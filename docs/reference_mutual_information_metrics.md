# Mutual Information Metrics for Collapse Detection

This document provides a comprehensive explanation of the mutual information (MI) based metrics used in RAGEN for detecting training collapse phenomena.

## 1. Overview: Two Core Metrics

We focus on two diagnostic quantities:

| Quantity | Meaning | Diagnostic Metric |
|--------------|------------|-------------------|
| **Within-input variability** | How much reasoning varies under the same input | $H(Z \mid X)$ |
| **Across-input dependence** | How much reasoning still depends on the input | $I(X; Z)$ |

**Key Insight**: We compute MI under the batch's empirical input distribution (uniform over prompts), not the true $p(x)$. This is exactly what's needed for diagnosing whether reasoning remains input-dependent inside a training batch.

---

## 2. Design Decision: Partitioning $X$ and $Z$

The choice of how to partition the sequence into conditioning context $X$ and reasoning $Z$ is crucial for meaningful collapse detection. The partition answers: **"Which segment of generation depends on which segment of input context?"**

### 2.1 Design Goal

We want to measure whether the reasoning content becomes increasingly input-independent (i.e., ignoring the environment state and producing generic outputs), while also tracking how much variability remains under the same input.

### 2.2 Recommended Partition

For a typical agent turn with structure:
```
[System Prompt] [User: State] [Assistant:] <think> reasoning content </think> <answer> action </answer>
```

We define:

| Variable | Content | Rationale |
|----------|---------|-----------|
| **$X$** | System prompt + User turn (state) + Assistant prefix + `<think>` tag | Everything the model sees *before* generating reasoning content |
| **$Z$** | Reasoning content tokens (between `<think>` and `</think>`, **excluding both tags**) | The actual reasoning we want to measure dependency for |

### 2.3 Why Include `<think>` in $X$?

The `<think>` tag should be part of $X$ (conditioning context), not $Z$ (reasoning):

1. **Semantic role**: `<think>` is a control token meaning "start generating reasoning" — it's a boundary marker, not reasoning content itself.

2. **Near-constant token**: `<think>` appears identically in every sample, so including it in $Z$ would:
   - Add no discriminative information between prompts
   - Dilute entropy/MI statistics with high-probability constant tokens

3. **Clean separation**: With `<think>` in $X$, the partition becomes: "everything before reasoning starts" vs "reasoning content itself"

### 2.4 Why Exclude `</think>` from $Z$?

The `</think>` closing tag should also be excluded from $Z$:

1. **Structural boundary**: Like `<think>`, it's a format token, not reasoning content.

2. **Format stability signal**: If `</think>` is included in $Z$, MI/entropy metrics would conflate:
   - Reasoning content dependency (what we want)
   - Format stability (whether the model reliably closes tags)

3. **Cleaner interpretation**: Excluding both tags means $Z$ purely measures "does the reasoning *content* depend on the input state?"

### 2.5 Implementation Mapping

In the codebase, this corresponds to:

| Field | Content |
|-------|---------|
| `first_turn_prompt_ids` | Tokens up to and including `<think>` |
| `first_turn_reasoning_ids` | Reasoning content tokens only (no `<think>`, no `</think>`) |

---

## 3. Notation and Definitions

### 3.1 Random Variables

| Symbol | Description |
|--------|-------------|
| $X$ | Input context: system prompt + user turn + assistant prefix + `<think>` |
| $Z$ | Reasoning content tokens (between `<think>` and `</think>`, excluding tags) |
| $x_j$ | The $j$-th unique prompt in the batch, $j \in \{1, \ldots, N\}$ |
| $z_{i,k}$ | The $k$-th reasoning sample for trajectory $i$ |
| $N$ | Number of unique prompts in the batch |
| $K$ | Number of reasoning samples per prompt (group size) |

### 3.2 Probability Distributions

| Symbol | Definition | Description |
|--------|------------|-------------|
| $p(z \mid x)$ | $\prod_{t=1}^{T} p_\theta(z_t \mid x, z_{1:t-1})$ | Conditional probability of reasoning $z$ given prompt $x$ under policy $\pi_\theta$ |
| $p_{\text{mix}}(z)$ | $\frac{1}{N} \sum_{j=1}^{N} p(z \mid x_j)$ | Marginal probability under uniform prompt mixture |
| $\hat{p}(x)$ | $\frac{1}{N}$ | Empirical (uniform) distribution over batch prompts |

---

## 4. Core Information-Theoretic Quantities

### 4.1 Conditional Entropy $H(Z \mid X)$

**Definition**: The expected uncertainty in the reasoning $Z$ given the prompt $X$.

$$H(Z \mid X) = -\mathbb{E}_{x \sim \hat{p}(x)} \mathbb{E}_{z \sim p(z|x)} \left[ \log p(z \mid x) \right]$$

**Estimation**: Using sampled (prompt, reasoning) pairs:

$$\hat{H}(Z \mid X) = -\frac{1}{NK} \sum_{i,k} \log p(z_{i,k} \mid x_i)$$

**Interpretation**:
- **High $H(Z \mid X)$**: Model generates diverse responses for each prompt (stochastic policy)
- **Low $H(Z \mid X)$**: Model generates deterministic/repetitive responses for each prompt

**Code Reference** (`collapse_metrics.py:675-700`):
```python
conditional_entropy = -matched.mean().item()  # H(Z|X) estimate
```

### 4.2 Marginal Entropy $H(Z)$

**Definition**: The total entropy of reasoning under the marginal distribution.

$$H(Z) = -\mathbb{E}_{z \sim p_{\text{mix}}(z)} \left[ \log p_{\text{mix}}(z) \right]$$

**Estimation**: Using the mixture distribution:

$$\hat{H}(Z) = -\frac{1}{NK} \sum_{i,k} \log p_{\text{mix}}(z_{i,k})$$

where:

$$p_{\text{mix}}(z) = \frac{1}{N} \sum_{j=1}^{N} p(z \mid x_j)$$

**Code Reference** (`collapse_metrics.py:675-700`):
```python
reasoning_entropy = -marginal.mean().item()  # H(Z) estimate
```

### 4.3 Mutual Information $I(X; Z)$

**Definition**: The amount of information that the reasoning $Z$ contains about the prompt $X$.

$$I(X; Z) = H(Z) - H(Z \mid X)$$

Equivalently:

$$I(X; Z) = \mathbb{E}_{x, z} \left[ \log \frac{p(z \mid x)}{p_{\text{mix}}(z)} \right]$$

**Estimation**:

$$\hat{I}(X; Z) = \frac{1}{NK} \sum_{i,k} \left[ \log p(z_{i,k} \mid x_i) - \log p_{\text{mix}}(z_{i,k}) \right]$$

**Interpretation**:
- **High $I(X; Z)$**: Reasoning is input-dependent (healthy)
- **Low $I(X; Z)$**: Reasoning has weak input dependence
- **Upper Bound**: $I(X; Z) \leq H(X) = \log N$ (when $X$ is uniform)

**Practical Note on Negative Values**:
- The true mutual information satisfies $I(X; Z) \geq 0$.
- Our logged `mi_estimate` and `mi_seq_estimate` are finite-sample Monte Carlo estimates, not exact MI.
- Because they average noisy sample terms of the form $\log p(z \mid x) - \log p_{\text{mix}}(z)$, they can temporarily dip below zero when the true MI is near zero or the sampled batch is noisy.
- In practice, a small negative value should usually be read as "approximately zero input dependence within estimation noise," not as a violation of information theory.

**Code Reference** (`collapse_metrics.py:563-590`):
```python
def _compute_mi_estimate(self, matched, marginal, N_prompts):
    mi = matched.mean().item() - marginal.mean().item()
    return {
        "collapse/mi_estimate": mi,
        "collapse/mi_upper_bound": math.log(N_prompts),
    }
```

---

## 5. Computation Pipeline

### 5.1 Cross Log-Probability Matrix

For each reasoning $z_{i,k}$ and each prompt $x_j$, we compute the cross log-probability:

$$\ell_j(z_{i,k}) = \log p(z_{i,k} \mid x_j) = \sum_{t=1}^{T} \log p_\theta(z_{i,k,t} \mid x_j, z_{i,k,1:t-1})$$

This forms a matrix $\mathbf{L} \in \mathbb{R}^{NK \times N}$ where:
- Rows index (trajectory, sample) pairs
- Columns index unique prompts

**Code Reference** (`collapse_metrics.py:452-546`):
```python
def _compute_cross_log_probs(self, ...):
    """
    For each reasoning z_{i,k} and each prompt x_j:
    1. Construct sequence [x_j | z_{i,k}]
    2. Compute teacher-forcing log prob
    3. Sum over reasoning tokens → ℓ_j(z_{i,k})
    """
    cross_log_probs = torch.zeros(NK, N, device=device)      # per-token mean
    cross_log_probs_sum = torch.zeros(NK, N, device=device)  # per-sequence sum
```

### 5.2 Matched vs Marginal Log-Probabilities

**Matched**: Log-probability of reasoning under its true prompt:
$$\text{matched}_{i,k} = \ell_i(z_{i,k}) = \log p(z_{i,k} \mid x_i)$$

**Marginal**: Log-probability under uniform prompt mixture:
$$\text{marginal}_{i,k} = \log p_{\text{mix}}(z_{i,k}) = \log \left( \frac{1}{N} \sum_{j=1}^{N} \exp(\ell_j(z_{i,k})) \right)$$

Using log-sum-exp for numerical stability:
$$\text{marginal}_{i,k} = \text{logsumexp}_j(\ell_j(z_{i,k})) - \log N$$

**Code Reference** (`collapse_metrics.py:548-561`):
```python
def _compute_log_prob_stats(self, cross_log_probs, col_ids):
    NK, N = cross_log_probs.shape
    matched = cross_log_probs[torch.arange(NK), col_ids]  # diagonal elements
    marginal = torch.logsumexp(cross_log_probs, dim=1) - math.log(N)
    return matched, marginal
```

---

## 6. Per-Token vs Per-Sequence Metrics

We compute two variants of each metric:

| Variant | Normalization | Use Case |
|---------|--------------|----------|
| **Per-token** (`_est`) | Divide by sequence length | Length-invariant comparison |
| **Per-sequence** (`_seq_est`) | Sum over tokens | Total information content |

### 6.1 Per-Token (Length-Normalized)

$$\bar{\ell}_j(z) = \frac{1}{T} \sum_{t=1}^{T} \log p(z_t \mid x_j, z_{1:t-1})$$

This reduces length bias when comparing reasoning of different lengths.

### 6.2 Per-Sequence (Sum)

$$\ell_j(z) = \sum_{t=1}^{T} \log p(z_t \mid x_j, z_{1:t-1})$$

This captures total log-probability without normalization.

**Base Metric Suffixes**:
- `collapse/mi_estimate` — Per-token MI
- `collapse/mi_seq_estimate` — Per-sequence MI
- `collapse/conditional_entropy_est` — Per-token $H(Z|X)$
- `collapse/conditional_entropy_seq_est` — Per-sequence $H(Z|X)$
- `collapse/reasoning_entropy_est` — Per-token $H(Z)$
- `collapse/reasoning_entropy_seq_est` — Per-sequence $H(Z)$

These are the raw suffixes produced inside `_compute_metrics_for_pairs`. In logged outputs, they are usually namespaced as `collapse_first_turn_sample/<suffix>` or `collapse_trajectory_sample/<suffix>`.

---

## 7. Additional Diagnostic Metrics

### 7.1 Retrieval Accuracy

**Definition**: Fraction of samples where the highest cross-log-probability matches the true prompt.
If multiple prompts are identical (same tokenized prompt text), they are treated as equivalent columns, and any of those columns counts as correct for retrieval accuracy and chance.

$$\text{Acc} = \frac{1}{NK} \sum_{i,k} \mathbf{1}\left[ \arg\max_j \ell_j(z_{i,k}) = i \right]$$

**Interpretation**:
- **High Accuracy** ($\approx 1$): Reasoning is highly prompt-specific
- **Chance Level** ($\approx 1/N$): Reasoning is prompt-independent

**Code Reference** (`collapse_metrics.py:592-673`):
```python
def _compute_retrieval_accuracy(self, cross_log_probs, col_ids, N_prompts):
    predicted_cols = torch.argmax(cross_log_probs, dim=1)
    correct = (predicted_cols == col_ids).float()
    accuracy = correct.mean().item()
    chance_level = 1.0 / N_prompts
```

**Base Metric Suffixes**:
- `collapse/retrieval_accuracy` — Top-1 accuracy
- `collapse/retrieval_accuracy@k` — Top-k accuracy (k ∈ {2, 4, 8})
- `collapse/retrieval_chance_level` — Expected accuracy under random guessing
- `collapse/retrieval_above_chance` — Accuracy improvement over chance
- `collapse/retrieval_chance_level@k` — Expected top-k accuracy under random guessing
- `collapse/retrieval_above_chance@k` — Top-k accuracy improvement over chance

### 7.2 MI Z-Score

**Definition**: Standardized MI using the marginal log-probability standard deviation.

$$\text{MI-ZScore} = \frac{\text{matched} - \text{marginal}}{\sigma_{\text{marginal}} + \epsilon}$$

where $\sigma_{\text{marginal}} = \text{std}(\text{marginal}_{i,k})$ and $\epsilon = 10^{-3}$ for stability.

**Interpretation**: Measures how many standard deviations the matched log-prob is above the marginal. More robust to scale changes during training.

**Practical Note on Extreme Negative Z-Scores**:
- Negative `mi_zscore*` values are normal; they simply mean the matched log-prob is below the marginal baseline on that batch.
- Very large-magnitude values, especially for `mi_zscore_seq`, often happen when `marginal_std` or `marginal_std_seq` becomes very small, so the normalization denominator is close to `std_eps`.
- When this happens, interpret `mi_zscore*` together with `marginal_std*` and `mi_estimate` rather than in isolation.

**Code Reference** (`collapse_metrics.py:302-320`):
```python
marginal_std = marginal.std(unbiased=False)
metrics["collapse/mi_zscore"] = ((matched - marginal) / (marginal_std + self.std_eps)).mean().item()
```

### 7.3 EMA-Normalized MI Z-Score

To handle variance drift during training, we track an exponential moving average of the marginal standard deviation:

$$\sigma_{\text{EMA}}^{(t)} = \alpha \cdot \sigma_{\text{EMA}}^{(t-1)} + (1 - \alpha) \cdot \sigma_{\text{marginal}}^{(t)}$$

where $\alpha = 0.9$ (default decay rate).

**Base Metric Suffixes**:
- `collapse/marginal_std` — Current batch marginal std
- `collapse/marginal_std_seq` — Current batch marginal std (per-sequence)
- `collapse/marginal_std_ema` — EMA of marginal std
- `collapse/mi_zscore_ema` — MI Z-score normalized by EMA std
- `collapse/marginal_std_ema_seq` — EMA of marginal std (per-sequence)
- `collapse/mi_zscore_seq` — MI Z-score (per-sequence)
- `collapse/mi_zscore_ema_seq` — MI Z-score normalized by EMA std (per-sequence)

---

## 8. Multi-Turn Sampling Strategies

For multi-turn trajectories, we support two sampling strategies:

### 8.1 Trajectory-Uniform Sampling

**Probability**: $\Pr(m, t) = \frac{1}{M} \cdot \frac{1}{T_m}$

- First sample trajectory $m$ uniformly
- Then sample turn $t$ uniformly within trajectory
- Each trajectory has equal weight regardless of length

**Code Reference** (`collapse_metrics.py:780-813`):
```python
def _sample_trajectory_uniform(self, ...):
    """Each trajectory has equal weight regardless of length."""
    for _ in range(num_to_sample):
        m = np.random.randint(M)  # uniform over trajectories
        t = np.random.randint(turn_counts[m])  # uniform over turns
```

### 8.2 Turn-Uniform Sampling (Disabled by Default)

**Probability**: $\Pr(m, t) = \frac{1}{\sum_m T_m}$

- Uniform over all (trajectory, turn) pairs
- Longer trajectories contribute more samples

---

## 9. Summary of All Logged Metrics

The code logs metrics in two layers:

1. **Sample-scoped diagnostic metrics**: computed on sampled $(x, z)$ pairs, then namespaced by sampling strategy.
2. **Global coverage / timing metrics**: logged directly without an additional sample prefix.

### 9.1 W&B Namespace Patterns

| Logged Key Pattern | When It Appears | Meaning |
|--------------------|-----------------|---------|
| `collapse_first_turn_sample/<suffix>` | `first_turn_enabled=True` and first-turn data exists | Diagnostics computed on first-turn $(x, z)$ pairs |
| `collapse_trajectory_sample/<suffix>` | `multi_turn_enabled=True` and multi-turn data exists | Diagnostics computed on trajectory-uniform multi-turn samples |
| `collapse_turn_sample/<suffix>` | Code path exists, but currently disabled by default | Diagnostics computed on turn-uniform multi-turn samples |
| `collapse/valid_thinking_rate` | `turn_counts_total` and `turn_counts` are available | Fraction of valid reasoning turns among all turns |
| `collapse/first_turn_num_total` | First-turn metrics enabled and data exists | Number of first-turn candidates before filtering empty reasoning |
| `collapse/first_turn_num_valid` | First-turn metrics enabled and data exists | Number of first-turn samples with non-empty reasoning |
| `collapse/first_turn_valid_rate` | First-turn metrics enabled and data exists | Valid first-turn fraction |
| `timing_s/collapse_multi_turn_step` | `multi_turn_enabled=True` | Wall-clock time for the multi-turn collapse pass |
| `timing_s/collapse_first_turn_step` | `first_turn_enabled=True` | Wall-clock time for the first-turn collapse pass |

The suffix tables below describe the metric families that can appear under `collapse_first_turn_sample/`, `collapse_trajectory_sample/`, and, if re-enabled, `collapse_turn_sample/`.

### 9.2 Core Information Metrics

| Suffix | Formula / Definition | Typical Reading |
|--------|----------------------|-----------------|
| `mi_estimate` | $\mathbb{E}[\log p(z \mid x) - \log p_{\text{mix}}(z)]$ | Higher means stronger input dependence |
| `mi_seq_estimate` | Sequence-sum version of MI | Same as above, but not length-normalized |
| `mi_upper_bound` | $\log N$ | Theoretical ceiling given $N$ unique prompts |
| `conditional_entropy_est` | $-\mathbb{E}[\log p(z \mid x)]$ | Higher means more within-input variability |
| `conditional_entropy_seq_est` | Sequence-sum version of $H(Z \mid X)$ | Total within-input uncertainty per sequence |
| `reasoning_entropy_est` | $-\mathbb{E}[\log p_{\text{mix}}(z)]$ | Total marginal diversity across prompts |
| `reasoning_entropy_seq_est` | Sequence-sum version of $H(Z)$ | Total marginal uncertainty per sequence |
| `matched_log_prob_mean` | $\mathbb{E}[\log p(z \mid x)]$ | Less negative is better fit to the true prompt |
| `marginal_log_prob_mean` | $\mathbb{E}[\log p_{\text{mix}}(z)]$ | Less negative means the response is broadly likely under the prompt mixture |

Example W&B keys:
- `collapse_first_turn_sample/mi_estimate`
- `collapse_trajectory_sample/conditional_entropy_est`
- `collapse_first_turn_sample/reasoning_entropy_seq_est`

### 9.3 Retrieval Metrics

| Suffix | Definition | Typical Reading |
|--------|------------|-----------------|
| `retrieval_accuracy` | Top-1 prompt retrieval accuracy from cross log-probs | Higher means reasoning is more prompt-specific |
| `retrieval_accuracy@2`, `@4`, `@8` | Top-k retrieval accuracy | Higher means prompt identity is easier to recover |
| `retrieval_chance_level` | Expected top-1 accuracy under random guessing | Baseline for comparison |
| `retrieval_chance_level@2`, `@4`, `@8` | Expected top-k accuracy under random guessing | Top-k baseline |
| `retrieval_above_chance` | `retrieval_accuracy - retrieval_chance_level` | Positive margin over chance |
| `retrieval_above_chance@2`, `@4`, `@8` | Top-k accuracy minus top-k chance | Positive margin over chance |

Example W&B keys:
- `collapse_first_turn_sample/retrieval_accuracy`
- `collapse_trajectory_sample/retrieval_accuracy@4`
- `collapse_first_turn_sample/retrieval_above_chance@8`

### 9.4 Variance-Normalized Metrics

| Suffix | Definition | Typical Reading |
|--------|------------|-----------------|
| `marginal_std` | $\text{std}(\text{marginal})$ | Current-batch spread of marginal log-probs |
| `marginal_std_seq` | Sequence-sum version of `marginal_std` | Current-batch spread on total log-prob scale |
| `marginal_std_ema` | EMA of `marginal_std` | Smoothed normalization scale |
| `marginal_std_ema_seq` | EMA of `marginal_std_seq` | Smoothed sequence-scale normalization |
| `mi_zscore` | $(\text{matched} - \text{marginal}) / (\text{marginal\_std} + \epsilon)$ | Standardized MI, batch-normalized |
| `mi_zscore_seq` | Sequence-sum version of `mi_zscore` | Standardized sequence-scale MI |
| `mi_zscore_ema` | MI normalized by `marginal_std_ema` | More stable across training drift |
| `mi_zscore_ema_seq` | Sequence-sum version of `mi_zscore_ema` | Stable sequence-scale normalization |

### 9.5 Directly Logged Coverage and Timing Metrics

| Logged Key | Meaning |
|------------|---------|
| `collapse/valid_thinking_rate` | Share of valid reasoning turns among all recorded turns |
| `collapse/first_turn_num_total` | Count of first-turn entries before removing empty reasoning |
| `collapse/first_turn_num_valid` | Count of first-turn entries with non-empty reasoning |
| `collapse/first_turn_valid_rate` | `first_turn_num_valid / first_turn_num_total` |
| `timing_s/collapse_multi_turn_step` | Time spent computing multi-turn collapse metrics on this step |
| `timing_s/collapse_first_turn_step` | Time spent computing first-turn collapse metrics on this step |

---

## 10. Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compute_freq` | 5 | Compute metrics every N steps |
| `micro_batch_size` | 128 | Batch size for cross-scoring |
| `first_turn_enabled` | True | Compute first-turn metrics |
| `multi_turn_enabled` | True | Enable multi-turn sampling |
| `num_samples` | 64 | Number of $(x, z)$ pairs to sample |
| `std_eps` | 1e-3 | Stability constant for std normalization |
| `ema_decay` | 0.9 | EMA decay for cross-time std tracking |

**Configuration in `base.yaml`** (`base.yaml:135-139`):
```yaml
collapse_detection:
  compute_freq: 5
  micro_batch_size: 128
  first_turn_enabled: true
  multi_turn_enabled: true
  num_samples: 64
```

---

## 11. Mathematical Derivations

### 11.1 MI Estimation via Importance Sampling

The mutual information is:

$$I(X; Z) = \mathbb{E}_{p(x,z)} \left[ \log \frac{p(z \mid x)}{p(z)} \right]$$

Under the empirical distribution $\hat{p}(x) = 1/N$ (uniform over batch prompts):

$$I(X; Z) = \mathbb{E}_{x \sim \hat{p}(x)} \mathbb{E}_{z \sim p(z|x)} \left[ \log \frac{p(z \mid x)}{p_{\text{mix}}(z)} \right]$$

where $p_{\text{mix}}(z) = \sum_j \hat{p}(x_j) p(z \mid x_j) = \frac{1}{N} \sum_j p(z \mid x_j)$.

Monte Carlo estimate with $K$ samples per prompt:

$$\hat{I}(X; Z) = \frac{1}{NK} \sum_{i=1}^{N} \sum_{k=1}^{K} \left[ \log p(z_{i,k} \mid x_i) - \log p_{\text{mix}}(z_{i,k}) \right]$$

### 11.2 Information-Theoretic Identity

The fundamental identity relating our metrics:

$$I(X; Z) = H(Z) - H(Z \mid X)$$

This means:
- If $H(Z|X)$ drops but $H(Z)$ stays constant → MI increases (good)
- If both $H(Z)$ and $H(Z|X)$ drop equally → MI stays constant
- If $H(Z) \to H(Z|X)$ → MI → 0 (input dependence vanishes)
