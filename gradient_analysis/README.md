# Gradient Analysis Plotting

This folder contains the plotting utilities for the gradient-analysis workflow.

There are two plotting entry points:

1. [plot_gradient_analysis.py](/Users/deimos/Desktop/ICML/RAGEN/plot_gradient_analysis.py)
- pulls one W&B run directly
- exports local `json` / `csv`
- writes per-step PNG plots

2. [plot_icml_steps.py](/Users/deimos/Desktop/ICML/RAGEN/gradient_analysis/plot_icml_steps.py)
- builds a fixed 3-step comparison figure from already-exported `metrics.json` files
- intended for paper-style summary figures

For the training-side workflow and arguments, see:
- [docs/gradient_analysis_walkthrough.md](/Users/deimos/Desktop/ICML/RAGEN/docs/gradient_analysis_walkthrough.md)

## Typical Workflow

### 1. Run one analysis job

Example helper runner:

```bash
bash scripts/runs/run_sokoban_ppo_filter_grad_analysis.sh \
  --steps 1 \
  --gpus 0,1,2,3,4,5,6,7
```

That job:
- runs one pre-train validation
- runs one gradient-analysis pass on step 1
- exits immediately after analysis

### 2. List available analysis steps in W&B

```bash
python plot_gradient_analysis.py \
  --wandb-path deimos-xing/ragen_gradient_analysis/<run_id> \
  --list-steps
```

### 3. Plot all analysis steps from that run

```bash
python plot_gradient_analysis.py \
  --wandb-path deimos-xing/ragen_gradient_analysis/<run_id>
```

Default output directory:

```text
gradient_analysis_outputs/<run_name>_<run_id>/
```

### 4. Plot only one step

```bash
python plot_gradient_analysis.py \
  --wandb-path deimos-xing/ragen_gradient_analysis/<run_id> \
  --step 1
```

### 5. Choose your own output directory

```bash
python plot_gradient_analysis.py \
  --wandb-path deimos-xing/ragen_gradient_analysis/<run_id> \
  --step 1 \
  --output-dir gradient_analysis_outputs/my_custom_dir
```

## Files Produced By `plot_gradient_analysis.py`

For each selected step, the script writes:

- `gradient_analysis_summary_step_<N>.png`
- `gradient_analysis_plots_step_<N>.png`
- `gradient_analysis_loss_plots_step_<N>.png`
- `gradient_analysis_reward_std_step_<N>.png`
- `gradient_analysis_normed_grads_step_<N>.png`
- `gradient_analysis_metrics_step_<N>.json`
- `gradient_analysis_bucket_rv_table_step_<N>.csv`

The `metrics.json` export is the bridge to the paper-style plotting script.

## Building A 3-Step Comparison Figure

If you have three exported step directories and want the fixed grid figure:

```bash
python gradient_analysis/plot_icml_steps.py \
  --mode ppo \
  --step0-dir /path/to/step0 \
  --step20-dir /path/to/step20 \
  --step40-dir /path/to/step40 \
  --out gradient_analysis_outputs/ppo_step0_20_40.png
```

Each step directory must contain:

```text
metrics.json
```

If your exported file is named `gradient_analysis_metrics_step_<N>.json`, copy or rename it to `metrics.json` inside each step directory before calling `plot_icml_steps.py`.

## What To Inspect First

For a new run, start with:

1. `gradient_analysis_summary_step_<N>.png`
2. `gradient_analysis_plots_step_<N>.png`
3. `gradient_analysis_metrics_step_<N>.json`

Those three are usually enough to tell:
- how many buckets were populated
- whether task gradients dominate regularizer gradients
- whether gradient magnitude is monotonic or non-monotonic in reward variance
