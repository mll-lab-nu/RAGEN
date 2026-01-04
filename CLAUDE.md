# CLAUDE.md - AI Assistant Guide for RAGEN

This document provides comprehensive guidance for AI assistants (like Claude) working with the RAGEN codebase. It explains the architecture, development workflows, key conventions, and common tasks.

**Last Updated:** 2026-01-04

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture & Design Patterns](#architecture--design-patterns)
3. [Directory Structure](#directory-structure)
4. [Core Components](#core-components)
5. [Environment System](#environment-system)
6. [Configuration System](#configuration-system)
7. [Training Workflow](#training-workflow)
8. [Development Workflows](#development-workflows)
9. [Testing Conventions](#testing-conventions)
10. [Common Tasks](#common-tasks)
11. [Key Conventions for AI Assistants](#key-conventions-for-ai-assistants)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

**RAGEN** (Reasoning Agent) is a reinforcement learning framework for training LLM agents in interactive, stochastic environments using the **StarPO** (State-Thinking-Actions-Reward Policy Optimization) algorithm.

### Key Concepts

- **Multi-turn Interactions**: Agents perform sequential decision-making across multiple turns
- **Stochastic Environments**: Same actions can lead to different outcomes
- **Trajectory-level Optimization**: Optimizes entire multi-turn trajectories, not just individual actions
- **Reasoning-guided Actions**: LLMs generate `<think>...</think><answer>...</answer>` pairs

### Technology Stack

- **Core**: PyTorch, Transformers, PEFT (LoRA)
- **RL Infrastructure**: veRL (submodule), Ray (distributed training)
- **Inference**: vLLM (fast LLM inference)
- **Configuration**: Hydra
- **Logging**: WandB, SwanLab
- **Environments**: Gymnasium, custom implementations

### Supported Environments

- **Sokoban**: Puzzle-solving (push boxes to targets)
- **FrozenLake**: Navigation with slippery ice
- **Bandit**: Multi-armed bandit problems
- **Countdown**: Math equation generation
- **MetaMathQA**: Math problem solving
- **WebShop**: E-commerce navigation
- **Lean**: Theorem proving
- **Sudoku**: Puzzle solving with feedback
- **Spatial**: Spatial reasoning tasks

---

## Architecture & Design Patterns

### Modular Three-Component Architecture

RAGEN follows a clean separation of concerns with three main components:

```
┌─────────────────┐
│  Agent Proxy    │  ← Orchestrates rollout process
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──────┐  │
│ Context  │  │  ← Translates between env and LLM
│ Manager  │  │
└───┬──────┘  │
    │         │
┌───▼─────────▼──┐
│  Environment   │  ← Manages multiple parallel environments
│ State Manager  │
└────────────────┘
```

#### 1. Environment State Manager (`ragen/llm_agent/es_manager.py`)

**Responsibilities:**
- Manages multiple environment instances (different seeds, configs)
- Executes actions and tracks environment states
- Handles parallel execution for complex environments
- Computes metrics (success rate, pass@k, custom metrics)
- Manages deterministic seed progression

**Key Methods:**
- `reset(seed)`: Initialize environments with deterministic seeds
- `step(actions)`: Execute actions in all environments
- `get_rollout_states()`: Gather final trajectories

#### 2. Context Manager (`ragen/llm_agent/ctx_manager.py`)

**Responsibilities:**
- **Environment → LLM**: Format observations as chat messages
- **LLM → Environment**: Parse LLM responses into actions
- **Trajectory Formulation**: Prepare rollouts for PPO training
- **Reward Normalization**: Apply groupwise normalization strategies

**Key Features:**
- Context window control (`max_context_window`)
- Response parsing (extracts `<think>` and `<answer>` tags)
- Multi-turn history management
- Token masking for training (only train on assistant responses)

#### 3. Agent Proxy (`ragen/llm_agent/agent_proxy.py`)

**Responsibilities:**
- High-level orchestration of rollout generation
- Coordinates between context manager and environment manager
- Manages LLM backend (vLLM, API calls)

### Design Patterns

#### Pattern 1: Configuration-Driven Development

All behavior is configurable via YAML:
- No hardcoded hyperparameters in code
- Environment definitions separate from implementation
- Easy experimentation with different setups

#### Pattern 2: Group-Based Training

Environments organized into groups with same seed/config:
- Group size typically 16 (for pass@k metrics)
- Enables curriculum learning (different groups = different tasks)
- Supports reward normalization within groups

#### Pattern 3: Rollout Filtering

Not all trajectories are equally useful:
- Keep top 25% by reward variance → encourages exploration
- Reduces noise in gradient estimates
- Improves sample efficiency

#### Pattern 4: Modular Environment Registration

Environments are registered in dictionaries:
```python
REGISTERED_ENVS = {'sokoban': SokobanEnv, ...}
REGISTERED_ENV_CONFIGS = {'sokoban': SokobanEnvConfig, ...}
```

---

## Directory Structure

```
/home/user/RAGEN/
├── ragen/                      # Main package
│   ├── llm_agent/             # Core agent components
│   │   ├── agent_proxy.py     # Rollout orchestration
│   │   ├── es_manager.py      # Environment state management
│   │   ├── ctx_manager.py     # Context/prompt management
│   │   ├── base_llm.py        # LLM abstractions
│   │   └── ap_webshop.py      # WebShop-specific proxy
│   ├── env/                   # Environment implementations
│   │   ├── base.py            # Abstract base classes
│   │   ├── sokoban/           # Sokoban puzzle
│   │   ├── frozen_lake/       # FrozenLake navigation
│   │   ├── bandit/            # Multi-armed bandit
│   │   ├── countdown/         # Countdown number game
│   │   ├── metamathqa/        # Math QA
│   │   ├── webshop/           # WebShop e-commerce
│   │   ├── lean/              # Lean theorem proving
│   │   ├── sudoku/            # Sudoku puzzle
│   │   └── spatial/           # Spatial reasoning
│   ├── trainer/               # Training infrastructure
│   │   ├── agent_trainer.py   # Main PPO trainer
│   │   └── rollout_filter.py  # Trajectory filtering
│   ├── workers/               # Distributed training workers
│   │   └── fsdp_workers.py    # FSDP worker implementations
│   ├── patches/               # Compatibility patches
│   └── utils.py               # Utilities (seeding, logging)
├── config/                    # Hydra configuration files
│   ├── base.yaml             # Main config (inherits from others)
│   ├── envs.yaml             # Environment definitions
│   ├── ppo_trainer.yaml      # Symlink to veRL config
│   ├── eval.yaml             # Evaluation config
│   └── _*.yaml               # Environment-specific configs
├── tests/                     # Test suite
│   ├── llm_agent/            # Agent component tests
│   ├── env/                  # Environment tests
│   └── es_manager/           # Manager tests
├── scripts/                   # Setup and utility scripts
│   └── setup_ragen.sh        # Environment setup script
├── verl/                      # veRL submodule (RL infrastructure)
├── external/                  # External dependencies (webshop, etc.)
├── train.py                   # Main training entry point
├── setup.py                   # Package installation
├── requirements.txt           # Dependencies
└── pytest.ini                 # Test configuration
```

### Important Files Quick Reference

| File | Purpose | When to Modify |
|------|---------|----------------|
| `train.py` | Main training entry point | Rarely (core training logic) |
| `ragen/llm_agent/agent_proxy.py` | Rollout orchestration | Adding new rollout features |
| `ragen/llm_agent/es_manager.py` | Environment management | Changing env execution logic |
| `ragen/llm_agent/ctx_manager.py` | Prompt engineering | Modifying prompt formats |
| `ragen/trainer/agent_trainer.py` | PPO training loop | Changing training algorithm |
| `ragen/env/base.py` | Environment base classes | Adding new env abstractions |
| `config/base.yaml` | Main configuration | Adjusting hyperparameters |
| `config/envs.yaml` | Environment definitions | Adding/modifying environments |

---

## Core Components

### Environment State Manager Details

**Location:** `ragen/llm_agent/es_manager.py`

**Key Data Structures:**

```python
EnvStatus: {
    truncated: bool,        # Done but not successful
    terminated: bool,       # Done and successful
    num_actions: int,       # Current action count
    rewards: List[float],   # Turn-wise rewards
    seed: Optional[int]     # Reset seed
}

rollout_cache: List[{
    env_id: int,
    group_id: int,
    tag: str,               # Environment type
    history: List[Dict],    # Full trajectory history
    penalty: float,         # Format violation penalties
    metrics: Dict           # Environment-specific metrics
}]
```

**Configuration Flow:**
1. Reads `es_manager.train/val` from config
2. Each environment tag maps to `custom_envs[tag]` definition
3. Supports `parallel_friendly` flag for threading

### Context Manager Details

**Location:** `ragen/llm_agent/ctx_manager.py`

**Critical Features:**

1. **Context Window Control:**
   - `max_context_window` limits history length
   - Prevents context overflow in long episodes
   - Set to -1 for unlimited context

2. **Response Parsing:**
   - Extracts `<think>...</think>` for reasoning
   - Extracts `<answer>...</answer>` for actions
   - Handles multiple actions separated by `||`

3. **without_history Mode:**
   - Treats each turn as independent sample (for GRPO)
   - During rollout: Only show current state
   - During training: Each turn becomes separate sample
   - Episode reward distributed to all turns

4. **Reward Normalization:**
   ```yaml
   reward_normalization:
     grouping: "state"     # state/batch/inductive
     method: "identity"    # identity/mean_std/mean/asym_clip
   ```

### Agent Proxy Rollout Flow

**Location:** `ragen/llm_agent/agent_proxy.py`

```python
def rollout(dataproto, val=False):
    1. es_manager.reset() → initial observations
    2. for turn in range(max_turn):
        a. ctx_manager.get_lm_inputs() → format prompts
        b. actor_wg.generate_sequences() → LLM generation
        c. ctx_manager.get_env_inputs() → parse actions
        d. es_manager.step() → execute in environments
    3. es_manager.get_rollout_states() → gather final states
    4. ctx_manager.formulate_rollouts() → prepare for training
```

**LLM Backend Support:**
- `VllmWrapperWg`: Local vLLM inference
- `ApiCallingWrapperWg`: External API calls (Anthropic, Together)
- `RayWorkerGroup`: Distributed vLLM with Ray

---

## Environment System

### Base Classes

**Location:** `ragen/env/base.py`

```python
class BaseEnv(ABC):
    """Abstract base for all environments"""

    @abstractmethod
    def reset(self, seed: Optional[int] = None):
        """Initialize with deterministic seed"""
        pass

    @abstractmethod
    def step(self, action):
        """Execute action, return (obs, reward, done, info)"""
        pass

    @abstractmethod
    def render(self) -> str:
        """Get current state observation"""
        pass

    @abstractmethod
    def close(self):
        """Cleanup resources"""
        pass

class BaseDiscreteActionEnv(BaseEnv):
    """For discrete actions (Sokoban, FrozenLake)"""
    pass

class BaseLanguageBasedEnv(BaseEnv):
    """For language actions (Countdown, MetaMathQA)"""
    pass
```

### Environment Registration

**Location:** `ragen/env/__init__.py`

Environments must be registered in two dictionaries:

```python
REGISTERED_ENVS = {
    'sokoban': SokobanEnv,
    'frozen_lake': FrozenLakeEnv,
    'bandit': BanditEnv,
    # ... etc
}

REGISTERED_ENV_CONFIGS = {
    'sokoban': SokobanEnvConfig,
    'frozen_lake': FrozenLakeEnvConfig,
    'bandit': BanditEnvConfig,
    # ... etc
}
```

### Environment Configuration Pattern

Each environment has a dataclass config:

```python
@dataclass
class SokobanEnvConfig:
    dim_room: Tuple[int, int] = (6, 6)
    max_steps: int = 100
    num_boxes: int = 3
    search_depth: int = 300
    grid_lookup: Dict[int, str]      # Rendering symbols
    action_lookup: Dict[int, str]    # Action names
    observation_format: str = "grid" # grid/coord/grid_coord
```

---

## Configuration System

### Configuration Hierarchy

RAGEN uses Hydra for configuration management:

```
base.yaml                 # Top-level config
├── defaults:
│   ├── ppo_trainer.yaml  # RL algorithm settings (from veRL)
│   └── envs.yaml         # Environment definitions
```

### Key Configuration Sections

#### 1. System & Seeds

```yaml
system:
  CUDA_VISIBLE_DEVICES: "0"

seed:
  train: 10000  # Increments each rollout for determinism
  val: 123      # Fixed for reproducibility
```

#### 2. Model & LoRA

```yaml
model_path: Qwen/Qwen2.5-3B-Instruct

lora:
  rank: 0       # Set to 64 for LoRA training
  alpha: 64
  target_modules: all-linear
```

#### 3. Agent Proxy

```yaml
agent_proxy:
  without_history: False    # True → each turn independent
  max_context_window: -1    # Limit history length (-1 = unlimited)
  max_turn: 5               # Multi-turn episodes
  action_sep: "||"          # Action delimiter
  max_actions_per_turn: 2   # Max actions per turn
  enable_think: True        # <think> tags for CoT
  use_turn_scores: False    # Use turn-level rewards
  reward_normalization:
    grouping: "state"       # Normalize within groups
    method: "identity"      # No normalization
```

#### 4. Environment Manager

```yaml
es_manager:
  format_penalty: -0.1      # Penalty for invalid formats
  train:
    env_groups: 8           # Number of prompt groups
    group_size: 16          # Trajectories per group
    env_configs:
      tags: ["CoordSokoban"]
      n_groups: [8]         # Must sum to env_groups
  val:
    env_groups: 32
    group_size: 16
    env_configs:
      tags: ["CoordSokoban"]
      n_groups: [32]
```

#### 5. Actor/Rollout

```yaml
actor_rollout_ref:
  rollout:
    max_model_len: 3600       # Context window
    response_length: 400      # Max generation length
    temperature: 1.0
    rollout_filter_ratio: 0.25  # Keep top 25%
    rollout_filter_type: largest
    rollout_filter_metric: reward_variance
```

#### 6. Training

```yaml
trainer:
  project_name: ragen_latest
  experiment_name: test
  local_log_dir: "results/"
  save_freq: 100
  total_training_steps: 200
  test_freq: 10
  val_before_train: True
  n_gpus_per_node: 1
  logger: ['console', 'wandb']
```

#### 7. Algorithm

```yaml
algorithm:
  gamma: 1.0              # Discount factor
  lam: 1.0                # GAE lambda
  adv_estimator: gae      # gae/grpo/reinforce_plus_plus/rloo
  bi_level_gae: False     # Hierarchical GAE
  kl_ctrl:
    type: fixed
    kl_coef: 0.000
```

### Configuration Resolution

Hydra resolves interpolations:
```yaml
ppo_mini_batch_size: 32
actor:
  ppo_mini_batch_size: ${ppo_mini_batch_size}  # References top-level
```

Custom resolvers:
```python
OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
OmegaConf.register_new_resolver("int_div", lambda x, y: int(x / y))
```

---

## Training Workflow

### Entry Point: `train.py`

**Main Flow:**

1. **Validate Config**: Check batch sizes, GPU counts, etc.
2. **Initialize Ray**: Set up distributed cluster
3. **Create ResourcePoolManager**: Allocate GPUs to roles
4. **Instantiate RayAgentTrainer**: Main training coordinator
5. **Init Workers**: Spawn actor/critic/ref workers
6. **Init Agent Proxy**: Create LLMAgentProxy
7. **Fit**: Run main training loop

### Training Loop (`ragen/trainer/agent_trainer.py`)

```python
for step in range(total_training_steps):
    # 1. Generate rollouts
    batch = agent_proxy.rollout(val=False)

    # 2. Filter trajectories (keep top 25% by default)
    batch = rollout_filter.filter(batch)

    # 3. Adjust batch size (divisible by ppo_mini_batch_size)
    batch = adjust_batch(batch, size_divisor)

    # 4. Compute old log probs (for PPO ratio)
    old_log_prob = actor_rollout_wg.compute_log_prob(batch)

    # 5. Compute reference log probs (for KL penalty)
    ref_log_prob = ref_policy_wg.compute_ref_log_prob(batch)

    # 6. Compute values (for GAE)
    values = critic_wg.compute_values(batch)

    # 7. Compute advantages
    batch = compute_advantage(batch, adv_estimator="gae")

    # 8. Update critic
    critic_wg.update_critic(batch)

    # 9. Update actor
    actor_rollout_wg.update_actor(batch)

    # 10. Validate periodically
    if step % test_freq == 0:
        val_metrics = _validate()
```

### Advantage Estimation

Supports multiple estimators:

- **GAE** (Generalized Advantage Estimation): Token-level, uses critic
- **Bi-level GAE**: Hierarchical GAE for multi-turn
- **GRPO** (Group Relative Policy Optimization): Trajectory-level, no critic
- **REINFORCE++**: Variance reduction without critic
- **RLOO** (REINFORCE Leave-One-Out): Group-based baseline

### Rollout Filtering

**Purpose:** Select top-performing trajectory groups to improve data quality.

**Configuration:**
```yaml
actor_rollout_ref:
  rollout:
    rollout_filter_ratio: 0.25  # Keep top 25%
    rollout_filter_type: largest # or smallest
    rollout_filter_metric: reward_variance
```

**Available Metrics:**
- `reward`: Mean reward per group
- `reward_variance`: Variance of rewards (exploration)
- `entropy`: Mean entropy (policy diversity)
- `entropy_variance`: Variance of entropy

---

## Development Workflows

### Setup Environment

```bash
# Quick setup
bash scripts/setup_ragen.sh

# Manual setup (if script fails)
# See scripts/setup_ragen.md for detailed instructions

# Base installation
pip install -e .

# With optional dependencies
pip install -e ".[webshop]"    # WebShop environment
pip install -e ".[lean]"        # Lean environment
pip install -e ".[all]"         # All optional dependencies
```

### Adding a New Environment

#### Step 1: Implement Environment

Create `ragen/env/new_env/env.py`:

```python
from ragen.env.base import BaseEnv
from dataclasses import dataclass

class NewEnv(BaseEnv):
    def __init__(self, config):
        self.config = config
        # Initialize your environment

    def reset(self, seed=None):
        # Reset environment with deterministic seed
        return observation

    def step(self, action):
        # Execute action
        return observation, reward, done, info

    def render(self) -> str:
        # Return current state as string
        return state_description

    def close(self):
        # Cleanup resources
        pass
```

Create `ragen/env/new_env/config.py`:

```python
from dataclasses import dataclass

@dataclass
class NewEnvConfig:
    param1: int = 10
    param2: str = "default"
    # Add all configurable parameters
```

Create `ragen/env/new_env/__init__.py`:

```python
from .env import NewEnv
from .config import NewEnvConfig

__all__ = ['NewEnv', 'NewEnvConfig']
```

#### Step 2: Register Environment

In `ragen/env/__init__.py`:

```python
from ragen.env.new_env import NewEnv, NewEnvConfig

REGISTERED_ENVS['new_env'] = NewEnv
REGISTERED_ENV_CONFIGS['new_env'] = NewEnvConfig
```

#### Step 3: Define in Configuration

In `config/envs.yaml`:

```yaml
custom_envs:
  NewEnvironmentTag:
    env_type: new_env
    max_actions_per_traj: 10
    env_instruction: |
      Task description and instructions for the LLM.
      Explain the goal, available actions, and expected format.
    max_tokens: 100
    parallel_friendly: false  # Set true for thread pool execution
    max_workers: 32           # Threads if parallel_friendly=true
    env_config:
      param1: 20
      param2: "custom_value"
```

#### Step 4: Use in Training

In `config/base.yaml`:

```yaml
es_manager:
  train:
    env_configs:
      tags: ["NewEnvironmentTag"]
      n_groups: [8]
```

### Training a Model

```bash
# Default training
python train.py --config-name base

# Train on specific environment
python train.py --config-name _2_sokoban

# Override parameters
python train.py \
  es_manager.train.env_configs.tags='["SimpleSokoban"]' \
  trainer.total_training_steps=500 \
  ppo_mini_batch_size=64

# LoRA training
python train.py --config-name base-lora

# Lower memory training (for 24GB GPUs)
python train.py \
  micro_batch_size_per_gpu=1 \
  ppo_mini_batch_size=8 \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.response_length=128
```

### Evaluating a Model

```bash
# Evaluate with default config
python -m ragen.llm_agent.agent_proxy --config-name eval

# Evaluate specific checkpoint
python -m ragen.llm_agent.agent_proxy \
  --config-name eval \
  actor_rollout_ref.model.path=results/exp_name/global_step_200/actor

# Limit context window during evaluation
python -m ragen.llm_agent.agent_proxy \
  --config-name eval \
  agent_proxy.max_context_window=5
```

### Modifying Hyperparameters

Common parameters to adjust:

```yaml
# Learning rates
actor_rollout_ref.actor.optim.lr: 1e-6
critic.optim.lr: 1e-5

# Batch sizes
ppo_mini_batch_size: 32
micro_batch_size_per_gpu: 4

# Rollout filtering
actor_rollout_ref.rollout.rollout_filter_ratio: 0.25

# Environment groups
es_manager.train.env_groups: 8
es_manager.train.group_size: 16

# Training duration
trainer.total_training_steps: 200
trainer.test_freq: 10

# Response generation
actor_rollout_ref.rollout.temperature: 1.0
actor_rollout_ref.rollout.response_length: 400
```

---

## Testing Conventions

### Test Organization

```
tests/
├── llm_agent/
│   └── test_context_window.py    # Context truncation
├── env/
│   └── test_sokoban_render.py    # Environment rendering
├── es_manager/
│   └── test_seed_iteration.py    # Seed management
└── test_rollout_filter.py         # Trajectory filtering
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/env/test_sokoban_render.py

# Run with verbose output
pytest -v

# Run specific test function
pytest tests/env/test_sokoban_render.py::test_grid_rendering
```

### Writing Tests

Use pytest with fixtures:

```python
import pytest
from omegaconf import OmegaConf

@pytest.fixture
def dummy_config():
    return OmegaConf.create({
        'param1': 'value1',
        'param2': 10
    })

def test_feature(dummy_config):
    # Arrange
    manager = ContextManager(config=dummy_config)

    # Act
    result = manager.method()

    # Assert
    assert result == expected_value
```

### Test Configuration

**File:** `pytest.ini`

```ini
[pytest]
pythonpath = .
```

---

## Common Tasks

### Task 1: Debugging Rollouts

1. **Enable Logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Visualize Generations:**
   - Check WandB dashboard → `val/generations` metric
   - Or set `trainer.generations_to_log_to_wandb.val` in config

3. **Inspect Rollout Cache:**
   Add breakpoint in `es_manager.get_rollout_states()`:
   ```python
   def get_rollout_states(self):
       breakpoint()  # Inspect self.rollout_cache here
       return self.rollout_cache
   ```

### Task 2: Changing Prompt Format

Modify `ctx_manager.py`:

```python
def get_lm_inputs(self, env_outputs):
    # Modify how observations are formatted into prompts
    for env_output in env_outputs:
        observation = env_output['observation']
        # Custom formatting logic here
        messages.append({
            'role': 'user',
            'content': f"Custom format: {observation}"
        })
    return messages
```

### Task 3: Adding Custom Reward Function

Create reward function file:

```python
# custom_reward.py
def my_custom_reward(data_item):
    # Extract information from data_item
    response = data_item.non_tensor_batch['response']

    # Compute custom reward
    score = compute_score(response)

    return score
```

Add to config:

```yaml
custom_reward_function:
  path: "path/to/custom_reward.py"
  name: "my_custom_reward"
```

### Task 4: Changing Response Parsing

Modify `ctx_manager.py`:

```python
def parse_response(self, response):
    # Default: extract <think>...</think> and <answer>...</answer>
    # Modify to use different tags or parsing logic

    # Example: Use different delimiters
    think_pattern = r'<reasoning>(.*?)</reasoning>'
    answer_pattern = r'<action>(.*?)</action>'

    # Extract and return parsed content
    ...
```

### Task 5: Running Experiments

```bash
# Experiment 1: No KL penalty, top 25% filtering
python train.py \
  algorithm.kl_ctrl.kl_coef=0.0 \
  actor_rollout_ref.rollout.rollout_filter_ratio=0.25 \
  trainer.experiment_name=exp1_no_kl_top25

# Experiment 2: With KL penalty, top 50% filtering
python train.py \
  algorithm.kl_ctrl.kl_coef=0.01 \
  actor_rollout_ref.rollout.rollout_filter_ratio=0.5 \
  trainer.experiment_name=exp2_kl_top50

# Experiment 3: GRPO (no critic)
python train.py \
  algorithm.adv_estimator=grpo \
  agent_proxy.without_history=True \
  trainer.experiment_name=exp3_grpo
```

---

## Key Conventions for AI Assistants

### 1. File Modification Guidelines

**DO:**
- Read files before modifying them
- Preserve existing code style and patterns
- Add comments for complex logic
- Update related tests when changing functionality
- Check configuration files for related settings

**DON'T:**
- Modify core training logic without understanding impact
- Change configuration defaults without documenting
- Add dependencies without updating setup.py
- Break existing API contracts

### 2. Configuration Changes

**ALWAYS:**
- Test configuration changes with `python train.py --config-name base --help`
- Validate batch size constraints (see `add_dependency_and_validate_config` in train.py)
- Check that `env_groups * group_size * rollout_filter_ratio >= ppo_mini_batch_size`
- Ensure `ppo_mini_batch_size` is divisible by `micro_batch_size_per_gpu * n_gpus_per_node`

**NEVER:**
- Set `without_history=True` without adjusting batch size calculations
- Change `max_turn` without considering context length
- Modify seed settings without understanding determinism requirements

### 3. Environment Development

**MUST HAVE:**
- Deterministic `reset(seed)` method
- Consistent `step()` return format: `(obs, reward, done, info)`
- String-based `render()` output
- Proper `close()` cleanup
- Configuration dataclass in `config.py`
- Registration in `ragen/env/__init__.py`

**BEST PRACTICES:**
- Use descriptive action/state spaces
- Provide helpful error messages
- Support multiple observation formats
- Include environment-specific metrics in `info`
- Document expected input/output formats in docstrings

### 4. Code Style

**Follow Existing Patterns:**
- Use dataclasses for configuration
- Type hints for function signatures
- Docstrings for public methods
- Snake_case for functions/variables
- CamelCase for classes
- UPPER_CASE for constants

**Example:**
```python
from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class MyConfig:
    """Configuration for MyComponent."""
    param1: int = 10
    param2: str = "default"

class MyComponent:
    """Component description."""

    def __init__(self, config: MyConfig):
        """Initialize component with config."""
        self.config = config

    def process(self, input_data: List[str]) -> Dict[str, float]:
        """Process input data and return results.

        Args:
            input_data: List of input strings

        Returns:
            Dictionary mapping strings to scores
        """
        # Implementation
        return results
```

### 5. Debugging Best Practices

**For Environment Issues:**
- Test environment in isolation with `python -m ragen.env.new_env.env`
- Check `render()` output matches expected format
- Verify `reset(seed)` is deterministic
- Ensure `step()` handles all edge cases

**For Training Issues:**
- Check WandB logs for NaN/Inf values
- Verify batch size calculations
- Inspect rollout cache during debugging
- Monitor GPU memory usage
- Check for reward scale issues

**For Configuration Issues:**
- Use `--help` to see all available options
- Check Hydra interpolation with `OmegaConf.to_container(config, resolve=True)`
- Verify environment tags match `envs.yaml` definitions
- Ensure paths are absolute, not relative

### 6. Git Workflow

**Committing:**
- Use descriptive commit messages
- Follow format: "Add feature X", "Fix bug in Y", "Update Z docs"
- Group related changes in single commit
- Don't commit experiment results or checkpoints

**Branch Naming:**
- Must start with `claude/` for automated workflows
- Should be descriptive: `claude/add-sudoku-env`

**Files to Never Commit:**
- `results/` directory
- `wandb/` logs
- `*.pt`, `*.ckpt` checkpoints
- `__pycache__/` directories
- Large datasets

### 7. Documentation Updates

**When Adding Features:**
- Update README.md if user-facing
- Update this CLAUDE.md for AI assistant guidance
- Add docstrings to new functions/classes
- Include usage examples in docstrings
- Update configuration comments

**When Fixing Bugs:**
- Document the fix in commit message
- Add test case if possible
- Update documentation if behavior changes

### 8. Common Pitfalls to Avoid

1. **Context Length Overflow:**
   - Always set reasonable `max_model_len` and `response_length`
   - Monitor context usage during long episodes
   - Use `max_context_window` to limit history

2. **Batch Size Mismatches:**
   - Verify all batch size constraints in config
   - Understand relationship between env_groups, group_size, and ppo_mini_batch_size
   - Account for rollout filtering ratio

3. **Seed Management:**
   - Don't modify seed in environment code
   - Let es_manager handle seed progression
   - Use fixed seeds for validation

4. **Memory Issues:**
   - Reduce `max_model_len` for smaller GPUs
   - Decrease `micro_batch_size_per_gpu`
   - Enable gradient checkpointing if needed
   - Use LoRA instead of full fine-tuning

5. **Environment Compatibility:**
   - Don't assume all environments are parallel-friendly
   - Check `parallel_friendly` flag before enabling threading
   - Test with single environment first

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solutions:**
```bash
# Reduce batch sizes
python train.py \
  micro_batch_size_per_gpu=1 \
  ppo_mini_batch_size=8

# Reduce context length
python train.py \
  actor_rollout_ref.rollout.max_model_len=2048 \
  actor_rollout_ref.rollout.response_length=128

# Use LoRA
python train.py --config-name base-lora

# Reduce GPU memory utilization
python train.py \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4
```

### Issue: Batch Size Validation Error

**Error:**
```
AssertionError: ppo_mini_batch_size (32) must be divisible by micro_batch_size_per_gpu * n_gpus_per_node (6)
```

**Solution:**
Adjust batch sizes to satisfy constraints:
```bash
python train.py \
  micro_batch_size_per_gpu=4 \
  ppo_mini_batch_size=32 \
  trainer.n_gpus_per_node=1
```

### Issue: Environment Not Found

**Error:**
```
KeyError: 'new_env' in REGISTERED_ENVS
```

**Solution:**
1. Check `ragen/env/__init__.py` for registration
2. Verify import statement
3. Ensure `env_type` in config matches registered name

### Issue: Rollout Filtering Produces Empty Batches

**Error:**
```
AssertionError: Batch size after filtering is 0
```

**Solution:**
```bash
# Increase rollout_filter_ratio
python train.py \
  actor_rollout_ref.rollout.rollout_filter_ratio=0.5

# Or increase env_groups * group_size
python train.py \
  es_manager.train.env_groups=16 \
  es_manager.train.group_size=16
```

### Issue: NaN/Inf in Training

**Possible Causes:**
- Learning rate too high
- Reward scale too large
- Gradient explosion

**Solutions:**
```bash
# Reduce learning rate
python train.py \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  critic.optim.lr=5e-6

# Enable gradient clipping (in veRL config)
python train.py \
  actor_rollout_ref.actor.optim.grad_clip=1.0

# Normalize rewards
python train.py \
  agent_proxy.reward_normalization.method=mean_std
```

### Issue: Slow Rollouts

**Solutions:**
```bash
# Enable parallel environments (if supported)
# In config/envs.yaml, set parallel_friendly: true

# Increase tensor parallelism
python train.py \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2

# Disable eager execution
python train.py \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False
```

### Issue: Determinism Problems

**Causes:**
- Random seeds not set properly
- Parallel execution with shared RNG state

**Solutions:**
```bash
# Fix training seed
python train.py seed.train=12345

# Disable parallel execution
# In config/envs.yaml, set parallel_friendly: false

# Use deterministic algorithms (slower)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

---

## Additional Resources

### Official Documentation

- **RAGEN Documentation:** https://ragen-doc.readthedocs.io/
- **Paper:** https://arxiv.org/abs/2504.20073
- **Homepage:** https://ragen-ai.github.io/

### Related Projects

- **veRL:** https://github.com/volcengine/verl
- **vLLM:** https://github.com/vllm-project/vllm
- **Hydra:** https://hydra.cc/

### Community

- **GitHub Issues:** https://github.com/RAGEN-AI/RAGEN/issues
- **Discussions:** Check GitHub Discussions tab

---

## Changelog

### 2026-01-04
- Initial comprehensive CLAUDE.md created
- Documented all core components and workflows
- Added detailed troubleshooting guide
- Included environment development guide

---

## Appendix: Quick Reference

### Important Commands

```bash
# Setup
bash scripts/setup_ragen.sh
pip install -e ".[all]"

# Training
python train.py --config-name base
python train.py --config-name _2_sokoban

# Evaluation
python -m ragen.llm_agent.agent_proxy --config-name eval

# Testing
pytest
pytest tests/env/test_sokoban_render.py -v

# Debugging
python -m pdb train.py --config-name base
```

### Important Paths

```
Config files:        /home/user/RAGEN/config/
Core code:           /home/user/RAGEN/ragen/
Environments:        /home/user/RAGEN/ragen/env/
Tests:               /home/user/RAGEN/tests/
Results:             /home/user/RAGEN/results/  (gitignored)
veRL submodule:      /home/user/RAGEN/verl/
```

### Key Constraints

```
env_groups * group_size * rollout_filter_ratio >= ppo_mini_batch_size
ppo_mini_batch_size % (micro_batch_size_per_gpu * n_gpus_per_node) == 0
max_model_len >= prompt_length + response_length
```

---

**End of CLAUDE.md**
