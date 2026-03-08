# MAGEN Curve Data

This folder stores curve data used for plotting RL training trends on the DrawSokoban task.

## Data Source

The CSV files in this directory are extracted from training logs for two reward settings:

- `dual_sokoban_Bi-level-GAE_with_reasoning_reward.csv`
- `dual_sokoban_Bi-level-GAE_without_reasoning_reward.csv`

Each CSV contains the following columns:

- `step`: training step
- `performance`: `train/trajectory_success`
- `entropy`: `actor/entropy_loss`
- `output length`: `response_length/mean`

These files are used to visualize how performance, entropy, and output length change during training.

## Task Setting

- Task: `DrawSokoban`
- Environment: multi-agent environment
- Number of agents: 2

## Training Setting

- Model: `Qwen 2.5-VL-7B-Instruct`
- Batch size: `128`
- Training steps: `200`

