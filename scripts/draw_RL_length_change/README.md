# Draw RL Length Change

This directory is for storing project-level CSV curve data and plotting figures.

## Data Organization

- Store all data under the CSV data folder: csv_data
- Create one folder per project: `csv_data/<project_name>/`
- Each project folder should contain:
  - one or more CSV files
  - one `README.md` describing the task and the data source

Example:

```text
csv_data/
  magen/
    README.md
    run_a.csv
    run_b.csv
```

## CSV Format

To use the plotting script in this repo, each CSV should contain:

- `step`
- `performance`
- `entropy`
- `output length`

Project-specific notes should be written in that project's `README.md`.

## Plotting

```bash
python scripts/draw_RL_length_change/plot_wandb_curves.py \
  draw_RL_length_change/csv_data/<project_name>/run_a.csv
```

```bash
python scripts/draw_RL_length_change/plot_wandb_curves.py \
  draw_RL_length_change/csv_data/<project_name>/run_a.csv \
  draw_RL_length_change/csv_data/<project_name>/run_b.csv
```

The script plots `Performance`, `Entropy`, and `Output Length` against `step`.
By default, figures are saved under `figure/<project_name>/`.
