#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FIGURE_DIR = SCRIPT_DIR / "figure"

RUN_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
]

METRICS = [
    ("performance", "Performance"),
    ("entropy", "Entropy"),
    ("output length", "Output Length"),
]


@dataclass
class RunData:
    label: str
    frame: pd.DataFrame
    color: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot performance / entropy / output length curves from one or more "
            "CSV files. Each input must contain the columns: step, performance, "
            "entropy, output length."
        ),
        epilog=(
            "Example: python draw_RL_length_change/plot_wandb_curves.py run1.csv\n"
            "Example: python draw_RL_length_change/plot_wandb_curves.py run1.csv run2.csv "
            "--labels baseline reasoning --output compare.png"
        ),
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        type=Path,
        help="One or more CSV files to plot.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Optional labels for each CSV. Defaults to each file stem.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional PNG output file. If omitted, save to "
            "draw_RL_length_change/figure/<project_name>/ with a filename "
            "derived from the input CSV names."
        ),
    )
    parser.add_argument(
        "--title",
        default="Performance / Entropy / Output Length",
        help="Figure title.",
    )
    return parser.parse_args()


def validate_columns(frame: pd.DataFrame, source: str) -> None:
    required = {"step"} | {metric_name for metric_name, _ in METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {', '.join(missing)}"
        )


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame[["step", *[metric_name for metric_name, _ in METRICS]]].copy()
    for metric_name in normalized.columns:
        normalized[metric_name] = pd.to_numeric(
            normalized[metric_name], errors="coerce"
        )
    normalized = normalized.dropna(subset=["step"])
    normalized = normalized.dropna(
        how="all", subset=[metric_name for metric_name, _ in METRICS]
    )
    return normalized.sort_values("step").reset_index(drop=True)


def load_runs(csv_paths: list[Path], labels: list[str] | None) -> list[RunData]:
    if labels and len(labels) != len(csv_paths):
        raise ValueError("Number of --labels must match the number of input files.")

    runs: list[RunData] = []
    resolved_labels = labels or [path.stem for path in csv_paths]

    for index, (path, label) in enumerate(zip(csv_paths, resolved_labels)):
        frame = pd.read_csv(path)
        validate_columns(frame, str(path))
        runs.append(
            RunData(
                label=label,
                frame=normalize_frame(frame),
                color=RUN_COLORS[index % len(RUN_COLORS)],
            )
        )
    return runs


def sanitize_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "plot"


def infer_project_name(csv_paths: list[Path]) -> str:
    project_names: list[str] = []

    for path in csv_paths:
        resolved = path.resolve()
        parts = resolved.parts
        project_name = resolved.parent.name

        if "csv_data" in parts:
            csv_data_index = parts.index("csv_data")
            if csv_data_index + 1 < len(parts):
                project_name = parts[csv_data_index + 1]

        project_names.append(project_name)

    unique_names = sorted(set(project_names))
    if len(unique_names) == 1:
        return sanitize_filename_part(unique_names[0])
    return "mixed_projects"


def build_default_output_path(csv_paths: list[Path]) -> Path:
    project_name = infer_project_name(csv_paths)
    name_parts = [sanitize_filename_part(path.stem) for path in csv_paths]
    filename = "__vs__".join(name_parts) if len(name_parts) > 1 else name_parts[0]
    return DEFAULT_FIGURE_DIR / project_name / f"{filename}.png"


def plot_curves(runs: list[RunData], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=1,
        figsize=(11, 9),
        sharex=True,
        constrained_layout=True,
    )

    for axis, (metric_name, metric_label) in zip(axes, METRICS):
        for run in runs:
            axis.plot(
                run.frame["step"],
                run.frame[metric_name],
                label=run.label,
                color=run.color,
                linewidth=2,
            )

        axis.set_ylabel(metric_label)
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].legend(frameon=False, loc="best")
    axes[-1].set_xlabel("Step")
    fig.suptitle(title, fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = load_runs(args.csv_paths, args.labels)
    output_path = args.output or build_default_output_path(args.csv_paths)
    plot_curves(runs, output_path, args.title)
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
