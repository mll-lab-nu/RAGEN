import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _load_metrics(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _bucket_sort_key(bucket: str) -> Tuple[int, int, str]:
    if bucket.startswith("bucket_"):
        suffix = bucket.split("_", 1)[1]
        if suffix.isdigit():
            return (0, int(suffix), bucket)
    return (1, 0, bucket)


def _extract_buckets(metrics: Dict) -> List[str]:
    buckets = set()
    for k in metrics.keys():
        if k.startswith("grad_norm/bucket_"):
            parts = k.split("/")
            if len(parts) >= 2:
                buckets.add(parts[1])
    return sorted(buckets, key=_bucket_sort_key)


def _rv_stats(metrics: Dict, buckets: List[str]) -> Tuple[List[float], List[float], List[float]]:
    means, mins, maxs = [], [], []
    for b in buckets:
        means.append(float(metrics.get(f"grad_norm/{b}/reward_std_mean", 0.0)))
        mins.append(float(metrics.get(f"grad_norm/{b}/reward_std_min", 0.0)))
        maxs.append(float(metrics.get(f"grad_norm/{b}/reward_std_max", 0.0)))
    return means, mins, maxs


def _grad_series(metrics: Dict, buckets: List[str]) -> Tuple[List[float], List[float], List[float]]:
    task = []
    kl = []
    ent = []
    for b in buckets:
        task.append(float(metrics.get(f"grad_norm/{b}/task", 0.0)))
        kl.append(float(metrics.get(f"grad_norm/{b}/kl", 0.0)))
        ent.append(float(metrics.get(f"grad_norm/{b}/entropy", 0.0)))
    return task, kl, ent


def _default_step_dir(mode: str, step: str) -> str:
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, "data", mode, step)


def main() -> None:
    parser = argparse.ArgumentParser(description="ICML paper plots: step 0/20/40 grid.")
    parser.add_argument("--mode", choices=["grpo", "ppo"], default="grpo", help="Which dataset to plot")
    parser.add_argument("--step0-dir", default=None, help="Directory with metrics json for step 0")
    parser.add_argument("--step20-dir", default=None, help="Directory with metrics json for step 20")
    parser.add_argument("--step40-dir", default=None, help="Directory with metrics json for step 40")
    parser.add_argument("--out", default="icml_step0_20_40_grid.png", help="Output PNG path")
    args = parser.parse_args()

    step0_dir = args.step0_dir or _default_step_dir(args.mode, "step0")
    step20_dir = args.step20_dir or _default_step_dir(args.mode, "step20")
    step40_dir = args.step40_dir or _default_step_dir(args.mode, "step40")

    metrics0 = _load_metrics(os.path.join(step0_dir, "metrics.json"))
    metrics20 = _load_metrics(os.path.join(step20_dir, "metrics.json"))
    metrics40 = _load_metrics(os.path.join(step40_dir, "metrics.json"))

    buckets = _extract_buckets(metrics20)
    buckets = [b for b in buckets if b in _extract_buckets(metrics40)]
    buckets = [b for b in buckets if b in _extract_buckets(metrics0)]
    labels = [b.replace("_", " ") for b in buckets]

    rv20_means, rv20_mins, rv20_maxs = _rv_stats(metrics20, buckets)
    rv40_means, rv40_mins, rv40_maxs = _rv_stats(metrics40, buckets)
    task20, kl20, ent20 = _grad_series(metrics20, buckets)
    task40, kl40, ent40 = _grad_series(metrics40, buckets)
    reg20 = [k + e for k, e in zip(kl20, ent20)]
    reg40 = [k + e for k, e in zip(kl40, ent40)]
    rv0_means, rv0_mins, rv0_maxs = _rv_stats(metrics0, buckets)
    task0, kl0, ent0 = _grad_series(metrics0, buckets)
    reg0 = [k + e for k, e in zip(kl0, ent0)]

    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex="col")
    color_rv = "#1f78b4"
    color_task = "#e67e22"
    color_reg = "#16a085"

    positions = np.arange(len(buckets))
    box_width = 0.35
    def _draw_interval_mean(ax, x, vmin, vmax, vmean, color):
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        yerr = [[max(0.0, vmean - vmin)], [max(0.0, vmax - vmean)]]
        ax.errorbar(
            [x],
            [vmean],
            yerr=yerr,
            fmt="o",
            color=color,
            markersize=5,
            capsize=4,
            linewidth=1.2,
        )

    # rows: step0 (if provided), step20, step40
    steps = [
        ("Step 0", rv0_means, rv0_mins, rv0_maxs, task0, reg0),
        ("Step 20", rv20_means, rv20_mins, rv20_maxs, task20, reg20),
        ("Step 40", rv40_means, rv40_mins, rv40_maxs, task40, reg40),
    ]

    col_titles = [
        "Reward Variance by bucket",
        "Task gradient norm vs Reward Variance",
        "Regularizer gradient norm (KL+Entropy) vs RV",
    ]
    col_captions = [
        "RV quantile buckets. (Q1 -> Q6)",
        "Bucket RV (log scale).",
        "Bucket RV (log scale).",
    ]

    for r, (step_name, rv_means, rv_mins, rv_maxs, task, reg) in enumerate(steps):
        # (a) RV per bucket interval + mean
        ax = axes[r][0]
        for i, x in enumerate(positions):
            _draw_interval_mean(
                ax,
                x,
                rv_mins[i],
                rv_maxs[i],
                rv_means[i],
                color=color_rv,
            )
        ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.15, linewidth=0.8)
        ax.set_ylabel(f"{step_name}\nReward Variance (Std)")
        if r == 0:
            ax.set_title("(a) " + col_titles[0], fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels if r == len(steps) - 1 else [])

        # (b) Task vs RV
        ax = axes[r][1]
        ax.plot(rv_means, task, linestyle="-", marker="o", color=color_task, markersize=5)
        ax.set_xscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.15, linewidth=0.8)
        ax.set_ylabel(f"{step_name}\nTask grad norm")
        if r == 0:
            ax.set_title("(b) " + col_titles[1], fontweight="bold")
        # if r == len(steps) - 1:
        #     ax.set_xlabel("RV mean")

        # (c) Reg vs RV
        ax = axes[r][2]
        ax.plot(rv_means, reg, linestyle="-", marker="o", color=color_reg, markersize=5)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 0.1)
        ax.grid(axis="y", linestyle="--", alpha=0.15, linewidth=0.8)
        ax.set_ylabel(f"{step_name}\nKL+Entropy grad norm")
        if r == 0:
            ax.set_title("(c) " + col_titles[2], fontweight="bold")
        # if r == len(steps) - 1:
        #     ax.set_xlabel("RV mean")

    # style spines
    for row in axes:
        for a in row:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

    # captions under each column
    for c, caption in enumerate(col_captions):
        ax = axes[-1][c]
        ax.text(
            0.5,
            -0.15,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved figure to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
