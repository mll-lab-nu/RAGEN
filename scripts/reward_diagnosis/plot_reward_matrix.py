#!/usr/bin/env python3
"""
Plot the reward triangular matrix from multi-rollout inference results.

Reads the JSON output from run_inference.py and generates a heatmap where:
  - Each row = one prompt, rollout rewards sorted high-to-low within the row
  - Rows sorted by mean reward descending (easy on top, hard on bottom)
  - Color: reward value (0 = red, 1 = green)
  - Region labels and RV annotations on the right margin
  - Classification: Easy (mean >= threshold), Hard (mean <= threshold), Mixed (in between)

The resulting upper-triangular shape shows:
  - Top rows: easy prompts (all green) — too easy, no RL signal
  - Middle rows: mixed prompts (left green, right red) — learnable
  - Bottom rows: hard prompts (all red) — too hard or broken

Usage:
    python scripts/reward_diagnosis/plot_reward_matrix.py \
        --input logs/inference_results.json \
        --output logs/reward_matrix.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    print("matplotlib is required: pip install matplotlib")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Plot reward triangular matrix")
    parser.add_argument("--input", required=True, help="Path to inference JSON from run_inference.py")
    parser.add_argument("--output", default=None, help="Output image path (default: <input_stem>_matrix.png)")
    parser.add_argument("--max_prompts", type=int, default=100, help="Max prompts to display")
    parser.add_argument("--easy_threshold", type=float, default=0.8,
                        help="Mean reward >= this is classified as Easy")
    parser.add_argument("--hard_threshold", type=float, default=0.2,
                        help="Mean reward <= this is classified as Hard")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    # Load data
    with open(args.input) as f:
        data = json.load(f)

    config = data["config"]
    prompts = data["prompts"]
    summary = data["summary"]
    n_rollouts = config["rollouts_per_prompt"]

    # Build matrix: each row = sorted rewards (descending) for one prompt
    reward_rows = []
    rv_values = []

    for p in prompts:
        rewards = sorted(p["rewards"], reverse=True)
        reward_rows.append(rewards)
        rv_values.append(p["reward_variance"])

    # Sort by mean reward descending (easy on top, hard on bottom → upper triangle)
    mean_rewards = [np.mean(row) for row in reward_rows]
    sort_idx = np.argsort(mean_rewards)[::-1]
    reward_rows = [reward_rows[i] for i in sort_idx]
    rv_values = [rv_values[i] for i in sort_idx]
    mean_rewards = [mean_rewards[i] for i in sort_idx]

    # Truncate for display
    n_display = min(len(reward_rows), args.max_prompts)
    reward_rows = reward_rows[:n_display]
    rv_values = rv_values[:n_display]

    matrix = np.array(reward_rows)

    # Classify by mean reward thresholds
    n_easy = sum(1 for m in mean_rewards if m >= args.easy_threshold)
    n_hard = sum(1 for m in mean_rewards if m <= args.hard_threshold)
    n_mixed = n_display - n_easy - n_hard
    n_other = 0

    # ── Figure layout ──
    fig_w = 10
    fig_h = max(5, n_display * 0.18 + 2.5)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#fafafa")

    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1, 0.04],
        height_ratios=[1, 0.08],
        hspace=0.35, wspace=0.08,
        left=0.08, right=0.85, top=0.88, bottom=0.08,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_summary = fig.add_subplot(gs[1, :])

    # ── Colormap ──
    cmap = LinearSegmentedColormap.from_list(
        "reward",
        [
            (0.0, "#c62828"),   # deep red
            (0.15, "#e53935"),  # red
            (0.35, "#ff8f00"),  # amber
            (0.50, "#fdd835"),  # yellow
            (0.65, "#7cb342"),  # light green
            (0.85, "#388e3c"),  # green
            (1.0, "#1b5e20"),   # deep green
        ],
    )

    # ── Main heatmap ──
    im = ax_main.imshow(
        matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1,
        interpolation="nearest",
    )

    # Cell value annotations (only if matrix is small enough)
    if n_display <= 40 and n_rollouts <= 16:
        for i in range(n_display):
            for j in range(n_rollouts):
                val = matrix[i, j]
                color = "white" if val < 0.4 or val > 0.8 else "black"
                ax_main.text(j, i, f"{val:.1f}", ha="center", va="center",
                             fontsize=6, color=color, fontweight="bold")

    # X axis
    ax_main.set_xlabel("Rollouts (sorted high → low)", fontsize=10, labelpad=8)
    ax_main.set_xticks(range(n_rollouts))
    ax_main.set_xticklabels([str(i + 1) for i in range(n_rollouts)], fontsize=8)
    ax_main.xaxis.set_ticks_position("bottom")

    # Y axis
    ax_main.set_ylabel("Prompts (sorted by mean reward ↓)", fontsize=10, labelpad=8)
    if n_display <= 50:
        ax_main.set_yticks(range(n_display))
        ax_main.set_yticklabels(range(1, n_display + 1), fontsize=6)
    else:
        step = max(1, n_display // 25)
        ticks = list(range(0, n_display, step))
        ax_main.set_yticks(ticks)
        ax_main.set_yticklabels([i + 1 for i in ticks], fontsize=7)

    # ── Region brackets on the right ──
    region_x = n_rollouts - 0.5 + 0.6  # just outside the matrix
    bracket_style = dict(fontsize=8, va="center", ha="left", fontweight="bold")

    # RV labels on the right of each row
    for i in range(n_display):
        rv_color = "#1565c0" if args.hard_threshold < mean_rewards[i] < args.easy_threshold else (
            "#2e7d32" if mean_rewards[i] >= args.easy_threshold else "#c62828")
        ax_main.text(
            n_rollouts - 0.5 + 0.3, i, f"{rv_values[i]:.3f}",
            fontsize=5, va="center", ha="left", color=rv_color,
            clip_on=False,
        )

    # Region label header
    ax_main.text(
        n_rollouts - 0.5 + 0.3, -0.8, "RV",
        fontsize=6, va="center", ha="left", color="#424242",
        fontweight="bold", clip_on=False,
    )

    # Divider lines between regions
    # Find boundaries: mixed (0.2 < mean < 0.8), easy (mean >= 0.8), hard (mean <= 0.2)
    # Since sorted by RV desc, regions may not be contiguous, so draw lines at transitions
    for i in range(n_display - 1):
        cat_i = "mixed" if args.hard_threshold < mean_rewards[i] < args.easy_threshold else ("easy" if mean_rewards[i] >= args.easy_threshold else "hard")
        cat_next = "mixed" if args.hard_threshold < mean_rewards[i+1] < args.easy_threshold else ("easy" if mean_rewards[i+1] >= args.easy_threshold else "hard")
        if cat_i != cat_next:
            ax_main.axhline(y=i + 0.5, color="#455a64", linewidth=1.0, linestyle="--", alpha=0.5)

    # Grid lines
    ax_main.set_xticks([x - 0.5 for x in range(1, n_rollouts)], minor=True)
    ax_main.set_yticks([y - 0.5 for y in range(1, n_display)], minor=True)
    ax_main.grid(which="minor", color="#e0e0e0", linewidth=0.3)
    ax_main.tick_params(which="minor", length=0)

    # ── Colorbar ──
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Reward", fontsize=9, labelpad=8)
    cbar.ax.tick_params(labelsize=8)

    # ── Title ──
    model_short = config["model"].split("/")[-1]
    fig.suptitle(
        f"Reward Matrix — {model_short}",
        fontsize=14, fontweight="bold", y=0.96,
    )
    ax_main.set_title(
        f"{n_display} prompts × {n_rollouts} rollouts  |  temp = {config['temperature']}",
        fontsize=10, color="#616161", pad=10,
    )

    # ── Summary bar at bottom ──
    ax_summary.set_xlim(0, 1)
    ax_summary.set_ylim(0, 1)
    ax_summary.axis("off")

    # Stacked bar showing proportions
    bar_y, bar_h = 0.55, 0.35
    segments = []
    if n_mixed > 0:
        segments.append((n_mixed / n_display, "#1565c0", f"Mixed: {n_mixed} ({n_mixed/n_display*100:.0f}%)"))
    if n_other > 0:
        segments.append((n_other / n_display, "#78909c", f"Other: {n_other}"))
    if n_easy > 0:
        segments.append((n_easy / n_display, "#43a047", f"Easy: {n_easy} ({n_easy/n_display*100:.0f}%)"))
    if n_hard > 0:
        segments.append((n_hard / n_display, "#e53935", f"Hard: {n_hard} ({n_hard/n_display*100:.0f}%)"))

    x_pos = 0.05
    bar_total_w = 0.6
    for frac, color, label in segments:
        w = frac * bar_total_w
        rect = FancyBboxPatch(
            (x_pos, bar_y), w, bar_h,
            boxstyle="round,pad=0.01", facecolor=color, edgecolor="white", linewidth=1.5,
        )
        ax_summary.add_patch(rect)
        if w > 0.05:
            ax_summary.text(x_pos + w / 2, bar_y + bar_h / 2, label,
                            ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        x_pos += w

    # Diagnosis text
    mean_rv_mixed = np.mean([rv for rv in rv_values if rv > 0]) if n_mixed > 0 else 0
    diag_x = 0.72
    ax_summary.text(diag_x, 0.85, f"Mean reward: {summary['mean_reward']:.3f}",
                    fontsize=8, color="#424242", transform=ax_summary.transAxes)
    ax_summary.text(diag_x, 0.55, f"Mean RV (mixed): {mean_rv_mixed:.4f}",
                    fontsize=8, color="#424242", transform=ax_summary.transAxes)

    mixed_pct = n_mixed / max(n_display, 1) * 100
    if mixed_pct >= 20:
        verdict = "✓ Good RL signal"
        verdict_color = "#2e7d32"
    elif mixed_pct >= 10:
        verdict = "~ Weak RL signal"
        verdict_color = "#f57f17"
    else:
        verdict = "✗ Poor RL signal"
        verdict_color = "#c62828"
    ax_summary.text(diag_x, 0.2, verdict,
                    fontsize=9, color=verdict_color, fontweight="bold", transform=ax_summary.transAxes)

    # ── Save ──
    if args.output is None:
        output_path = Path(args.input).with_name(Path(args.input).stem + "_matrix.png")
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved reward matrix to {output_path}")

    # Text summary
    print(f"\n{'=' * 60}")
    print(f"REWARD MATRIX SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model:           {config['model']}")
    print(f"Prompts:         {n_display}")
    print(f"Rollouts/prompt: {n_rollouts}")
    print(f"Temperature:     {config['temperature']}")
    print()
    print(f"Mixed (RV > 0):  {n_mixed:4d}  ({n_mixed/n_display*100:5.1f}%)  <- RL can learn from these")
    print(f"All correct:     {n_easy:4d}  ({n_easy/n_display*100:5.1f}%)  <- too easy, no signal")
    print(f"All wrong:       {n_hard:4d}  ({n_hard/n_display*100:5.1f}%)  <- too hard or broken")
    print()
    print(f"Mean RV (mixed only): {mean_rv_mixed:.4f}")
    print(f"Overall mean reward:  {summary['mean_reward']:.4f}")
    print()

    if n_hard / max(n_display, 1) > 0.5:
        print("DIAGNOSIS: >50% prompts are all-wrong.")
        print("  -> Check: Is the environment set up correctly?")
        print("  -> Check: Does the model understand the expected action format?")
    elif n_easy / max(n_display, 1) > 0.5:
        print("DIAGNOSIS: >50% prompts are all-correct.")
        print("  -> Task may be too easy. Consider harder subset or lower temperature.")
    elif n_mixed / max(n_display, 1) < 0.2:
        print("DIAGNOSIS: <20% prompts have mixed rewards. Weak RL signal.")
        print("  -> Adjust temperature, check environment setup, or use different data.")
    else:
        print(f"DIAGNOSIS: Good RL signal. {n_mixed/n_display*100:.0f}% prompts are learnable.")
        print("  -> Proceed to training.")


if __name__ == "__main__":
    main()
