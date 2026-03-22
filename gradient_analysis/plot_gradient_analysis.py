import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import wandb

DEFAULT_BUCKETS = [
    "bucket_1",
    "bucket_2",
    "bucket_3",
    "bucket_4",
    "bucket_5",
    "bucket_6",
]
COMPONENTS = ["kl", "entropy", "task"]
LOSS_COMPONENTS = ["policy", "entropy", "kl", "total"]

def _sanitize_dir_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)

def _bucket_sort_key(bucket_name: str):
    if bucket_name.startswith("bucket_"):
        suffix = bucket_name.split("_", 1)[1]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, bucket_name)

def _extract_buckets(metric_source: dict) -> list[str]:
    buckets = set()
    for key in metric_source.keys():
        if not key.startswith("grad_norm/bucket_"):
            continue
        parts = key.split("/")
        if len(parts) >= 2:
            buckets.add(parts[1])
    if not buckets:
        return DEFAULT_BUCKETS
    return sorted(buckets, key=_bucket_sort_key)

def get_bucket_label(bucket_name):
    """Formats bucket names for the plot axis."""
    if bucket_name.startswith("bucket_"):
        return bucket_name.replace("_", " ")
    return bucket_name

def main():
    parser = argparse.ArgumentParser(
        description="Plot gradient-analysis metrics from a W&B run.",
        epilog=(
            "Examples:\n"
            "  python gradient_analysis/plot_gradient_analysis.py --wandb-path entity/project/run_id\n"
            "  python gradient_analysis/plot_gradient_analysis.py --wandb-path entity/project/run_id --step 1\n"
            "  python gradient_analysis/plot_gradient_analysis.py --wandb-path entity/project/run_id "
            "--output-dir gradient_analysis_outputs/my_run\n"
            "  python gradient_analysis/plot_gradient_analysis.py --wandb-path entity/project/run_id --list-steps"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--wandb-path",
        required=True,
        help="W&B run path like entity/project/run_id",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for generated plots and exported metrics. "
            "Defaults to gradient_analysis_outputs/<run_name>_<run_id>."
        ),
    )
    parser.add_argument(
        "--step",
        dest="steps",
        type=int,
        nargs="+",
        default=None,
        help="One or more training steps to plot. Default: all available gradient-analysis steps.",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List available gradient-analysis steps in the run and exit.",
    )
    args = parser.parse_args()

    print(f"Connecting to WandB run: {args.wandb_path}...")
    api = wandb.Api()
    try:
        run = api.run(args.wandb_path)
    except Exception as e:
        print(f"Error accessing run: {e}")
        return

    summary = run.summary
    step_metrics = {}
    available_steps = set()
    for row in run.scan_history():
        bucket_items = [(k, v) for k, v in row.items() if k.startswith("grad_norm/bucket_")]
        has_bucket_metrics = any(v is not None for _, v in bucket_items)
        has_nonzero_bucket_metrics = any((v is not None and v != 0) for _, v in bucket_items)
        if not has_bucket_metrics or not has_nonzero_bucket_metrics:
            continue
        step = row.get("_step")
        if step is None:
            continue
        available_steps.add(step)
        if step not in step_metrics:
            step_metrics[step] = {}
        for k, v in row.items():
            if v is None:
                continue
            step_metrics[step][k] = v

    if available_steps:
        print(f"Found grad_norm bucket metrics at steps: {sorted(available_steps)}")
    else:
        print("Warning: no grad_norm metrics found in history; falling back to run summary.")
        step_metrics = {"summary": summary}

    if args.list_steps:
        if available_steps:
            print("Available gradient-analysis steps:")
            for step in sorted(available_steps):
                print(step)
        else:
            print("No gradient-analysis steps found.")
        return

    default_dir = os.path.join(
        "gradient_analysis_outputs",
        f"{_sanitize_dir_name(run.name)}_{run.id}",
    )
    output_dir = args.output_dir or default_dir
    os.makedirs(output_dir, exist_ok=True)
    titles = {
        "kl": "KL Gradient Norm",
        "entropy": "Entropy Gradient Norm",
        "task": "Task (Policy) Gradient Norm"
    }
    
    colors = ["#3498db", "#2ecc71", "#e74c3c"] # Blue, Green, Red
    loss_titles = {
        "policy": "Policy (Task) Loss",
        "entropy": "Entropy Loss",
        "kl": "KL Loss",
        "total": "Total Loss",
    }
    loss_colors = ["#8e44ad", "#27ae60", "#2980b9", "#c0392b"]  # Purple, Green, Blue, Red
    norm_titles = {
        "kl": "KL Grad Norm (Per Sample vs Per Token)",
        "entropy": "Entropy Grad Norm (Per Sample vs Per Token)",
        "task": "Task Grad Norm (Per Sample vs Per Token)",
    }

    steps_to_plot = sorted(step_metrics.keys(), key=lambda x: (isinstance(x, str), x))
    if args.steps is not None:
        requested_steps = set(args.steps)
        steps_to_plot = [s for s in steps_to_plot if s in requested_steps]
        if not steps_to_plot:
            print(f"Error: none of the requested steps {sorted(requested_steps)} were found.")
            return

    for step_key in steps_to_plot:
        metric_source = step_metrics[step_key]
        buckets = _extract_buckets(metric_source)
        x_labels = [get_bucket_label(b) for b in buckets]
        step_tag = f"step_{step_key}"
        output_file = os.path.join(output_dir, f"gradient_analysis_plots_{step_tag}.png")
        output_file_loss = os.path.join(output_dir, f"gradient_analysis_loss_plots_{step_tag}.png")
        output_file_rv = os.path.join(output_dir, f"gradient_analysis_reward_std_{step_tag}.png")
        output_file_normed = os.path.join(output_dir, f"gradient_analysis_normed_grads_{step_tag}.png")
        output_file_summary = os.path.join(output_dir, f"gradient_analysis_summary_{step_tag}.png")
        output_metrics_json = os.path.join(output_dir, f"gradient_analysis_metrics_{step_tag}.json")
        output_rv_table = os.path.join(output_dir, f"gradient_analysis_bucket_rv_table_{step_tag}.csv")

        # Create subplots for gradient norms
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        plt.subplots_adjust(wspace=0.3, top=0.62, bottom=0.12)

        bucket_rv = {
            b: {
                "mean": metric_source.get(f"grad_norm/{b}/reward_std_mean", 0),
                "min": metric_source.get(f"grad_norm/{b}/reward_std_min", 0),
                "max": metric_source.get(f"grad_norm/{b}/reward_std_max", 0),
            }
            for b in buckets
        }
        bucket_rv_values = {b: [] for b in buckets}
        table_keys = [f"grad_norm/{b}/group_rv_table" for b in buckets]
        for row in run.scan_history(keys=["_step", *table_keys]):
            if row.get("_step") != step_key:
                continue
            for b in buckets:
                key = f"grad_norm/{b}/group_rv_table"
                table_meta = row.get(key)
                if not isinstance(table_meta, dict) or "path" not in table_meta:
                    continue
                table_path = table_meta["path"]
                try:
                    file_ref = run.file(table_path)
                    local_path = file_ref.download(replace=True).name
                    with open(local_path, "r") as f:
                        table_json = json.load(f)
                    # table_json has keys: columns, data
                    data_rows = table_json.get("data", [])
                    # columns: bucket, group_id, reward_std
                    for row_vals in data_rows:
                        if len(row_vals) >= 3:
                            bucket_rv_values[b].append(float(row_vals[2]))
                except Exception:
                    continue
        # Save raw metric snapshot and RV table values for this step
        try:
            with open(output_metrics_json, "w") as f:
                json.dump(metric_source, f, indent=2, sort_keys=True)
        except Exception as e:
            print(f"Warning: failed to write metrics json: {e}")

        try:
            with open(output_rv_table, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["bucket", "reward_std"])
                for b in buckets:
                    for rv in bucket_rv_values[b]:
                        writer.writerow([b, rv])
        except Exception as e:
            print(f"Warning: failed to write rv table csv: {e}")
        legend_lines = []
        for label, bucket in zip(x_labels, buckets):
            rv = bucket_rv.get(bucket, {})
            legend_lines.append(
                f"{label}: mean={rv.get('mean', 0):.3f} min={rv.get('min', 0):.3f} max={rv.get('max', 0):.3f}"
            )
        for ax, comp, color in zip(axes, COMPONENTS, colors):
            y_values = []
            for bucket in buckets:
                key = f"grad_norm/{bucket}/{comp}"
                val = metric_source.get(key, 0)
                y_values.append(val)

            bars = ax.bar(x_labels, y_values, color=color, alpha=0.8, edgecolor='black', linewidth=1)
            ax.set_title(titles[comp], fontsize=16, fontweight='bold', pad=15)
            ax.set_ylabel("Grad Norm Magnitude", fontsize=12)
            ax.set_xlabel("Reward Variance Bucket", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.6)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (max(y_values)*0.01 if y_values else 0.01),
                        f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        fig.suptitle(f"Gradient Norms - Run: {run.name} (Step {step_key})", fontsize=20, y=0.98)
        fig.text(0.5, 0.88, "\n".join(legend_lines), ha="center", va="top", fontsize=8)
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"\nSuccess! Results visualization saved to: {os.path.abspath(output_file)}")
        plt.close(fig)

        # Create subplots for per-component losses
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
        plt.subplots_adjust(hspace=0.35, wspace=0.25)

        for ax, comp, color in zip(axes2.flatten(), LOSS_COMPONENTS, loss_colors):
            y_values = []
            for bucket in buckets:
                key = f"grad_norm/{bucket}/loss/{comp}"
                val = metric_source.get(key, 0)
                y_values.append(val)

            bars = ax.bar(x_labels, y_values, color=color, alpha=0.8, edgecolor="black", linewidth=1)
            ax.set_title(loss_titles[comp], fontsize=14, fontweight="bold", pad=10)
            ax.set_ylabel("Loss", fontsize=11)
            ax.set_xlabel("Reward Variance Bucket", fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.6)

            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (max(y_values) * 0.01 if y_values else 0.01),
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        fig2.suptitle(f"Per-Component Losses - Run: {run.name} (Step {step_key})", fontsize=18, y=1.02)
        plt.tight_layout()
        plt.savefig(output_file_loss, bbox_inches="tight", dpi=300)
        print(f"Success! Loss visualization saved to: {os.path.abspath(output_file_loss)}")
        plt.close(fig2)

    # Create plot for per-bucket mean reward variance (std)
        rv_values = []
        for bucket in buckets:
            rv_values.append(metric_source.get(f"grad_norm/{bucket}/reward_std_mean", 0))
        if any(v != 0 for v in rv_values):
            fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
            bars = ax3.bar(x_labels, rv_values, color="#f39c12", alpha=0.85, edgecolor="black", linewidth=1)
            ax3.set_title(f"Reward Std Mean by Bucket - Run: {run.name} (Step {step_key})", fontsize=14, fontweight="bold", pad=10)
            ax3.set_ylabel("Reward Std (Mean)", fontsize=11)
            ax3.set_xlabel("Reward Variance Bucket", fontsize=11)
            ax3.grid(axis="y", linestyle="--", alpha=0.6)
            for bar in bars:
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (max(rv_values) * 0.01 if rv_values else 0.01),
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
            plt.tight_layout()
            plt.savefig(output_file_rv, bbox_inches="tight", dpi=300)
            print(f"Success! Reward std visualization saved to: {os.path.abspath(output_file_rv)}")
            plt.close(fig3)
        else:
            print(f"Warning: No reward std mean metrics found at step {step_key}; skipping reward std plot.")

        # Create plots for per-sample and per-token grad norms (combined per component)
        fig4, axes4 = plt.subplots(1, 3, figsize=(20, 6))
        plt.subplots_adjust(wspace=0.3)
        for ax, comp in zip(axes4, COMPONENTS):
            per_sample = []
            per_token = []
            for bucket in buckets:
                per_sample.append(metric_source.get(f"grad_norm/{bucket}/per_sample/{comp}", 0))
                per_token.append(metric_source.get(f"grad_norm/{bucket}/per_token/{comp}", 0))

            x = range(len(x_labels))
            width = 0.38
            bars1 = ax.bar([i - width / 2 for i in x], per_sample, width=width, label="per_sample", color="#16a085", alpha=0.85)
            bars2 = ax.bar([i + width / 2 for i in x], per_token, width=width, label="per_token", color="#f39c12", alpha=0.85)
            ax.set_xticks(list(x))
            ax.set_xticklabels(x_labels)
            ax.set_title(norm_titles[comp], fontsize=14, fontweight="bold", pad=10)
            ax.set_ylabel("Grad Norm", fontsize=11)
            ax.set_xlabel("Reward Variance Bucket", fontsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            ax.legend(frameon=False, fontsize=9)

            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (max(per_sample + per_token) * 0.01 if (per_sample + per_token) else 0.01),
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        fig4.suptitle(f"Normalized Grad Norms - Run: {run.name} (Step {step_key})", fontsize=18, y=1.03)
        plt.tight_layout()
        plt.savefig(output_file_normed, bbox_inches="tight", dpi=300)
        print(f"Success! Normalized grad visualization saved to: {os.path.abspath(output_file_normed)}")
        plt.close(fig4)

        # Summary 3-panel plot with available aggregates
        rv_means = [bucket_rv[b]["mean"] for b in buckets]
        rv_mins = [bucket_rv[b]["min"] for b in buckets]
        rv_maxs = [bucket_rv[b]["max"] for b in buckets]
        task_grads = [metric_source.get(f"grad_norm/{b}/task", 0) for b in buckets]
        kl_grads = [metric_source.get(f"grad_norm/{b}/kl", 0) for b in buckets]
        ent_grads = [metric_source.get(f"grad_norm/{b}/entropy", 0) for b in buckets]
        reg_grads = [k + e for k, e in zip(kl_grads, ent_grads)]

        fig5, axes5 = plt.subplots(1, 3, figsize=(20, 6))
        plt.subplots_adjust(wspace=0.35, top=0.80, bottom=0.15)

        # Left: RV mean with min/max error bars per bucket
        ax = axes5[0]
        use_boxplot = any(bucket_rv_values[b] for b in buckets)
        if use_boxplot:
            data = [bucket_rv_values[b] for b in buckets]
            data_mins = [min(v) if v else 0 for v in data]
            data_maxs = [max(v) if v else 0 for v in data]
            # sanity check: compare against logged min/max (per-sample)
            mismatch = any(
                abs(dm - rm) > 1e-3 or abs(dx - rx) > 1e-3
                for dm, rm, dx, rx in zip(data_mins, rv_mins, data_maxs, rv_maxs)
            )
            if mismatch:
                print(f"Warning: bucket RV table min/max mismatch at step {step_key}; falling back to error bars.")
                use_boxplot = False
        if use_boxplot:
            ax.boxplot(data, tick_labels=x_labels, showfliers=False)
            ax.set_title("RV by Bucket (Boxplot)", fontsize=13, fontweight="bold")
        else:
            # Guard against negative error bars from inconsistent min/max logging.
            yerr = [
                [max(0.0, m - lo) for m, lo in zip(rv_means, rv_mins)],
                [max(0.0, hi - m) for m, hi in zip(rv_means, rv_maxs)],
            ]
            ax.errorbar(x_labels, rv_means, yerr=yerr, fmt="o-", color="#6c5ce7", ecolor="#2d3436", capsize=4)
            ax.set_title("RV by Bucket (Mean ± Min/Max)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Bucket")
        ax.set_ylabel("Reward Variance (Std)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Middle: task grad norm vs mean RV
        ax = axes5[1]
        ax.plot(rv_means, task_grads, "o-", color="#e67e22")
        for i, (xv, yv) in enumerate(zip(rv_means, task_grads), start=1):
            ax.text(xv, yv, f"Q{i}", fontsize=8, ha="left", va="bottom")
        ax.set_title("Task Grad Norm vs RV Mean", fontsize=13, fontweight="bold")
        ax.set_xlabel("RV Mean")
        ax.set_ylabel("Task Grad Norm")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        mid_ylim = ax.get_ylim()

        # Right: regularizer grad norm (KL+Entropy) vs mean RV
        ax = axes5[2]
        ax.plot(rv_means, reg_grads, "o-", color="#16a085")
        for i, (xv, yv) in enumerate(zip(rv_means, reg_grads), start=1):
            ax.text(xv, yv, f"Q{i}", fontsize=8, ha="left", va="bottom")
        ax.set_title("Reg Grad Norm vs RV Mean (KL+Ent)", fontsize=13, fontweight="bold")
        ax.set_xlabel("RV Mean")
        ax.set_ylabel("KL+Entropy Grad Norm")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig5.suptitle(f"Gradient Summary - Run: {run.name} (Step {step_key})", fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(output_file_summary, bbox_inches="tight", dpi=300)
        print(f"Success! Summary visualization saved to: {os.path.abspath(output_file_summary)}")
        plt.close(fig5)

    print(f"All outputs written to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
