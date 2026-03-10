import contextlib
from typing import Dict

from verl.utils.metric import reduce_metrics


def _set_meta_flag(meta: Dict, key: str, value: bool):
    existing = meta.get(key, None)
    meta[key] = value

    @contextlib.contextmanager
    def _restore():
        try:
            yield
        finally:
            if existing is None:
                meta.pop(key, None)
            else:
                meta[key] = existing

    return _restore()


def run_gradient_analysis(trainer, batch, metrics):
    print(f"[Gradient Analysis] Step {trainer.global_steps}: Running analysis on buckets...")

    try:
        buckets = trainer.rollout_filter.split_into_buckets(batch)
    except AttributeError:
        print("[Gradient Analysis] Rollout filter does not support 'split_into_buckets'. Using default batch.")
        buckets = {"all": batch}

    for bucket_name, sub_batch in buckets.items():
        count = sub_batch.batch.batch_size[0]
        if count == 0:
            print(f"[Gradient Analysis] Bucket '{bucket_name}' is empty. Skipping.")
            continue
        dp_size = int(getattr(trainer.config.trainer, "n_gpus_per_node", 1))
        if dp_size > 1 and count % dp_size != 0:
            new_count = count - (count % dp_size)
            if new_count == 0:
                print(
                    f"[Gradient Analysis] Bucket '{bucket_name}' size {count} not divisible by dp_size={dp_size}. Skipping."
                )
                continue
            print(
                f"[Gradient Analysis] Bucket '{bucket_name}' size {count} not divisible by dp_size={dp_size}. Dropping to {new_count}."
            )
            sub_batch = sub_batch.slice(0, new_count)
            count = new_count

        print(f"[Gradient Analysis] Processing bucket '{bucket_name}' with {count} samples.")

        total_samples = batch.batch.batch_size[0]
        metrics[f"grad_norm/{bucket_name}/sample_count"] = count
        if total_samples > 0:
            metrics[f"grad_norm/{bucket_name}/sample_pct"] = count / total_samples

        with _set_meta_flag(sub_batch.meta_info, "skip_optimizer_step", True), _set_meta_flag(
            sub_batch.meta_info, "grad_component_analysis", True
        ):
            actor_output = trainer.actor_rollout_wg.update_actor(sub_batch)

        bucket_metrics = reduce_metrics(actor_output.meta_info["metrics"])

        reward_std_mean = sub_batch.meta_info.get("bucket_reward_std_mean", None)
        bucket_reward_std_values = sub_batch.meta_info.get("bucket_reward_std_values", None)
        bucket_group_ids = sub_batch.meta_info.get("bucket_group_ids", None)
        if reward_std_mean is None and "reward_std" in sub_batch.batch:
            reward_std_mean = sub_batch.batch["reward_std"].mean().item()
        reward_std_min = None
        reward_std_max = None
        if "reward_std" in sub_batch.batch:
            reward_std_min = sub_batch.batch["reward_std"].min().item()
            reward_std_max = sub_batch.batch["reward_std"].max().item()
        if reward_std_mean is not None:
            metrics[f"grad_norm/{bucket_name}/reward_std_mean"] = reward_std_mean
        if reward_std_min is not None:
            metrics[f"grad_norm/{bucket_name}/reward_std_min"] = reward_std_min
        if reward_std_max is not None:
            metrics[f"grad_norm/{bucket_name}/reward_std_max"] = reward_std_max
        if bucket_reward_std_values is not None and bucket_group_ids is not None:
            metrics[f"grad_norm/{bucket_name}/group_rv_count"] = len(bucket_reward_std_values)
            try:
                import wandb  # local import to avoid hard dependency if wandb disabled

                table = wandb.Table(columns=["bucket", "group_id", "reward_std"])
                for gid, rv in zip(bucket_group_ids, bucket_reward_std_values):
                    table.add_data(bucket_name, int(gid), float(rv))
                metrics[f"grad_norm/{bucket_name}/group_rv_table"] = table
            except Exception:
                pass

        num_tokens = None
        if "response_mask" in sub_batch.batch:
            num_tokens = sub_batch.batch["response_mask"].sum().item()
        for k, v in bucket_metrics.items():
            if k.startswith("actor/grad_norm/"):
                component = k.split("/", 2)[2]
                metrics[f"grad_norm/{bucket_name}/{component}"] = v
                if count > 0:
                    metrics[f"grad_norm/{bucket_name}/per_sample/{component}"] = v / count
                if num_tokens and num_tokens > 0:
                    metrics[f"grad_norm/{bucket_name}/per_token/{component}"] = v / num_tokens
            elif k.startswith("actor/loss/"):
                component = k.split("/", 2)[2]
                metrics[f"grad_norm/{bucket_name}/loss/{component}"] = v
            else:
                metrics[f"grad_norm/{bucket_name}/{k}"] = v
            if bucket_name == "all":
                metrics[k] = v

        kl = bucket_metrics.get("actor/loss/kl", 0)
        entropy = bucket_metrics.get("actor/loss/entropy", 0)
        policy = bucket_metrics.get("actor/loss/policy", 0)
        total = bucket_metrics.get("actor/loss/total", 0)
        grad_task = bucket_metrics.get("actor/grad_norm/task", 0)
        grad_entropy = bucket_metrics.get("actor/grad_norm/entropy", 0)
        grad_kl = bucket_metrics.get("actor/grad_norm/kl", 0)

        print(f"[Gradient Analysis] Bucket '{bucket_name}' Metrics:")
        print(f"    - KL Loss:      {kl:>10.6f}")
        print(f"    - Entropy Loss: {entropy:>10.6f}")
        print(f"    - Policy Loss:  {policy:>10.6f} (Task)")
        print(f"    - Total Loss:   {total:>10.6f}")
        print(
            f"    - Grad Norms:   task={grad_task:>10.6f} | entropy={grad_entropy:>10.6f} | kl={grad_kl:>10.6f}"
        )
        if reward_std_mean is not None:
            if reward_std_min is not None and reward_std_max is not None:
                print(
                    f"    - Reward Std:   mean={reward_std_mean:>10.6f} | min={reward_std_min:>10.6f} | max={reward_std_max:>10.6f}"
                )
            else:
                print(f"    - Reward Std:   mean={reward_std_mean:>10.6f}")
        print("")
