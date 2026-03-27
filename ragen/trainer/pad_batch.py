from __future__ import annotations

from typing import Iterable

import torch

from verl import DataProto


PAD_ZERO_BATCH_KEYS: tuple[str, ...] = (
    "response_mask",
    "loss_mask",
    "advantages",
    "returns",
    "token_level_scores",
    "token_level_rewards",
    "old_log_probs",
    "ref_log_prob",
    "values",
)

SEQ_MEAN_LOSS_AGG_MODES = {
    "seq-mean-token-sum",
    "seq-mean-token-mean",
    "seq-mean-token-sum-norm",
}


def build_padded_training_batch(
    batch: DataProto,
    mini_batch_size: int,
    zero_batch_keys: Iterable[str] = PAD_ZERO_BATCH_KEYS,
) -> tuple[DataProto, int]:
    """Pad a training batch to the next mini-batch multiple using zero-loss pad rows."""
    if mini_batch_size <= 0:
        raise ValueError(f"mini_batch_size must be positive, got {mini_batch_size}")

    if batch.batch is None:
        raise ValueError("pad batch padding requires tensor data")

    real_batch_size = len(batch)
    remainder = real_batch_size % mini_batch_size
    if remainder == 0:
        return batch, 0

    pad_count = mini_batch_size - remainder
    pad_parts = []
    remaining = pad_count
    while remaining > 0:
        take_size = min(remaining, real_batch_size)
        pad_parts.append(batch[:take_size])
        remaining -= take_size

    padded_batch = DataProto.concat([batch] + pad_parts)
    padded_batch.meta_info = dict(batch.meta_info or {})

    first_tensor_key = next(iter(padded_batch.batch.keys()))
    device = padded_batch.batch[first_tensor_key].device
    pad_mask = torch.zeros(len(padded_batch), dtype=torch.bool, device=device)
    pad_mask[real_batch_size:] = True
    padded_batch.batch["pad_mask"] = pad_mask

    for key in zero_batch_keys:
        if key in padded_batch.batch.keys():
            padded_batch.batch[key][real_batch_size:] = 0

    if "attention_mask" in padded_batch.batch.keys():
        padded_batch.meta_info["global_token_num"] = torch.sum(padded_batch.batch["attention_mask"], dim=-1).tolist()

    padded_batch.meta_info["real_batch_size"] = real_batch_size
    padded_batch.meta_info["pad_count"] = pad_count
    return padded_batch, pad_count


def get_pad_seq_mean_scale(
    pad_mask: torch.Tensor | None,
    loss_agg_mode: str,
    configured_mini_batch_size: int,
) -> float:
    """Scale seq-mean losses so pad rows still count toward the configured mini-batch denominator."""
    if pad_mask is None or loss_agg_mode not in SEQ_MEAN_LOSS_AGG_MODES:
        return 1.0

    if configured_mini_batch_size <= 0:
        raise ValueError(f"configured_mini_batch_size must be positive, got {configured_mini_batch_size}")

    pad_mask = pad_mask.to(dtype=torch.bool).reshape(-1)
    if not pad_mask.any().item():
        return 1.0

    real_batch_size = (~pad_mask).sum().item()
    return float(real_batch_size) / float(configured_mini_batch_size)
