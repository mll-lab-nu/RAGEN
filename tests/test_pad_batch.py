import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto

from ragen.trainer.pad_batch import build_padded_training_batch, get_pad_seq_mean_scale


def _make_training_batch(batch_size: int = 34, seq_len: int = 5, response_len: int = 3) -> DataProto:
    batch = TensorDict(
        {
            "input_ids": torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len),
            "attention_mask": torch.ones(batch_size, seq_len),
            "position_ids": torch.arange(seq_len, dtype=torch.long).repeat(batch_size, 1),
            "responses": torch.arange(batch_size * response_len, dtype=torch.long).reshape(batch_size, response_len),
            "response_mask": torch.ones(batch_size, response_len),
            "loss_mask": torch.ones(batch_size, response_len),
            "advantages": torch.full((batch_size, response_len), 3.0),
            "returns": torch.full((batch_size, response_len), 4.0),
            "token_level_scores": torch.full((batch_size, response_len), 5.0),
            "token_level_rewards": torch.full((batch_size, response_len), 6.0),
            "old_log_probs": torch.full((batch_size, response_len), 7.0),
            "ref_log_prob": torch.full((batch_size, response_len), 8.0),
            "values": torch.full((batch_size, response_len), 9.0),
        },
        batch_size=[batch_size],
    )
    non_tensor_batch = {
        "uid": np.arange(batch_size),
        "group_ids": np.arange(batch_size),
    }
    return DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info={
            "metrics": {"foo": 1.0},
            "global_token_num": torch.sum(batch["attention_mask"], dim=-1).tolist(),
        },
    )


def test_build_padded_training_batch_zeroes_only_pad_rows():
    batch = _make_training_batch()

    padded, pad_count = build_padded_training_batch(batch, mini_batch_size=32)

    assert pad_count == 30
    assert len(padded) == 64
    assert padded.meta_info["metrics"] == {"foo": 1.0}
    assert len(padded.meta_info["global_token_num"]) == 64
    assert padded.meta_info["real_batch_size"] == 34
    assert padded.meta_info["pad_count"] == 30

    assert torch.equal(padded.batch["input_ids"][:34], batch.batch["input_ids"])
    assert torch.equal(padded.batch["input_ids"][34:], batch.batch["input_ids"][:30])
    assert not padded.batch["pad_mask"][:34].any().item()
    assert padded.batch["pad_mask"][34:].all().item()

    zero_keys = [
        "response_mask",
        "loss_mask",
        "advantages",
        "returns",
        "token_level_scores",
        "token_level_rewards",
        "old_log_probs",
        "ref_log_prob",
        "values",
    ]
    for key in zero_keys:
        assert torch.count_nonzero(padded.batch[key][34:]).item() == 0


def test_build_padded_training_batch_is_noop_when_already_divisible():
    batch = _make_training_batch(batch_size=64)

    padded, pad_count = build_padded_training_batch(batch, mini_batch_size=32)

    assert pad_count == 0
    assert padded is batch
    assert "pad_mask" not in padded.batch.keys()


def test_pad_seq_mean_scale_only_applies_to_seq_mean_modes():
    pad_mask = torch.tensor([False, False, True, True])

    assert get_pad_seq_mean_scale(pad_mask, "seq-mean-token-mean", 4) == pytest.approx(2 / 4)
    assert get_pad_seq_mean_scale(pad_mask, "seq-mean-token-sum", 4) == pytest.approx(2 / 4)
    assert get_pad_seq_mean_scale(pad_mask, "seq-mean-token-sum-norm", 4) == 1.0
    assert get_pad_seq_mean_scale(pad_mask, "token-mean", 4) == 1.0
    assert get_pad_seq_mean_scale(torch.zeros(4, dtype=torch.bool), "seq-mean-token-mean", 4) == 1.0


def test_pad_seq_mean_scale_preserves_original_mini_batch_weighting():
    micro_batch_size = 4
    mini_batch_size = 32
    gradient_accumulation = mini_batch_size // micro_batch_size
    micro_batch_real_counts = [4, 4, 4, 4, 2, 0, 0, 0]

    total_weight = 0.0
    for real_count in micro_batch_real_counts:
        pad_mask = torch.tensor([False] * real_count + [True] * (micro_batch_size - real_count), dtype=torch.bool)
        scale = get_pad_seq_mean_scale(pad_mask, "seq-mean-token-mean", micro_batch_size)
        total_weight += scale / gradient_accumulation

    assert total_weight == pytest.approx(18 / 32)
