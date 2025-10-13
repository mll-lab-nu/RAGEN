import sys
import types

import numpy as np
import torch
from tensordict import TensorDict


if "verl" not in sys.modules:
    stub = types.ModuleType("verl")

    class DummyDataProto:
        def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
            self.batch = batch
            self.non_tensor_batch = non_tensor_batch or {}
            self.meta_info = meta_info or {}

        def union(self, other):
            if other.batch is not None:
                for key, value in other.batch.items():
                    self.batch[key] = value
            if other.non_tensor_batch:
                self.non_tensor_batch.update(other.non_tensor_batch)
            if other.meta_info:
                self.meta_info.update(other.meta_info)
            return self

    stub.DataProto = DummyDataProto
    sys.modules["verl"] = stub


from ragen.trainer.rollout_filter import RolloutFilterConfig, RewardVarianceRolloutFilter, EntropyVarianceRolloutFilter


def _make_reward_batch(num_groups: int, group_size: int, traj_len: int):
    total = num_groups * group_size
    rm_scores = torch.arange(total * traj_len, dtype=torch.float32).reshape(total, traj_len)
    loss_mask = torch.ones(total, traj_len)
    batch = TensorDict(
        {
            "original_rm_scores": rm_scores,
            "loss_mask": loss_mask,
        },
        batch_size=[total],
    )
    non_tensor_batch = {"uids": np.arange(total)}
    return sys.modules["verl"].DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={})


def test_reward_variance_filter_reduces_batch_size():
    num_groups, group_size, traj_len = 4, 2, 3
    batch = _make_reward_batch(num_groups, group_size, traj_len)

    rollout_filter = RewardVarianceRolloutFilter(
        RolloutFilterConfig(
            ratio=0.5,
            filter_type="std",
            num_groups=num_groups,
            group_size=group_size,
        )
    )

    filtered_batch, metrics = rollout_filter.filter(batch)

    assert filtered_batch.batch["original_rm_scores"].shape[0] == group_size * max(int(0.5 * num_groups), 1)
    assert "rollout/in_group_std" in metrics


def test_entropy_variance_filter_uses_compute_log_prob():
    num_groups, group_size, traj_len = 2, 3, 4
    batch = _make_reward_batch(num_groups, group_size, traj_len)

    entropies = torch.linspace(0.1, 1.0, steps=num_groups * group_size * traj_len).reshape(num_groups * group_size, traj_len)
    old_log_probs = -entropies

    def fake_compute_log_prob(data_proto):
        td = TensorDict(
            {
                "old_log_probs": old_log_probs,
                "entropys": entropies,
            },
            batch_size=[num_groups * group_size],
        )
        return sys.modules["verl"].DataProto(batch=td, non_tensor_batch={}, meta_info={})

    rollout_filter = EntropyVarianceRolloutFilter(
        RolloutFilterConfig(
            ratio=0.5,
            filter_type="std",
            num_groups=num_groups,
            group_size=group_size,
            metric="entropy",
        ),
        compute_log_prob=fake_compute_log_prob,
    )

    filtered_batch, metrics = rollout_filter.filter(batch)

    expected = group_size * max(int(0.5 * num_groups), 1)
    assert filtered_batch.batch["loss_mask"].shape[0] == expected
    assert "old_log_probs" in filtered_batch.batch.keys()
    assert "rollout/in_group_entropy_std" in metrics
