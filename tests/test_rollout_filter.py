import numpy as np
import pytest
import torch
from tensordict import TensorDict

try:
    from verl import DataProto
except Exception:
    class DataProto:
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


from ragen.trainer.rollout_filter import (
    RolloutFilterConfig,
    RewardRolloutFilter,
    EntropyRolloutFilter,
    build_rollout_filter,
)


def _make_top_p_filter(
    num_groups: int,
    value: float,
    top_p_prob_mode: str,
    include_zero: bool,
    selection_eps: float = 0.01,
):
    return RewardRolloutFilter(
        RolloutFilterConfig(
            value=value,
            filter_type="largest",
            num_groups=num_groups,
            group_size=2,
            top_p_prob_mode=top_p_prob_mode,
            include_zero=include_zero,
            selection_eps=selection_eps,
        )
    )


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
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={})


def test_reward_variance_filter_reduces_batch_size():
    num_groups, group_size, traj_len = 4, 2, 3
    batch = _make_reward_batch(num_groups, group_size, traj_len)

    rollout_filter = RewardRolloutFilter(
        RolloutFilterConfig(
            value=0.5,
            filter_type="largest",
            num_groups=num_groups,
            group_size=group_size,
            strategy="top_k",
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
        return DataProto(batch=td, non_tensor_batch={}, meta_info={})

    rollout_filter = EntropyRolloutFilter(
        RolloutFilterConfig(
            value=0.5,
            filter_type="largest",
            num_groups=num_groups,
            group_size=group_size,
            metric="entropy",
            strategy="top_k",
        ),
        compute_log_prob=fake_compute_log_prob,
    )

    filtered_batch, metrics = rollout_filter.filter(batch)

    expected = group_size * max(int(0.5 * num_groups), 1)
    assert filtered_batch.batch["loss_mask"].shape[0] == expected
    assert "entropys" in filtered_batch.batch.keys()
    assert "rollout/in_group_entropy_std" in metrics


def test_reward_metric_selects_high_mean_group():
    num_groups, group_size, traj_len = 2, 2, 1
    batch = _make_reward_batch(num_groups, group_size, traj_len)

    # Overwrite scores: first group has higher mean, second has higher variance.
    batch.batch["original_rm_scores"] = torch.tensor(
        [
            [10.0],
            [11.0],
            [0.0],
            [5.0],
        ]
    )

    rollout_filter = RewardRolloutFilter(
        RolloutFilterConfig(
            value=0.5,
            filter_type="largest",
            num_groups=num_groups,
            group_size=group_size,
            metric="reward",
            strategy="top_k",
        )
    )

    filtered_batch, _ = rollout_filter.filter(batch)

    # Highest mean group is the first one, so we expect its entries to remain.
    retained = filtered_batch.batch["original_rm_scores"].squeeze(-1)
    assert torch.allclose(retained, torch.tensor([10.0, 11.0]))


def test_reward_sum_metric_keeps_top_half_groups():
    num_groups, group_size, traj_len = 4, 2, 2
    batch = _make_reward_batch(num_groups, group_size, traj_len)

    batch.batch["original_rm_scores"] = torch.tensor(
        [
            [3.0, 2.0],
            [1.0, 4.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [1.0, 2.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )

    rollout_filter = RewardRolloutFilter(
        RolloutFilterConfig(
            value=0.5,
            filter_type="largest",
            num_groups=num_groups,
            group_size=group_size,
            metric="reward_sum",
            strategy="top_k",
        )
    )

    filtered_batch, metrics = rollout_filter.filter(batch)

    retained = filtered_batch.batch["original_rm_scores"]
    expected = torch.tensor(
        [
            [3.0, 2.0],
            [1.0, 4.0],
            [2.0, 2.0],
            [1.0, 2.0],
        ]
    )
    assert torch.allclose(retained, expected)
    assert metrics["rollout/in_group_reward_sum"].item() == pytest.approx(6.0)
    assert metrics["rollout/chosen_in_group_reward_sum"].item() == pytest.approx(8.5)


def test_build_rollout_filter_supports_reward_sum():
    rollout_filter = build_rollout_filter(
        value=0.5,
        filter_type="largest",
        num_groups=4,
        group_size=2,
        metric="reward_sum",
        strategy="top_k",
    )

    assert isinstance(rollout_filter, RewardRolloutFilter)
    assert rollout_filter.config.metric == "reward_sum"


def test_top_p_softmax_mode_preserves_previous_behavior():
    rollout_filter = RewardRolloutFilter(
        RolloutFilterConfig(
            value=0.6,
            filter_type="largest",
            num_groups=3,
            group_size=2,
            top_p_prob_mode="softmax",
        )
    )

    selected = rollout_filter._select_top_groups(torch.tensor([1.0, 1.0, 1.0]))

    # softmax over tied logits is uniform, so top_p must still keep enough groups
    # to exceed the threshold instead of returning an empty selection.
    assert torch.equal(selected, torch.tensor([0, 1]))


def test_top_p_linear_mode_matches_score_sum_rule():
    rollout_filter = RewardRolloutFilter(
        RolloutFilterConfig(
            value=0.6,
            filter_type="largest",
            num_groups=4,
            group_size=2,
            top_p_prob_mode="linear",
        )
    )

    selected = rollout_filter._select_top_groups(torch.tensor([4.0, 3.0, 2.0, 1.0]))

    # threshold = 0.6 * 10 - 0.01 = 5.99, so we need 4 + 3 and keep indices [0, 1]
    assert torch.equal(selected, torch.tensor([0, 1]))


def test_top_p_linear_mode_returns_empty_when_mass_is_non_positive():
    rollout_filter = RewardRolloutFilter(
        RolloutFilterConfig(
            value=0.9,
            filter_type="largest",
            num_groups=3,
            group_size=2,
            top_p_prob_mode="linear",
        )
    )

    selected = rollout_filter._select_top_groups(torch.tensor([0.0, 0.0, 0.0]))

    assert selected.numel() == 0


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([4.0, 2.0, 0.0, 0.0], [0, 1, 2, 3]),
        ([0.0, 0.0, 0.0], [0, 1, 2]),
        ([1.0, 3.0, 2.0], [0, 1, 2]),
    ],
)
def test_top_p_setting_nofilter_keeps_all_groups_for_various_lists(scores, expected):
    rollout_filter = _make_top_p_filter(
        num_groups=len(scores),
        value=1.0,
        top_p_prob_mode="linear",
        include_zero=True,
    )

    selected = rollout_filter._select_top_groups(torch.tensor(scores))

    assert torch.equal(selected, torch.tensor(expected))


@pytest.mark.parametrize(
    ("scores", "top_p", "expected"),
    [
        ([10.0, 1.0, 0.0, 0.0], 0.9, [0]),
        ([4.0, 2.0, 0.0, 0.0], 0.9, [0, 1]),
        ([1.0, 1.0, 1.0, 0.0], 0.6, [0, 1]),
    ],
)
def test_top_p_setting_softmax_excludes_zero_for_various_lists(scores, top_p, expected):
    rollout_filter = _make_top_p_filter(
        num_groups=len(scores),
        value=top_p,
        top_p_prob_mode="softmax",
        include_zero=False,
    )

    selected = rollout_filter._select_top_groups(torch.tensor(scores))

    assert torch.equal(selected, torch.tensor(expected))


@pytest.mark.parametrize(
    ("scores", "top_p", "expected"),
    [
        ([10.0, 1.0, 0.0, 0.0], 0.9, [0]),
        ([4.0, 2.0, 0.0, 0.0], 0.9, [0, 1]),
        ([3.0, 1.0, 1.0, 0.0], 0.9, [0, 1, 2]),
    ],
)
def test_top_p_setting_linear_excludes_zero_for_various_lists(scores, top_p, expected):
    rollout_filter = _make_top_p_filter(
        num_groups=len(scores),
        value=top_p,
        top_p_prob_mode="linear",
        include_zero=False,
        selection_eps=0.01,
    )

    selected = rollout_filter._select_top_groups(torch.tensor(scores))

    assert torch.equal(selected, torch.tensor(expected))
