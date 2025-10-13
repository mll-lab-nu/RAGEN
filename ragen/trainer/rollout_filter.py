"""Utilities for filtering rollout trajectories before PPO updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from verl import DataProto


@dataclass
class RolloutFilterConfig:
    """Configuration container for rollout filtering."""

    ratio: float
    filter_type: str
    group_size: int
    num_groups: int
    metric: str = "reward_variance"


class RolloutFilter:
    """Base class for rollout filters."""

    def __init__(self, config: RolloutFilterConfig):
        self.config = config

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    @property
    def ratio(self) -> float:
        return self.config.ratio

    @property
    def filter_type(self) -> str:
        return self.config.filter_type

    @property
    def group_size(self) -> int:
        return self.config.group_size

    @property
    def num_groups(self) -> int:
        return self.config.num_groups

    def _select_top_groups(self, scores: torch.Tensor) -> torch.Tensor:
        rollout_filter_ratio = self.ratio
        if rollout_filter_ratio >= 1:
            return torch.arange(self.num_groups, device=scores.device)

        k = max(int(rollout_filter_ratio * self.num_groups), 1)

        if self.filter_type == "smallest":
            top_groups = (-scores).topk(k).indices
        elif self.filter_type == "largest":
            top_groups = scores.topk(k).indices
        else:
            raise ValueError(f"Invalid rollout filter type: {self.filter_type}")

        return top_groups

    def _groups_to_mask(self, top_groups: torch.Tensor) -> torch.Tensor:
        device = top_groups.device
        mask = torch.zeros(self.num_groups, dtype=torch.bool, device=device)
        if top_groups.numel() > 0:
            mask[top_groups] = True
        mask = mask.unsqueeze(1).expand(-1, self.group_size).reshape(-1).cpu()
        return mask

    def _apply_mask(self, batch: DataProto, mask: torch.Tensor) -> DataProto:
        batch.batch = batch.batch[mask]

        if batch.non_tensor_batch is not None:
            np_mask = mask.cpu().numpy()
            for key, value in batch.non_tensor_batch.items():
                if isinstance(value, np.ndarray):
                    batch.non_tensor_batch[key] = value[np_mask]
                else:
                    batch.non_tensor_batch[key] = [v for v, m in zip(value, np_mask) if m]

        return batch

    def _build_base_metrics(
        self,
        in_group_std: torch.Tensor,
        in_group_max: torch.Tensor,
        in_group_mean: torch.Tensor,
        top_groups: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        metrics = {
            "rollout/in_group_std": in_group_std.mean(),
            "rollout/in_group_max": in_group_max.mean(),
            "rollout/in_group_mean": in_group_mean.mean(),
        }

        chosen = top_groups
        metrics.update(
            {
                "rollout/chosen_in_group_std": in_group_std[chosen].mean(),
                "rollout/chosen_in_group_max": in_group_max[chosen].mean(),
                "rollout/chosen_in_group_mean": in_group_mean[chosen].mean(),
            }
        )
        return metrics


class RewardVarianceRolloutFilter(RolloutFilter):
    """Filters rollouts based on reward variance within groups."""

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, torch.Tensor]]:
        rollout_filter_ratio = self.ratio
        num_groups, group_size = self.num_groups, self.group_size

        rm_scores = batch.batch["original_rm_scores"].sum(dim=-1).view(num_groups, group_size)
        in_group_std = rm_scores.std(dim=-1)
        in_group_max = rm_scores.max(dim=-1).values
        in_group_mean = rm_scores.mean(dim=-1)

        top_groups = self._select_top_groups(in_group_std)

        metrics = self._build_base_metrics(in_group_std, in_group_max, in_group_mean, top_groups)
        metrics.update(
            {
                "rollout/in_group_reward_std": in_group_std.mean(),
                "rollout/in_group_reward_max": in_group_max.mean(),
                "rollout/in_group_reward_mean": in_group_mean.mean(),
                "rollout/chosen_in_group_reward_std": in_group_std[top_groups].mean(),
                "rollout/chosen_in_group_reward_max": in_group_max[top_groups].mean(),
                "rollout/chosen_in_group_reward_mean": in_group_mean[top_groups].mean(),
            }
        )

        if rollout_filter_ratio >= 1:
            return batch, metrics

        mask = self._groups_to_mask(top_groups)
        batch = self._apply_mask(batch, mask)

        return batch, metrics


class EntropyRolloutFilter(RolloutFilter):
    """Filters rollouts based on policy entropy amount within groups."""

    def __init__(
        self,
        config: RolloutFilterConfig,
        compute_log_prob: Callable[[DataProto], DataProto],
    ) -> None:
        super().__init__(config)
        self._compute_log_prob = compute_log_prob

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, torch.Tensor]]:
        rollout_filter_ratio = self.ratio

        if "entropys" not in batch.batch:
            log_prob = self._compute_log_prob(batch)
            batch = batch.union(log_prob)

        entropys = batch.batch["entropys"]
        loss_mask = batch.batch.get("loss_mask")
        if loss_mask is None:
            loss_mask = batch.batch.get("response_mask")
        if loss_mask is None:
            raise ValueError("EntropyVarianceRolloutFilter requires loss_mask or response_mask in the batch")

        loss_mask = loss_mask.to(entropys.device)
        token_counts = loss_mask.sum(dim=-1).clamp(min=1)
        entropy_per_traj = (entropys * loss_mask).sum(dim=-1) / token_counts

        num_groups, group_size = self.num_groups, self.group_size
        entropy_per_group = entropy_per_traj.view(num_groups, group_size)
        in_group_std = entropy_per_group.std(dim=-1)
        in_group_max = entropy_per_group.max(dim=-1).values
        in_group_mean = entropy_per_group.mean(dim=-1)

        top_groups = self._select_top_groups(in_group_mean)

        metrics = self._build_base_metrics(in_group_std, in_group_max, in_group_mean, top_groups)
        metrics.update(
            {
                "rollout/in_group_entropy_std": in_group_std.mean(),
                "rollout/in_group_entropy_max": in_group_max.mean(),
                "rollout/in_group_entropy_mean": in_group_mean.mean(),
                "rollout/chosen_in_group_entropy_std": in_group_std[top_groups].mean(),
                "rollout/chosen_in_group_entropy_max": in_group_max[top_groups].mean(),
                "rollout/chosen_in_group_entropy_mean": in_group_mean[top_groups].mean(),
            }
        )

        if rollout_filter_ratio >= 1:
            return batch, metrics

        mask = self._groups_to_mask(top_groups)
        batch = self._apply_mask(batch, mask)

        return batch, metrics


def build_rollout_filter(
    ratio: float,
    filter_type: str,
    num_groups: int,
    group_size: int,
    metric: Optional[str],
    compute_log_prob: Optional[Callable[[DataProto], DataProto]] = None,
) -> RolloutFilter:
    metric = (metric or "reward_variance").lower()

    config = RolloutFilterConfig(
        ratio=ratio,
        filter_type=filter_type,
        num_groups=num_groups,
        group_size=group_size,
        metric=metric,
    )

    if metric in {"reward", "reward_variance", "reward_std"}:
        return RewardVarianceRolloutFilter(config)
    if metric in {"entropy", "entropy_variance", "entropy_std"}:
        if compute_log_prob is None:
            raise ValueError("Entropy variance filtering requires a compute_log_prob callable")
        return EntropyVarianceRolloutFilter(config, compute_log_prob=compute_log_prob)

    raise ValueError(f"Unsupported rollout filter metric: {metric}")
