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

    def _groups_to_mask(self, top_groups: torch.Tensor, group_size: int = None) -> torch.Tensor:
        device = top_groups.device
        if group_size is None:
            group_size = self.group_size
        mask = torch.zeros(self.num_groups, dtype=torch.bool, device=device)
        if top_groups.numel() > 0:
            mask[top_groups] = True
        mask = mask.unsqueeze(1).expand(-1, group_size).reshape(-1).cpu()
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


class RewardRolloutFilter(RolloutFilter):
    """Filters rollouts based on reward statistics within groups."""

    _METRIC_OPTIONS = {"reward", "reward_variance"}

    def __init__(self, config: RolloutFilterConfig) -> None:
        super().__init__(config)
        if config.metric not in self._METRIC_OPTIONS:
            raise ValueError(
                f"RewardRolloutFilter only supports metrics {self._METRIC_OPTIONS}, got {config.metric}"
            )

    def _selection_scores(
        self, in_group_std: torch.Tensor, in_group_mean: torch.Tensor
    ) -> torch.Tensor:
        if self.config.metric == "reward":
            return in_group_mean
        return in_group_std

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, torch.Tensor]]:
        rollout_filter_ratio = self.ratio
        num_groups = self.num_groups

        # Check if this is turn-level mode (single_turn/limited_multi_turn, indicated by episode_ids)
        has_episode_ids = (
            batch.non_tensor_batch is not None
            and "episode_ids" in batch.non_tensor_batch
        )

        if has_episode_ids:
            # Turn-level mode: aggregate by episode first
            episode_ids = batch.non_tensor_batch["episode_ids"]
            group_ids = batch.non_tensor_batch["group_ids"]
            all_scores = batch.batch["original_rm_scores"].sum(dim=-1)

            # Get unique episodes and their rewards
            unique_episodes = []
            episode_to_first_idx = {}
            for i, eid in enumerate(episode_ids):
                if eid not in episode_to_first_idx:
                    unique_episodes.append(eid)
                    episode_to_first_idx[eid] = i

            # Get episode-level rewards and group_ids
            num_episodes = len(unique_episodes)
            episode_rewards = torch.zeros(num_episodes, device=all_scores.device)
            episode_group_ids = []
            for i, eid in enumerate(unique_episodes):
                idx = episode_to_first_idx[eid]
                episode_rewards[i] = all_scores[idx]
                episode_group_ids.append(group_ids[idx])

            # Calculate group_size as episodes per group
            group_size = num_episodes // num_groups
            
            if num_episodes % num_groups != 0:
                raise ValueError(
                    f"Number of episodes ({num_episodes}) must be divisible by num_groups ({num_groups})"
                )
            
            # Reshape to (num_groups, group_size)
            rm_scores = episode_rewards.view(num_groups, group_size)
        else:
            # Original mode: each sample is an episode
            actual_batch_size = batch.batch["original_rm_scores"].shape[0]
            group_size = actual_batch_size // num_groups
            rm_scores = batch.batch["original_rm_scores"].sum(dim=-1).view(num_groups, group_size)

        in_group_std = rm_scores.std(dim=-1)
        in_group_max = rm_scores.max(dim=-1).values
        in_group_mean = rm_scores.mean(dim=-1)

        selection_scores = self._selection_scores(in_group_std, in_group_mean)
        top_groups = self._select_top_groups(selection_scores)

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

        if has_episode_ids:
            # Build mask for turn-level samples based on selected groups
            # First, find which episodes belong to selected groups
            selected_episodes = set()
            for gid in top_groups.cpu().tolist():
                start_ep = gid * group_size
                end_ep = start_ep + group_size
                for ep_idx in range(start_ep, end_ep):
                    selected_episodes.add(unique_episodes[ep_idx])

            # Build turn-level mask
            mask = torch.tensor(
                [episode_ids[i] in selected_episodes for i in range(len(episode_ids))],
                dtype=torch.bool
            )
        else:
            mask = self._groups_to_mask(top_groups, group_size)

        batch = self._apply_mask(batch, mask)

        return batch, metrics


class EntropyRolloutFilter(RolloutFilter):
    """Filters rollouts based on policy entropy statistics within groups."""

    _METRIC_OPTIONS = {"entropy", "entropy_variance"}

    def __init__(
        self,
        config: RolloutFilterConfig,
        compute_log_prob: Callable[[DataProto], DataProto],
    ) -> None:
        super().__init__(config)
        if config.metric not in self._METRIC_OPTIONS:
            raise ValueError(
                f"EntropyRolloutFilter only supports metrics {self._METRIC_OPTIONS}, got {config.metric}"
            )
        self._compute_log_prob = compute_log_prob

    def _selection_scores(
        self, in_group_std: torch.Tensor, in_group_mean: torch.Tensor
    ) -> torch.Tensor:
        if self.config.metric == "entropy":
            return in_group_mean
        return in_group_std

    def filter(self, batch: DataProto) -> Tuple[DataProto, Dict[str, torch.Tensor]]:
        rollout_filter_ratio = self.ratio
        num_groups = self.num_groups

        if "entropys" not in batch.batch:
            log_prob = self._compute_log_prob(batch)
            batch = batch.union(log_prob)

        entropys = batch.batch["entropys"]
        loss_mask = batch.batch.get("loss_mask")
        if loss_mask is None:
            loss_mask = batch.batch.get("response_mask")
        if loss_mask is None:
            raise ValueError("EntropyRolloutFilter requires loss_mask or response_mask in the batch")

        loss_mask = loss_mask.to(entropys.device)
        token_counts = loss_mask.sum(dim=-1).clamp(min=1)
        entropy_per_traj = (entropys * loss_mask).sum(dim=-1) / token_counts

        # Check if this is turn-level mode (single_turn/limited_multi_turn, indicated by episode_ids)
        has_episode_ids = (
            batch.non_tensor_batch is not None
            and "episode_ids" in batch.non_tensor_batch
        )

        if has_episode_ids:
            # Turn-level mode: aggregate by episode first
            episode_ids = batch.non_tensor_batch["episode_ids"]

            # Get unique episodes and their entropy (average across turns)
            unique_episodes = []
            episode_to_indices = {}
            for i, eid in enumerate(episode_ids):
                if eid not in episode_to_indices:
                    unique_episodes.append(eid)
                    episode_to_indices[eid] = []
                episode_to_indices[eid].append(i)

            # Get episode-level entropy (mean of all turns)
            num_episodes = len(unique_episodes)
            episode_entropy = torch.zeros(num_episodes, device=entropy_per_traj.device)
            for i, eid in enumerate(unique_episodes):
                indices = episode_to_indices[eid]
                episode_entropy[i] = entropy_per_traj[indices].mean()

            # Calculate group_size as episodes per group
            group_size = num_episodes // num_groups

            if num_episodes % num_groups != 0:
                raise ValueError(
                    f"Number of episodes ({num_episodes}) must be divisible by num_groups ({num_groups})"
                )

            # Reshape to (num_groups, group_size)
            entropy_per_group = episode_entropy.view(num_groups, group_size)
        else:
            # Original mode: each sample is an episode
            actual_batch_size = entropy_per_traj.shape[0]
            group_size = actual_batch_size // num_groups
            entropy_per_group = entropy_per_traj.view(num_groups, group_size)

        in_group_std = entropy_per_group.std(dim=-1)
        in_group_max = entropy_per_group.max(dim=-1).values
        in_group_mean = entropy_per_group.mean(dim=-1)

        selection_scores = self._selection_scores(in_group_std, in_group_mean)
        top_groups = self._select_top_groups(selection_scores)

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

        if has_episode_ids:
            # Build mask for turn-level samples based on selected groups
            selected_episodes = set()
            for gid in top_groups.cpu().tolist():
                start_ep = gid * group_size
                end_ep = start_ep + group_size
                for ep_idx in range(start_ep, end_ep):
                    selected_episodes.add(unique_episodes[ep_idx])

            # Build turn-level mask
            mask = torch.tensor(
                [episode_ids[i] in selected_episodes for i in range(len(episode_ids))],
                dtype=torch.bool
            )
        else:
            mask = self._groups_to_mask(top_groups, group_size)

        batch = self._apply_mask(batch, mask)

        return batch, metrics


# Backwards compatibility: preserve older class names.
RewardVarianceRolloutFilter = RewardRolloutFilter
EntropyVarianceRolloutFilter = EntropyRolloutFilter


def build_rollout_filter(
    ratio: float,
    filter_type: str,
    num_groups: int,
    group_size: int,
    metric: Optional[str],
    compute_log_prob: Optional[Callable[[DataProto], DataProto]] = None,
) -> RolloutFilter:
    metric = (metric or "reward_variance").lower()
    metric = {
        "reward_std": "reward_variance",
        "entropy_std": "entropy_variance",
    }.get(metric, metric)

    config = RolloutFilterConfig(
        ratio=ratio,
        filter_type=filter_type,
        num_groups=num_groups,
        group_size=group_size,
        metric=metric,
    )

    if metric in {"reward", "reward_variance"}:
        return RewardRolloutFilter(config)
    if metric in {"entropy", "entropy_variance"}:
        if compute_log_prob is None:
            raise ValueError("Entropy filtering requires a compute_log_prob callable")
        return EntropyRolloutFilter(config, compute_log_prob=compute_log_prob)

    raise ValueError(f"Unsupported rollout filter metric: {metric}")
