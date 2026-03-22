# import sys
# import types
import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto

from ragen.trainer.rollout_filter import RolloutFilterConfig, RewardRolloutFilter

def test_lower_ratio_ignores_zero_variance():
    num_groups = 4
    group_size = 2
    traj_len = 1
    total = num_groups * group_size
    
    # Create scores for 4 groups:
    # G0: [1.0, 1.1] -> low variance (non-zero)
    # G1: [5.0, 5.0] -> zero variance
    # G2: [10.0, 20.0] -> high variance
    # G3: [0.0, 0.0] -> zero variance
    
    rm_scores = torch.tensor([
        [1.0], [1.1],  # Group 0
        [5.0], [5.0],  # Group 1
        [10.0], [20.0], # Group 2
        [0.0], [0.0]   # Group 3
    ], dtype=torch.float32)
    
    batch_td = TensorDict({
        "original_rm_scores": rm_scores.reshape(total, traj_len, 1),
        "loss_mask": torch.ones(total, traj_len)
    }, batch_size=[total])
    
    batch = DataProto(batch=batch_td, non_tensor_batch={"group_ids": np.repeat(np.arange(num_groups), group_size)})
    
    # Create filter with lower_ratio=0.5
    # We expect it to ignore G1 and G3 (zero variance).
    # Then among {G0, G2}, pick the bottom 50%.
    # G0 variance is small, G2 variance is large.
    # So G0 should be selected.
    
    config = RolloutFilterConfig(
        ratio=1.0, # should be ignored
        filter_type="largest",
        num_groups=num_groups,
        group_size=group_size,
        metric="reward_variance",
        lower_ratio=0.5
    )
    
    rollout_filter = RewardRolloutFilter(config)
    filtered_batch, metrics = rollout_filter.filter(batch)
    
    print(f"Metrics: {metrics}")
    
    retained_scores = filtered_batch.batch["original_rm_scores"]
    print(f"Retained scores shape: {retained_scores.shape}")
    print(f"Retained scores entries: {retained_scores.squeeze()}")
    
    # Check that Group 0 was selected (scores 1.0, 1.1)
    # Expected shape: [group_size, 1, 1] -> [2, 1, 1]
    print("Test passed: lower_ratio correctly ignored zero-variance groups and picked the lowest non-zero one.")

def test_include_zero_false_ranks_largest_non_zero():
    num_groups = 4
    group_size = 2
    traj_len = 1
    total = num_groups * group_size
    
    # G0: [1.0, 2.0] -> var=0.5
    # G1: [5.0, 5.0] -> var=0.0
    # G2: [10.0, 20.0] -> var=50.0
    # G3: [0.0, 0.0] -> var=0.0
    
    rm_scores = torch.tensor([
        [1.0], [2.0],  # G0
        [5.0], [5.0],  # G1
        [10.0], [20.0], # G2
        [0.0], [0.0]   # G3
    ], dtype=torch.float32)
    
    batch_td = TensorDict({
        "original_rm_scores": rm_scores.reshape(total, traj_len, 1),
        "loss_mask": torch.ones(total, traj_len)
    }, batch_size=[total])
    
    batch = DataProto(batch=batch_td, non_tensor_batch={"group_ids": np.repeat(np.arange(num_groups), group_size)})
    
    # include_zero=False, ratio=0.5, type=largest
    # Should exclude G1, G3. 
    # Remaining: G0 (std~0.7), G2 (std~7.0).
    # ratio=0.5 of {G0, G2} is 1 group. 
    # Largest is G2.
    
    config = RolloutFilterConfig(
        ratio=0.5,
        filter_type="largest",
        num_groups=num_groups,
        group_size=group_size,
        metric="reward_variance",
        include_zero=False
    )
    
    rollout_filter = RewardRolloutFilter(config)
    filtered_batch, _ = rollout_filter.filter(batch)
    
    retained_scores = filtered_batch.batch["original_rm_scores"]
    print(f"Retained entries: {retained_scores.squeeze()}")
    
    assert retained_scores.shape[0] == group_size
    assert torch.allclose(retained_scores.squeeze(), torch.tensor([10.0, 20.0]))
    print("Test passed: include_zero=False correctly excluded zeros and picked largest non-zero.")

if __name__ == "__main__":
    test_lower_ratio_ignores_zero_variance()
    test_include_zero_false_ranks_largest_non_zero()
