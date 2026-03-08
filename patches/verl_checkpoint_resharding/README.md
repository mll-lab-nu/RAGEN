# FSDP Checkpoint Resharding Patch

## Problem

verl's `FSDPCheckpointManager` saves checkpoints as `model_world_size_{N}_rank_{R}.pt`. When resuming with a **different number of GPUs** than the original training run, the checkpoint files don't match and loading fails:

```
FileNotFoundError: model_world_size_4_rank_1.pt
# (checkpoint was saved with 1 GPU as model_world_size_1_rank_0.pt)
```

## Fix

This patched `fsdp_checkpoint_manager.py` adds automatic resharding: when the expected checkpoint file doesn't exist but a `model_world_size_1_rank_0.pt` is found, it loads the full state dict and lets FSDP reshard across the current world size.

### What changes:

1. **Model loading**: Detects world_size mismatch, switches from `SHARDED_STATE_DICT` to `FULL_STATE_DICT` loading mode so FSDP can reshard automatically
2. **Optimizer**: Skipped during resharding (optimizer state is world_size-specific and must reinitialize)
3. **Extra state** (lr_scheduler, rng): Falls back to the 1-GPU file if available, or skips gracefully

## How to Apply

Copy the patched file over the verl submodule file:

```bash
cp patches/verl_checkpoint_resharding/fsdp_checkpoint_manager.py \
   verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py
```

## Limitations

- Only supports resharding **from 1 GPU** to N GPUs (not arbitrary M→N)
- Optimizer state is not transferred (will reinitialize from scratch)
- LR scheduler state may not transfer if extra state was not saved
