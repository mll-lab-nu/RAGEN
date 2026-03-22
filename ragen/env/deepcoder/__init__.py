"""
DeepCoder coding environment for competitive programming problems.

This environment uses datasets from multiple sources:
- PrimeIntellect synthetic data (https://github.com/PrimeIntellect-ai/synthetic1M)
- TACO: https://arxiv.org/abs/2310.20466
- LiveCodeBench v5: https://arxiv.org/abs/2403.07974

Reference: https://www.together.ai/blog/deepcoder
Used to train DeepSeek-R1-Distill-Qwen-14B with reinforcement learning.
"""
from .env import DeepCoderEnv
from .config import DeepCoderEnvConfig

__all__ = ["DeepCoderEnv", "DeepCoderEnvConfig"]
