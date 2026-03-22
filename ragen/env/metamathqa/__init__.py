"""
MetaMathQA environment for mathematical reasoning.

Dataset: MetaMathQA (https://huggingface.co/datasets/meta-math/MetaMathQA)
Citation: Yu et al. (2023). MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models
Paper: https://arxiv.org/abs/2309.12284
License: MIT
"""
from .env import MetaMathQAEnv
from .config import MetaMathQAEnvConfig

__all__ = ["MetaMathQAEnv", "MetaMathQAEnvConfig"]
