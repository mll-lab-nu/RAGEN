"""
Countdown numbers game environment for compositional arithmetic reasoning.

Inspired by the numbers game from the TV show "Countdown".
Implementation adapted from TinyZero and veRL codebases.

Reference: Katz et al. (2025). Countdown environment for mathematical reasoning.
TinyZero: https://github.com/tinyzero-ai/tinyzero
veRL: https://github.com/volcengine/verl

We plan to generalize this environment to support any sort of static problem sets.
"""

from .env import CountdownEnv
from .config import CountdownEnvConfig

__all__ = ["CountdownEnv", "CountdownEnvConfig"]
