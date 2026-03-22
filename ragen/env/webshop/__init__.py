"""
WebShop environment for interactive e-commerce task.

Original Source: WebShop (https://github.com/princeton-nlp/WebShop)
Citation: Yao et al. (2022). WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
Paper: https://arxiv.org/abs/2207.01206
License: MIT

This implementation uses a minimal version of the WebShop environment
adapted for the RAGEN framework.
"""
from .env import WebShopEnv
from .config import WebShopEnvConfig

__all__ = ["WebShopEnv", "WebShopEnvConfig"]
