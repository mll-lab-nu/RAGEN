from typing import Optional
from .base import BaseMemory
from .simple_memory import SimpleMemory
from .alfworld_memory import AlfWorldMemory

# Environment type to Memory class mapping
MEMORY_REGISTRY = {
    "alfworld": AlfWorldMemory,
    # Add more environment-specific memory classes here
    # "webshop": WebShopMemory,
}


def get_memory_class(env_type: str) -> type:
    """Get memory class for environment type."""
    return MEMORY_REGISTRY.get(env_type.lower(), SimpleMemory)


def create_memory(env_type: str) -> BaseMemory:
    """Create memory instance for environment type."""
    memory_class = get_memory_class(env_type)
    return memory_class()
