from typing import List, Dict, Any
from .base import BaseMemory


class SimpleMemory(BaseMemory):
    """
    Default memory implementation using direct concatenation.
    Used for most environments (Sokoban, FrozenLake, etc.)

    Format:
    Summary: N step(s) completed so far. Showing the last K turn(s)...
    ---
    Turn 1 State: ...
    Turn 1 Action: ...
    Turn 1 Reward: ...
    ---
    Current Turn (Turn N):
    State: ...
    """

    def __init__(self):
        self._data = None
        self.batch_size = 0

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self.batch_size = batch_size

    def store(self, record: Dict[str, List[Any]]):
        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in record.keys()})

    def build_user_content(
        self,
        env_output: Dict,
        history: List[Dict],
        turn_idx: int,
        history_start: int,
        turn_offset: int,
        include_warning: bool,
        format_prompt: str,
        length_prompt: str,
    ) -> str:
        """Build user content using direct concatenation (original method)."""
        content = ""

        # Build history section
        if history_start < turn_idx:
            completed_steps = turn_idx + turn_offset
            history_length = turn_idx - history_start
            content += (
                f"Summary: {completed_steps} step(s) completed so far. "
                f"Showing the last {history_length} turn(s) with state/action/reward details. "
            )
            content += "Recent turns:\n---\n"
            for h_idx in range(history_start, turn_idx):
                actual_turn = h_idx + 1 + turn_offset
                h_turn = history[h_idx]
                content += f"Turn {actual_turn} State:\n{h_turn.get('state', '')}\n"
                content += f"Turn {actual_turn} Action: {h_turn.get('llm_response', '')}\n"
                if 'reward' in h_turn:
                    content += f"Turn {actual_turn} Reward: {h_turn['reward']}\n"
                content += "---\n"

        # Build current turn section
        actual_turn = turn_idx + 1 + turn_offset
        current_turn_prefix = "Current Turn " if history_start < turn_idx else ""

        warning = ""
        if include_warning and history[turn_idx].get('manager_invalid_action'):
            warning = "No valid action provided previously. Environment state remains the same. Please try again.\n"

        separator = "\n" if content else ""
        content += f"{separator}{current_turn_prefix}(Turn {actual_turn}):\n"
        content += (
            f"State:\n{history[turn_idx]['state']}\n{warning}"
            f"You have {history[turn_idx]['actions_left']} actions left. Always output: {format_prompt} "
            f"with no extra text. Strictly follow this format. {length_prompt}\n"
        )
        return content
