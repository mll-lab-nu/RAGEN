import re
from typing import List, Dict, Any, Optional
from .base import BaseMemory


class AlfWorldMemory(BaseMemory):
    """
    ALFWorld-specific memory implementation using verl-agent style formatting.

    Features:
    - Extracts task description from first observation
    - Cleans observations (removes welcome message, format instructions, etc.)
    - Uses compact history format: [Observation N: '...', Action N: '...']
    - Only shows admissible actions for current turn

    Format:
    Your task is to: <task>.
    Prior to this step, you have already taken N step(s).
    Below are the most recent K observations and the corresponding actions you took:
    [Observation 1: '...', Action 1: '...']
    ...
    You are now at step N+1 and your current observation is: <obs>
    Your admissible actions of the current situation are: [...]
    Now it's your turn to take an action...
    """

    # Patterns to remove from observations
    CLEAN_PATTERNS = [
        r"-= Welcome to TextWorld, ALFRED! =-",
        r"You have \d+ actions left\.?",
        r"Always output:.*?(?=\n|$)",
        r"Strictly follow this format\..*?(?=\n|$)",
        r"Max response length:.*?(?=\n|$)",
    ]

    def __init__(self):
        self._data = None
        self.batch_size = 0
        self._tasks = None  # Extracted task descriptions per environment

    def reset(self, batch_size: int):
        if self._data is not None:
            self._data.clear()
        self._data = [[] for _ in range(batch_size)]
        self._tasks = [None] * batch_size
        self.batch_size = batch_size

    def store(self, record: Dict[str, List[Any]]):
        for env_idx in range(self.batch_size):
            self._data[env_idx].append({k: record[k][env_idx] for k in record.keys()})

    def _clean_observation(self, obs: str) -> str:
        """Remove boilerplate from observation text."""
        cleaned = obs
        for pattern in self.CLEAN_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE | re.DOTALL)

        # Remove admissible actions section
        cleaned = re.sub(r"Admissible actions:.*?(?=\n|$)", "", cleaned, flags=re.DOTALL)

        # Clean up excessive whitespace
        cleaned = re.sub(r"\n\s*\n+", "\n", cleaned).strip()
        return cleaned

    def _extract_task(self, state: str) -> Optional[str]:
        """Extract task description from state."""
        match = re.search(r"Your task is to: (.+?)(?:\.\n|\n|$)", state, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_admissible_actions(self, state: str) -> Optional[str]:
        """Extract admissible actions list from state."""
        match = re.search(r"Admissible actions: \[(.+?)\]", state, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_action_from_response(self, llm_response: str) -> str:
        """Extract clean action from LLM response (removes think/answer tags)."""
        match = re.search(r'<answer>(.*?)</answer>', llm_response, re.DOTALL)
        return match.group(1).strip() if match else llm_response.strip()

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
        """Build user content using verl-agent style compact format."""
        env_id = env_output.get("env_id", 0)
        content_parts = []

        # 1. Extract and include task description (from first state)
        if history:
            if self._tasks[env_id % self.batch_size] is None:
                task = self._extract_task(history[0].get('state', ''))
                self._tasks[env_id % self.batch_size] = task
            task = self._tasks[env_id % self.batch_size]
            if task:
                content_parts.append(f"Your task is to: {task}.")

        # 2. Step count summary
        total_steps = turn_idx + turn_offset
        if total_steps > 0:
            content_parts.append(
                f"Prior to this step, you have already taken {total_steps} step(s)."
            )

        # 3. Compact history section
        if history_start < turn_idx:
            history_count = turn_idx - history_start
            content_parts.append(
                f"Below are the most recent {history_count} observations and "
                "the corresponding actions you took:"
            )

            for h_idx in range(history_start, turn_idx):
                h_turn = history[h_idx]
                actual_turn = h_idx + 1 + turn_offset

                # Clean observation
                clean_obs = self._clean_observation(h_turn.get('state', ''))
                # Truncate if too long
                if len(clean_obs) > 200:
                    clean_obs = clean_obs[:197] + "..."

                # Extract action from response
                action = self._extract_action_from_response(h_turn.get('llm_response', ''))

                content_parts.append(
                    f"[Observation {actual_turn}: '{clean_obs}', Action {actual_turn}: '{action}']"
                )

        # 4. Current step and observation
        current_turn = turn_idx + 1 + turn_offset
        current_state = history[turn_idx].get('state', '')
        clean_current = self._clean_observation(current_state)
        content_parts.append(
            f"You are now at step {current_turn} and your current observation is: {clean_current}"
        )

        # 5. Current admissible actions
        admissible = self._extract_admissible_actions(current_state)
        if admissible:
            content_parts.append(
                f"Your admissible actions of the current situation are: [{admissible}]"
            )

        # 6. Warning for invalid action
        if include_warning and history[turn_idx].get('manager_invalid_action'):
            content_parts.append(
                "No valid action provided previously. Environment state remains the same."
            )

        # 7. Format instructions
        content_parts.append(
            f"Now it's your turn to take an action. "
            f"You have {history[turn_idx]['actions_left']} actions left. "
            f"Always output: {format_prompt} with no extra text. {length_prompt}"
        )

        return "\n".join(content_parts)
