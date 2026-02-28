import json
import random
from typing import Any, Dict, Optional, Tuple

from datasets import load_dataset
from ragen.env.base import BaseLanguageBasedEnv
from ragen.utils import all_seed
from .config import DeepCoderEnvConfig
from .utils import prepare_deepcoder_data, run_deepcoder_sandbox


class DeepCoderEnv(BaseLanguageBasedEnv):
    def __init__(self, config: Optional[DeepCoderEnvConfig] = None):
        super(DeepCoderEnv, self).__init__()
        self.config = config if config is not None else DeepCoderEnvConfig()
        self.render_mode = self.config.render_mode

        self.render_cache: Optional[str] = None
        self.current_prompt: Optional[str] = None
        self.current_solution: Optional[str] = None
        self.current_tests: Optional[list] = None
        self.current_metadata: Optional[dict] = None
        self.current_starter_code: Optional[str] = None
        self.last_feedback: Optional[str] = None
        self.step_num = 0
        self._load_data()

    def _load_data(self) -> None:
        dataset_path = self.config.dataset_path
        if dataset_path:
            self.dataset = load_dataset(path=dataset_path, cache_dir=self.config.cache_dir)
            return

        dataset_bundle = None
        try:
            train_dataset, test_dataset = prepare_deepcoder_data()
            dataset_bundle = {"train": train_dataset, "test": test_dataset}
        except Exception:
            dataset_bundle = None

        if dataset_bundle is not None:
            self.dataset = dataset_bundle
        else:
            self.dataset = None

    def _sample_problem(self, seed: Optional[int] = None) -> Tuple[str, str]:
        if self.dataset is None:
            prompt = "Write a function add(a, b) that returns a + b."
            solution = "def add(a, b):\n    return a + b"
            return prompt, solution

        split = self.config.split or next(iter(self.dataset.keys()))
        dataset_split = self.dataset[split]
        index = random.randint(0, len(dataset_split) - 1)
        item = dataset_split[index]
        prompt = item.get("question", item.get("prompt", item.get("problem", str(item))))
        solution = item.get("canonical_solution", item.get("solution", ""))
        self.current_starter_code = item.get("starter_code", "") or ""
        self.current_metadata = {}
        raw_meta = item.get("metadata", {})
        if isinstance(raw_meta, str):
            try:
                self.current_metadata = json.loads(raw_meta)
            except Exception:
                self.current_metadata = {}
        elif isinstance(raw_meta, dict):
            self.current_metadata = raw_meta
        raw_tests = item.get("ground_truth", item.get("tests", "[]"))
        if isinstance(raw_tests, str):
            try:
                self.current_tests = json.loads(raw_tests)
            except Exception:
                self.current_tests = []
        else:
            self.current_tests = raw_tests if isinstance(raw_tests, list) else []
        return prompt, solution

    def reset(self, seed: Optional[int] = None, mode: Optional[str] = None) -> Any:
        with all_seed(seed):
            self.current_prompt, self.current_solution = self._sample_problem(seed=seed)
        self.step_num = 0
        self.last_feedback = None
        self.render_cache = self.current_prompt
        return self.render_cache

    def step(self, action: str) -> Tuple[Any, float, bool, Dict]:
        is_valid = bool(action.strip())
        if not is_valid:
            observation = "Invalid action."
            reward = 0.0
            done = False
            info = {
                "action_is_effective": False,
                "action_is_valid": False,
                "success": False,
                "error": "Empty action",
            }
            self.last_feedback = info["error"]
            if self.current_prompt:
                self.render_cache = f"{self.current_prompt}\n\nFeedback: {self.last_feedback}"
            else:
                self.render_cache = observation
            return observation, reward, done, info

        is_correct, detail = run_deepcoder_sandbox(
            action,
            tests=self.current_tests or [],
            metadata=self.current_metadata or {},
            starter_code=self.current_starter_code or "",
        )
        reward = 1.0 if is_correct else 0.0
        observation = "Correct!" if is_correct else "Incorrect."
        done = True if is_correct else (self.step_num + 1) >= self.config.max_steps
        self.step_num += 1
        self.last_feedback = detail
        info = {
            "action_is_effective": is_correct,
            "action_is_valid": is_valid,
            "success": is_correct,
            "detail": detail,
        }
        if not done and self.current_prompt:
            self.render_cache = f"{self.current_prompt}\n\nFeedback: {detail}"
        else:
            self.render_cache = observation
        return observation, reward, done, info

    def render(self, mode: Optional[str] = None) -> Any:
        return self.render_cache

    def compute_reward(self, action: str, **kwargs) -> float:
        is_correct, _ = run_deepcoder_sandbox(
            action,
            tests=self.current_tests or [],
            metadata=self.current_metadata or {},
            starter_code=self.current_starter_code or "",
        )
        return 1.0 if is_correct else 0.0

    def close(self) -> None:
        self.render_cache = None


if __name__ == "__main__":
    try:
        env = DeepCoderEnv()
        obs = env.reset(seed=42)
        print("Reset OK. Observation preview:")
        print(obs if isinstance(obs, str) else obs)
        sample_code = (
            "class Solution:\n"
            "    def shiftDistance(self, s: str, t: str, nextCost: list[int], previousCost: list[int]) -> int:\n"
            "        n = len(s)\n"
            "        # prefix sums for forward (next) costs\n"
            "        pref_next = [0] * 27\n"
            "        for i in range(26):\n"
            "            pref_next[i + 1] = pref_next[i] + nextCost[i]\n"
            "        # prefix sums for backward (previous) costs\n"
            "        pref_prev = [0] * 27\n"
            "        for i in range(26):\n"
            "            pref_prev[i + 1] = pref_prev[i] + previousCost[i]\n"
            "\n"
            "        def cost_next(i: int, j: int) -> int:\n"
            "            if i <= j:\n"
            "                return pref_next[j] - pref_next[i]\n"
            "            return pref_next[26] - pref_next[i] + pref_next[j]\n"
            "\n"
            "        def cost_prev(i: int, j: int) -> int:\n"
            "            if i >= j:\n"
            "                return pref_prev[i + 1] - pref_prev[j + 1]\n"
            "            return pref_prev[i + 1] + (pref_prev[26] - pref_prev[j + 1])\n"
            "\n"
            "        total = 0\n"
            "        for a, b in zip(s, t):\n"
            "            i = ord(a) - 97\n"
            "            j = ord(b) - 97\n"
            "            forward = cost_next(i, j)\n"
            "            backward = cost_prev(i, j)\n"
            "            total += forward if forward <= backward else backward\n"
            "        return total\n"
        )
        obs2, reward, done, info = env.step(sample_code)
        print("Step OK.")
        print({"reward": reward, "done": done, "info": info})
        print("Observation:", obs2[:200] if isinstance(obs2, str) else obs2)
    except Exception as e:
        print(f"Smoke test failed: {e}")
