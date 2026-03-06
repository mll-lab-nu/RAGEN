"""
Search reward function for HotpotQA-style question answering.
Migrated from rllm/rewards/search_reward.py, with rllm-specific dependencies removed.

Evaluation uses:
- Exact Match (EM): normalized string comparison
- F1 Score: token-level precision/recall

Reference: HotpotQA / SQuAD evaluation standards.
"""

import re
import string
from collections import Counter
from typing import Any, List, Tuple, Union


class SearchRewardFn:
    """Reward function for search-based QA tasks using F1 and Exact Match."""

    def __init__(self, correct_reward: float = 1.0, incorrect_reward: float = 0.0, f1_threshold: float = 0.3):
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.f1_threshold = f1_threshold

    def normalize_answer(self, s: str) -> str:
        """Normalize answer text for evaluation (following HotpotQA/SQuAD standards)."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(self, prediction: str, ground_truth: str) -> Tuple[float, float, float]:
        """Calculate token-level F1 score between prediction and ground truth."""
        normalized_prediction = self.normalize_answer(prediction)
        normalized_ground_truth = self.normalize_answer(ground_truth)

        ZERO_METRIC = (0.0, 0.0, 0.0)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            return ZERO_METRIC

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return ZERO_METRIC
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

    def exact_match_score(self, prediction: str, ground_truth: str) -> bool:
        """Calculate exact match score after normalization."""
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)

    def extract_answer_from_response(self, response: str) -> str:
        """
        Fallback: extract answer from free-form LLM response text.
        Used when the agent doesn't follow the finish[...] format.
        Migrated from rllm's RewardSearchFn.extract_answer_from_response().
        """
        response = response.strip()

        # Remove thinking tags
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response = re.sub(r"\s+", " ", response).strip()

        if not response:
            return ""

        # 1. Look for \boxed{} content (rllm format)
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
        if boxed_match:
            return boxed_match.group(1).strip()

        # 2. Bold text
        bold_patterns = [r"\*\*([^*]+)\*\*", r"\*([^*]+)\*"]
        for pattern in bold_patterns:
            matches = re.findall(pattern, response)
            substantive = [m.strip() for m in matches if len(m.strip()) > 2 and not re.match(r"^[^\w]*$", m.strip())]
            if substantive:
                return substantive[0]

        # 3. Direct answer patterns
        answer_patterns = [
            r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*:?\s*([^.!?]+)",
            r"(?:therefore|thus|so|hence)\s*,?\s*([^.!?]+)",
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                answer = re.sub(r"^\W+|\W+$", "", answer)
                if len(answer) > 3:
                    return answer

        # 4. Fallback: first substantial sentence
        sentences = [s.strip() for s in re.split(r"[.!?]+", response) if len(s.strip()) > 5]
        if sentences:
            return sentences[0]

        return response[:100].strip()

    def evaluate_answer(self, model_answer: str, ground_truth: Union[str, List[str]]) -> Tuple[bool, float, dict]:
        """
        Evaluate model answer against ground truth(s).

        Returns:
            (is_correct, max_f1, metadata_dict)
        """
        if isinstance(ground_truth, str):
            ground_truths = [ground_truth]
        else:
            ground_truths = ground_truth

        max_f1 = 0.0
        max_em = False
        best_match = ""
        best_precision = 0.0
        best_recall = 0.0
        eval_method = None

        for gt in ground_truths:
            gt_str = str(gt).strip()

            em = self.exact_match_score(model_answer, gt_str)
            if em:
                max_em = True
                max_f1 = 1.0
                best_match = gt_str
                best_precision = 1.0
                best_recall = 1.0
                eval_method = "exact_match"
                break

            f1, precision, recall = self.f1_score(model_answer, gt_str)
            if f1 > max_f1:
                max_f1 = f1
                best_match = gt_str
                best_precision = precision
                best_recall = recall
                eval_method = "f1_score"

        is_correct = max_em or max_f1 >= self.f1_threshold

        metadata = {
            "extracted_answer": model_answer,
            "ground_truths": ground_truths,
            "best_match": best_match,
            "f1_score": max_f1,
            "precision": best_precision,
            "recall": best_recall,
            "exact_match": max_em,
            "evaluation_method": eval_method,
            "f1_threshold": self.f1_threshold,
        }

        return is_correct, max_f1, metadata

    def compute_reward(self, model_answer: str, ground_truth: Union[str, List[str]]) -> Tuple[float, dict]:
        """
        Compute reward for a model answer.

        Returns:
            (reward_float, metadata_dict)
        """
        is_correct, f1, metadata = self.evaluate_answer(model_answer, ground_truth)

        if is_correct:
            if metadata.get("exact_match", False):
                reward = self.correct_reward
            else:
                reward = self.correct_reward * f1
        else:
            reward = self.incorrect_reward

        metadata["reward"] = reward
        return reward, metadata
