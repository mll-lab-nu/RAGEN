# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ROUGE-based reward scoring for ELI5 dataset.
Based on the evaluation approach used in the ELI5 repository.
"""

import re
from typing import Dict, List, Union
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not installed. Install with: pip install rouge-score")


def extract_solution(solution_str: str) -> str:
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    # If there are 0 matches, return None (no answer found)
    if len(matches) == 0:
        return None
    
    # If there are 1 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_rouge_score(prediction: str, reference: Union[str, List[str]], rouge_type: str = "rougeL") -> float:
    """
    Compute ROUGE score between prediction and reference(s).
    
    Args:
        prediction: The predicted answer text
        reference: The reference answer(s) - can be a string or list of strings
        rouge_type: Type of ROUGE to compute ('rouge1', 'rouge2', 'rougeL', 'rougeLsum')
    
    Returns:
        ROUGE F1 score (float between 0 and 1)
    """
    if not ROUGE_AVAILABLE:
        raise ImportError("rouge-score library is required. Install with: pip install rouge-score")
    
    if not prediction or not reference:
        return 0.0
    
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    
    # Handle multiple references by taking the maximum score
    if isinstance(reference, list):
        scores = []
        for ref in reference:
            if ref:  # Skip empty references
                score = scorer.score(ref, prediction)
                scores.append(score[rouge_type].fmeasure)
        return max(scores) if scores else 0.0
    else:
        score = scorer.score(reference, prediction)
        return score[rouge_type].fmeasure


def compute_score_rouge(solution_str: str, ground_truth: Dict, method: str = 'rougeL', format_score: float = 0.0, score: float = 1.0, return_dict: bool = False) -> Union[float, Dict]:
    """
    The scoring function for ROUGE-based evaluation (for ELI5 dataset).
    
    Args:
        solution_str: The solution text containing <answer> tags
        ground_truth: The ground truth dict with 'target' key containing reference answer(s)
        method: ROUGE type to use ('rouge1', 'rouge2', 'rougeL', 'rougeLsum')
        format_score: The score for reasonable format (when ROUGE is low but answer is valid)
        score: The maximum score for high ROUGE match
        return_dict: If True, return a dict with separate metrics; if False, return total reward (backward compatible)
    
    Returns:
        If return_dict=False: Reward score (float)
        If return_dict=True: Dict with keys: 'total_reward', 'rouge_score', 'format_reward'
    """
    if not ROUGE_AVAILABLE:
        raise ImportError("rouge-score library is required. Install with: pip install rouge-score")
    
    answer = extract_solution(solution_str=solution_str)
    
    if answer is None:
        print(f"[DEBUG ROUGE] Failed to extract answer from solution string")
        print(f"[DEBUG ROUGE] Solution string (first 200 chars): {solution_str[:200]}")
        if return_dict:
            return {"total_reward": 0.0, "rouge_score": 0.0, "format_reward": 0.0}
        return 0.0
    
    # Get reference answers
    references = ground_truth.get('target', [])
    if not references:
        print(f"[DEBUG ROUGE] No references found in ground_truth")
        final_reward = format_score if answer.strip() else 0.0
        if return_dict:
            return {"total_reward": final_reward, "rouge_score": 0.0, "format_reward": format_score if answer.strip() else 0.0}
        return final_reward
    
    # Ensure references is a list
    if not isinstance(references, list):
        references = [references]
    
    print(f"[DEBUG ROUGE] Number of references: {len(references)}")
    
    # Compute ROUGE score
    rouge_score = compute_rouge_score(answer, references, rouge_type=method)
    
    # Ensure answer is non-empty and has minimum length
    answer_stripped = answer.strip()
    if not answer_stripped or len(answer_stripped) <= 10:
        print(f"[DEBUG ROUGE] Answer too short or empty: len={len(answer_stripped) if answer_stripped else 0}")
        if return_dict:
            return {"total_reward": 0.0, "rouge_score": 0.0, "format_reward": 0.0}
        return 0.0
    
    # Debug logging
    print(f"[DEBUG ROUGE] Answer length: {len(answer_stripped)}, ROUGE score: {rouge_score:.4f}")
    if len(references) > 0:
        print(f"[DEBUG ROUGE] Reference length: {len(references[0]) if isinstance(references[0], str) else 'N/A'}")
    
    # Simple reward formula: format_score + (score - format_score) * rouge_score
    # This gives: reward = 0.1 + 0.9 * rouge_score (when format_score=0.1, score=1.0)
    # Format check is already done in the environment, so we only get here if format is correct
    final_reward = format_score + (score - format_score) * rouge_score
    print(f"[DEBUG ROUGE] ROUGE={rouge_score:.4f}, reward = {final_reward:.4f}")
    
    if return_dict:
        return {
            "total_reward": final_reward,
            "rouge_score": rouge_score,  # ROUGE score (used for evaluation)
            "format_reward": format_score
        }
    return final_reward

