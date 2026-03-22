#!/usr/bin/env python3
"""
Multi-rollout inference for reward distribution diagnosis.

For each prompt, runs N rollouts (with temperature sampling) and records
per-rollout rewards. Output is a JSON file consumed by plot_reward_matrix.py.

This script provides the scaffolding for multi-turn inference. To adapt
for your environment, implement the functions marked with PLACEHOLDER.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/reward_diagnosis/run_inference.py \
        --model Qwen/Qwen2.5-3B-Instruct \
        --n_prompts 50 \
        --rollouts_per_prompt 8 \
        --temperature 0.7 \
        --max_turns 5 \
        --output logs/inference_results.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ============================================================================
# PLACEHOLDER: Environment-specific configuration
# Modify the following sections to adapt for your environment.
# ============================================================================

# PLACEHOLDER: Import your environment's reward function

# PLACEHOLDER: System prompt for the agent
SYSTEM_PROMPT = "PLACEHOLDER: Replace with your environment's system prompt."


def extract_action(response: str) -> str:
    """PLACEHOLDER: Extract the action from model response.

    Parse the model's raw text output and return the action string.
    """
    raise NotImplementedError(
        "extract_action() must be implemented for your environment."
    )


def get_observation(action: str, args) -> str:
    """PLACEHOLDER: Execute an action and return the environment observation.

    Adapt this for your environment's action execution logic.
    Return a string observation if the action is non-terminal (episode continues),
    or None if the action is terminal (episode should end and compute reward).

    Example for SearchQA: parse search[query], call retrieval server, return results.
    Example for other envs: execute the action in your env, return the observation.
    """
    raise NotImplementedError(
        "get_observation() must be implemented for your environment. "
        "See PLACEHOLDER comments for guidance."
    )


def compute_reward(action: str, ground_truth: str) -> float:
    """PLACEHOLDER: Compute reward for a terminal action.

    Adapt this for your environment's reward function.
    Should return a float reward value (typically 0.0 to 1.0).
    """
    raise NotImplementedError(
        "compute_reward() must be implemented for your environment. "
        "See PLACEHOLDER comments for guidance."
    )


def load_data(data_path: str):
    """PLACEHOLDER: Load evaluation data.

    Must return a list of dicts, each with at least 'question' and 'ground_truth' keys.
    Adapt the column names for your environment's data format.
    """
    df = pd.read_parquet(data_path)
    return df


# ============================================================================
# Core inference logic (environment-agnostic)
# ============================================================================

def build_initial_messages(question: str) -> list:
    """PLACEHOLDER: Build the initial conversation messages for an episode.

    Adapt the user message format for your environment.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]


def is_terminal_action(action: str) -> bool:
    """PLACEHOLDER: Check if an action is terminal (ends the episode).

    Return True if the action should end the episode and compute reward.
    Return False if the action is non-terminal and the episode should continue.
    """
    raise NotImplementedError(
        "is_terminal_action() must be implemented for your environment."
    )


def format_observation(action: str, obs: str) -> str:
    """PLACEHOLDER: Format the environment observation as a user message.

    Given the action taken and the raw observation string,
    return the text to append as the next user message.
    """
    raise NotImplementedError(
        "format_observation() must be implemented for your environment."
    )


def extract_answer(action: str) -> str:
    """PLACEHOLDER: Extract the final answer from a terminal action.

    Given a terminal action string, return the answer to be evaluated.
    """
    raise NotImplementedError(
        "extract_answer() must be implemented for your environment."
    )


def run_episode(question, ground_truth, llm, tokenizer, sampling_params, args):
    """Run one multi-turn episode. Returns (reward, n_turns, action_types, final_answer)."""
    messages = build_initial_messages(question)

    action_types = []
    for turn in range(1, args.max_turns + 1):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = len(tokenizer.encode(prompt))
        if prompt_tokens > args.max_model_len - args.max_new_tokens:
            action_types.append("truncated")
            return 0.0, turn, action_types, ""
        try:
            outputs = llm.generate([prompt], sampling_params)
        except ValueError:
            action_types.append("truncated")
            return 0.0, turn, action_types, ""
        response = outputs[0].outputs[0].text

        action = extract_action(response)

        if is_terminal_action(action):
            action_types.append("terminal")
            answer = extract_answer(action)
            reward = compute_reward(answer, ground_truth)
            return reward, turn, action_types, answer
        else:
            action_types.append("non_terminal")
            obs = get_observation(action, args)
            obs_text = format_observation(action, obs)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": obs_text})

    return 0.0, args.max_turns, action_types, ""


def main():
    parser = argparse.ArgumentParser(description="Multi-rollout inference for reward distribution diagnosis")
    # Model settings
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=5000)
    parser.add_argument("--max_new_tokens", type=int, default=300, help="Max tokens per generation")
    # Inference settings
    parser.add_argument("--n_prompts", type=int, default=50, help="Number of prompts to evaluate")
    parser.add_argument("--rollouts_per_prompt", type=int, default=8, help="Number of rollouts per prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_turns", type=int, default=5, help="Max turns per episode")
    # PLACEHOLDER: Add environment-specific arguments here
    # Data settings
    parser.add_argument("--data_path", required=True, help="Path to evaluation data (parquet)")
    parser.add_argument("--output", default="logs/inference_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}...")
    df = load_data(args.data_path)
    n_prompts = min(args.n_prompts, len(df))

    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(df))[:n_prompts]

    print(f"Loading model: {args.model}")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    print(f"Running {n_prompts} prompts x {args.rollouts_per_prompt} rollouts "
          f"(temp={args.temperature}, max_turns={args.max_turns})")
    print("=" * 60)

    results = []
    t0 = time.time()

    for pi, idx in enumerate(indices):
        row = df.iloc[idx]
        question = row["question"]
        ground_truth = row["ground_truth"]

        rollout_rewards = []
        rollout_details = []

        for ri in range(args.rollouts_per_prompt):
            reward, turns, action_types, answer = run_episode(
                question, ground_truth, llm, tokenizer, sampling_params, args
            )
            rollout_rewards.append(reward)
            rollout_details.append({
                "rollout_idx": ri,
                "reward": reward,
                "turns": turns,
                "action_types": action_types,
                "answer": answer,
            })

        rewards_arr = np.array(rollout_rewards)
        rv = float(np.var(rewards_arr))
        mean_r = float(np.mean(rewards_arr))

        results.append({
            "prompt_idx": int(idx),
            "question": question,
            "ground_truth": ground_truth if isinstance(ground_truth, str) else str(ground_truth),
            "rewards": rollout_rewards,
            "reward_mean": mean_r,
            "reward_variance": rv,
            "rollouts": rollout_details,
        })

        if (pi + 1) % 10 == 0 or pi == 0:
            elapsed = time.time() - t0
            print(f"  [{pi+1}/{n_prompts}] q=\"{question[:60]}...\" "
                  f"rewards={rollout_rewards} mean={mean_r:.3f} RV={rv:.4f} "
                  f"| {elapsed:.1f}s")

    elapsed = time.time() - t0

    # Summary stats
    all_rvs = [r["reward_variance"] for r in results]
    all_means = [r["reward_mean"] for r in results]
    n_zero_rv = sum(1 for rv in all_rvs if rv == 0.0)
    n_all_correct = sum(1 for r in results if all(rw == 1.0 for rw in r["rewards"]))
    n_all_wrong = sum(1 for r in results if all(rw == 0.0 for rw in r["rewards"]))
    n_mixed = n_prompts - n_all_correct - n_all_wrong

    print(f"\n{'=' * 60}")
    print(f"INFERENCE SUMMARY ({n_prompts} prompts x {args.rollouts_per_prompt} rollouts, {elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"Mean reward across all:  {np.mean(all_means):.4f}")
    print(f"Mean RV:                 {np.mean(all_rvs):.4f}")
    print(f"Prompts with RV=0:       {n_zero_rv}/{n_prompts} ({n_zero_rv/n_prompts*100:.1f}%)")
    print(f"  - All correct (easy):  {n_all_correct}")
    print(f"  - All wrong (hard):    {n_all_wrong}")
    print(f"  - Mixed (learnable):   {n_mixed}")
    print()

    if n_mixed / max(n_prompts, 1) < 0.2:
        print("WARNING: <20% prompts have mixed rewards. RL signal may be weak.")
        print("  -> Check prompt format, environment setup, and model capability.")
    else:
        print(f"OK: {n_mixed/n_prompts*100:.0f}% prompts have mixed rewards. RL should have signal.")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "model": args.model,
            "n_prompts": n_prompts,
            "rollouts_per_prompt": args.rollouts_per_prompt,
            "temperature": args.temperature,
            "max_turns": args.max_turns,
            "data_path": args.data_path,
            "seed": args.seed,
        },
        "summary": {
            "mean_reward": float(np.mean(all_means)),
            "mean_rv": float(np.mean(all_rvs)),
            "n_all_correct": n_all_correct,
            "n_all_wrong": n_all_wrong,
            "n_mixed": n_mixed,
            "elapsed_seconds": elapsed,
        },
        "prompts": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
    print(f"Next step: python scripts/reward_diagnosis/plot_reward_matrix.py --input {output_path}")


if __name__ == "__main__":
    main()
