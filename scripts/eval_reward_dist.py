#!/usr/bin/env python3
"""
Quick reward distribution eval for SearchQA.
Runs Qwen2.5-7B on a sample of questions with multi-turn search,
then reports reward distribution.

Usage:
    cd /workspace/RAGEN
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_reward_dist.py [--n 100] [--temperature 0.5]
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ragen.env.search.reward import SearchRewardFn

SYSTEM_PROMPT = (
    "You are a search agent answering questions by searching for information.\n"
    "Use search[your query] to find relevant documents, and finish[your answer] to submit your final answer.\n\n"
    "You should first reason step-by-step about the current situation. "
    "This reasoning process MUST be enclosed within <think> </think> tags.\n"
    "Then provide your action within <answer>...</answer> tags.\n\n"
    "Examples:\n"
    "  <think>I need to find information about Ben Platt's father.</think>"
    "<answer>search[Ben Platt father parent]</answer>\n"
    "  <think>Based on the search results, Ben Platt's father is Henry Platt.</think>"
    "<answer>finish[Henry Platt]</answer>\n"
)


def extract_action(response: str) -> str:
    """Extract action from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    for pattern in [r"(search\[.*?\])", r"(finish\[.*?\])"]:
        m = re.search(pattern, response, re.DOTALL)
        if m:
            return m.group(1).strip()
    return response.strip()


def search_retrieval(query: str, port: int, top_k: int = 5) -> str:
    """Call retrieval server directly."""
    try:
        resp = requests.post(f"http://127.0.0.1:{port}/retrieve",
                             json={"query": query, "top_k": top_k}, timeout=30)
        data = resp.json()
        results = data.get("results", [])
        lines = []
        total_chars = 0
        for i, r in enumerate(results[:top_k], 1):
            content = r.get("content", "")
            if total_chars + len(content) > 4000:
                content = content[:max(0, 4000 - total_chars)]
            total_chars += len(content)
            score = r.get("score", 0.0)
            lines.append(f"[{i}] (score: {score:.4f}) {content}")
        return "\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Search error: {e}"


def run_episode(question, ground_truth, llm, tokenizer, sampling_params, args):
    """Run one multi-turn episode. Returns (reward, turns, action_types)."""
    reward_fn = SearchRewardFn()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\nAvailable actions: search[<query>], finish[<answer>]"},
    ]

    action_types = []
    for turn in range(1, args.max_turns + 1):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text

        action = extract_action(response)
        if action.startswith("search[") and action.endswith("]"):
            action_types.append("search")
            query = action[7:-1]
            results = search_retrieval(query, args.retrieval_port)
            obs = f"Search results for '{query}':\n{results}\n\nAvailable actions: search[<query>], finish[<answer>]"
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": obs})
        elif action.startswith("finish[") and action.endswith("]"):
            action_types.append("finish")
            answer = action[7:-1]
            reward, _ = reward_fn.compute_reward(answer, ground_truth)
            return reward, turn, action_types
        else:
            action_types.append("other")
            extracted = reward_fn.extract_answer_from_response(action)
            reward, _ = reward_fn.compute_reward(extracted, ground_truth)
            return reward, turn, action_types

    return 0.0, args.max_turns, action_types


def run_eval(args):
    # Load data BEFORE any CUDA operations
    print("Loading eval data...")
    df = pd.read_parquet("data/search/val.parquet")

    print(f"Loading model: {args.model}")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.85,
        max_model_len=5000,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=300,
    )

    n_samples = min(args.n, len(df))
    print(f"Evaluating {n_samples} questions, max_turns={args.max_turns}, temp={args.temperature}")

    rewards = []
    reward_details = []
    all_action_counts = Counter()
    turn_counts = []

    t0 = time.time()
    for i in range(n_samples):
        row = df.iloc[i]
        question = row["question"]
        ground_truth = row["ground_truth"]

        reward, turns, action_types = run_episode(
            question, ground_truth, llm, tokenizer, sampling_params, args
        )

        rewards.append(reward)
        turn_counts.append(turns)
        for at in action_types:
            all_action_counts[at] += 1

        reward_details.append({
            "idx": i,
            "question": question,
            "ground_truth": ground_truth,
            "reward": reward,
            "turns": turns,
            "action_types": action_types,
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            mean_r = sum(rewards) / len(rewards)
            print(f"  [{i+1}/{n_samples}] mean_reward={mean_r:.3f} | elapsed={elapsed:.1f}s")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"REWARD DISTRIBUTION ({n_samples} samples, {elapsed:.1f}s)")
    print(f"{'='*60}")

    reward_counter = Counter()
    for r in rewards:
        if r == 0.0:
            reward_counter["0.0"] += 1
        elif r == 1.0:
            reward_counter["1.0 (exact)"] += 1
        elif r > 0:
            bucket = f"{r:.1f}"
            reward_counter[bucket] += 1

    print(f"\nReward buckets:")
    for bucket in sorted(reward_counter.keys()):
        cnt = reward_counter[bucket]
        pct = cnt / n_samples * 100
        bar = "#" * int(pct / 2)
        print(f"  {bucket:>12s}: {cnt:4d} ({pct:5.1f}%) {bar}")

    mean_r = sum(rewards) / len(rewards)
    nonzero = sum(1 for r in rewards if r > 0)
    mean_turns = sum(turn_counts) / len(turn_counts)

    print(f"\nMean reward:    {mean_r:.4f}")
    print(f"Nonzero reward: {nonzero}/{n_samples} ({nonzero/n_samples*100:.1f}%)")
    print(f"Mean turns:     {mean_turns:.2f}")
    print(f"Action types:   {dict(all_action_counts)}")

    # Save details
    out_path = Path("logs/reward_dist.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": {"mean_reward": mean_r, "nonzero_pct": nonzero/n_samples,
                                "mean_turns": mean_turns, "action_counts": dict(all_action_counts),
                                "n_samples": n_samples, "model": args.model, "temperature": args.temperature},
                    "details": reward_details}, f, indent=2, ensure_ascii=False)
    print(f"\nDetails saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--retrieval-port", type=int, default=8000)
    args = parser.parse_args()
    run_eval(args)
