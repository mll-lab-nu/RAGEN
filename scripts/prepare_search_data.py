#!/usr/bin/env python3
"""
Prepare HotpotQA data for the RAGEN Search environment.

Downloads HotpotQA from HuggingFace and saves as parquet files with columns:
  - question (str)
  - ground_truth (str)
  - data_source (str)

Usage:
    python scripts/prepare_search_data.py
    python scripts/prepare_search_data.py --train_size 20000 --test_size 1000
    python scripts/prepare_search_data.py --output_dir data/search
"""

import argparse
import os

import pandas as pd
from datasets import load_dataset


def prepare_hotpotqa(output_dir: str = "data/search", train_size: int = None, test_size: int = None):
    """Download HotpotQA and convert to parquet format for RAGEN."""

    os.makedirs(output_dir, exist_ok=True)

    print("Loading HotpotQA dataset (distractor split)...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", trust_remote_code=True)

    # Process train split
    train_data = dataset["train"]
    if train_size is not None:
        train_data = train_data.select(range(min(train_size, len(train_data))))

    train_records = [
        {
            "question": ex["question"],
            "ground_truth": ex["answer"],
            "data_source": "hotpotqa",
        }
        for ex in train_data
    ]
    train_df = pd.DataFrame(train_records)
    train_path = os.path.join(output_dir, "train.parquet")
    train_df.to_parquet(train_path, index=False)
    print(f"Saved {len(train_df)} train examples to {train_path}")

    # Process validation split (used as test in RAGEN)
    val_data = dataset["validation"]
    if test_size is not None:
        val_data = val_data.select(range(min(test_size, len(val_data))))

    val_records = [
        {
            "question": ex["question"],
            "ground_truth": ex["answer"],
            "data_source": "hotpotqa",
        }
        for ex in val_data
    ]
    val_df = pd.DataFrame(val_records)
    val_path = os.path.join(output_dir, "val.parquet")
    val_df.to_parquet(val_path, index=False)
    print(f"Saved {len(val_df)} val examples to {val_path}")

    print(f"\nDone! Data saved to {output_dir}/")
    print(f"  train: {len(train_df)} examples")
    print(f"  val:   {len(val_df)} examples")
    print(f"\nSample question: {train_records[0]['question']}")
    print(f"Sample answer:   {train_records[0]['ground_truth']}")


def main():
    parser = argparse.ArgumentParser(description="Prepare HotpotQA data for RAGEN Search environment")
    parser.add_argument("--output_dir", default="data/search", help="Output directory for parquet files")
    parser.add_argument("--train_size", type=int, default=None, help="Max train examples (default: all ~90k)")
    parser.add_argument("--test_size", type=int, default=None, help="Max test examples (default: all ~7k)")
    args = parser.parse_args()

    prepare_hotpotqa(
        output_dir=args.output_dir,
        train_size=args.train_size,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
