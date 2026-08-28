"""
Analyze token/length statistics of the training and validation data.
Usage: python myscripts/data_stats.py
"""
import json
import pandas as pd
import numpy as np


def analyze_parquet(path, name):
    print(f"\n{'='*60}")
    print(f"  {name}: {path}")
    print(f"{'='*60}")
    df = pd.read_parquet(path)
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Prompt length (chars / 4 ≈ tokens)
    if "prompt" in df.columns:
        prompt_token_est = df["prompt"].apply(
            lambda x: len(str(x[0]["content"])) // 4 if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0
        )
        print(f"\nPrompt estimated tokens (chars/4):")
        print(prompt_token_est.describe().to_string())
        print(f"\nPrompt percentiles:")
        for p in [90, 95, 99, 100]:
            print(f"  p{p}: {prompt_token_est.quantile(p/100):.0f}")

    # Response / ground_truth length
    if "response" in df.columns:
        resp_token_est = df["response"].apply(lambda x: len(str(x)) // 4)
        print(f"\nResponse estimated tokens (chars/4):")
        print(resp_token_est.describe().to_string())
        print(f"\nResponse percentiles:")
        for p in [90, 95, 99, 100]:
            print(f"  p{p}: {resp_token_est.quantile(p/100):.0f}")

    if "reward_model" in df.columns:
        resp_token_est = df["reward_model"].apply(
            lambda x: len(json.loads(x)["ground_truth"]) // 4
        )
        print(f"\nGround truth (from reward_model) estimated tokens (chars/4):")
        print(resp_token_est.describe().to_string())
        print(f"\nGround truth percentiles:")
        for p in [90, 95, 99, 100]:
            print(f"  p{p}: {resp_token_est.quantile(p/100):.0f}")

    # eval_type distribution
    if "extra_info" in df.columns:
        eval_types = df["extra_info"].apply(lambda x: json.loads(x).get("eval_type", ""))
        print(f"\neval_type distribution:")
        print(eval_types.value_counts().to_string())

    # Prompt word count (secondary metric)
    if "prompt" in df.columns:
        word_counts = df["prompt"].apply(
            lambda x: len(str(x[0]["content"]).split()) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else 0
        )
        print(f"\nPrompt word counts:")
        print(word_counts.describe().to_string())


def main():
    files = [
        ("data/train_sft.parquet", "SFT Training"),
        ("data/train_rl.parquet", "RL Training"),
        ("data/val_sft.parquet", "SFT Validation"),
        ("data/val_rl.parquet", "RL Validation"),
    ]
    for path, name in files:
        try:
            analyze_parquet(path, name)
        except Exception as e:
            print(f"\nError processing {path}: {e}")

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    for path, name in files:
        try:
            df = pd.read_parquet(path)
            print(f"  {name}: {len(df)} samples")
        except Exception:
            pass


if __name__ == "__main__":
    main()
