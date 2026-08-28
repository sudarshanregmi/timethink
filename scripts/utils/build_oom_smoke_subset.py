"""Build a worst-case subset of train_sft_nothink_fair.parquet for OOM smoke testing.

Picks the top-K rows by an effective-sequence-length proxy:
    proxy = sum(len(ts) for ts in row.timeseries) + len(row.response) + len(prompt_text)

This biases toward the longest-context samples (length-extended RL prompts up to
seq_len=768 plus multivariate samples with multiple time series + long responses).
Forwarding and backpropping these worst-case rows at a candidate
`micro_batch_size_per_gpu` is the strict OOM test for the full run.

Output: data/train_oom_smoke.parquet (default K=256 rows).

Usage:
    python scripts/utils/build_oom_smoke_subset.py
    python scripts/utils/build_oom_smoke_subset.py --k 512
"""
import argparse
import os

import numpy as np
import pandas as pd


def proxy_len(row) -> int:
    ts_total = 0
    ts = row["timeseries"]
    if ts is not None:
        for arr in ts:
            ts_total += len(arr)
    resp = row["response"] if isinstance(row["response"], str) else ""
    prompt_text = ""
    if row["prompt"] is not None:
        for msg in row["prompt"]:
            c = msg.get("content", "") if isinstance(msg, dict) else ""
            if isinstance(c, str):
                prompt_text += c
    return ts_total + len(resp) + len(prompt_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/train_sft_nothink_fair.parquet")
    ap.add_argument("--out", default="data/train_oom_smoke.parquet")
    ap.add_argument("--k", type=int, default=256, help="Number of worst-case rows to keep.")
    args = ap.parse_args()

    print(f"Loading {args.src}")
    df = pd.read_parquet(args.src)
    print(f"Total rows: {len(df)}")

    print("Computing length proxies...")
    proxies = np.array([proxy_len(df.iloc[i]) for i in range(len(df))])
    p50 = int(np.percentile(proxies, 50))
    p90 = int(np.percentile(proxies, 90))
    p99 = int(np.percentile(proxies, 99))
    pmax = int(proxies.max())
    print(f"Proxy stats: p50={p50}  p90={p90}  p99={p99}  max={pmax}")

    top_idx = np.argsort(-proxies)[: args.k]
    subset = df.iloc[top_idx].reset_index(drop=True)
    sub_proxies = proxies[top_idx]
    print(f"Subset({args.k}) proxy stats: min={int(sub_proxies.min())}  median={int(np.median(sub_proxies))}  max={int(sub_proxies.max())}")

    subset.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}: {len(subset)} rows")
    print(f"Subset[0] preview:")
    print(f"  ts shape: {[len(t) for t in subset.iloc[0]['timeseries']]}")
    print(f"  response[:120]: {subset.iloc[0]['response'][:120]!r}")


if __name__ == "__main__":
    main()
