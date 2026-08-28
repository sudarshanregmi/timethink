"""Build the "fair" no-think SFT dataset.

The plain `sft_nothink` baseline trains only on `train_sft_nothink.parquet`
(95K rows = post-`</think>` answers from the SFT split). That undercounts vs.
SFT+RL, which additionally exposes the model to all RL prompts during the
RL phase.

For an apples-to-apples comparison we union the post-think NL answers from
both splits and train pure SFT on the merged set:

    train_sft_nothink_fair = train_sft_nothink (95,354)
                           ∪ strip_think(train_rl_public)  (115,200)
                           = 210,554 rows

`train_rl_public.parquet` has no `response` column (RL only stores
`reward_model.ground_truth`), so the post-think answer is sourced from
`train_rl_public.jsonl::output`, which has the full
`<think>...</think>{NL_answer}` shape. Verified row-aligned 1:1 with the
parquet (eval_type matches across all 115,200 rows).

Val mirrors the same union (1K + 1K = 2K) but the val RL parquet does have
a `response` column, so we strip directly from the parquet.

Usage:
    python scripts/utils/build_sft_nothink_fair.py
"""
import argparse
import json
import os
import re

import pandas as pd

THINK_RE = re.compile(r"^<think>.*?</think>\s*", flags=re.DOTALL)


def strip_think(text):
    if not isinstance(text, str):
        return text
    return THINK_RE.sub("", text, count=1)


def build_train(data_dir: str, out_path: str) -> None:
    sft_path = os.path.join(data_dir, "train_sft_nothink.parquet")
    rl_pq_path = os.path.join(data_dir, "train_rl_public.parquet")
    rl_jl_path = os.path.join(data_dir, "train_rl_public.jsonl")

    sft = pd.read_parquet(sft_path)
    rl = pd.read_parquet(rl_pq_path)
    print(f"Loaded SFT: {len(sft)} rows, RL: {len(rl)} rows")

    responses = []
    with open(rl_jl_path) as f:
        for line in f:
            rec = json.loads(line)
            responses.append(strip_think(rec["output"]))
    assert len(responses) == len(rl), \
        f"jsonl/parquet length mismatch: {len(responses)} vs {len(rl)}"

    rl = rl.drop(columns=["reward_model"])
    rl["response"] = responses
    rl = rl[sft.columns.tolist()]

    n_residual = int(rl["response"].astype(str).str.contains("</think>").sum())
    n_empty = int((rl["response"].astype(str).str.strip() == "").sum())
    assert n_residual == 0, f"residual </think> in {n_residual} RL rows"
    assert n_empty == 0, f"{n_empty} RL rows empty after strip"

    combined = pd.concat([sft, rl], ignore_index=True)
    combined.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}: {len(combined)} rows ({len(sft)} SFT + {len(rl)} RL)")
    print(f"  RL sample[0]: {rl['response'].iloc[0][:160]!r}")


def build_val(data_dir: str, out_path: str) -> None:
    sft_path = os.path.join(data_dir, "val_sft_nothink.parquet")
    rl_path = os.path.join(data_dir, "val_rl.parquet")

    sft = pd.read_parquet(sft_path)
    rl = pd.read_parquet(rl_path)
    print(f"Loaded val SFT: {len(sft)} rows, val RL: {len(rl)} rows")

    rl["response"] = rl["response"].map(strip_think)
    rl = rl[sft.columns.tolist()]

    n_residual = int(rl["response"].astype(str).str.contains("</think>").sum())
    n_empty = int((rl["response"].astype(str).str.strip() == "").sum())
    assert n_residual == 0, f"residual </think> in {n_residual} val RL rows"
    assert n_empty == 0, f"{n_empty} val RL rows empty after strip"

    combined = pd.concat([sft, rl], ignore_index=True)
    combined.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}: {len(combined)} rows ({len(sft)} SFT + {len(rl)} RL)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    build_train(args.data_dir, os.path.join(args.data_dir, "train_sft_nothink_fair.parquet"))
    build_val(args.data_dir, os.path.join(args.data_dir, "val_sft_nothink_fair.parquet"))


if __name__ == "__main__":
    main()
