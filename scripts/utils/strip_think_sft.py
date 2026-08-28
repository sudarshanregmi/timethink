"""Strip <think>...</think> blocks from SFT parquet responses.

Produces a no-think carbon copy of train_sft.parquet (and val_sft.parquet
used for SFT validation) so we can train an SFT baseline on just the
final post-think answer.

Usage:
    python scripts/utils/strip_think_sft.py \
        --data-dir data \
        --suffix _nothink
"""
import argparse
import os
import re

import pandas as pd

THINK_RE = re.compile(r"^<think>.*?</think>\s*", flags=re.DOTALL)


def strip_think(text: str) -> str:
    if not isinstance(text, str):
        return text
    return THINK_RE.sub("", text, count=1)


def process(in_path: str, out_path: str, response_col: str = "response") -> None:
    df = pd.read_parquet(in_path)
    assert response_col in df.columns, f"{response_col} missing in {in_path}"

    n = len(df)
    n_with_think = int(df[response_col].astype(str).str.contains("</think>").sum())

    df[response_col] = df[response_col].map(strip_think)

    n_residual = int(df[response_col].astype(str).str.contains("</think>").sum())
    n_empty = int((df[response_col].astype(str).str.strip() == "").sum())

    print(f"{in_path} -> {out_path}")
    print(f"  rows: {n}")
    print(f"  had <think>: {n_with_think}")
    print(f"  residual </think> after strip: {n_residual}")
    print(f"  empty after strip: {n_empty}")
    sample = str(df.iloc[0][response_col])[:160]
    print(f"  sample[0]: {sample!r}")

    df.to_parquet(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--suffix", default="_nothink",
                    help="Suffix appended to the output filenames before .parquet")
    ap.add_argument("--inputs", nargs="+",
                    default=["train_sft.parquet", "val_sft.parquet"],
                    help="Parquet files (under --data-dir) to strip.")
    args = ap.parse_args()

    for name in args.inputs:
        in_path = os.path.join(args.data_dir, name)
        stem, ext = os.path.splitext(name)
        out_path = os.path.join(args.data_dir, f"{stem}{args.suffix}{ext}")
        process(in_path, out_path)


if __name__ == "__main__":
    main()
