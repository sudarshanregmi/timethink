#!/usr/bin/env python3
"""Exit 0 if an eval dir's results look valid, 1 if they need re-running.

An eval is considered broken/stale when ANY of:
  * judge was down → >30% of rows have a missing reasoning_score (NAN_THRESHOLD)
  * generated_answer.json was regenerated but CSV is stale → row count mismatch
    between the JSON input and the CSV output

The row-count check guards against a real failure mode (caught 2026-04-25):
re-running inference with a different sample size produces a new
generated_answer.json, but the prior CSV remains "valid" by NaN fraction
and the wrapper would skip re-judging — silently shipping stale numbers.

Usage:
    python scripts/utils/check_eval_valid.py exp/sft/test_lite
"""

import json
import sys
from pathlib import Path

import pandas as pd

NAN_THRESHOLD = 0.30


def is_valid(exp_dir: Path) -> tuple[bool, str]:
    csv = exp_dir / "evaluation_report_optimized.csv"
    summary = exp_dir / "evaluation_report_summary.txt"
    generated = exp_dir / "generated_answer.json"
    if not csv.exists():
        return False, f"missing {csv.name}"
    if not summary.exists():
        return False, f"missing {summary.name}"
    try:
        df = pd.read_csv(csv)
    except Exception as e:
        return False, f"csv unreadable: {e}"
    if len(df) == 0:
        return False, "csv is empty"
    if "reasoning_score" not in df.columns:
        return False, "no reasoning_score column"
    # Row-count vs input JSON. If the input was regenerated with different
    # sample size, the prior CSV is stale even if NaN fraction is fine.
    if generated.exists():
        try:
            with open(generated) as f:
                n_input = len(json.load(f))
        except Exception as e:
            return False, f"generated_answer.json unreadable: {e}"
        if n_input != len(df):
            return False, (
                f"row count mismatch: generated_answer.json has {n_input} rows "
                f"but CSV has {len(df)} (input was regenerated; CSV is stale)"
            )
    nan_frac = df["reasoning_score"].isna().mean()
    if nan_frac > NAN_THRESHOLD:
        return False, f"reasoning_score NaN fraction {nan_frac:.0%} > {NAN_THRESHOLD:.0%} (judge likely didn't respond)"
    return True, f"ok (n={len(df)}, nan_frac={nan_frac:.2%})"


def main():
    if len(sys.argv) != 2:
        print("usage: check_eval_valid.py <exp_dir>", file=sys.stderr)
        sys.exit(2)
    ok, reason = is_valid(Path(sys.argv[1]))
    print(reason)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
