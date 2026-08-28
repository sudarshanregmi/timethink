"""Build a balanced eval_type subset of train_rl_public.parquet.

Each eval_type contributes as close to target_size / num_types samples as
possible. Types with fewer than the per-type quota contribute all of their
rows; the remaining budget is redistributed across the remaining types
(water-filling). Total row count is exactly --target-size.

Examples:
    python scripts/utils/build_balanced_rl_subset.py
    python scripts/utils/build_balanced_rl_subset.py --target-size 20000 --seed 0
    python scripts/utils/build_balanced_rl_subset.py --in data/train_rl_public.parquet \\
        --out data/train_rl_public_balanced_10k.parquet
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_eval_type(extra_info):
    if isinstance(extra_info, dict):
        return extra_info.get("eval_type")
    if isinstance(extra_info, str):
        try:
            return json.loads(extra_info).get("eval_type")
        except (ValueError, TypeError):
            return None
    return None


def waterfill_allocation(counts: dict[str, int], target: int) -> dict[str, int]:
    """Allocate `target` slots across keys, never exceeding each key's count.

    Process keys from smallest to largest available count. At each key,
    the per-type share is `remaining_budget // remaining_types`. If the
    key has fewer rows than its share, take all of them and let the
    deficit be redistributed across the (larger) remaining types.

    Any leftover slots after the pass are sprinkled onto types with spare
    capacity, one row at a time, until the budget is exhausted or every
    type is at its cap.
    """
    types = sorted(counts.items(), key=lambda x: x[1])
    assigned = {t: 0 for t, _ in types}
    remaining = target
    n_left = len(types)
    for t, c in types:
        share = remaining // n_left if n_left else 0
        a = min(c, share)
        assigned[t] = a
        remaining -= a
        n_left -= 1

    while remaining > 0:
        progressed = False
        for t, c in types:
            if assigned[t] < c:
                assigned[t] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return assigned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="inp",
        default="data/train_rl_public.parquet",
        help="Input parquet path.",
    )
    ap.add_argument(
        "--out",
        dest="out",
        default=None,
        help="Output parquet path. Default: data/train_rl_public_balanced_<target>.parquet",
    )
    ap.add_argument("--target-size", type=int, default=10000, help="Total rows in subset.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for per-type sampling.")
    ap.add_argument(
        "--report",
        default=None,
        help="Output JSON report path. Default: <out>.report.json",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    inp = (repo_root / args.inp).resolve() if not Path(args.inp).is_absolute() else Path(args.inp)
    if args.out is None:
        out = repo_root / f"data/train_rl_public_balanced_{args.target_size}.parquet"
    else:
        out = Path(args.out)
        if not out.is_absolute():
            out = repo_root / args.out
    report_path = Path(args.report) if args.report else out.with_suffix(".report.json")

    print(f"[load] reading {inp}")
    df = pd.read_parquet(inp)
    print(f"[load] {len(df):,} rows, {len(df.columns)} columns")

    print("[parse] extracting eval_type from extra_info")
    eval_types = df["extra_info"].map(parse_eval_type)
    if eval_types.isna().any():
        n_missing = int(eval_types.isna().sum())
        raise SystemExit(f"error: {n_missing} rows have missing/unparseable eval_type")

    counts = Counter(eval_types)
    print(f"[parse] {len(counts)} distinct eval_types, total {sum(counts.values()):,}")

    if args.target_size > sum(counts.values()):
        raise SystemExit(
            f"error: target {args.target_size} > available {sum(counts.values())}"
        )

    quotas = waterfill_allocation(dict(counts), args.target_size)
    assert sum(quotas.values()) == args.target_size, (
        f"allocation sum {sum(quotas.values())} != target {args.target_size}"
    )

    rng = np.random.default_rng(args.seed)
    by_type: dict[str, list[int]] = defaultdict(list)
    for pos, et in enumerate(eval_types):
        by_type[et].append(pos)

    selected_positions: list[int] = []
    for et, q in quotas.items():
        idxs = np.asarray(by_type[et], dtype=np.int64)
        if q >= len(idxs):
            picked = idxs
        else:
            picked = rng.choice(idxs, size=q, replace=False)
        selected_positions.extend(int(p) for p in picked)

    selected_positions.sort()
    subset = df.iloc[selected_positions].reset_index(drop=True)

    print(f"[write] writing {len(subset):,} rows -> {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(out, index=False)

    sel_eval_types = subset["extra_info"].map(parse_eval_type)
    sel_counts = dict(Counter(sel_eval_types))
    report = {
        "input_parquet": str(inp),
        "output_parquet": str(out),
        "target_size": args.target_size,
        "actual_size": int(len(subset)),
        "seed": args.seed,
        "num_eval_types": len(counts),
        "per_type_input_counts": dict(sorted(counts.items(), key=lambda x: -x[1])),
        "per_type_quota": dict(sorted(quotas.items(), key=lambda x: x[0])),
        "per_type_actual": dict(sorted(sel_counts.items(), key=lambda x: x[0])),
        "min_per_type": int(min(sel_counts.values())),
        "max_per_type": int(max(sel_counts.values())),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[write] report -> {report_path}")

    quota_hist = Counter(quotas.values())
    print("\nallocation summary:")
    for q, n in sorted(quota_hist.items()):
        print(f"  {q:>5d} samples × {n} eval_type(s)")
    print(f"  min={report['min_per_type']}  max={report['max_per_type']}  total={report['actual_size']}")


if __name__ == "__main__":
    main()
