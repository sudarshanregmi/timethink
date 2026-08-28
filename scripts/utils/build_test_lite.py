#!/usr/bin/env python3
"""Build a stratified ~5K subset of test.jsonl for fast evaluation.

Stratifies by (eval_type, sub_type, verdict_class) so every bucket that
exists in test.jsonl shows up in the lite subset with at least 1 sample.
Remaining quota is distributed proportionally to bucket size.

Default: data/test.jsonl (15,357 rows) -> data/test_lite.jsonl (5,000 rows).

Usage:
    python scripts/utils/build_test_lite.py
    python scripts/utils/build_test_lite.py --target 5000 --seed 42
"""

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


_SUB_TYPE_KEYS = ("sub_type", "subtype", "question_type", "variant", "stat_type", "bridge_type")


def _sub_type(md: dict) -> str:
    for k in _SUB_TYPE_KEYS:
        if k in md:
            return str(md[k])
    return "-"


_ANSWER_RE = re.compile(r"Answer:\s*(\S+)", re.IGNORECASE)


def _verdict_class(output_text: str) -> str:
    m = _ANSWER_RE.search(output_text or "")
    v = m.group(1).rstrip(".").lower() if m else "-"
    if v in {"yes", "no", "true", "false"}:
        return v
    try:
        float(v)
        return "num"
    except ValueError:
        return "str"


def _bucket_key(sample: dict) -> tuple[str, str, str]:
    md = sample.get("eval_metadata", {}) or {}
    return (
        sample.get("eval_type", "-"),
        _sub_type(md),
        _verdict_class(sample.get("output", "")),
    )


def stratified_sample(rows: list[dict], target: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        buckets[_bucket_key(row)].append(idx)

    for idxs in buckets.values():
        rng.shuffle(idxs)

    if target >= len(rows):
        return list(rows)

    n_buckets = len(buckets)
    if target < n_buckets:
        raise ValueError(
            f"target={target} < n_buckets={n_buckets}; cannot guarantee >=1 per bucket"
        )

    total = len(rows)
    quotas: dict[tuple, int] = {}
    # Proportional allocation with floor of 1 per bucket.
    remaining_target = target - n_buckets
    for k, idxs in buckets.items():
        quotas[k] = 1  # guaranteed floor
    # Distribute the remaining_target proportionally to (bucket_size - 1),
    # so large buckets get more and already-1-sample buckets stay at 1.
    weights = {k: max(0, len(v) - 1) for k, v in buckets.items()}
    wsum = sum(weights.values())
    if wsum > 0 and remaining_target > 0:
        allocated = 0
        leftovers = []
        for k in buckets:
            share = remaining_target * weights[k] / wsum
            add = math.floor(share)
            # Cap by available samples in bucket.
            add = min(add, len(buckets[k]) - 1)
            quotas[k] += add
            allocated += add
            leftovers.append((share - math.floor(share), k))
        # Distribute remainder by largest fractional part, respecting caps.
        deficit = remaining_target - allocated
        leftovers.sort(reverse=True)
        for _, k in leftovers:
            if deficit <= 0:
                break
            if quotas[k] < len(buckets[k]):
                quotas[k] += 1
                deficit -= 1
        # If still short (all buckets maxed), distribute anywhere with room.
        if deficit > 0:
            for k, idxs in buckets.items():
                while deficit > 0 and quotas[k] < len(idxs):
                    quotas[k] += 1
                    deficit -= 1
                if deficit <= 0:
                    break

    picked: list[int] = []
    for k, idxs in buckets.items():
        picked.extend(idxs[: quotas[k]])

    rng.shuffle(picked)
    return [rows[i] for i in picked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/test.jsonl")
    ap.add_argument("--output", default="data/test_lite.jsonl")
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")

    rows = [json.loads(line) for line in src.open()]
    print(f"Loaded {len(rows)} rows from {src}")

    sampled = stratified_sample(rows, args.target, args.seed)
    print(f"Sampled {len(sampled)} rows")

    # Sanity: bucket coverage
    src_buckets = {_bucket_key(r) for r in rows}
    dst_buckets = {_bucket_key(r) for r in sampled}
    missing = src_buckets - dst_buckets
    print(f"Buckets in source: {len(src_buckets)} | in sample: {len(dst_buckets)} | missing: {len(missing)}")
    if missing:
        for k in list(missing)[:5]:
            print(f"  missing: {k}")

    with dst.open("w") as f:
        for r in sampled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
