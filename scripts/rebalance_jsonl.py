"""Rebalance binary verdict types in existing JSONL files.

Mirrors `balance_binary_results()` from synth/align/dataset.py but operates
on JSONL dicts rather than SampleResult objects. Used to rebalance files
that were written outside the main pipeline (e.g., via extend_lengths.py)
where source-level balance wasn't applied.

For each binary eval_type, downsamples the majority verdict to match the
minority count — producing a hard 50/50 distribution. Non-binary types
and types with only one verdict present pass through unchanged.

Usage:
    python scripts/rebalance_jsonl.py path/to/data.jsonl
    # In-place rewrite; original is copied to path/to/data.jsonl.preblance
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from synth.align.dataset import _BINARY_VERDICT_TYPES  # noqa: E402


def _get_verdict(row: dict) -> str:
    """Return normalized verdict string or '' if not a known binary verdict row."""
    et = row.get('eval_type', '')
    md = row.get('eval_metadata') or {}
    verdict = str(md.get('verdict', '')).lower() if isinstance(md, dict) else ''
    if not verdict and et == 'yes_no':
        output = row.get('output', '')
        m = re.search(r'answer:\s*(yes|no)', output, re.IGNORECASE)
        verdict = m.group(1).lower() if m else ''
    # Normalize native verdicts to yes/no for binary balancing
    # (must match synth/align/dataset.py _get_result_verdict)
    if et == 'correlation':
        if verdict == 'similar': verdict = 'yes'
        elif verdict == 'different': verdict = 'no'
    elif et == 'anticorrelation':
        if verdict == 'opposite': verdict = 'yes'
        elif verdict == 'not_opposite': verdict = 'no'
    return verdict


def rebalance_file(path: Path, seed: int = 0xC0FFEE, backup: bool = True,
                   verbose: bool = True) -> tuple[int, int]:
    """Rebalance binary verdict types in the JSONL at `path` in place.

    Returns (n_input, n_output) counts.
    """
    random.seed(seed)
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    n_in = len(rows)

    # Bucket: (eval_type, verdict) → list of rows
    buckets: dict = defaultdict(list)
    non_binary = []
    for r in rows:
        et = r.get('eval_type', '')
        if et in _BINARY_VERDICT_TYPES:
            v = _get_verdict(r)
            if v:
                buckets[(et, v)].append(r)
            else:
                non_binary.append(r)
        else:
            non_binary.append(r)

    out = list(non_binary)
    eval_types_seen = sorted({et for et, _ in buckets})
    total_dropped = 0
    changes = []
    for et in eval_types_seen:
        verdicts = {v: buckets[(et, v)] for (e, v) in buckets if e == et}
        if len(verdicts) < 2:
            for v_list in verdicts.values():
                out.extend(v_list)
            if verbose and verdicts:
                only = next(iter(verdicts))
                changes.append(
                    f"  {et:<34} single-verdict ({only!r})  passthrough n={len(verdicts[only])}"
                )
            continue
        counts = {v: len(items) for v, items in verdicts.items()}
        min_count = min(counts.values())
        for v, items in verdicts.items():
            if len(items) > min_count:
                random.shuffle(items)
                out.extend(items[:min_count])
                dropped = len(items) - min_count
                total_dropped += dropped
                changes.append(
                    f"  {et:<34} {v:<6} {len(items)} -> {min_count} (dropped {dropped})"
                )
            else:
                out.extend(items)
    random.shuffle(out)

    if backup:
        bak = path.with_suffix(path.suffix + '.preblance')
        if not bak.exists():
            shutil.copy2(path, bak)

    with open(path, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    if verbose:
        print(f"[rebalance] {path}: {n_in} -> {len(out)} "
              f"(dropped {total_dropped} majority)")
        for line in changes:
            print(line)
    return n_in, len(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('path', type=Path, help='JSONL file to rebalance in-place')
    p.add_argument('--seed', type=int, default=0xC0FFEE)
    p.add_argument('--no-backup', action='store_true',
                   help='Skip writing .preblance backup')
    args = p.parse_args()
    rebalance_file(args.path, seed=args.seed, backup=not args.no_backup)


if __name__ == '__main__':
    main()
