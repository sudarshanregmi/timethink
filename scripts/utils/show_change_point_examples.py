#!/usr/bin/env python3
"""Extract and display exact change_point examples from a JSONL data file.

Usage:
    python scripts/utils/show_change_point_examples.py [path_to_jsonl] [--max N]

Defaults to data/train_sft.jsonl, shows 1 example per sub_type.
"""
import json
import sys
from collections import defaultdict


def main():
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'data/train_sft.jsonl'
    max_per = 1
    if '--max' in sys.argv:
        max_per = int(sys.argv[sys.argv.index('--max') + 1])

    examples = defaultdict(list)  # sub_type -> list of records

    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('eval_type') != 'change_point':
                continue
            sub = rec.get('eval_metadata', {}).get('sub_type', 'unknown')
            if len(examples[sub]) < max_per:
                examples[sub].append(rec)

    if not examples:
        print(f"No change_point examples found in {path}")
        return

    for sub_type in sorted(examples):
        for i, rec in enumerate(examples[sub_type]):
            print(f"{'='*80}")
            print(f"SUB_TYPE: {sub_type}  (example {i+1})")
            print(f"{'='*80}")
            print(f"\n--- QUESTION ---")
            # Strip timeseries tokens from input for readability
            inp = rec['input']
            ts_end = inp.rfind('</ts>')
            if ts_end != -1:
                inp = inp[ts_end + len('</ts>'):].lstrip('. ;')
            print(inp)
            print(f"\n--- ANSWER (full output) ---")
            print(rec['output'])
            print(f"\n--- EVAL METADATA ---")
            meta = {k: v for k, v in rec.get('eval_metadata', {}).items()}
            print(json.dumps(meta, indent=2))
            print()


if __name__ == '__main__':
    main()
