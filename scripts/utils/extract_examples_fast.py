#!/usr/bin/env python3
"""Fast extraction: stop as soon as we have 1 example per target type."""
import json
import sys


def strip_ts(text):
    idx = text.rfind('</ts>')
    if idx != -1:
        text = text[idx + len('</ts>'):].lstrip('. ;')
    return text.strip()


def extract(path, target_types, max_output_len=800):
    found = {}
    with open(path) as f:
        for line in f:
            if len(found) == len(target_types):
                break
            rec = json.loads(line)
            et = rec.get('eval_type', '')
            if et in target_types and et not in found:
                if len(rec['output']) <= max_output_len:
                    found[et] = rec
    return found


def print_examples(found, label):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    for et in sorted(found):
        rec = found[et]
        q = strip_ts(rec['input'])
        meta = rec.get('eval_metadata', {})
        sub = meta.get('sub_type', '')
        print(f"\n--- eval_type={et}" + (f"  sub_type={sub}" if sub else "") + " ---")
        print(f"Q: ...{q[-250:]}")
        print(f"A:\n{rec['output']}")
        print(f"META: {json.dumps({k:v for k,v in meta.items() if k != 'length'})}")


# Training atomic examples
train_types = {
    'stat_numerical', 'segment_trend_dominance', 'change_point',
    'local_enumeration', 'segment_enumeration',
    'duration_proportion',
}
train_found = extract('data/train_sft.jsonl', train_types)
print_examples(train_found, "TRAINING (ATOMIC) EXAMPLES")

# OOD test examples
ood_types = {
    'ood_conditional_stat', 'ood_nested_extrema', 'ood_event_density',
    'ood_conditional_count', 'ood_trend_reversal', 'ood_range_norm_amp',
    'ood_segment_stat_cmp',
}
ood_found = extract('data/test.jsonl', ood_types, max_output_len=1200)
print_examples(ood_found, "OOD (TEST-ONLY) EXAMPLES")

# Quick counts for OOD
print(f"\n{'='*80}")
print("  OOD COUNTS IN TEST SET")
print(f"{'='*80}")
counts = {}
with open('data/test.jsonl') as f:
    for line in f:
        et = json.loads(line).get('eval_type', '')
        if et.startswith('ood_'):
            counts[et] = counts.get(et, 0) + 1
for k in sorted(counts):
    print(f"  {k}: {counts[k]}")
print(f"  TOTAL OOD: {sum(counts.values())}")
