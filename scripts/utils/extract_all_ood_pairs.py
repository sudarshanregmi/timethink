#!/usr/bin/env python3
"""Extract ALL atomic training examples needed for OOD composition,
plus all 7 OOD test examples. Grouped by OOD type."""
import json
from collections import defaultdict

def strip_ts(text):
    idx = text.rfind('</ts>')
    if idx != -1:
        text = text[idx + len('</ts>'):].lstrip('. ;')
    return text.strip()

# What atomic skills each OOD type needs (eval_type -> preferred sub_types)
OOD_COMPOSITION = {
    'ood_conditional_stat': {
        'skills': [
            ('stat_numerical', ['stat_mean_val']),
            ('segment_enumeration', ['segment_count_by_type']),
        ],
        'desc': 'Filter data points by trend type, THEN compute statistic on filtered subset',
    },
    'ood_nested_extrema': {
        'skills': [
            ('segment_enumeration', ['longest_segment']),
            ('local_enumeration', ['max_amplitude']),
        ],
        'desc': 'Find longest segment, THEN find max event amplitude WITHIN that segment',
    },
    'ood_event_density': {
        'skills': [
            ('local_enumeration', ['count_all', 'count_by_amplitude']),
            ('duration_proportion', ['longest_total_duration', 'duration_by_type']),
        ],
        'desc': 'Count events per trend type, compute duration per trend type, divide for density, compare',
    },
    'ood_conditional_count': {
        'skills': [
            ('segment_enumeration', ['segment_count_by_type']),
            ('stat_numerical', ['stat_mean_val']),
        ],
        'desc': 'Count segments of a given type whose mean exceeds the global mean',
    },
    'ood_trend_reversal': {
        'skills': [
            ('change_point', ['largest_level_shift', 'change_point_positions']),
            ('segment_trend_dominance', []),
        ],
        'desc': 'Find largest change point, THEN check if dominant trend reverses across it',
    },
    'ood_range_normalized_amplitude': {
        'skills': [
            ('local_enumeration', ['highest_amplitude']),
            ('stat_numerical', ['stat_range']),
        ],
        'desc': 'Get max event amplitude AND total value range, THEN compare (amplitude > range/2?)',
    },
    'ood_segment_stat_compare': {
        'skills': [
            ('segment_enumeration', ['segment_count_all', 'longest_segment']),
            ('stat_numerical', ['stat_mean_val']),
        ],
        'desc': 'Identify first/last segment, compute mean of each, THEN compare',
    },
}

# Collect all needed atomic eval_types
needed_types = set()
for ood_info in OOD_COMPOSITION.values():
    for et, _ in ood_info['skills']:
        needed_types.add(et)

# Extract training examples — one per (eval_type, sub_type), prefer short
train_examples = {}  # (eval_type, sub_type) -> rec
with open('data/train_sft.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        et = rec.get('eval_type', '')
        if et not in needed_types:
            continue
        sub = rec.get('eval_metadata', {}).get('sub_type', '')
        key = (et, sub)
        if key not in train_examples and len(rec['output']) < 900:
            train_examples[key] = rec

# Extract OOD examples
ood_names = set(OOD_COMPOSITION.keys())
# Also handle actual names in data
ood_name_map = {
    'ood_range_normalized_amplitude': 'ood_range_normalized_amplitude',
    'ood_segment_stat_compare': 'ood_segment_stat_compare',
}
ood_examples = {}
with open('data/test.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        et = rec.get('eval_type', '')
        if et.startswith('ood_') and et not in ood_examples:
            ood_examples[et] = rec

# Print everything grouped by OOD type
for ood_type, info in OOD_COMPOSITION.items():
    print(f"\n{'#'*90}")
    print(f"# OOD TYPE: {ood_type}")
    print(f"# Composition: {info['desc']}")
    print(f"{'#'*90}")

    # Show each atomic skill
    for skill_et, preferred_subs in info['skills']:
        print(f"\n  ---- ATOMIC SKILL: {skill_et} ----")
        # Find best match
        found = False
        for sub in preferred_subs:
            key = (skill_et, sub)
            if key in train_examples:
                rec = train_examples[key]
                q = strip_ts(rec['input'])
                meta = rec.get('eval_metadata', {})
                print(f"  sub_type: {meta.get('sub_type', '(none)')}")
                print(f"  Q: ...{q[-220:]}")
                print(f"  A:")
                for aline in rec['output'].split('\n'):
                    print(f"    {aline}")
                found = True
                break
        if not found:
            # Try any sub_type for this eval_type
            for key, rec in train_examples.items():
                if key[0] == skill_et:
                    q = strip_ts(rec['input'])
                    meta = rec.get('eval_metadata', {})
                    print(f"  sub_type: {meta.get('sub_type', '(none)')}")
                    print(f"  Q: ...{q[-220:]}")
                    print(f"  A:")
                    for aline in rec['output'].split('\n'):
                        print(f"    {aline}")
                    found = True
                    break
        if not found:
            print(f"  !! NO EXAMPLE FOUND for {skill_et}")

    # Show OOD question
    print(f"\n  ---- OOD QUESTION: {ood_type} ----")
    # Try exact match or close match
    ood_rec = ood_examples.get(ood_type)
    if not ood_rec:
        # Try fuzzy
        for k, v in ood_examples.items():
            if ood_type.replace('_', '') in k.replace('_', ''):
                ood_rec = v
                break
    if ood_rec:
        q = strip_ts(ood_rec['input'])
        meta = ood_rec.get('eval_metadata', {})
        print(f"  eval_type in data: {ood_rec['eval_type']}")
        print(f"  sub_type: {meta.get('sub_type', '(none)')}")
        print(f"  Q: ...{q[-280:]}")
        print(f"  A:")
        for aline in ood_rec['output'].split('\n'):
            print(f"    {aline}")
    else:
        print(f"  !! NO OOD EXAMPLE FOUND")

# Also list all available training sub_types we found
print(f"\n\n{'='*90}")
print("ALL ATOMIC TRAINING EXAMPLES FOUND:")
print(f"{'='*90}")
for (et, sub), rec in sorted(train_examples.items()):
    print(f"  {et}/{sub}  (output len={len(rec['output'])})")

print(f"\n{'='*90}")
print("ALL OOD TYPES IN TEST DATA:")
print(f"{'='*90}")
for et in sorted(ood_examples):
    print(f"  {et}")
