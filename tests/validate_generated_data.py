"""
Post-generation data quality validation.

Run AFTER generating data to verify structural integrity before training:

    python -m tests.validate_generated_data inspect_data/debug.jsonl

Checks:
  1. Self-consistency:  compute_score(gt, gt) >= 0.95 for every sample
  2. Think block structure: ===, answer:, data-only for lookups
  3. Format contradictions: no "X.XX > X.XX" where both sides equal
  4. Eval type coverage: all expected types present
  5. Verdict distribution: flag severe skew (>85% single verdict)
  6. Coherency parsing: returns 1.0 (not neutral 0.5) for applicable types
  7. Evidence quality: >= 0.70 for all non-description samples
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_ONLY_SUBTYPES = frozenset({
    'stat_min_val', 'stat_max_val', 'stat_std_val',
    'stat_median', 'period_estimate',
})

# Eval types that use minimal think blocks (no === separator required).
# RL taxonomy types let the model discover its own reasoning style.
# OOD types use minimal think for eval-only generation.
# Bridge types may use full decomposition but don't require ===.
_MINIMAL_THINK_PREFIXES = ('rl_', 'ood_', 'atomic_', 'bridge_')

# eval_types that SHOULD have coherency scoring (and which function)
COHERENCY_MAP = {
    'segment_judgment': 'condition_verdict',
    'cross_stat_judgment': 'condition_verdict',
    'anti_judgment': 'condition_verdict',
    'segment_trend_dominance': 'duration_dominance',
    'cross_trend_query': 'duration_dominance',
    'correlation': 'mts_verdict',
    'anticorrelation': 'mts_verdict',
    'clustering': 'mts_cluster',
    'anticlustering': 'mts_cluster',
    'yes_no': 'yes_no',
}

# stat_numerical sub_types that use categorical coherency
STAT_CATEGORICAL_SUBTYPES = frozenset({
    'stat_segment_compare',
})

SKEW_THRESHOLD = 0.85
EVIDENCE_FLOOR = 0.70
SELF_CONSISTENCY_FLOOR = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_think(output: str) -> str:
    m = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    return m.group(1) if m else ''


def _load_samples(path: str):
    samples = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            # NEVER read 'timeseries' — huge and useless
            samples.append({
                'output': rec['output'],
                'eval_type': rec['eval_type'],
                'eval_metadata': rec.get('eval_metadata', {}),
            })
    return samples


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_self_consistency(samples):
    """compute_score(gt, gt) >= threshold for every sample."""
    from reward import compute_score
    from reward.evidence import set_context_seq_len
    set_context_seq_len(256)

    failures = []
    for i, s in enumerate(samples):
        if s['eval_type'] == 'description':
            continue
        # OOD types return 0.5 by design (eval-only, never reward-scored)
        if s['eval_type'].startswith('ood_'):
            continue
        try:
            score = compute_score(
                s['output'], s['output'],
                eval_type=s['eval_type'],
                eval_metadata=s['eval_metadata'],
            )
            if score < SELF_CONSISTENCY_FLOOR:
                sub = s['eval_metadata'].get('sub_type', '')
                failures.append(
                    f"  [{i}] {s['eval_type']}/{sub} score={score:.4f}"
                )
        except Exception as e:
            failures.append(f"  [{i}] {s['eval_type']} ERROR: {e}")
    return failures


def check_think_block_structure(samples):
    """Verify ===, answer:, data-only format for lookups."""
    issues = []
    for i, s in enumerate(samples):
        et = s['eval_type']
        sub = s['eval_metadata'].get('sub_type', '')
        think = _extract_think(s['output'])
        if not think:
            if et != 'description':
                issues.append(f"  [{i}] {et}: missing <think> block")
            continue

        has_sep = '===' in think
        is_lookup = sub in DATA_ONLY_SUBTYPES
        is_desc = et == 'description'

        # Lookup types should NOT have ===
        if is_lookup and has_sep:
            issues.append(f"  [{i}] {et}/{sub}: lookup type has === (should be data-only)")

        # Non-lookup, non-description, non-taxonomy should HAVE ===
        # Taxonomy types (rl_*, ood_*, atomic_*, bridge_*) use free-form
        # think blocks — the model reasons however it wants.
        is_taxonomy = et.startswith(_MINIMAL_THINK_PREFIXES)
        if not is_lookup and not is_desc and not is_taxonomy and not has_sep:
            issues.append(f"  [{i}] {et}/{sub}: missing === separator")

        # Must have answer: (except description and taxonomy types)
        if not is_desc and not is_taxonomy and 'answer:' not in think.lower():
            issues.append(f"  [{i}] {et}/{sub}: missing 'answer:' line")
    return issues


def check_format_contradictions(samples):
    """No 'X.XX > X.XX' or 'X.XX < X.XX' where both sides are equal."""
    issues = []
    for i, s in enumerate(samples):
        think = _extract_think(s['output'])
        for m in re.finditer(r'(-?\d+\.\d{2})\s*([><])\s*(-?\d+\.\d{2})', think):
            left, op, right = m.group(1), m.group(2), m.group(3)
            if left == right:
                sub = s['eval_metadata'].get('sub_type', '')
                issues.append(
                    f"  [{i}] {s['eval_type']}/{sub}: '{left} {op} {right}'"
                )
    return issues


def check_eval_type_coverage(samples):
    """All expected eval_types should appear at least once."""
    from evaluation.eval.config import ALL_EVAL_TYPES
    expected = ALL_EVAL_TYPES
    found = {s['eval_type'] for s in samples}
    missing = expected - found
    if not missing:
        return []
    # With small datasets (<500), rare types may not appear — warn, don't fail
    n = len(samples)
    if n < 500 and missing:
        return [f"  WARNING (n={n}, may be too small): missing {sorted(missing)}"]
    return [f"  Missing eval_types: {sorted(missing)}"]


def check_verdict_distribution(samples):
    """Flag eval_types with severely skewed verdict distribution."""
    verdict_dist = defaultdict(Counter)
    for s in samples:
        verdict = s['eval_metadata'].get('verdict', '')
        if verdict:
            verdict_dist[s['eval_type']][str(verdict).lower()] += 1

    issues = []
    for et, counts in sorted(verdict_dist.items()):
        total = sum(counts.values())
        if total < 4:
            continue  # too few samples to judge
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count / total > SKEW_THRESHOLD:
            dominant = counts.most_common(1)[0][0]
            issues.append(
                f"  {et}: '{dominant}' = {most_common_count}/{total} "
                f"({most_common_count/total:.0%})"
            )
    return issues


def check_coherency_parsing(samples):
    """Coherency functions should return 1.0 (not neutral 0.5) on GT."""
    from reward.coherency import (
        coherency_condition_verdict, coherency_duration_dominance,
        coherency_mts_verdict, coherency_mts_cluster, coherency_yes_no,
        coherency_stats_categorical,
    )
    from reward.parsing import extract_think_content

    dispatchers = {
        'condition_verdict': coherency_condition_verdict,
        'duration_dominance': coherency_duration_dominance,
        'mts_verdict': coherency_mts_verdict,
        'mts_cluster': coherency_mts_cluster,
        'yes_no': coherency_yes_no,
    }

    issues = []
    for i, s in enumerate(samples):
        et = s['eval_type']
        sub = s['eval_metadata'].get('sub_type', '')
        think = extract_think_content(s['output'])
        if not think:
            continue

        # Direct coherency map
        coh_type = COHERENCY_MAP.get(et)
        # stat_numerical categorical sub_types
        if et == 'stat_numerical' and sub in STAT_CATEGORICAL_SUBTYPES:
            coh_type = 'stats_categorical'

        if coh_type is None:
            continue

        if coh_type == 'stats_categorical':
            score = coherency_stats_categorical(think)
        else:
            fn = dispatchers[coh_type]
            score = fn(think)

        if score == 0.5:
            issues.append(
                f"  [{i}] {et}/{sub}: coherency_{coh_type} returned neutral 0.5"
            )
    return issues


def check_evidence_quality(samples):
    """Evidence quality should be >= threshold for all non-description GT."""
    from reward.evidence import score_evidence_quality, set_context_seq_len
    from reward.parsing import extract_think_content
    set_context_seq_len(256)

    issues = []
    for i, s in enumerate(samples):
        if s['eval_type'] == 'description':
            continue
        think = extract_think_content(s['output'])
        if not think:
            continue
        score = score_evidence_quality(think, think)
        if score < EVIDENCE_FLOOR:
            sub = s['eval_metadata'].get('sub_type', '')
            issues.append(
                f"  [{i}] {s['eval_type']}/{sub}: evidence={score:.3f}"
            )
    return issues


def check_verdict_metadata_match(samples):
    """answer: in think block must match eval_metadata verdict."""
    issues = []
    for i, s in enumerate(samples):
        verdict = str(s['eval_metadata'].get('verdict', ''))
        if not verdict:
            continue
        think = _extract_think(s['output'])
        answer_match = re.search(r'answer:\s*(.+)', think, re.IGNORECASE)
        if not answer_match:
            continue
        answer_val = answer_match.group(1).strip()
        if answer_val == verdict:
            continue
        # Allow minor numeric diffs
        try:
            if abs(float(answer_val) - float(verdict)) <= 0.01:
                continue
        except ValueError:
            pass
        if answer_val.lower() == verdict.lower():
            continue
        sub = s['eval_metadata'].get('sub_type', '')
        issues.append(
            f"  [{i}] {s['eval_type']}/{sub}: answer='{answer_val}' != verdict='{verdict}'"
        )
    return issues


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CHECKS = [
    ("Self-consistency (score >= {:.2f})".format(SELF_CONSISTENCY_FLOOR),
     check_self_consistency),
    ("Think block structure", check_think_block_structure),
    ("Format contradictions", check_format_contradictions),
    ("Eval type coverage", check_eval_type_coverage),
    ("Verdict distribution (skew > {:.0%})".format(SKEW_THRESHOLD),
     check_verdict_distribution),
    ("Coherency parsing (no neutrals on GT)", check_coherency_parsing),
    ("Evidence quality (>= {:.2f})".format(EVIDENCE_FLOOR),
     check_evidence_quality),
    ("Verdict-metadata consistency", check_verdict_metadata_match),
]


def main():
    parser = argparse.ArgumentParser(
        description="Validate generated data quality before training."
    )
    parser.add_argument("jsonl_path", help="Path to debug.jsonl")
    args = parser.parse_args()

    path = Path(args.jsonl_path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Loading {path}...")
    samples = _load_samples(str(path))
    print(f"Loaded {len(samples)} samples\n")

    # Eval type distribution
    type_counts = Counter(s['eval_type'] for s in samples)
    print("Eval type distribution:")
    for et, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {et:40s} {count:4d}")
    print()

    # Run checks
    all_passed = True
    for name, check_fn in CHECKS:
        print(f"Checking: {name}...", end=" ", flush=True)
        issues = check_fn(samples)
        if issues:
            # WARNING lines don't count as failures
            real_issues = [x for x in issues if 'WARNING' not in x]
            warnings = [x for x in issues if 'WARNING' in x]
            if real_issues:
                all_passed = False
                print(f"FAIL ({len(real_issues)} issues)")
                for issue in real_issues[:15]:
                    print(issue)
                if len(real_issues) > 15:
                    print(f"  ... and {len(real_issues) - 15} more")
            elif warnings:
                print(f"WARN ({len(warnings)})")
                for w in warnings:
                    print(w)
            else:
                print("PASS")
        else:
            print("PASS")
        print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED — data is ready for training.")
        sys.exit(0)
    else:
        print("SOME CHECKS FAILED — review issues above before training.")
        sys.exit(1)


if __name__ == '__main__':
    main()
