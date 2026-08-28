#!/usr/bin/env python3
"""Detect reward hacking in RL evaluation results.

Checks for majority-class exploitation, verdict distribution shift,
accuracy-reasoning divergence, and eval/reward misalignment signals.

Usage:
    # Compare SFT baseline (post-SFT stage) vs RL model (post-RL stage)
    python scripts/utils/check_reward_hacking.py exp/sft exp/rl

    # Check a single model for red flags
    python scripts/utils/check_reward_hacking.py exp/rl
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Binary verdict eval_types to check.  Update when adding new binary tasks.
# Matches preprocess.py BINARY_VERDICT_TYPES exactly.
BINARY_VERDICT_TYPES = {
    'yes_no',
    'anti_judgment', 'segment_judgment', 'cross_stat_judgment',
    # OOD binary verdict types (eval-only)
    'ood_trend_reversal', 'ood_range_normalized_amplitude', 'ood_segment_stat_compare',
    'ood_max_before_min', 'ood_std_exceeds_half_range',
    'ood_max_in_highest_mean_quarter', 'ood_quarter_mean_ordering',
    'ood_symmetric_recovery', 'ood_cycle_mean_trend', 'ood_event_amplitude_vs_std',
    'ood_amplitude_vs_segment_std', 'ood_symmetric_trend_sequence', 'ood_trend_follows_mean',
    # RL single-metric binary types
    'rl_amplitude_vs_range', 'rl_amplitude_vs_std', 'rl_condition_recovery',
    'rl_event_in_trend_type', 'rl_event_near_extremum', 'rl_extrema_same_half',
    'rl_half_mean_compare', 'rl_has_periodicity', 'rl_interval_comparison',
    'rl_max_in_first_half', 'rl_max_in_trend_type', 'rl_mean_shift',
    'rl_mean_stability', 'rl_median_mean_close', 'rl_monotonic_chunks',
    'rl_segment_mean_compare', 'rl_volatility_change',
    # Cross-metric taxonomy binary types
    'atomic_cross_stat_compare', 'atomic_cross_trend_align',
    'rl_cross_stat_ratio', 'rl_cross_event_sync', 'rl_cross_period_compare',
    'rl_cross_attribute_corr',
    'rl_corr_conditional',
    'ood_cross_extrema_alignment', 'ood_cross_range_overlap',
    'ood_cluster_singleton', 'ood_corr_transitivity',
    'ood_cross_trend_convergence',
    # OOD compositional probes (eval-only) — added 2026-04-24
    'ood_peak_in_longest_segment', 'ood_cross_event_causality',
}

# Thresholds for flagging
CLASS_COLLAPSE_THRESHOLD = 0.90   # One class >90% accuracy = likely collapse
CLASS_NEGLECT_THRESHOLD = 0.15    # Other class <15% accuracy = model ignoring it
IMBALANCE_WARNING = 0.60          # GT class >60% = imbalanced
REASONING_DIVERGENCE = 0.03       # Reasoning up but accuracy down by >3pp
DISTRIBUTION_SHIFT = 0.10         # Prediction distribution shifted by >10pp


def load_eval(exp_dir: Path) -> pd.DataFrame:
    csv_path = exp_dir / "evaluation_report_optimized.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(csv_path)


def _get_verdict(df):
    """Normalize verdict column to string."""
    return df['gt_verdict'].fillna('').astype(str)


def check_single(df, name="Model"):
    """Check a single model's results for reward hacking signals."""
    warnings = []
    infos = []

    for task in sorted(BINARY_VERDICT_TYPES):
        sub = df[df['task_type'] == task]
        if len(sub) < 5:
            continue

        verdicts = _get_verdict(sub)
        unique_verdicts = sorted(verdicts.unique())
        if len(unique_verdicts) != 2:
            continue

        v_pos, v_neg = unique_verdicts[1], unique_verdicts[0]
        pos = sub[verdicts == v_pos]
        neg = sub[verdicts == v_neg]
        gt_pos_frac = len(pos) / len(sub)

        pos_acc = pos['binary_accuracy'].mean() if len(pos) > 0 else 0
        neg_acc = neg['binary_accuracy'].mean() if len(neg) > 0 else 0
        overall_acc = sub['binary_accuracy'].mean()

        # Check 1: Class collapse
        if (pos_acc > CLASS_COLLAPSE_THRESHOLD and neg_acc < CLASS_NEGLECT_THRESHOLD):
            warnings.append(
                f"  COLLAPSE  {task}: always predicts '{v_pos}' "
                f"(acc: {v_pos}={pos_acc:.0%}, {v_neg}={neg_acc:.0%}, n={len(sub)})"
            )
        elif (neg_acc > CLASS_COLLAPSE_THRESHOLD and pos_acc < CLASS_NEGLECT_THRESHOLD):
            warnings.append(
                f"  COLLAPSE  {task}: always predicts '{v_neg}' "
                f"(acc: {v_pos}={pos_acc:.0%}, {v_neg}={neg_acc:.0%}, n={len(sub)})"
            )
        # Check 2: Significant per-class accuracy gap (not full collapse but skewed)
        elif abs(pos_acc - neg_acc) > 0.30:
            warnings.append(
                f"  SKEWED    {task}: large per-class gap "
                f"(acc: {v_pos}={pos_acc:.2f}, {v_neg}={neg_acc:.2f}, n={len(sub)})"
            )

        # Check 3: GT class imbalance
        if gt_pos_frac > IMBALANCE_WARNING or gt_pos_frac < (1 - IMBALANCE_WARNING):
            majority = v_pos if gt_pos_frac > 0.5 else v_neg
            majority_frac = max(gt_pos_frac, 1 - gt_pos_frac)
            infos.append(
                f"  IMBALANCE {task}: GT '{majority}' = {majority_frac:.0%} (n={len(sub)})"
            )

        # Check 4: Prediction distribution (if pred_verdict available)
        if sub['pred_verdict'].notna().sum() > len(sub) * 0.5:
            pred_pos_frac = (sub['pred_verdict'].astype(str) == v_pos).mean()
            if abs(pred_pos_frac - gt_pos_frac) > DISTRIBUTION_SHIFT:
                direction = "over" if pred_pos_frac > gt_pos_frac else "under"
                warnings.append(
                    f"  PRED_SHIFT {task}: {direction}-predicts '{v_pos}' "
                    f"(GT={gt_pos_frac:.0%}, Pred={pred_pos_frac:.0%})"
                )

    return warnings, infos


def check_comparison(df_base, df_rl, name_base="SFT", name_rl="RL"):
    """Compare two models for reward hacking signals."""
    warnings = []

    for task in sorted(BINARY_VERDICT_TYPES):
        sub_b = df_base[df_base['task_type'] == task]
        sub_r = df_rl[df_rl['task_type'] == task]
        if len(sub_b) < 5 or len(sub_r) < 5:
            continue

        # Check: Accuracy-reasoning divergence
        b_acc = sub_b['binary_accuracy'].mean()
        r_acc = sub_r['binary_accuracy'].mean()
        b_rea = sub_b['reasoning_score'].mean()
        r_rea = sub_r['reasoning_score'].mean()

        if (r_rea - b_rea > REASONING_DIVERGENCE and r_acc < b_acc - REASONING_DIVERGENCE):
            warnings.append(
                f"  DIVERGE   {task}: reasoning up ({b_rea:.2f}→{r_rea:.2f}) "
                f"but accuracy down ({b_acc:.2f}→{r_acc:.2f}) — "
                f"model may explain wrong answers convincingly"
            )

        # Check: Per-class accuracy shift
        verdicts_b = _get_verdict(sub_b)
        verdicts_r = _get_verdict(sub_r)
        unique = sorted(verdicts_b.unique())
        if len(unique) != 2:
            continue

        v_pos, v_neg = unique[1], unique[0]

        b_pos_acc = sub_b[verdicts_b == v_pos]['binary_accuracy'].mean()
        r_pos_acc = sub_r[verdicts_r == v_pos]['binary_accuracy'].mean()
        b_neg_acc = sub_b[verdicts_b == v_neg]['binary_accuracy'].mean()
        r_neg_acc = sub_r[verdicts_r == v_neg]['binary_accuracy'].mean()

        # One class improved significantly while the other crashed
        if (r_neg_acc - b_neg_acc > 0.10 and b_pos_acc - r_pos_acc > 0.10):
            warnings.append(
                f"  TRADEOFF  {task}: '{v_neg}' improved ({b_neg_acc:.2f}→{r_neg_acc:.2f}) "
                f"but '{v_pos}' crashed ({b_pos_acc:.2f}→{r_pos_acc:.2f}) — "
                f"majority-class exploitation"
            )
        elif (r_pos_acc - b_pos_acc > 0.10 and b_neg_acc - r_neg_acc > 0.10):
            warnings.append(
                f"  TRADEOFF  {task}: '{v_pos}' improved ({b_pos_acc:.2f}→{r_pos_acc:.2f}) "
                f"but '{v_neg}' crashed ({b_neg_acc:.2f}→{r_neg_acc:.2f}) — "
                f"majority-class exploitation"
            )

    return warnings


def check_numeric_tasks(df_base, df_rl, name_base="SFT", name_rl="RL"):
    """Check numeric verdict tasks for near-zero accuracy."""
    warnings = []
    numeric_types = {'temporal_position', 'stat_numerical', 'duration_proportion'}

    for task in sorted(numeric_types):
        sub_b = df_base[df_base['task_type'] == task]
        sub_r = df_rl[df_rl['task_type'] == task]
        if len(sub_b) < 5:
            continue

        b_acc = sub_b['binary_accuracy'].mean()
        r_acc = sub_r['binary_accuracy'].mean() if len(sub_r) > 0 else 0

        if b_acc < 0.05 and r_acc < 0.05:
            warnings.append(
                f"  ZERO_ACC  {task}: both models ~0% accuracy "
                f"({name_base}={b_acc:.2f}, {name_rl}={r_acc:.2f}, n={len(sub_b)}) — "
                f"task may be too hard or eval metric too strict"
            )

    return warnings


def check_generation_balance(jsonl_path):
    """Check a generated JSONL file for distribution issues BEFORE training.

    This catches problems at the source — before they become reward hacking.
    """
    import json
    from collections import Counter, defaultdict

    print(f"\n--- Generation Balance Check: {jsonl_path} ---")
    warnings = []

    type_counts = Counter()
    verdict_counts = defaultdict(Counter)
    total = 0

    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            et = rec.get('eval_type', 'unknown')
            meta = rec.get('eval_metadata', {})
            verdict = str(meta.get('verdict', ''))

            type_counts[et] += 1
            if verdict:
                verdict_counts[et][verdict] += 1
            total += 1

    if total == 0:
        print("  ERROR: empty file")
        return ["  ERROR: empty JSONL file"]

    # 1. Eval type distribution
    print(f"\n  Total samples: {total}")
    print(f"\n  Eval type distribution:")
    for et, cnt in type_counts.most_common():
        pct = cnt / total * 100
        flag = ""
        if pct > 25:
            flag = " ⚠ DOMINANT"
            warnings.append(f"  TYPE_DOMINANT {et}: {pct:.1f}% of data — will drown out rare types in RL")
        elif cnt < 20:
            flag = " ⚠ RARE"
            warnings.append(f"  TYPE_RARE    {et}: only {cnt} samples — model barely sees this during RL")
        print(f"    {et:35s} {cnt:6d} ({pct:5.1f}%){flag}")

    # 2. Binary verdict balance
    print(f"\n  Binary verdict balance:")
    for et in sorted(BINARY_VERDICT_TYPES):
        vc = verdict_counts.get(et, {})
        if len(vc) != 2:
            continue
        total_et = sum(vc.values())
        majority_frac = max(vc.values()) / total_et
        majority_label = max(vc, key=vc.get)
        if majority_frac > 0.60:
            warnings.append(
                f"  VERDICT_SKEW {et}: '{majority_label}' = {majority_frac:.0%} (n={total_et})"
            )
            print(f"    {et:35s} ⚠ '{majority_label}' = {majority_frac:.0%} (n={total_et})")
        else:
            print(f"    {et:35s} ✓ balanced (n={total_et})")

    # 3. Concentration ratio — top 2 types shouldn't be >60% combined
    top2 = type_counts.most_common(2)
    top2_pct = sum(c for _, c in top2) / total * 100
    if top2_pct > 60:
        warnings.append(
            f"  CONCENTRATED top 2 types = {top2_pct:.0f}% of data "
            f"({top2[0][0]}={top2[0][1]}, {top2[1][0]}={top2[1][1]})"
        )

    return warnings


def main():
    parser = argparse.ArgumentParser(description="Detect reward hacking in RL evaluation results")
    parser.add_argument("dirs", nargs="*",
                        help="1 or 2 experiment directories (if 2: first=baseline, second=RL)")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Display names (default: folder names)")
    parser.add_argument("--check-gen", default=None, metavar="JSONL",
                        help="Check a generated JSONL file for distribution balance")
    args = parser.parse_args()

    if args.check_gen:
        print("=" * 70)
        print("GENERATION BALANCE CHECK")
        print("=" * 70)
        warnings = check_generation_balance(args.check_gen)
        if warnings:
            print(f"\n  WARNINGS:")
            for w in warnings:
                print(w)
            print(f"\n{'=' * 70}")
            print(f"VERDICT: {len(warnings)} warning(s) — review before training.")
            print("=" * 70)
            return 1
        else:
            print(f"\n{'=' * 70}")
            print("VERDICT: No generation balance issues detected.")
            print("=" * 70)
            return 0

    dirs = [Path(d) for d in args.dirs]
    names = args.names or [d.name.replace("_", " ").title() for d in dirs]

    dfs = [load_eval(d) for d in dirs]

    print("=" * 70)
    print("REWARD HACKING DETECTION REPORT")
    print("=" * 70)

    all_warnings = []
    all_infos = []

    # Single-model checks
    for df, name in zip(dfs, names):
        print(f"\n--- {name} ---")
        warnings, infos = check_single(df, name)
        if warnings:
            print("\n  WARNINGS:")
            for w in warnings:
                print(w)
            all_warnings.extend(warnings)
        if infos:
            print("\n  INFO:")
            for i in infos:
                print(i)
            all_infos.extend(infos)
        if not warnings and not infos:
            print("  No issues detected.")

    # Comparison checks (if 2 models)
    if len(dfs) == 2:
        print(f"\n--- Comparison: {names[0]} vs {names[1]} ---")
        comp_warnings = check_comparison(dfs[0], dfs[1], names[0], names[1])
        numeric_warnings = check_numeric_tasks(dfs[0], dfs[1], names[0], names[1])
        comp_warnings.extend(numeric_warnings)

        if comp_warnings:
            print("\n  WARNINGS:")
            for w in comp_warnings:
                print(w)
            all_warnings.extend(comp_warnings)
        else:
            print("  No comparison issues detected.")

    # Summary
    print(f"\n{'=' * 70}")
    if all_warnings:
        print(f"VERDICT: {len(all_warnings)} warning(s) detected — review before trusting results.")
    else:
        print("VERDICT: No reward hacking signals detected.")
    print("=" * 70)

    return 1 if all_warnings else 0


if __name__ == "__main__":
    sys.exit(main())
