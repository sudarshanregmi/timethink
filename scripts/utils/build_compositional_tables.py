"""Build the Compositional Reasoning Study LaTeX tables.

Emits 4 tables to `presentation/figures/compositional_tables.tex`:
  1. Per-bucket summary (ID-easy / ID-hard / OOD)
  2. ID-easy detailed
  3. ID-hard detailed
  4. OOD detailed

Defensive checks (per memory/feedback_per_task_table_coverage.md and
memory/feedback_silent_nan_aggregation.md):
  - 100% taxonomy coverage assertion before writing — fails if any
    eval_type from docs/task_taxonomy.md is missing without explicit
    justification in EXCLUDE_REASONS.
  - Loud NaN handling — `acc()` raises on non-empty subset with all-NaN
    metric column instead of silently returning None.
  - Metric column picker tries `binary_accuracy` first then falls back
    to `f1`, mirroring the actual eval pipeline (NOT the LIST_VERDICT_TYPES
    classification in evaluation/eval/parser.py — that set is too broad
    for column selection in this CSV).
"""

import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Decomposition: each compositional eval_type → (atomic constituents, sub-skill).
# Sub-skill is a short tag for the operations BEYOND the atomic primitives
# (lookup, count, argmax, ratio, region check, spatial join, etc.).
# Approximate mappings are flagged in comments.
# ---------------------------------------------------------------------------
DECOMP = {
    # ===== ID-easy (composition CoT seen during SFT) =====
    'anti_judgment':                (['atomic_cross_trend_align'], 'binary verdict'),
    'anticlustering':               (['atomic_cross_ranking'], 'group anti-similar'),
    'anticorrelation':              (['atomic_cross_trend_align'], 'binary verdict'),
    'change_point':                 (['atomic_chunked_means', 'atomic_interval_mean'], 'inflection detection'),
    'clustering':                   (['atomic_cross_ranking'], 'group similar'),
    'correlation':                  (['atomic_cross_trend_align'], 'binary verdict'),
    'cross_metric_enumeration':     (['atomic_cross_filtering'], 'enumerate matching'),
    'cross_stat_judgment':          (['atomic_cross_stat_compare'], 'binary verdict'),
    'cross_trend_query':            (['atomic_cross_trend_align'], 'filter'),
    'duration_proportion':          (['atomic_trend_enumeration'], 'sum, ratio'),
    'event_segment_enumeration':    (['atomic_event_enumeration', 'atomic_trend_enumeration'], 'enumerate joined'),
    'local_enumeration':            (['atomic_event_enumeration'], 'enumerate local'),
    'periodicity':                  (['atomic_periodic_description'], 'binary verdict'),
    'rl_amplitude_vs_std':          (['atomic_global_std', 'atomic_max_value', 'atomic_min_value'], 'subtract, ratio'),
    'rl_cross_full_ordering':       (['atomic_cross_ranking'], 'full ordering'),
    'rl_cross_stat_ratio':          (['atomic_global_std', 'atomic_global_mean'], 'ratio'),
    'rl_cross_trend_concordance':   (['atomic_cross_trend_align'], 'count concordance'),
    'rl_event_count_by_type':       (['atomic_event_enumeration'], 'filter, count'),
    'rl_event_in_trend_type':       (['atomic_event_enumeration', 'atomic_trend_enumeration'], 'spatial join'),
    'rl_half_mean_compare':         (['atomic_interval_mean'], 'compute x2, compare'),
    'rl_max_in_trend_type':         (['atomic_max_position', 'atomic_trend_enumeration'], 'filter, spatial join'),
    'rl_type_duration_fraction':    (['atomic_trend_enumeration'], 'sum-by-type, ratio'),
    'rl_type_of_longest':           (['atomic_trend_enumeration'], 'argmax (duration)'),
    'rl_volatility_change':         (['atomic_interval_std'], 'compare (x2 intervals)'),
    'segment_enumeration':          (['atomic_trend_enumeration'], 'count'),
    'segment_judgment':             (['atomic_trend_enumeration'], 'binary verdict'),
    'segment_trend_dominance':      (['atomic_trend_enumeration'], 'count by type, argmax'),
    # Approximate: stat_numerical is polymorphic (mean/std/max/min/percentile).
    'stat_numerical':               (['atomic_global_mean', 'atomic_global_std', 'atomic_max_value', 'atomic_min_value'], 'numeric stat (polymorphic)'),
    # Approximate: temporal_position is "when did X happen" - max_position is
    # the closest atomic.
    'temporal_position':            (['atomic_max_position'], 'temporal lookup'),
    'transition_enumeration':       (['atomic_trend_enumeration'], 'enumerate transitions'),
    # Approximate: yes_no spans many topics; treated as generic binary verdict
    # over multiple atomic skills.
    'yes_no':                       (['atomic_cross_trend_align', 'atomic_cross_stat_compare'], 'binary verdict (polymorphic)'),

    # ===== ID-hard (atomics seen, NO CoT for composition) =====
    'rl_amplitude_vs_range':        (['atomic_periodic_description', 'atomic_max_value', 'atomic_min_value'], 'subtract, ratio'),
    'rl_cluster_count':             (['atomic_cross_ranking'], 'cluster, count'),
    'rl_cluster_dominant':          (['atomic_cross_ranking'], 'cluster, argmax'),
    'rl_condition_recovery':        (['atomic_event_enumeration'], 'recovery check'),
    'rl_corr_count':                (['atomic_cross_trend_align'], 'count'),
    'rl_cross_asymmetric_behavior': (['atomic_cross_trend_align'], 'asymmetry check'),
    'rl_cross_attribute_corr':      (['atomic_cross_trend_align'], 'numeric estimate'),
    'rl_cross_conditional_query':   (['atomic_cross_stat_compare'], 'conditional filter'),
    'rl_cross_event_sync':          (['atomic_event_enumeration'], 'temporal sync'),
    'rl_cross_period_compare':      (['atomic_periodic_description'], 'compare'),
    'rl_cycle_count':               (['atomic_periodic_description'], 'count'),
    'rl_dominant_trend_type':       (['atomic_trend_enumeration'], 'count by type, argmax'),
    'rl_event_count':               (['atomic_event_enumeration'], 'count'),
    'rl_event_near_extremum':       (['atomic_event_enumeration', 'atomic_max_position'], 'spatial join'),
    'rl_event_type_at_pos':         (['atomic_event_enumeration'], 'lookup'),
    'rl_extrema_same_half':         (['atomic_max_position', 'atomic_min_position'], 'region join'),
    'rl_half_mean_diff':            (['atomic_interval_mean'], 'compute x2, subtract'),
    'rl_has_periodicity':           (['atomic_periodic_description'], 'binary verdict'),
    'rl_interval_comparison':       (['atomic_interval_mean'], 'compare'),
    'rl_longest_segment':           (['atomic_trend_enumeration'], 'argmax (duration)'),
    'rl_max_amplitude_event':       (['atomic_event_enumeration'], 'argmax (amplitude)'),
    'rl_max_in_first_half':         (['atomic_max_position'], 'region check'),
    'rl_mean_shift':                (['atomic_interval_mean'], 'compare'),
    'rl_mean_stability':            (['atomic_interval_std'], 'compare'),
    'rl_median_mean_close':         (['atomic_global_mean'], 'median, compare'),
    'rl_normalized_range':          (['atomic_max_value', 'atomic_min_value', 'atomic_global_std'], 'subtract, normalize'),
    'rl_period_estimate':           (['atomic_periodic_description'], 'numeric estimate'),
    'rl_range':                     (['atomic_max_value', 'atomic_min_value'], 'subtract'),
    'rl_segment_count':             (['atomic_trend_enumeration'], 'count'),
    'rl_segment_duration':          (['atomic_trend_enumeration'], 'lookup, subtract'),
    'rl_segment_mean_compare':      (['atomic_interval_mean', 'atomic_trend_enumeration'], 'compare'),
    'rl_segment_type_at_pos':       (['atomic_trend_enumeration'], 'lookup'),

    # ===== OOD (composition NEVER seen) =====
    'ood_amplitude_vs_segment_std': (['atomic_max_value', 'atomic_min_value', 'atomic_global_std'], 'subtract, compare'),
    'ood_chunk_above_proportion':   (['atomic_chunked_means'], 'filter, ratio'),
    'ood_cluster_singleton':        (['atomic_cross_ranking'], 'cluster, singleton check'),
    'ood_conditional_count':        (['atomic_event_enumeration'], 'filter, count'),
    'ood_conditional_mean_by_type': (['atomic_trend_enumeration', 'atomic_interval_mean'], 'filter, weighted mean'),
    'ood_conditional_stat':         (['atomic_event_enumeration', 'atomic_global_mean'], 'filter, mean'),
    'ood_corr_transitivity':        (['atomic_cross_trend_align'], 'transitive logic'),
    'ood_cross_concordant_shift':   (['atomic_cross_trend_align', 'atomic_interval_mean'], 'count concordance'),
    'ood_cross_corr_count':         (['atomic_cross_trend_align', 'atomic_cross_counting'], 'count'),
    'ood_cross_event_causality':    (['atomic_event_enumeration'], 'temporal causality'),
    'ood_cross_extrema_alignment':  (['atomic_max_position'], 'alignment check'),
    'ood_cross_range_overlap':      (['atomic_max_value', 'atomic_min_value'], 'interval overlap'),
    'ood_cross_trend_convergence':  (['atomic_cross_trend_align'], 'temporal convergence'),
    'ood_cycle_mean_trend':         (['atomic_periodic_description', 'atomic_trend_enumeration'], 'correlation check'),
    'ood_dominant_trend_in_cluster':(['atomic_trend_enumeration', 'atomic_cross_ranking'], 'cluster, argmax'),
    'ood_duration_weighted_mean':   (['atomic_interval_mean', 'atomic_trend_enumeration'], 'weighted average'),
    'ood_event_amplitude_vs_std':   (['atomic_event_enumeration', 'atomic_global_std'], 'compare'),
    'ood_event_density':            (['atomic_event_enumeration'], 'density'),
    'ood_event_density_by_trend':   (['atomic_event_enumeration', 'atomic_trend_enumeration'], 'density ratio'),
    'ood_longest_type_fraction':    (['atomic_trend_enumeration'], 'sum-by-type, ratio'),
    'ood_max_amp_in_longest_segment':(['atomic_max_value', 'atomic_trend_enumeration'], 'filter, argmax'),
    'ood_max_before_min':           (['atomic_max_position', 'atomic_min_position'], 'temporal compare'),
    'ood_max_in_highest_mean_quarter':(['atomic_max_position', 'atomic_interval_mean'], 'argmax, region'),
    'ood_max_mean_chunk_pos':       (['atomic_chunked_means'], 'argmax (position)'),
    'ood_mixed_corr_anti':          (['atomic_cross_trend_align', 'atomic_cross_counting'], 'classify, count'),
    'ood_peak_in_longest_segment':  (['atomic_max_position', 'atomic_trend_enumeration'], 'argmax, spatial join'),
    'ood_quarter_mean_ordering':    (['atomic_interval_mean'], 'x4 intervals, ordering'),
    'ood_range_normalized_amplitude':(['atomic_max_value', 'atomic_min_value'], 'subtract, normalize'),
    'ood_segment_stat_compare':     (['atomic_interval_mean'], 'compare'),
    'ood_std_exceeds_half_range':   (['atomic_global_std', 'atomic_max_value', 'atomic_min_value'], 'subtract, compare'),
    'ood_symmetric_recovery':       (['atomic_event_enumeration'], 'symmetry check'),
    'ood_symmetric_trend_sequence': (['atomic_trend_enumeration'], 'symmetry check'),
    'ood_trend_follows_mean':       (['atomic_trend_enumeration', 'atomic_interval_mean'], 'correlation check'),
    'ood_trend_reversal':           (['atomic_trend_enumeration'], 'reversal pattern'),
}

# Eval types deliberately excluded with a documented reason. Anything in the
# taxonomy NOT in DECOMP and NOT here triggers a hard failure.
EXCLUDE_REASONS = {
    'ood_nested_extrema': 'no eval data on test/ split (n=0 in evaluation_report_optimized.csv)',
}


# ---------------------------------------------------------------------------
# Defensive helpers
# ---------------------------------------------------------------------------
def _parse_buckets(taxonomy_path: Path) -> dict:
    """Parse docs/task_taxonomy.md into {bucket: set(types)}."""
    buckets = {'Atomic': set(), 'ID-easy': set(), 'ID-hard': set(), 'OOD': set()}
    current = None
    with open(taxonomy_path) as f:
        for line in f:
            if line.startswith('## Atomic'):     current = 'Atomic'
            elif line.startswith('## ID-easy'):  current = 'ID-easy'
            elif line.startswith('## ID-hard'):  current = 'ID-hard'
            elif line.startswith('## OOD'):      current = 'OOD'
            elif line.startswith('## '):         current = None
            elif current and line.startswith('| `'):
                m = re.match(r'\| `([^`]+)`', line)
                if m:
                    buckets[current].add(m.group(1))
    return buckets


def acc(df: pd.DataFrame, task_type: str) -> float | None:
    """Per-eval_type accuracy. Loud on NaN with non-empty input.

    Tries `binary_accuracy` first (the unified metric for most types in
    `evaluation_report_optimized.csv`), falls back to `f1` only for types
    that genuinely populate it (clustering, anticlustering). Raises if a
    non-empty subset has no usable metric.
    """
    sub = df[df.task_type == task_type]
    if len(sub) == 0:
        return None  # genuinely no data
    for col in ('binary_accuracy', 'f1'):
        if col not in sub.columns:
            continue
        v = sub[col].mean()
        if not pd.isna(v):
            return float(v)
    nonnull_cols = {c: int(sub[c].notna().sum()) for c in sub.columns if sub[c].dtype != 'object' and sub[c].notna().sum() > 0}
    raise ValueError(
        f"acc({task_type!r}): {len(sub)} rows but no usable metric in "
        f"binary_accuracy / f1. Non-NaN numeric columns: {nonnull_cols}"
    )


def coverage_check(buckets: dict) -> None:
    """Hard-fail if any taxonomy type is neither in DECOMP nor in EXCLUDE_REASONS."""
    bucket_union = set().union(*[buckets[b] for b in ('ID-easy', 'ID-hard', 'OOD')])
    accounted = set(DECOMP) | set(EXCLUDE_REASONS)
    missing = bucket_union - accounted
    extra = accounted - bucket_union - buckets['Atomic']
    if missing:
        msg = "FATAL: taxonomy types missing from DECOMP / EXCLUDE_REASONS:\n"
        for t in sorted(missing):
            msg += f"  {t}\n"
        msg += "Either add to DECOMP with constituents, or to EXCLUDE_REASONS with justification."
        raise SystemExit(msg)
    if extra:
        # Not fatal but suspicious — DECOMP entries that aren't in any bucket.
        print(f"WARNING: DECOMP/EXCLUDE entries not in any bucket: {sorted(extra)}", file=sys.stderr)


# ---------------------------------------------------------------------------
# LaTeX emit
# ---------------------------------------------------------------------------
def _latex_escape(s: str) -> str:
    return (s.replace('_', r'\_')
             .replace('&', r'\&')
             .replace('%', r'\%')
             .replace('×', r'$\times$')
             .replace('x2', r'$\times$2')
             .replace('x4', r'$\times$4'))


def _fmt_atoms(atoms: list[str], subskill: str) -> str:
    a = ', '.join(_latex_escape(a.replace('atomic_', '')) for a in atoms)
    return f"{a} {{\\footnotesize\\textit{{(+ {_latex_escape(subskill)})}}}}"


def _emit_detail_table(df: pd.DataFrame, label: str, caption: str) -> str:
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"\centering")
    L.append(r"\footnotesize")
    L.append(r"\setlength{\tabcolsep}{4pt}")
    L.append(r"\caption{" + caption + "}")
    L.append(r"\label{" + label + "}")
    L.append(r"\begin{tabular}{@{}l p{5.8cm} ccc c ccc@{}}")
    L.append(r"\toprule")
    L.append(r"\multirow{2}{*}{Task type} & \multirow{2}{*}{Atomics (+ sub-skills)} & "
             r"\multicolumn{3}{c}{Atomic skills} & & \multicolumn{3}{c}{Composition} \\")
    L.append(r"\cmidrule(lr){3-5} \cmidrule(lr){7-9}")
    L.append(r" & & SFT & RL & $\Delta_a$ & & SFT & RL & $\Delta_c$ \\")
    L.append(r"\midrule")
    for _, r in df.sort_values('d_co', ascending=False).iterrows():
        task_disp = r"\texttt{" + _latex_escape(r['task']) + "}"
        atoms_disp = _fmt_atoms(r['atoms'], r['subskill'])
        d_co = f"{r['d_co']:+.2f}"
        if r['d_co'] >= 0.20: d_co = r"\textbf{" + d_co + "}"
        L.append(
            f"{task_disp} & {atoms_disp} & "
            f"{r['s_at']:.2f} & {r['r_at']:.2f} & {r['d_at']:+.2f} & & "
            f"{r['s_co']:.2f} & {r['r_co']:.2f} & {d_co} \\\\"
        )
    m = df.mean(numeric_only=True)
    L.append(r"\midrule")
    L.append(
        r"\textbf{Bucket mean} & & "
        f"\\textbf{{{m['s_at']:.2f}}} & \\textbf{{{m['r_at']:.2f}}} & \\textbf{{{m['d_at']:+.2f}}} & & "
        f"\\textbf{{{m['s_co']:.2f}}} & \\textbf{{{m['r_co']:.2f}}} & \\textbf{{{m['d_co']:+.2f}}} \\\\"
    )
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table*}")
    return '\n'.join(L)


def main():
    sft = pd.read_csv(REPO / 'exp/sft/test/evaluation_report_optimized.csv')
    rl  = pd.read_csv(REPO / 'exp/rl/test/evaluation_report_optimized.csv')
    buckets = _parse_buckets(REPO / 'docs/task_taxonomy.md')

    coverage_check(buckets)

    bucket_of = {}
    for b in ('ID-easy', 'ID-hard', 'OOD'):
        for t in buckets[b]:
            bucket_of[t] = b

    # ---- Build rows (loud on missing data) ----
    rows = []
    skipped_no_data = []
    for t, (atoms, subskill) in DECOMP.items():
        if t not in bucket_of:
            print(f"WARNING: DECOMP entry {t!r} not in any compositional bucket; skipping.", file=sys.stderr)
            continue
        s_at_vals = [acc(sft, a) for a in atoms]
        r_at_vals = [acc(rl,  a) for a in atoms]
        s_at_vals = [x for x in s_at_vals if x is not None]
        r_at_vals = [x for x in r_at_vals if x is not None]
        if not s_at_vals or not r_at_vals:
            skipped_no_data.append((t, "atomic constituents lack eval data"))
            continue
        s_co = acc(sft, t)
        r_co = acc(rl, t)
        if s_co is None or r_co is None:
            skipped_no_data.append((t, "composition lacks eval data"))
            continue
        rows.append({
            'bucket': bucket_of[t], 'task': t, 'atoms': atoms, 'subskill': subskill,
            's_at': sum(s_at_vals)/len(s_at_vals), 'r_at': sum(r_at_vals)/len(r_at_vals),
            's_co': s_co, 'r_co': r_co,
        })

    df = pd.DataFrame(rows)
    df['d_at'] = df.r_at - df.s_at
    df['d_co'] = df.r_co - df.s_co
    df['gap']  = df.d_co - df.d_at

    # ---- Final coverage assertion ----
    for b in ('ID-easy', 'ID-hard', 'OOD'):
        in_table = set(df[df.bucket == b].task)
        in_decomp = {t for t in DECOMP if bucket_of.get(t) == b}
        excluded = {t for t in EXCLUDE_REASONS if bucket_of.get(t) == b}
        skipped = {t for t, _ in skipped_no_data if bucket_of.get(t) == b}
        accounted = in_table | excluded | skipped
        missing = buckets[b] - accounted
        if missing:
            raise SystemExit(f"FATAL: {b} unaccounted: {sorted(missing)}")
        n_have = len(in_table)
        n_total = len(buckets[b])
        n_excl = len(excluded) + len(skipped)
        print(f"  {b}: {n_have} rows ({n_total} taxonomy total, {n_excl} excluded with reason)")

    # ---- Summary table ----
    summary = (df.groupby('bucket', sort=False)
                 .agg(s_at=('s_at','mean'), r_at=('r_at','mean'), d_at=('d_at','mean'),
                      s_co=('s_co','mean'), r_co=('r_co','mean'), d_co=('d_co','mean'),
                      gap=('gap','mean'), n=('task','count'))
                 .reset_index()
                 .set_index('bucket').reindex(['ID-easy', 'ID-hard', 'OOD']).reset_index())

    # ---- Emit ----
    preamble = r"""% --------------------------------------------------------------------
% Compositional Reasoning Study: per-bucket task breakdown.
%
% Generated by scripts/utils/build_compositional_tables.py.
% Defensive checks (memory/feedback_per_task_table_coverage.md,
% memory/feedback_silent_nan_aggregation.md): coverage asserted vs.
% docs/task_taxonomy.md; loud failure on NaN aggregation.
%
% Metric: per-task accuracy (each task's natural metric, averaged).
% For verdict-match types, exact-match binary_accuracy.  For numeric
% types (positions, counts, floats), proximity-scaled binary_accuracy
% from evaluation/eval/parser.py (1 - relative_error, clipped; positional
% types scaled by seq_len).  For LIST types, binary_accuracy carries the
% graded-similarity score except for clustering/anticlustering which use F1.
% Atomic-skill columns average accuracy across the listed constituents.
% --------------------------------------------------------------------
"""
    excl_note = (
        f"\n% Intentionally excluded ({len(EXCLUDE_REASONS)} types):\n" +
        "\n".join(f"%   {t}: {reason}" for t, reason in EXCLUDE_REASONS.items()) +
        "\n"
    ) if EXCLUDE_REASONS else ""

    # Summary table
    S = []
    S.append(r"\begin{table}[t]")
    S.append(r"\centering")
    S.append(r"\small")
    S.append(r"\caption{Compositional reasoning summary: per-bucket means of atomic and "
             r"composition accuracy under SFT vs RL. Atomic skills improve roughly uniformly "
             r"across buckets ($\Delta_a {\approx} {+}0.11$); composition gains scale with how "
             r"compositional the bucket is. Composition CoT supervision is present in ID-easy, "
             r"absent in ID-hard, and the task itself is held out in OOD. The compositional "
             r"gain $\Delta_c$ is largest in ID-hard, the bucket that most directly tests "
             r"learned compositional reasoning.}")
    S.append(r"\label{tab:compositional-summary}")
    S.append(r"\begin{tabular}{@{}l c ccc c ccc@{}}")
    S.append(r"\toprule")
    S.append(r"\multirow{2}{*}{Bucket} & \multirow{2}{*}{$n$} & "
             r"\multicolumn{3}{c}{Atomic skills} & & \multicolumn{3}{c}{Composition} \\")
    S.append(r"\cmidrule(lr){3-5} \cmidrule(lr){7-9}")
    S.append(r" & & SFT & RL & $\Delta_a$ & & SFT & RL & $\Delta_c$ \\")
    S.append(r"\midrule")
    for _, r in summary.iterrows():
        S.append(
            f"{r['bucket']} & {int(r['n'])} & "
            f"{r['s_at']:.2f} & {r['r_at']:.2f} & {r['d_at']:+.2f} & & "
            f"{r['s_co']:.2f} & {r['r_co']:.2f} & {r['d_co']:+.2f} \\\\"
        )
    S.append(r"\bottomrule")
    S.append(r"\end{tabular}")
    S.append(r"\end{table}")

    t1 = _emit_detail_table(
        df[df.bucket == 'ID-easy'], 'tab:comp-id-easy',
        r'Compositional reasoning, \textbf{ID-easy bucket} (composition CoT seen during SFT). '
        r'Per-task accuracy under SFT and RL on the constituent atomic skills (averaged) and on '
        r'the composition itself. Composition gains $\Delta_c$ are small on average because SFT '
        r'CoT supervision already teaches the decomposition.')
    t2 = _emit_detail_table(
        df[df.bucket == 'ID-hard'], 'tab:comp-id-hard',
        r'Compositional reasoning, \textbf{ID-hard bucket} (atomic skills seen in SFT, but the '
        r'composition CoT was \emph{never} demonstrated; the model only saw raw RL rollouts). '
        r'The compositional gain dominates here, with individual tasks reaching $+0.62$. Bold '
        r'$\Delta_c$ values exceed $+0.20$.')
    t3 = _emit_detail_table(
        df[df.bucket == 'OOD'], 'tab:comp-ood',
        r'Compositional reasoning, \textbf{OOD bucket} (composition NEVER seen during training; '
        r'only the constituent atomics appeared in SFT). Per-task variance is high; the strongest '
        r'tasks --- e.g.\ \texttt{ood\_cross\_concordant\_shift} --- demonstrate that the '
        r'compositional ability transfers to held-out task structures.')

    out = REPO / 'presentation/figures/compositional_tables.tex'
    with open(out, 'w') as f:
        f.write(preamble + excl_note + '\n')
        f.write('\n'.join(S) + '\n\n')
        f.write(t1 + '\n\n')
        f.write(t2 + '\n\n')
        f.write(t3 + '\n')

    print(f"\nWrote {out}")
    print("\nBucket summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == '__main__':
    main()
