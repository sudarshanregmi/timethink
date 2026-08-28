#!/usr/bin/env python3
"""Extract representative QA examples from JSONL and produce a PDF.

Usage:
    # From dryrun output (no GPU needed):
    python -m synth.align --debug --dryrun
    python scripts/utils/extract_examples_for_paper.py inspect_data/debug.jsonl

    # From production data:
    python scripts/utils/extract_examples_for_paper.py data/train_sft.jsonl data/test.jsonl
"""
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict


def strip_ts(text):
    """Remove timeseries tokens, keep the readable question part."""
    idx = text.rfind('</ts>')
    if idx != -1:
        text = text[idx + len('</ts>'):].lstrip('. ;')
    while '<ts>' in text:
        s = text.find('<ts>')
        e = text.find('</ts>', s)
        if e == -1:
            break
        text = text[:s] + text[e + len('</ts>'):]
    return text.strip()


def sanitize(text):
    """Remove non-ASCII bytes that break pdflatex."""
    return text.encode('ascii', errors='ignore').decode('ascii')


def escape_latex(text):
    """Escape special LaTeX characters for lstlisting-safe content."""
    text = sanitize(text)
    replacements = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
        '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def pick_best(examples):
    """Pick the example with shortest output (most readable)."""
    return min(examples, key=lambda r: len(r.get('output', '')))


def load_examples(jsonl_paths):
    """Load all examples grouped by eval_type from one or more JSONL files."""
    by_type = defaultdict(list)
    for path in jsonl_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                et = rec.get('eval_type', '')
                if not et:
                    continue
                # Prefer short examples; always keep at least one per type
                out_len = len(rec.get('output', ''))
                if out_len < 1200 or len(by_type[et]) == 0:
                    if len(by_type[et]) < 5:
                        by_type[et].append(rec)
    return by_type


def build_latex(by_type):
    """Build LaTeX document with representative examples."""
    # Group eval_types by category
    categories = [
        ("SFT Thinking (force-SFT, single-step skills)", [
            'description', 'yes_no', 'correlation', 'anticorrelation',
            'clustering', 'anticlustering', 'segment_judgment',
            'segment_trend_dominance', 'anti_judgment', 'cross_trend_query',
            'stat_numerical', 'periodicity', 'change_point',
            'local_enumeration', 'segment_enumeration',
            'cross_metric_enumeration', 'transition_enumeration',
            'event_segment_enumeration', 'temporal_position',
            'duration_proportion', 'cross_stat_judgment',
        ]),
        ("Taxonomy Atomic SFT (single-metric)", [
            'atomic_global_mean', 'atomic_global_std',
            'atomic_min_value', 'atomic_max_value',
            'atomic_min_position', 'atomic_max_position',
            'atomic_percentile', 'atomic_interval_mean', 'atomic_interval_std',
            'atomic_chunked_means', 'atomic_trend_enumeration',
            'atomic_event_enumeration', 'atomic_periodic_description',
        ]),
        ("Taxonomy Atomic SFT (cross-metric)", [
            'atomic_cross_stat_compare', 'atomic_cross_ranking',
            'atomic_cross_filtering', 'atomic_cross_counting',
            'atomic_cross_trend_align',
        ]),
        ("RL Compositions (single-metric)", [
            'rl_amplitude_vs_range', 'rl_amplitude_vs_std',
            'rl_condition_recovery', 'rl_cycle_count',
            'rl_dominant_trend_type', 'rl_event_count',
            'rl_event_count_by_type', 'rl_event_in_trend_type',
            'rl_event_near_extremum', 'rl_event_type_at_pos',
            'rl_extrema_same_half', 'rl_half_mean_compare',
            'rl_half_mean_diff', 'rl_has_periodicity',
            'rl_interval_comparison', 'rl_longest_segment',
            'rl_max_amplitude_event', 'rl_max_in_first_half',
            'rl_max_in_trend_type', 'rl_mean_shift',
            'rl_mean_stability', 'rl_median_mean_close',
            'rl_monotonic_chunks', 'rl_normalized_range',
            'rl_period_estimate', 'rl_range',
            'rl_segment_count', 'rl_segment_duration',
            'rl_segment_mean_compare', 'rl_segment_type_at_pos',
            'rl_type_duration_fraction', 'rl_type_of_longest',
            'rl_volatility_change',
        ]),
        ("RL Compositions (cross-metric)", [
            'rl_cross_stat_ratio', 'rl_cross_full_ordering',
            'rl_cross_event_sync', 'rl_cross_period_compare',
            'rl_cross_trend_concordance', 'rl_cross_conditional_query',
            'rl_cross_attribute_corr', 'rl_cross_asymmetric_behavior',
            'rl_cluster_count', 'rl_cluster_dominant',
            'rl_corr_count', 'rl_corr_conditional',
        ]),
        ("OOD Eval-Only (single-metric)", [
            'ood_conditional_stat', 'ood_nested_extrema', 'ood_event_density',
            'ood_conditional_count', 'ood_trend_reversal',
            'ood_range_normalized_amplitude', 'ood_segment_stat_compare',
            'ood_max_before_min',
            'ood_std_exceeds_half_range', 'ood_max_in_highest_mean_quarter',
            'ood_quarter_mean_ordering',
            'ood_max_mean_chunk_pos', 'ood_chunk_above_proportion',
            'ood_symmetric_recovery', 'ood_cycle_mean_trend',
            'ood_trend_follows_mean', 'ood_conditional_mean_by_type',
            'ood_longest_type_fraction', 'ood_event_amplitude_vs_std',
            'ood_amplitude_vs_segment_std', 'ood_symmetric_trend_sequence',
            'ood_event_density_by_trend', 'ood_max_amp_in_longest_segment',
        ]),
        ("OOD Eval-Only (cross-metric)", [
            'ood_cross_corr_count', 'ood_cross_trend_convergence',
            'ood_cross_extrema_alignment',
            'ood_cross_range_overlap', 'ood_cross_concordant_shift',
            'ood_cluster_singleton', 'ood_corr_transitivity',
            'ood_mixed_corr_anti',
        ]),
        ("TSEvol", ['tsevol']),
    ]

    lines = [
        r'\documentclass[10pt,a4paper]{article}',
        r'\usepackage[margin=1.5cm]{geometry}',
        r'\usepackage{listings}',
        r'\usepackage{xcolor}',
        r'\usepackage{hyperref}',
        r'\usepackage{booktabs}',
        r'\lstdefinestyle{qa}{',
        r'  basicstyle=\ttfamily\scriptsize,',
        r'  breaklines=true, frame=single, backgroundcolor=\color{gray!5},',
        r'  xleftmargin=4pt, xrightmargin=4pt, aboveskip=4pt, belowskip=4pt,',
        r'}',
        r'\setlength{\parindent}{0pt}',
        r'\begin{document}',
        r'\section*{Representative QA Examples by Category}',
        r'\small',
    ]

    total_shown = 0
    total_types_with_data = 0

    for cat_name, type_list in categories:
        # Collect types that have data
        present = [(et, by_type[et]) for et in type_list if et in by_type]
        if not present:
            continue

        lines.append(rf'\subsection*{{{escape_latex(cat_name)} ({len(present)} of {len(type_list)} types)}}')

        for et, examples in present:
            rec = pick_best(examples)
            q = strip_ts(rec.get('input', ''))
            # Get just the question part (last sentence)
            a = rec.get('output', '')
            verdict = rec.get('eval_metadata', {}).get('verdict', '')

            lines.append(rf'\paragraph{{\texttt{{{escape_latex(et)}}}}} verdict: \texttt{{{escape_latex(str(verdict))}}}')
            lines.append('')
            lines.append(r'\textbf{Q:} ' + escape_latex(q[-300:]))
            lines.append('')
            lines.append(r'\begin{lstlisting}[style=qa]')
            lines.append(sanitize(a))
            lines.append(r'\end{lstlisting}')
            lines.append('')

            total_shown += 1
            total_types_with_data += 1

    # Summary table
    lines.append(r'\newpage')
    lines.append(r'\section*{Coverage Summary}')
    lines.append(r'\begin{tabular}{@{}lr@{}}')
    lines.append(r'\toprule')
    lines.append(rf'Total eval\_types with examples & {total_types_with_data} \\')
    lines.append(rf'Total examples shown & {total_shown} \\')
    lines.append(rf'Total eval\_types in data & {len(by_type)} \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    lines.append(r'\bigskip')
    lines.append(r'\subsection*{All eval\_types in data}')
    lines.append(r'\begin{tabular}{@{}lr@{}}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{eval\_type} & \textbf{count} \\')
    lines.append(r'\midrule')
    for et in sorted(by_type.keys()):
        lines.append(rf'\texttt{{{escape_latex(et)}}} & {len(by_type[et])} \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    lines.append(r'\end{document}')
    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_examples_for_paper.py <jsonl_file> [jsonl_file2 ...]")
        print("  e.g.: python extract_examples_for_paper.py inspect_data/debug.jsonl")
        sys.exit(1)

    jsonl_paths = sys.argv[1:]
    print(f"Loading from: {', '.join(jsonl_paths)}")
    by_type = load_examples(jsonl_paths)
    print(f"Found {len(by_type)} eval_types, {sum(len(v) for v in by_type.values())} examples")

    latex_content = build_latex(by_type)

    # Write and compile
    out_dir = os.path.dirname(jsonl_paths[0]) or '.'
    tex_path = os.path.join(out_dir, 'qa_showcase.tex')
    pdf_path = os.path.join(out_dir, 'qa_showcase.pdf')

    with open(tex_path, 'w') as f:
        f.write(latex_content)
    print(f"LaTeX written to: {tex_path}")

    # Compile to PDF (two passes)
    for pass_num in range(2):
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', '-output-directory', out_dir, tex_path],
            capture_output=True,
        )
        if result.returncode != 0 and pass_num == 1:
            print("LaTeX compilation failed. Check the .log file.", file=sys.stderr)
            sys.exit(1)

    # Clean aux files
    for ext in ['.aux', '.log', '.out']:
        aux = os.path.join(out_dir, 'qa_showcase' + ext)
        if os.path.exists(aux):
            os.remove(aux)

    print(f"PDF generated: {pdf_path}")


if __name__ == '__main__':
    main()
