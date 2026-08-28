#!/usr/bin/env python3
"""Generate a human-auditable PDF showcasing all QA types by training tier.

Groups every eval_type into its training tier (SFT Atomic, SFT Thinking,
Bridge, RL Taxonomy, RL Thinking, OOD Evaluation) and renders 1 example
per eval_type as a formatted LaTeX document compiled to PDF.

Usage:
    python scripts/utils/generate_qa_showcase.py data/debug.jsonl
    python scripts/utils/generate_qa_showcase.py data/debug.jsonl -o my_showcase.pdf
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

_SFT_THINKING_TYPES = frozenset({
    'description', 'segment_judgment', 'cross_stat_judgment',
    'anti_judgment', 'segment_trend_dominance', 'cross_trend_query',
    'stat_numerical', 'periodicity', 'change_point',
    'local_enumeration', 'segment_enumeration', 'transition_enumeration',
    'event_segment_enumeration', 'temporal_position', 'duration_proportion',
    'yes_no', 'correlation', 'anticorrelation', 'clustering', 'anticlustering',
    'cross_metric_enumeration',
})

TIER_ORDER = [
    'SFT Atomic',
    'SFT Thinking',
    'Bridge',
    'RL Taxonomy',
    'RL Thinking',
    'OOD Evaluation',
]

TIER_DESCRIPTIONS = {
    'SFT Atomic': (
        "Atomic skills taught via supervised fine-tuning. Each type teaches one "
        "fundamental operation (lookup, comparison, enumeration) with a full "
        "chain-of-thought decomposition. The model learns HOW to reason about "
        "time series from these examples."
    ),
    'SFT Thinking': (
        "Structured thinking QA types trained via SFT with full chain-of-thought "
        "think blocks (data above \\texttt{===}, computation below). "
        "These cover segment-level reasoning, statistical queries, periodicity, "
        "anomaly detection, and multi-metric comparisons."
    ),
    'Bridge': (
        "Bridge patterns that teach explicit decomposition strategies for "
        "compositional tasks. These use full step-by-step think blocks that "
        "show the model how to break complex questions into atomic operations. "
        "Trained via SFT (force-SFT routing)."
    ),
    'RL Taxonomy': (
        "Compositional tasks where the model must discover its own reasoning "
        "strategy via reinforcement learning. Only the answer line is checked "
        "by the reward function --- the model is free to think however it wants. "
        "Think blocks are minimal (just `answer: X`)."
    ),
    'RL Thinking': (
        "Compositional types trained via RL with structured thinking. "
        "These include segment judgment, trend dominance, and other "
        "multi-step reasoning tasks."
    ),
    'OOD Evaluation': (
        "Out-of-distribution evaluation types. These are NEVER seen during "
        "training --- they test whether the model can generalize its learned "
        "skills to novel question compositions. Each OOD type combines "
        "familiar operations in unfamiliar ways."
    ),
}

TIER_COLORS = {
    'SFT Atomic': ('blue!8', 'blue!60!black'),
    'SFT Thinking': ('blue!5', 'blue!40!black'),
    'Bridge': ('green!8', 'green!50!black'),
    'RL Taxonomy': ('orange!8', 'orange!60!black'),
    'RL Thinking': ('orange!5', 'orange!40!black'),
    'OOD Evaluation': ('red!8', 'red!50!black'),
}


def classify_tier(eval_type: str, eval_metadata: dict) -> str:
    if eval_metadata.get('bridge'):
        return 'Bridge'
    if eval_type.startswith('atomic_'):
        return 'SFT Atomic'
    if eval_type.startswith('ood_'):
        return 'OOD Evaluation'
    if eval_type.startswith('rl_'):
        return 'RL Taxonomy'
    if eval_type in _SFT_THINKING_TYPES:
        return 'SFT Thinking'
    if eval_type == 'tsevol':
        return 'SFT Thinking'
    return 'RL Thinking'


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_question(input_text: str) -> str:
    """Extract the actual question from the input prompt, stripping context preamble."""
    # Remove time series data markers
    text = input_text.replace('<ts>', '[time series data]')

    # For MTS: "In a X system, there are N metrics:\n metric1...\n QUESTION"
    # The question usually starts after the last semicolon-terminated metric line
    lines = text.split('\n')
    question_lines = []
    found_question = False
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Stop when we hit a metric line (ends with ;) or context line
        if stripped.endswith(';') or stripped.startswith('In a') or stripped.startswith('In an'):
            break
        question_lines.insert(0, stripped)
        found_question = True

    if found_question and question_lines:
        return ' '.join(question_lines)

    # For UTS: "You are a time series analysis expert. ... <ts>. QUESTION"
    # Extract after last [time series data]
    parts = text.split('[time series data]')
    if len(parts) > 1:
        after = parts[-1].strip()
        # Remove leading punctuation
        after = after.lstrip('.;, ')
        if after:
            return after

    # Fallback: return last 200 chars
    return text[-200:] if len(text) > 200 else text


def extract_think_block(output: str) -> str:
    """Extract content between <think> and </think>."""
    m = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    return m.group(1).strip() if m else ''


def extract_response(output: str) -> str:
    """Extract the natural language response after </think>."""
    parts = output.split('</think>')
    if len(parts) > 1:
        return parts[-1].strip()
    return output.strip()


# ---------------------------------------------------------------------------
# LaTeX escaping
# ---------------------------------------------------------------------------

# Unicode (∩, ∪, →, ≥, ≤, ≠, ×, —, –, …) passes through untouched — rendered
# natively by lualatex with DejaVu Sans Mono for Verbatim and Latin Modern
# (via fontspec) for text.

def tex_escape(text: str) -> str:
    """Escape special LaTeX characters."""
    # Order matters: & first, then others
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def tex_verb(text: str, max_lines: int = None) -> str:
    """Prepare text for a Verbatim block (no escaping, no truncation).

    The Verbatim environment is configured with breaklines=true, so long lines
    wrap onto the next line instead of overflowing. Never truncate — always
    show the full thinking block.
    """
    return text


# ---------------------------------------------------------------------------
# LaTeX document generation
# ---------------------------------------------------------------------------

def generate_latex(samples_by_tier: dict, stats: dict) -> str:
    """Generate full LaTeX document content."""

    doc = []
    doc.append(r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage[dvipsnames]{xcolor}
\usepackage{tcolorbox}
\usepackage{fancyvrb}
\usepackage{enumitem}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{hyperref}
\usepackage{fontspec}
\setmainfont{Latin Modern Roman}
\setsansfont{Latin Modern Sans}
\setmonofont{DejaVu Sans Mono}[Scale=0.85]

\tcbuselibrary{breakable,skins}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!50!black,
    urlcolor=blue!50!black,
    pdfauthor={ChatTS Pipeline},
    pdftitle={QA Generation Showcase},
}

% Custom tcolorbox styles
\newtcolorbox{questionbox}[1][]{
    colback=gray!5, colframe=gray!50!black,
    title={\textbf{Question}}, fonttitle=\small,
    breakable, left=4pt, right=4pt, top=2pt, bottom=2pt,
    #1
}

\newtcolorbox{thinkbox}[1][]{
    colback=yellow!3, colframe=yellow!50!black,
    title={\textbf{Think Block} \textit{\small(model's chain-of-thought)}},
    fonttitle=\small,
    breakable, left=4pt, right=4pt, top=2pt, bottom=2pt,
    #1
}

\newtcolorbox{responsebox}[1][]{
    colback=green!3, colframe=green!40!black,
    title={\textbf{Response} \textit{\small(what evaluation LLM sees)}},
    fonttitle=\small,
    breakable, left=4pt, right=4pt, top=2pt, bottom=2pt,
    #1
}

\newtcolorbox{metabox}[1][]{
    colback=blue!3, colframe=blue!30!black,
    breakable, left=4pt, right=4pt, top=2pt, bottom=2pt,
    #1
}

\newtcolorbox{tierbox}[2][]{
    colback=#2, colframe=gray!60!black,
    fonttitle=\bfseries\large,
    title={#1}, breakable,
    left=6pt, right=6pt, top=4pt, bottom=4pt,
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}

\begin{document}

\begin{center}
{\LARGE\bfseries ChatTS QA Generation Showcase}\\[6pt]
{\large Human Audit Document}\\[4pt]
{\small Generated from pipeline dryrun data}\\[12pt]
\end{center}
""")

    # Summary table
    doc.append(r"""
\section*{Summary}

\begin{center}
\begin{tabular}{lrr}
\toprule
\textbf{Training Tier} & \textbf{Eval Types} & \textbf{Sample Count} \\
\midrule
""")

    for tier in TIER_ORDER:
        types = samples_by_tier.get(tier, {})
        count = sum(len(v) for v in types.values())
        doc.append(f"{tex_escape(tier)} & {len(types)} & {count} \\\\\n")

    doc.append(r"""\midrule
""")
    total_types = sum(len(v) for v in samples_by_tier.values())
    total_samples = stats['total']
    doc.append(f"\\textbf{{Total}} & {total_types} & {total_samples} \\\\\n")
    doc.append(r"""\bottomrule
\end{tabular}
\end{center}
""")

    # Architecture explanation
    doc.append(r"""
\subsection*{Training Architecture}
\begin{itemize}[nosep]
\item \textbf{SFT (Supervised Fine-Tuning):} Model learns to produce correct format and reasoning from examples.
\item \textbf{Bridge (SFT):} Teaches decomposition strategies for compositional tasks. Force-routed to SFT.
\item \textbf{RL (Reinforcement Learning):} Model discovers reasoning via reward signal on the \texttt{answer:} line.
\item \textbf{OOD (Out-of-Distribution):} Eval-only. Never trained. Tests generalization.
\end{itemize}

\subsection*{Think Block Structure}
\begin{itemize}[nosep]
\item \textbf{SFT Atomic:} Full decomposition --- data display, step-by-step computation, answer line.
\item \textbf{SFT Thinking:} Structured format with data above \texttt{===}, computation below, answer line.
\item \textbf{Bridge:} Full decomposition plan with step results and combination.
\item \textbf{RL / OOD:} Minimal --- just \texttt{answer: X}. Model thinks freely; only answer is scored.
\end{itemize}

\subsection*{Scoring Architecture}
\begin{itemize}[nosep]
\item \textbf{Reward (training):} Rule-based. Parses \texttt{answer:} line from \texttt{<think>} block.
\item \textbf{Evaluation:} LLM-as-judge. Sees ONLY the response after \texttt{</think>}. Never sees think block.
\end{itemize}

\clearpage
\tableofcontents
\clearpage
""")

    # Each tier
    for tier in TIER_ORDER:
        types = samples_by_tier.get(tier, {})
        if not types:
            continue

        bg_color, _ = TIER_COLORS.get(tier, ('gray!5', 'gray!60!black'))
        desc = TIER_DESCRIPTIONS.get(tier, '')

        ets_in_tier = len({et for (et, _) in types})
        pairs_in_tier = len(types)
        doc.append(
            f"\n\\section{{{tex_escape(tier)} "
            f"({ets_in_tier} types, {pairs_in_tier} sub-types)}}\n"
        )
        doc.append(f"\\textit{{{tex_escape(desc)}}}\n\n")

        for (eval_type, sub_type_key) in sorted(types.keys()):
            sample = types[(eval_type, sub_type_key)]
            et = sample['eval_type']
            meta = sample.get('eval_metadata', {})
            sub_type = meta.get('sub_type', '') or sub_type_key
            verdict = meta.get('verdict', '')
            seq_len = meta.get('length', '')

            question = extract_question(sample['input'])
            think = extract_think_block(sample['output'])
            response = extract_response(sample['output'])

            # Subsection for this (eval_type, sub_type) — include sub_type in
            # the heading so every sub-family is individually auditable.
            heading = tex_escape(et)
            if sub_type:
                heading = f"{heading} --- {tex_escape(str(sub_type))}"
            doc.append(f"\n\\subsection{{{heading}}}\n")

            # Metadata line
            meta_parts = [f"\\textbf{{eval\\_type:}} \\texttt{{{tex_escape(et)}}}"]
            if sub_type:
                meta_parts.append(f"\\textbf{{sub\\_type:}} \\texttt{{{tex_escape(str(sub_type))}}}")
            if verdict:
                meta_parts.append(f"\\textbf{{verdict:}} \\texttt{{{tex_escape(str(verdict))}}}")
            if seq_len:
                meta_parts.append(f"\\textbf{{seq\\_len:}} {seq_len}")

            doc.append("\\begin{metabox}\n")
            doc.append(" \\quad \\textbar\\quad ".join(meta_parts))
            doc.append("\n\\end{metabox}\n\n")

            # Question
            doc.append("\\begin{questionbox}\n")
            doc.append(f"{tex_escape(question)}\n")
            doc.append("\\end{questionbox}\n\n")

            # Think block
            if think:
                doc.append("\\begin{thinkbox}\n")
                doc.append("\\begin{Verbatim}[fontsize=\\small,breaklines=true,breaksymbol=]\n")
                think_text = tex_verb(think)
                doc.append(think_text)
                doc.append("\n\\end{Verbatim}\n")
                doc.append("\\end{thinkbox}\n\n")

            # Response
            if response:
                doc.append("\\begin{responsebox}\n")
                doc.append(f"{tex_escape(response)}\n")
                doc.append("\\end{responsebox}\n\n")

            doc.append("\\vspace{8pt}\n")

    doc.append(r"""
\end{document}
""")

    return ''.join(doc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate human-auditable PDF showcasing all QA types."
    )
    parser.add_argument("jsonl_path", help="Path to debug.jsonl")
    parser.add_argument("-o", "--output", default="qa_showcase.pdf",
                        help="Output PDF path (default: qa_showcase.pdf)")
    parser.add_argument("--max-per-type", type=int, default=1,
                        help="Max examples per eval_type (default: 1)")
    args = parser.parse_args()

    # Load data
    path = Path(args.jsonl_path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Loading {path}...")
    samples = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            # Skip timeseries field (huge, useless for showcase)
            rec.pop('timeseries', None)
            samples.append(rec)

    print(f"Loaded {len(samples)} samples")

    # Group by (eval_type, sub_type) — must cover every pair, not just every
    # eval_type. Missing a sub_type hides a whole question family from audit.
    by_pair = defaultdict(list)
    for s in samples:
        et = s.get('eval_type', 'unknown')
        st = s.get('eval_metadata', {}).get('sub_type', '') or ''
        by_pair[(et, st)].append(s)

    # Pick one representative per (eval_type, sub_type). Tier is per-eval_type.
    samples_by_tier = defaultdict(dict)
    eval_types_seen = set()
    for (et, st), pair_samples in sorted(by_pair.items()):
        best = max(pair_samples[:20], key=lambda s: len(extract_think_block(s.get('output', ''))))
        tier = classify_tier(et, best.get('eval_metadata', {}))
        samples_by_tier[tier][(et, st)] = best
        eval_types_seen.add(et)

    stats = {
        'total': len(samples),
        'types': len(eval_types_seen),
        'pairs': len(by_pair),
    }

    # Print tier summary
    print("\nTier distribution (eval_type / sub_type pairs):")
    for tier in TIER_ORDER:
        pairs = samples_by_tier.get(tier, {})
        ets = len({et for (et, _) in pairs})
        print(f"  {tier:20s}: {ets:3d} eval_types, {len(pairs):3d} (type,sub_type) pairs")

    # Generate LaTeX
    print("\nGenerating LaTeX...")
    latex_content = generate_latex(samples_by_tier, stats)

    # Write and compile
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "showcase.tex")
        with open(tex_path, 'w') as f:
            f.write(latex_content)

        # Also save .tex for debugging
        tex_backup = args.output.replace('.pdf', '.tex')
        with open(tex_backup, 'w') as f:
            f.write(latex_content)
        print(f"LaTeX source saved to: {tex_backup}")

        # Compile (run twice for TOC). lualatex for native Unicode support
        # (∩, ∪, →, ≥, ≤, ≠, ×) in both text and Verbatim.
        print("Compiling PDF...")
        for run in range(2):
            subprocess.run(
                ['lualatex', '-interaction=nonstopmode', '-output-directory', tmpdir, tex_path],
                capture_output=True, text=True, timeout=180,
            )

        # Copy PDF to output
        pdf_src = os.path.join(tmpdir, "showcase.pdf")
        if os.path.exists(pdf_src):
            import shutil
            shutil.copy2(pdf_src, args.output)
            print(f"\nPDF generated: {args.output}")
            print(f"  Size: {os.path.getsize(args.output) / 1024:.1f} KB")
        else:
            print("PDF not generated. Check LaTeX errors.")
            sys.exit(1)


if __name__ == '__main__':
    main()
