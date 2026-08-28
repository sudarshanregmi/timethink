#!/usr/bin/env python3
"""Generate LaTeX content demonstrating the TSEvol pipeline for Overleaf.

Usage:
    python scripts/utils/tsevol_demo_latex.py
    python scripts/utils/tsevol_demo_latex.py -o tsevol_demo.tex
"""

import argparse
from pathlib import Path

STRATEGIES = [
    ("Situation", "Embed in a real-world scenario requiring full analysis",
     "In a reactor cooling system monitoring Temperature, does the observed trend "
     "indicate an unsafe thermal drift that would trigger an automatic shutdown?"),
    ("Constraints", "Add requirements that force using all computed values",
     "If operational safety requires the trend slope to be below 0.05 AND noise "
     "strength below 0.3, does Temperature satisfy both constraints simultaneously?"),
    ("Deepen", "Increase breadth or depth of the question",
     "Beyond the overall trend, how do the noise characteristics and any local "
     "anomalies interact to affect the reliability of the trend estimate?"),
    ("Concretize", "Replace abstract concepts with specific formulas or thresholds",
     "Given that stability requires std dev $< 5.0$ and trend slope $< |0.02|$, "
     "does Temperature meet these specifications with std~$=3.1$ and slope~$=0.01$?"),
    ("Complex Reasoning", "Require multi-step computation chains",
     "Calculate the ratio of local event amplitude to the global std dev. "
     "If this ratio exceeds 3.0, classify it as an outlier. Is the spike at position 234 an outlier?"),
    ("Deductive Reasoning", "Yes/No with formal logical conditions",
     "If a metric is `stable' when (a) trend is steady, (b) noise $< 1.0$, and "
     "(c) no local events exist, is Temperature stable?"),
    ("Causal Reasoning", "Multiple-choice causal analysis",
     "Which factor most likely explains the observed pattern: "
     "(A) seasonal periodicity, (B) sensor drift, (C) external disturbance, or (D) random noise?"),
]


def generate():
    lines = []

    lines.append("% Auto-generated TSEvol demo — paste into Overleaf")
    lines.append("% Requires: booktabs, xcolor, tcolorbox, amsmath")
    lines.append("% \\usepackage{booktabs, xcolor, tcolorbox, amsmath}")
    lines.append("")

    # ── Section: Overview ──────────────────────────────────────────────
    lines.append("\\subsection{TSEvol: Evolutionary QA Augmentation}")
    lines.append("")
    lines.append("TSEvol takes seed question--answer pairs generated from time series data "
                 "and evolves them into harder, more diverse variants using an LLM-driven "
                 "evolutionary pipeline. Each evolved QA must ground its answer in the "
                 "\\emph{entire} chain-of-thought reasoning block from the seed, ensuring "
                 "no information is discarded.")
    lines.append("")

    # ── Figure: Pipeline ───────────────────────────────────────────────
    lines.append("\\subsubsection{Pipeline Overview}")
    lines.append("")
    lines.append("\\begin{enumerate}")
    lines.append("  \\item \\textbf{Seed Generation}: Initial QA pairs are generated from "
                 "time series via domain-specific generators (description, correlation, "
                 "clustering, etc.), each with a structured \\texttt{<think>} block.")
    lines.append("  \\item \\textbf{Selection}: 30\\% of seeds are randomly selected for evolution.")
    lines.append("  \\item \\textbf{Batched Evolution}: Seeds are submitted to an LLM with a "
                 "\\emph{strategy catalog}. The LLM picks a strategy and rewrites the QA pair.")
    lines.append("  \\item \\textbf{Validation}: A validator checks that the evolved answer "
                 "references all lines of the original \\texttt{<think>} block and contains "
                 "no leaked numeric values in the question.")
    lines.append("  \\item \\textbf{Retry \\& Balance}: Failed or rejected samples re-enter "
                 "the pool. Filled strategies are removed from the catalog. Up to 10 rounds "
                 "run until all 7 strategies have equal representation.")
    lines.append("\\end{enumerate}")
    lines.append("")

    # ── Table: Strategies ──────────────────────────────────────────────
    lines.append("\\subsubsection{Evolution Strategies}")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{The 7 TSEvol evolution strategies. Each strategy transforms a "
                 "seed QA pair in a different way while preserving full reasoning grounding.}")
    lines.append("\\label{tab:tsevol-strategies}")
    lines.append("\\begin{tabular}{@{} l p{9cm} @{}}")
    lines.append("\\toprule")
    lines.append("Strategy & Description \\\\")
    lines.append("\\midrule")
    for name, desc, _ in STRATEGIES:
        lines.append(f"\\textsc{{{name}}} & {desc} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # ── Example: Before / After ────────────────────────────────────────
    lines.append("\\subsubsection{Example: Seed $\\rightarrow$ Evolved}")
    lines.append("")

    # Seed box
    lines.append("\\begin{tcolorbox}[title={Seed QA (description task)}, "
                 "colback=gray!5, colframe=gray!50, fonttitle=\\bfseries\\small]")
    lines.append("\\small")
    lines.append("\\textbf{Think Block:}")
    lines.append("\\begin{verbatim}")
    lines.append("<think>")
    lines.append("Metric: Temperature")
    lines.append("- Trend: steady (slope = 0.01)")
    lines.append("- Noise: smooth (strength = 0.15)")
    lines.append("- No seasonal component detected")
    lines.append("- Local event: spike at pos 234, amplitude 12.3")
    lines.append("- Mean: 45.2, Std: 3.1, Range: [38.1, 52.0]")
    lines.append("</think>")
    lines.append("\\end{verbatim}")
    lines.append("\\textbf{Question:} Describe the overall characteristics of Temperature.\\\\")
    lines.append("\\textbf{Answer:} Temperature shows a steady trend with low noise. "
                 "There is a notable spike at position 234 with amplitude 12.3. "
                 "No seasonal pattern is present.")
    lines.append("\\end{tcolorbox}")
    lines.append("")

    # Evolved examples
    examples = [
        ("Situation",
         "In a chemical reactor where Temperature must remain within $\\pm 10$\\textdegree C "
         "of the setpoint (45\\textdegree C) to prevent runaway reactions, does the observed "
         "spike at position 234 pose a safety risk?",
         "The spike amplitude is 12.3, pushing the peak to 52.0 (range confirms max = 52.0). "
         "Deviation from setpoint: $52.0 - 45.2 = 6.8$\\textdegree C $< 10$\\textdegree C "
         "$\\Rightarrow$ within tolerance. The steady trend (slope = 0.01) and low noise "
         "(strength = 0.15) confirm no sustained drift. \\textbf{No immediate safety risk}, "
         "but the spike warrants investigation."),
        ("Concretize",
         "Given that an anomaly is defined as any local event with amplitude $> 3\\sigma$ "
         "(where $\\sigma$ is the standard deviation), does the spike at position 234 qualify?",
         "Standard deviation $\\sigma = 3.1$. Threshold: $3 \\times 3.1 = 9.3$. "
         "Spike amplitude: 12.3 $> 9.3$ $\\Rightarrow$ \\textbf{Yes}, the spike qualifies "
         "as an anomaly under the $3\\sigma$ rule."),
        ("Deductive Reasoning",
         "A metric is classified as ``quiet stable'' if ALL of the following hold: "
         "(a) trend is steady, (b) noise strength $< 0.3$, (c) no local events with "
         "amplitude $> 10$. Is Temperature quiet stable?",
         "(a) Trend = steady (slope 0.01) $\\checkmark$. "
         "(b) Noise = 0.15 $< 0.3$ $\\checkmark$. "
         "(c) Spike amplitude = 12.3 $> 10$ $\\times$. "
         "Condition (c) fails $\\Rightarrow$ \\textbf{No}, Temperature is not quiet stable."),
    ]

    for strat, q, a in examples:
        color = "blue" if strat == "Situation" else ("teal" if strat == "Concretize" else "violet")
        lines.append(f"\\begin{{tcolorbox}}[title={{Evolved: \\textsc{{{strat}}} strategy}}, "
                     f"colback={color}!3, colframe={color}!40, fonttitle=\\bfseries\\small]")
        lines.append("\\small")
        lines.append(f"\\textbf{{Question:}} {q}\\\\[4pt]")
        lines.append(f"\\textbf{{Answer:}} {a}")
        lines.append("\\end{tcolorbox}")
        lines.append("\\vspace{2pt}")
        lines.append("")

    # ── Key properties ─────────────────────────────────────────────────
    lines.append("\\subsubsection{Key Properties}")
    lines.append("\\begin{itemize}")
    lines.append("  \\item \\textbf{Full grounding}: Every evolved answer must reference "
                 "\\emph{all} values from the think block---no information discarded.")
    lines.append("  \\item \\textbf{No leakage}: Numeric values from the think block must "
                 "not appear verbatim in the evolved question.")
    lines.append("  \\item \\textbf{Strategy balance}: Final dataset contains equal counts "
                 "per strategy (trimmed after all rounds).")
    lines.append("  \\item \\textbf{Batched retry}: Failed samples re-enter the pool with a "
                 "narrowing strategy catalog, focusing effort on underrepresented strategies.")
    lines.append("\\end{itemize}")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate TSEvol demo LaTeX")
    parser.add_argument("-o", "--output", default=None, help="Output .tex file")
    args = parser.parse_args()

    latex = generate()

    out_path = Path(args.output) if args.output else Path("exp/tsevol_demo.tex")
    out_path.write_text(latex)
    print(f"Written to {out_path}", file=__import__("sys").stderr)
    print(latex)


if __name__ == "__main__":
    main()
