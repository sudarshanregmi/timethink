"""NeurIPS-grade figure for OOD compositional reasoning results.

Native palette + dims match scripts/utils/plot_compositional_decomposition.py
so the two figures pair cleanly side-by-side as subfigures.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np
from pathlib import Path as PathLib

# -----------------------------------------------------------------------------
# Tol-inspired palette — matches neurips_bundle/make_figs.py.
# Native models keep their distinct colors (per-model identity); text baselines
# are uniformly muted gray (group identity, not individual identity).
# -----------------------------------------------------------------------------
C_CHATTS = '#B79268'   # warm sand
C_TTSFT  = '#6E9CA8'   # muted teal
C_TTRL   = '#2B3A67'   # deep indigo
C_TEXT   = '#9A9A9A'   # uniform muted gray for the unclear-OOD group
C_GRID   = '#E6E6E6'

DATA = [
    ('Time-MQA',                                 0.500, 'text-baseline'),
    ('Time-R1',                                  0.536, 'text-baseline'),
    ('TimeOmni-1',                               0.530, 'text-baseline'),
    ('ChatTS',                                   0.490, 'native-baseline'),
    (r'\textsc{TimeThink} (SFT)',                0.533, 'ours-sft'),
    (r'\textbf{\textsc{TimeThink} (RL)}',        0.629, 'ours-rl'),
]

GROUP_COLOR = {
    'text-baseline':   C_TEXT,
    'native-baseline': C_CHATTS,
    'ours-sft':        C_TTSFT,
    'ours-rl':         C_TTRL,
}

NATIVE_GROUPS = {'native-baseline', 'ours-sft', 'ours-rl'}

# Bracket aesthetic.
NATIVE_COLOR = '#1A7F37'   # forest green — strictly fair
TEXT_COLOR   = '#7A7A7A'   # gray — caveat


def _draw_latex_brace(ax, x, y_bot, y_top, color: str,
                      rule_pt: float = 32.0, fontsize: float = 12.0) -> None:
    """Real LaTeX `}` rendered via `text.usetex`.

    A `\\rule{0pt}{Hpt}` of height `rule_pt` is wrapped in
    `\\left.\\right\\}` so the brace auto-scales to that height.
    `(x, ymid)` is the position of the brace's vertical center.
    """
    ymid = 0.5 * (y_top + y_bot)
    expr = rf'$\left.\rule{{0pt}}{{{rule_pt}pt}}\right\}}$'
    ax.text(x, ymid, expr,
            ha='left', va='center',
            fontsize=fontsize, color=color,
            clip_on=False, zorder=6)


def main() -> None:
    mpl.rcParams.update({
        'text.usetex':       True,
        'text.latex.preamble': r'\usepackage{mathptmx}',
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'Times', 'STIX', 'DejaVu Serif'],
        'mathtext.fontset':  'stix',
        'pdf.fonttype':      42,
        'ps.fonttype':       42,
        'font.size':         8.5,
        'axes.titlesize':    9.0,
        'axes.labelsize':    8.5,
        'xtick.labelsize':   7.6,
        'ytick.labelsize':   7.8,
        'legend.fontsize':   7.6,
        'axes.linewidth':    0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size':  2.2,
        'ytick.major.size':  2.2,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.spines.left':  False,
        'axes.edgecolor':    '#444',
        'axes.labelcolor':   '#222',
        'xtick.color':       '#444',
        'ytick.color':       '#444',
        'text.color':        '#222',
    })

    # Group + sort within each group ascending.
    text_rows   = sorted([r for r in DATA if r[2] not in NATIVE_GROUPS],
                         key=lambda r: r[1])
    native_rows = sorted([r for r in DATA if r[2] in     NATIVE_GROUPS],
                         key=lambda r: r[1])

    GAP = 0.9
    text_y   = list(range(len(text_rows)))
    native_y = [len(text_rows) + GAP + i for i in range(len(native_rows))]

    rows   = text_rows + native_rows
    ys     = text_y + native_y
    names  = [r[0] for r in rows]
    scores = [r[1] for r in rows]
    groups = [r[2] for r in rows]
    colors = [GROUP_COLOR[g] for g in groups]

    fig, ax = plt.subplots(figsize=(3.4, 1.6), dpi=200)

    # Faint group-background shading.
    text_y_min, text_y_max = min(text_y) - 0.45, max(text_y) + 0.45
    nat_y_min,  nat_y_max  = min(native_y) - 0.45, max(native_y) + 0.45
    ax.axhspan(text_y_min, text_y_max, color=TEXT_COLOR,   alpha=0.05, zorder=0)
    ax.axhspan(nat_y_min,  nat_y_max,  color=NATIVE_COLOR, alpha=0.05, zorder=0)

    bars = ax.barh(ys, scores, color=colors,
                   edgecolor='#333', linewidth=0.4, height=0.62, zorder=3)

    ax.set_yticks(ys)
    ax.set_yticklabels(names)

    # Winner highlight.
    winner_idx = int(np.argmax(scores))
    bars[winner_idx].set_path_effects([
        pe.withStroke(linewidth=2.2, foreground=C_TTRL + '40'),
    ])

    # Score labels.
    for i, (y, s) in enumerate(zip(ys, scores)):
        is_winner = (i == winner_idx)
        kwargs = dict(va='center', ha='left',
                      fontsize=7.4, color='#222',
                      fontweight='bold' if is_winner else 'normal',
                      zorder=10)
        if is_winner:
            kwargs['path_effects'] = [pe.withStroke(linewidth=2.4, foreground='white')]
        ax.text(s + 0.005, y, f'{s:.3f}', **kwargs)

    # Real LaTeX curly braces (text.usetex=True; \left.\rule\right\}).
    BRACE_X = 0.78
    LABEL_X = 0.86
    _draw_latex_brace(ax, BRACE_X, nat_y_min, nat_y_max,
                      color=NATIVE_COLOR, rule_pt=24.0, fontsize=12.0)
    ax.text(LABEL_X, 0.5 * (nat_y_min + nat_y_max),
            'no train–test \nleakage',
            ha='left', va='center', fontsize=8.0, color=NATIVE_COLOR,
            fontweight='bold', linespacing=1.2, clip_on=False, zorder=6)

    _draw_latex_brace(ax, BRACE_X, text_y_min, text_y_max,
                      color=TEXT_COLOR, rule_pt=24.0, fontsize=12.0)
    ax.text(LABEL_X, 0.5 * (text_y_min + text_y_max),
            'possible partial\n train–test leakage',
            ha='left', va='center', fontsize=7.6, color=TEXT_COLOR,
            fontstyle='italic', linespacing=1.2, clip_on=False, zorder=6)

    ax.set_xlim(0.0, 1.18)
    #ax.set_xlabel('Overall performance across tasks', labelpad=4)
    ax.set_xticks(np.arange(0.0, 0.751, 0.1))
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', linestyle=':', linewidth=0.5, color='#bbb', zorder=0)

    plt.subplots_adjust(top=0.96, bottom=0.20, left=0.30, right=0.99)

    out_dir = PathLib('figures')
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / 'ood_main_results.pdf'
    png = out_dir / 'ood_main_results.png'
    fig.savefig(pdf, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(png, bbox_inches='tight', pad_inches=0.04, dpi=300)
    print(f'wrote {pdf}')
    print(f'wrote {png}')


if __name__ == '__main__':
    main()
