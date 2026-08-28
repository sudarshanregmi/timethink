"""NeurIPS-grade figure for the compositional decomposition story.

Native palette + dims match scripts/utils/plot_ood_results.py so the two
figures pair cleanly side-by-side as subfigures.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# Palette + rcParams: copied verbatim from neurips_bundle/make_figs.py.
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    'text.usetex':       True,
    'text.latex.preamble': r'\usepackage{mathptmx}',
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Times', 'STIX', 'DejaVu Serif'],
    'mathtext.fontset':  'stix',
    'font.size':         8.0,
    'axes.titlesize':    8.5,
    'axes.labelsize':    8.0,
    'xtick.labelsize':   7.2,
    'ytick.labelsize':   7.6,
    'legend.fontsize':   7.4,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  2.2,
    'ytick.major.size':  2.2,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.edgecolor':    '#444',
    'axes.labelcolor':   '#222',
    'xtick.color':       '#444',
    'ytick.color':       '#444',
    'text.color':        '#222',
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'savefig.dpi':       600,
})

C_SFT  = '#6E9CA8'   # muted teal
C_RL   = '#2B3A67'   # deep indigo
C_GRID = '#E6E6E6'

# -----------------------------------------------------------------------------
# Data — Table tab:compositional-summary in the paper.
# -----------------------------------------------------------------------------
buckets  = ['Easy', 'Hard', 'OOD']
atom_sft = [0.59, 0.64, 0.57]
atom_rl  = [0.70, 0.75, 0.68]
comp_sft = [0.69, 0.55, 0.52]
comp_rl  = [0.72, 0.73, 0.61]

# Δ values from the paper table (rounded mean of per-task gains).
DELTA_A = [0.11, 0.10, 0.11]
DELTA_C = [0.04, 0.18, 0.09]

HARD_IDX = 1


def _draw_panel(ax, sft, rl, deltas, title: str,
                highlight_idx: int | None = None) -> None:
    y = np.arange(len(buckets))

    if highlight_idx is not None:
        ax.axhspan(highlight_idx - 0.42, highlight_idx + 0.42,
                   color=C_RL, alpha=0.06, zorder=0)

    for i in range(len(buckets)):
        ax.hlines(y=i, xmin=sft[i], xmax=rl[i],
                  color=C_RL, lw=0.8, alpha=0.35, zorder=1)

    ax.scatter(sft, y, s=22, marker='s',
               facecolor='white', edgecolor=C_SFT, linewidths=1.0, zorder=3)
    ax.scatter(rl,  y, s=34, marker='D',
               facecolor=C_RL, edgecolor=C_RL, linewidths=0.5, zorder=4)

    # Δ annotation — compact, no subscript (title disambiguates panel).
    for i, (r, d) in enumerate(zip(rl, deltas)):
        is_hl = (highlight_idx is not None and i == highlight_idx)
        fw = 'bold' if is_hl else 'normal'
        ax.annotate(rf'+{d:.2f}',
                    xy=(r, i),
                    xytext=(5, 0), textcoords='offset points',
                    ha='left', va='center',
                    fontsize=7.0, color=C_RL,
                    fontweight=fw)

    ax.set_xlim(0.42, 0.92)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.set_xticklabels(['0.5', '0.6', '0.7', '0.8'])
    #ax.set_xlabel('Accuracy', labelpad=3)
    ax.set_title(title, fontweight='bold', pad=3, color='#222')
    ax.xaxis.grid(True, color=C_GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.6), sharey=True)

    _draw_panel(axes[0], atom_sft, atom_rl, DELTA_A,
                title=r'\textbf{Atomic skills} ($\Delta_a$)')

    _draw_panel(axes[1], comp_sft, comp_rl, DELTA_C,
                title=r'\textbf{Composition} ($\Delta_c$)',
                highlight_idx=HARD_IDX)

    axes[0].set_yticks(np.arange(len(buckets)))
    axes[0].set_yticklabels(buckets, fontsize=7.8)
    axes[0].invert_yaxis()

    # Compact legend at top.
    sft_handle = mlines.Line2D([], [], color=C_SFT, marker='s', linestyle='None',
                                markerfacecolor='white', markeredgewidth=1.0,
                                ms=4.4, label=r'\textsc{TimeThink} (SFT)')
    rl_handle  = mlines.Line2D([], [], color=C_RL, marker='D', linestyle='None',
                                markerfacecolor=C_RL, ms=4.8,
                                label=r'\textbf{\textsc{TimeThink} (RL)}')
    leg = fig.legend(handles=[sft_handle, rl_handle],
                     loc='upper center', bbox_to_anchor=(0.5, 1.04),
                     ncol=2, frameon=False, handlelength=0.9,
                     handletextpad=0.3, columnspacing=1.2,
                     fontsize=7.4, borderaxespad=0.0)
    for txt, col in zip(leg.get_texts(), [C_SFT, C_RL]):
        txt.set_color(col)
    leg.get_texts()[1].set_fontweight('bold')

    plt.subplots_adjust(left=0.12, right=0.97, top=0.78, bottom=0.24, wspace=0.34)

    out_dir = Path('figures')
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / 'fig_compositional_decomposition.pdf'
    png = out_dir / 'fig_compositional_decomposition.png'
    fig.savefig(pdf, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(png, bbox_inches='tight', pad_inches=0.04, dpi=300)
    plt.close(fig)
    print(f'wrote {pdf}')
    print(f'wrote {png}')


if __name__ == '__main__':
    main()
