"""Build a paper figure showcasing how RLVR aligns the multi-segment trend
description with ground truth, while SFT and SFT-no-think collapse to
oversimplified single-segment claims (or get the direction wrong).

Three cherry-picked examples (high RL reasoning_score, low SFT/sft_nothink):
  - test_len_near idx=24 (Crop Rotation Metrics — single metric)
  - test_len_far idx=761 (User Demographics + Sentiment Analysis — 2 metrics)
  - test_len_near idx=605 (Subscription Rates — single metric)

Output: figures/description_alignment.pdf  +  figures/description_alignment.tex
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
os.chdir(str(_REPO))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Trend colour palette — colourblind-friendly, paper-printable
TREND_COLORS = {
    "increase": "#2E7D32",   # green
    "decrease": "#C62828",   # red
    "keep steady": "#9E9E9E", # grey
    "steady": "#9E9E9E",
    "stable": "#9E9E9E",
}

# Hard-coded parsed segments per model per example (verified from manual
# inspection of generated_answer.json).
EXAMPLES = [
    {
        "label": "(a) Crop Rotation Metrics — Agriculture",
        "ds": "test_len_near",
        "parent_idx": 24,
        "metric_idx_in_ts": 1,
        "metric_name": "Crop Rotation Metrics",
        "seq_len": 568,
        "gt_segments": [("increase", 0, 386), ("decrease", 386, 567)],
        "gt_overall": "increase",
        "preds": {
            "sft_nothink": {
                "segments": [("increase", 0, 284), ("decrease", 284, 567)],
                "overall": "decrease",  # WRONG direction
                "summary": "Wrong overall direction: claims decreasing.",
            },
            "sft": {
                "segments": [("increase", 0, 567)],
                "overall": "increase",
                "summary": "Single segment — misses the late decrease.",
            },
            "rl": {
                "segments": [("increase", 0, 567)],
                "overall": "increase",
                "summary": "Correct overall direction.",
            },
        },
        "score": {"sft_nothink": 0.20, "sft": 0.45, "rl": 0.85},
    },
    {
        "label": "(b) Sentiment Analysis — Social Media",
        "ds": "test_len_far",
        "parent_idx": 761,
        "metric_idx_in_ts": 0,
        "metric_name": "Sentiment Analysis",
        "seq_len": 1289,
        "gt_segments": [("decrease", 0, 322), ("increase", 322, 610), ("decrease", 610, 1288)],
        "gt_overall": "decrease",
        "preds": {
            "sft_nothink": {
                "segments": [("decrease", 0, 1288)],
                "overall": "decrease",
                "summary": "Single segment — misses both internal turning points.",
            },
            "sft": {
                "segments": [("decrease", 0, 1288)],
                "overall": "decrease",
                "summary": "Single segment — misses both internal turning points.",
            },
            "rl": {
                "segments": [("decrease", 0, 433), ("increase", 433, 921), ("decrease", 921, 1288)],
                "overall": "decrease",
                "summary": "Recovers the 3-segment structure, breakpoints align.",
            },
        },
        "score": {"sft_nothink": 0.30, "sft": 0.30, "rl": 0.85},
    },
    {
        "label": "(c) Subscription Rates — Media",
        "ds": "test_len_near",
        "parent_idx": 605,
        "metric_idx_in_ts": 4,
        "metric_name": "Subscription Rates",
        "seq_len": 736,
        "gt_segments": [("decrease", 0, 344), ("increase", 344, 735)],
        "gt_overall": "decrease",
        "preds": {
            "sft_nothink": {
                "segments": [("decrease", 0, 367), ("increase", 367, 735)],
                "overall": "increase",  # contradicts segments
                "summary": "Wrong overall direction (claims increase) despite identifying segments.",
            },
            "sft": {
                "segments": [("decrease", 0, 735)],
                "overall": "increase",
                "summary": "Single segment + contradictory overall direction.",
            },
            "rl": {
                "segments": [("decrease", 0, 311), ("keep steady", 311, 501), ("increase", 501, 735)],
                "overall": "decrease",
                "summary": "Recovers V-shape structure with breakpoints aligned.",
            },
        },
        "score": {"sft_nothink": 0.20, "sft": 0.20, "rl": 0.85},
    },
]


def load_ts(ds, parent_idx, metric_idx):
    with open(f"data/{ds}.jsonl") as f:
        line = f.readlines()[parent_idx]
    src = json.loads(line)
    return np.array(src["timeseries"][metric_idx])


def draw_segment_bar(ax, segments, y_center, height=0.6, label=None, label_x=None):
    """Draw a horizontal bar split into colored segments (one per trend)."""
    for trend, s, e in segments:
        c = TREND_COLORS.get(trend, "#777777")
        ax.barh(y_center, e - s, left=s, height=height, color=c, edgecolor="white", linewidth=0.5)
    if label and label_x is not None:
        ax.text(label_x, y_center, label, ha="right", va="center",
                fontsize=10, fontweight="bold")


def draw_example(fig, gs_row, ex):
    seq_len = ex["seq_len"]
    # Two columns: timeseries plot (left, wide) + segment ribbons (right, narrow)
    ax_ts = fig.add_subplot(gs_row[0, 0])
    ax_seg = fig.add_subplot(gs_row[0, 1])

    # === LEFT: timeseries plot with GT segment shading ===
    ts = load_ts(ex["ds"], ex["parent_idx"], ex["metric_idx_in_ts"])
    x = np.arange(len(ts))
    ax_ts.plot(x, ts, color="#222222", linewidth=0.9, zorder=3)

    # Shade by GT segment for visual cue
    for trend, s, e in ex["gt_segments"]:
        c = TREND_COLORS.get(trend, "#777777")
        ax_ts.axvspan(s, e, color=c, alpha=0.08, zorder=1)

    # Vertical lines at GT segment boundaries
    for trend, s, e in ex["gt_segments"][:-1]:
        ax_ts.axvline(e, color="#444444", linewidth=0.7, linestyle="--", alpha=0.6, zorder=2)

    ax_ts.set_title(f"{ex['label']}", fontsize=11, fontweight="bold", loc="left", pad=4)
    ax_ts.set_xlim(0, seq_len - 1)
    ax_ts.set_xlabel("timestep", fontsize=9)
    ax_ts.set_ylabel(ex["metric_name"], fontsize=9)
    ax_ts.tick_params(labelsize=8)
    for sp in ax_ts.spines.values():
        sp.set_linewidth(0.5)

    # === RIGHT: segment-alignment ribbons ===
    rows = [
        ("Ground Truth", ex["gt_segments"], ex["score"].get("gt", None)),
        ("ChatTS-style (sft-nothink)", ex["preds"]["sft_nothink"]["segments"], ex["score"]["sft_nothink"]),
        ("+ SFT (think)", ex["preds"]["sft"]["segments"], ex["score"]["sft"]),
        ("+ RL (RLVR)", ex["preds"]["rl"]["segments"], ex["score"]["rl"]),
    ]
    y_positions = list(range(len(rows), 0, -1))  # top→bottom
    for y, (name, segs, score) in zip(y_positions, rows):
        draw_segment_bar(ax_seg, segs, y, height=0.62)
        # Label on the left of the ribbon
        ax_seg.text(-0.02 * seq_len, y, name, ha="right", va="center",
                    fontsize=9, fontweight="bold" if "RL" in name else "normal")
        # Score on the right
        if score is not None:
            ax_seg.text(seq_len * 1.02, y, f"{score:.2f}",
                        ha="left", va="center", fontsize=9,
                        color="#1B5E20" if score >= 0.7 else "#B71C1C",
                        fontweight="bold")

    ax_seg.set_xlim(-0.20 * seq_len, seq_len * 1.13)
    ax_seg.set_ylim(0.4, len(rows) + 0.6)
    ax_seg.set_xlabel("timestep", fontsize=9)
    ax_seg.set_yticks([])
    ax_seg.tick_params(axis="x", labelsize=8)
    for sp in ax_seg.spines.values():
        sp.set_visible(False)
    ax_seg.spines["bottom"].set_visible(True)
    ax_seg.spines["bottom"].set_linewidth(0.5)
    ax_seg.set_title("predicted segments  (reasoning_score →)", fontsize=10, loc="left", pad=4)


def main():
    figs_dir = Path("figures")
    figs_dir.mkdir(exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.5,
    })

    fig = plt.figure(figsize=(13, 9.5))
    outer = GridSpec(3, 1, figure=fig, hspace=0.55, top=0.94, bottom=0.10, left=0.06, right=0.97)
    for i, ex in enumerate(EXAMPLES):
        sub = outer[i].subgridspec(1, 2, width_ratios=[1.0, 1.4], wspace=0.1)
        draw_example(fig, sub, ex)

    # Legend
    legend_handles = [
        mpatches.Patch(color=TREND_COLORS["increase"], label="increase"),
        mpatches.Patch(color=TREND_COLORS["decrease"], label="decrease"),
        mpatches.Patch(color=TREND_COLORS["keep steady"], label="keep steady"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=3, fontsize=10,
        frameon=False, bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        "RLVR aligns trend descriptions with ground-truth segment structure",
        fontsize=13, fontweight="bold", y=0.985,
    )

    out_pdf = figs_dir / "description_alignment.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    out_png = figs_dir / "description_alignment.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
