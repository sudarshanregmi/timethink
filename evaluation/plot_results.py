import json
import os
import re
import textwrap
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

SEED = 42
DATA_PATH = "newdata/test.jsonl"
OUTPUT_DIR = "figures"
random.seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

THINK = {
    "clustering":  {"F1": 0.8298,  "Reasoning": 0.7746},
    "correlation": {"Binary Acc": 0.8321, "Reasoning": 0.6769},
    "description": {"Reasoning": 0.8180, "Overall": 0.7898},
    "yes_no":      {"Binary Acc": 0.9184, "Reasoning": 0.7721},
}
NONTHINK = {
    "clustering":  {"F1": 0.5524,  "Reasoning": 0.6256},
    "correlation": {"Binary Acc": 0.7006, "Reasoning": 0.5562},
    "description": {"Reasoning": 0.8123, "Overall": 0.7845},
    "yes_no":      {"Binary Acc": 0.8337, "Reasoning": 0.7063},
}

CLUSTER_GTLEN_F1 = {
    0:  (0.9111, 0.8288),
    1:  (0.5430, 0.1623),
    2:  (0.7181, 0.2810),
    3:  (0.8101, 0.3802),
    4:  (0.8205, 0.3878),
    5:  (0.8602, 0.5496),
    6:  (0.8850, 0.5426),
    7:  (0.8835, 0.5108),
    8:  (0.8491, 0.5414),
    9:  (0.9005, 0.7913),
    10: (0.9868, 0.9327),
}
CLUSTER_GTLEN_N = {
    0: 742, 1: 78, 2: 372, 3: 251, 4: 199,
    5: 132, 6: 59,  7: 14,  8: 14,  9: 13,  10: 4,
}

# Description sub-metric breakdown
#   key -> (think_score, nonthink_score)
DESC_SUB = {
    "Trend\n(Cat)":        (0.8967, 0.8752),
    "Trend\n(Num)":        (0.7268, 0.7361),
    "Trend Num\n(Amp)":    (0.6938, 0.7000),
    "Trend Num\n(Pos)":    (0.7434, 0.7542),
    "Noise\n(Cat)":        (0.8246, 0.8123),
    "Noise\n(Num/Amp)":    (0.8872, 0.8860),
}

REASONING = {
    "Clustering":  (0.7746, 0.6256),
    "Correlation": (0.6769, 0.5562),
    "Description": (0.8180, 0.8123),
    "Yes/No":      (0.7721, 0.7063),
}

SERIES_COLORS = [
    "#2196F3", "#E91E63", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#FF5722", "#795548",
]
THINK_CLR    = "#1565C0"
NONTHINK_CLR = "#B71C1C"

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
})

TASK_INFO = {
    "clustering": {
        "fig_title": "Clustering",
        "subtitle":  "Group time series by similar global shape / trend",
        "n_label":   "1882 samples",
    },
    "correlation": {
        "fig_title": "Correlation",
        "subtitle":  "Identify correlated local events across multiple series",
        "n_label":   "3210 samples",
    },
    "description": {
        "fig_title": "Description",
        "subtitle":  "Describe trend, noise, seasonality and local events",
        "n_label":   "5254 samples",
    },
    "yes_no": {
        "fig_title": "Yes / No",
        "subtitle":  "Binary question about time series properties",
        "n_label":   "1654 samples",
    },
}

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def group_by_task(data):
    g = defaultdict(list)
    for r in data:
        g[r["eval_type"]].append(r)
    return g

_TS_PATTERN = re.compile(r"\[\s*[-\d.,\s]{40,}\]")
_TS_TAG     = re.compile(r"<ts(?:\s[^>]*)?>.*?</ts>", re.DOTALL | re.IGNORECASE)

def clean_question(text):
    """Remove everything up to and including the last <ts>...</ts> block."""
    matches = list(_TS_TAG.finditer(text))
    if matches:
        text = text[matches[-1].end():].strip()
    else:
        for tag in ["</ts>", "<ts"]:
            idx = text.lower().rfind(tag)
            if idx != -1:
                end = text.find(">", idx) + 1 if tag == "<ts" else idx + len("</ts>")
                text = text[end:].strip()
                break
    text = _TS_PATTERN.sub("[<series>]", text)
    return text.strip()

def parse_answer(text):
    """Return (think_content, answer_content). think_content is None if absent."""
    text = str(text).strip()
    m = re.match(r"^<think>(.*?)</think>\s*(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, text

def wrap(text, width=42):
    return "\n".join(textwrap.wrap(text, width=width))

def draw_ts(ax, timeseries, task_type, eval_metadata):
    n = len(timeseries)
    metric = eval_metadata.get("target_metric", "")
    mode   = eval_metadata.get("mode", "")

    for i, s in enumerate(timeseries):
        arr = np.array(s, dtype=float)
        lw  = 1.6 if n == 1 else 1.1
        ax.plot(arr, color=SERIES_COLORS[i % len(SERIES_COLORS)],
                linewidth=lw, alpha=0.85, label=f"S{i + 1}")

    if n > 1:
        ax.legend(fontsize=7, ncol=min(n, 6), loc="upper right", framealpha=0.5)

    if n == 1:
        arr = np.array(timeseries[0], dtype=float)
        ax.fill_between(range(len(arr)), arr.min(), arr,
                        alpha=0.08, color=SERIES_COLORS[0])

    info = TASK_INFO[task_type]
    ax.set_title(info["subtitle"] + f"\n({info['n_label']})",
                 fontsize=9.5, color="#444", style="italic", pad=6)

    xlabel = f"Time   ·   metric: {metric}" if metric else "Time"
    if mode:
        xlabel += f"   ·   mode: {mode}"
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Value", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, linestyle="--")

def draw_right(fig, gs_right, task_type, sample, qa_height_in):
    ax_qa = fig.add_subplot(gs_right[0])
    ax_tb = fig.add_subplot(gs_right[1])

    ax_qa.axis("off")

    q             = clean_question(sample.get("input", ""))
    think_text, answer_text = parse_answer(str(sample.get("output", "")))
    q_wrapped     = wrap(q)
    a_wrapped     = wrap(answer_text)
    think_wrapped = wrap(think_text) if think_text else None

    NORMAL_LS = 1.4;  NORMAL_PT = 11.0
    SMALL_LS  = 1.3;  SMALL_PT  = 9.0
    LABEL_PT  = 12.5
    normal_line_h = NORMAL_PT * NORMAL_LS / 72 / qa_height_in
    small_line_h  = SMALL_PT  * SMALL_LS  / 72 / qa_height_in
    label_h       = LABEL_PT  * 1.5       / 72 / qa_height_in
    gap           = normal_line_h * 1.2

    y = 0.98

    ax_qa.text(0, y, "Example Question:", fontsize=LABEL_PT, fontweight="bold",
               va="top", transform=ax_qa.transAxes, color="#212121")
    y -= label_h
    ax_qa.text(0, y, q_wrapped, fontsize=NORMAL_PT, va="top",
               transform=ax_qa.transAxes, color="#555",
               style="italic", linespacing=NORMAL_LS)
    y -= (q_wrapped.count("\n") + 1) * normal_line_h + gap

    ax_qa.text(0, y, "Ground Truth Answer:", fontsize=LABEL_PT, fontweight="bold",
               va="top", transform=ax_qa.transAxes, color="#212121")
    y -= label_h

    if think_wrapped:
        ax_qa.text(0, y, think_wrapped, fontsize=SMALL_PT, va="top",
                   transform=ax_qa.transAxes, color="#999",
                   style="italic", linespacing=SMALL_LS)
        y -= (think_wrapped.count("\n") + 1) * small_line_h + gap * 0.6

    ax_qa.text(0, y, a_wrapped, fontsize=NORMAL_PT, va="top",
               transform=ax_qa.transAxes, color="#1B5E20", linespacing=NORMAL_LS)

    ax_tb.axis("off")

    think_m   = THINK[task_type]
    nonthink_m = NONTHINK[task_type]

    col_labels = ["Metric", "Thinking ▶", "Non-Thinking"]
    rows = []
    for metric, val in think_m.items():
        nt = nonthink_m[metric]
        nt_str = f"{nt:.4f}" if isinstance(nt, float) else "—"
        rows.append([metric, f"{val:.4f}", nt_str])

    tbl = ax_tb.table(
        cellText=rows,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
        bbox=[0.0, 0.05, 1.0, 0.82],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    # header
    for j in range(3):
        c = tbl[0, j]
        c.set_facecolor("#263238")
        c.set_text_props(color="white", fontweight="bold")

    # data rows
    for i in range(1, len(rows) + 1):
        tbl[i, 0].set_facecolor("#F5F5F5")
        tbl[i, 1].set_facecolor("#E3F2FD")
        tbl[i, 1].set_text_props(color=THINK_CLR, fontweight="bold")
        tbl[i, 2].set_facecolor("#FFEBEE")
        tbl[i, 2].set_text_props(color=NONTHINK_CLR)

    ax_tb.set_title("Model Performance", fontsize=9.5,
                    fontweight="bold", pad=6, color="#212121")

def make_task_figure(task_type, sample):
    # Estimate content size to set figure height
    q           = clean_question(sample.get("input", ""))
    think_text, answer_text = parse_answer(str(sample.get("output", "")))
    q_lines     = wrap(q).count("\n") + 1
    a_lines     = wrap(answer_text).count("\n") + 1
    think_lines = (wrap(think_text).count("\n") + 1) if think_text else 0

    # Height in inches per line type
    NORMAL_IN = 7.5 * 1.4 / 72   # ~0.146
    SMALL_IN  = 5.5 * 1.3 / 72   # ~0.099
    LABEL_IN  = 9.0 * 1.5 / 72   # ~0.188

    content_in = (2 * LABEL_IN + q_lines * NORMAL_IN +
                  think_lines * SMALL_IN + a_lines * NORMAL_IN + 1.5)
    QA_FRAC = 0.50  # QA sub-axes is ~50% of figure height
    fig_h = max(8.0, min(content_in / QA_FRAC * 1.2, 40.0))
    qa_height_in = fig_h * QA_FRAC

    fig = plt.figure(figsize=(16, fig_h))
    fig.suptitle(
        TASK_INFO[task_type]["fig_title"],
        fontsize=15, fontweight="bold", x=0.5, y=1.02,
    )

    outer = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[1.7, 1], wspace=0.08,
    )
    ax_ts   = fig.add_subplot(outer[0])
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1], hspace=0.12, height_ratios=[1.3, 1],
    )

    draw_ts(ax_ts, sample["timeseries"], task_type,
            sample.get("eval_metadata", {}))
    draw_right(fig, gs_right, task_type, sample, qa_height_in)

    path = os.path.join(OUTPUT_DIR, f"{task_type}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def make_clustering_gtlen():
    lens   = sorted(CLUSTER_GTLEN_F1.keys())
    t_f1s  = [CLUSTER_GTLEN_F1[l][0] for l in lens]
    nt_f1s = [CLUSTER_GTLEN_F1[l][1] for l in lens]
    ns     = [CLUSTER_GTLEN_N[l]     for l in lens]

    x = np.arange(len(lens))
    w = 0.38

    fig, ax = plt.subplots(figsize=(13, 5))
    bars_t  = ax.bar(x - w / 2, t_f1s,  w, color=THINK_CLR,    alpha=0.82, zorder=3, label="Thinking")
    bars_nt = ax.bar(x + w / 2, nt_f1s, w, color=NONTHINK_CLR, alpha=0.72, zorder=3, label="Non-Thinking")

    # annotate with sample count above the taller bar
    for i, (t, nt, n) in enumerate(zip(t_f1s, nt_f1s, ns)):
        top = max(t, nt) + 0.015
        ax.text(x[i], top, f"n={n}", ha="center", va="bottom",
                fontsize=7, color="#333")

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(lens)
    ax.set_xlabel("Ground-truth cluster size  (# series in GT cluster)", fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=10)
    ax.set_title(
        "Clustering — F1 by Ground-Truth Cluster Size",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.28, linestyle="--", zorder=0)
    ax.axhline(THINK["clustering"]["F1"],    color=THINK_CLR,    linestyle=":",
               linewidth=1.4, label=f"Thinking overall  F1={THINK['clustering']['F1']:.4f}")
    ax.axhline(NONTHINK["clustering"]["F1"], color=NONTHINK_CLR, linestyle="--",
               linewidth=1.4, label=f"Non-Think overall F1={NONTHINK['clustering']['F1']:.4f}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "clustering_by_gtlen.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def make_description_breakdown():
    labels     = list(DESC_SUB.keys())
    think_vals = [v[0] for v in DESC_SUB.values()]
    nt_vals    = [v[1] for v in DESC_SUB.values()]

    x = np.arange(len(labels))
    w = 0.34

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, think_vals, w, color=THINK_CLR,    alpha=0.82, label="Thinking",     zorder=3)
    ax.bar(x + w / 2, nt_vals,    w, color=NONTHINK_CLR, alpha=0.72, label="Non-Thinking", zorder=3)

    for xi, val in zip(x - w / 2, think_vals):
        ax.text(xi, val + 0.012, f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, color=THINK_CLR, fontweight="bold")
    for xi, val in zip(x + w / 2, nt_vals):
        ax.text(xi, val + 0.012, f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, color=NONTHINK_CLR, fontweight="bold")

    # shade regions to group Trend vs Noise
    ax.axvspan(-0.55, 3.45, alpha=0.04, color="#2196F3", label="Trend questions")
    ax.axvspan(3.55, 5.55, alpha=0.04, color="#E91E63", label="Noise questions")
    ax.axvline(3.5, color="gray", linestyle="--", linewidth=0.8)

    ax.text(1.5, 1.03, "Trend", ha="center", fontsize=10, color="#1565C0",
            fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(4.5, 1.03, "Noise", ha="center", fontsize=10, color="#B71C1C",
            fontweight="bold", transform=ax.get_xaxis_transform())

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(
        "Description Task — Score by Sub-category\n"
        "Cat = Categorical  ·  Num = Numerical  ·  Amp = Amplitude  ·  Pos = Position",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.28, linestyle="--", zorder=0)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "description_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def make_reasoning_overview():
    tasks    = list(REASONING.keys())
    t_scores = [v[0] for v in REASONING.values()]
    nt_scores = [v[1] for v in REASONING.values()]

    x = np.arange(len(tasks))
    w = 0.34

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, t_scores,  w, color=THINK_CLR,    alpha=0.82, label="Thinking",     zorder=3)
    ax.bar(x + w / 2, nt_scores, w, color=NONTHINK_CLR, alpha=0.72, label="Non-Thinking", zorder=3)

    for xi, val in zip(x - w / 2, t_scores):
        ax.text(xi, val + 0.012, f"{val:.4f}", ha="center", va="bottom",
                fontsize=9.5, color=THINK_CLR, fontweight="bold")
    for xi, val in zip(x + w / 2, nt_scores):
        ax.text(xi, val + 0.012, f"{val:.4f}", ha="center", va="bottom",
                fontsize=9.5, color=NONTHINK_CLR, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=12)
    ax.set_ylabel("Reasoning Score", fontsize=11)
    ax.set_title("Reasoning Score — All Tasks", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.28, linestyle="--", zorder=0)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "reasoning_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def make_summary_overview():
    """Bar chart of primary metrics per task in a single figure."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    fig.suptitle("Thinking Model vs Non-Thinking",
                 fontsize=13, fontweight="bold", y=1.02)

    tasks = ["clustering", "correlation", "description", "yes_no"]
    titles = ["Clustering", "Correlation", "Description", "Yes / No"]
    task_colors = [THINK_CLR, "#00695C", "#6A1B9A", "#E65100"]

    for ax, task, title, clr in zip(axes, tasks, titles, task_colors):
        metrics  = list(THINK[task].keys())
        t_vals   = [THINK[task][m]    for m in metrics]
        nt_vals  = [NONTHINK[task][m] for m in metrics]

        x = np.arange(len(metrics))
        w = 0.38

        ax.bar(x - w / 2, t_vals,  w, color=clr,         alpha=0.82, label="Thinking",     zorder=3)
        ax.bar(x + w / 2, nt_vals, w, color=NONTHINK_CLR, alpha=0.72, label="Non-Thinking", zorder=3)

        for xi, val in zip(x - w / 2, t_vals):
            ax.text(xi, val + 0.012, f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=clr, fontweight="bold")
        for xi, val in zip(x + w / 2, nt_vals):
            ax.text(xi, val + 0.012, f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=NONTHINK_CLR, fontweight="bold")

        ax.set_ylim(0, 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--", zorder=0)
        ax.tick_params(labelsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Score", fontsize=10)

    # unified legend
    legend_elements = [
        Patch(facecolor=THINK_CLR,    alpha=0.82, label="Thinking model"),
        Patch(facecolor=NONTHINK_CLR, alpha=0.72, label="Non-Thinking model"),
    ]
    fig.legend(handles=legend_elements, fontsize=10, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(OUTPUT_DIR, "summary_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def main():
    print(f"Loading data from: {DATA_PATH}")
    data   = load_jsonl(DATA_PATH)
    groups = group_by_task(data)
    print(f"Loaded {len(data)} samples across {len(groups)} task types.\n")

    # one figure per task
    for task in ["clustering", "correlation", "description", "yes_no"]:
        samples = groups.get(task, [])
        if not samples:
            print(f"[WARN] No samples found for task: {task}")
            continue
        sample = random.choice(samples)
        make_task_figure(task, sample)

    # breakdown / summary figures
    make_clustering_gtlen()
    make_description_breakdown()
    make_reasoning_overview()
    make_summary_overview()

    print(f"\nDone. All figures saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
