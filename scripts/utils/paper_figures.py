"""
paper_figures.py — NeurIPS method-section figure generator for ChatTS.

================================================================================
Purpose
================================================================================
The paper claims: "RL composes atomic skills taught in SFT into new compositional
tasks, and this composition transfers to held-out OOD probes."

The method section needs figures that make that story intuitive at a glance.
This script builds them from the real pipeline data — the same samples the model
saw in SFT / RL / held-out OOD eval. No hand-constructed toys.

Two curated storylines back the claim:

STORY A — PRIMITIVE REUSE  (one atomic primitive, progressively more context)
  SFT atomic : atomic_max_position
               "At what index is the maximum?"   [teaches argmax primitive]
               SFT 0.79 → RL 0.80   (preserved)
  RL task    : rl_max_in_first_half
               "Is the maximum in the first half?"   [argmax + half boundary]
               SFT 0.60 → RL 0.85   (+0.25 — RL learns the composition)
  OOD probe  : ood_peak_in_longest_segment
               "Is the peak inside the longest trend segment?"
               [argmax + segment duration — held out from RL]

STORY B — TWO PRIMITIVES COMBINED
  SFT atomic  : atomic_max_position   (argmax primitive)
                SFT 0.79 → RL 0.80
  SFT atomic  : atomic_min_position   (argmin primitive)
                SFT 0.80 → RL 0.77
  RL task     : rl_extrema_same_half
                "Are max and min in the same half?"   [argmax + argmin + halves]
                SFT 0.48 → RL 0.69   (+0.21)
  OOD probe   : ood_max_before_min
                "Does the max come before the min?"   [argmax + argmin + ordering]
                SFT 0.51 → RL 0.74   (+0.23 — composition transfers)

STORY C — SINGLE-CHANNEL PRIMITIVE GENERALIZES TO MULTI-CHANNEL (MTS)
  Motivation: the atomic primitives are *taught* as "find-X on this one
  timeseries." But real evaluation requires reasoning across multiple
  channels simultaneously (cross-metric comparisons, alignments, joint
  behaviors). Story C shows the primitive transfers from UTS to MTS
  without ever being explicitly taught cross-metric.
  SFT atomic  : atomic_max_position   (argmax on ONE metric at a time)
                SFT 0.79 → RL 0.80
  SFT atomic  : atomic_global_mean    (mean on ONE metric at a time)
                SFT 0.80 → RL 0.84
  RL task     : rl_cross_asymmetric_behavior
                "Is one metric more asymmetric than another?"
                SFT 0.19 → RL 0.40   (+0.21 — primitive extends to cross-metric)
  OOD probe   : ood_cross_extrema_alignment
                "Do the extrema of these metrics align in time?"
                [held-out — applies argmax/argmin across channels]
                SFT 0.54 → RL 0.62   (+0.08)

Rejected candidate — ood_max_in_highest_mean_quarter:
  Numerically flat (0.55 → 0.55). Including it as a motivating example would
  misrepresent the method. Kept in the paper's quantitative tables (as a
  failure case with a documented "missing primitive" diagnosis), but NOT used
  for method-section illustration.

================================================================================
What each figure contains
================================================================================

For Atomic and RL categories:
  [top]     TS plot with annotations (argmax marker, half boundary, etc.)
  [bottom]  Question text block, then the ground-truth response verbatim.
            <think>…</think> block rendered in a muted gray box to make
            the reasoning/answer separation visible.

For OOD category:
  [top]     TS plot with annotations for the features the OOD question touches.
  [bottom]  Question only — these are held-out probes, so we don't show a
            response the model hasn't been trained to emit.

================================================================================
Outputs (per example idx)
================================================================================
{out_dir}/story_{A|B}/{category}/{eval_type}_{idx}.pdf
    Combined figure (TS plot + Q/R text). Drop-in for the paper figure.

{out_dir}/story_{A|B}/{category}/{eval_type}_{idx}_plot.pdf
    TS plot only. Use when you want to typeset Q/R with native LaTeX.

{out_dir}/story_{A|B}/{category}/{eval_type}_{idx}.tex
    LaTeX snippet with question and response, ready to include with
    \\input{}. Escapes special chars; renders <think> as a shaded listing.

================================================================================
Typical usage
================================================================================
  # Produce 5 of each panel for both stories (useful for appendix):
  python scripts/utils/paper_figures.py --story all --n 5

  # One polished showcase per panel (method section body):
  python scripts/utils/paper_figures.py --story all --n 1 --seed 7

  # Just story A, lots of variants for picking-and-choosing:
  python scripts/utils/paper_figures.py --story A --n 20 \\
      --out-dir figures/paper_appendix

================================================================================
Design notes (so you can extend this later)
================================================================================
1. eval_metadata in the JSONL files is LEAN — just verdict/length/tsevol.
   We compute annotations (argmax, argmin, halves, segments) directly from
   the timeseries array, not from metadata. This is robust to metadata drift.

2. We preferentially filter to UTS (1-channel) samples for cleaner
   single-line plots. If fewer than --n UTS samples exist for an eval_type,
   we fall back to MTS and extract the primary channel by matching the
   question's metric name against the channel labels in `input`.

3. Style is NeurIPS-friendly: vector PDF, serif body, minimal color palette
   (accent + two neutral grays), 9-10pt captions, 7-8pt TS tick labels.
   Figure width fits a 2-column layout (~3.3in wide) by default but you can
   toggle --wide for full-textwidth renders.

4. Segment detection for ood_peak_in_longest_segment: uses a simple
   sign-of-difference rule with a minimum-run filter. This is approximate
   (not the exact segment boundaries the reward function uses) but gives a
   visually-sensible "longest trend segment" annotation that matches what
   the question is asking.

5. To add a new storyline: append to STORIES below. Each story is a list
   of (category, eval_type) panels; category ∈ {atomic, rl, ood}.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Matplotlib: configure BEFORE importing pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"

# Where each panel's samples come from.
# Atomic and RL use train_sft / train_rl because the `output` field there
# is the IDEAL training target (the response we want the model to produce).
# OOD uses test.jsonl because that's where the probes live — we only show
# the question + TS (no response) since it's held-out.
DEFAULT_SOURCES = {
    "atomic_max_position":           DATA / "train_sft.jsonl",
    "atomic_min_position":           DATA / "train_sft.jsonl",
    "atomic_global_mean":            DATA / "train_sft.jsonl",
    "rl_max_in_first_half":          DATA / "train_rl.jsonl",
    "rl_extrema_same_half":          DATA / "train_rl.jsonl",
    "rl_cross_asymmetric_behavior":  DATA / "train_rl.jsonl",
    "ood_peak_in_longest_segment":   DATA / "test.jsonl",
    "ood_max_before_min":            DATA / "test.jsonl",
    "ood_cross_extrema_alignment":   DATA / "test.jsonl",
}

# Eval types that inherently use multiple metrics. These render differently
# (stacked mini-subplots for the referenced channels, not a single TS plot).
CROSS_METRIC_TYPES = {
    "rl_cross_asymmetric_behavior",
    "ood_cross_extrema_alignment",
}


# --------------------------------------------------------------------------
# Verdict-consistency filters
# --------------------------------------------------------------------------
# Training data can drift from its label (TSEvol paraphrase can outlive the
# original TS, resulting in verdict/metadata mismatch with the actual
# timeseries). For method-section figures we want samples where the TS
# visibly agrees with the stated verdict. Each entry is a callable
#   (sample, primary_ts) -> bool
# returning True when the sample is internally consistent. Types without
# an entry accept everything (no validation available).

def _verdict_yes_no(md_verdict) -> Optional[bool]:
    if isinstance(md_verdict, str):
        v = md_verdict.strip().lower()
        if v in ("yes", "true"):  return True
        if v in ("no", "false"):  return False
    if isinstance(md_verdict, bool):
        return md_verdict
    return None


def _consistent_atomic_max_position(s, ts):
    v = s.eval_metadata.get("verdict")
    try: v = int(v)
    except Exception: return True
    return int(np.argmax(ts)) == v


def _consistent_atomic_min_position(s, ts):
    v = s.eval_metadata.get("verdict")
    try: v = int(v)
    except Exception: return True
    return int(np.argmin(ts)) == v


def _consistent_atomic_global_mean(s, ts):
    v = s.eval_metadata.get("verdict")
    try: v = float(v)
    except Exception: return True
    mu = float(np.mean(ts))
    denom = max(abs(v), abs(mu), 1.0)
    return abs(mu - v) / denom < 0.05   # within 5% rel tol


def _consistent_max_in_first_half(s, ts):
    want = _verdict_yes_no(s.eval_metadata.get("verdict"))
    if want is None: return True
    return (int(np.argmax(ts)) < len(ts) // 2) == want


def _consistent_extrema_same_half(s, ts):
    want = _verdict_yes_no(s.eval_metadata.get("verdict"))
    if want is None: return True
    half = len(ts) // 2
    same = (int(np.argmax(ts)) < half) == (int(np.argmin(ts)) < half)
    return same == want


def _consistent_max_before_min(s, ts):
    want = _verdict_yes_no(s.eval_metadata.get("verdict"))
    if want is None: return True
    return (int(np.argmax(ts)) < int(np.argmin(ts))) == want


# Cross-metric consistency — checks argmax positions across the channels
def _consistent_cross_extrema_alignment(s, channels, tol: int = 10):
    want = _verdict_yes_no(s.eval_metadata.get("verdict"))
    if want is None or not channels or len(channels) < 2:
        return True
    argmaxes = [int(np.argmax(ts)) for _, _, ts in channels[:2]]
    aligned = abs(argmaxes[0] - argmaxes[1]) <= tol
    return aligned == want


VERDICT_CONSISTENCY = {
    "atomic_max_position":   _consistent_atomic_max_position,
    "atomic_min_position":   _consistent_atomic_min_position,
    "atomic_global_mean":    _consistent_atomic_global_mean,
    "rl_max_in_first_half":  _consistent_max_in_first_half,
    "rl_extrema_same_half":  _consistent_extrema_same_half,
    "ood_max_before_min":    _consistent_max_before_min,
}

# Cross-metric checkers take (sample, channels) rather than (sample, ts).
CROSS_VERDICT_CONSISTENCY = {
    "ood_cross_extrema_alignment": _consistent_cross_extrema_alignment,
}

# Per-panel human-readable caption (printed as the TS plot title).
PANEL_TITLES = {
    "atomic_max_position":          "Atomic: find argmax (single metric)",
    "atomic_min_position":          "Atomic: find argmin (single metric)",
    "atomic_global_mean":           "Atomic: compute mean (single metric)",
    "rl_max_in_first_half":         "RL composition: argmax + half boundary",
    "rl_extrema_same_half":         "RL composition: argmax + argmin + halves",
    "rl_cross_asymmetric_behavior": "RL cross-metric: asymmetry across channels",
    "ood_peak_in_longest_segment":  "OOD held-out: argmax + segment duration",
    "ood_max_before_min":           "OOD held-out: argmax + argmin + ordering",
    "ood_cross_extrema_alignment":  "OOD held-out: extrema alignment across channels",
}

STORIES = {
    "A": {
        "title": "Primitive reused across contexts (argmax)",
        "panels": [
            ("atomic", "atomic_max_position"),
            ("rl",     "rl_max_in_first_half"),
            ("ood",    "ood_peak_in_longest_segment"),
        ],
    },
    "B": {
        "title": "Two primitives combined (argmax + argmin)",
        "panels": [
            ("atomic", "atomic_max_position"),
            ("atomic", "atomic_min_position"),
            ("rl",     "rl_extrema_same_half"),
            ("ood",    "ood_max_before_min"),
        ],
    },
    "C": {
        "title": "Single-channel primitive generalizes to multi-channel (MTS)",
        "panels": [
            ("atomic", "atomic_max_position"),
            ("atomic", "atomic_global_mean"),
            ("rl",     "rl_cross_asymmetric_behavior"),
            ("ood",    "ood_cross_extrema_alignment"),
        ],
    },
}

# Palette — muted, print-safe.
COLOR_TS       = "#2C3E50"   # main timeseries line
COLOR_MAX      = "#C0392B"   # argmax marker
COLOR_MIN      = "#2980B9"   # argmin marker
COLOR_HALF     = "#7F8C8D"   # half-boundary line
COLOR_SEGMENT  = "#F39C12"   # longest-segment shade
COLOR_QBG      = "#F7F4EE"   # question background
COLOR_THINKBG  = "#ECEFF1"   # <think> block background
COLOR_ANSBG    = "#E8F5E9"   # final answer background


# --------------------------------------------------------------------------
# Data loading & filtering
# --------------------------------------------------------------------------

@dataclass
class Sample:
    eval_type: str
    input_text: str
    output_text: str
    timeseries: list          # list[list[float]] or list[float]
    eval_metadata: dict
    source_path: Path
    source_idx: int           # line number in the source file

    # Derived — single-channel mode (Stories A/B)
    primary_channel: Optional[int] = None     # which channel to plot
    primary_ts: Optional[np.ndarray] = None
    metric_name: Optional[str] = None         # e.g. "Wind Speed"

    # Derived — cross-metric mode (Story C)
    # list of (channel_idx, channel_label, ts) for the metrics referenced
    # in the question. Populated when the eval_type is in CROSS_METRIC_TYPES.
    cross_channels: Optional[list[tuple[int, str, np.ndarray]]] = None


def _extract_metric_name(sample: Sample) -> Optional[str]:
    """Pull the metric name that the QUESTION asks about.

    Strategy (in priority order):
      1. Match channel labels against the question text (text after the
         last `<ts>`-terminated clause). The question is authoritative
         about which metric is being asked about.
      2. Fallback: match against the response text.
      3. Fallback: first channel label in the header.

    This avoids two historical bugs:
      - greedy regex matches like "Visibility found in the first half" when
        the question uses "of" as a connective, not as "of <metric>";
      - MTS samples where response mentions a different metric than the
        question (paraphrase drift); we prefer the question's metric so
        the plotted channel matches the asked-about metric.
    """
    labels = _parse_channel_labels(sample.input_text)
    if not labels:
        return None
    # 1) question first — authoritative about what's being asked
    q = _question_text_raw(sample)
    hits = _find_channels_in_text(labels, q)
    if hits:
        return labels[hits[0]]
    # 2) response
    hits = _find_channels_in_text(labels, sample.output_text)
    if hits:
        return labels[hits[0]]
    # 3) first channel in header
    return labels[0]


_CHANNEL_LABEL_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+\.\s*)?([A-Z][A-Za-z \-&/]+?)\s+is of length",
)
# OOD benchmark phrasing (UTS): "This is a metric called <NAME> collected from..."
_OOD_METRIC_RE = re.compile(
    r"This is (?:a )?metric called ([A-Z][A-Za-z \-&/]+?)\s+collected from",
)


def _parse_channel_labels(input_text: str) -> list[str]:
    """Return ordered list of channel labels from the input header.

    Handles both training-data phrasing (`Foo is of length 256: <ts>;`)
    and OOD-benchmark UTS phrasing (`This is a metric called Foo
    collected from ...`).
    """
    labels = [m.group(1).strip() for m in _CHANNEL_LABEL_RE.finditer(input_text)]
    if labels:
        return labels
    # OOD UTS fallback
    m = _OOD_METRIC_RE.search(input_text)
    return [m.group(1).strip()] if m else []


def _find_channels_in_text(labels: list[str], text: str) -> list[int]:
    """Find channel labels that are literally mentioned in the given text.

    Returns indices preserving order of appearance in `text` (so the first
    metric mentioned in the question appears first in our list).
    """
    hits: list[tuple[int, int]] = []   # (position_in_text, channel_idx)
    for idx, lab in enumerate(labels):
        if not lab:
            continue
        # word-boundary match, case-sensitive (labels are Capitalized)
        pattern = re.compile(r"\b" + re.escape(lab) + r"\b")
        m = pattern.search(text)
        if m:
            hits.append((m.start(), idx))
    hits.sort()
    # preserve order, deduplicate
    seen = set(); out = []
    for _, idx in hits:
        if idx not in seen:
            seen.add(idx); out.append(idx)
    return out


def _pick_primary_channel(sample: Sample) -> None:
    """Set sample.primary_channel and sample.primary_ts.

    For UTS (1 channel) → channel 0.
    For MTS → match the metric name against the channel labels in `input`.
    Fall back to channel 0 if matching fails.
    """
    ts = sample.timeseries
    if not (isinstance(ts, list) and ts):
        sample.primary_channel = None
        sample.primary_ts = None
        return

    is_mts = isinstance(ts[0], list)
    if not is_mts:
        sample.primary_channel = 0
        sample.primary_ts = np.asarray(ts, dtype=float)
        return

    # MTS: match metric name against the numbered channel list in `input`.
    metric = sample.metric_name or _extract_metric_name(sample)
    if metric:
        labels = _parse_channel_labels(sample.input_text)
        for idx, lab in enumerate(labels):
            if lab == metric:
                sample.primary_channel = idx
                sample.primary_ts = np.asarray(ts[idx], dtype=float)
                sample.metric_name = metric
                return

    # Fall back: channel 0
    sample.primary_channel = 0
    sample.primary_ts = np.asarray(ts[0], dtype=float)


def _pick_cross_channels(sample: Sample, max_channels: int = 2) -> None:
    """Populate sample.cross_channels for cross-metric types.

    Parses the question to find the metrics it asks about, then extracts
    the matching channel indices + labels + data. If fewer than max_channels
    metrics are named in the question, falls back to the first
    max_channels channels.
    """
    ts = sample.timeseries
    if not (isinstance(ts, list) and ts and isinstance(ts[0], list)):
        # UTS fallback — treat primary as the one cross channel
        sample.cross_channels = [(sample.primary_channel or 0,
                                   sample.metric_name or "channel 0",
                                   sample.primary_ts)]
        return

    labels = _parse_channel_labels(sample.input_text)
    question = _question_text_raw(sample)
    mentioned = _find_channels_in_text(labels, question)

    # If the question names only 1 or 0 metrics, try the response text
    if len(mentioned) < max_channels:
        more = _find_channels_in_text(labels, sample.output_text)
        for idx in more:
            if idx not in mentioned:
                mentioned.append(idx)

    # Still short? pad with first unused channels
    if len(mentioned) < max_channels:
        for idx in range(len(labels)):
            if idx not in mentioned:
                mentioned.append(idx)
            if len(mentioned) >= max_channels:
                break

    picked = mentioned[:max_channels]
    sample.cross_channels = [
        (idx, labels[idx] if idx < len(labels) else f"channel {idx}",
         np.asarray(ts[idx], dtype=float))
        for idx in picked
    ]


_TS_SENTINEL_RE = re.compile(
    r"(?:is of length|with length of)\s+\d+\s*:\s*<ts>\s*[;.]"
)


def _question_text_raw(sample: Sample) -> str:
    """Strip the input preamble and return just the question.

    Handles three header styles:
      - train MTS  : "Foo is of length 256: <ts>; Bar is of length 256: <ts>."
      - train UTS  : "Foo is of length 256: <ts>."
      - OOD UTS    : "You are ... This is a metric called Foo collected from
                      Bar with length of 256: <ts>."
    In all cases the question follows the last <ts>-terminated clause.
    """
    inp = sample.input_text.strip()
    m = list(_TS_SENTINEL_RE.finditer(inp))
    if m:
        q = inp[m[-1].end():].strip()
        if q:
            return q
    return inp


def load_samples(eval_type: str,
                 *,
                 n: int,
                 seed: int,
                 prefer_uts: Optional[bool] = None,
                 source_path: Optional[Path] = None) -> list[Sample]:
    """Return up to n samples of the given eval_type.

    prefer_uts: if None (default), auto — UTS for single-channel panels,
                MTS for CROSS_METRIC_TYPES. Pass True/False to override.
    Deterministic for a given (eval_type, n, seed) triple.
    """
    is_cross = eval_type in CROSS_METRIC_TYPES
    if prefer_uts is None:
        prefer_uts = not is_cross

    path = source_path or DEFAULT_SOURCES[eval_type]
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    candidates: list[Sample] = []
    with path.open() as f:
        for idx, line in enumerate(f):
            e = json.loads(line)
            if e.get("eval_type") != eval_type:
                continue
            s = Sample(
                eval_type=eval_type,
                input_text=e.get("input", ""),
                output_text=e.get("output", ""),
                timeseries=e.get("timeseries", []),
                eval_metadata=e.get("eval_metadata", {}),
                source_path=path,
                source_idx=idx,
            )
            s.metric_name = _extract_metric_name(s)
            _pick_primary_channel(s)
            if s.primary_ts is None or len(s.primary_ts) < 8:
                continue
            if is_cross:
                _pick_cross_channels(s)
                # For cross-metric panels we need at least 2 real channels
                if not s.cross_channels or len(s.cross_channels) < 2:
                    continue
                # Verdict-consistency check for cross types
                cross_check = CROSS_VERDICT_CONSISTENCY.get(eval_type)
                if cross_check is not None and not cross_check(s, s.cross_channels):
                    continue
            else:
                # Verdict-consistency check for single-channel types
                check = VERDICT_CONSISTENCY.get(eval_type)
                if check is not None and not check(s, s.primary_ts):
                    continue
            # Q/R metric-agreement check (catches paraphrase-drift samples):
            # if both the question and the response mention channel labels,
            # and they disagree, the sample is corrupted — skip.
            labels_all = _parse_channel_labels(s.input_text)
            if labels_all:
                q_hits = _find_channels_in_text(labels_all,
                                                 _question_text_raw(s))
                r_hits = _find_channels_in_text(labels_all, s.output_text)
                if q_hits and r_hits and q_hits[0] != r_hits[0]:
                    continue
            candidates.append(s)

    if prefer_uts:
        uts = [s for s in candidates
               if s.timeseries and not isinstance(s.timeseries[0], list)]
        if len(uts) >= n:
            candidates = uts
    elif is_cross:
        # For cross panels prefer samples where the question explicitly
        # names ≥2 metrics (more informative plots).
        rich = []
        for s in candidates:
            labels = _parse_channel_labels(s.input_text)
            mentioned = _find_channels_in_text(labels, _question_text_raw(s))
            if len(mentioned) >= 2:
                rich.append(s)
        if len(rich) >= n:
            candidates = rich

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


# --------------------------------------------------------------------------
# Segment detection (needed for ood_peak_in_longest_segment)
# --------------------------------------------------------------------------

def detect_trend_segments(ts: np.ndarray, min_len: int = 8) -> list[tuple[int, int, str]]:
    """Approximate monotonic-run segmentation.

    Returns a list of (start_idx, end_idx_exclusive, direction) where
    direction ∈ {'up', 'down', 'flat'}. Adjacent same-direction runs are
    merged. Runs shorter than min_len are absorbed into the neighbour.

    This is approximate — it doesn't match the reward-side segmenter
    exactly — but it's close enough for a method-section illustration.
    """
    if len(ts) < 2:
        return [(0, len(ts), "flat")]

    diffs = np.diff(ts)
    # Smooth-ish: treat very small diffs as flat
    thr = max(1e-9, 0.01 * (np.max(ts) - np.min(ts)))
    sign = np.where(diffs > thr, 1,
           np.where(diffs < -thr, -1, 0))

    # Collapse runs
    raw_runs = []
    start = 0
    for i in range(1, len(sign)):
        if sign[i] != sign[i - 1]:
            raw_runs.append((start, i, int(sign[i - 1])))
            start = i
    raw_runs.append((start, len(sign), int(sign[-1])))

    # Convert to ts-indices (run [a, b) over diff = ts[a:b+1])
    segs = [(a, b + 1, s) for (a, b, s) in raw_runs]

    # Merge short runs into neighbour
    def merge_short(segments):
        changed = True
        while changed and len(segments) > 1:
            changed = False
            for i, (a, b, s) in enumerate(segments):
                if b - a < min_len:
                    # merge with the longer neighbour
                    left = segments[i - 1] if i > 0 else None
                    right = segments[i + 1] if i + 1 < len(segments) else None
                    target = None
                    if left and right:
                        target = i - 1 if (left[1] - left[0]) >= (right[1] - right[0]) else i + 1
                    elif left:
                        target = i - 1
                    elif right:
                        target = i + 1
                    if target is None:
                        continue
                    if target < i:
                        la, lb, ls = segments[target]
                        segments[target] = (la, b, ls)
                    else:
                        ra, rb, rs = segments[target]
                        segments[target] = (a, rb, rs)
                    segments.pop(i)
                    changed = True
                    break
        return segments

    segs = merge_short(segs)

    def name(s: int) -> str:
        return "up" if s > 0 else ("down" if s < 0 else "flat")

    return [(a, b, name(s)) for (a, b, s) in segs]


# --------------------------------------------------------------------------
# Matplotlib style
# --------------------------------------------------------------------------

def _set_style(wide: bool = False) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# --------------------------------------------------------------------------
# Annotators — per-eval-type visual highlights
# --------------------------------------------------------------------------

def _annot_argmax(ax, ts: np.ndarray, *, label: str = "argmax") -> int:
    i = int(np.argmax(ts))
    ax.scatter([i], [ts[i]], color=COLOR_MAX, s=55, zorder=5,
               edgecolor="white", linewidth=1.0, label=label)
    ax.axvline(i, color=COLOR_MAX, linewidth=0.7, linestyle=":", alpha=0.55)
    return i


def _annot_argmin(ax, ts: np.ndarray, *, label: str = "argmin") -> int:
    i = int(np.argmin(ts))
    ax.scatter([i], [ts[i]], color=COLOR_MIN, s=55, zorder=5,
               edgecolor="white", linewidth=1.0, label=label, marker="v")
    ax.axvline(i, color=COLOR_MIN, linewidth=0.7, linestyle=":", alpha=0.55)
    return i


def _annot_half_boundary(ax, ts: np.ndarray) -> int:
    mid = len(ts) // 2
    ax.axvline(mid, color=COLOR_HALF, linewidth=1.0, linestyle="--",
               alpha=0.8, label="half boundary")
    y_lo, y_hi = ax.get_ylim()
    ax.axvspan(0, mid, color=COLOR_HALF, alpha=0.04)
    ax.text(mid * 0.5, y_hi, "first half", ha="center", va="top",
            fontsize=7, color=COLOR_HALF, alpha=0.9)
    ax.text(mid + (len(ts) - mid) * 0.5, y_hi, "second half",
            ha="center", va="top", fontsize=7, color=COLOR_HALF, alpha=0.9)
    return mid


def _annot_longest_segment(ax, ts: np.ndarray) -> tuple[int, int, str]:
    segments = detect_trend_segments(ts, min_len=max(8, len(ts) // 24))
    if not segments:
        return 0, len(ts), "flat"
    longest = max(segments, key=lambda s: s[1] - s[0])
    a, b, direction = longest
    ax.axvspan(a, b, color=COLOR_SEGMENT, alpha=0.18,
               label=f"longest segment ({direction})")
    # faint vertical lines at all segment boundaries
    for (sa, sb, _) in segments:
        if sa not in (0, a):
            ax.axvline(sa, color=COLOR_SEGMENT, linewidth=0.4,
                       alpha=0.3, linestyle="-")
    return a, b, direction


def _annot_mean_line(ax, ts: np.ndarray, *, label: str = "mean") -> float:
    """Horizontal line at the global mean of the series."""
    mu = float(np.mean(ts))
    ax.axhline(mu, color="#8E44AD", linewidth=1.0, linestyle="--",
               alpha=0.75, label=f"{label} = {mu:.2f}")
    return mu


ANNOTATORS = {
    "atomic_max_position":         lambda ax, ts: _annot_argmax(ax, ts),
    "atomic_min_position":         lambda ax, ts: _annot_argmin(ax, ts),
    "atomic_global_mean":          lambda ax, ts: _annot_mean_line(ax, ts),
    "rl_max_in_first_half": lambda ax, ts: (_annot_argmax(ax, ts),
                                             _annot_half_boundary(ax, ts)),
    "rl_extrema_same_half": lambda ax, ts: (_annot_argmax(ax, ts),
                                             _annot_argmin(ax, ts),
                                             _annot_half_boundary(ax, ts)),
    "ood_peak_in_longest_segment": lambda ax, ts: (_annot_longest_segment(ax, ts),
                                                    _annot_argmax(ax, ts, label="peak (argmax)")),
    "ood_max_before_min":          lambda ax, ts: (_annot_argmax(ax, ts),
                                                    _annot_argmin(ax, ts)),
}


# Per-channel annotator used when drawing each stacked subplot of a
# cross-metric sample. Returns nothing; annotates the given ax in place.
CROSS_CHANNEL_ANNOTATORS = {
    "rl_cross_asymmetric_behavior": lambda ax, ts: (_annot_argmax(ax, ts),
                                                     _annot_argmin(ax, ts),
                                                     _annot_mean_line(ax, ts)),
    "ood_cross_extrema_alignment":  lambda ax, ts: (_annot_argmax(ax, ts),
                                                     _annot_argmin(ax, ts)),
}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _pad_xlim(ax, n: int, frac: float = 0.02) -> None:
    """Expand x-limits by `frac` of the series length on each side so
    markers sitting at index 0 or index n-1 aren't clipped by the axes.
    Uses max(0.6, n*frac) as the absolute pad — keeps short series
    (n=50) readable and long series (n=4000) from looking squashed.
    """
    pad = max(0.6, n * frac)
    ax.set_xlim(-pad, (n - 1) + pad)


def render_ts_plot(ax, sample: Sample) -> None:
    """Draw the primary TS with type-specific annotations (single-metric)."""
    ts = sample.primary_ts
    assert ts is not None
    x = np.arange(len(ts))
    ax.plot(x, ts, color=COLOR_TS, linewidth=1.1, alpha=0.95)
    _pad_xlim(ax, len(ts))
    ax.set_xlabel("timestep", fontsize=8, labelpad=2)
    name = sample.metric_name or f"channel {sample.primary_channel}"
    ax.set_ylabel(name, fontsize=8, labelpad=2)
    ax.set_title(PANEL_TITLES.get(sample.eval_type, sample.eval_type),
                 fontsize=9.5, pad=6)

    annot = ANNOTATORS.get(sample.eval_type)
    if annot is not None:
        annot(ax, ts)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="best", frameon=False,
                  fontsize=7, handlelength=1.2, borderaxespad=0.3)


def render_ts_area(fig, bbox: tuple[float, float, float, float],
                   sample: Sample) -> None:
    """Fill the given figure bbox (x, y, w, h) with the TS visualization.

    Dispatches: single-metric types get one axes; cross-metric types get
    stacked mini-subplots (one per referenced channel) so multiple channels
    are readable side-by-side.
    """
    x, y, w, h = bbox
    if sample.eval_type in CROSS_METRIC_TYPES and sample.cross_channels:
        channels = sample.cross_channels
        n = len(channels)
        # Reserve absolute inches for title clearance, then convert to
        # fig-fraction using the actual figure height.
        fig_h_in = fig.get_figheight()
        title_pad_in   = 0.34       # title + clearance
        between_pad_in = 0.18       # gap between stacked subplots
        title_pad      = title_pad_in / fig_h_in
        pad_between    = between_pad_in / fig_h_in
        avail = h - title_pad
        per_h = (avail - pad_between * (n - 1)) / n
        # Draw figure-level title centered above the TS subplots.
        fig.text(x + w / 2, y + h - 0.20 / fig_h_in,
                 PANEL_TITLES.get(sample.eval_type, sample.eval_type),
                 ha="center", va="top", fontsize=9.5, fontweight="normal")
        annotator = CROSS_CHANNEL_ANNOTATORS.get(sample.eval_type)
        for i, (ch_idx, ch_name, ts) in enumerate(channels):
            # Top-to-bottom layout
            ax_y = y + h - title_pad - (i + 1) * per_h - i * pad_between
            ax = fig.add_axes([x, ax_y, w, per_h])
            ts_x = np.arange(len(ts))
            ax.plot(ts_x, ts, color=COLOR_TS, linewidth=1.0, alpha=0.95)
            _pad_xlim(ax, len(ts))
            ax.set_ylabel(ch_name, fontsize=7.8, labelpad=2)
            if i < n - 1:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                ax.set_xlabel("timestep", fontsize=8, labelpad=2)
            if annotator is not None:
                annotator(ax, ts)
            handles, labels = ax.get_legend_handles_labels()
            if handles and i == 0:
                ax.legend(handles, labels, loc="best", frameon=False,
                          fontsize=6.8, handlelength=1.0, borderaxespad=0.2)
    else:
        ax = fig.add_axes([x, y, w, h - 0.02])
        render_ts_plot(ax, sample)


# --------------------------------------------------------------------------
# Response rendering (combined figure)
# --------------------------------------------------------------------------

def _split_think_and_answer(output_text: str) -> tuple[str, str]:
    """Return (think_inner, final_after_think). Either may be empty."""
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", output_text, re.DOTALL)
    if think_match:
        inner = think_match.group(1).strip()
        after = output_text[think_match.end():].strip()
    else:
        inner = ""
        after = output_text.strip()
    return inner, after


def _question_text(sample: Sample) -> str:
    """Extract the question sentence(s) from `input` (see _question_text_raw)."""
    inp = sample.input_text.strip()
    m = list(_TS_SENTINEL_RE.finditer(inp))
    if m:
        q = inp[m[-1].end():].strip()
        if q:
            return q
    return inp


def _draw_text_block(ax, y_top: float, height: float, lines: list[str],
                     *, bg_color: str,
                     font_family: str = "serif",
                     font_size: float = 8.0,
                     rounded: bool = True,
                     edge_color: str = "#B0BEC5",
                     inner_pad_frac: float = 0.14) -> None:
    """Draw a colored rectangle containing wrapped text. No separate
    label row — the box color and position are the indicator.

    inner_pad_frac: top/bottom padding as a fraction of the block height.
    """
    boxstyle = "round,pad=0.004,rounding_size=0.010" if rounded \
               else "square,pad=0.004"
    rect = FancyBboxPatch((0.01, y_top - height), 0.98, height,
                          boxstyle=boxstyle,
                          transform=ax.transAxes,
                          facecolor=bg_color, edgecolor=edge_color,
                          linewidth=0.6, clip_on=False)
    ax.add_patch(rect)
    body_y = y_top - height * inner_pad_frac
    ax.text(0.025, body_y, "\n".join(lines), transform=ax.transAxes,
            fontsize=font_size, family=font_family, va="top", ha="left",
            linespacing=1.20)


def _wrap_lines(text: str, width: int) -> list[str]:
    """Wrap text preserving explicit newlines."""
    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
            continue
        out.extend(textwrap.wrap(para, width=width) or [""])
    return out


def _wrap_hyphenated(text: str, width: int) -> list[str]:
    """Word-wrap, densely filling lines. Words longer than `width` are
    broken mid-word with a trailing '-' (the reader sees "word-\\nbreak").
    Preserves explicit newlines as hard line breaks.
    """
    out: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
            continue
        words = para.split()
        cur: list[str] = []
        cur_len = 0
        for w in words:
            # If a single word doesn't fit, split it mid-word with '-'.
            if len(w) > width:
                # flush current line first
                if cur:
                    out.append(" ".join(cur))
                    cur, cur_len = [], 0
                remaining = w
                while len(remaining) > width:
                    out.append(remaining[: width - 1] + "-")
                    remaining = remaining[width - 1 :]
                if remaining:
                    cur, cur_len = [remaining], len(remaining)
                continue
            add = len(w) + (1 if cur else 0)
            if cur_len + add > width:
                out.append(" ".join(cur))
                cur, cur_len = [w], len(w)
            else:
                cur.append(w)
                cur_len += add
        if cur:
            out.append(" ".join(cur))
    return out


def _collapse_reasoning(think_inner: str) -> str:
    """Flatten the think block's reasoning into a single paragraph,
    keeping the `answer: X` line on its own (that's the one newline we
    preserve). Empty inner → empty string.
    """
    if not think_inner.strip():
        return ""
    lines = think_inner.strip().split("\n")
    reasoning = [l.strip() for l in lines
                 if l.strip() and not l.strip().lower().startswith("answer:")]
    answer = next((l.strip() for l in lines
                   if l.strip().lower().startswith("answer:")), None)
    flat = " ".join(reasoning)
    if answer:
        return (flat + "\n" + answer) if flat else answer
    return flat


def _rl_response_content(sample: Sample) -> tuple[str, str]:
    """For an RL panel, render the response as its training-rollout shape:
    think block with a [ROLLOUT] placeholder + answer line, and an
    explicit '[No rollout after </think>]' marker in the post-think
    region (RL rollout stops at </think> — see memory/rollout_stop_think).
    """
    verdict = sample.eval_metadata.get("verdict", "...")
    think_body = f"[ROLLOUT]\nanswer: {verdict}"
    final      = "[No rollout after </think>]"
    return think_body, final


def render_combined(sample: Sample, category: str, *, out_pdf: Path,
                    wide: bool = False) -> None:
    """Render the combined figure (TS plot + Q/R text) for one sample.

    Layout — two blocks, no verbose labels:
      [ TS PLOT                                    ]
      [ Q: ... question text ...                  ] (peach background)
      [ <think>                                    ] \
      [ [rollout / reasoning] answer: X            ]  |--> one Response
      [ </think>                                   ]  |    block, split
      [ post-think response OR [No rollout ...]    ] /     visually
    """
    is_cross = sample.eval_type in CROSS_METRIC_TYPES

    width_in = 7.0 if wide else 4.5

    if is_cross:
        n_ch = len(sample.cross_channels or [])
        subplot_h = 1.20 if wide else 1.05
        ts_height_in = 0.40 + (n_ch - 1) * 0.20 + n_ch * subplot_h
    else:
        ts_height_in = 2.1 if wide else 1.85

    text_width_chars = 110 if wide else 62

    FS_BODY = 8.0
    FS_MONO = 7.6
    LH_BODY = FS_BODY * 1.20 / 72.0
    LH_MONO = FS_MONO * 1.20 / 72.0
    PAD_IN  = 0.09   # top/bottom padding inside each box

    # ---------- Question block ----------
    question = _question_text(sample)
    q_lines = _wrap_hyphenated(question, text_width_chars)
    q_body_h = len(q_lines) * LH_BODY
    q_box_h  = q_body_h + 2 * PAD_IN

    blocks: list[dict] = [{
        "kind": "question",
        "lines": q_lines,
        "bg": COLOR_QBG,
        "family": "serif",
        "fontsize": FS_BODY,
        "line_h": LH_BODY,
        "height_in": q_box_h,
    }]

    # ---------- Response block(s) ----------
    # Rendered as TWO adjacent sub-boxes (zero gap between) so they read
    # as one visually-unified Response region with think vs post-think
    # colors doing the signalling.
    if category != "ood":
        if category == "rl":
            think_inner, final = _rl_response_content(sample)
        else:
            think_inner_raw, final = _split_think_and_answer(sample.output_text)
            think_inner = _collapse_reasoning(think_inner_raw)

        # Compose full "think" text including the <think>/</think> tags as
        # visible markers. Single paragraph for reasoning; 'answer:' on its
        # own line (enforced by _collapse_reasoning + RL override).
        think_full = f"<think>\n{think_inner}\n</think>"
        t_lines = _wrap_hyphenated(think_full, text_width_chars)
        f_lines = _wrap_hyphenated(final or "", text_width_chars) if final else []

        t_body_h = len(t_lines) * LH_MONO
        blocks.append({
            "kind": "think",
            "lines": t_lines,
            "bg": COLOR_THINKBG,
            "family": "monospace",
            "fontsize": FS_MONO,
            "line_h": LH_MONO,
            "height_in": t_body_h + 2 * PAD_IN,
        })
        if f_lines:
            f_body_h = len(f_lines) * LH_BODY
            blocks.append({
                "kind": "post",
                "lines": f_lines,
                "bg": COLOR_ANSBG,
                "family": "serif",
                "fontsize": FS_BODY,
                "line_h": LH_BODY,
                "height_in": f_body_h + 2 * PAD_IN,
            })

    # ---------- Figure layout ----------
    TS_TOP_PAD     = 0.28
    TS_BOTTOM_PAD  = 0.40
    GAP_Q_TO_RESP  = 0.09   # gap between Question and think
    GAP_WITHIN_RESP = 0.0   # no gap between think and post-think
    FIG_BOTTOM_PAD = 0.08

    def gap_after(i: int) -> float:
        if i == len(blocks) - 1:
            return 0.0
        cur, nxt = blocks[i]["kind"], blocks[i + 1]["kind"]
        if cur == "question":
            return GAP_Q_TO_RESP
        return GAP_WITHIN_RESP

    text_total = sum(b["height_in"] for b in blocks) \
                 + sum(gap_after(i) for i in range(len(blocks)))
    height_in = TS_TOP_PAD + ts_height_in + TS_BOTTOM_PAD + text_total + FIG_BOTTOM_PAD

    fig = plt.figure(figsize=(width_in, height_in))

    # TS area
    ts_y_in = height_in - TS_TOP_PAD - ts_height_in
    ts_x_in, ts_w_in = 0.11 * width_in, 0.86 * width_in
    render_ts_area(fig,
                   (ts_x_in / width_in, ts_y_in / height_in,
                    ts_w_in / width_in, ts_height_in / height_in),
                   sample)

    # Text axes
    text_top_in   = ts_y_in - TS_BOTTOM_PAD
    text_bottom_in = FIG_BOTTOM_PAD
    ax_tx = fig.add_axes([0.0, text_bottom_in / height_in,
                          1.0, (text_top_in - text_bottom_in) / height_in])
    ax_tx.set_xlim(0, 1); ax_tx.set_ylim(0, 1)
    ax_tx.axis("off")
    text_area_h_in = text_top_in - text_bottom_in

    y = 1.0
    for i, b in enumerate(blocks):
        h_frac = b["height_in"] / text_area_h_in
        _draw_text_block(ax_tx, y_top=y, height=h_frac,
                         lines=b["lines"], bg_color=b["bg"],
                         font_family=b["family"],
                         font_size=b["fontsize"])
        y -= h_frac + gap_after(i) / text_area_h_in

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


def render_plot_only(sample: Sample, *, out_pdf: Path, wide: bool = False) -> None:
    """Render the TS-plot-only PDF (for when you want to typeset Q/R in LaTeX)."""
    is_cross = sample.eval_type in CROSS_METRIC_TYPES
    width_in  = 6.2 if wide else 3.3
    if is_cross:
        n_ch = len(sample.cross_channels or [])
        height_in = (1.0 + 0.9 * n_ch) if wide else (0.8 + 0.75 * n_ch)
    else:
        height_in = 2.2 if wide else 1.9
    fig = plt.figure(figsize=(width_in, height_in))
    # leave top+bottom padding for title/xlabel
    render_ts_area(fig, (0.12, 0.08, 0.86, 0.88), sample)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)


# --------------------------------------------------------------------------
# LaTeX snippet output
# --------------------------------------------------------------------------

_LATEX_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "<": r"\textless{}", ">": r"\textgreater{}",
}


def _tex_escape(s: str) -> str:
    return "".join(_LATEX_ESCAPE.get(c, c) for c in s)


def write_tex_snippet(sample: Sample, category: str, out_tex: Path) -> None:
    """Emit a self-contained LaTeX snippet with question + response boxes."""
    question = _question_text(sample)
    parts = [
        "% Auto-generated by scripts/utils/paper_figures.py",
        f"% eval_type: {sample.eval_type}   source: {sample.source_path.name}:{sample.source_idx}",
        r"\begin{minipage}{\linewidth}",
        r"  \small",
        r"  \textbf{Question:}\\",
        "  " + _tex_escape(question).replace("\n", r" \\ "),
    ]
    if category != "ood":
        think_inner, final = _split_think_and_answer(sample.output_text)
        parts += [
            r"  \par\vspace{2pt}",
            r"  \textbf{\small Expected \texttt{<think>} block:}",
            r"  \begin{quote}\ttfamily\footnotesize",
            "  " + _tex_escape(think_inner).replace("\n", r"\\"),
            r"  \end{quote}",
            r"  \textbf{\small Expected post-think response:}\\",
            "  " + _tex_escape(final),
        ]
    else:
        parts += [
            r"  \par\vspace{2pt}",
            r"  \textit{\small Held-out OOD probe — response not shown.}",
        ]
    parts += [r"\end{minipage}%"]

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(parts) + "\n")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def produce_panel(story_id: str, category: str, eval_type: str, *,
                  n: int, seed: int, out_dir: Path,
                  wide: bool, want_plot_only: bool, want_tex: bool) -> int:
    """Produce n examples for this panel. Returns count successfully written."""
    samples = load_samples(eval_type, n=n, seed=seed)
    if not samples:
        print(f"  [{eval_type}] NO SAMPLES found — skipping", file=sys.stderr)
        return 0

    panel_dir = out_dir / f"story_{story_id}" / category
    print(f"  [{eval_type}] producing {len(samples)} example(s) -> {panel_dir}")

    for idx, sample in enumerate(samples):
        stem = f"{eval_type}_{idx:02d}"
        combined = panel_dir / f"{stem}.pdf"
        render_combined(sample, category, out_pdf=combined, wide=wide)
        if want_plot_only:
            plot_only = panel_dir / f"{stem}_plot.pdf"
            render_plot_only(sample, out_pdf=plot_only, wide=wide)
        if want_tex:
            tex_file = panel_dir / f"{stem}.tex"
            write_tex_snippet(sample, category, tex_file)
    return len(samples)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("=" * 80)[0].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--story", choices=["A", "B", "C", "all"], default="all",
                   help="Which storyline to render (default: all).")
    p.add_argument("--n", type=int, default=3,
                   help="Examples per panel (default: 3).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sample selection (default: 42).")
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "figures" / "paper",
                   help="Output root directory (default: figures/paper/).")
    p.add_argument("--wide", action="store_true",
                   help="Use full-textwidth figure size (default: 2-column).")
    p.add_argument("--no-plot-only", action="store_true",
                   help="Skip emitting TS-plot-only PDFs.")
    p.add_argument("--no-tex", action="store_true",
                   help="Skip emitting .tex Q/R snippets.")
    p.add_argument("--list-stories", action="store_true",
                   help="Print storyline config and exit.")
    args = p.parse_args()

    if args.list_stories:
        for sid, s in STORIES.items():
            print(f"Story {sid}: {s['title']}")
            for cat, et in s["panels"]:
                print(f"  [{cat:5s}] {et}")
        return 0

    _set_style(wide=args.wide)

    stories = ["A", "B", "C"] if args.story == "all" else [args.story]
    total = 0
    for sid in stories:
        story = STORIES[sid]
        print(f"Story {sid}: {story['title']}")
        for category, eval_type in story["panels"]:
            total += produce_panel(
                sid, category, eval_type,
                n=args.n, seed=args.seed + hash(eval_type) % 1000,
                out_dir=args.out_dir, wide=args.wide,
                want_plot_only=not args.no_plot_only,
                want_tex=not args.no_tex,
            )
    print(f"\nDone. {total} example(s) written under {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
