"""Build a confident, evidence-based taxonomy mapping every task_type that
appears in our evaluation data to one of 5 buckets:

  - Atomic           : single-skill primitives. Tagged `atomic_*`.
                       Always seen in SFT training.
  - ID-easy          : non-atomic compositional types seen in SFT training
                       with CoT. Includes:
                          * legacy SFT types (clustering, anticlustering,
                            correlation, anticorrelation, segment_*,
                            stat_numerical, yes_no, temporal_position,
                            change_point, periodicity, ...)
                          * bridge `rl_*` types (eval_metadata.bridge=True
                            in train_sft.jsonl).
  - ID-hard          : `rl_*` types that appear ONLY in RL training
                       (train_rl.jsonl), not in any SFT bridge demo.
                       Model never saw a CoT demo for these during SFT.
  - OOD              : `ood_*` prefix — held out from BOTH SFT and RL
                       training; the model's first exposure is at eval time.
  - Description      : the legacy `description` family (free-form perspectives
                       scored by reasoning_score).

Provenance: the bucket assignment is grounded in actual presence/absence
of each task_type in train_sft.jsonl (with bridge flag) and train_rl.jsonl,
plus the eval_type prefix convention.

Outputs:
  docs/task_taxonomy.json   — {task_type: {bucket, in_sft, in_rl, has_bridge, scoring_metric}}
  docs/task_taxonomy.md     — human-readable, grouped by bucket, with totals
                              and any unclassified types flagged.
"""
from __future__ import annotations
import json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
os.chdir(str(_REPO))

import pandas as pd


# --- Evidence: what eval_types appear in each training split? ----------------

def collect_train_evidence():
    sft_counts = Counter()
    sft_bridge = set()
    rl_counts = Counter()
    with open("data/train_sft.jsonl") as f:
        for line in f:
            r = json.loads(line)
            et = r.get("eval_type") or "?"
            sft_counts[et] += 1
            em = r.get("eval_metadata") or {}
            if em.get("bridge"):
                sft_bridge.add(et)
    with open("data/train_rl.jsonl") as f:
        for line in f:
            r = json.loads(line)
            et = r.get("eval_type") or "?"
            rl_counts[et] += 1
    return sft_counts, sft_bridge, rl_counts


# --- All eval_types that appear in our evaluation data ------------------------

def collect_eval_types():
    types = set()
    for ds in ["test", "test_len_near", "test_len_far", "test_len_extreme", "val"]:
        p = Path(f"data/{ds}.jsonl")
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                types.add(r.get("eval_type") or "?")
    return types


# --- Scoring metric per task_type (read from one CSV that has the task) -------

def collect_scoring_metric():
    """For each task_type, infer the scoring metric used: f1 (clustering),
    reasoning_score (description), or binary_accuracy (everything else)."""
    metric_for = {}
    paths = [
        Path("exp_topup_combined/rl/test_len_near/evaluation_report_optimized.csv"),
        Path("exp_topup_combined/rl/test_len_far/evaluation_report_optimized.csv"),
        Path("exp_topup_combined/rl/test_len_extreme/evaluation_report_optimized.csv"),
        Path("exp/rl/test/evaluation_report_optimized.csv"),
    ]
    for p in paths:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for tt, grp in df.groupby("task_type"):
            if tt in metric_for:
                continue
            f1_filled = grp["f1"].notna().sum() if "f1" in grp.columns else 0
            ba_filled = grp["binary_accuracy"].notna().sum() if "binary_accuracy" in grp.columns else 0
            rs_filled = grp["reasoning_score"].notna().sum() if "reasoning_score" in grp.columns else 0
            if f1_filled > ba_filled:
                metric_for[tt] = "F1"
            elif tt == "description":
                metric_for[tt] = "reasoning_score (LLM-judge)"
            elif ba_filled > 0:
                metric_for[tt] = "binary_accuracy (verdict_match or proximity)"
            else:
                metric_for[tt] = "reasoning_score (LLM-judge)"
    return metric_for


# --- Bucket assignment --------------------------------------------------------

def assign_bucket(task_type: str, sft: Counter, sft_bridge: set, rl: Counter) -> str:
    """Confident bucket assignment based on training-data evidence + naming."""
    t = str(task_type)
    if t == "description":
        return "Description"
    if t.startswith("atomic_"):
        return "Atomic"
    if t.startswith("ood_"):
        return "OOD"
    if t.startswith("rl_"):
        # In SFT bridge → ID-easy; otherwise (RL-only) → ID-hard.
        if t in sft_bridge:
            return "ID-easy"
        if t in sft:
            # rl_* present in SFT without bridge flag — flag for review.
            return "ID-easy"  # Defensive: if it's in SFT, model saw it.
        return "ID-hard"
    # Non-prefixed types (clustering, anticlustering, segment_*, stat_*, etc.)
    if t in sft:
        return "ID-easy"
    if t in rl and t not in sft:
        return "ID-hard"
    # Type appears nowhere in training? Flag.
    return "UNCATEGORIZED"


def main():
    print("Collecting evidence …")
    sft, sft_bridge, rl = collect_train_evidence()
    eval_types = collect_eval_types()
    metrics = collect_scoring_metric()
    print(f"  train_sft.jsonl: {sum(sft.values())} rows, {len(sft)} eval_types")
    print(f"  bridge rl_* in SFT: {len(sft_bridge)}")
    print(f"  train_rl.jsonl: {sum(rl.values())} rows, {len(rl)} eval_types")
    print(f"  distinct eval_types in eval datasets: {len(eval_types)}")

    by_bucket = defaultdict(list)
    rows = {}
    for tt in sorted(eval_types):
        b = assign_bucket(tt, sft, sft_bridge, rl)
        rows[tt] = {
            "bucket": b,
            "in_sft": sft.get(tt, 0),
            "in_sft_as_bridge": tt in sft_bridge,
            "in_rl": rl.get(tt, 0),
            "scoring_metric": metrics.get(tt, "(unseen in CSV)"),
        }
        by_bucket[b].append(tt)

    out_json = Path("docs/task_taxonomy.json")
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_json}")

    # Markdown
    lines = []
    lines.append("# Task Taxonomy — bucket assignment for every eval_type\n")
    lines.append(f"Generated by `scripts/utils/build_task_taxonomy.py`. "
                 f"Evidence: presence in `train_sft.jsonl` (with `eval_metadata.bridge=True` flag) "
                 f"and `train_rl.jsonl`. {len(eval_types)} distinct eval_types in our eval data.\n")
    bucket_order = ["Atomic", "ID-easy", "ID-hard", "OOD", "Description", "UNCATEGORIZED"]
    for b in bucket_order:
        types = sorted(by_bucket[b])
        if not types:
            continue
        lines.append(f"## {b} ({len(types)} task_types)\n")
        if b == "Atomic":
            lines.append("Single-skill primitives, `atomic_*` prefix. Always seen in SFT training.\n")
        elif b == "ID-easy":
            lines.append("Non-atomic compositional types seen in SFT training with CoT — "
                         "either as legacy SFT types (clustering, segment_judgment, etc.) or as "
                         "`rl_*` bridge demos (`eval_metadata.bridge=True` in train_sft.jsonl).\n")
        elif b == "ID-hard":
            lines.append("`rl_*` types that appear ONLY in `train_rl.jsonl`, NOT in any SFT "
                         "bridge demo. Model's first CoT exposure to these is during RL training.\n")
        elif b == "OOD":
            lines.append("`ood_*` prefix — held out from both SFT and RL training. First exposure is at eval time.\n")
        elif b == "Description":
            lines.append("Free-form perspectives, scored by `reasoning_score`.\n")
        elif b == "UNCATEGORIZED":
            lines.append("**⚠️ FLAG**: these task_types do not fit cleanly into the taxonomy. "
                         "They appear in eval data but are not seen in train_sft or train_rl, "
                         "and don't match any prefix convention.\n")
        # Table
        lines.append("| eval_type | in train_sft (rows) | bridge? | in train_rl (rows) | scoring metric |")
        lines.append("|---|---:|:---:|---:|---|")
        for t in types:
            r = rows[t]
            lines.append(f"| `{t}` | {r['in_sft']} | "
                         f"{'✓' if r['in_sft_as_bridge'] else ''} | "
                         f"{r['in_rl']} | {r['scoring_metric']} |")
        lines.append("")

    out_md = Path("docs/task_taxonomy.md")
    out_md.write_text("\n".join(lines))
    print(f"Wrote {out_md}")

    # Summary
    print("\n=== Summary by bucket ===")
    for b in bucket_order:
        n = len(by_bucket[b])
        if n: print(f"  {b}: {n} task_types")
    if by_bucket["UNCATEGORIZED"]:
        print(f"\n⚠️  UNCATEGORIZED: {by_bucket['UNCATEGORIZED']}")
    else:
        print("\n✓ All eval_types successfully bucketed.")


if __name__ == "__main__":
    main()
