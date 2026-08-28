"""
Aggregation functions for evaluation results.

Builds per-category summaries from the raw per-sample evaluation CSV.
Used by both evaluation.eval.main (inline after scoring) and
scripts/utils/csv_to_summary.py (standalone from existing CSV).
"""

import ast

import pandas as pd


def _as_list(val):
    """Coerce a value to a Python list.

    Handles both actual lists (from in-memory DataFrame) and
    string-serialized lists (from CSV via pd.read_csv).
    """
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
            return []
        except (ValueError, SyntaxError):
            return []
    return []


def _compute_micro_f1(group_df):
    """Compute micro F1 by pooling TP/FP/FN across samples.

    Requires gt_entities and pred_entities columns (Python lists or
    string-serialized lists from CSV).
    Returns (micro_precision, micro_recall, micro_f1) or (None, None, None).
    """
    if "gt_entities" not in group_df.columns or "pred_entities" not in group_df.columns:
        return None, None, None
    total_tp = total_gt = total_pred = n_valid = 0
    for _, row in group_df.iterrows():
        raw_gt = row.get("gt_entities")
        raw_pred = row.get("pred_entities")
        # Skip rows where extraction failed (None / NaN)
        if raw_gt is None or raw_pred is None:
            continue
        if isinstance(raw_gt, float) or isinstance(raw_pred, float):
            continue  # NaN from pandas
        n_valid += 1
        gt = set(_as_list(raw_gt))
        pred = set(_as_list(raw_pred))
        total_tp += len(gt & pred)
        total_gt += len(gt)
        total_pred += len(pred)
    if n_valid == 0:
        return None, None, None
    if total_gt == 0 and total_pred == 0:
        return 1.0, 1.0, 1.0
    micro_p = total_tp / total_pred if total_pred else 0.0
    micro_r = total_tp / total_gt if total_gt else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    return round(micro_p, 4), round(micro_r, 4), round(micro_f1, 4)


def build_task_type_summary(df):
    """Section 1: FINAL EVALUATION SUMMARY — one row per task_type."""
    agg_cols = {
        "f1": "mean",
        "binary_accuracy": "mean",
        "reasoning_score": "mean",
        "idx": "count",
    }
    if "description_overall" in df.columns:
        agg_cols["description_overall"] = "mean"

    summary = df.groupby("task_type").agg(agg_cols).rename(columns={"idx": "count"})

    # Add micro F1 for entity-set task types (clustering, anticlustering).
    # macro F1 (the "f1" column) averages per-sample F1 — weighs each sample equally.
    # micro F1 pools TP/FP/FN across samples — weighs each entity prediction equally.
    set_types = {"clustering", "anticlustering"}
    for tt in set_types & set(summary.index):
        group = df[df["task_type"] == tt]
        mp, mr, mf1 = _compute_micro_f1(group)
        summary.loc[tt, "micro_p"] = mp
        summary.loc[tt, "micro_r"] = mr
        summary.loc[tt, "micro_f1"] = mf1

    return summary.round(4)


def build_clustering_by_gt_length(df):
    """Section 2: CLUSTERING INSIGHTS BY GT LENGTH."""
    cdf = df[df["task_type"] == "clustering"]
    if cdf.empty:
        return None
    tbl = cdf.groupby("gt_len")[["f1", "precision", "recall"]].mean()
    tbl["sample_count"] = cdf.groupby("gt_len")["idx"].count()
    # Add micro P/R/F1 per GT-length bucket
    for gt_len, group in cdf.groupby("gt_len"):
        mp, mr, mf1 = _compute_micro_f1(group)
        tbl.loc[gt_len, "micro_p"] = mp
        tbl.loc[gt_len, "micro_r"] = mr
        tbl.loc[gt_len, "micro_f1"] = mf1
    return tbl.round(4)


def build_description_perspectives(df):
    """Section 4: DESCRIPTION QA BREAKDOWN — one row per perspective."""
    desc_df = df[df["task_type"] == "description"]
    if desc_df.empty:
        return None

    perspective_cols = {
        "trend": ("trend_cat", "trend_num"),
        "seasonal": ("seasonal_cat", "seasonal_num"),
        "noise": ("noise_cat", "noise_num"),
        "local": ("local_cat", "local_num"),
    }
    num_suffixes = ["amplitude", "period", "position", "strength"]

    rows = []
    for p_name, (cat_col, num_col) in perspective_cols.items():
        if cat_col not in desc_df.columns:
            continue
        cat_vals = desc_df[cat_col].dropna()
        if cat_vals.empty:
            continue
        row = {
            "perspective": p_name,
            "cls": round(cat_vals.mean(), 4),
            "num": round(desc_df[num_col].dropna().mean(), 4) if num_col in desc_df.columns else None,
            "count": cat_vals.shape[0],
        }
        for suffix in num_suffixes:
            col = f"{p_name}_num_{suffix}"
            if col in desc_df.columns:
                vals = desc_df[col].dropna()
                if not vals.empty:
                    row[f"num_{suffix}"] = round(vals.mean(), 4)
        rows.append(row)

    return pd.DataFrame(rows).set_index("perspective") if rows else None


def build_trend_detail(df):
    """Section 5: TREND DIRECTION DETAIL."""
    desc_df = df[df["task_type"] == "description"]
    if desc_df.empty or "trend_overall_cat" not in desc_df.columns:
        return None

    overall = desc_df["trend_overall_cat"].dropna()
    if overall.empty:
        return None

    row = {"direction_cls_acc": round(overall.mean(), 4), "direction_n": overall.shape[0]}

    # Use seg_f1 count as the canonical segment n (matches original main.py)
    seg_f1_col = "trend_segment_f1"
    if seg_f1_col in desc_df.columns:
        seg_f1_vals = desc_df[seg_f1_col].dropna()
        if not seg_f1_vals.empty:
            row["seg_n"] = seg_f1_vals.shape[0]

    for col, key in [
        ("trend_segment_precision", "seg_precision"),
        ("trend_segment_recall", "seg_recall"),
        ("trend_segment_f1", "seg_f1"),
        ("trend_segment_num", "seg_boundary_pos_acc"),
    ]:
        if col in desc_df.columns:
            vals = desc_df[col].dropna()
            if not vals.empty:
                row[key] = round(vals.mean(), 4)

    if "reasoning_score" in desc_df.columns:
        rs = desc_df["reasoning_score"].dropna()
        if not rs.empty:
            row["reasoning_score"] = round(rs.mean(), 4)
            row["reasoning_n"] = rs.shape[0]

    if "description_overall" in desc_df.columns:
        ov = desc_df["description_overall"].dropna()
        if not ov.empty:
            row["overall_score"] = round(ov.mean(), 4)
            row["overall_n"] = ov.shape[0]

    return pd.DataFrame([row])


LENGTH_BUCKET_ORDER = ['in_sft', 'rl_only', 'ood_near', 'ood_far', 'unknown']


def build_length_bucket_summary(df):
    """Per-length-bucket × per-task-type accuracy table.

    Stratifies every task_type by training envelope:
      in_sft   : seq_len ≤ 256        (SFT + RL both trained)
      rl_only  : 256 < seq_len ≤ 768  (RL trained, SFT did not)
      ood_near : 768 < seq_len ≤ 1536 (neither trained — near OOD)
      ood_far  : seq_len > 1536       (extreme OOD)
    """
    if 'length_bucket' not in df.columns:
        return None
    sub = df.copy()
    # Prefer binary_accuracy; fall back to reasoning_score / f1 where binary
    # is null (description / tsevol / segment-family numeric types).
    sub['_score'] = sub.get('binary_accuracy')
    if 'reasoning_score' in sub.columns:
        sub['_score'] = sub['_score'].fillna(sub['reasoning_score'])
    if 'f1' in sub.columns:
        sub['_score'] = sub['_score'].fillna(sub['f1'])
    sub = sub.dropna(subset=['_score'])
    if sub.empty:
        return None
    tbl = sub.groupby(['task_type', 'length_bucket'])['_score'].agg(
        ['mean', 'count']
    ).unstack(fill_value=None)
    # Reorder columns so buckets appear in length order, not alphabetical.
    score_cols = [c for c in LENGTH_BUCKET_ORDER if ('mean', c) in tbl.columns]
    count_cols = [c for c in LENGTH_BUCKET_ORDER if ('count', c) in tbl.columns]
    ordered = [('mean', c) for c in score_cols] + [('count', c) for c in count_cols]
    return tbl[ordered].round(4)


def build_length_bucket_overall(df):
    """Overall score × length bucket (single row per bucket, across all task types).

    Useful as the headline length-generalization chart: one score per length
    regime, averaged over all task types weighted by sample count.
    """
    if 'length_bucket' not in df.columns:
        return None
    sub = df.copy()
    sub['_score'] = sub.get('binary_accuracy')
    if 'reasoning_score' in sub.columns:
        sub['_score'] = sub['_score'].fillna(sub['reasoning_score'])
    if 'f1' in sub.columns:
        sub['_score'] = sub['_score'].fillna(sub['f1'])
    sub = sub.dropna(subset=['_score'])
    if sub.empty:
        return None
    tbl = sub.groupby('length_bucket').agg(
        mean_score=('_score', 'mean'),
        n_samples=('_score', 'count'),
    )
    present = [b for b in LENGTH_BUCKET_ORDER if b in tbl.index]
    return tbl.loc[present].round(4)


def build_tsevol_strategy_summary(df):
    """Section 6: TSEVOL PERFORMANCE BY STRATEGY — one row per evolution strategy."""
    tdf = df[df["task_type"] == "tsevol"]
    if tdf.empty or "evol_strategy" not in tdf.columns:
        return None

    agg = tdf.groupby("evol_strategy").agg(
        reasoning_score=("reasoning_score", "mean"),
        count=("idx", "count"),
    )
    return agg.round(4)


def save_aggregated_csvs(df, out_dir):
    """Save all per-category aggregated CSVs to out_dir."""
    import os

    tbl = build_task_type_summary(df)
    tbl.to_csv(os.path.join(out_dir, "summary_by_task_type.csv"))

    clu = build_clustering_by_gt_length(df)
    if clu is not None:
        clu.to_csv(os.path.join(out_dir, "clustering_by_gt_length.csv"))

    persp = build_description_perspectives(df)
    if persp is not None:
        persp.to_csv(os.path.join(out_dir, "description_perspectives.csv"))

    trend = build_trend_detail(df)
    if trend is not None:
        trend.to_csv(os.path.join(out_dir, "description_trend_detail.csv"), index=False)

    tsevol = build_tsevol_strategy_summary(df)
    if tsevol is not None:
        tsevol.to_csv(os.path.join(out_dir, "tsevol_by_strategy.csv"))

    len_over = build_length_bucket_overall(df)
    if len_over is not None:
        len_over.to_csv(os.path.join(out_dir, "length_bucket_overall.csv"))

    len_by_type = build_length_bucket_summary(df)
    if len_by_type is not None:
        len_by_type.to_csv(os.path.join(out_dir, "length_bucket_by_task_type.csv"))


def write_summary_txt(df, path):
    """Write the human-readable summary txt (mirrors original main.py format)."""
    lines = []

    def p(text=""):
        lines.append(str(text))

    p()
    p("=" * 50)
    p("FINAL EVALUATION SUMMARY")
    p("=" * 50)

    if df.empty:
        p("No results processed.")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return

    # Section 1
    summary = build_task_type_summary(df)
    p(summary.to_string())

    # Section 2
    p()
    p("=" * 50)
    p("CLUSTERING INSIGHTS BY GT LENGTH")
    p("=" * 50)
    clu = build_clustering_by_gt_length(df)
    if clu is not None:
        p(clu.to_string())
    else:
        p("No clustering tasks found.")

    # Micro F1 for entity-set types (read from summary to avoid recomputing)
    if "micro_f1" in summary.columns:
        for tt in ("clustering", "anticlustering"):
            if tt in summary.index and pd.notna(summary.loc[tt, "micro_f1"]):
                mp = summary.loc[tt, "micro_p"]
                mr = summary.loc[tt, "micro_r"]
                mf1 = summary.loc[tt, "micro_f1"]
                macro_f1 = summary.loc[tt, "f1"]
                n = int(summary.loc[tt, "count"])
                p(f"  {tt} micro: P={mp:.4f}  R={mr:.4f}  F1={mf1:.4f}  (macro F1={macro_f1:.4f}, n={n})")

    # Section 3: reasoning breakdown
    if "reasoning_score" in df.columns:
        rs_data = df[df["reasoning_score"].notna()]
        if not rs_data.empty:
            p()
            p("=" * 50)
            p("REASONING SCORE BREAKDOWN BY TASK TYPE")
            p("=" * 50)
            for tt, group in rs_data.groupby("task_type"):
                rs_vals = group["reasoning_score"].dropna()
                if not rs_vals.empty:
                    p(f"  {tt:12s}  score={rs_vals.mean():.4f}  n={rs_vals.shape[0]}")

    # Section 4: description QA breakdown
    desc_df = df[df["task_type"] == "description"]
    if not desc_df.empty:
        p()
        p("=" * 50)
        p("DESCRIPTION QA BREAKDOWN")
        p("  cls = categorical classification accuracy | quant = numerical value accuracy")
        p("=" * 50)

        perspective_cols = {
            "trend": ("trend_cat", "trend_num"),
            "seasonal": ("seasonal_cat", "seasonal_num"),
            "noise": ("noise_cat", "noise_num"),
            "local": ("local_cat", "local_num"),
        }
        num_type_suffixes = ["amplitude", "period", "position", "strength"]
        suffix_labels = {"amplitude": "amp", "period": "period", "position": "pos", "strength": "str"}

        for p_name, (cat_col, num_col) in perspective_cols.items():
            if cat_col in desc_df.columns:
                cat_mean = desc_df[cat_col].dropna().mean()
                num_mean = desc_df[num_col].dropna().mean() if num_col in desc_df.columns else float("nan")
                count = desc_df[cat_col].dropna().shape[0]
                if count > 0:
                    type_parts = []
                    for suffix in num_type_suffixes:
                        col = f"{p_name}_num_{suffix}"
                        if col in desc_df.columns:
                            vals = desc_df[col].dropna()
                            if not vals.empty:
                                label = suffix_labels[suffix]
                                type_parts.append(f"{label}={vals.mean():.4f}")
                    type_str = "  ".join(type_parts)
                    if type_str:
                        p(f"  {p_name:12s}  cls={cat_mean:.4f}  quant={num_mean:.4f}  [{type_str}]  n={count}")
                    else:
                        p(f"  {p_name:12s}  cls={cat_mean:.4f}  quant={num_mean:.4f}  n={count}")

        # Section 5: trend detail
        if "trend_overall_cat" in desc_df.columns:
            overall = desc_df["trend_overall_cat"].dropna()
            if not overall.empty:
                p()
                p("=" * 50)
                p("TREND DIRECTION DETAIL  (overall direction type + segment-level matching)")
                p("=" * 50)
                p(f"    {'direction_cls':14s}  acc={overall.mean():.4f}  n={overall.shape[0]}")

            seg_f1 = desc_df["trend_segment_f1"].dropna() if "trend_segment_f1" in desc_df.columns else pd.Series()
            seg_p = desc_df["trend_segment_precision"].dropna() if "trend_segment_precision" in desc_df.columns else pd.Series()
            seg_r = desc_df["trend_segment_recall"].dropna() if "trend_segment_recall" in desc_df.columns else pd.Series()
            seg_num = desc_df["trend_segment_num"].dropna() if "trend_segment_num" in desc_df.columns else pd.Series()
            if not seg_f1.empty:
                num_str = f"  boundary_pos_acc={seg_num.mean():.4f}" if not seg_num.empty else ""
                p(f"    {'segments':14s}  P={seg_p.mean():.4f}  R={seg_r.mean():.4f}  F1={seg_f1.mean():.4f}{num_str}  n={seg_f1.shape[0]}")

        if "reasoning_score" in desc_df.columns:
            rs = desc_df["reasoning_score"].dropna()
            if not rs.empty:
                p(f"  {'reasoning':14s}  score={rs.mean():.4f}  n={rs.shape[0]}")

        if "local_reasoning_score" in desc_df.columns:
            lrs = desc_df["local_reasoning_score"].dropna()
            if not lrs.empty:
                p(f"  {'local_reasoning':15s}  score={lrs.mean():.4f}  n={lrs.shape[0]}")

        if "description_overall" in desc_df.columns:
            ov = desc_df["description_overall"].dropna()
            if not ov.empty:
                p(f"  OVERALL (avg of cls + quant + reasoning across all perspectives)  score={ov.mean():.4f}  n={ov.shape[0]}")

    # Section 5b: Length-bucket generalization
    len_over = build_length_bucket_overall(df)
    if len_over is not None and not len_over.empty:
        p()
        p("=" * 50)
        p("LENGTH GENERALIZATION (overall score by length bucket)")
        p("  in_sft ≤256 | rl_only 257-768 | ood_near 769-1536 | ood_far >1536")
        p("=" * 50)
        for bucket, row in len_over.iterrows():
            p(f"  {bucket:10s}  score={row['mean_score']:.4f}  n={int(row['n_samples'])}")

    # Section 6: TSEvol strategy breakdown
    tsevol_df = df[df["task_type"] == "tsevol"]
    if not tsevol_df.empty and "evol_strategy" in tsevol_df.columns:
        p()
        p("=" * 50)
        p("TSEVOL PERFORMANCE BY STRATEGY")
        p("=" * 50)
        for strategy, group in tsevol_df.groupby("evol_strategy"):
            rs = group["reasoning_score"].dropna()
            if not rs.empty:
                p(f"  {strategy:22s}  score={rs.mean():.4f}  n={rs.shape[0]}")
        total_rs = tsevol_df["reasoning_score"].dropna()
        if not total_rs.empty:
            p(f"  {'OVERALL':22s}  score={total_rs.mean():.4f}  n={total_rs.shape[0]}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
