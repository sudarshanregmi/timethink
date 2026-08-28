"""
Preprocess ChatTS JSONL data into verl-compatible Parquet format.

Reads train_sft.jsonl, train_rl.jsonl, and val.jsonl produced by
`python -m synth.align` and writes:

  train_sft.parquet  — SFT training (prompt + response)
  train_rl.parquet   — RL training  (prompt + reward_model)
  val_sft.parquet    — SFT validation (length-filtered ≤256)
  val_rl.parquet     — RL validation  (full length pool, mixes ≤256 + extension)

The key design choice: `reward_model` carries `eval_type` and `eval_metadata`
so the reward function can route by QA type directly from kwargs.
"""

import argparse
import json
import os
from collections import defaultdict

import datasets
import pandas as pd

DATA_SOURCE_NAME = "timethink"

# Binary verdict eval_types that need class balancing.
# Only types with genuinely binary verdicts (yes/no, true/false).
BINARY_VERDICT_TYPES = frozenset({
    'yes_no',
    'anti_judgment', 'segment_judgment', 'cross_stat_judgment',
    # OOD binary verdict types (eval-only)
    'ood_trend_reversal', 'ood_range_normalized_amplitude', 'ood_segment_stat_compare',
    'ood_max_before_min', 'ood_std_exceeds_half_range',
    'ood_max_in_highest_mean_quarter', 'ood_quarter_mean_ordering',
    'ood_symmetric_recovery', 'ood_cycle_mean_trend', 'ood_event_amplitude_vs_std',
    'ood_amplitude_vs_segment_std', 'ood_symmetric_trend_sequence', 'ood_trend_follows_mean',
    # RL single-metric binary types
    'rl_amplitude_vs_range', 'rl_amplitude_vs_std', 'rl_condition_recovery',
    'rl_event_in_trend_type', 'rl_event_near_extremum', 'rl_extrema_same_half',
    'rl_half_mean_compare', 'rl_has_periodicity', 'rl_interval_comparison',
    'rl_max_in_first_half', 'rl_max_in_trend_type', 'rl_mean_shift',
    'rl_mean_stability', 'rl_median_mean_close', 'rl_monotonic_chunks',
    'rl_segment_mean_compare', 'rl_volatility_change',
    # Cross-metric taxonomy binary types
    'atomic_cross_stat_compare', 'atomic_cross_trend_align',
    'rl_cross_stat_ratio', 'rl_cross_event_sync', 'rl_cross_period_compare',
    'rl_cross_attribute_corr',
    'rl_corr_conditional',
    'ood_cross_extrema_alignment', 'ood_cross_range_overlap',
    'ood_cluster_singleton', 'ood_corr_transitivity',
    'ood_cross_trend_convergence',
    # OOD compositional probes (reverse-engineered)
    'ood_peak_in_longest_segment', 'ood_cross_event_causality',
})

# Types where verdict is in output text, not eval_metadata['verdict'].
# Need special extraction during balancing.
_VERDICT_IN_OUTPUT = frozenset({'yes_no'})

# Maximum majority fraction after balancing (0.5 = hard 50/50).
_BALANCE_MAX_MAJORITY = 0.5


def _base_fields(row, split):
    """Fields common to all output schemas."""
    ts_data = row["timeseries"]
    assert isinstance(ts_data, list), f"timeseries must be a list, got {type(ts_data)}"

    eval_type = row.get("eval_type", "")
    eval_metadata = row.get("eval_metadata", {})

    return {
        "data_source": DATA_SOURCE_NAME,
        "prompt": [{"role": "user", "content": row["input"]}],
        "timeseries": ts_data,
        "ability": "time-series-reasoning",
        "extra_info": json.dumps({
            "split": split,
            "index": row.name if hasattr(row, "name") else 0,
            "eval_type": eval_type,
            "eval_metadata": eval_metadata,
            "sub_type": eval_metadata.get("sub_type", ""),
        }),
    }


def _reward_model_field(row):
    return json.dumps({
        "style": "rule",
        "ground_truth": row["output"],
        "eval_type": row.get("eval_type", ""),
        "eval_metadata": row.get("eval_metadata", {}),
    })


def process_sft_row(row, split):
    rec = _base_fields(row, split)
    rec["response"] = row["output"]
    return rec


def process_rl_row(row, split):
    rec = _base_fields(row, split)
    rec["reward_model"] = _reward_model_field(row)
    return rec


def process_val_row(row, split):
    rec = _base_fields(row, split)
    rec["response"] = row["output"]
    rec["reward_model"] = _reward_model_field(row)
    return rec


def balance_eval_type_distribution(df, max_per_type=None):
    """Cap over-represented eval_types and report distribution.

    If max_per_type is None, uses the 75th-percentile count across all
    eval_types as the cap.  This prevents dominant types (tsevol, description)
    from drowning out rare types during RL training, without throwing away
    ALL excess data.

    Returns the balanced DataFrame and prints a summary.
    """
    df = df.copy()
    et_col = df['eval_type'].fillna('')

    type_counts = et_col.value_counts()
    if max_per_type is None:
        # Use 75th percentile as cap — keeps most types intact,
        # only trims the few extreme outliers
        max_per_type = int(type_counts.quantile(0.75))
        # But never cap below the median to avoid throwing away too much
        max_per_type = max(max_per_type, int(type_counts.median()))
        # And ensure at least 100 samples per type
        max_per_type = max(max_per_type, 100)

    print(f"  Eval-type cap: {max_per_type} samples per type")

    keep_mask = pd.Series(True, index=df.index)
    total_dropped = 0

    for et, cnt in type_counts.items():
        if cnt <= max_per_type:
            continue
        drop_count = cnt - max_per_type
        et_indices = df[et_col == et].index
        drop_indices = et_indices.to_series().sample(n=drop_count, random_state=42).index
        keep_mask.loc[drop_indices] = False
        total_dropped += drop_count
        print(f"  Cap {et}: {cnt} → {max_per_type} (dropped {drop_count})")

    df = df[keep_mask].reset_index(drop=True)
    if total_dropped > 0:
        print(f"  Eval-type balancing: dropped {total_dropped} samples, {len(df)} remaining.")
    else:
        print(f"  Eval-type balancing: no types exceeded cap, {len(df)} samples kept.")

    # Report final distribution for visibility
    final_counts = df['eval_type'].fillna('').value_counts()
    print(f"  Final distribution: min={final_counts.min()}, "
          f"max={final_counts.max()}, "
          f"median={int(final_counts.median())}, "
          f"types={len(final_counts)}")

    return df


def balance_binary_verdicts(df):
    """Hard-balance majority class for binary verdict types.

    - Extracts verdict from eval_metadata OR output text (type-dependent)
    - Balances ALL binary types regardless of sample count
    - Hard 50/50 balance to eliminate majority-class exploitation
    - Only targets genuinely binary types (BINARY_VERDICT_TYPES)
    """
    import re

    def _get_verdict(row):
        et = row.get('eval_type', '')
        # Types that store verdict in output text, not metadata
        if et in _VERDICT_IN_OUTPUT:
            output = str(row.get('output', ''))
            m = re.search(r'answer:\s*(yes|no)', output, re.IGNORECASE)
            return m.group(1).lower() if m else ''
        # All other types: verdict in eval_metadata
        meta = row.get('eval_metadata')
        if isinstance(meta, dict):
            return str(meta.get('verdict', ''))
        return ''

    df = df.copy()
    df['_verdict'] = df.apply(_get_verdict, axis=1)
    df['_eval_type'] = df['eval_type'].fillna('')

    keep_mask = pd.Series(True, index=df.index)
    total_dropped = 0

    for et in BINARY_VERDICT_TYPES:
        et_mask = df['_eval_type'] == et
        et_df = df[et_mask]
        if len(et_df) == 0:
            continue

        verdict_counts = et_df['_verdict'].value_counts()
        if len(verdict_counts) != 2:
            continue  # Not binary in practice — skip

        total_et = len(et_df)
        minority_count = verdict_counts.min()
        majority_label = verdict_counts.idxmax()
        majority_count = verdict_counts.max()

        # Hard balance: downsample majority to match minority
        max_allowed = minority_count
        if majority_count <= max_allowed:
            skew = majority_count / total_et
            print(f"  OK {et}: {majority_label}={majority_count}/{total_et} "
                  f"({skew:.0%} skew, within {_BALANCE_MAX_MAJORITY:.0%} cap)")
            continue

        # Downsample majority to match minority
        drop_count = majority_count - max_allowed
        majority_indices = et_df[et_df['_verdict'] == majority_label].index
        drop_indices = majority_indices.to_series().sample(n=drop_count, random_state=42).index
        keep_mask.loc[drop_indices] = False
        total_dropped += drop_count

        kept = total_et - drop_count
        new_skew = max_allowed / kept
        print(f"  Balance {et}: {majority_label}={majority_count} → {max_allowed} "
              f"(dropped {drop_count}, {total_et} → {kept}, skew {new_skew:.0%})")

    df = df[keep_mask].drop(columns=['_verdict', '_eval_type']).reset_index(drop=True)
    if total_dropped > 0:
        print(f"  Total: dropped {total_dropped} majority-class samples, {len(df)} rows remaining.")
    else:
        print(f"  Verdict balancing: no types exceeded {_BALANCE_MAX_MAJORITY:.0%} cap.")
    return df


def drop_unfixable_types_for_training(df):
    """Drop eval_types that have no meaningful learning signal in training.

    These are types where the generator can only produce one verdict in
    practice (e.g. rl_corr_conditional's sync condition at tolerance=20
    over independently-placed events = 100% "no"). Keeping them in
    training data teaches the model to always output the dominant
    verdict — pure reward-hacking surface.

    Applied ONLY to train_sft / train_rl. Val/test keep these so the model
    can still be evaluated on them (and fail honestly rather than game them).
    """
    DROP_FROM_TRAINING = frozenset({
        # 100% "no" in practice — sync condition over random event positions
        # is essentially unreachable. Dropped for training; kept in val/test.
        'rl_corr_conditional',
    })
    et_col = df['eval_type'].fillna('')
    mask = ~et_col.isin(DROP_FROM_TRAINING)
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"  Unfixable-for-training filter: dropped {dropped} samples "
              f"({', '.join(sorted(DROP_FROM_TRAINING))})")
    return df[mask].reset_index(drop=True)


def rebalance_numeric_verdicts_for_training(df, max_majority_ratio=2.0):
    """Cap majority-value count for numeric verdict types in training data.

    Integer-count verdicts (rl_segment_count, rl_event_count_by_type) have
    a natural distribution peak at "1" (~82%). The reward uses EXACT match
    for small-integer counts, so a model that always outputs "1" collects
    ~82% reward — majority-class exploitation. Cap per-value sample count
    at `max_majority_ratio × median` to force gradient across value levels.

    Applied ONLY to train_sft / train_rl.
    """
    import numpy as np

    NUMERIC_COUNT_TYPES = frozenset({
        'rl_segment_count', 'rl_event_count_by_type',
        'rl_event_count', 'ood_conditional_count',
        # Atomic count/enumeration types with natural mode at "1"
        # (single trend segment / single event). Without rebalance the
        # "always predict 1" shortcut yields 75% reward.
        'atomic_trend_enumeration', 'atomic_event_enumeration',
    })
    et_col = df['eval_type'].fillna('')
    mask_keep = pd.Series(True, index=df.index)
    total_dropped = 0

    for et in NUMERIC_COUNT_TYPES & set(et_col):
        et_df = df[et_col == et]
        if len(et_df) < 50:
            continue
        # Extract verdict values
        verdicts = et_df['eval_metadata'].apply(
            lambda m: (m or {}).get('verdict') if isinstance(m, dict) else None
        )
        verdict_counts = verdicts.value_counts()
        if verdict_counts.empty:
            continue
        median_count = int(verdict_counts.median())
        cap = max(median_count * int(max_majority_ratio), 20)
        for v, cnt in verdict_counts.items():
            if cnt <= cap:
                continue
            v_indices = et_df[verdicts == v].index
            drop_n = cnt - cap
            drop_idx = v_indices.to_series().sample(n=drop_n, random_state=42).index
            mask_keep.loc[drop_idx] = False
            total_dropped += drop_n
            print(f"  Numeric-count rebalance {et}/verdict={v}: "
                  f"{cnt} → {cap} (dropped {drop_n})")

    if total_dropped > 0:
        print(f"  Numeric-count rebalance: dropped {total_dropped} total, "
              f"{mask_keep.sum()} remaining")
    return df[mask_keep].reset_index(drop=True)


def stratified_val_sample(df, max_samples=1000):
    """Stratified downsample for validation: equal representation per eval_type.

    Ensures the small val set (~1K) covers ALL eval_types, so validation
    loss during RL rollout reflects all task families, not just common ones.
    """
    et_col = df['eval_type'].fillna('')
    unique_types = et_col.unique()
    n_types = len(unique_types)

    if n_types == 0:
        return df
    if len(df) <= max_samples:
        return df

    per_type = max(1, max_samples // n_types)
    sampled_parts = []
    for et in unique_types:
        et_df = df[et_col == et]
        n = min(len(et_df), per_type)
        sampled_parts.append(et_df.sample(n=n, random_state=42))

    result = pd.concat(sampled_parts, ignore_index=True)

    # If under budget, fill remaining slots from unsampled rows
    if len(result) < max_samples:
        used_indices = set(result.index)
        remaining = df[~df.index.isin(used_indices)]
        extra = min(max_samples - len(result), len(remaining))
        if extra > 0:
            result = pd.concat([
                result,
                remaining.sample(n=extra, random_state=42)
            ], ignore_index=True)

    # If over budget, trim
    if len(result) > max_samples:
        result = result.sample(n=max_samples, random_state=42).reset_index(drop=True)

    final_counts = result['eval_type'].fillna('').value_counts()
    print(f"  Stratified val: {len(df)} → {len(result)} samples, "
          f"{n_types} eval_types, {final_counts.min()}-{final_counts.max()} per type")

    return result


def convert_jsonl_to_parquet(jsonl_path, parquet_path, split_name, row_fn,
                             balance=False, stratify_val=False,
                             length_filter=None, training_quality_filter=False):
    """Read a JSONL file, process it, and save as Parquet.

    Args:
        length_filter: Optional (min_len, max_len) inclusive; rows whose
            eval_metadata.length is outside this range are dropped before
            stratification/balancing. Used to emit val_sft (≤256-only) and
            val_rl (full pool) from the same source val.jsonl.
        training_quality_filter: If True, drops eval_types that have no
            meaningful training signal (100% single-verdict bugs) and caps
            majority-value counts for numeric types. ONLY set for
            train_sft / train_rl; NOT for val/test (those keep everything
            for honest evaluation).
    """
    if not os.path.exists(jsonl_path):
        print(f"Warning: {jsonl_path} does not exist. Skipping.")
        return

    print(f"Processing {split_name} data from: {jsonl_path}")

    try:
        df = pd.read_json(jsonl_path, lines=True)
    except ValueError as e:
        print(f"Error reading {jsonl_path}: {e}")
        return

    print(f"  -> Found {len(df)} rows.")

    if length_filter is not None:
        lo, hi = length_filter
        lens = df['eval_metadata'].apply(
            lambda m: (m or {}).get('length') if isinstance(m, dict) else None
        )
        keep = lens.between(lo, hi, inclusive='both').fillna(False)
        dropped = (~keep).sum()
        df = df[keep].reset_index(drop=True)
        print(f"  Length filter [{lo}, {hi}]: dropped {dropped}, {len(df)} remaining.")

    if training_quality_filter:
        df = drop_unfixable_types_for_training(df)
        df = rebalance_numeric_verdicts_for_training(df)

    if stratify_val:
        df = stratified_val_sample(df, max_samples=1000)

    if balance:
        # Report distribution for awareness (no samples dropped).
        # Balancing is unnecessary because:
        #   - Generation enforces 50/50 verdict coin flip with skip-on-failure
        #   - WRONG_VERDICT_CAP (0.30) in reward prevents verdict exploitation
        #   - GRPO per-prompt advantage normalization prevents cross-prompt gaming
        #   - TSEvol diversity means more samples = more unique patterns, not bias
        et_counts = df['eval_type'].fillna('').value_counts()
        print(f"  Distribution: {len(et_counts)} types, "
              f"min={et_counts.min()}, max={et_counts.max()}, "
              f"median={int(et_counts.median())}")

    processed_df = df.apply(
        lambda row: row_fn(row, split_name), axis=1, result_type="expand"
    )

    hf_dataset = datasets.Dataset.from_pandas(processed_df)
    hf_dataset.to_parquet(parquet_path)
    print(f"  -> Saved to: {parquet_path}")

    example = hf_dataset[0]
    example_path = parquet_path.replace(".parquet", "_example.json")
    with open(example_path, "w") as f:
        json.dump(example, f, indent=2)
    print(f"  -> Example saved to: {example_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ChatTS JSONL data into verl Parquet format."
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory containing train_sft.jsonl, train_rl.jsonl, and val.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="./data",
        help="Output directory for parquet files (defaults to --data_dir)",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS path to upload to")
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    output_dir = os.path.expanduser(args.output_dir) if args.output_dir else data_dir
    os.makedirs(output_dir, exist_ok=True)

    convert_jsonl_to_parquet(
        os.path.join(data_dir, "train_sft.jsonl"),
        os.path.join(output_dir, "train_sft.parquet"),
        "train_sft",
        process_sft_row,
        balance=True,
        training_quality_filter=True,
    )
    convert_jsonl_to_parquet(
        os.path.join(data_dir, "train_rl.jsonl"),
        os.path.join(output_dir, "train_rl.parquet"),
        "train_rl",
        process_rl_row,
        balance=True,
        training_quality_filter=True,
    )
    # val_sft.parquet: SFT-phase validation — length-filtered to ≤256 so
    # the validation distribution matches the SFT training distribution
    # (which was generated with seq_len capped at 256 pre-gate-relaxation).
    convert_jsonl_to_parquet(
        os.path.join(data_dir, "val.jsonl"),
        os.path.join(output_dir, "val_sft.parquet"),
        "val_sft",
        process_val_row,
        stratify_val=True,
        length_filter=(0, 256),
    )
    # val_rl.parquet: RL-phase validation — full length pool (≤256 AND
    # [32, 768] extension), matches the 50/50 RL training distribution.
    convert_jsonl_to_parquet(
        os.path.join(data_dir, "val.jsonl"),
        os.path.join(output_dir, "val_rl.parquet"),
        "val_rl",
        process_val_row,
        stratify_val=True,
    )
    # Test OOD length buckets — optional, only emitted if the JSONL exists.
    # Used for measuring length generalization at eval time, stratified by
    # bucket (near / far / extreme) in the results reporting.
    for bucket in ("test_len_near", "test_len_far", "test_len_extreme"):
        convert_jsonl_to_parquet(
            os.path.join(data_dir, f"{bucket}.jsonl"),
            os.path.join(output_dir, f"{bucket}.parquet"),
            bucket,
            process_val_row,
        )
    # NOTE: `train_sft_only.parquet` and `train_rl_all.parquet` are no
    # longer emitted. Under Option A (train_sft_only == train_sft), they
    # were byte-equivalent to train_sft.parquet. The RL scripts now point
    # at train_rl.parquet directly (the paraphrase-only, reward-verifiable
    # pool we actually want RL to train on) rather than train_rl_all
    # (which was derived from train_sft_only and would have trained RL on
    # SFT data under Option A — defeating the whole thesis).

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        print(f"\nUploading to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=output_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
