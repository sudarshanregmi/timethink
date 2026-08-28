"""
Pre-flight estimator: project per-eval_type sample counts from datagen config.

Run BEFORE expensive data generation to validate coverage:
    python scripts/utils/estimate_distribution.py [--config config/datagen_config.yaml]
"""
import argparse
import yaml
from collections import defaultdict


# Mapping: qa_type -> list of eval_types it can generate
# Based on the generator routing in synth/align/factory.py
QA_TYPE_TO_EVAL_TYPES = {
    "description": ["description"],
    "yes_no": ["yes_no"],
    "correlation": ["correlation"],
    "anticorrelation": ["anticorrelation"],
    "clustering": ["clustering"],
    "anticlustering": ["anticlustering"],
    "segment_mask": [
        "segment_judgment",
        "cross_stat_judgment",
        "anti_judgment",
        "segment_trend_dominance",
        "cross_trend_query",
        "stat_numerical",
        "periodicity",

        "change_point",
        "local_enumeration",
        "segment_enumeration",
        "transition_enumeration",
        "event_segment_enumeration",
        "temporal_position",
        "duration_proportion",
        "cross_metric_enumeration",
    ],
}

# Default equal weights for segment_mask families
SEGMENT_MASK_CORE = [
    "segment_judgment", "cross_stat_judgment", "anti_judgment",
    "segment_trend_dominance", "cross_trend_query",
    "stat_numerical", "periodicity", "change_point",
    "local_enumeration", "segment_enumeration", "transition_enumeration",
    "event_segment_enumeration", "temporal_position", "duration_proportion",
    "cross_metric_enumeration",
]


def estimate_distribution(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    num_data = cfg["num_data"]
    tsevol_ratio = cfg.get("tsevol_ratio", 0.15)
    qa_weights = cfg.get("qa_type_weights", {})
    eval_weights = cfg.get("eval_type_weights", {})
    mode_weights = cfg.get("mode_weights", {"uts": 0.30, "mts_local": 0.35, "mts_shape": 0.35})

    # Split ratios (must match synth/align/dataset.py)
    train_ratio = 0.58
    val_ratio = 0.02
    sft_ratio = 0.20

    total_w = sum(qa_weights.values())
    if abs(total_w - 1.0) > 0.01:
        print(f"WARNING: qa_type_weights sum to {total_w:.3f}, not 1.0")

    # Estimate base seeds (non-tsevol)
    n_seeds = int(num_data * (1 - tsevol_ratio))
    n_tsevol = num_data - n_seeds

    print(f"{'='*70}")
    print(f"  Data Generation Distribution Estimate")
    print(f"{'='*70}")
    print(f"  num_data:      {num_data:,}")
    print(f"  tsevol_ratio:  {tsevol_ratio} ({n_tsevol:,} evolved, {n_seeds:,} seeds)")
    print()

    # Estimate per-eval_type raw counts
    eval_type_counts = defaultdict(float)

    for qa_type, qa_w in qa_weights.items():
        qa_share = n_seeds * (qa_w / total_w)
        eval_types = QA_TYPE_TO_EVAL_TYPES.get(qa_type, [qa_type])

        if len(eval_types) == 1:
            eval_type_counts[eval_types[0]] += qa_share
        else:
            # Apply eval_type_weights for two-level sampling
            et_w = {et: eval_weights.get(et, 1.0) for et in eval_types}
            et_total = sum(et_w.values())
            for et in eval_types:
                eval_type_counts[et] += qa_share * (et_w[et] / et_total)

    # Add tsevol (primarily augments description)
    eval_type_counts["tsevol"] += n_tsevol

    # Sort by count descending
    sorted_types = sorted(eval_type_counts.items(), key=lambda x: -x[1])

    # Compute split counts
    train_frac = train_ratio
    sft_frac = train_frac * sft_ratio
    rl_frac = train_frac * (1 - sft_ratio)
    val_frac = val_ratio
    test_frac = 1.0 - train_ratio - val_ratio

    print(f"  Split ratios: train={train_frac:.0%} (SFT={sft_frac:.0%}, RL={rl_frac:.0%}), "
          f"val={val_frac:.0%} (stratified→1K), test={test_frac:.0%}")
    print()

    print(f"  {'eval_type':<30} {'Total':>8} {'SFT':>8} {'RL':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    total_all = 0
    min_rl = float('inf')
    min_rl_type = ""

    for et, count in sorted_types:
        count = int(count)
        sft_n = int(count * sft_frac)
        rl_n = int(count * rl_frac)
        val_n = min(27, int(count * val_frac))  # stratified cap
        test_n = int(count * test_frac)
        total_all += count

        if rl_n < min_rl and et != "tsevol":
            min_rl = rl_n
            min_rl_type = et

        print(f"  {et:<30} {count:>8,} {sft_n:>8,} {rl_n:>8,} {val_n:>8} {test_n:>8,}")

    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'TOTAL':<30} {total_all:>8,} {int(total_all*sft_frac):>8,} "
          f"{int(total_all*rl_frac):>8,} {'1000':>8} {int(total_all*test_frac):>8,}")

    print()
    print(f"  Rarest type in RL: {min_rl_type} ({min_rl:,} samples)")

    # Training step estimates. Under Option A, the SFT stage IS the SFT-only
    # baseline — same data, same checkpoint, one run.
    sft_samples = int(total_all * sft_frac)
    rl_samples = int(total_all * rl_frac)

    sft_steps = sft_samples * 3 // 192
    rl_steps = rl_samples // 256

    print()
    print(f"  Training step estimates:")
    print(f"    SFT: {sft_samples:,} samples × 3 epochs / batch 192 = {sft_steps:,} steps")
    print(f"    RL:  {rl_samples:,} samples / batch 256 = {rl_steps:,} steps")
    print(f"    (SFT stage doubles as the SFT-only baseline — no separate run.)")
    print()

    # Coverage check
    n_types = len([et for et, c in eval_type_counts.items() if int(c) > 0])
    print(f"  Coverage: {n_types} eval_types with >0 samples")

    if min_rl < 100:
        print(f"  WARNING: {min_rl_type} has only {min_rl} RL samples — consider boosting its weight")
    elif min_rl < 500:
        print(f"  NOTE: {min_rl_type} has {min_rl} RL samples — adequate but thin")
    else:
        print(f"  OK: All types have ≥{min_rl} RL samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate data distribution from config")
    parser.add_argument("--config", default="config/datagen_config.yaml")
    args = parser.parse_args()
    estimate_distribution(args.config)
