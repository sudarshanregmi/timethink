import numpy as np
from typing import Dict, Any, List, Tuple


def calculate_statistics(y: np.ndarray, seq_len: int) -> Dict[str, Any]:
    n = len(y)
    mean_val = float(np.mean(y))
    std_val  = float(np.std(y))
    max_val  = float(np.max(y))
    min_val  = float(np.min(y))

    stats = {
        "mean":    round(mean_val, 2),
        "std":     round(std_val, 2),
        "max":     round(max_val, 2),
        "min":     round(min_val, 2),
        "max_pos": int(np.argmax(y)),
        "min_pos": int(np.argmin(y)),
        "range":   round(max_val - min_val, 2),
        # Quartiles
        "median":  round(float(np.median(y)), 2),
        "q25":     round(float(np.quantile(y, 0.25)), 2),
        "q75":     round(float(np.quantile(y, 0.75)), 2),
    }

    # Outlier detection (beyond ±2σ) — use rounded values so counts match
    # the 2dp mean/std shown in think blocks.
    r_mean_o = round(mean_val, 2)
    r_std_o  = round(std_val, 2)
    if r_std_o > 0:
        outlier_mask = np.abs(y - r_mean_o) > 2 * r_std_o
        stats["outlier_2sigma_count"] = int(np.sum(outlier_mask))
        z_scores = np.abs(y - r_mean_o) / r_std_o
        stats["extreme_outlier_pos"] = int(np.argmax(z_scores))
    else:
        stats["outlier_2sigma_count"] = 0
        stats["extreme_outlier_pos"] = stats["max_pos"]
    stats["outlier_2sigma_fraction"] = round(
        stats["outlier_2sigma_count"] / n, 2
    ) if n > 0 else 0.0

    # Windowed means at two granularities
    for step in (16, 32):
        stats[f"segment_means_{step}"] = [
            round(float(np.mean(y[i:i + step])), 2)
            for i in range(0, n, step)
        ]

    # Mean crossings: sign-changes of (y - rounded_mean) to match think block display.
    deviations = y - round(mean_val, 2)
    sign_changes = np.diff(np.sign(deviations))
    crossing_mask = sign_changes != 0
    stats["mean_crossings"] = int(np.sum(crossing_mask))
    stats["mean_crossing_indices"] = np.where(crossing_mask)[0].tolist()  # 0-indexed

    # Threshold duration counts — use ROUNDED thresholds so counts match
    # the 2dp values shown in think blocks (see design rule #65).
    r_mean = round(mean_val, 2)
    r_std  = round(std_val, 2)
    stats["above_mean_count"]     = int(np.sum(y > r_mean))
    stats["above_mean_std_count"] = int(np.sum(y > r_mean + r_std))
    stats["below_mean_std_count"] = int(np.sum(y < r_mean - r_std))

    # Split-half statistics
    mid = n // 2
    if 0 < mid < n:
        first  = y[:mid]
        second = y[mid:]
        stats["first_half_mean"]  = round(float(np.mean(first)),  2)
        stats["second_half_mean"] = round(float(np.mean(second)), 2)
        stats["first_half_std"]   = round(float(np.std(first)),   2)
        stats["second_half_std"]  = round(float(np.std(second)),  2)
        stats["split_mid"]        = mid

    return stats

def calculate_segment_means(
    time_series: np.ndarray,
    num_segments: int
) -> List[float]:
    seq_len = len(time_series)
    segment_size = max(1, seq_len // num_segments)
    means = [
        round(float(np.mean(time_series[i:i + segment_size])), 2)
        for i in range(0, seq_len, segment_size)
    ]
    return means


def get_actual_segment_count(seq_len: int, num_segments: int) -> int:
    """Return the actual number of segments produced by calculate_segment_means."""
    segment_size = max(1, seq_len // num_segments)
    return len(range(0, seq_len, segment_size))

def get_segment_count(seq_len: int) -> int:
    if seq_len >= 64:
        return 32
    elif seq_len >= 32:
        return 16
    else:
        return max(1, seq_len)

def add_statistics_to_pool(
    attribute_pool: Dict[str, Any],
    y: np.ndarray,
    seq_len: int
) -> None:
    attribute_pool["statistics"] = calculate_statistics(y, seq_len)
    attribute_pool["seq_len"] = seq_len


