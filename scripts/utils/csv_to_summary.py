"""
Convert the raw per-sample evaluation_report_optimized.csv into:
  1. Per-category aggregated CSVs (saved alongside the input CSV)
  2. The summary txt file (identical to what main.py produces)

Usage:
    python scripts/utils/csv_to_summary.py exp/rl/evaluation_report_optimized.csv
    python scripts/utils/csv_to_summary.py exp/sft/evaluation_report_optimized.csv

Outputs (all in same directory as input CSV):
    summary_by_task_type.csv        — f1, binary_accuracy, reasoning_score, count, description_overall
    clustering_by_gt_length.csv     — f1, precision, recall, sample_count  (only if clustering rows exist)
    description_perspectives.csv    — cls, num, sub-metric breakdown per perspective
    description_trend_detail.csv    — direction_cls acc, segment P/R/F1/boundary_pos_acc
    evaluation_report_summary.txt   — human-readable summary (mirrors main.py output)
"""

import sys
import os
import pandas as pd

# Allow running from repo root: python scripts/utils/csv_to_summary.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from evaluation.eval.aggregation import save_aggregated_csvs, write_summary_txt


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/utils/csv_to_summary.py <path_to_evaluation_report_optimized.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(csv_path))
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    save_aggregated_csvs(df, out_dir)

    txt_path = os.path.join(out_dir, "evaluation_report_summary.txt")
    write_summary_txt(df, txt_path)

    print(f"Outputs saved to {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
