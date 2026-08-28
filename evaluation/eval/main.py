import argparse
import json
import asyncio
import os
import time
import pandas as pd
from loguru import logger

from evaluation.eval.config import MTYPE
from evaluation.eval.prompts import prepare_prompts
from evaluation.eval.inference import run_massive_inference
from evaluation.eval.parser import parse_and_score
from evaluation.eval.aggregation import save_aggregated_csvs, write_summary_txt


def run(exp_dir: str):
    """Score generated_answer.json inside exp_dir using the LLM-as-judge pipeline."""
    exp_dir = os.path.abspath(exp_dir)
    results_path = os.path.join(exp_dir, "generated_answer.json")
    output_csv = os.path.join(exp_dir, "evaluation_report_optimized.csv")

    if not os.path.exists(results_path):
        logger.error(f"File not found: {results_path}")
        return

    logger.info(f"Input:  {results_path}")

    with open(results_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    prepared_tasks = prepare_prompts(data_list)
    start_time = time.time()

    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("Using uvloop for high performance asyncio.")
    except ImportError:
        pass

    raw_results = asyncio.run(run_massive_inference(prepared_tasks))
    duration = time.time() - start_time
    logger.info(f"Inference complete. Time: {duration:.2f}s")
    metrics_data = parse_and_score(raw_results)
    df = pd.DataFrame(metrics_data)

    # Save raw per-sample CSV
    df.to_csv(output_csv, index=False)

    # Save aggregated per-category CSVs
    save_aggregated_csvs(df, exp_dir)

    # Save human-readable summary txt
    summary_txt_path = os.path.join(exp_dir, "evaluation_report_summary.txt")
    write_summary_txt(df, summary_txt_path)

    with open(summary_txt_path, "r") as f:
        print(f.read())

    logger.info(f"Detailed CSV report saved to: {output_csv}")
    logger.info(f"Aggregated CSVs saved to: {exp_dir}/")
    logger.info(f"Summary text report saved to: {summary_txt_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Run LLM-as-judge scoring on a generated_answer.json."
    )
    ap.add_argument(
        "--exp-dir",
        default=None,
        help=(
            "Directory containing generated_answer.json. "
            "Default: exp/{MTYPE} from config for back-compat. "
            "Use exp/{model}/{dataset_stem} for the per-dataset layout."
        ),
    )
    args = ap.parse_args()

    exp_dir = args.exp_dir or os.path.abspath(f"exp/{MTYPE}")
    run(exp_dir)


if __name__ == "__main__":
    main()
