"""Build the public train_rl parquet from a stopped RL run.

Two phases (run with --phase filter, then --phase build):

  1. filter: re-run TSRLHFDataset's overlong-prompt filter on train_rl.parquet
     and cache the surviving original-row indices to a .npy file.
     (Slow — minutes to tens of minutes. Only needs to run once per parquet.)

  2. build: load the saved StatefulDataLoader state from
     outputs/.../global_step_K/data.pt, replay the dataloader to enumerate
     never-visited filtered indices, map back to original rows, and write
     a new parquet containing only visited rows.

The resulting parquet is the actual training data the model saw — safe to
release publicly.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def phase_filter(args):
    from omegaconf import OmegaConf
    from transformers import AutoProcessor, AutoTokenizer

    from ts_data import TSRLHFDataset

    tokenizer = AutoTokenizer.from_pretrained(args.sft_ckpt, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.sft_ckpt, trust_remote_code=True)

    cfg = OmegaConf.create({
        "prompt_key": "prompt",
        "image_key": "images",
        "video_key": "videos",
        "max_prompt_length": args.max_prompt_length,
        "truncation": "error",
        "filter_overlong_prompts": True,
        "filter_overlong_prompts_workers": args.num_workers,
        "trust_remote_code": True,
        "apply_chat_template_kwargs": {},
        "cache_dir": "~/.cache/verl/rlhf",
        "use_shm": False,
        "return_raw_chat": False,
        "return_full_prompt": False,
        "return_multi_modal_inputs": True,
        "shuffle": True,
        "seed": None,
        "timeseries_key": "timeseries",
    })

    ds = TSRLHFDataset(args.parquet, tokenizer, cfg, processor)

    df = ds.dataframe
    if df._indices is not None:
        kept_orig = np.asarray(df._indices.column(0).to_pylist(), dtype=np.int64)
    else:
        kept_orig = np.arange(len(df), dtype=np.int64)

    print(f"[filter] kept {len(kept_orig)} of {pq.read_metadata(args.parquet).num_rows} rows")
    np.save(args.filtered_indices, kept_orig)
    print(f"[filter] wrote {args.filtered_indices}")


def _write_jsonl_from_parquet(table, out_path, chunk_size=2000):
    """Reconstruct the original JSONL schema from the verl-preprocessed parquet.

    Reverse of preprocess.py::process_rl_row + _base_fields:
        input          <- prompt[0]['content']
        output         <- json.loads(reward_model)['ground_truth']
        timeseries     <- timeseries  (list of list of float)
        eval_type      <- json.loads(extra_info)['eval_type']
        eval_metadata  <- json.loads(extra_info)['eval_metadata']
    """
    n = table.num_rows
    with open(out_path, "w") as f:
        for start in range(0, n, chunk_size):
            chunk = table.slice(start, min(chunk_size, n - start)).to_pylist()
            for row in chunk:
                extra = json.loads(row["extra_info"])
                rm = json.loads(row["reward_model"])
                f.write(json.dumps({
                    "input": row["prompt"][0]["content"],
                    "output": rm["ground_truth"],
                    "timeseries": row["timeseries"],
                    "eval_type": extra.get("eval_type", ""),
                    "eval_metadata": extra.get("eval_metadata", {}),
                }) + "\n")


def phase_build(args):
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torchdata.stateful_dataloader.sampler import RandomSampler

    kept_orig = np.load(args.filtered_indices)
    filtered_size = len(kept_orig)

    sd = torch.load(args.ckpt_data_pt, weights_only=False)
    snapshot_step = sd["_snapshot"]["_snapshot_step"]
    samples_yielded = sd["_snapshot"]["_main_snapshot"]["_sampler_iter_state"]["samples_yielded"]
    expected = snapshot_step * args.batch_size
    assert samples_yielded == expected, (
        f"data.pt samples_yielded={samples_yielded} != snapshot_step*batch_size={expected}; "
        "did you change batch_size between training and replay?"
    )
    train_epoch_batches = filtered_size // args.batch_size  # drop_last=True at training
    print(f"[build] filtered pool: {filtered_size}, batch={args.batch_size}, "
          f"train_epoch_batches={train_epoch_batches} (drop_last=True), "
          f"snapshot_step={snapshot_step}")

    class _IndexDS(Dataset):
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return int(i)

    # Replay with drop_last=False so partial-batch orphans (filtered_size %
    # batch_size) are also surfaced — those indices were yielded by the sampler
    # in epoch 1 but discarded by the trainer's drop_last=True. They are
    # equally "unvisited" from the model's perspective.
    ds = _IndexDS(filtered_size)
    sampler = RandomSampler(data_source=ds, generator=torch.Generator())
    loader = StatefulDataLoader(
        dataset=ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        sampler=sampler,
    )
    loader.load_state_dict(sd)

    unvisited_filtered = []
    for batch in loader:
        if isinstance(batch, torch.Tensor):
            unvisited_filtered.extend(int(x) for x in batch.tolist())
        else:
            unvisited_filtered.extend(int(x) for x in batch)
    unvisited_filtered = np.asarray(unvisited_filtered, dtype=np.int64)

    expected_unvisited = filtered_size - samples_yielded
    assert len(unvisited_filtered) == expected_unvisited, (
        f"replay yielded {len(unvisited_filtered)} unvisited samples, "
        f"expected {expected_unvisited}. Sampler state did not load cleanly."
    )

    # uniqueness check: indices yielded post-resume should not collide
    if len(set(unvisited_filtered.tolist())) != len(unvisited_filtered):
        raise SystemExit("Duplicate indices in replay — sampler state load is wrong.")

    unvisited_orig = kept_orig[unvisited_filtered]
    visited_orig = np.setdiff1d(kept_orig, unvisited_orig, assume_unique=True)
    expected_visited = filtered_size - expected_unvisited
    assert len(visited_orig) == expected_visited, (
        f"visited count {len(visited_orig)} != expected {expected_visited}"
    )

    print(f"[build] unvisited (drop): {len(unvisited_orig)}, visited (publish): {len(visited_orig)}")

    table = pq.read_table(args.parquet)
    public = table.take(visited_orig)
    pq.write_table(public, args.out)
    print(f"[build] wrote {args.out} ({public.num_rows} rows)")

    if args.out_jsonl:
        _write_jsonl_from_parquet(public, args.out_jsonl)
        print(f"[build] wrote {args.out_jsonl} ({public.num_rows} rows)")

    report = {
        "parquet_in": str(args.parquet),
        "parquet_out": str(args.out),
        "ckpt_data_pt": str(args.ckpt_data_pt),
        "filtered_indices_cache": str(args.filtered_indices),
        "rows_total_in_parquet": int(table.num_rows),
        "rows_dropped_overlong_filter": int(table.num_rows - filtered_size),
        "rows_filtered_pool": int(filtered_size),
        "rows_unvisited_dropped": int(len(unvisited_orig)),
        "rows_visited_published": int(len(visited_orig)),
        "snapshot_step": int(snapshot_step),
        "batch_size": int(args.batch_size),
        "samples_yielded_in_ckpt": int(samples_yielded),
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"[build] wrote report to {args.report}")
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["filter", "build", "all"], required=True)
    ap.add_argument("--parquet", default=str(REPO_ROOT / "data/train_rl.parquet"))
    ap.add_argument("--filtered-indices", default=str(REPO_ROOT / "data/train_rl_filter_indices.npy"))
    ap.add_argument("--sft-ckpt", default=None,
                    help="HF dir for tokenizer/processor; required for --phase filter.")
    ap.add_argument("--max-prompt-length", type=int, default=8192)
    ap.add_argument("--ckpt-data-pt", default=None,
                    help="Path to outputs/.../global_step_K/data.pt; required for --phase build.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default=str(REPO_ROOT / "data/train_rl_public.parquet"))
    ap.add_argument("--out-jsonl", default=str(REPO_ROOT / "data/train_rl_public.jsonl"),
                    help="Also write a JSONL with the original {input, output, timeseries, "
                         "eval_type, eval_metadata} schema. Pass empty string to skip.")
    ap.add_argument("--report", default=str(REPO_ROOT / "data/train_rl_public_report.json"))
    args = ap.parse_args()

    if args.phase in ("filter", "all"):
        if not args.sft_ckpt:
            raise SystemExit("--sft-ckpt required for filter phase")
        phase_filter(args)
    if args.phase in ("build", "all"):
        if not args.ckpt_data_pt:
            raise SystemExit("--ckpt-data-pt required for build phase")
        phase_build(args)


if __name__ == "__main__":
    main()
