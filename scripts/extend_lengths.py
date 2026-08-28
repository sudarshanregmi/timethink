"""Length-extension data generator.

Generates additional JSONL samples at specified seq_len ranges to
complement the existing 256-dominant data. Used for:
  - RL phase:    uniform[32, 768] rl_* samples to give RL length diversity
  - Val:         uniform[32, 768] mixed samples to track length generalization
  - Test OOD:    [257, 768] / [769, 1536] / [1537, 4096] buckets for
                 measuring length extrapolation at eval time

Writes JSONL in the same schema as `synth.align` produces (one record
per sample: input/output/timeseries/eval_type/eval_metadata). Designed to
be APPENDED to existing split files via the --append flag.

Usage examples:
    # Extend train_rl with 48K rl_* samples uniform[32, 768]
    python scripts/extend_lengths.py \\
        --split train_rl --num 48000 \\
        --min-len 32 --max-len 768 --default-prob 0.0 \\
        --qa-filter rl_only \\
        --output data/train_rl.jsonl --append

    # Build test_len_near from scratch
    python scripts/extend_lengths.py \\
        --split test_len_near --num 3000 \\
        --min-len 257 --max-len 768 --default-prob 0.0 \\
        --output data/test_len_near.jsonl

    # Cap channels at 8 for long-length test (MTS token budget)
    python scripts/extend_lengths.py \\
        --split test_len_extreme --num 1500 \\
        --min-len 1537 --max-len 4096 --default-prob 0.0 \\
        --max-channels 8 \\
        --output data/test_len_extreme.jsonl

The script bypasses TSEvol and the token filter (token filter should be
redundant since we cap seq_len and channels); those are SFT-phase concerns
only. Channel cap is enforced by rejecting MTS samples with more than
--max-channels timeseries.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

os.environ.setdefault("LOGURU_LEVEL", "ERROR")

# Repo root on path so workers can import synth.*. Also chdir there so any
# module-level `open("config/...")` relative paths in the chatts package
# (e.g. synth/utils/llm_utils.py loads config/datagen_config.yaml at
# import time) resolve correctly regardless of caller's CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(str(_REPO_ROOT))

from synth.align.config import Config, Mode, PromptIndexer, Difficulty  # noqa: E402
from synth.align.factory import generate_sample  # noqa: E402
from synth.align.dataset import _is_force_sft  # noqa: E402
from synth.ts_generator.utils.common_utils import NumpyEncoder  # noqa: E402


# ---------- length sampler ----------

def sample_seq_len(min_len: int, max_len: int, default_prob: float,
                   default_len: int = 256) -> int:
    """Pick a sequence length.

    With probability default_prob return default_len (256). Otherwise
    uniform int in [min_len, max_len]. If default_prob == 0.0 the sampler
    never returns the default.
    """
    if default_prob > 0.0 and random.random() < default_prob:
        return default_len
    return random.randint(min_len, max_len)


# ---------- worker ----------

# Worker-global config (set via initializer to avoid pickling per task).
_G_CONFIG: Optional[Config] = None
_G_ARGS: Optional[argparse.Namespace] = None


def _worker_init(config_yaml: str, args_dict: dict, base_seed: int) -> None:
    global _G_CONFIG, _G_ARGS
    _G_CONFIG = Config.from_yaml(config_yaml)
    _G_ARGS = argparse.Namespace(**args_dict)
    # Per-worker RNG seed so samples differ across workers
    seed = base_seed + os.getpid()
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)


def _mode_weights_to_list(config: Config):
    mw = config.mode_weights
    modes = []
    weights = []
    for name, mode in (("uts", Mode.UTS), ("mts_local", Mode.MTS_LOCAL),
                       ("mts_shape", Mode.MTS_SHAPE)):
        w = mw.get(name, 0.0)
        if w > 0:
            modes.append(mode)
            weights.append(w)
    return modes, weights


def _override_qa_weights_for_rl(config: Config) -> Config:
    """Force qa_type_weights to select only rl_composition (non-bridge rl_*)."""
    new_weights = {k: 0.0 for k in config.qa_type_weights}
    if "rl_composition" not in new_weights:
        # Fail fast if schema changed
        raise KeyError("rl_composition missing from qa_type_weights")
    new_weights["rl_composition"] = 1.0
    return replace(config, qa_type_weights=new_weights)


def _generate_one(task_id: int) -> Optional[dict]:
    """Attempt to produce one sample. Returns record dict or None."""
    assert _G_CONFIG is not None and _G_ARGS is not None
    args = _G_ARGS
    cfg = _G_CONFIG
    if args.qa_filter == "rl_only":
        cfg = _override_qa_weights_for_rl(cfg)

    modes, weights = _mode_weights_to_list(cfg)

    # Up to 5 attempts per task slot before giving up — some combinations
    # (e.g., periodicity at short seq_len) can fail.
    for _ in range(5):
        seq_len = sample_seq_len(args.min_len, args.max_len, args.default_prob)
        mode = random.choices(modes, weights=weights, k=1)[0]
        try:
            indexer = PromptIndexer()
            result = generate_sample(
                mode=mode, config=cfg, indexer=indexer,
                seq_len=seq_len, difficulty=Difficulty.EASY,
            )
        except Exception:
            continue

        # Filter: rl_only means result must be rl_* non-bridge
        if args.qa_filter == "rl_only":
            if _is_force_sft(result):
                continue
            if not (result.eval_task or "").startswith("rl_"):
                continue
        elif args.qa_filter == "ood_only":
            # OOD-only: post-filter to keep eval_task starting with ood_.
            # OOD types are eval-only (force_sft); we don't override
            # qa_type_weights since OOD is generated via the existing
            # ood_eval qa_type alongside everything else.
            if not (result.eval_task or "").startswith("ood_"):
                continue

        # Channel cap
        if args.max_channels > 0 and result.original_timeseries is not None:
            if len(result.original_timeseries) > args.max_channels:
                continue

        # Build JSONL record in same schema as write_split_files
        ts_len = (len(result.original_timeseries[0])
                  if result.original_timeseries else None)
        eval_meta = dict(result.eval_metadata) if result.eval_metadata else {}
        if ts_len is not None and "length" not in eval_meta:
            eval_meta["length"] = ts_len

        raw_input = f"{result.base_prompt.rstrip(';.')}. {result.questions[0]}"
        record = {
            "input": raw_input,
            "output": result.answers[0],
            "timeseries": [list(map(float, ts))
                           for ts in result.original_timeseries],
            "eval_type": result.eval_task or result.qa_types[0],
            "eval_metadata": eval_meta,
        }
        return record

    return None


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--split", required=True,
                   help="Label for logging (e.g., train_rl, test_len_near)")
    p.add_argument("--num", type=int, required=True,
                   help="Target sample count")
    p.add_argument("--min-len", type=int, required=True)
    p.add_argument("--max-len", type=int, required=True)
    p.add_argument("--default-prob", type=float, default=0.0,
                   help="Probability of picking default_len=256 (0 = never)")
    p.add_argument("--qa-filter", choices=["none", "rl_only", "ood_only"], default="none",
                   help="rl_only: keep only rl_* non-bridge samples; "
                        "ood_only: keep only eval_type starting with ood_")
    p.add_argument("--max-channels", type=int, default=0,
                   help="Cap MTS channel count (0 = no cap)")
    p.add_argument("--output", required=True,
                   help="Output JSONL path")
    p.add_argument("--append", action="store_true",
                   help="Append to existing file (default: overwrite)")
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 8) - 2))
    p.add_argument("--config",
                   default=str(_REPO_ROOT / "config" / "datagen_config.yaml"))
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--chunksize", type=int, default=16,
                   help="imap_unordered chunksize")
    p.add_argument("--skip-rebalance", action="store_true",
                   help="Skip the post-generation 50/50 verdict rebalance")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"

    # Over-provision task count because some attempts fail
    oversubscribe = 1.4
    n_tasks = int(args.num * oversubscribe)

    print(f"[extend_lengths] split={args.split} target={args.num} "
          f"tasks={n_tasks} workers={args.workers}")
    print(f"[extend_lengths] seq_len ~ uniform[{args.min_len}, {args.max_len}] "
          f"(default_prob={args.default_prob})")
    print(f"[extend_lengths] qa_filter={args.qa_filter} "
          f"max_channels={args.max_channels}")
    print(f"[extend_lengths] output={out_path}  mode={'append' if args.append else 'write'}")

    t0 = time.time()
    written = 0
    # Spawn to avoid fork-after-numpy/CUDA quirks (defensive)
    ctx = mp.get_context("spawn")
    args_dict = vars(args).copy()
    # argparse Namespace attrs use hyphen→underscore for attr access
    args_dict = {k.replace("-", "_"): v for k, v in args_dict.items()}
    with open(out_path, mode, encoding="utf-8") as f:
        with ctx.Pool(
            processes=args.workers,
            initializer=_worker_init,
            initargs=(args.config, args_dict, args.seed),
        ) as pool:
            for rec in pool.imap_unordered(_generate_one,
                                            range(n_tasks),
                                            chunksize=args.chunksize):
                if rec is None:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False, cls=NumpyEncoder) + "\n")
                written += 1
                if written % 1000 == 0:
                    rate = written / max(1.0, time.time() - t0)
                    print(f"  [{args.split}] {written}/{args.num}  ({rate:.1f} samples/s)")
                if written >= args.num:
                    pool.terminate()
                    break

    elapsed = time.time() - t0
    print(f"[extend_lengths] done. wrote {written} samples to {out_path} "
          f"in {elapsed:.1f}s ({written/elapsed:.1f}/s)")

    # Source-level 50/50 verdict balance for binary types. At long seq_len
    # some RL generators skew (rl_monotonic_chunks, rl_has_periodicity,
    # rl_cross_stat_ratio lean "no"). Without this, the model can learn to
    # always output the majority verdict and collect reward.
    if not args.skip_rebalance:
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from rebalance_jsonl import rebalance_file
        print(f"[extend_lengths] rebalancing {out_path}...")
        rebalance_file(out_path, seed=args.seed, backup=False, verbose=False)
        n_final = sum(1 for _ in open(out_path))
        print(f"[extend_lengths] post-rebalance: {n_final} samples")


if __name__ == "__main__":
    main()
