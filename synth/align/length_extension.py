"""In-pipeline length-extension for SFT→RL curriculum.

Produces additional samples at expanded seq_len ranges beyond the
default-256-heavy main pool. Called from __main__.py after the main
split + TSEvol, so extension samples do NOT go through TSEvol (raw
paraphrase of original QA is sufficient, LLM cost avoided).

Distributions per split:
  train_rl       : +N samples uniform[32, 768], rl_* only
  val            : +N samples uniform[32, 768], all eval_types
  test_len_near  : N samples uniform[257, 768], all eval_types
  test_len_far   : N samples uniform[769, 1536], all eval_types, channels ≤8
  test_len_extreme: N samples uniform[1537, 4096], all eval_types, channels ≤8

Can also be invoked standalone via scripts/extend_lengths.py for
post-hoc appends to existing JSONL.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

from synth.align.config import (
    Config, Difficulty, Mode, PromptIndexer, SampleResult,
)
from synth.align.factory import generate_sample
from synth.align.dataset import _is_force_sft, balance_binary_results


# ---------- length sampler ----------

def sample_seq_len(min_len: int, max_len: int, default_prob: float,
                   default_len: int = 256) -> int:
    """Bimodal sampler: default_prob -> default_len, else uniform[min, max]."""
    if default_prob > 0.0 and random.random() < default_prob:
        return default_len
    return random.randint(min_len, max_len)


# ---------- worker (module-level for picklability) ----------

_G_CONFIG: Optional[Config] = None
_G_PARAMS: Optional[dict] = None


def _worker_init(config_yaml: str, params: dict, base_seed: int) -> None:
    """Spawned-worker initializer. Loads config and seeds RNG."""
    global _G_CONFIG, _G_PARAMS
    _G_CONFIG = Config.from_yaml(config_yaml)
    _G_PARAMS = params
    seed = base_seed + os.getpid()
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)


def _mode_weights_to_list(config: Config):
    out_modes, out_weights = [], []
    for name, mode in (("uts", Mode.UTS), ("mts_local", Mode.MTS_LOCAL),
                       ("mts_shape", Mode.MTS_SHAPE)):
        w = config.mode_weights.get(name, 0.0)
        if w > 0:
            out_modes.append(mode)
            out_weights.append(w)
    return out_modes, out_weights


def _override_qa_weights_for_rl(config: Config) -> Config:
    """Force qa_type_weights so only rl_composition samples survive."""
    new_weights = {k: 0.0 for k in config.qa_type_weights}
    if "rl_composition" not in new_weights:
        raise KeyError("rl_composition missing from qa_type_weights")
    new_weights["rl_composition"] = 1.0
    return replace(config, qa_type_weights=new_weights)


def _generate_one(_task_id: int):
    """Worker entry: attempt to produce one SampleResult dict, or None."""
    assert _G_CONFIG is not None and _G_PARAMS is not None
    p = _G_PARAMS
    cfg = _G_CONFIG
    if p["qa_filter"] == "rl_only":
        cfg = _override_qa_weights_for_rl(cfg)

    modes, weights = _mode_weights_to_list(cfg)

    for _ in range(5):
        seq_len = sample_seq_len(p["min_len"], p["max_len"], p["default_prob"])
        mode = random.choices(modes, weights=weights, k=1)[0]
        try:
            indexer = PromptIndexer()
            result = generate_sample(
                mode=mode, config=cfg, indexer=indexer,
                seq_len=seq_len, difficulty=Difficulty.EASY,
            )
        except Exception:
            continue

        if p["qa_filter"] == "rl_only":
            if _is_force_sft(result):
                continue
            if not (result.eval_task or "").startswith("rl_"):
                continue

        if p["max_channels"] > 0 and result.original_timeseries is not None:
            if len(result.original_timeseries) > p["max_channels"]:
                continue

        # Return SampleResult directly (picklable — dataclass of numpy + str)
        return result
    return None


# ---------- public API ----------

def extend_with_lengths(
    *,
    config_yaml: str,
    split_label: str,
    target_count: int,
    min_len: int,
    max_len: int,
    default_prob: float = 0.0,
    qa_filter: str = "none",
    max_channels: int = 0,
    workers: Optional[int] = None,
    chunksize: int = 16,
    seed: int = 0xC0FFEE,
) -> List[SampleResult]:
    """Generate length-extended samples via spawned workers.

    Returns a list of SampleResult. Caller is responsible for appending
    them to the appropriate split and writing to disk.
    """
    if workers is None:
        # Cap at 64 — beyond that, spawn overhead + Config.from_yaml per
        # worker dominates, and later pool-creations deadlock under heavy
        # process accumulation. Previously: 128+ workers, several pools
        # per run, hung after ~1–2 pools.
        workers = min(64, max(1, (os.cpu_count() or 8) - 2))

    params = {
        "min_len": min_len,
        "max_len": max_len,
        "default_prob": default_prob,
        "qa_filter": qa_filter,
        "max_channels": max_channels,
    }

    # Over-provision tasks because some attempts fail.
    n_tasks = int(target_count * 1.4)

    logger.info(
        f"[length-extension:{split_label}] target={target_count} "
        f"tasks={n_tasks} workers={workers} "
        f"seq_len∈[{min_len},{max_len}] default_prob={default_prob} "
        f"qa_filter={qa_filter} max_channels={max_channels}"
    )

    ctx = mp.get_context("spawn")
    t0 = time.time()
    collected: List[SampleResult] = []
    pool = ctx.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(config_yaml, params, seed),
    )
    try:
        for result in pool.imap_unordered(
            _generate_one, range(n_tasks), chunksize=chunksize
        ):
            if result is None:
                continue
            collected.append(result)
            if len(collected) >= target_count:
                break
    finally:
        # Hot-stop pattern: terminate workers OUTSIDE the iterator loop
        # (calling terminate() while imap_unordered is being consumed can
        # leave the iterator and its shm queue in an inconsistent state,
        # deadlocking the subsequent pool.join()). terminate() here is
        # safe because we've already broken out of the iterator.
        pool.terminate()
        pool.join()

    elapsed = time.time() - t0
    logger.info(
        f"[length-extension:{split_label}] produced {len(collected)} samples "
        f"in {elapsed:.1f}s ({len(collected)/max(elapsed, 1e-3):.1f}/s)"
    )

    # Source-level 50/50 verdict balance for binary types — at long seq_len
    # some RL generators skew (e.g. rl_monotonic_chunks → "no", because
    # strict monotonicity over 48 chunks is rare). Without this, the model
    # can learn to always output the majority verdict and collect reward.
    before = len(collected)
    collected = balance_binary_results(collected)
    if len(collected) != before:
        logger.info(
            f"[length-extension:{split_label}] post-balance: "
            f"{before} -> {len(collected)} samples"
        )
    return collected
