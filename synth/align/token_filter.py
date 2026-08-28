"""Token length filtering for SampleResult lists.

Computes exact token counts accounting for chat template overhead and
time series patch expansion, then filters samples exceeding max_token_length.

Token count formula per sample:
  text_tokens(chat_template(input, output))
  + sum_over_ts(ceil(len(ts) / patch_size) - 1 + prefix_tokens_for_this_ts)

The -1 accounts for the single <ts> already present in the input text per TS.
The prefix tokens cover the [offset=X.XXXX|scaling=X.XXXX] prefix that
the processor inserts before each <ts> at inference time. Prefix token count
varies with float magnitude (18–26 tokens), so it is computed per TS.
"""

import math
from typing import List

import numpy as np
from loguru import logger
from transformers import AutoTokenizer

from synth.align.config import SampleResult

_PATCH_SIZE = 8


def _prefix_token_count(ts: np.ndarray, tokenizer: AutoTokenizer) -> int:
    """Compute exact prefix token count for a single time series.

    Replicates the sp_encoding prefix logic from processing_qwen3_ts.py:
      [offset={-mean:.4f}|scaling={scale_factor:.4f}]
    """
    mean = float(np.mean(ts))
    scaled = ts - mean
    scale_factor = 1.0
    if np.any(np.abs(scaled) >= 3.0):
        scale_factor = float(np.max(np.abs(scaled))) / 3.0
    prefix = f"[offset={-mean:.4f}|scaling={scale_factor:.4f}]"
    return len(tokenizer.encode(prefix, add_special_tokens=False))


def _compute_sample_tokens(
    result: SampleResult,
    tokenizer: AutoTokenizer,
) -> int:
    """Compute total token count for a SampleResult (prompt + response + TS expansion)."""
    input_text = f"{result.base_prompt.rstrip(';.')}. {result.questions[0]}"
    output_text = result.answers[0]

    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    text_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))

    # TS expansion: each TS of length L contributes ceil(L/8)-1 extra <ts> tokens
    # plus prefix tokens for [offset=...|scaling=...] (varies with float magnitude)
    ts_extra = 0
    for ts in result.original_timeseries:
        num_patches = math.ceil(len(ts) / _PATCH_SIZE)
        ts_extra += (num_patches - 1) + _prefix_token_count(ts, tokenizer)

    return text_tokens + ts_extra


def filter_by_token_length(
    results: List[SampleResult],
    tokenizer: AutoTokenizer,
    max_tokens: int,
    stage: str = "",
) -> List[SampleResult]:
    """Filter out samples whose total token count exceeds max_tokens.

    Args:
        results: List of SampleResult to filter.
        tokenizer: Tokenizer for text tokenization.
        max_tokens: Maximum allowed token count (inclusive).
        stage: Label for log messages (e.g. "seed", "post-tsevol").

    Returns:
        Filtered list (order preserved, overlength samples removed).
    """
    kept = []
    dropped = 0
    max_seen = 0

    for r in results:
        n_tokens = _compute_sample_tokens(r, tokenizer)
        max_seen = max(max_seen, n_tokens)
        if n_tokens <= max_tokens:
            kept.append(r)
        else:
            dropped += 1

    label = f"[{stage}] " if stage else ""
    logger.info(
        f"{label}Token filter: {len(results)} → {len(kept)} samples "
        f"(dropped {dropped}, max_tokens={max_tokens}, longest={max_seen})"
    )
    return kept
