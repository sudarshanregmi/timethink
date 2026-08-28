"""Text rendering of `<ts>` placeholders for non-multimodal baselines.

Our ChatTS-style model consumes time series via a multi-modal embedding layer:
the input prompt contains `<ts>` placeholders that the vLLM `vllm-ts` engine
maps onto float tensors carried in `multi_modal_data`.

Text-only baselines (e.g. TimeOmni-1, plain Qwen-Instruct) have no embedding
layer for time series. They expect the numeric values inlined into the user
prompt as Python lists, e.g.

    "...The time series of J96A is: [3.35, 2.92, 2.61, ...]..."

`inline_timeseries_as_text` walks each `<ts>` placeholder and substitutes the
corresponding array (rounded to a fixed precision, default dp=2 to match the
representation used by TimeOmni-1's released inference examples).

The function is engine-agnostic; the caller decides whether to consume the
rendered text (text-only path) or keep the original `<ts>` prompt (multi-modal
path). Used by `synth.utils.inference_tsmllm_vllm` for the `timeomni-1-7b`
baseline integration.
"""
from __future__ import annotations
from typing import Iterable, List, Sequence
import numpy as np


PLACEHOLDER = "<ts>"


def _format_ts_array(arr: Sequence[float], dp: int) -> str:
    """Render a single 1-D series as a Python-style list literal.

    Uses fixed-point with `dp` decimals to keep tokens predictable and
    match the format TimeOmni-1's example prompt ships with.
    """
    a = np.asarray(arr, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{v:.{dp}f}" for v in a) + "]"


def inline_timeseries_as_text(
    input_text: str,
    ts_list: Iterable[Sequence[float]],
    dp: int = 2,
) -> str:
    """Replace each `<ts>` placeholder with the next array in `ts_list`.

    Order matches positional order of `<ts>` occurrences in `input_text`,
    which is the same order the multi-modal pipeline assumes.

    Raises:
        ValueError: if placeholder count != len(ts_list).
    """
    ts_list = list(ts_list)
    n_ph = input_text.count(PLACEHOLDER)
    if n_ph != len(ts_list):
        raise ValueError(
            f"<ts> placeholder count ({n_ph}) != timeseries count ({len(ts_list)})"
        )
    out = input_text
    for arr in ts_list:
        out = out.replace(PLACEHOLDER, _format_ts_array(arr, dp), 1)
    return out
