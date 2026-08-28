"""Configuration and data structures for QA generation."""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from synth.ts_generator.utils.common_utils import load_cfg
from synth.ts_generator.utils.probability_utils import DEFAULT_QA_TYPE_WEIGHTS


DEFAULT_MODE_WEIGHTS: Dict[str, float] = {
    "uts": 0.30,
    "mts_local": 0.35,
    "mts_shape": 0.35,
}


def _validate_qa_type_weights(weights: Dict[str, float], config_path: str) -> None:
    """Fail-fast if qa_type_weights is out of sync with the QAType enum.

    Guards against the silent-drop class of bug: if the codebase emits a
    qa_type that isn't in the yaml, `select_weighted_qa_index` would give
    it weight 0 and discard every sample of that type. This check aborts
    generation loudly instead.

    Also warns (but does not abort) on dead weights — yaml keys that match
    no QAType enum value — to catch renames/removals without blocking runs.
    """
    # Avoid circular import at module top
    from synth.ts_generator.utils.probability_utils import QAType

    code_types = {qt.value for qt in QAType}
    yaml_types = set(weights.keys())

    missing_in_yaml = code_types - yaml_types
    if missing_in_yaml:
        raise ValueError(
            f"qa_type_weights in {config_path} is missing entries for "
            f"qa_types the codebase emits: {sorted(missing_in_yaml)}. "
            "Without explicit weights, select_weighted_qa_index would drop "
            "every sample of these types during production generation. Add "
            "weights for each (or update the QAType enum if the type was "
            "renamed/removed)."
        )

    dead_in_yaml = yaml_types - code_types
    if dead_in_yaml:
        import warnings
        warnings.warn(
            f"qa_type_weights in {config_path} has entries not recognized "
            f"by the QAType enum: {sorted(dead_in_yaml)}. These are dead "
            "weights and will not affect generation. Remove them or rename.",
            RuntimeWarning,
        )

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-3:
        import warnings
        warnings.warn(
            f"qa_type_weights in {config_path} sum to {total:.4f}, not 1.0. "
            "Weights will be normalized at selection time, but explicit "
            "normalization in yaml makes intent clearer.",
            RuntimeWarning,
        )


class Mode(Enum):
    """Generation mode for time series QA."""
    UTS = "uts"           # Single time series
    MTS_LOCAL = "local"   # Multiple TS, local event correlation
    MTS_SHAPE = "shape"   # Multiple TS, trend/shape correlation


class Difficulty(Enum):
    """Difficulty level for yes/no QA generation.

    Controls WHERE the query point is placed relative to the threshold boundary.
    50/50 verdict balance is maintained at all difficulty levels.

    EASY:   Clear-cut — yes points well inside threshold, no points far from events.
    MEDIUM: Boundary — yes points near threshold edge, no points just past it.
    HARD:   Ambiguous — yes/no points barely inside/outside threshold.
            MTS adds cross-metric confusion (other metrics active at query point).
    """
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"



@dataclass
class Config:
    """Configuration for LLM QA generation."""
    num_data: int
    encoding_method: str
    output_base_dir: str
    dryrun: bool
    debug: bool
    local_llm_path: str
    disable_metric_config: bool
    metric_config: List[Dict]
    strip: bool = False
    debug_filter: Optional[List[str]] = None
    qa_type_weights: Dict[str, float] = None
    mode_weights: Dict[str, float] = None
    eval_type_weights: Dict[str, float] = None
    # --- LLM / GPU config (read from YAML, used by LLMClient) ---
    num_gpus: int = 4
    gpu_per_model: int = 1
    ctx_length: int = 30000
    batch_size: int = 32
    engine: str = "vllm"
    tsevol_ratio: float = 0.3
    # --- Token length filtering (training model tokenizer) ---
    max_token_length: int = 8192
    tokenizer_path: str = "ickpt"
    # --- OOD generation cap (0 = unlimited) ---
    max_ood_samples: int = 10000
    # --- Minimum samples per eval_type (top-up generation fills deficits) ---
    min_samples_per_eval_type: int = 500
    # --- Length-extension curriculum (Option C, 2026-04-21) ---
    # Main pool is capped at this length so SFT data stays in-curriculum
    # (≤256 only). The length_extension phase adds longer samples to
    # RL + val + OOD test buckets. Set to 4096 to disable the curriculum
    # and generate old-style mixed-length main pool.
    main_pool_max_seq_len: int = 256
    # Post-split, post-TSEvol phase that appends length-diverse samples to
    # train_rl + val (50% at 256, 50% uniform[32, 768]) and emits OOD test
    # buckets (near/far/extreme). See synth/align/length_extension.py.
    length_extension: Dict = None

    @property
    def num_data_seed(self) -> int:
        """Seeds kept unmodified (70% of num_data when tsevol_ratio=0.3)."""
        return self.num_data - self.num_data_tsevol

    @property
    def num_data_tsevol(self) -> int:
        """Target number of seeds to replace with evolved QA (30% of num_data)."""
        return int(self.num_data * self.tsevol_ratio)

    @classmethod
    def from_yaml(cls, config_path: str = "config/datagen_config.yaml") -> "Config":
        """Load configuration from YAML file."""
        cfg = load_cfg(config_path)
        metric_config = load_cfg("config/metric_set.json")
        qa_type_weights = cfg.get("qa_type_weights", DEFAULT_QA_TYPE_WEIGHTS)
        _validate_qa_type_weights(qa_type_weights, config_path)
        return cls(
            num_data=cfg["num_data"],
            encoding_method=cfg["encoding_method"],
            output_base_dir=cfg["data_output_dir"],
            dryrun=cfg["dryrun"],
            debug=cfg.get("debug", False),
            local_llm_path=cfg["local_llm_path"],
            disable_metric_config=cfg.get("disable_metric_config", False),
            metric_config=metric_config,
            qa_type_weights=qa_type_weights,
            mode_weights=cfg.get("mode_weights", DEFAULT_MODE_WEIGHTS),
            eval_type_weights=cfg.get("eval_type_weights", {}),
            num_gpus=cfg.get("num_gpus", 4),
            gpu_per_model=cfg.get("gpu_per_model", 1),
            ctx_length=cfg.get("ctx_length", 30000),
            batch_size=cfg.get("batch_size", 32),
            engine=cfg.get("engine", "vllm"),
            tsevol_ratio=cfg.get("tsevol_ratio", 0.3),
            max_token_length=cfg.get("max_token_length", 8192),
            tokenizer_path=cfg.get("tokenizer_path", "ickpt"),
            max_ood_samples=cfg.get("max_ood_samples", 10000),
            min_samples_per_eval_type=cfg.get("min_samples_per_eval_type", 500),
            main_pool_max_seq_len=cfg.get("main_pool_max_seq_len", 256),
            length_extension=cfg.get("length_extension", None),
        )

    @property
    def output_path(self) -> Path:
        """Path for main JSONL output."""
        return Path(self.output_base_dir) / "qa.jsonl"


class PromptIndexer:
    """Helper for managing placeholder prompts."""

    def __init__(self):
        self._idx = 0

    def next(self) -> int:
        """Get next index and increment."""
        i = self._idx
        self._idx += 1
        return i

    def current(self) -> int:
        """Get current index without incrementing."""
        return self._idx

    def reset(self):
        """Reset index to zero."""
        self._idx = 0

    def placeholder(self, idx: int = None) -> str:
        """Generate placeholder string."""
        return f"<|prompt{idx if idx is not None else self.next()}|>"


@dataclass
class SampleResult:
    """Unified result structure for all generation modes."""
    mode: Mode
    original_timeseries: List[np.ndarray]
    encoded_timeseries: List[np.ndarray]
    metrics: List[str]
    attributes: List[Dict]
    base_prompt: str
    questions: List[str]
    answers: List[str]
    llm_prompts: List[List[str]]
    fields: List[Dict]
    corr_pool: List[Any]
    label: Dict
    qa_types: List[str] = field(default_factory=list)
    eval_tasks: List[str] = field(default_factory=list)
    eval_metadatas: List[Dict[str, Any]] = field(default_factory=list)
    eval_task: Optional[str] = None
    eval_metadata: Optional[Dict[str, Any]] = None
    situation: Optional[str] = None
