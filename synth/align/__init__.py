"""QA Generation package for time series analysis."""

from synth.align.config import (
    Config,
    Mode,
    Difficulty,
    SampleResult,
    PromptIndexer,
)
from synth.align.base_generator import BaseQAGenerator
from synth.align.generators import (
    UTSQAGenerator,
    MTSLocalQAGenerator,
    MTSShapeQAGenerator,
)
from synth.align.factory import (
    generate_sample,
    generate_uts_sample,
    generate_mts_local_sample,
    generate_mts_shape_sample,
    select_single_qa,
    select_metrics_for_generation,
)
from synth.align.dataset import DatasetGenerator
from synth.align.utils import replace_prompts_in_obj

__all__ = [
    # Config
    "Config",
    "Mode",
    "Difficulty",
    "SampleResult",
    "PromptIndexer",
    # Base
    "BaseQAGenerator",
    # Generators
    "UTSQAGenerator",
    "MTSLocalQAGenerator",
    "MTSShapeQAGenerator",
    # Factory
    "generate_sample",
    "generate_uts_sample",
    "generate_mts_local_sample",
    "generate_mts_shape_sample",
    "select_single_qa",
    "select_metrics_for_generation",
    # Dataset
    "DatasetGenerator",
    # Utils
    "replace_prompts_in_obj",
]
