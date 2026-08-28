import random
from loguru import logger
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import yaml


class SequenceLengthThresholds:
    """Thresholds for sequence length-based behavior."""
    VERY_SHORT = 8
    SHORT = 24
    MEDIUM = 32
    LONG = 64
    DEFAULT = 512
    MIN = 5
    MAX = 4096


class SeasonalConstants:
    """Constants for seasonal wave generation."""
    MIN_PERIOD = 6
    SLIDING_WINDOW = 5
    DEFAULT_SIN_PROBABILITY = 0.7
    DEFAULT_SQUARE_PROBABILITY = 0.15
    DEFAULT_TRIANGLE_PROBABILITY = 0.15


class NoiseConstants:
    """Constants for noise generation."""
    MIN_NOISE_STD_FACTOR = 0.03
    MAX_NOISE_STD_FACTOR = 0.15


class AmplitudeConstants:
    """Constants for amplitude generation."""
    # SNR note: at exponent -2 (10^-2 overall_amplitude, ~10% of samples),
    # event amplitude ~0.008-0.023 vs noise std ~0.0003-0.0015 → SNR ~15:1, safe.
    AMPLITUDE_EXPONENTS = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
    AMPLITUDE_PROBABILITIES = [0.1, 0.2, 0.2, 0.3, 0.1, 0.04, 0.03, 0.02, 0.008, 0.002]

@dataclass
class DatagenConfig:
    """Configuration for data generation."""
    enable_multiple_trend: bool = False
    enable_drop_prompt: bool = True
    local_change_verbose: bool = False

    @classmethod
    def from_yaml(cls, config_path: str = "config/datagen_config.yaml") -> "DatagenConfig":
        """Load configuration from YAML file."""
        try:
            path = Path(config_path)
            if not path.exists():
                logger.warning(f"Config file not found at {config_path}, using defaults")
                return cls()

            with open(path, 'r') as f:
                config_data = yaml.safe_load(f)

            return cls(
                enable_multiple_trend=config_data.get("enable_multiple_trend", False),
                enable_drop_prompt=config_data.get("enable_drop_prompt", True),
                local_change_verbose=config_data.get("local_change_verbose", False),
            )
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return cls()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DatagenConfig":
        """Create config from dictionary."""
        return cls(
            enable_multiple_trend=config_dict.get("enable_multiple_trend", False),
            enable_drop_prompt=config_dict.get("enable_drop_prompt", True),
            local_change_verbose=config_dict.get("local_change_verbose", False),
        )

# Global config management (unchanged)
_config: Optional[DatagenConfig] = None

def get_config() -> DatagenConfig:
    """Get the current configuration, loading if necessary."""
    global _config
    if _config is None:
        _config = DatagenConfig.from_yaml()
    return _config

def set_config(config: DatagenConfig) -> None:
    """Set the configuration (useful for testing)."""
    global _config
    _config = config

@contextmanager
def temporary_config(config: DatagenConfig):
    """Context manager for temporarily changing config."""
    global _config
    old_config = _config
    _config = config
    try:
        yield
    finally:
        _config = old_config

@dataclass
class AttributeSet:
    """Configuration for attribute generation probabilities."""
    overall_attribute: Dict[str, Dict[str, float]] = field(default_factory=dict)
    change: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def get_default(cls) -> "AttributeSet":
        """Get default attribute set."""
        return cls(
            overall_attribute={
                "seasonal": {
                    "no periodic fluctuation": 0.7,
                    "sin periodic fluctuation": 0.25,
                    "square periodic fluctuation": 0.02,
                    "triangle periodic fluctuation": 0.03
                },
                "trend": {
                    "decrease": 0.3,
                    "increase": 0.3,
                    "keep steady": 0.3,
                    "multiple": 0.1
                },
                "frequency": {
                    "high frequency": 0.5,
                    "low frequency": 0.5
                },
                "noise": {
                    "noisy": 0.2,
                    "smooth": 0.8
                }
            },
            change={
                "shake": 2,
                "upward spike": 12,
                "downward spike": 10,
                "continuous upward spike": 3,
                "continuous downward spike": 3,
                "upward convex": 2,
                "downward convex": 2,
                "sudden increase": 10,
                "sudden decrease": 10,
                "rapid rise followed by slow decline": 2,
                "slow rise followed by rapid decline": 2,
                "rapid decline followed by slow rise": 2,
                "slow decline followed by rapid rise": 2,
                "decrease after upward spike": 1,
                "increase after downward spike": 1,
                "increase after upward spike": 1,
                "decrease after downward spike": 1,
                "wide upward spike": 2,
                "wide downward spike": 2
            }
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AttributeSet":
        return cls(
            overall_attribute=config_dict.get("overall_attribute", {}),
            change=config_dict.get("change", {})
        )

# Default attribute set instance
DEFAULT_ATTRIBUTE_SET = AttributeSet.get_default()

def generate_split_points(seq_len: int, num_segments: int) -> List[int]:
    """Generate random split points for segmenting a sequence."""
    if num_segments < 1:
        raise ValueError("Number of segments must be at least 1.")
    if seq_len < num_segments:
        raise ValueError("Sequence length must be at least equal to the number of segments.")

    min_segment_len = seq_len / num_segments / 2
    split_points = [0]

    for i in range(num_segments - 1):
        min_point = split_points[-1] + min_segment_len
        max_point = seq_len - (num_segments - len(split_points)) * min_segment_len

        if min_point >= max_point:
            raise ValueError("Cannot generate split points satisfying the constraints.")

        split_points.append(int(random.uniform(min_point, max_point)))

    split_points.append(seq_len)
    return split_points
