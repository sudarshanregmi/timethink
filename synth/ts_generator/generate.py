import numpy as np
import random
import re
from loguru import logger
from typing import Dict, List, Optional, Tuple, Any, Union

from synth.ts_generator.local_changes import generate_local_chars
from synth.ts_generator.utils.probability_utils import (
    weighted_random_choice,
    weighted_random_choices
)
from synth.ts_generator.utils.validation_utils import validate_sequence_length
from synth.ts_generator.utils.serialization_utils import deep_copy_dict
from synth.ts_generator.utils.statistics_utils import (
    calculate_segment_means,
    get_segment_count,
    add_statistics_to_pool
)

# Re-export from constants
from synth.ts_generator.constants import (
    SequenceLengthThresholds,
    SeasonalConstants,
    AmplitudeConstants,
    DatagenConfig,
    get_config,
    AttributeSet,
    DEFAULT_ATTRIBUTE_SET,
)

# Re-export from generators
from synth.ts_generator.generators import (
    WaveGenerator,
    NoiseGenerator,
    SeasonalGenerator,
    TrendGenerator,
)


class AttributeGenerator:
    """Generates attribute pools for time series generation."""

    # Change types that require longer sequences
    COMPLEX_CHANGE_TYPES = [
        "upward convex",
        "downward convex",
        "rapid rise followed by slow decline",
        "slow rise followed by rapid decline",
        "rapid decline followed by slow rise",
        "slow decline followed by rapid rise",
        "decrease after upward spike",
        "increase after downward spike",
        "increase after upward spike",
        "decrease after downward spike",
        "wide upward spike",
        "wide downward spike"
    ]

    # Change types excluded for very short sequences
    SHORT_SEQUENCE_EXCLUDED = [
        "shake",
        "sudden increase",
        "sudden decrease"
    ]

    def __init__(
        self,
        attribute_set: Optional[AttributeSet] = None,
        config: Optional[DatagenConfig] = None
    ):
        self.attribute_set = attribute_set or DEFAULT_ATTRIBUTE_SET
        self.config = config or get_config()

    def _filter_change_attributes(
        self,
        change_attribute: Dict[str, float],
        seq_len: int,
        trend_type: str
    ) -> Dict[str, float]:
        """Filter change attributes based on sequence length and trend."""
        filtered = change_attribute.copy()

        if seq_len <= SequenceLengthThresholds.LONG and trend_type == "multiple":
            for change_type in self.COMPLEX_CHANGE_TYPES:
                filtered.pop(change_type, None)

        if seq_len <= SequenceLengthThresholds.VERY_SHORT:
            for change_type in self.SHORT_SEQUENCE_EXCLUDED:
                filtered.pop(change_type, None)

        return filtered

    def _select_trend(self, trend_options: Dict[str, float]) -> str:
        """Select a trend type based on probabilities."""
        return weighted_random_choice(trend_options)

    def _generate_seasonal_attribute(
        self,
        seasonal_options: Dict[str, float],
        seq_len: int
    ) -> Dict[str, Any]:
        """Generate seasonal attribute based on options and sequence length."""
        if seq_len >= SequenceLengthThresholds.SHORT:
            seasonal_type = weighted_random_choice(seasonal_options)
        else:
            seasonal_type = "no periodic fluctuation"

        return {"type": seasonal_type}

    def _generate_frequency_attribute(
        self,
        frequency_options: Dict[str, float],
        seasonal_type: str,
        seq_len: int
    ) -> str:
        """Generate frequency attribute based on seasonal type and length."""
        if 'no periodic fluctuation' in seasonal_type or seq_len < SequenceLengthThresholds.SHORT:
            return 'no periodicity'

        if seq_len <= SequenceLengthThresholds.LONG:
            return 'low frequency'

        return weighted_random_choice(frequency_options)

    def _generate_noise_attribute(
        self,
        noise_options: Dict[str, float],
        seq_len: int
    ) -> Dict[str, Any]:
        """Generate noise attribute based on options and sequence length."""
        if seq_len <= SequenceLengthThresholds.MEDIUM:
            return {'type': 'smooth'}

        return {'type': weighted_random_choice(noise_options)}

    def _generate_local_attributes(
        self,
        change_attribute: Dict[str, float],
        change_positions: List[Tuple[Optional[int], Optional[float]]]
    ) -> List[Dict[str, Any]]:
        """Generate local change attributes."""
        if not change_attribute:
            return []

        num_local_chars = len(change_positions)
        local_chars = weighted_random_choices(change_attribute, num_local_chars)

        local_attributes = []
        positions = list(change_positions)

        for char in local_chars:
            if positions:
                local_position, local_amplitude = positions.pop()
            else:
                local_position, local_amplitude = None, None

            local_attributes.append({
                "type": char,
                "position_start": local_position,
                "amplitude": local_amplitude
            })

        return local_attributes

    def generate_random_attributes(
        self,
        change_positions: Optional[List[Tuple[Optional[int], Optional[float]]]] = None,
        seq_len: int = SequenceLengthThresholds.DEFAULT
    ) -> Dict[str, Any]:
        """Generate a random attribute pool."""
        validate_sequence_length(seq_len)

        if change_positions is None:
            change_positions = [(None, None) for _ in range(random.randint(0, 3))]
        else:
            change_positions = list(change_positions)

        overall = self.attribute_set.overall_attribute
        change = self.attribute_set.change

        attribute_pool = {}

        attribute_pool["seasonal"] = self._generate_seasonal_attribute(
            overall['seasonal'], seq_len
        )

        freq_type = self._generate_frequency_attribute(
            overall['frequency'],
            attribute_pool["seasonal"]['type'],
            seq_len
        )
        attribute_pool["seasonal"]["frequency_type"] = freq_type

        trend_type = self._select_trend(overall['trend'])
        attribute_pool["trend"] = {"type": trend_type}

        filtered_change = self._filter_change_attributes(change, seq_len, trend_type)
        attribute_pool["local"] = self._generate_local_attributes(
            filtered_change, change_positions
        )

        attribute_pool["noise"] = self._generate_noise_attribute(
            overall['noise'], seq_len
        )

        attribute_pool["seq_len"] = seq_len
        return attribute_pool

    def generate_controlled_attributes(
        self,
        attribute_set: Dict[str, Any],
        change_positions: Optional[List[Tuple[Optional[int], Optional[float]]]] = None,
        seq_len: int = SequenceLengthThresholds.DEFAULT
    ) -> Dict[str, Any]:
        """Generate attributes with controlled parameters."""
        validate_sequence_length(seq_len)

        if change_positions is None:
            change_positions = [(None, None) for _ in range(random.randint(0, 3))]
        else:
            change_positions = list(change_positions)

        attribute_set = deep_copy_dict(attribute_set)

        description = {}
        default_overall = self.attribute_set.overall_attribute
        default_change = self.attribute_set.change

        # Seasonal
        seasonal_attrs = attribute_set['seasonal']['attributes']
        seasonal_p = [default_overall['seasonal'].get(i, 0.1) for i in seasonal_attrs]
        seasonal_p = np.array(seasonal_p) / sum(seasonal_p)
        description["seasonal"] = {
            "type": np.random.choice(seasonal_attrs, p=seasonal_p),
            "amplitude": round(random.uniform(
                attribute_set['seasonal']['amplitude']['min'],
                attribute_set['seasonal']['amplitude']['max']
            ), 2)
        }

        # Frequency
        if 'no periodic fluctuation' not in description["seasonal"]['type']:
            period = max(
                random.uniform(
                    attribute_set['seasonal']['period']['min'],
                    attribute_set['seasonal']['period']['max']
                ),
                SeasonalConstants.MIN_PERIOD
            )
            freq_type = 'high frequency' if period < seq_len // 8 else 'low frequency'
            description["seasonal"]["frequency_type"] = freq_type
            description["seasonal"]["period"] = round(period, 1)
        else:
            description["seasonal"]["frequency_type"] = 'no periodicity'
            description["seasonal"]["period"] = 0.0
            description["seasonal"]["amplitude"] = 0.0

        # Trend
        trend_attrs = attribute_set['trend']['attributes']
        if not self.config.enable_multiple_trend:
            if "multiple" in trend_attrs:
                trend_attrs.remove("multiple")
            if len(trend_attrs) == 0:
                trend_attrs = ['increase', 'decrease', 'keep steady']

        trend_p = [default_overall['trend'].get(i, 0.1) for i in trend_attrs]
        trend_p = np.array(trend_p) / sum(trend_p)
        description["trend"] = {
            "type": np.random.choice(trend_attrs, p=trend_p),
            "start": random.uniform(
                attribute_set['trend']['start']['min'],
                attribute_set['trend']['start']['max']
            ),
            "amplitude": round(random.uniform(
                attribute_set['trend']['amplitude']['min'],
                attribute_set['trend']['amplitude']['max']
            ), 2)
        }

        # Local changes
        num_local_chars = len(change_positions)
        change_attrs = attribute_set['change']['attributes']
        change_p = [default_change.get(i, 1) for i in change_attrs]
        change_p = np.array(change_p) / sum(change_p)
        local_chars = list(np.random.choice(change_attrs, size=num_local_chars, p=change_p))

        description["local"] = []
        for char in local_chars:
            description["local"].append({
                "type": char,
                "position_start": None,
                "amplitude": round(random.uniform(
                    attribute_set['change']['amplitude']['min'],
                    attribute_set['change']['amplitude']['max']
                ), 2)
            })

        # Noise
        noise_attrs = attribute_set['noise']['attributes']
        noise_p = [default_overall['noise'].get(i, 0.1) for i in noise_attrs]
        noise_p = np.array(noise_p) / sum(noise_p)
        description["noise"] = {
            'type': np.random.choice(noise_attrs, p=noise_p)
        }

        description["seq_len"] = seq_len

        return description


class TimeSeriesGenerator:
    """
    Builds a single time series from component parts.

    This is the core low-level generator that assembles trend, seasonal,
    noise, and local change components into a complete time series.
    """

    def __init__(self, config: Optional[DatagenConfig] = None):
        self.config = config or get_config()
        self.attribute_generator = AttributeGenerator(config=self.config)
        self.seasonal_generator = SeasonalGenerator(config=self.config)
        self.trend_generator = TrendGenerator(config=self.config)
        self.noise_generator = NoiseGenerator(config=self.config)

    def build(
        self,
        attribute_pool: Dict[str, Any],
        seq_len: int = SequenceLengthThresholds.DEFAULT
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Build a time series from an attribute pool.

        Args:
            attribute_pool: Dictionary specifying time series attributes
            seq_len: Length of the time series to generate

        Returns:
            Tuple of (time_series_array, updated_attribute_pool)
        """
        validate_sequence_length(seq_len)

        # Create a copy to avoid mutating input
        attribute_pool = deep_copy_dict(attribute_pool)

        # Generate base time series
        y = np.zeros(seq_len)

        # Setup frequency/period
        self._setup_frequency(attribute_pool, seq_len)

        # Setup amplitude and bias
        overall_amplitude, overall_bias = self._setup_amplitude_bias(attribute_pool)

        # Apply seasonal component
        y += self.seasonal_generator.generate_seasonal(
            attribute_pool, overall_amplitude, seq_len
        )

        # Apply local changes
        y += generate_local_chars(attribute_pool, overall_amplitude, seq_len)

        # Apply trend
        y = self.trend_generator.generate_trend(
            attribute_pool, y, overall_amplitude, overall_bias, seq_len
        )

        # Replace placeholders in local char details
        self._replace_local_char_placeholders(attribute_pool, y, seq_len)

        # Add noise
        y += self.noise_generator.generate_noise(
            attribute_pool, y, overall_amplitude, seq_len
        )

        # Add statistics
        add_statistics_to_pool(attribute_pool, y, seq_len)

        return y, attribute_pool

    def _setup_frequency(
        self,
        attribute_pool: Dict[str, Any],
        seq_len: int
    ) -> None:
        """Setup frequency and period in attribute pool."""
        if "seasonal" not in attribute_pool:
            return

        freq_type = attribute_pool["seasonal"].get("frequency_type")

        if not freq_type:
            return

        if freq_type == "no periodicity":
            attribute_pool["seasonal"]['period'] = 0.0
            attribute_pool["seasonal"]['frequency_detail'] = (
                "No significant periodic fluctuations observed, "
                "overall almost no periodicity."
            )
            return

        # Calculate period if not already set
        if 'period' not in attribute_pool["seasonal"]:
            if freq_type == "high frequency":
                max_period = seq_len // 8
                min_period = max(seq_len // 16, SeasonalConstants.MIN_PERIOD)
            else:  # low frequency
                max_period = seq_len // 3
                min_period = max(seq_len // 8, SeasonalConstants.MIN_PERIOD)

            # Ensure min_period <= max_period (can invert for small seq_len)
            if min_period > max_period:
                min_period, max_period = max_period, min_period
            max_period = max(max_period, SeasonalConstants.MIN_PERIOD)

            period = random.uniform(min_period, max_period)
        else:
            period = attribute_pool["seasonal"]['period']

        attribute_pool["seasonal"]['period'] = round(period, 2)
        attribute_pool["seasonal"]['frequency_detail'] = (
            f"Each fluctuation period is approximately {period:.2f} points, "
            f"thus the overall fluctuation is {freq_type}."
        )

    def _setup_amplitude_bias(
        self,
        attribute_pool: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Setup overall amplitude and bias."""
        if 'overall_amplitude' in attribute_pool and 'overall_bias' in attribute_pool:
            return (
                attribute_pool['overall_amplitude'],
                attribute_pool['overall_bias']
            )

        # Generate random amplitude and bias
        overall_amplitude_e = np.random.choice(
            AmplitudeConstants.AMPLITUDE_EXPONENTS,
            p=AmplitudeConstants.AMPLITUDE_PROBABILITIES
        )

        overall_amplitude = round(
            np.random.uniform(
                10.0 ** (overall_amplitude_e - 1),
                10.0 ** (overall_amplitude_e + 1)
            ),
            2
        )

        overall_bias = round(
            np.random.uniform(
                -(10.0 ** (overall_amplitude_e + 1)),
                10.0 ** (overall_amplitude_e + 1)
            ),
            2
        )

        attribute_pool['overall_amplitude'] = round(overall_amplitude, 2)
        attribute_pool['overall_bias'] = round(overall_bias, 2)

        return overall_amplitude, overall_bias

    def _replace_local_char_placeholders(
        self,
        attribute_pool: Dict[str, Any],
        y: np.ndarray,
        seq_len: int
    ) -> None:
        """Replace placeholder values in local character details."""
        pattern = re.compile(r'<\|(\d+)\|>')

        def replacer(match: re.Match) -> str:
            n = int(match.group(1))
            if n < 0 or n >= seq_len:
                logger.warning(
                    f"Invalid index {n} in local char detail, seq_len={seq_len}"
                )
                n = max(0, min(n, seq_len - 1))
            return f"{y[n]:.2f}"

        for local_char in attribute_pool["local"]:
            if 'detail' in local_char:
                local_char['detail'] = pattern.sub(replacer, local_char['detail'])

            # Key point resolution
            if "key_points_intent" in local_char:
                resolved_points = []
                for kp in local_char["key_points_intent"]:
                    idx = kp["index"]
                    safe_idx = max(0, min(idx, seq_len - 1))
                    resolved_points.append({
                        "label": kp["label"],
                        "index": safe_idx,
                        "value": round(float(y[safe_idx]), 2)
                    })
                local_char["key_points_resolved"] = resolved_points

            # Add start/end values
            p_start = local_char.get("position_start")
            p_end = local_char.get("position_end")
            if p_start is not None and 0 <= p_start < seq_len:
                local_char["value_start"] = round(float(y[p_start]), 2)
            if p_end is not None and p_end > 0:
                # position_end is exclusive (0-indexed past-the-end), so last
                # inclusive point is p_end - 1.  Clamp to seq_len - 1 for
                # events that extend to the very end of the sequence.
                last_idx = min(p_end - 1, seq_len - 1)
                local_char["value_end"] = round(float(y[last_idx]), 2)


class TextGenerator:
    """Generates text descriptions from time series attributes."""

    DEFAULT_INCLUDE_ATTRIBUTES = [
        'length', 'trend', 'periodicity', 'frequency', 'noise', 'local', 'statistic'
    ]

    @staticmethod
    def attribute_to_text(
        time_series: np.ndarray,
        attribute_pool: Dict[str, Any],
        generate_values: bool = False,
        include_attributes: Optional[List[str]] = None
    ) -> str:
        """Convert attribute pool to descriptive text."""
        if include_attributes is None:
            include_attributes = list(TextGenerator.DEFAULT_INCLUDE_ATTRIBUTES)
        else:
            mapping = {
                "local events": "local",
                "statistics": "statistic",
                "periodicity": "periodicity"
            }
            include_attributes = [mapping.get(a, a) for a in include_attributes]

        # Handle legacy parameter
        if not generate_values and 'statistic' in include_attributes:
            include_attributes.remove('statistic')
        elif generate_values and 'statistic' not in include_attributes:
            include_attributes.append('statistic')

        seq_len = len(time_series)
        description_parts = []

        for attr in include_attributes:
            if attr == 'length':
                description_parts.append(f"The length of the time series is {seq_len}.")

            elif attr == 'trend' and 'trend' in attribute_pool:
                description_parts.append(attribute_pool['trend']['detail'])

            elif attr == 'periodicity' and 'seasonal' in attribute_pool:
                description_parts.append(attribute_pool['seasonal']['detail'])

            elif attr == 'frequency' and 'seasonal' in attribute_pool:
                if "no" not in attribute_pool['seasonal'].get('type', 'no'):
                    freq_detail = attribute_pool['seasonal'].get('frequency_detail')
                    if freq_detail:
                        description_parts.append(freq_detail)

            elif attr == 'noise' and 'noise' in attribute_pool:
                description_parts.append(attribute_pool['noise']['detail'])

            elif attr == 'local':
                description_parts.append(
                    TextGenerator._generate_local_description(attribute_pool)
                )

            elif attr == 'statistic':
                max_value = round(np.max(time_series), 2)
                min_value = round(np.min(time_series), 2)
                description_parts.append(
                    TextGenerator._generate_statistic_description(
                        time_series, seq_len, max_value, min_value
                    )
                )

        return ' '.join(description_parts).replace('  ', ' ').strip()

    @staticmethod
    def _generate_local_description(attribute_pool: Dict[str, Any]) -> str:
        """Generate description of local events."""
        local_chars = attribute_pool.get("local", [])

        if not local_chars:
            return "No local events are found."

        local_descriptions = [
            f"{item['detail']}, forming a {item['type']}"
            for item in local_chars
            if 'detail' in item
        ]

        if local_descriptions:
            return f"In terms of local events, {'; '.join(local_descriptions)}."

        return "No local events are found."

    @staticmethod
    def _generate_statistic_description(
        time_series: np.ndarray,
        seq_len: int,
        max_value: float,
        min_value: float
    ) -> str:
        """Generate statistical description."""
        segments = get_segment_count(seq_len)
        segment_mean = calculate_segment_means(time_series, segments)
        actual_segments = len(segment_mean)
        segment_size = seq_len // segments

        return (
            f"Specific data details: The time series is divided into {actual_segments} segments, "
            f"with the approximate mean values for each {segment_size}-point interval being: "
            f"{segment_mean}. The maximum value of the entire series is {max_value}, "
            f"and the minimum value is {min_value}."
        )



# =============================================================================
# Public API Functions (module-level wrappers around class methods)
# =============================================================================

def generate_random_attributes(
    overall_attribute: Optional[Dict[str, Dict[str, float]]] = None,
    change_attribute: Optional[Dict[str, float]] = None,
    change_positions: Optional[List[Tuple[Optional[int], Optional[float]]]] = None,
    seq_len: int = SequenceLengthThresholds.DEFAULT
) -> Dict[str, Any]:
    """Generate random attribute pool for time series generation."""
    if overall_attribute is None:
        overall_attribute = DEFAULT_ATTRIBUTE_SET.overall_attribute
    if change_attribute is None:
        change_attribute = DEFAULT_ATTRIBUTE_SET.change

    attribute_set = AttributeSet(
        overall_attribute=overall_attribute,
        change=change_attribute
    )
    generator = AttributeGenerator(attribute_set=attribute_set)
    return generator.generate_random_attributes(
        change_positions=change_positions,
        seq_len=seq_len
    )


def generate_controlled_attributes(
    attribute_set: Dict[str, Any],
    change_positions: Optional[List[Tuple[Optional[int], Optional[float]]]] = None,
    seq_len: int = SequenceLengthThresholds.DEFAULT
) -> Dict[str, Any]:
    """Generate attribute pool with controlled parameters."""
    generator = AttributeGenerator()
    return generator.generate_controlled_attributes(
        attribute_set=attribute_set,
        change_positions=change_positions,
        seq_len=seq_len
    )


def generate_time_series(
    attribute_pool: Dict[str, Any],
    seq_len: int = SequenceLengthThresholds.DEFAULT
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a complete time series from attribute pool."""
    builder = TimeSeriesGenerator()
    return builder.build(attribute_pool=attribute_pool, seq_len=seq_len)


def attribute_to_text(
    time_series: np.ndarray,
    attribute_pool: Dict[str, Any],
    generate_values: bool = False,
    include_attributes: Optional[List[str]] = None
) -> str:
    """Convert attribute pool to descriptive text."""
    if include_attributes is None:
        include_attributes = TextGenerator.DEFAULT_INCLUDE_ATTRIBUTES
    return TextGenerator.attribute_to_text(
        time_series=time_series,
        attribute_pool=attribute_pool,
        generate_values=generate_values,
        include_attributes=include_attributes
    )


# Expose the default attribute set for backward compatibility
all_attribute_set = {
    "overall_attribute": DEFAULT_ATTRIBUTE_SET.overall_attribute,
    "change": DEFAULT_ATTRIBUTE_SET.change
}
