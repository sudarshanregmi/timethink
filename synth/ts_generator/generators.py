import numpy as np
import random
from loguru import logger
from typing import Dict, List, Optional, Tuple, Any

from synth.ts_generator.constants import (
    SeasonalConstants,
    NoiseConstants,
    DatagenConfig,
    get_config,
)

from synth.ts_generator.utils.trend_utils import (
    generate_random_points,
    generate_trend_prompt,
    generate_trend_curve,
    generate_trend_list,
)
from synth.ts_generator.utils.change_utils import generate_ts_change


class WaveGenerator:

    @staticmethod
    def generate_seasonal_wave(
        period: float,
        amplitude_list: List[float],
        split_points: List[int],
        seq_len: int,
        wave_type: Optional[str] = None
    ) -> np.ndarray:
        t = np.linspace(0, seq_len, seq_len)
        data = np.zeros(seq_len)
        base_frequency = 1 / period

        # Build amplitude series
        amplitude_series = np.zeros(seq_len)
        for i in range(len(amplitude_list)):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]
            amplitude_series[start_idx:end_idx] = amplitude_list[i]

        # Smooth amplitude series
        sliding_window = SeasonalConstants.SLIDING_WINDOW
        for i in range(seq_len - sliding_window):
            mid_idx = i + sliding_window // 2
            amplitude_series[mid_idx] = np.mean(amplitude_series[i:i + sliding_window])

        # Select wave type if not specified
        if wave_type is None:
            wave_type = str(np.random.choice(
                ['sin', 'square', 'triangle'],
                p=[
                    SeasonalConstants.DEFAULT_SIN_PROBABILITY,
                    SeasonalConstants.DEFAULT_SQUARE_PROBABILITY,
                    SeasonalConstants.DEFAULT_TRIANGLE_PROBABILITY
                ]
            ))

        if wave_type == 'sin':
            data = WaveGenerator._generate_sin_wave(
                t, data, amplitude_series, base_frequency, period, seq_len
            )
        elif wave_type == 'square':
            data = WaveGenerator._generate_square_wave(
                t, data, amplitude_series, period, seq_len
            )
        else:  # triangle
            data = WaveGenerator._generate_triangle_wave(
                t, data, amplitude_series, period, seq_len
            )

        # Normalize to amplitude
        data_range = data.max() - data.min() + 1e-7
        data = data / data_range * max(amplitude_list)
        data -= np.mean(data)

        return data

    @staticmethod
    def _generate_sin_wave(
        t: np.ndarray,
        data: np.ndarray,
        amplitude_series: np.ndarray,
        base_frequency: float,
        period: float,
        seq_len: int,
        aligned: bool = False
    ) -> np.ndarray:
        num_harmonics = np.random.randint(1, max(2, min(int(period // 6), 10)))

        for n in range(1, num_harmonics + 1):
            # aligned=True: phase=0 so the wave starts and ends at zero
            # (valid when caller guarantees integer number of cycles)
            phase = 0.0 if aligned else np.random.uniform(0, 2 * np.pi)

            # Amplitude modulation
            modulation = 1 + np.random.uniform(0, 0.05) * np.sin(
                np.random.uniform(1, 3) * np.pi * t / seq_len +
                np.random.uniform(0, 2 * np.pi)
            )
            harmonic_amplitude = amplitude_series / n * modulation

            data += harmonic_amplitude * np.sin(
                2 * np.pi * base_frequency * n * t + phase
            )

        return data

    @staticmethod
    def _generate_square_wave(
        t: np.ndarray,
        data: np.ndarray,
        amplitude_series: np.ndarray,
        period: float,
        seq_len: int,
        aligned: bool = False
    ) -> np.ndarray:
        start = 0.0 if aligned else np.random.uniform(0, 0.3)
        duration = np.random.uniform(0.1, 0.3)

        for i in range(seq_len):
            cycle_pos = (t[i] % period) / period
            if start <= cycle_pos < start + duration:
                data[i] = amplitude_series[i]
            else:
                data[i] = 0.0

        return data

    @staticmethod
    def _generate_triangle_wave(
        t: np.ndarray,
        data: np.ndarray,
        amplitude_series: np.ndarray,
        period: float,
        seq_len: int,
        aligned: bool = False
    ) -> np.ndarray:
        start = 0.0 if aligned else np.random.uniform(0, 0.3)
        duration = np.random.uniform(0.1, 0.6)
        end = start + duration
        mid = (start + end) / 2

        for i in range(seq_len):
            cycle_pos = (t[i] % period) / period
            if start <= cycle_pos < end:
                if cycle_pos < mid:
                    data[i] = amplitude_series[i] * 2 * (cycle_pos - start) / duration
                else:
                    data[i] = amplitude_series[i] * 2 * (end - cycle_pos) / duration
            else:
                data[i] = 0.0

        return data

    @staticmethod
    def generate_single_segment_wave(
        wave_type: str,
        period: float,
        amplitude: float,
        length: int,
        aligned: bool = False
    ) -> np.ndarray:
        """Generate a wave segment of given type, period, amplitude and length."""
        if wave_type == "none" or length == 0:
            return np.zeros(length)

        t = np.linspace(0, length, length)
        data = np.zeros(length)
        amp_series = np.full(length, amplitude)

        if wave_type == "sin":
            base_frequency = 1.0 / period
            data = WaveGenerator._generate_sin_wave(t, data, amp_series, base_frequency, period, length, aligned=aligned)
        elif wave_type == "square":
            data = WaveGenerator._generate_square_wave(t, data, amp_series, period, length, aligned=aligned)
        else:  # triangle
            data = WaveGenerator._generate_triangle_wave(t, data, amp_series, period, length, aligned=aligned)

        data_range = data.max() - data.min() + 1e-7
        data = data / data_range * amplitude
        data -= np.mean(data)
        return data



class NoiseGenerator:
    """Generates noise components for time series."""

    def __init__(self, config: Optional[DatagenConfig] = None):
        self.config = config or get_config()

    def generate_noise(
        self,
        attribute_pool: Dict[str, Any],
        y: np.ndarray,
        overall_amplitude: float,
        seq_len: int
    ) -> np.ndarray:
        """Generate noise based on attribute pool settings.

        Uses overall_amplitude as the ONLY noise reference (from AMPLITUDE_EXPONENTS).
        Stores relative strength (ratio) in noise['strength'].
        """
        noise_level = attribute_pool["noise"]['type']

        if noise_level == "noisy":
            return self._generate_noisy(attribute_pool, overall_amplitude, seq_len)
        else:
            attribute_pool["noise"]["strength"] = 0.0
            attribute_pool["noise"]["detail"] = "The curve is smooth with no noise."
            return np.zeros(seq_len)

    def _generate_noisy(
        self,
        attribute_pool: Dict[str, Any],
        overall_amplitude: float,
        seq_len: int
    ) -> np.ndarray:
        """Generate Gaussian noise with relative strength in [0.03, 0.15]."""
        strength = np.random.uniform(
            NoiseConstants.MIN_NOISE_STD_FACTOR,
            NoiseConstants.MAX_NOISE_STD_FACTOR
        )
        noise_std = strength * overall_amplitude
        noise = np.random.normal(0, noise_std, seq_len)

        strength_rounded = round(strength, 2)
        attribute_pool["noise"]["strength"] = strength_rounded
        attribute_pool["noise"]["detail"] = (
            f"The noise strength is around {strength_rounded * 100:.2f}% of the signal amplitude, "
            f"indicating a noisy curve."
        )
        return noise


class SeasonalGenerator:
    """Generates seasonal components for time series."""

    def __init__(self, config: Optional[DatagenConfig] = None):
        self.config = config or get_config()

    def generate_seasonal(
        self,
        attribute_pool: Dict[str, Any],
        overall_amplitude: float,
        seq_len: int
    ) -> np.ndarray:
        """Generate seasonal component based on attribute pool."""
        y = np.zeros(seq_len)
        seasonal_type = attribute_pool["seasonal"]['type']

        if "no period" in seasonal_type:
            return self._handle_no_periodicity(attribute_pool, y)

        if seasonal_type == "no periodic fluctuation":
            return self._handle_no_fluctuation(attribute_pool, y)

        return self._generate_periodic(attribute_pool, y, overall_amplitude, seq_len)

    def _handle_no_periodicity(
        self,
        attribute_pool: Dict[str, Any],
        y: np.ndarray
    ) -> np.ndarray:
        """Handle case with no periodicity or fluctuation."""
        attribute_pool["seasonal"]["segments"] = []
        attribute_pool["seasonal"]["amplitude"] = 0.0
        attribute_pool["seasonal"]['detail'] = (
            f"It shows {attribute_pool['seasonal']['type']}."
        )
        return y

    def _handle_no_fluctuation(
        self,
        attribute_pool: Dict[str, Any],
        y: np.ndarray
    ) -> np.ndarray:
        """Handle case with no fluctuation."""
        return self._handle_no_periodicity(attribute_pool, y)

    def _generate_periodic(
        self,
        attribute_pool: Dict[str, Any],
        y: np.ndarray,
        overall_amplitude: float,
        seq_len: int
    ) -> np.ndarray:
        """Generate periodic seasonal pattern (single type, single period)."""
        seasonal_type = attribute_pool["seasonal"]['type']

        # Determine wave type
        if seasonal_type == "periodic fluctuation":
            wave_type = None
        else:
            wave_type = seasonal_type.split(" ")[0]

        # Determine amplitudes
        if 'amplitude' not in attribute_pool['seasonal']:
            amplitudes, split_points = self._generate_multiple_amplitudes(
                overall_amplitude, seq_len
            )
        else:
            amplitudes = [attribute_pool['seasonal']['amplitude']]
            split_points = [0, seq_len]

        period = attribute_pool["seasonal"].get('period')

        if period is None:
            raise ValueError(
                f"Period missing in seasonal attribute for type '{seasonal_type}'. "
                f"_setup_frequency() must be called before generate_seasonal()."
            )

        # Generate wave
        y += WaveGenerator.generate_seasonal_wave(
            period,
            amplitudes,
            split_points,
            seq_len,
            wave_type
        )

        # Update attribute pool with details
        self._update_seasonal_details(attribute_pool, amplitudes, split_points)

        return y

    def _generate_multiple_amplitudes(
        self,
        overall_amplitude: float,
        seq_len: int
    ) -> Tuple[List[float], List[int]]:
        """Generate a single amplitude for the whole series."""
        amplitude = random.uniform(1.0, 2.0) * overall_amplitude
        return [amplitude], [0, seq_len]

    def _update_seasonal_details(
        self,
        attribute_pool: Dict[str, Any],
        amplitudes: List[float],
        split_points: List[int]
    ) -> None:
        """Update attribute pool with seasonal details."""
        seasonal_type = attribute_pool["seasonal"]['type']
        attribute_pool["seasonal"]['detail'] = (
            f"The time series is showing {seasonal_type}: "
        )
        attribute_pool["seasonal"]["segments"] = []
        attribute_pool["seasonal"]["amplitude"] = round(float(np.mean(amplitudes)), 2)

        for i, amp in enumerate(amplitudes):
            start = split_points[i]
            end = split_points[i + 1]

            segment_info = {
                "amplitude": round(amp, 2),
                "position_start": start,
                "position_end": end,
                "description": (
                    f"the amplitude of the periodic fluctuation is {amp:.2f} "
                    f"between point {start} and point {end}"
                )
            }
            attribute_pool["seasonal"]["segments"].append(segment_info)
            attribute_pool["seasonal"]['detail'] += (
                f"the amplitude of the periodic fluctuation is {amp:.2f} "
                f"between point {start} and point {end}, "
            )

        # Clean trailing punctuation
        attribute_pool["seasonal"]['detail'] = (
            attribute_pool["seasonal"]['detail'].rstrip(", ") + "."
        )


class TrendGenerator:
    """Generates trend components for time series."""

    def __init__(self, config: Optional[DatagenConfig] = None):
        self.config = config or get_config()

    def generate_trend(
        self,
        attribute_pool: Dict[str, Any],
        y: np.ndarray,
        overall_amplitude: float,
        overall_bias: float,
        seq_len: int
    ) -> np.ndarray:
        """Generate trend component based on attribute pool."""
        trend = attribute_pool["trend"]["type"]

        amplitude = attribute_pool['trend'].get(
            'amplitude',
            random.uniform(0.8, 3.0) * overall_amplitude
        )
        bias = attribute_pool['trend'].get('start', overall_bias)

        trend_handlers = {
            "decrease": self._apply_decrease,
            "increase": self._apply_increase,
            "multiple": self._apply_multiple,
            "keep steady": self._apply_steady
        }

        handler = trend_handlers.get(trend, self._apply_steady)
        y = handler(attribute_pool, y, amplitude, bias, seq_len)

        self._add_local_phase_info(attribute_pool)
        self._update_trend_details(attribute_pool, y)

        return y

    def _apply_decrease(
        self,
        attribute_pool: Dict[str, Any],
        y: np.ndarray,
        amplitude: float,
        bias: float,
        seq_len: int
    ) -> np.ndarray:
        """Apply decreasing trend."""
        cur_value = generate_ts_change(seq_len, -amplitude, add_random_noise=False) + bias
        y += cur_value
        attribute_pool["trend"]["detail"] = (
            "From the perspective of the slope, the overall trend is decreasing."
        )
        attribute_pool["trend_list"] = [("decrease", 0, seq_len - 1)]
        return y

    def _apply_increase(
        self,
        attribute_pool: Dict[str, Any],
        y: np.ndarray,
        amplitude: float,
        bias: float,
        seq_len: int
    ) -> np.ndarray:
        """Apply increasing trend."""
        cur_value = generate_ts_change(seq_len, amplitude, add_random_noise=False) + bias
        y += cur_value
        attribute_pool["trend"]["detail"] = (
            "From the perspective of the slope, the overall trend is increasing."
        )
        attribute_pool["trend_list"] = [("increase", 0, seq_len - 1)]
        return y

    def _apply_multiple(
        self,
        attribute_pool: Dict[str, Any],
        y: np.ndarray,
        amplitude: float,
        bias: float,
        seq_len: int
    ) -> np.ndarray:
        """Apply multiple trend segments."""
        max_attempts = 100
        for _ in range(max_attempts):
            points, _ = generate_random_points(seq_len=seq_len)
            if len(generate_trend_list(points, seq_len)) > 1:
                break
        else:
            logger.warning("Could not generate multiple trends, falling back to steady")
            return self._apply_steady(attribute_pool, y, amplitude, bias, seq_len)

        _, trend_ts, _ = generate_trend_curve(seq_len=seq_len, points=points)
        y += trend_ts * amplitude

        attribute_pool["trend"]["detail"] = (
            "From the perspective of the slope, the overall trend contains "
            "multiple different segments: " + generate_trend_prompt(points)
        )
        attribute_pool["trend_list"] = generate_trend_list(points, seq_len)

        return y

    def _apply_steady(
        self,
        attribute_pool: Dict[str, Any],
        y: np.ndarray,
        amplitude: float,
        bias: float,
        seq_len: int
    ) -> np.ndarray:
        """Apply steady (no) trend."""
        y += bias
        attribute_pool["trend"]["detail"] = (
            "From the perspective of the slope, the overall trend is steady."
        )
        attribute_pool["trend_list"] = [("keep steady", 0, seq_len - 1)]
        return y

    def _add_local_phase_info(self, attribute_pool: Dict[str, Any]) -> None:
        """Add information about local phase changes to trend detail."""
        local_phase_changes = [
            item['type'] for item in attribute_pool["local"]
            if 'increase' in item['type'] or 'decrease' in item['type']
        ]

        if local_phase_changes:
            changes_str = ', '.join(local_phase_changes)
            attribute_pool["trend"]["detail"] = attribute_pool["trend"]["detail"].rstrip()
            attribute_pool["trend"]["detail"] += (
                f" However, local phase changes were observed, including: {changes_str}."
            )

    def _update_trend_details(
        self,
        attribute_pool: Dict[str, Any],
        y: np.ndarray
    ) -> None:
        """Update attribute pool with final trend details."""
        seq_len = len(y)
        slice_5 = max(1, seq_len // 20)
        start_val = float(np.mean(y[:slice_5]))
        end_val = float(np.mean(y[seq_len - slice_5:]))

        diff = end_val - start_val
        data_range = np.max(y) - np.min(y)

        threshold = 0.1 * data_range if data_range > 1e-6 else 1e-6

        if diff > threshold:
            net_type = "increase"
            sentence_start = "From the perspective of the slope, the overall trend is increasing."
        elif diff < -threshold:
            net_type = "decrease"
            sentence_start = "From the perspective of the slope, the overall trend is decreasing."
        else:
            net_type = "keep steady"
            sentence_start = "From the perspective of the slope, the overall trend is steady."

        was_multiple = attribute_pool["trend"]["type"] == "multiple"

        if not was_multiple:
            attribute_pool["trend"]["type"] = net_type
            attribute_pool["trend"]["detail"] = sentence_start
            attribute_pool["trend_list"] = [(net_type, 0, seq_len - 1)]
        else:
            attribute_pool["trend"]["type"] = net_type
            net_label = {
                "increase": "increasing",
                "decrease": "decreasing",
                "keep steady": "steady",
            }.get(net_type, net_type)
            attribute_pool["trend"]["detail"] = (
                attribute_pool["trend"]["detail"].rstrip()
                + f" Overall, the net direction of the trend is {net_label}."
            )

        attribute_pool["trend"]["start"] = round(start_val, 2)
        attribute_pool["trend"]["amplitude"] = round(diff, 2)
        attribute_pool["trend"]["end"] = round(end_val, 2)

        attribute_pool["trend"]["detail"] = attribute_pool["trend"]["detail"].rstrip()
        attribute_pool["trend"]["detail"] += (
            f" The value of time series starts from around {start_val:.2f} "
            f"and ends at around {end_val:.2f}, with an overall amplitude "
            f"of {diff:.2f}."
        )

        # Add segment breakdown for single-segment trends
        # (multiple-segment trends already have segments in their detail)
        if not was_multiple:
            vocab = {
                "increase": "an increasing",
                "decrease": "a decreasing",
                "keep steady": "a stable",
            }
            article = vocab.get(net_type, "a")
            attribute_pool["trend"]["detail"] += (
                f" From point 0 to point {seq_len - 1},"
                f" there is {article} trend."
            )
