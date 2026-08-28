from abc import abstractmethod, ABC
import numpy as np
from loguru import logger
from typing import List, Optional, Dict, Any, Type
from synth.ts_generator.constants import get_config
from synth.ts_generator.utils.change_utils import generate_ts_change, generate_spike

class BaseChange(ABC):
    """Base class for all local changes in time series"""
    
    def __init__(self, change_type: str, position_start: Optional[int] = None, 
                 amplitude: Optional[float] = None, rng: Optional[np.random.Generator] = None):
        self.change_type = change_type
        self.position_start = position_start
        self.amplitude = amplitude
        self.position_end: Optional[int] = None
        self.detail = ""
        self.rng = rng if rng is not None else np.random.default_rng()
        # Store metadata here (sub-points, specific peak indices, etc.)
        self.context_info: Dict[str, Any] = {
            "key_points": [],  # List of {label: str, index: int}
            "params": {}       # Structural params like direction, sub-amplitudes
        }
    
    @abstractmethod
    def get_min_length(self) -> int:
        """Return minimum length required for this change type"""
        pass
    
    @abstractmethod
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        """Apply the change to the time series and return modified array"""
        pass
    
    def set_position_if_none(self, seq_len: int, existing_objs: List['BaseChange']):
        """
        Set position if not provided using a vectorized mask approach.
        """
        if self.position_start is not None:
            return
        min_length = self.get_min_length()
        max_start_pos = seq_len - min_length
        if max_start_pos < 0:
            raise ValueError(f"Sequence length {seq_len} is too short for {self.change_type} (min {min_length}).")
        
        valid_starts = np.ones(max_start_pos + 1, dtype=bool)
        min_interval = int(max(seq_len / 8, min_length, 20))
        
        for obj in existing_objs:
            block_start = obj.position_start - min_interval - min_length
            block_end = obj.position_end + min_interval
            idx_s = max(0, int(block_start))
            idx_e = min(max_start_pos, int(block_end))
            if idx_s <= idx_e:
                valid_starts[idx_s : idx_e + 1] = False
        
        available_indices = np.flatnonzero(valid_starts)
        if len(available_indices) == 0:
            raise RuntimeError(f"Cannot find a valid position for {self.change_type} (Seq: {seq_len}). Sequence is too crowded.")
        self.position_start = int(self.rng.choice(available_indices))

    def get_remaining_length(self, seq_len: int) -> int:
        """Get remaining length from current position to end of sequence"""
        if self.position_start is None:
            return 0
        return seq_len - self.position_start

    def set_amplitude_if_none(self, overall_amplitude: float, base_factor: float = 0.8, variance: float = 2.0):
        """Set amplitude if not provided"""
        if self.amplitude is None:
            noise = np.abs(self.rng.normal(0.0, variance))
            self.amplitude = (base_factor + noise) * overall_amplitude

    def safe_add(self, y: np.ndarray, start_idx: int, values: np.ndarray) -> int:
        """
        Safely adds values to y starting at start_idx, clipping if it exceeds bounds.
        Returns the effective end index.
        """
        if start_idx >= len(y):
            return start_idx
            
        end_idx = start_idx + len(values)
        if end_idx > len(y):
            valid_len = len(y) - start_idx
            y[start_idx:] += values[:valid_len]
            return len(y)
        else:
            y[start_idx:end_idx] += values
            return end_idx


class ShakeChange(BaseChange):
    """Represents a shake/vibration change"""
    
    def get_min_length(self) -> int:
        return 8
    
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        
        peak_start = self.position_start
        remaining_length = self.get_remaining_length(seq_len)
        
        max_len = max(int(seq_len * 0.15), 16)
        peak_length = min(self.rng.integers(8, max_len + 1), remaining_length)
        
        # Determine shake type
        is_sinusoidal = self.rng.random() > 0.5
        if not is_sinusoidal:
            vals = self.rng.uniform(-1, 1, peak_length) * self.amplitude / 2
            shake_type = "random noise"
        else:
            vals = np.sin(np.linspace(0, 5.0, peak_length)) * self.amplitude / 2
            shake_type = "sinusoidal oscillation"
            
        self.position_end = self.safe_add(y, peak_start, vals)
        
        safe_start = max(0, peak_start)
        safe_end = min(self.position_end - 1, seq_len - 1)

        self.context_info["key_points"] = [
            {"label": "start_context", "index": safe_start},
            {"label": "end_context", "index": safe_end}
        ]
        self.context_info["params"] = {
            "shake_type": shake_type,
            "duration": int(self.position_end - peak_start)
        }
        
        self.detail = (f"a {shake_type} shake with an amplitude of about {self.amplitude:.2f} "
                       f"occurred between point {peak_start} and point {self.position_end}, "
                       f"causing the time series to fluctuate around its baseline for {self.position_end - peak_start} points")
        
        return y


class SpikeChange(BaseChange):
    """Base class for spike changes"""
    
    def get_min_length(self) -> int:
        return 3
    
    def _generate_spike_detail(self, peak_start: int, peak_end: int, spike_top_idx: int, direction: str, seq_len: int):
        """Generate detail description for spike"""
        safe_start = max(0, peak_start)
        safe_top = min(spike_top_idx, seq_len - 1)
        safe_end = min(peak_end - 1, seq_len - 1)
        
        duration = int(peak_end - peak_start)

        self.context_info["key_points"] = [
            {"label": "start_context", "index": safe_start},
            {"label": "peak", "index": safe_top},
            {"label": "end_context", "index": safe_end}
        ]
        self.context_info["params"] = {
            "direction": direction,
            "duration": duration
        }
        
        if direction == "upward":
            self.detail = (f"an upward spike with an amplitude of {self.amplitude:.2f} occurred over {duration} points "
                        f"between point {peak_start} and point {peak_end}, with the time series value rapidly rising "
                        f"from around <|{safe_start}|> to around <|{safe_top}|> and then quickly falling back to around <|{safe_end}|>")
        else:
            self.detail = (f"a downward spike with an amplitude of {self.amplitude:.2f} occurred over {duration} points "
                        f"between point {peak_start} and point {peak_end}, with the time series value rapidly falling "
                        f"from around <|{safe_start}|> to around <|{safe_top}|> and then quickly rising back to around <|{safe_end}|>")

class UpwardSpikeChange(SpikeChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        peak_start = self.position_start
        remaining_length = self.get_remaining_length(seq_len)
        spike = generate_spike(self.amplitude, remaining_length)
        self.position_end = self.safe_add(y, peak_start, spike)
        actual_length = self.position_end - peak_start
        if actual_length > 0:
            spike_top_idx = peak_start + np.argmax(np.abs(spike[:actual_length]))
            self.context_info['sub_points'] = [max(0, peak_start - 1), spike_top_idx]
            self._generate_spike_detail(peak_start, self.position_end, spike_top_idx, "upward", seq_len)
        return y


class DownwardSpikeChange(SpikeChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        peak_start = self.position_start
        remaining_length = self.get_remaining_length(seq_len)
        spike = generate_spike(-self.amplitude, remaining_length)
        self.position_end = self.safe_add(y, peak_start, spike)
        actual_length = self.position_end - peak_start
        if actual_length > 0:
            spike_top_idx = peak_start + np.argmax(np.abs(spike[:actual_length]))
            self.context_info['sub_points'] = [max(0, peak_start - 1), spike_top_idx]
            self._generate_spike_detail(peak_start, self.position_end, spike_top_idx, "downward", seq_len)
        return y


class ContinuousSpikeChange(BaseChange):
    """Base class for continuous spike changes"""
    
    def get_min_length(self) -> int:
        return 10
    
    def _apply_continuous_spikes(self, y: np.ndarray, seq_len: int, direction: int):
        """Apply multiple consecutive spikes"""
        current_pos = self.position_start
        remaining_length = self.get_remaining_length(seq_len)
        
        num_peaks = min(self.rng.integers(2, 6), max(1, remaining_length // 3))
        
        peaks_desc = []
        spike_top_ids = []
        all_amplitudes = []
        all_gaps = []
        key_points = []
        
        key_points.append({"label": "start_context", "index": max(0, self.position_start)})
        
        for i in range(num_peaks):
            rem_len = seq_len - current_pos
            if rem_len < 3:
                break
            
            gap = self.rng.integers(0, min(4, max(1, rem_len - 3)))
            all_gaps.append(int(gap))
            peak_start = current_pos + gap
            
            cur_amplitude = self.rng.uniform(self.amplitude * 0.6, self.amplitude * 1.5)
            all_amplitudes.append(cur_amplitude)
            peaks_desc.append(f"point {peak_start}")
            
            spike_len = seq_len - peak_start
            spike = generate_spike(direction * cur_amplitude, spike_len)
            
            actual_end = self.safe_add(y, peak_start, spike)
            actual_len = actual_end - peak_start
            if actual_len > 0:
                top_idx = peak_start + np.argmax(np.abs(spike[:actual_len]))
                safe_top = min(top_idx, seq_len - 1)
                spike_top_ids.append(safe_top)
                key_points.append({"label": f"peak_{i}", "index": safe_top})
            
            current_pos = actual_end
        
        self.position_end = current_pos
        self.amplitude = float(np.mean(all_amplitudes)) if all_amplitudes else self.amplitude

        safe_end = min(current_pos - 1, seq_len - 1)
        key_points.append({"label": "end_context", "index": safe_end})
        
        self.context_info["key_points"] = key_points
        
        direction_word = "upward" if direction > 0 else "downward"
        action_word = "rising" if direction > 0 else "falling"
        min_amp = min(all_amplitudes) if all_amplitudes else 0
        max_amp = max(all_amplitudes) if all_amplitudes else 0
        safe_start = max(0, self.position_start)
        ids_str = '|> and <|'.join(map(str, spike_top_ids))
        
        avg_gap = np.mean(all_gaps) if all_gaps else 0

        self.context_info["params"] = {
            "num_spikes": len(all_amplitudes),
            # "sub_amplitudes": [round(a, 2) for a in all_amplitudes],
            # "inter_spike_gaps": all_gaps,
            "min_amplitude": round(min_amp, 2), # Used in text
            "max_amplitude": round(max_amp, 2), # Used in text
            "avg_gap": round(avg_gap, 1),         # Used in text
            "direction": "upward" if direction > 0 else "downward"
        }

        self.detail = (f"at {' and '.join(peaks_desc)}, there were {len(all_amplitudes)} consecutive {direction_word} "
                       f"spikes with amplitudes ranging from {min_amp:.2f} to {max_amp:.2f} "
                       f"(average gap between spikes: {avg_gap:.1f} points), with the time series value "
                       f"repeatedly {action_word} sharply from around <|{safe_start}|> to peaks at around <|"
                       f"{ids_str}|>, and then quickly falling back to around <|{safe_end}|>")
        return y


class ContinuousUpwardSpikeChange(ContinuousSpikeChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        return self._apply_continuous_spikes(y, seq_len, 1)


class ContinuousDownwardSpikeChange(ContinuousSpikeChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        return self._apply_continuous_spikes(y, seq_len, -1)


class ConvexChange(BaseChange):
    """Base class for convex changes"""
    def get_min_length(self) -> int:
        return 15
    
    def _apply_convex(self, y: np.ndarray, seq_len: int, direction: int):
        convex_start = self.position_start
        remaining_length = self.get_remaining_length(seq_len)
        
        start_length = min(self.rng.integers(1, 5), remaining_length // 3)
        end_length = min(self.rng.integers(1, 5), max(1, (remaining_length - start_length) // 2))
        min_c = max(int(seq_len * 0.03), 6)
        max_c = max(int(seq_len * 0.2), 16)
        avail = remaining_length - start_length - end_length
        convex_length = max(1, min(self.rng.integers(min_c, max_c + 1), avail))
        
        curr_idx = self.safe_add(y, convex_start, generate_ts_change(start_length, direction * self.amplitude))
        plateau_start_idx = curr_idx
        body_vals = np.full(convex_length, direction * self.amplitude)
        curr_idx = self.safe_add(y, curr_idx, body_vals)
        plateau_end_idx = curr_idx
        return_vals = generate_ts_change(end_length, -direction * self.amplitude) + (direction * self.amplitude)
        self.position_end = self.safe_add(y, curr_idx, return_vals)
        
        safe_start = max(0, convex_start)
        safe_plateau_start = min(plateau_start_idx, seq_len - 1)
        safe_plateau_end = min(plateau_end_idx, seq_len - 1)
        safe_end = min(self.position_end - 1, seq_len - 1)
        
        self.context_info["key_points"] = [
            {"label": "start_context", "index": safe_start},
            {"label": "plateau_start", "index": safe_plateau_start},
            {"label": "plateau_end", "index": safe_plateau_end},
            {"label": "end_context", "index": safe_end}
        ]
        # For upward: start_length is the rise, end_length is the fall.
        # For downward: start_length is the fall (down to valley), end_length is the rise (back up).
        if direction > 0:
            _rise_len, _fall_len = int(start_length), int(end_length)
        else:
            _fall_len, _rise_len = int(start_length), int(end_length)
        self.context_info["params"] = {
            "direction": "upward" if direction > 0 else "downward",
            "rise_length": _rise_len,
            "plateau_length": int(convex_length),
            "fall_length": _fall_len
        }
        
        direction_word = "upward" if direction > 0 else "downward"
        action_words = ("rises", "falls") if direction > 0 else ("falls", "rises")
        
        self.detail = (f"starting from point {convex_start}, the time series value {action_words[0]} over {start_length} points "
                       f"from around <|{safe_start}|> to around <|{safe_plateau_start}|>, then maintains {'an' if direction_word[0] in 'aeiou' else 'a'} {direction_word} plateau "
                       f"with an amplitude of about {self.amplitude:.2f} for {convex_length} points until point {plateau_end_idx}, "
                       f"and finally {action_words[1]} back over {end_length} points to around <|{safe_end}|>")
        return y


class UpwardConvexChange(ConvexChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        return self._apply_convex(y, seq_len, 1)


class DownwardConvexChange(ConvexChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        return self._apply_convex(y, seq_len, -1)


class SuddenChange(BaseChange):
    def get_min_length(self) -> int:
        return 3
    
    def _apply_sudden_change(self, y: np.ndarray, seq_len: int, direction: int):
        remaining_length = self.get_remaining_length(seq_len)
        drop_length = min(self.rng.integers(1, 11), remaining_length)
        vals = generate_ts_change(drop_length, direction * self.amplitude)
        _ = self.safe_add(y, self.position_start, vals)
        self.position_end = self.position_start + drop_length
        if self.position_end < seq_len:
            y[self.position_end:] += direction * self.amplitude
        
        safe_prev = max(0, self.position_start - 1)
        safe_end = min(self.position_end - 1, seq_len - 1)

        # Initialize context_info
        self.context_info["key_points"] = [
            {"label": "start_context", "index": safe_prev},
            {"label": "end_context", "index": safe_end}
        ]
        self.context_info["params"] = {
            "direction": "increase" if direction > 0 else "decrease",
            "change_length": int(drop_length),
            "has_recovery": False,
            "recovery_amplitude": None,
            "recovery_length": None
        }
        
        action_word = "increase" if direction > 0 else "decrease"
        movement_word = "rising" if direction > 0 else "falling"
        
        self.detail = (f"a sudden {action_word} with an amplitude of {self.amplitude:.2f} occurred over {drop_length} points "
                       f"between point {self.position_start} and point {self.position_end}, "
                       f"with the time series value {movement_word} from around <|{safe_prev}|> to around <|{safe_end}|>")
        
        # Recovery logic
        enable_drop = get_config().enable_drop_prompt
        if self.rng.random() < 0.5:
            avail_rec = seq_len - self.position_end
            recover_length = min(self.rng.integers(1, 11), avail_rec)
            if recover_length > 0:
                recover_amplitude = self.rng.uniform(0, self.amplitude / 3)
                rec_vals = generate_ts_change(recover_length, -direction * recover_amplitude)
                _ = self.safe_add(y, self.position_end, rec_vals)
                rec_end = self.position_end + recover_length
                if rec_end < seq_len:
                    y[rec_end:] -= direction * recover_amplitude
                
                safe_rec_end = min(rec_end - 1, seq_len - 1)
                
                # Update context_info with recovery details
                self.context_info["key_points"].append({"label": "recovery_end", "index": safe_rec_end})
                self.context_info["params"]["has_recovery"] = True
                self.context_info["params"]["recovery_amplitude"] = round(recover_amplitude, 2)
                self.context_info["params"]["recovery_length"] = int(recover_length)
                
                if enable_drop:
                    recovery_word = "partial recovery drop" if direction > 0 else "partial recovery rise"
                    recovery_movement = "falling" if direction > 0 else "rising"
                    self.detail += (f"; this was followed by a {recovery_word} of {recover_amplitude:.2f} over {recover_length} points "
                                    f"between point {self.position_end} and point {rec_end}, "
                                    f"with the time series value {recovery_movement} back to around <|{safe_rec_end}|>")
        return y


class SuddenIncreaseChange(SuddenChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        return self._apply_sudden_change(y, seq_len, 1)


class SuddenDecreaseChange(SuddenChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        return self._apply_sudden_change(y, seq_len, -1)


class TwoPhaseChange(BaseChange):
    """Base class for two-phase changes"""
    def get_min_length(self) -> int:
        return 10


class RapidRiseSlowDeclineChange(TwoPhaseChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        rem_len = self.get_remaining_length(seq_len)

        rise_len = min(self.rng.integers(1, 6), rem_len // 2)
        fall_len = min(self.rng.integers(max(int(seq_len * 0.05), 8), max(int(seq_len * 0.15), 20)), rem_len - rise_len)

        mid_point = self.position_start + rise_len
        curr = self.safe_add(y, self.position_start, generate_ts_change(rise_len, self.amplitude))
        self.position_end = self.safe_add(y, curr, generate_ts_change(fall_len, -self.amplitude) + self.amplitude)

        safe_start = max(0, self.position_start - 1)
        safe_mid = min(mid_point, seq_len - 1)
        safe_end = min(self.position_end - 1, seq_len - 1)

        self.context_info["key_points"] = [
            {"label": "start_context", "index": safe_start},
            {"label": "peak", "index": safe_mid},
            {"label": "end_context", "index": safe_end}
        ]
        self.context_info["params"] = {
            "pattern": "rapid_rise_slow_decline",
            "rise_length": int(rise_len),
            "fall_length": int(fall_len),
            "rise_rate": round(self.amplitude / rise_len, 2) if rise_len > 0 else 0,
            "fall_rate": round(self.amplitude / fall_len, 2) if fall_len > 0 else 0
        }
        
        rise_rate = round(self.amplitude / rise_len, 2) if rise_len > 0 else 0
        fall_rate = round(self.amplitude / fall_len, 2) if fall_len > 0 else 0

        self.detail = (f"a rapid rise with an amplitude of {self.amplitude:.2f} occurred over {rise_len} points "
                       f"(rate: {rise_rate:.3f}/point) between point {self.position_start} and point {mid_point}, "
                       f"with the time series value rising quickly from around <|{safe_start}|> to around <|{safe_mid}|>, "
                       f"followed by a slow decline over {fall_len} points (rate: {fall_rate:.3f}/point) "
                       f"between point {mid_point} and point {self.position_end} gradually returning to around <|{safe_end}|>")
        return y


class SlowRiseRapidDeclineChange(TwoPhaseChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        rem_len = self.get_remaining_length(seq_len)
        
        rise_len = min(self.rng.integers(max(int(seq_len * 0.05), 8), max(int(seq_len * 0.15), 20)), rem_len // 2)
        fall_len = min(self.rng.integers(1, 6), rem_len - rise_len)
        
        peak_pt = self.position_start + rise_len
        
        curr = self.safe_add(y, self.position_start, generate_ts_change(rise_len, self.amplitude))
        self.position_end = self.safe_add(y, curr, generate_ts_change(fall_len, -self.amplitude) + self.amplitude)
        
        safe_start = max(0, self.position_start)
        safe_peak = min(peak_pt, seq_len - 1)
        safe_end = min(self.position_end - 1, seq_len - 1)

        rise_rate = round(self.amplitude / rise_len, 2) if rise_len > 0 else 0
        fall_rate = round(self.amplitude / fall_len, 2) if fall_len > 0 else 0

        self.context_info["key_points"] = [
            {"label": "start_context", "index": safe_start},
            {"label": "peak", "index": safe_peak},
            {"label": "end_context", "index": safe_end}
        ]
        self.context_info["params"] = {
            "pattern": "slow_rise_rapid_decline",
            "rise_length": int(rise_len),
            "fall_length": int(fall_len),
            "rise_rate": rise_rate,
            "fall_rate": fall_rate
        }
        
        self.detail = (f"starting from point {self.position_start}, the time series value slowly rises over {rise_len} points "
                       f"(rate: {rise_rate:.3f}/point) from around <|{safe_start}|>, reaching a peak of amplitude {self.amplitude:.2f} "
                       f"at point {peak_pt} around <|{safe_peak}|>, followed by a rapid decline over {fall_len} points "
                       f"(rate: {fall_rate:.3f}/point) between point {peak_pt} and point {self.position_end} "
                       f"quickly dropping back to around <|{safe_end}|>")
        return y


class RapidDeclineSlowRiseChange(TwoPhaseChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        rem_len = self.get_remaining_length(seq_len)
        
        drop_len = min(self.rng.integers(1, 6), rem_len // 2)
        rise_len = min(self.rng.integers(max(int(seq_len * 0.05), 8), max(int(seq_len * 0.15), 20)), rem_len - drop_len)
        
        trough_pt = self.position_start + drop_len

        curr = self.safe_add(y, self.position_start, generate_ts_change(drop_len, -self.amplitude))
        self.position_end = self.safe_add(y, curr, generate_ts_change(rise_len, self.amplitude) - self.amplitude)

        safe_start = max(0, self.position_start - 1)
        safe_trough = min(trough_pt, seq_len - 1)
        safe_end = min(self.position_end - 1, seq_len - 1)

        drop_rate = round(self.amplitude / drop_len, 2) if drop_len > 0 else 0
        rise_rate = round(self.amplitude / rise_len, 2) if rise_len > 0 else 0

        self.context_info["key_points"] = [
            {"label": "start_context", "index": safe_start},
            {"label": "trough", "index": safe_trough},
            {"label": "end_context", "index": safe_end}
        ]
        self.context_info["params"] = {
            "pattern": "rapid_decline_slow_rise",
            "drop_length": int(drop_len),
            "rise_length": int(rise_len),
            "drop_rate": drop_rate,
            "rise_rate": rise_rate
        }

        self.detail = (f"a rapid decline with an amplitude of {self.amplitude:.2f} occurred over {drop_len} points "
                       f"(rate: {drop_rate:.3f}/point) between point {self.position_start} and point {trough_pt}, "
                       f"with the time series value falling quickly from around <|{safe_start}|> to around <|{safe_trough}|>, "
                       f"followed by a slow rise over {rise_len} points (rate: {rise_rate:.3f}/point) "
                       f"between point {trough_pt} and point {self.position_end} gradually returning to around <|{safe_end}|>")
        return y


class SlowDeclineRapidRiseChange(TwoPhaseChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude)
        rem_len = self.get_remaining_length(seq_len)
        
        drop_len = min(self.rng.integers(max(int(seq_len * 0.05), 8), max(int(seq_len * 0.15), 20)), rem_len // 2)
        rise_len = min(self.rng.integers(1, 6), rem_len - drop_len)
        
        trough_pt = self.position_start + drop_len
        
        curr = self.safe_add(y, self.position_start, generate_ts_change(drop_len, -self.amplitude))
        self.position_end = self.safe_add(y, curr, generate_ts_change(rise_len, self.amplitude) - self.amplitude)
        
        safe_start = max(0, self.position_start)
        safe_trough = min(trough_pt, seq_len - 1)
        safe_end = min(self.position_end - 1, seq_len - 1)

        drop_rate = round(self.amplitude / drop_len, 2) if drop_len > 0 else 0
        rise_rate = round(self.amplitude / rise_len, 2) if rise_len > 0 else 0

        self.context_info["key_points"] = [
            {"label": "start_context", "index": safe_start},
            {"label": "trough", "index": safe_trough},
            {"label": "end_context", "index": safe_end}
        ]
        self.context_info["params"] = {
            "pattern": "slow_decline_rapid_rise",
            "drop_length": int(drop_len),
            "rise_length": int(rise_len),
            "drop_rate": drop_rate,
            "rise_rate": rise_rate
        }
        
        self.detail = (f"starting from point {self.position_start}, the time series value slowly declines over {drop_len} points "
                       f"(rate: {drop_rate:.3f}/point) from around <|{safe_start}|>, reaching a low point of amplitude {self.amplitude:.2f} "
                       f"at point {trough_pt} around <|{safe_trough}|>, followed by a rapid rise over {rise_len} points "
                       f"(rate: {rise_rate:.3f}/point) between point {trough_pt} and point {self.position_end} "
                       f"quickly returning to around <|{safe_end}|>")
        return y


class SpikeFollowedByChange(BaseChange):
    def get_min_length(self) -> int:
        return 8


class DecreaseAfterUpwardSpikeChange(SpikeFollowedByChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        rem_len = self.get_remaining_length(seq_len)
        
        fall_amplitude = self.rng.uniform(0.1, 0.7) * self.amplitude
        
        spike = generate_spike(self.amplitude, rem_len)
        peak_len = min(len(spike), rem_len // 2)
        fall_len = min(self.rng.integers(2, max(int(seq_len * 0.05), 12) + 1), rem_len - peak_len)
        
        curr = self.position_start
        spike_part = spike[:peak_len]
        y[curr : curr + len(spike_part)] += spike_part
        spike_top_idx = curr + np.argmax(np.abs(spike_part))
        transition_pt = curr + peak_len
        curr = transition_pt
        
        self.position_end = self.safe_add(y, curr, generate_ts_change(fall_len, -fall_amplitude))
        if self.position_end < seq_len:
            y[self.position_end:] -= fall_amplitude

        safe_start = max(0, self.position_start - 1)
        safe_peak = min(spike_top_idx, seq_len - 1)
        safe_transition = min(transition_pt, seq_len - 1)
        safe_end = min(self.position_end - 1, seq_len - 1)

        self.context_info["key_points"] = [
            {"label": "start_context", "index": safe_start},
            {"label": "spike_peak", "index": safe_peak},
            {"label": "transition_point", "index": safe_transition},
            {"label": "end_context", "index": safe_end}
        ]
        self.context_info["params"] = {
            "spike_amplitude": round(float(self.amplitude), 2),
            "spike_direction": "upward",
            "spike_length": int(peak_len),
            "followup_type": "decrease",
            "followup_amplitude": round(float(fall_amplitude), 2),
            "followup_length": int(fall_len)
        }
        
        self.detail = (f"an upward spike with an amplitude of {self.amplitude:.2f} occurred over {peak_len} points "
                       f"between point {self.position_start} and point {transition_pt}, "
                       f"with the time series value rapidly rising from around <|{safe_start}|> to around <|{safe_peak}|> "
                       f"and quickly falling back to around <|{safe_transition}|>; "
                       f"this was followed by a further decline of {fall_amplitude:.2f} over {fall_len} points "
                       f"between point {transition_pt} and point {self.position_end} to around <|{safe_end}|>")
        return y


class IncreaseAfterDownwardSpikeChange(SpikeFollowedByChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        rem_len = self.get_remaining_length(seq_len)
        
        rise_amplitude = self.rng.uniform(0.1, 0.7) * self.amplitude
        
        spike = generate_spike(-self.amplitude, rem_len)
        peak_len = min(len(spike), rem_len // 2)
        rise_len = min(self.rng.integers(2, max(int(seq_len * 0.05), 12) + 1), rem_len - peak_len)
        
        curr = self.position_start
        spike_part = spike[:peak_len]
        y[curr : curr + len(spike_part)] += spike_part
        spike_top_idx = curr + np.argmax(np.abs(spike_part))
        transition_pt = curr + peak_len
        curr = transition_pt
        
        self.position_end = self.safe_add(y, curr, generate_ts_change(rise_len, rise_amplitude))
        if self.position_end < seq_len:
            y[self.position_end:] += rise_amplitude

        safe_start = max(0, self.position_start - 1)
        safe_trough = min(spike_top_idx, seq_len - 1)
        safe_transition = min(transition_pt, seq_len - 1)
        safe_end = min(self.position_end - 1, seq_len - 1)

        self.context_info["key_points"] = [
            {"label": "start_context", "index": safe_start},
            {"label": "spike_trough", "index": safe_trough},
            {"label": "transition_point", "index": safe_transition},
            {"label": "end_context", "index": safe_end}
        ]
        self.context_info["params"] = {
            "spike_amplitude": round(float(self.amplitude), 2),
            "spike_direction": "downward",
            "spike_length": int(peak_len),
            "followup_type": "increase",
            "followup_amplitude": round(float(rise_amplitude), 2),
            "followup_length": int(rise_len)
        }
        
        self.detail = (f"a downward spike with an amplitude of {self.amplitude:.2f} occurred over {peak_len} points "
                       f"between point {self.position_start} and point {transition_pt}, "
                       f"with the time series value rapidly falling from around <|{safe_start}|> to around <|{safe_trough}|> "
                       f"and quickly rising back to around <|{safe_transition}|>; "
                       f"this was followed by a further increase of {rise_amplitude:.2f} over {rise_len} points "
                       f"between point {transition_pt} and point {self.position_end} to around <|{safe_end}|>")
        return y


class IncreaseAfterUpwardSpikeChange(SpikeFollowedByChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        rem_len = self.get_remaining_length(seq_len)
        
        rise_amplitude = self.rng.uniform(0.1, 0.7) * self.amplitude
        
        spike = generate_spike(self.amplitude, rem_len)
        peak_len = min(len(spike), rem_len // 2)
        rise_len = min(self.rng.integers(2, max(int(seq_len * 0.05), 12) + 1), rem_len - peak_len)
        
        curr = self.position_start
        spike_part = spike[:peak_len]
        y[curr : curr + len(spike_part)] += spike_part
        spike_top_idx = curr + np.argmax(np.abs(spike_part))
        transition_pt = curr + peak_len
        curr = transition_pt
        
        curr = self.safe_add(y, curr, generate_ts_change(rise_len, rise_amplitude))
        if curr < seq_len:
            y[curr:] += rise_amplitude
        self.position_end = curr

        safe_prev = max(0, self.position_start - 1)
        safe_peak = min(spike_top_idx, seq_len - 1)
        safe_transition = min(transition_pt, seq_len - 1)
        safe_end = min(self.position_end - 1, seq_len - 1)

        self.context_info["key_points"] = [
            {"label": "start_context", "index": safe_prev},
            {"label": "spike_peak", "index": safe_peak},
            {"label": "transition_point", "index": safe_transition},
            {"label": "end_context", "index": safe_end}
        ]
        self.context_info["params"] = {
            "spike_amplitude": round(float(self.amplitude), 2),
            "spike_direction": "upward",
            "spike_length": int(peak_len),
            "followup_type": "increase",
            "followup_amplitude": round(float(rise_amplitude), 2),
            "followup_length": int(rise_len)
        }
        
        self.detail = (f"an upward spike with an amplitude of {self.amplitude:.2f} occurred over {peak_len} points "
                       f"between point {self.position_start} and point {transition_pt}, "
                       f"with the time series value rapidly rising from around <|{safe_prev}|> to around <|{safe_peak}|> "
                       f"and quickly falling back to around <|{safe_transition}|>; "
                       f"this was followed by a further increase of {rise_amplitude:.2f} over {rise_len} points "
                       f"between point {transition_pt} and point {self.position_end} to around <|{safe_end}|>")
        return y


class DecreaseAfterDownwardSpikeChange(SpikeFollowedByChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        rem_len = self.get_remaining_length(seq_len)
        
        fall_amplitude = self.rng.uniform(0.1, 0.7) * self.amplitude
        
        spike = generate_spike(-self.amplitude, rem_len)
        peak_len = min(len(spike), rem_len // 2)
        fall_len = min(self.rng.integers(2, max(int(seq_len * 0.05), 12) + 1), rem_len - peak_len)
        
        curr = self.position_start
        spike_part = spike[:peak_len]
        y[curr : curr + len(spike_part)] += spike_part
        spike_top_idx = curr + np.argmax(np.abs(spike_part))
        transition_pt = curr + peak_len
        curr = transition_pt
        
        curr = self.safe_add(y, curr, generate_ts_change(fall_len, -fall_amplitude))
        if curr < seq_len:
            y[curr:] -= fall_amplitude
        self.position_end = curr
        
        safe_start = max(0, self.position_start)
        safe_trough = min(spike_top_idx, seq_len - 1)
        safe_transition = min(transition_pt, seq_len - 1)
        safe_end = min(self.position_end - 1, seq_len - 1)

        self.context_info["key_points"] = [
            {"label": "start_context", "index": safe_start},
            {"label": "spike_trough", "index": safe_trough},
            {"label": "transition_point", "index": safe_transition},
            {"label": "end_context", "index": safe_end}
        ]
        self.context_info["params"] = {
            "spike_amplitude": round(float(self.amplitude), 2),
            "spike_direction": "downward",
            "spike_length": int(peak_len),
            "followup_type": "decrease",
            "followup_amplitude": round(float(fall_amplitude), 2),
            "followup_length": int(fall_len)
        }
        
        self.detail = (f"a downward spike with an amplitude of {self.amplitude:.2f} occurred over {peak_len} points "
                       f"between point {self.position_start} and point {transition_pt}, "
                       f"with the time series value rapidly falling from around <|{safe_start}|> to around <|{safe_trough}|> "
                       f"and quickly rising back to around <|{safe_transition}|>; "
                       f"this was followed by a further decline of {fall_amplitude:.2f} over {fall_len} points "
                       f"between point {transition_pt} and point {self.position_end} to around <|{safe_end}|>")
        return y


class WideSpikeChange(BaseChange):
    def get_min_length(self) -> int:
        return 16


class WideUpwardSpikeChange(WideSpikeChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        rem_len = self.get_remaining_length(seq_len)
        
        min_slope = max(int(seq_len * 0.02), 4)
        max_slope = max(int(seq_len * 0.08), 8)
        
        rise_len = min(self.rng.integers(min_slope, max_slope + 1), rem_len // 3)
        peak_len = min(self.rng.integers(1, 4), (rem_len - rise_len) // 2)
        fall_len = min(self.rng.integers(min_slope, max_slope + 1), rem_len - rise_len - peak_len)
        
        curr = self.position_start
        peak_start_pt = curr + rise_len
        
        # Apply Rise
        curr = self.safe_add(y, curr, generate_ts_change(rise_len, self.amplitude))
        peak_end_pt = curr + peak_len
        
        # Apply Plateau
        curr = self.safe_add(y, curr, np.full(peak_len, self.amplitude))
        
        # Apply Fall
        self.position_end = self.safe_add(y, curr, generate_ts_change(fall_len, -self.amplitude) + self.amplitude)

        safe_start = max(0, self.position_start - 1)
        safe_peak_start = min(peak_start_pt, seq_len - 1)
        safe_peak_end = min(peak_end_pt, seq_len - 1)
        safe_end = min(self.position_end - 1, seq_len - 1)
        
        rise_rate = round(self.amplitude / rise_len, 2) if rise_len > 0 else 0
        fall_rate = round(self.amplitude / fall_len, 2) if fall_len > 0 else 0

        self.context_info["key_points"] = [
            {"label": "start_context", "index": safe_start},
            {"label": "plateau_start", "index": safe_peak_start},
            {"label": "plateau_end", "index": safe_peak_end},
            {"label": "end_context", "index": safe_end}
        ]
        self.context_info["params"] = {
            "direction": "upward",
            "rise_length": int(rise_len),
            "plateau_length": int(peak_len),
            "fall_length": int(fall_len),
            "rise_rate": rise_rate,
            "fall_rate": fall_rate
        }
        
        self.detail = (f"a wide upward spike with an amplitude of {self.amplitude:.2f} occurred between point {self.position_start} "
                       f"and point {self.position_end}: the time series value slowly rises over {rise_len} points "
                       f"(rate: {rise_rate:.3f}/point) from around <|{safe_start}|> to around <|{safe_peak_start}|>, "
                       f"maintains a short plateau for {peak_len} points until point {peak_end_pt} around <|{safe_peak_end}|>, "
                       f"then slowly declines over {fall_len} points (rate: {fall_rate:.3f}/point) back to around <|{safe_end}|>")        
        return y


class WideDownwardSpikeChange(WideSpikeChange):
    def apply_change(self, y: np.ndarray, seq_len: int, overall_amplitude: float) -> np.ndarray:
        self.set_amplitude_if_none(overall_amplitude, variance=6.0)
        rem_len = self.get_remaining_length(seq_len)
        
        min_slope = max(int(seq_len * 0.02), 4)
        max_slope = max(int(seq_len * 0.08), 8)
        
        drop_len = min(self.rng.integers(min_slope, max_slope + 1), rem_len // 3)
        trough_len = min(self.rng.integers(1, 4), (rem_len - drop_len) // 2)
        rise_len = min(self.rng.integers(min_slope, max_slope + 1), rem_len - drop_len - trough_len)
        
        curr = self.position_start
        trough_start_pt = curr + drop_len
        
        # Apply Drop
        curr = self.safe_add(y, curr, generate_ts_change(drop_len, -self.amplitude))
        trough_end_pt = curr + trough_len
        
        # Apply Trough (Bottom Plateau)
        curr = self.safe_add(y, curr, np.full(trough_len, -self.amplitude))
        
        # Apply Rise
        self.position_end = self.safe_add(y, curr, generate_ts_change(rise_len, self.amplitude) - self.amplitude)

        safe_start = max(0, self.position_start - 1)
        safe_trough_start = min(trough_start_pt, seq_len - 1)
        safe_trough_end = min(trough_end_pt, seq_len - 1)
        safe_end = min(self.position_end - 1, seq_len - 1)
        
        drop_rate = round(self.amplitude / drop_len, 2) if drop_len > 0 else 0
        rise_rate = round(self.amplitude / rise_len, 2) if rise_len > 0 else 0

        self.context_info["key_points"] = [
            {"label": "start_context", "index": safe_start},
            {"label": "trough_start", "index": safe_trough_start},
            {"label": "trough_end", "index": safe_trough_end},
            {"label": "end_context", "index": safe_end}
        ]
        self.context_info["params"] = {
            "direction": "downward",
            "drop_length": int(drop_len),
            "trough_length": int(trough_len),
            "rise_length": int(rise_len),
            "drop_rate": drop_rate,
            "rise_rate": rise_rate
        }
        self.detail = (f"a wide downward spike with an amplitude of {self.amplitude:.2f} occurred between point {self.position_start} "
                       f"and point {self.position_end}: the time series value slowly declines over {drop_len} points "
                       f"(rate: {drop_rate:.3f}/point) from around <|{safe_start}|> to around <|{safe_trough_start}|>, "
                       f"maintains a short trough for {trough_len} points until point {trough_end_pt} around <|{safe_trough_end}|>, "
                       f"then slowly rises over {rise_len} points (rate: {rise_rate:.3f}/point) back to around <|{safe_end}|>")
        return y


# Factory class to create appropriate change objects
class ChangeFactory:
    """Factory class to create change objects based on change type"""
    
    _change_classes = {
        "shake": ShakeChange,
        "upward spike": UpwardSpikeChange,
        "downward spike": DownwardSpikeChange,
        "continuous upward spike": ContinuousUpwardSpikeChange,
        "continuous downward spike": ContinuousDownwardSpikeChange,
        "upward convex": UpwardConvexChange,
        "downward convex": DownwardConvexChange,
        "sudden increase": SuddenIncreaseChange,
        "sudden decrease": SuddenDecreaseChange,
        "rapid rise followed by slow decline": RapidRiseSlowDeclineChange,
        "slow rise followed by rapid decline": SlowRiseRapidDeclineChange,
        "rapid decline followed by slow rise": RapidDeclineSlowRiseChange,
        "slow decline followed by rapid rise": SlowDeclineRapidRiseChange,
        "decrease after upward spike": DecreaseAfterUpwardSpikeChange,
        "increase after downward spike": IncreaseAfterDownwardSpikeChange,
        "increase after upward spike": IncreaseAfterUpwardSpikeChange,
        "decrease after downward spike": DecreaseAfterDownwardSpikeChange,
        "wide upward spike": WideUpwardSpikeChange,
        "wide downward spike": WideDownwardSpikeChange,
    }
    
    @classmethod
    def create_change(cls, change_type: str, position_start: Optional[int] = None, 
                      amplitude: Optional[float] = None, rng: Optional[np.random.Generator] = None) -> BaseChange:
        """Create a change object based on change type"""
        if change_type not in cls._change_classes:
            raise ValueError(f"Unknown change type: {change_type}")
        
        return cls._change_classes[change_type](change_type, position_start, amplitude, rng)
    
    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported change types"""
        return list(cls._change_classes.keys())


# Change types that blend into periodic/seasonal fluctuations and are not easily distinguishable.
# These are gradual, slow-moving shapes (convex plateaus, multi-phase rises/falls, wide spikes,
# shake oscillations) that look visually similar to a seasonal wave segment.
_PERIODIC_EXCLUDED_TYPES: frozenset = frozenset({
    'shake',
    'upward convex', 'downward convex',
    'wide upward spike', 'wide downward spike',
    'rapid rise followed by slow decline', 'slow rise followed by rapid decline',
    'rapid decline followed by slow rise', 'slow decline followed by rapid rise',
})


def generate_local_chars(attribute_pool: Dict[str, Any], overall_amplitude: float, seq_len: int, seed: Optional[int] = None):
    """
    Generate a time series with local characteristics using object-oriented approach.

    Args:
        attribute_pool (dict): Pool of attributes containing local characteristics
        overall_amplitude (float): Overall amplitude for scaling
        seq_len (int): Length of the time series
        seed (int): Optional random seed for reproducibility

    Returns:
        np.ndarray: Modified time series with local changes applied
    """
    y = np.zeros(seq_len)
    existing_objs: List[BaseChange] = []

    # Initialize Random Number Generator
    rng = np.random.default_rng(seed)

    verbose = get_config().local_change_verbose

    # If the TS has a seasonal/periodic component, skip change types whose gradual shape
    # blends visually with the seasonal wave (making the local event hard to see).
    seasonal_amp = attribute_pool.get('seasonal', {}).get('amplitude', 0.0) if isinstance(attribute_pool.get('seasonal'), dict) else 0.0
    has_periodicity = seasonal_amp > 0

    updated_local = []

    for local_char in attribute_pool.get("local", []):
        try:
            # Skip gradual change types that are not distinguishable on periodic time series.
            if has_periodicity and local_char.get("type") in _PERIODIC_EXCLUDED_TYPES:
                if verbose:
                    logger.debug(f"Skipping {local_char['type']}: not visible in periodic TS")
                continue

            change_obj = ChangeFactory.create_change(
                local_char["type"],
                local_char.get("position_start"),
                local_char.get("amplitude"),
                rng=rng
            )

            # Set position using vectorized mask approach
            change_obj.set_position_if_none(seq_len, existing_objs)
            # Check if there's enough space for this change type
            if change_obj.get_remaining_length(seq_len) < change_obj.get_min_length():
                if verbose:
                    logger.debug(f"Skipping {local_char['type']}: not enough remaining length")
                continue
            existing_objs.append(change_obj)
            
            # Apply the current change
            y = change_obj.apply_change(y, seq_len, overall_amplitude)
            
            # --- META DATA TRANSFER START ---
            
            # 1. Pass the Blueprint (Intent) - key points for downstream processing
            if change_obj.context_info.get("key_points"):
                local_char["key_points_intent"] = change_obj.context_info["key_points"]
                
            # 2. Pass structural parameters
            if change_obj.context_info.get("params"):
                local_char["params"] = change_obj.context_info["params"]
                
            # 3. Pass sub_points if available (for backward compatibility)
            if change_obj.context_info.get("sub_points"):
                local_char["sub_points"] = change_obj.context_info["sub_points"]
                
            # --- META DATA TRANSFER END ---
            
            # Final Sanity Check
            if change_obj.position_end > seq_len:
                if verbose:
                    logger.warning(f"Change {change_obj.change_type} clipped at {seq_len}")
            
            # Update local_char with computed values
            local_char.update({
                "position_start": int(change_obj.position_start),
                "position_end": int(change_obj.position_end),
                "amplitude": round(float(change_obj.amplitude), 2),
                "detail": change_obj.detail
            })
            updated_local.append(local_char)
            
            if verbose:
                logger.debug(f"Applied {change_obj.change_type} at [{change_obj.position_start}, {change_obj.position_end}] "
                             f"with amplitude {change_obj.amplitude:.2f}")
            
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Skipping change '{local_char.get('type')}': {e}")
            if verbose:
                logger.opt(exception=True).debug("Full traceback")
            continue
        except Exception as e:
            logger.warning(f"Error applying change '{local_char.get('type')}' ({seq_len=}): {e}")
            logger.opt(exception=True).debug("Full traceback")
            continue
    
    # Sort by position
    updated_local.sort(key=lambda x: x["position_start"])
    attribute_pool["local"] = updated_local
    
    return y