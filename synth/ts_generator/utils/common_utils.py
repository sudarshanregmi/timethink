"""
Common utilities for time series batch generation and dataset creation.

This module provides high-level utilities for generating batches of time series
for training and evaluation, built on top of the core generation in ts_generator.
"""

import random
import copy
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union
from pathlib import Path
from synth.ts_generator.utils.probability_utils import weighted_random_choice

DEFAULT_THRESHOLD = 15

_VOWEL_SOUNDS = frozenset('aeiouAEIOU')


def article(word: str) -> str:
    """Return 'an' if *word* starts with a vowel sound, else 'a'."""
    return 'an' if word and word[0] in _VOWEL_SOUNDS else 'a'

# Core generation imports
from synth.ts_generator.generate import (
    generate_random_attributes,
    generate_time_series
)
from synth.ts_generator.utils.trend_utils import (
    generate_random_points,
    generate_trend_prompt,
    generate_trend_curve,
    generate_trend_list
)

# Utility imports
from synth.ts_generator.utils.serialization_utils import (
    NumpyEncoder,
    deep_copy_dict,
    attribute_pool_to_json
)
from synth.ts_generator.utils.statistics_utils import (
    add_statistics_to_pool
)
from synth.ts_generator.utils.formatting_utils import (
    format_float,
    format_list_natural_language
)

# Re-export for backward compatibility
__all__ = [
    # Classes
    'BatchSampleGenerator',
    'NumpyEncoder',
    # Functions
    'load_cfg',
    'format_float',
    'format_list_natural_language',
    'attribute_pool_to_json',
    'has_local_event_near',
    'has_local_event_end_near',
    'get_local_event_positions',
    'get_local_event_end_positions',
    'sanitize_attributes_for_sync',
    'determine_sequence_length',
    'generate_threshold_for_sample',
    'shuffle_aligned_data',
    'get_change_position_range',
    'get_random_change_position',
    'apply_trend_to_timeseries',
    'inject_varied_noise',
    'find_point_far_from_all_events',
    'find_point_at_distance_range',
    # Configs
    'DEFAULT_LOCAL_FEATURE_CONFIG',
    'DEFAULT_SHAPE_FEATURE_CONFIG',
]


# =============================================================================
# Configuration Loading
# =============================================================================

def load_cfg(config_path: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration dictionary or list
        
    Raises:
        ValueError: If file extension is not supported
    """
    import yaml
    import json
    
    path = Path(config_path)
    extension = path.suffix.lower()
    
    with open(path, 'r', encoding='utf-8') as f:
        if extension in ('.yaml', '.yml'):
            return yaml.safe_load(f)
        elif extension == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")


# =============================================================================
# Noise Injection (Post-Generation Noise Augmentation)
# =============================================================================

def inject_varied_noise(
    timeseries: np.ndarray,
    ref_amplitude: Optional[float] = None
) -> np.ndarray:
    """Inject Gaussian noise into a time series.

    Noise strength varies uniformly in [2%, 8%] of ref_amplitude.

    Callers should pass overall_amplitude for ALL modes (from AMPLITUDE_EXPONENTS).

    Args:
        timeseries: Input time series array
        ref_amplitude: Reference amplitude for noise scaling.
            If None, computed as y_range from data.

    Returns:
        Time series with added noise
    """
    if ref_amplitude is None:
        ref_amplitude = np.max(timeseries) - np.min(timeseries)
        if ref_amplitude < 1e-8:
            ref_amplitude = 1.0  # Avoid division by zero

    # Sample noise strength as percentage of ref_amplitude [2%, 8%]
    noise_strength_pct = random.uniform(0.02, 0.08)
    noise_std = noise_strength_pct * ref_amplitude
    seq_len = len(timeseries)

    noise = np.random.normal(0, noise_std, seq_len)
    return timeseries + noise


# =============================================================================
# Feature Configurations
# =============================================================================

DEFAULT_LOCAL_FEATURE_CONFIG = {
    "overall_attribute": {
        "seasonal": {
            "no periodic fluctuation": 0.7,
            "periodic fluctuation": 0.3
        },
        "trend": {
            "decrease": 0.2,
            "increase": 0.2,
            "keep steady": 0.6
        },
        "frequency": {
            "high frequency": 0.5,
            "low frequency": 0.5
        },
        "noise": {
            "noisy": 0.3,
            "smooth": 0.7
        }
    },
    "change": {
        "shake": 2,
        "upward spike": 10,
        "downward spike": 6,
        "continuous upward spike": 4,
        "continuous downward spike": 2,
        "upward convex": 2,
        "downward convex": 2,
        "sudden increase": 2,
        "sudden decrease": 2,
        "rapid rise followed by slow decline": 2,
        "slow rise followed by rapid decline": 2,
        "rapid decline followed by slow rise": 2,
        "slow decline followed by rapid rise": 2,
        "decrease after upward spike": 3,
        "increase after downward spike": 3,
        "increase after upward spike": 3,
        "decrease after downward spike": 3,
        "wide upward spike": 3,
        "wide downward spike": 3
    }
}

DEFAULT_SHAPE_FEATURE_CONFIG = {
    "overall_attribute": {
        "seasonal": {
            "no periodic fluctuation": 0.9,
            "periodic fluctuation": 0.1
        },
        "trend": {
            "keep steady": 0.6
        },
        "frequency": {
            "high frequency": 0.5,
            "low frequency": 0.5
        },
        "noise": {
            "noisy": 0.5,
            "smooth": 0.5
        },
    },
    "change": {
        "shake": 2,
        "upward spike": 10,
        "downward spike": 6,
        "continuous upward spike": 4,
        "continuous downward spike": 2,
        "wide upward spike": 3,
        "wide downward spike": 3,
    }
}


# =============================================================================
# Sequence Length Utilities
# =============================================================================

def determine_sequence_length(
    configured_len: Optional[int] = None,
    default_len: int = 256,
    min_len: int = 16,
    max_len: int = 4096,
    default_probability: float = 0.5
) -> int:
    """
    Determine sequence length with optional randomization.
    
    Args:
        configured_len: Pre-configured length (if any)
        default_len: Default length to use
        min_len: Minimum random length
        max_len: Maximum random length
        default_probability: Probability of using default_len vs random
        
    Returns:
        Determined sequence length
    """
    if configured_len is not None:
        return configured_len
    if random.random() > (1 - default_probability):
        return default_len
    return random.randint(min_len, max_len)


def generate_threshold_for_sample(
    seq_len: int,
    force_none_probability: float = 0.2
) -> Optional[int]:
    """
    Generate a threshold value for sample generation.
    
    Args:
        seq_len: Sequence length
        force_none_probability: Probability of returning None
        
    Returns:
        Threshold value or None
    """
    if random.random() < force_none_probability:
        return None
    
    max_threshold = max(1, int(seq_len * 0.15))
    return random.randint(1, max_threshold)


# =============================================================================
# Change Position Utilities
# =============================================================================

def get_change_position_range(
    seq_len: int,
    min_ratio: float = 0.02,
    max_ratio: float = 0.95
) -> Tuple[int, int]:
    """
    Get valid range for change positions.
    
    Args:
        seq_len: Sequence length
        min_ratio: Minimum position as ratio of seq_len
        max_ratio: Maximum position as ratio of seq_len
        
    Returns:
        Tuple of (min_position, max_position)
    """
    return int(min_ratio * seq_len), int(max_ratio * seq_len)


def get_random_change_position(
    seq_len: int,
    min_ratio: float = 0.02,
    max_ratio: float = 0.95
) -> int:
    """
    Get a random valid change position.
    
    Args:
        seq_len: Sequence length
        min_ratio: Minimum position as ratio of seq_len
        max_ratio: Maximum position as ratio of seq_len
        
    Returns:
        Random position within valid range
    """
    mn, mx = get_change_position_range(seq_len, min_ratio, max_ratio)
    return random.randint(mn, mx)


# =============================================================================
# Shuffling Utilities
# =============================================================================

def shuffle_aligned_data(
    *lists: List[Any],
    return_indices: bool = True
) -> Tuple[Any, ...]:
    """
    Shuffle multiple lists in unison.
    
    Args:
        *lists: Lists to shuffle together
        return_indices: Whether to return the shuffle indices
        
    Returns:
        Tuple of shuffled lists, optionally with indices
    """
    if not lists:
        return tuple()
    
    n = len(lists[0])
    indices = np.random.permutation(n)
    shuffled = tuple([lst[i] for i in indices] for lst in lists)
    
    if return_indices:
        return shuffled + (indices,)
    return shuffled


def has_local_event_near(
    attr: Dict,
    target_pos: int,
    threshold: int = 15
) -> Optional[str]:
    """
    Check if there's a local event near the target position.
    
    Args:
        attr: Attribute dictionary
        target_pos: Target position to check
        threshold: Maximum distance to consider "near"
        
    Returns:
        Event type if found, None otherwise
    """
    local_events = attr.get('local', [])
    if not local_events:
        return None
    
    for event in local_events:
        pos = event.get('position_start', event.get('position', -999))
        if abs(pos - target_pos) <= threshold:
            e_type = event.get('type', 'change')
            if isinstance(e_type, list):
                e_type = e_type[0]
            return e_type
    
    return None


def get_local_event_positions(attr: Dict) -> List[int]:
    """
    Get all local event positions from attributes.
    
    Args:
        attr: Attribute dictionary
        
    Returns:
        List of event positions
    """
    local_events = attr.get('local', [])
    positions = []
    
    for event in local_events:
        pos = event.get('position_start', event.get('position'))
        if pos is not None:
            positions.append(pos)
    
    return positions


def has_local_event_end_near(
    attr: Dict,
    target_pos: int,
    threshold: int = 15
) -> Optional[str]:
    """
    Check if there's a local event whose *end* is near the target position.

    Args:
        attr: Attribute dictionary
        target_pos: Target position to check against event ends
        threshold: Maximum distance to consider "near"

    Returns:
        Event type if found, None otherwise
    """
    local_events = attr.get('local', [])
    for event in local_events:
        pos_end = event.get('position_end')
        if pos_end is not None and abs(pos_end - target_pos) <= threshold:
            e_type = event.get('type', 'change')
            if isinstance(e_type, list):
                e_type = e_type[0]
            return e_type
    return None


def get_local_event_end_positions(attr: Dict) -> List[int]:
    """
    Get all local event *end* positions from attributes.

    Args:
        attr: Attribute dictionary

    Returns:
        List of event end positions (only events that have position_end set)
    """
    local_events = attr.get('local', [])
    positions = []
    for event in local_events:
        pos_end = event.get('position_end')
        if pos_end is not None:
            positions.append(pos_end)
    return positions


def sanitize_attributes_for_sync(attr: Dict) -> Dict:
    """
    Sanitize attributes for synchronization analysis.
    
    Removes non-local attributes to focus on local events only.
    
    Args:
        attr: Attribute dictionary
        
    Returns:
        Sanitized attribute dictionary
    """
    sanitized = copy.deepcopy(attr)
    
    keys_to_remove = [
        'periodicity', 'trend', 'noise', 'seasonal', 'frequency', 'statistics'
    ]
    for k in keys_to_remove:
        sanitized.pop(k, None)
    
    if 'local' in sanitized:
        new_local = []
        for event in sanitized['local']:
            pos = event.get('position_start', event.get('position'))
            if pos is not None:
                e_type = event.get('type', '')
                if isinstance(e_type, list):
                    e_type = e_type[0] if e_type else ''
                entry: Dict = {'type': e_type, 'position_start': pos}
                if 'value_start' in event:
                    entry['value_start'] = event['value_start']
                pos_end = event.get('position_end')
                if pos_end is not None:
                    entry['position_end'] = pos_end
                    if 'value_end' in event:
                        entry['value_end'] = event['value_end']
                new_local.append(entry)
        sanitized['local'] = new_local
    
    return sanitized


# =============================================================================
# Trend Utilities
# =============================================================================

def apply_trend_to_timeseries(
    ts: np.ndarray,
    curve_y: np.ndarray,
    amplitude: float
) -> np.ndarray:
    """
    Apply a trend curve to an existing time series.
    
    Args:
        ts: Base time series
        curve_y: Trend curve values
        amplitude: Overall amplitude for scaling
        
    Returns:
        Time series with trend applied
    """
    curve_range = curve_y.max() - curve_y.min()
    if curve_range > 1e-3:
        scale = amplitude * random.uniform(3.0, 15.0)
        ts = ts + curve_y / curve_range * scale
    return ts


def find_point_far_from_all_events(
    all_positions: List[int],
    seq_len: int,
    threshold: int = 15,
    max_attempts: int = 200
) -> Optional[int]:
    """Find a point that is far from all given event positions."""
    for _ in range(max_attempts):
        point = random.randint(0, seq_len - 1)
        if all(abs(point - pos) >= threshold for pos in all_positions):
            return point
    return None


def find_point_at_distance_range(
    all_positions: List[int],
    seq_len: int,
    min_distance: int,
    max_distance: Optional[int] = None,
    max_attempts: int = 200,
) -> Optional[int]:
    """Find a point whose min distance from all positions is in [min_distance, max_distance].

    If max_distance is None, only checks >= min_distance (no upper bound).
    """
    for _ in range(max_attempts):
        point = random.randint(0, seq_len - 1)
        min_dist = min(abs(point - pos) for pos in all_positions) if all_positions else seq_len
        if min_dist < min_distance:
            continue
        if max_distance is not None and min_dist > max_distance:
            continue
        return point
    return None


def derive_trend_info(
    points: List[Tuple[int, float]],
    seq_len: int
) -> Dict[str, Any]:
    """
    Derive standard trend dictionary from points.
    
    Args:
        points: List of (x, y) tuples defining the trend
        seq_len: Sequence length
        
    Returns:
        Trend info dictionary
    """
    segment_text = generate_trend_prompt(points, seq_len=seq_len)

    vals = [p[1] for p in points]
    start, end = vals[0], vals[-1]

    val_range = max(vals) - min(vals)
    diff = end - start

    if val_range == 0 or abs(diff) < 0.10 * val_range:
        t_type = "keep steady"
        overall = "From the perspective of the slope, the overall trend is steady."
    elif diff > 0:
        t_type = "increase"
        overall = "From the perspective of the slope, the overall trend is increasing."
    else:
        t_type = "decrease"
        overall = "From the perspective of the slope, the overall trend is decreasing."

    detail = (
        f"{overall}"
        f" The value of time series starts from around {start:.2f}"
        f" and ends at around {end:.2f}, with an overall amplitude"
        f" of {diff:.2f}. {segment_text}"
    )

    return {
        "type": t_type,
        "start": round(start, 2),
        "end": round(end, 2),
        "amplitude": round(diff, 2),
        "detail": detail
    }


def select_metrics_from_clusters(
    cluster: Dict,
    num_positive_clusters: int = None,
    max_negative: int = 5,
    target_positive_size: Optional[int] = None,
) -> Tuple[List[List[str]], List[str], Dict]:
    """Standalone metric selector (Shape style).

    Args:
        target_positive_size: When set, create exactly one positive cluster of
            this size (for balanced GT-length generation).  Metrics are drawn
            from a single semantic cluster when possible, otherwise pooled
            across all available metrics.
    """
    metric_to_cluster = {m: c for c, ms in cluster.items() for m in ms}

    # --- Targeted path: exact cluster size for GT-length balancing ---
    if target_positive_size is not None:
        all_metrics = list(metric_to_cluster.keys())
        target = min(target_positive_size, len(all_metrics))

        if target <= 0:
            pos_clusters: List[List[str]] = []
            pos_metrics_flat: List[str] = []
        else:
            # Prefer a single semantic cluster large enough
            big_clusters = [c for c in cluster if len(cluster[c]) >= target]
            if big_clusters:
                c_name = random.choice(big_clusters)
                selected = list(np.random.choice(
                    cluster[c_name], size=target, replace=False
                ))
            else:
                # Pool across all available metrics
                selected = list(np.random.choice(
                    all_metrics, size=target, replace=False
                ))
            pos_clusters = [selected]
            pos_metrics_flat = list(selected)

        neg_cands = sorted(set(all_metrics) - set(pos_metrics_flat))
        # Ensure enough negatives for small clusters
        min_neg = min(2, len(neg_cands)) if target <= 3 else 0
        neg_count = random.randint(min_neg, min(max_negative, len(neg_cands))) if neg_cands else 0
        neg_metrics = random.sample(neg_cands, neg_count) if neg_count else []
        return pos_clusters, neg_metrics, metric_to_cluster

    # --- Original randomized path (unchanged) ---
    if num_positive_clusters is None:
        num_positive_clusters = random.randint(1, 3)

    visited_metrics: set = set()
    visited_clusters: set = set()
    pos_clusters = []
    pos_metrics_flat = []

    for _ in range(num_positive_clusters):
        # Prefer selecting from a real cluster (coherent group); fall back to random pick
        candidates_cluster = [
            c for c in cluster
            if len(set(cluster[c]) - visited_metrics) > 1 and c not in visited_clusters
        ]
        candidates_random = [m for m in metric_to_cluster if m not in visited_metrics]

        selected = []
        if random.random() > 0.5 and candidates_cluster:
            # Cluster path: pick a cluster and sample 2+ metrics from it
            c_name = random.choice(candidates_cluster)
            avail = list(set(cluster[c_name]) - visited_metrics)
            selected = list(np.random.choice(avail, size=random.randint(2, len(avail)), replace=False))
            visited_clusters.add(c_name)
        elif len(candidates_random) >= 2:
            # Random path: sample 2–5 metrics from whatever is available
            selected = list(np.random.choice(
                candidates_random, size=random.randint(2, min(len(candidates_random), 5)), replace=False
            ))

        if selected:
            visited_metrics.update(selected)
            pos_metrics_flat.extend(selected)
            pos_clusters.append(selected)

    # Everything not chosen as positive becomes a negative candidate
    neg_cands = sorted(set(metric_to_cluster) - set(pos_metrics_flat))
    neg_metrics = (
        random.sample(neg_cands, random.randint(0, min(max_negative, len(neg_cands))))
        if neg_cands else []
    )
    return pos_clusters, neg_metrics, metric_to_cluster

class BatchSampleGenerator:
    """
    High-level generator for creating batches of time series samples.
    
    This class is designed for generating positive/negative pairs for
    training ML models, with support for different generation modes.
    
    Modes:
        - "local": Generates samples based on local change features
        - "shape": Generates samples based on overall shape/trend features
    
    Example:
        >>> generator = BatchSampleGenerator(mode="local")
        >>> pos_ts, pos_attrs, anchor, _ = generator.generate_positive_timeseries(
        ...     count=5, seq_len=256
        ... )
        >>> neg_ts, neg_attrs, _, _ = generator.generate_negative_timeseries(
        ...     count=5, positive_anchor=anchor, seq_len=256
        ... )
    """
    
    def __init__(
        self,
        mode: str = "local",
        feature_config: Optional[Dict] = None,
        max_retries: int = 10000
    ):
        """
        Initialize the batch sample generator.

        Args:
            mode: Generation mode ("local" or "shape")
            feature_config: Custom feature configuration dict
            max_retries: Maximum retry attempts for generation
        """
        self.mode = mode.lower()

        if feature_config is not None:
            self.feature_config = feature_config
        elif self.mode == "local":
            self.feature_config = DEFAULT_LOCAL_FEATURE_CONFIG
        elif self.mode == "shape":
            self.feature_config = DEFAULT_SHAPE_FEATURE_CONFIG
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'local' or 'shape'.")

        self.max_retries = max_retries
    
    def generate_positive_timeseries(
        self,
        count: int,
        seq_len: int = 256,
        change_position: Optional[int] = None,
        validate_local_count: bool = False,
        threshold: int = DEFAULT_THRESHOLD
    ) -> Tuple[List[np.ndarray], List[Dict], Any, Optional[List[List[Tuple[int, float]]]]]:
        """
        Generate positive (matching) time series samples.

        Args:
            count: Number of samples to generate
            seq_len: Length of each time series
            change_position: Fixed change position (local mode only)
            validate_local_count: Whether to validate local event count
            threshold: Maximum distance from anchor for event placement (local mode)

        Returns:
            Tuple of:
                - timeseries: List of generated time series arrays
                - attributes: List of attribute dictionaries
                - anchor_data: Mode-specific anchor (int for local, points for shape)
                - points_list: List of perturbed points (shape mode only)
        """
        if self.mode == "local":
            return self._generate_positive_local(
                count, seq_len, change_position, validate_local_count, threshold
            )
        elif self.mode == "shape":
            return self._generate_positive_shape(count, seq_len, threshold)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _generate_positive_local(
        self,
        count: int,
        seq_len: int,
        change_position: Optional[int],
        validate_local_count: bool,
        threshold: int = DEFAULT_THRESHOLD
    ) -> Tuple[List[np.ndarray], List[Dict], int, None]:
        """Generate positive samples for local mode.

        Each positive sample gets 1-3 local fluctuations:
        - 1 guaranteed within threshold of the anchor position
        - 0-2 additional at random positions elsewhere

        The event is placed uniformly within [-threshold, +threshold] of the
        anchor, clamped to valid sequence bounds.
        """
        if change_position is None:
            change_position = get_random_change_position(seq_len)

        timeseries, attributes = [], []

        for _ in range(count):
            ts, attr_pool = None, None

            for _ in range(self.max_retries):
                # 1 near anchor (within threshold) + 0-2 extra at random positions
                # 0 extra: 50%, 1 extra: 25%, 2 extra: 25%
                num_extra = int(weighted_random_choice({"0": 0.5, "1": 0.25, "2": 0.25}))
                anchor_pos = int(change_position + random.randint(
                    -threshold, threshold
                ))
                anchor_pos = max(0, min(seq_len - 1, anchor_pos))
                changes = {(anchor_pos, None)}

                mn, mx = get_change_position_range(seq_len)
                for _ in range(num_extra):
                    extra_pos = random.randint(mn, mx)
                    changes.add((extra_pos, None))

                attr_pool = generate_random_attributes(
                    self.feature_config['overall_attribute'],
                    self.feature_config['change'],
                    changes.copy(),
                    seq_len
                )
                ts, attr_pool = generate_time_series(attr_pool, seq_len)

                if not validate_local_count or len(attr_pool.get('local', [])) == len(changes):
                    break

            # Inject varied noise scaled to overall_amplitude (protects local events)
            ts = inject_varied_noise(ts, attr_pool['overall_amplitude'])
            add_statistics_to_pool(attr_pool, ts, seq_len)

            timeseries.append(ts)
            attributes.append(attr_pool)

        return timeseries, attributes, change_position, None

    def _generate_positive_shape(
        self,
        count: int,
        seq_len: int,
        threshold: int = 15,
    ) -> Tuple[List[np.ndarray], List[Dict], List[Tuple[int, float]], List[List[Tuple[int, float]]]]:
        """Generate positive samples for shape mode."""
        timeseries, attributes, all_perturbed = [], [], []
        
        # Generate base points for all positive samples
        base_points, _ = generate_random_points(seq_len)
        
        for _ in range(count):
            attr_pool = generate_random_attributes(
                self.feature_config['overall_attribute'],
                self.feature_config['change'],
                {},
                seq_len
            )
            ts, attr_pool = generate_time_series(attr_pool, seq_len)
            
            # Perturb points slightly
            perturbed = self._perturb_points(base_points, seq_len, threshold)
            all_perturbed.append(perturbed)
            
            # Apply trend curve
            _, curve_y, _ = generate_trend_curve(seq_len, perturbed)
            ts = apply_trend_to_timeseries(ts, curve_y, attr_pool['overall_amplitude'])

            # Inject varied noise to make data realistic
            ts = inject_varied_noise(ts, attr_pool['overall_amplitude'])

            timeseries.append(ts)

            # Update trend info
            attr_pool['trend'] = derive_trend_info(perturbed, seq_len)
            attr_pool['trend_list'] = generate_trend_list(perturbed, seq_len)
            add_statistics_to_pool(attr_pool, ts, seq_len)
            attributes.append(attr_pool)

        return timeseries, attributes, base_points, all_perturbed
    
    def _perturb_points(
        self,
        base_points: List[Tuple[int, float]],
        seq_len: int,
        threshold: int = 15,
        min_gap: int = 5
    ) -> List[Tuple[int, float]]:
        """Apply small perturbations to base points."""
        y_vals = [p[1] for p in base_points]
        y_range = max(y_vals) - min(y_vals) if y_vals else 1.0

        num_pts = len(base_points)
        perturbed = [None] * num_pts

        for i in range(num_pts):
            orig_x, orig_y = base_points[i]

            # --- Y JITTER (Independent) ---
            new_y = orig_y + random.uniform(-0.05, 0.05) * y_range

            # --- X JITTER (Neighbor-Aware) ---
            if i == 0:
                # Anchor start
                new_x = 0
            elif i == num_pts - 1:
                # Anchor end
                new_x = seq_len - 1
            else:
                # Left: must be > previous perturbed x + min_gap
                left_limit = perturbed[i-1][0] + min_gap
                # Right: must leave room for ALL remaining points (each needs min_gap)
                # e.g. if there are 2 more points after this one, we need 2*min_gap of room
                remaining = num_pts - 1 - i  # points still to be placed after this one
                right_limit = min(
                    base_points[i+1][0] - min_gap,
                    seq_len - 1 - remaining * min_gap
                )

                # The target range based on the threshold
                target_min = orig_x - threshold
                target_max = orig_x + threshold

                # Intersect the target range with the safe limit
                safe_min = max(target_min, left_limit)
                safe_max = min(target_max, right_limit)

                if safe_min < safe_max:
                    new_x = random.randint(safe_min, safe_max)
                else:
                    # Fallback: clamp to valid range to preserve strict ordering
                    new_x = max(left_limit, min(safe_min, right_limit))

            perturbed[i] = (new_x, new_y)

        return perturbed
    
    def generate_negative_timeseries(
        self,
        count: int,
        positive_anchor: Any,
        seq_len: int = 256,
        threshold: int = DEFAULT_THRESHOLD,
        validate_local_count: bool = False
    ) -> Tuple[List[np.ndarray], List[Dict], Optional[List], Optional[List[List[Tuple[int, float]]]]]:
        """
        Generate negative (non-matching) time series samples.

        Args:
            count: Number of samples to generate
            positive_anchor: Anchor from positive generation (position or points)
            seq_len: Length of each time series
            threshold: Events must be placed farther than this from positive anchors
            validate_local_count: Whether to validate local event count

        Returns:
            Tuple of:
                - timeseries: List of generated time series arrays
                - attributes: List of attribute dictionaries
                - diff_info: Mode-specific difference info (shape mode only)
                - points_list: List of points (shape mode only)
        """
        if self.mode == "local":
            ts, attrs = self._generate_negative_local(
                count, positive_anchor, seq_len, threshold, validate_local_count
            )
            return ts, attrs, None, None
        elif self.mode == "shape":
            return self._generate_negative_shape(count, positive_anchor, seq_len, threshold)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _generate_negative_local(
        self,
        count: int,
        positive_positions: Union[int, List[int]],
        seq_len: int,
        threshold: int,
        validate_local_count: bool
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """Generate negative samples for local mode.

        Events are placed at distance > threshold from all positive anchor
        positions, guaranteeing that verification against those anchors will
        correctly report no nearby event.
        """
        # Ensure positive_positions is a list
        if isinstance(positive_positions, int):
            positive_positions = [positive_positions]

        timeseries, attributes = [], []
        negative_positions = set()

        for _ in range(count):
            changes = set()

            # Sometimes add a change at a different position
            if random.random() < 0.2:
                mn, mx = get_change_position_range(seq_len)
                for _ in range(self.max_retries):
                    cand = random.randint(mn, mx)
                    all_positions = list(positive_positions) + list(negative_positions)
                    if all(abs(cand - p) > threshold for p in all_positions):
                        changes = {(cand, None)}
                        negative_positions.add(cand)
                        break
            
            # Generate time series
            ts, attr_pool = None, None
            retry_count = self.max_retries if validate_local_count else 1
            
            for _ in range(retry_count):
                attr_pool = generate_random_attributes(
                    self.feature_config['overall_attribute'],
                    self.feature_config['change'],
                    changes.copy(),
                    seq_len
                )
                ts, attr_pool = generate_time_series(attr_pool, seq_len)
                
                if not validate_local_count or len(attr_pool.get('local', [])) == len(changes):
                    break

            # Inject varied noise scaled to overall_amplitude (protects local events)
            ts = inject_varied_noise(ts, attr_pool['overall_amplitude'])
            add_statistics_to_pool(attr_pool, ts, seq_len)

            timeseries.append(ts)
            attributes.append(attr_pool)

        return timeseries, attributes

    def _generate_negative_shape(
        self,
        count: int,
        positive_points: List[Tuple[int, float]],
        seq_len: int,
        threshold: int = 15,  # <--- Add this
        min_gap: int = 5      # <--- Add this for safety
    ) -> Tuple[List[np.ndarray], List[Dict], List[Optional[Tuple[int, float]]], List[List[Tuple[int, float]]]]:
        """Generate negative samples for shape mode."""
        timeseries, attributes, diff_types, res_points = [], [], [], []
        
        for _ in range(count):
            attr_pool = generate_random_attributes(
                self.feature_config['overall_attribute'],
                self.feature_config['change'],
                {},
                seq_len
            )
            ts, attr_pool = generate_time_series(attr_pool, seq_len)
            
            # Decide if this is a "hard" negative (similar but different), cant create the thing if we have too few points, otherwise we might end up with the same shape after modification.
            is_hard = random.random() <= 0.6 and len(positive_points) > 3
            
            if not is_hard:
                # Completely different points
                points, _ = generate_random_points(seq_len)
                diff_info = None
            else:
                # "Hard Negative": Modify ONE point from positive
                # Still, it could generate the positive shape, and yes we have handled that thing.!
                points = copy.deepcopy(positive_points)
                idx = random.choice(range(len(points)))
                
                # --- 1. Modify Y (Make the shape value different) ---
                y_range = max(p[1] for p in points) - min(p[1] for p in points)
                diff = random.choice([-1, 1]) * random.uniform(0.5, 1.0) * y_range
                new_y = points[idx][1] + diff
                
                # --- 2. Modify X (Smart Jitter) ---
                orig_x = points[idx][0]
                
                if idx == 0:
                    new_x = 0
                elif idx == len(points) - 1:
                    new_x = seq_len - 1
                else:
                    # Look at neighbors to define the Safe Zone
                    prev_x = points[idx-1][0]
                    next_x = points[idx+1][0]
                    
                    # Calculate safe bounds
                    safe_min = prev_x + min_gap
                    safe_max = next_x - min_gap
                    
                    # Calculate target jitter bounds based on threshold
                    target_min = orig_x - threshold
                    target_max = orig_x + threshold
                    
                    # Intersect bounds
                    final_min = max(safe_min, target_min)
                    final_max = min(safe_max, target_max)
                    
                    if final_min < final_max:
                        new_x = random.randint(final_min, final_max)
                    else:
                        new_x = final_min # Fallback if squeezed tight

                points[idx] = (new_x, new_y)
                diff_info = (idx, float(diff))
            
            res_points.append(points)
            diff_types.append(diff_info)
            
            # Apply trend curve
            _, curve_y, _ = generate_trend_curve(seq_len, points)
            ts = apply_trend_to_timeseries(ts, curve_y, attr_pool['overall_amplitude'])

            # Inject varied noise to make data realistic
            ts = inject_varied_noise(ts, attr_pool['overall_amplitude'])

            timeseries.append(ts)

            # Update trend info
            attr_pool['trend'] = derive_trend_info(points, seq_len)
            attr_pool['trend_list'] = generate_trend_list(points, seq_len)
            add_statistics_to_pool(attr_pool, ts, seq_len)
            attributes.append(attr_pool)

        return timeseries, attributes, diff_types, res_points

    def generate_anti_trend_timeseries(
        self,
        base_points: List[Tuple[int, float]],
        count: int,
        seq_len: int = 256,
        threshold: int = DEFAULT_THRESHOLD
    ) -> Tuple[List[np.ndarray], List[Dict], List[List[Tuple[int, float]]]]:
        """Generate time series with opposite trend to base_points.

        Only supported in shape mode. Returns empty lists if the base
        trend is all-steady (no meaningful anti-trend possible).

        Returns:
            (timeseries, attributes, perturbed_points_list)
        """
        if self.mode != "shape":
            raise ValueError("Anti-trend generation only supported in shape mode.")
        return self._generate_anti_trend_shape(base_points, count, seq_len, threshold)

    def _generate_anti_trend_shape(
        self,
        base_points: List[Tuple[int, float]],
        count: int,
        seq_len: int,
        threshold: int = 15,
    ) -> Tuple[List[np.ndarray], List[Dict], List[List[Tuple[int, float]]]]:
        """Generate anti-trend TS by negating Y values of base points.

        Flips increase<->decrease; steady stays steady.
        Returns empty lists if base trend is all-steady.
        """
        base_trend_list = generate_trend_list(base_points, seq_len)
        if all(seg[0] == 'keep steady' for seg in base_trend_list):
            return [], [], []

        # Negate Y to flip trends
        anti_base = [(x, -y) for x, y in base_points]

        timeseries, attributes, all_perturbed = [], [], []
        for _ in range(count):
            attr_pool = generate_random_attributes(
                self.feature_config['overall_attribute'],
                self.feature_config['change'],
                {},
                seq_len
            )
            ts, attr_pool = generate_time_series(attr_pool, seq_len)

            # Perturb anti-base points (same perturbation as positive)
            perturbed = self._perturb_points(anti_base, seq_len, threshold)
            all_perturbed.append(perturbed)

            # Apply trend curve
            _, curve_y, _ = generate_trend_curve(seq_len, perturbed)
            ts = apply_trend_to_timeseries(ts, curve_y, attr_pool['overall_amplitude'])

            # Inject varied noise to make data realistic
            ts = inject_varied_noise(ts, attr_pool['overall_amplitude'])

            timeseries.append(ts)

            # Derive trend info from the perturbed anti-trend points
            attr_pool['trend'] = derive_trend_info(perturbed, seq_len)
            attr_pool['trend_list'] = generate_trend_list(perturbed, seq_len)
            add_statistics_to_pool(attr_pool, ts, seq_len)
            attributes.append(attr_pool)

        return timeseries, attributes, all_perturbed

    def generate_similar_timeseries(
        self,
        base_attributes: Dict,
        count: int,
        seq_len: int = 256
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Generate time series similar to a base attribute set.
        
        Args:
            base_attributes: Base attribute dictionary to use as template
            count: Number of samples to generate
            seq_len: Length of each time series
            
        Returns:
            Tuple of (timeseries_list, attributes_list)
        """
        timeseries, attributes = [], []
        
        for _ in range(count):
            # Deep copy to avoid mutations
            clean = deep_copy_dict(base_attributes)
            ts, attr = generate_time_series(clean, seq_len)

            # Inject varied noise scaled to overall_amplitude (protects local events)
            ts = inject_varied_noise(ts, attr['overall_amplitude'])
            add_statistics_to_pool(attr, ts, seq_len)

            timeseries.append(ts)
            attributes.append(attr)

        return timeseries, attributes
