import math
import random
import numpy as np
from scipy.interpolate import PchipInterpolator


def generate_random_points(seq_len, y_range=(-1, 1), bezier_prob=0.99):
    """
    Generates random key points for the curve based on the sequence length.

    Parameters:
        seq_len (int): The total number of points on the x-axis.
        y_range (tuple): The (min, max) range for y-values.
        bezier_prob (float): Probability of selecting 'Bezier' curve type.

    Returns:
        points (list of tuples): The list of generated key points as (x, y) tuples.
        curve_type (str): The type of curve used ('Bezier' or 'Straight Line').
    """
    min_distance = math.ceil(seq_len / 8)
    num_turning_points = random.randint(0, 3)
    total_key_points = 2 + num_turning_points
    total_min_distance = (total_key_points - 1) * min_distance
    total_distance = seq_len - 1
    extra_distance = total_distance - total_min_distance

    while extra_distance < 0 and num_turning_points > 0:
        num_turning_points -= 1
        total_key_points = 2 + num_turning_points
        total_min_distance = (total_key_points - 1) * min_distance
        extra_distance = total_distance - total_min_distance
    
    if extra_distance < 0:
        raise ValueError("seq_len is too small")
    
    gaps = [min_distance] * (total_key_points - 1)
    for _ in range(extra_distance):
        idx = random.randint(0, total_key_points - 2)
        gaps[idx] += 1
    
    key_x = [0]
    for gap in gaps:
        key_x.append(key_x[-1] + gap)
    
    # Use configurable range
    y_positions = np.random.uniform(y_range[0], y_range[1], total_key_points)
    points = list(zip(key_x, y_positions))

    # Use configurable probability
    if random.random() < bezier_prob:
        curve_type = "Bezier"
    else:
        curve_type = "Straight Line"
    
    return points, curve_type

def generate_trend_curve(seq_len, points, bezier_prob=0.99):
    """
    Generates the curve based on the key points.
    
    Parameters:
        seq_len (int): The total number of points on the x-axis.
        points (list of tuples): The list of generated key points as (x, y) tuples.
        bezier_prob (float): Probability of selecting 'Bezier' curve type.
    """
    # Extract x and y from points
    key_x = [point[0] for point in points]
    key_y = [point[1] for point in points]
    
    # Decide whether to use Bezier curves or straight lines
    curve_x = np.arange(seq_len)
    
    # Use configurable probability
    if random.random() < bezier_prob:
        curve_type = "Bezier"
        interpolator = PchipInterpolator(key_x, key_y)
        curve_y = interpolator(curve_x)
    else:
        curve_type = "Straight Line"
        curve_y = np.interp(np.arange(seq_len), key_x, key_y)
    
    return curve_x, curve_y, curve_type

def _analyze_trend_segments(points, threshold_ratio):
    """
    Helper function to calculate merged trend segments.
    Returns: list of (trend_type, start_index, end_index)
    trend_type is normalized to: 'increase', 'decrease', 'stable'
    """
    if not points or len(points) < 2:
        return []

    y_values = [y for _, y in points]
    curve_range = max(y_values) - min(y_values)

    # Handle flat data explicitly
    if curve_range == 0:
        return [("stable", 0, len(points) - 1)]

    trends = []
    limit = threshold_ratio * curve_range

    for i in range(len(points) - 1):
        delta_y = y_values[i+1] - y_values[i]

        if delta_y > limit:
            trends.append("increase")
        elif delta_y < -limit:
            trends.append("decrease")
        else:
            trends.append("stable")

    if not trends:
        return []

    merged_trends = []
    current_trend = trends[0]
    start_idx = 0

    for i in range(1, len(trends)):
        if trends[i] != current_trend:
            merged_trends.append((current_trend, start_idx, i))
            current_trend = trends[i]
            start_idx = i
            
    # Append the last trend
    merged_trends.append((current_trend, start_idx, len(trends)))
    
    return merged_trends

def generate_trend_prompt(points, seq_len=None, threshold=0.05):
    # This function expects 'points' to be a list of (x, y) tuples.
    merged_trends = _analyze_trend_segments(points, threshold)

    if not merged_trends:
        return "Insufficient points to determine trends."

    prompt_segments = []
    vocab_map = {
        "increase": ("increasing", "an increasing trend"),
        "decrease": ("decreasing", "a decreasing trend"),
        "stable": ("stable", "a stable trend")
    }

    for i, (trend_key, start, end) in enumerate(merged_trends):
        point_start = points[start]
        point_end = points[end]

        # 0-indexed display
        start_x = int(point_start[0])
        end_x = int(point_end[0])

        if i == 0: start_x = 0
        if i == len(merged_trends) - 1 and seq_len is not None:
            end_x = seq_len - 1
        elif i == len(merged_trends) - 1:
            end_x = int(points[-1][0])

        _, article = vocab_map[trend_key]
        if end - start > 1:
            variation_note = "with some variation in slope"
        else:
            variation_note = ""

        sentence = f"From point {start_x} to point {end_x}, there is {article}{' ' + variation_note if variation_note else ''}."
        prompt_segments.append(sentence)

    return " ".join(prompt_segments)

def generate_trend_list(points, seq_len, threshold=0.05):
    """
    Generates a list describing the trend between each pair of adjacent points.
    
    Returns:
        trend_list: [(increase/decrease/keep steady, start_point_x, end_point_x)]
    """
    merged_trends = _analyze_trend_segments(points, threshold)

    if not merged_trends:
        return []

    final_list = []
    
    # Mapping normalized trends to list-specific vocabulary
    vocab_map = {
        "increase": "increase",
        "decrease": "decrease",
        "stable": "keep steady"
    }

    for i, (trend_key, start_idx, end_idx) in enumerate(merged_trends):
        output_trend = vocab_map[trend_key]
        
        start_x = int(points[start_idx][0])
        end_x = int(points[end_idx][0])

        final_list.append((output_trend, start_x, end_x))

    return final_list