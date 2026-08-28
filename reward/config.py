import math
from typing import Dict, Tuple


class RewardConfig:
    # Numeric sensitivity (lower k = faster decay = more sensitive to error)
    K_POSITION  = 0.05
    K_AMPLITUDE = 0.10
    K_PERIOD    = 0.10
    K_LENGTH    = 0.05
    K_VALUE     = 0.10   # for key_point values

    # Verdict/cluster correctness | intermediate verification | attribute evidence
    # W_VERDICT is the direct verdict weight used by mts_verdict, mts_set, yes_no,
    # trend_dominance (0.55).  Compound types use 0.60.  Coherency (0.05) is separate.
    W_VERDICT      = 0.55
    W_VERIFICATION = 0.15   # per-metric vs PASS/FAIL in verification section
    W_EVIDENCE     = 0.25

    # Partial credit when the structural element is absent in pred
    PARTIAL_NO_VERDICT      = 0.20
    PARTIAL_NO_CLUSTER_LINE = 0.10
    PARTIAL_NO_PASS_FAIL    = 0.20

    # Hallucinated-metric penalty per extra metric in compound description
    EXTRA_METRIC_PENALTY = 0.10

    PENALTY_PARSE_ERROR = 0.0

    # Baby-step anti-cliff constants (RL gradient smoothing)
    # Reduced from initial values now that model has strong SFT baseline —
    # sharper gradient between wrong and right without full cliff.
    LOCAL_DETECTION_FLOOR = 0.08
    LOCAL_COUNT_BONUS = 0.05
    LOCAL_HALLUCINATION_PER_EVENT = 0.20
    LOCAL_UNNECESSARY_PER_EVENT = 0.05
    SEGMENT_DETECTION_FLOOR = 0.05
    SEGMENT_COUNT_BONUS = 0.05
    VERDICT_WRONG_FLOOR = 0.05

    # Cap on total reward when verdict is wrong.  Prevents majority-class
    # exploitation: even with perfect structure, a wrong verdict can earn
    # at most this value.  Without this cap, the model learns that
    # always-guessing the majority class yields higher expected reward
    # than actually trying on hard samples.
    WRONG_VERDICT_CAP = 0.10

    # Coherency weight — internal reasoning consistency (taken from verdict budget)
    W_COHERENCY = 0.05


TYPE_SIMILARITY: Dict[Tuple[str, str], float] = {
    # Opposite-direction pairs: gradient for direction learning
    ('upward spike', 'downward spike'): 0.10,
    ('sudden increase', 'sudden decrease'): 0.10,
    ('upward spike', 'sudden increase'): 0.6,
    ('downward spike', 'sudden decrease'): 0.6,
    ('upward spike', 'continuous upward spike'): 0.8,
    ('downward spike', 'continuous downward spike'): 0.8,
    ('upward convex', 'wide upward spike'): 0.75,
    ('downward convex', 'wide downward spike'): 0.75,
    ('upward spike', 'rapid rise followed by slow decline'): 0.6,
    ('upward spike', 'slow rise followed by rapid decline'): 0.6,
    ('downward spike', 'rapid decline followed by slow rise'): 0.6,
    ('downward spike', 'slow decline followed by rapid rise'): 0.6,
    ('upward spike', 'decrease after upward spike'): 0.5,
    ('upward spike', 'increase after upward spike'): 0.5,
    ('downward spike', 'increase after downward spike'): 0.5,
    ('downward spike', 'decrease after downward spike'): 0.5,
    ('linear increase', 'increase'): 0.9,
    ('linear decrease', 'decrease'): 0.9,
    ('no trend', 'steady'): 0.9,
    ('no trend', 'flat'): 0.9,
    # Noise types
    ('smooth', 'smooth'): 1.0,
    # Seasonal segment types
    ('none', 'no periodic fluctuation'): 0.9,
    # --- Convex / spike shape cross-family (orphan GT types) ---
    ('downward spike', 'downward convex'): 0.5,
    ('upward spike', 'upward convex'): 0.5,
    ('downward convex', 'downward spike'): 0.5,
    ('upward convex', 'upward spike'): 0.5,
    # --- Shake: rapid oscillation, partial credit from any spike-like type ---
    ('shake', 'downward spike'): 0.3,
    ('shake', 'upward spike'): 0.3,
    ('shake', 'sudden decrease'): 0.3,
    ('shake', 'sudden increase'): 0.3,
    # --- Trend segment synonyms ---
    ('steady', 'keep steady'): 0.95,
    ('no trend', 'keep steady'): 0.9,
    ('steady', 'flat'): 0.9,
    ('keep steady', 'flat'): 0.9,
    ('steady', 'no change'): 0.9,
    # --- Seasonal subtype: correct detection but missing sin/square qualifier ---
    ('periodic fluctuation', 'sin periodic fluctuation'): 0.85,
    ('periodic fluctuation', 'square periodic fluctuation'): 0.85,
}


_LOCAL_EVENT_KEYWORDS = {'spike', 'convex', 'sudden', 'shake'}


def get_type_similarity(t1: str, t2: str) -> float:
    t1 = str(t1).lower().strip().replace('_', ' ')
    t2 = str(t2).lower().strip().replace('_', ' ')
    if t1 == t2:
        return 1.0
    if (t1, t2) in TYPE_SIMILARITY:
        return TYPE_SIMILARITY[(t1, t2)]
    if (t2, t1) in TYPE_SIMILARITY:
        return TYPE_SIMILARITY[(t2, t1)]
    if t1 and t2 and len(t1) >= 4 and len(t2) >= 4 and (t1 in t2 or t2 in t1):
        return 0.4
    # Category-based keyword fallback: shared event keywords → partial credit
    if t1 and t2:
        t1_words = set(t1.split())
        t2_words = set(t2.split())
        if t1_words & t2_words & _LOCAL_EVENT_KEYWORDS:
            return 0.15
    return 0.0


def score_numeric_proximity(pred: float, gt: float, scale: float, k: float) -> float:
    scale = max(abs(scale), 1e-6)
    return math.exp(-abs(pred - gt) / (k * scale))


def score_verdict_simple(
    pred_raw, gt_raw, *,
    wrong_floor: float = RewardConfig.VERDICT_WRONG_FLOOR,
    missing_floor: float = RewardConfig.PARTIAL_NO_VERDICT,
) -> float:
    """Standard verdict scoring: match=1.0, missing pred=missing_floor, wrong=wrong_floor.
    For unparseable GT (None), returns 1.0 (can't penalize what we can't parse).
    Works with any comparable types (strings, booleans, etc.)."""
    if gt_raw is None:
        return 1.0
    if pred_raw is None:
        return missing_floor
    if pred_raw == gt_raw:
        return 1.0
    return wrong_floor
