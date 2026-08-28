import random
import numpy as np
from enum import Enum
from typing import Dict, List, Optional

def normalize_probabilities(probabilities: Dict[str, float]) -> np.ndarray:
    values = np.array(list(probabilities.values()), dtype=float)
    total = values.sum()
    if total == 0:
        raise ValueError("Probabilities sum to zero")
    return values / total

def weighted_random_choice(options: Dict[str, float]) -> str:
    keys = list(options.keys())
    probs = normalize_probabilities(options)
    return str(np.random.choice(keys, p=probs))

def weighted_random_choices(options: Dict[str, float], size: int) -> List[str]:
    keys = list(options.keys())
    probs = normalize_probabilities(options)
    return list(np.random.choice(keys, size=size, p=probs))


class QAType(str, Enum):
    DESCRIPTION = "description"
    YES_NO = "yes_no"
    CORRELATION = "correlation"
    CLUSTERING = "clustering"
    ANTICORRELATION = "anticorrelation"
    ANTICLUSTERING = "anticlustering"
    SEGMENT_MASK = "segment_mask"
    # Semantic split of the former TAXONOMY lump — each maps to a distinct
    # thesis role: SFT_ATOMIC teaches primitives (TS encoder alignment),
    # SFT_BRIDGE teaches decomposition (meta-skill, force-SFT), RL_COMPOSITION
    # provides compositional reasoning signal for the RL phase.
    SFT_ATOMIC = "sft_atomic"
    SFT_BRIDGE = "sft_bridge"
    RL_COMPOSITION = "rl_composition"


DEFAULT_QA_TYPE_WEIGHTS: Dict[str, float] = {
    QAType.DESCRIPTION: 0.06,
    QAType.YES_NO: 0.07,
    QAType.CORRELATION: 0.06,
    QAType.ANTICORRELATION: 0.06,
    QAType.CLUSTERING: 0.05,
    QAType.ANTICLUSTERING: 0.05,
    QAType.SEGMENT_MASK: 0.18,
    QAType.SFT_ATOMIC: 0.22,
    QAType.SFT_BRIDGE: 0.09,
    QAType.RL_COMPOSITION: 0.16,
}


def select_weighted_qa_index(
    qa_types: List[str],
    weights: Dict[str, float],
    eval_types: Optional[List[str]] = None,
    eval_type_weights: Optional[Dict[str, float]] = None,
) -> int:
    """Select a QA index using two-level category-weighted sampling.

    Level 1: Pick a qa_type proportional to its weight (among types with ≥1 QA).
    Level 2: Within the chosen qa_type, group entries by eval_type, pick an
             eval_type proportional to eval_type_weights (default 1.0 for unlisted).
    Level 3: Uniform random within the chosen eval_type group.

    If eval_types is None, Level 2 is skipped (original single-level behavior).
    """
    # Group indices by qa_type
    type_to_indices: Dict[str, List[int]] = {}
    for i, t in enumerate(qa_types):
        type_to_indices.setdefault(t, []).append(i)

    # Filter to available types and normalize weights.
    # Default 1.0 (not 0.0) for missing keys — fail-open, matches level-2
    # eval_type_weights. Startup validation in Config.from_yaml catches
    # intentional omissions; this default protects against runtime-injected
    # weights dicts that don't cover every QAType (e.g. `replace(config,
    # qa_type_weights={'clustering': 1.0})` for targeted generation).
    available = list(type_to_indices.keys())
    w = [weights.get(t, 1.0) for t in available]
    total = sum(w)
    if total == 0:
        return random.randint(0, len(qa_types) - 1)  # fallback

    # Level 1: Pick qa_type
    chosen_type = random.choices(available, weights=w, k=1)[0]
    candidates = type_to_indices[chosen_type]

    # Level 2: Sub-select by eval_type if provided
    if eval_types is not None and eval_type_weights is not None and len(candidates) > 1:
        et_to_indices: Dict[str, List[int]] = {}
        for idx in candidates:
            et = eval_types[idx] if idx < len(eval_types) else "__unknown__"
            et_to_indices.setdefault(et, []).append(idx)

        if len(et_to_indices) > 1:
            et_keys = list(et_to_indices.keys())
            et_w = [eval_type_weights.get(k, 1.0) for k in et_keys]
            et_total = sum(et_w)
            if et_total > 0:
                chosen_et = random.choices(et_keys, weights=et_w, k=1)[0]
                candidates = et_to_indices[chosen_et]

    return random.choice(candidates)
