"""Utility functions for QA generation."""
from typing import Any, Dict
from pathlib import Path
import json
import numpy as np
import copy


CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / 'config' / 'metric_set.json'

_metric_to_attributes = None

def _load_metric_attributes():
    global _metric_to_attributes
    if _metric_to_attributes is None:
        with open(CONFIG_PATH) as f:
            control_attribute_data = json.load(f)
        _metric_to_attributes = {}
        for category in control_attribute_data:
            for k, v in category['attributes'].items():
                _metric_to_attributes[k] = v
    return _metric_to_attributes

def metric_to_controlled_attributes(metric: str):
    return _load_metric_attributes().get(metric, None)

def replace_prompts_in_obj(obj: Any, replacements: Dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: replace_prompts_in_obj(v, replacements) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_prompts_in_obj(item, replacements) for item in obj]
    elif isinstance(obj, str):
        for k, v in replacements.items():
            obj = obj.replace(k, v)
        return obj
    return obj

def no_encoding(timeseries: np.ndarray):
    return np.array(timeseries), "<ts>", {}

def timeseries_encoding(timeseries: np.ndarray, method: str):
    if method == 'no':
        return no_encoding(timeseries)
    else:
        raise NotImplementedError(f"Timeseries encoding method: {method} not implemented!")

def timeseries_to_list(timeseries, digits: int=6, cp=True):
    if cp:
        result = copy.deepcopy(timeseries)
    else:
        result = timeseries
    
    if isinstance(result, np.ndarray):
        result = result.tolist()
    
    if isinstance(result, list):
        if len(result) > 0 and isinstance(result[0], (float, int, np.float64, np.float32)):
            for i in range(len(result)):
                result[i] = round(float(result[i]), digits)
        else:
            for i in range(len(result)):
                result[i] = timeseries_to_list(result[i], digits, cp=False)
                
    return result