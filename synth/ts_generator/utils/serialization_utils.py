import json
import copy
import numpy as np
from typing import Any, Dict

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def deep_copy_dict(d: Dict) -> Dict:
    return json.loads(json.dumps(d, cls=NumpyEncoder))

def attribute_pool_to_json(
    attribute_pool: Dict[str, Any], 
    clean: bool = True,
    round_decimals: int = 2
) -> str:
    pool_copy = copy.deepcopy(attribute_pool)
    
    # Round amplitudes in local
    if 'local' in pool_copy and isinstance(pool_copy['local'], list):
        for local_attr in pool_copy['local']:
            if 'amplitude' in local_attr:
                local_attr['amplitude'] = round(local_attr['amplitude'], round_decimals)
    
    # Clean internal keys if requested
    if clean:
        keys_to_remove = ['overall_amplitude', 'overall_bias', 'statistics', 'trend_list']
        for k in keys_to_remove:
            pool_copy.pop(k, None)
    
    return json.dumps(pool_copy, cls=NumpyEncoder, ensure_ascii=False)
