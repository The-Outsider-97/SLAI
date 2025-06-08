
import json
import numpy as np

from typing import Any

class NumpyEncoder(json.JSONEncoder):
    """
    JSON Encoder that serializes NumPy data types into JSON-compatible formats.
    Supports integers, floats, arrays, booleans, complex numbers, and generic scalars.
    """
    
    def default(self, obj: Any) -> Any:
        # Handle NumPy scalar types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.complexfloating):
            return {"__type__": "complex", "real": obj.real, "imag": obj.imag}
        elif isinstance(obj, np.generic):  # fallback for any other scalar
            return obj.item()

        # Handle NumPy arrays
        elif isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": obj.tolist()
            }

        # Handle callable or method types gracefully
        elif callable(obj):
            return {"__type__": "callable", "name": getattr(obj, '__name__', str(obj))}

        # Handle objects with __dict__ (custom classes)
        elif hasattr(obj, '__dict__'):
            try:
                return obj.__dict__
            except Exception as e:
                return {"__type__": "unserializable", "error": str(e), "repr": repr(obj)}

        # As a last resort, convert to string
        try:
            return str(obj)
        except Exception as e:
            return {"__type__": "unserializable", "error": str(e), "repr": repr(obj)}
