"""Strict, deterministic and bounded JSON primitives."""
import hashlib
import json
import math
from .errors import ValidationError, ResourceLimitError

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_STRING = 1_000_000
MAX_NODES = 300_000
MAX_DEPTH = 64

def finite(value, label="number", *, limit=1e12) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > limit:
        raise ValidationError(label + " must be finite and within bounds")
    return float(value)

def vector(value, dimension: int, label="vector") -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != dimension:
        raise ValidationError(label + " must have " + str(dimension) + " components")
    return tuple(finite(v, label) for v in value)

def json_safe(value, *, max_depth=MAX_DEPTH, max_nodes=MAX_NODES):
    remaining = [max_nodes]
    def walk(item, depth):
        remaining[0] -= 1
        if remaining[0] < 0 or depth > max_depth:
            raise ResourceLimitError("JSON structural budget exceeded")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, str):
            if len(item) > MAX_STRING:
                raise ResourceLimitError("string too long")
            return item
        if isinstance(item, (int, float)):
            finite(item, "JSON numeric value", limit=1e18)
            return item
        if isinstance(item, (list, tuple)):
            return [walk(x, depth + 1) for x in item]
        if isinstance(item, dict):
            if any(not isinstance(k, str) or len(k) > MAX_STRING for k in item):
                raise ValidationError("JSON objects require bounded string keys")
            return {k: walk(v, depth + 1) for k, v in item.items()}
        raise ValidationError("unsupported JSON value: " + type(item).__name__)
    return walk(value, 0)

def canonical_json(value) -> bytes:
    safe = json_safe(value)
    raw = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(raw) > MAX_JSON_BYTES:
        raise ResourceLimitError("JSON member too large")
    return raw

def parse_json(raw: bytes):
    if len(raw) > MAX_JSON_BYTES:
        raise ResourceLimitError("JSON member too large")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValidationError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid_number(token):
        raise ValidationError("invalid JSON numeric token: " + token)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid_number)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError("malformed UTF-8 JSON") from exc
    return json_safe(value)

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
