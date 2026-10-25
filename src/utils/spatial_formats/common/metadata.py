"""Validated metadata and attributable claims; none are executable."""
from .errors import ValidationError
from .helpers import json_safe, finite
from .ids import require_id

def require_metadata(value):
    if not isinstance(value, dict):
        raise ValidationError("metadata must be a JSON object")
    safe = json_safe(value)
    for key in safe:
        if ":" not in key and not key.startswith("slai."):
            raise ValidationError("metadata keys must be namespaced")
    return safe

def require_claim(claim):
    if not isinstance(claim, dict) or not isinstance(claim.get("predicate"), str):
        raise ValidationError("claim requires a predicate")
    subject = require_id(claim.get("subject"))
    evidence = claim.get("evidence")
    if evidence is not None and (not isinstance(evidence, str) or not evidence):
        raise ValidationError("claim evidence must be a nonempty source identifier")
    confidence = finite(claim.get("confidence", 1.0), "confidence", limit=1)
    if confidence < 0:
        raise ValidationError("confidence must be between 0 and 1")
    return json_safe({**claim, "subject":subject, "confidence":confidence})
