"""Artifact-local provenance claims never imply authorship verification."""
from datetime import datetime,timezone
from ..common.metadata import require_claim
from ..common.helpers import json_safe,digest,canonical_json
from ..common.ids import require_id
from ..common.errors import ValidationError

def make_observation(subject,predicate,*,evidence,confidence=1.0,agent=""):
    require_id(subject)
    if not isinstance(predicate,str) or not predicate or len(predicate)>255:
        raise ValidationError("invalid spatial claim predicate")
    if not isinstance(agent,str) or len(agent)>255:
        raise ValidationError("invalid agent label")
    return require_claim({"subject":subject,"predicate":predicate,"evidence":evidence,
                          "confidence":confidence,"agent":agent,
                          "timestamp":datetime.now(timezone.utc).isoformat()})

def claim_digest(claim):
    return digest(canonical_json(require_claim(claim)))
