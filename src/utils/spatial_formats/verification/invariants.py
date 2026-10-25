"""Common independent document invariants; no agent-runtime imports."""
from ..common.errors import ValidationError
from ..common.helpers import json_safe
from ..common.ids import require_id, assert_unique
from ..common.units import require_unit

def validate_document(manifest: dict, document: dict) -> None:
    from ..container.reader import _manifest
    kind = _manifest(manifest)
    if not isinstance(document, dict):
        raise ValidationError("spatial document must be a JSON object")
    json_safe(document)
    require_unit(manifest["units"])
    if document.get("document_id") != manifest["document_id"]:
        raise ValidationError("document ID differs from manifest ID")
    if document.get("revision") != manifest["revision"]:
        raise ValidationError("document revision differs from manifest")
    key = "objects" if kind == "slai.sgeo" else "entities"
    objects = document.get(key, [])
    if not isinstance(objects, list):
        raise ValidationError(key + " must be an array")
    if len(objects) > 100_000:
        raise ValidationError("too many spatial elements")
    if any(not isinstance(obj, dict) for obj in objects):
        raise ValidationError("spatial elements must be objects")
    assert_unique(obj.get("id") for obj in objects)
