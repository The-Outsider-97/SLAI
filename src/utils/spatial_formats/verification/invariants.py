"""Common and profile-specific semantic invariants; import cycles avoided."""
from ..common.errors import ValidationError
from ..common.helpers import json_safe
from ..common.ids import assert_unique
from ..common.units import require_unit

def validate_document(manifest:dict,document:dict)->None:
    from ..container.reader import _manifest
    format_id=_manifest(manifest)
    if not isinstance(document,dict):
        raise ValidationError("document must be a JSON object")
    json_safe(document)
    require_unit(manifest["units"])
    if document.get("document_id")!=manifest["document_id"] or document.get("revision")!=manifest["revision"]:
        raise ValidationError("manifest/document identity and revision must agree")
    if format_id=="slai.sgeo":
        from ..sgeo.scene import Scene
        Scene(**document,unit=manifest["units"],document_kind=manifest["document_kind"])
    else:
        from ..sdrw.drawing import Drawing
        Drawing(**document,unit=manifest["units"])
