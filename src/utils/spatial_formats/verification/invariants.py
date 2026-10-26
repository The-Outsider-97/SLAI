"""Common and profile-specific semantic invariants; import cycles avoided."""
from ..common.errors import ValidationError
from ..common.helpers import json_safe
from ..common.units import require_unit
from functools import lru_cache
from pathlib import Path
import json

@lru_cache(maxsize=2)
def _validator(profile):
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    directory = Path(__file__).resolve().parents[1] / "schemas"
    schemas = [json.loads((directory / name).read_text())
               for name in ("spatial_core.json", "sdrw.json", "sgeo.json")]
    registry = Registry().with_resources((schema["$id"], Resource.from_contents(schema))
                                         for schema in schemas)
    return Draft202012Validator(next(s for s in schemas if s["$id"].endswith(profile + ".json")),
                                registry=registry)

def validate_document(manifest:dict,document:dict)->None:
    from ..container.reader import _manifest
    format_id=_manifest(manifest)
    if not isinstance(document,dict):
        raise ValidationError("document must be a JSON object")
    document = json_safe(document)
    error = next(_validator(format_id.split(".")[-1]).iter_errors(document), None)
    if error is not None:
        location = "/".join(map(str, error.absolute_path)) or "document"
        raise ValidationError(location + ": " + error.message)
    require_unit(manifest["units"])
    if document.get("document_id")!=manifest["document_id"] or document.get("revision")!=manifest["revision"]:
        raise ValidationError("manifest/document identity and revision must agree")
    if format_id=="slai.sgeo":
        from ..sgeo.scene import Scene
        Scene(**document,unit=manifest["units"],document_kind=manifest["document_kind"])
    else:
        from ..sdrw.drawing import Drawing
        Drawing(**document,unit=manifest["units"])
