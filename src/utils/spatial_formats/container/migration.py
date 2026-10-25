"""Conservative, explicit version negotiation for spatial artifact migrations."""
from copy import deepcopy
from ..common.errors import MigrationError, UnsupportedFeatureError
from ..common.helpers import digest, canonical_json

CURRENT_VERSION="0.1.0"

def inspect_migration(manifest):
    if not isinstance(manifest,dict):
        raise MigrationError("manifest is not a mapping")
    version=manifest.get("format_version")
    if version==CURRENT_VERSION:
        return {"source_version":version,"target_version":CURRENT_VERSION,"compatible":True,
                "required":False,"losses":[]}
    return {"source_version":version,"target_version":CURRENT_VERSION,"compatible":False,
            "required":True,"losses":["unsupported schema version"]}

def migrate_document(manifest, document, *, target_version=CURRENT_VERSION):
    """Return independently copied content only for identical already-supported versions.

    Never silently reinterpret past or future formats, and never mutate the input.
    """
    status=inspect_migration(manifest)
    if target_version!=CURRENT_VERSION or not status["compatible"]:
        raise UnsupportedFeatureError("no validated migration path exists for this version pair")
    from ..verification.invariants import validate_document
    validate_document(manifest,document)
    return deepcopy(manifest),deepcopy(document),status
