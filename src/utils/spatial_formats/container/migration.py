"""Conservative, explicit version negotiation and migration for spatial artifacts.

Supported paths: ``0.1.0 -> 0.2.0``. A legacy document is first validated
against its *own* frozen schema and content rules (a file labelled 0.1.0 may not
contain anything introduced later), then upgraded by adding the empty registries
the current schema requires. Unknown or future versions are never reinterpreted,
and inputs are never mutated.
"""
from copy import deepcopy
from ..common.errors import MigrationError, UnsupportedFeatureError, ValidationError
from .versions import (CURRENT_VERSION, LEGACY_VERSIONS, SUPPORTED_VERSIONS, FEATURES, LEGACY_SDRW_KINDS,
                       LEGACY_SDRW_STYLE_KEYS, LEGACY_SGEO_KINDS, LEGACY_SGEO_MESH_KEYS,
                       LEGACY_SGEO_OBJECT_KEYS, schema_id)

__all__ = ["CURRENT_VERSION", "inspect_migration", "migrate_document", "upgrade_loaded"]


def inspect_migration(manifest):
    """Describe what loading ``manifest`` requires; never raises for an unknown version."""
    if not isinstance(manifest, dict):
        raise MigrationError("manifest is not a mapping")
    version = manifest.get("format_version")
    if version == CURRENT_VERSION:
        return {"source_version": version, "target_version": CURRENT_VERSION, "compatible": True,
                "required": False, "losses": []}
    if isinstance(version, str) and version in LEGACY_VERSIONS:
        return {"source_version": version, "target_version": CURRENT_VERSION, "compatible": True,
                "required": True, "losses": []}
    return {"source_version": version, "target_version": CURRENT_VERSION, "compatible": False,
            "required": True, "losses": ["unsupported schema version"]}


def _check_legacy_sdrw(document):
    for entity in document.get("entities", ()):
        if entity.get("kind") not in LEGACY_SDRW_KINDS:
            raise MigrationError("a 0.1.0 drawing cannot contain entity kind %r" % (entity.get("kind"),))
        if set(entity.get("style", {})) - LEGACY_SDRW_STYLE_KEYS:
            raise MigrationError("a 0.1.0 drawing cannot use newer style properties")


def _check_legacy_sgeo(document):
    for obj in document.get("objects", ()):
        if set(obj) - LEGACY_SGEO_OBJECT_KEYS:
            raise MigrationError("a 0.1.0 scene object cannot use newer fields")
        geometry = obj.get("geometry")
        if geometry is None:
            continue
        kind = geometry.get("kind")
        if kind not in LEGACY_SGEO_KINDS and kind != "mesh_buffer":
            raise MigrationError("a 0.1.0 scene cannot contain geometry kind %r" % (kind,))
        if kind == "mesh" and set(geometry) - LEGACY_SGEO_MESH_KEYS:
            raise MigrationError("a 0.1.0 mesh cannot carry newer attributes")


def upgrade_loaded(manifest, document):
    """Upgrade a *hydrated* legacy ``(manifest, document)`` pair to the current format.

    Returns new objects; raises ``MigrationError``/``SchemaError`` when the legacy
    content is not valid for its own version.
    """
    from ..verification.invariants import validate_legacy
    status = inspect_migration(manifest)
    if not status["compatible"]:
        raise UnsupportedFeatureError("no validated migration path exists for this version",
                                      context={"version": manifest.get("format_version")})
    if not status["required"]:
        return deepcopy(manifest), deepcopy(document)
    validate_legacy(manifest, document)
    format_id = manifest["format_id"]
    upgraded = deepcopy(document)
    if format_id == "slai.sdrw":
        _check_legacy_sdrw(upgraded)
        upgraded.setdefault("linetypes", [])
        upgraded.setdefault("blocks", [])
    else:
        _check_legacy_sgeo(upgraded)
        upgraded.setdefault("materials", [])
        upgraded.setdefault("images", [])
    new_manifest = deepcopy(manifest)
    new_manifest["format_version"] = CURRENT_VERSION
    new_manifest["schema"] = schema_id(format_id, CURRENT_VERSION)
    return new_manifest, upgraded


def migrate_document(manifest, document, *, target_version=CURRENT_VERSION):
    """Return independently copied, validated content at ``target_version``.

    Identical versions are validated and copied; a legacy version is upgraded.
    Never silently reinterprets unknown or future formats and never mutates the input.
    """
    status = inspect_migration(manifest)
    if target_version != CURRENT_VERSION or not status["compatible"]:
        raise UnsupportedFeatureError("no validated migration path exists for this version pair")
    from ..verification.invariants import validate_document
    new_manifest, new_document = upgrade_loaded(manifest, document)
    validate_document(new_manifest, new_document)
    return new_manifest, new_document, status
