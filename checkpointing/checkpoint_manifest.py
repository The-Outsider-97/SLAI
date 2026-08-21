"""Schema-v3 checkpoint manifest construction, migration, and verification.

The manifest is the immutable, auditable description of a committed
checkpoint.  It records what was actually written, not what a caller intended
to write.  This module validates schema structure, upgrades SLAI v1/v2
manifests in memory, performs exact-file verification, and delegates atomic
bytes persistence and filesystem observation to ``checkpoint_storage``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import time
import uuid

from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence, overload

from .checkpoint_errors import *
from .checkpoint_storage import *
from .checkpoint_types import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]


logger = get_logger("Checkpoint Manifest")
printer = PrettyPrinter()


_V3_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "checkpoint_id",
        "version",
        "created_at",
        "format",
        "artifacts",
        "producer",
        "epoch",
        "step",
        "metrics",
        "metadata",
        "provenance",
        "compatibility",
    }
)


def make_checkpoint_id() -> str:
    """Return an opaque, collision-resistant checkpoint identity."""

    return uuid.uuid4().hex


def make_version(version: str | None = None) -> str:
    """Return a validated caller version or a sortable generated version."""

    if version is not None:
        try:
            return validate_version(version)
        except ValueError as exc:
            raise CheckpointManifestError(
                str(exc),
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
                version=version,
            ) from exc
    timestamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"v_{timestamp}_{uuid.uuid4().hex[:8]}"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not permitted: {value}")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode strict, deterministic UTF-8 JSON suitable for hashing."""

    try:
        frozen = freeze_json(dict(value))
        encoded = json.dumps(
            thaw_json(frozen),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointManifestError(
            f"manifest contains a non-JSON value: {exc}",
            operation=CheckpointOperation.SAVE,
            stage=CheckpointStage.MANIFEST,
        ) from exc
    return encoded.encode("utf-8")


def encode_manifest(
    manifest: CheckpointManifest,
    *,
    pretty: bool = True,
    limits: ManifestLimits | None = None,
) -> bytes:
    """Validate and encode a schema-v3 manifest."""

    active_limits = limits or ManifestLimits()
    try:
        payload = manifest.to_json_dict()
        if pretty:
            data = (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                    separators=(",", ": "),
                )
                + "\n"
            ).encode("utf-8")
        else:
            data = canonical_json_bytes(payload)
    except (TypeError, ValueError) as exc:
        raise CheckpointManifestError(
            f"failed to encode checkpoint manifest: {exc}",
            operation=CheckpointOperation.SAVE,
            stage=CheckpointStage.MANIFEST,
            version=manifest.version,
            checkpoint_id=manifest.checkpoint_id,
        ) from exc
    if len(data) > active_limits.max_bytes:
        raise CheckpointManifestTooLargeError(
            "encoded checkpoint manifest exceeds the configured limit",
            operation=CheckpointOperation.SAVE,
            stage=CheckpointStage.MANIFEST,
            version=manifest.version,
            checkpoint_id=manifest.checkpoint_id,
            details={"size_bytes": len(data), "max_bytes": active_limits.max_bytes},
        )
    return data


def manifest_digest(manifest: CheckpointManifest) -> ArtifactDigest:
    """Return the SHA-256 digest of the canonical manifest representation."""

    digest = hashlib.sha256(canonical_json_bytes(manifest.to_json_dict())).hexdigest()
    return ArtifactDigest("sha256", digest)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return value


def _string(value: Any, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _integer(value: Any, field: str, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _only_keys(value: Mapping[str, Any], allowed: Iterable[str], field: str) -> None:
    unexpected = set(value) - set(allowed)
    if unexpected:
        raise ValueError(f"{field} contains unknown fields: {sorted(unexpected)}")


def _parse_digest(value: Any, field: str) -> ArtifactDigest:
    data = _mapping(value, field)
    _only_keys(data, {"algorithm", "value"}, field)
    return ArtifactDigest(
        algorithm=str(_string(data.get("algorithm"), f"{field}.algorithm")),
        value=str(_string(data.get("value"), f"{field}.value")),
    )


def _parse_artifact(value: Any, index: int) -> CheckpointArtifact:
    field = f"artifacts[{index}]"
    data = _mapping(value, field)
    allowed = {
        "path",
        "component",
        "size_bytes",
        "digest",
        "codec",
        "codec_version",
        "media_type",
        "required",
        "metadata",
    }
    _only_keys(data, allowed, field)
    size_bytes = _integer(data.get("size_bytes"), f"{field}.size_bytes")
    if size_bytes is None:
        raise ValueError(f"{field}.size_bytes must be an integer")
    return CheckpointArtifact(
        relative_path=str(_string(data.get("path"), f"{field}.path")),
        component=str(_string(data.get("component"), f"{field}.component")),
        size_bytes=size_bytes,
        digest=_parse_digest(data.get("digest"), f"{field}.digest"),
        codec=_string(data.get("codec"), f"{field}.codec", optional=True),
        codec_version=_string(
            data.get("codec_version"), f"{field}.codec_version", optional=True
        ),
        media_type=str(
            _string(
                data.get("media_type", "application/octet-stream"),
                f"{field}.media_type",
            )
        ),
        required=_boolean(data.get("required", True), f"{field}.required"),
        metadata=dict(_mapping(data.get("metadata", {}), f"{field}.metadata")),
    )


def _parse_producer(value: Any) -> ProducerInfo:
    data = _mapping(value, "producer")
    _only_keys(data, {"name", "version", "environment"}, "producer")
    return ProducerInfo(
        name=str(_string(data.get("name"), "producer.name")),
        version=_string(data.get("version"), "producer.version", optional=True),
        environment=dict(_mapping(data.get("environment", {}), "producer.environment")),
    )


def _parse_lineage(value: Any) -> CheckpointLineage:
    data = _mapping(value, "provenance.lineage")
    allowed = {
        "parent_checkpoint_id",
        "restored_from_checkpoint_id",
        "generation",
    }
    _only_keys(data, allowed, "provenance.lineage")
    generation_value = _integer(data.get("generation", 0), "provenance.lineage.generation")
    if generation_value is None:
        raise ValueError("provenance.lineage.generation must be an integer")
    return CheckpointLineage(
        parent_checkpoint_id=_string(
            data.get("parent_checkpoint_id"),
            "provenance.lineage.parent_checkpoint_id",
            optional=True,
        ),
        restored_from_checkpoint_id=_string(
            data.get("restored_from_checkpoint_id"),
            "provenance.lineage.restored_from_checkpoint_id",
            optional=True,
        ),
        generation=generation_value,
    )


def _parse_objective(value: Any) -> PrimaryMetric:
    data = _mapping(value, "provenance.objective")
    _only_keys(data, {"name", "value", "direction", "step"}, "provenance.objective")
    number = data.get("value")
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        raise ValueError("provenance.objective.value must be numeric")
    return PrimaryMetric(
        name=str(_string(data.get("name"), "provenance.objective.name")),
        value=float(number),
        direction=MetricDirection(
            str(_string(data.get("direction"), "provenance.objective.direction"))
        ),
        step=_integer(data.get("step"), "provenance.objective.step", optional=True),
    )


def _parse_provenance(value: Any) -> CheckpointProvenance:
    data = _mapping(value, "provenance")
    allowed = {
        "agent_id",
        "run_id",
        "trace_id",
        "checkpoint_reason",
        "code_revision",
        "config_digest",
        "model_signature",
        "objective",
        "lineage",
        "requested_components",
        "omitted_components",
    }
    _only_keys(data, allowed, "provenance")
    requested_values = _sequence(
        data.get("requested_components", []),
        "provenance.requested_components",
    )
    omitted_values = _sequence(data.get("omitted_components", []), "provenance.omitted_components")
    omitted: list[OmittedComponent] = []
    for index, raw in enumerate(omitted_values):
        item = _mapping(raw, f"provenance.omitted_components[{index}]")
        _only_keys(
            item,
            {"component", "reason"},
            f"provenance.omitted_components[{index}]",
        )
        omitted.append(
            OmittedComponent(
                component=str(
                    _string(
                        item.get("component"),
                        f"provenance.omitted_components[{index}].component",
                    )
                ),
                reason=str(
                    _string(
                        item.get("reason"),
                        f"provenance.omitted_components[{index}].reason",
                    )
                ),
            )
        )
    return CheckpointProvenance(
        agent_id=_string(data.get("agent_id"), "provenance.agent_id", optional=True),
        run_id=_string(data.get("run_id"), "provenance.run_id", optional=True),
        trace_id=_string(data.get("trace_id"), "provenance.trace_id", optional=True),
        checkpoint_reason=_string(
            data.get("checkpoint_reason"), "provenance.checkpoint_reason", optional=True
        ),
        code_revision=_string(
            data.get("code_revision"), "provenance.code_revision", optional=True
        ),
        config_digest=(
            _parse_digest(data["config_digest"], "provenance.config_digest")
            if data.get("config_digest") is not None
            else None
        ),
        model_signature=(
            _parse_digest(data["model_signature"], "provenance.model_signature")
            if data.get("model_signature") is not None
            else None
        ),
        objective=(
            _parse_objective(data["objective"])
            if data.get("objective") is not None
            else None
        ),
        lineage=_parse_lineage(data.get("lineage", {})),
        requested_components=tuple(
            str(_string(item, "provenance.requested_components[]"))
            for item in requested_values
        ),
        omitted_components=tuple(omitted),
    )


def _parse_compatibility(value: Any) -> CompatibilityConstraints:
    data = _mapping(value, "compatibility")
    allowed = {
        "slai_version",
        "python_version",
        "platform_tags",
        "required_codecs",
        "component_schemas",
    }
    _only_keys(data, allowed, "compatibility")
    tags = _sequence(data.get("platform_tags", []), "compatibility.platform_tags")
    codecs = _mapping(data.get("required_codecs", {}), "compatibility.required_codecs")
    schemas = _mapping(data.get("component_schemas", {}), "compatibility.component_schemas")
    return CompatibilityConstraints(
        slai_version=_string(
            data.get("slai_version"), "compatibility.slai_version", optional=True
        ),
        python_version=_string(
            data.get("python_version"), "compatibility.python_version", optional=True
        ),
        platform_tags=tuple(str(_string(tag, "compatibility.platform_tags[]")) for tag in tags),
        required_codecs={
            str(name): str(_string(version, f"compatibility.required_codecs.{name}"))
            for name, version in codecs.items()
        },
        component_schemas={
            str(name): int(_integer(version, f"compatibility.component_schemas.{name}", optional=False) or 0)
            for name, version in schemas.items()
        },
    )


def _legacy_component(relative_path: str) -> str:
    if relative_path in {TORCH_CHECKPOINT_NAME}:
        return StandardComponent.CHECKPOINT_PAYLOAD.value
    if relative_path in {LEGACY_TORCH_CHECKPOINT_NAME, NPZ_WEIGHTS_NAME}:
        return StandardComponent.MODEL.value
    if relative_path in {TOKENIZER_VOCAB_NAME, TOKENIZER_STATE_NAME} or relative_path.startswith(
        f"{TOKENIZER_DIR_NAME}/"
    ):
        return StandardComponent.TOKENIZER.value
    return StandardComponent.LEGACY_PAYLOAD.value


def _legacy_codec(relative_path: str, tokenizer_kind: Any) -> str | None:
    if relative_path in {TORCH_CHECKPOINT_NAME, LEGACY_TORCH_CHECKPOINT_NAME}:
        return "torch"
    if relative_path == NPZ_WEIGHTS_NAME:
        return "numpy"
    if relative_path in {TOKENIZER_VOCAB_NAME, TOKENIZER_STATE_NAME} or relative_path.startswith(
        f"{TOKENIZER_DIR_NAME}/"
    ):
        return str(tokenizer_kind) if tokenizer_kind else None
    return None


def _legacy_checkpoint_id(data: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(data)).hexdigest()
    return f"legacy-{digest[:32]}"


def _upgrade_legacy_mapping(
    data: Mapping[str, Any],
    *,
    source_schema: int,
    fallback_created_at: str | None,
) -> Mapping[str, Any]:
    """Convert the exact SLAI v1/v2 manifest shape to schema v3 in memory."""

    version = str(_string(data.get("version"), "version"))
    validate_version(version)
    checkpoint_format = normalize_format(str(_string(data.get("format"), "format")))
    files = _mapping(data.get("files"), "files")
    if not files:
        raise ValueError("legacy manifest files cannot be empty")

    artifacts: list[dict[str, Any]] = []
    for relative_path, raw_info in sorted(files.items()):
        canonical = validate_relative_path(str(relative_path))
        info = _mapping(raw_info, f"files.{relative_path}")
        size = _integer(info.get("size_bytes"), f"files.{relative_path}.size_bytes")
        digest = _string(info.get("sha256"), f"files.{relative_path}.sha256")
        artifacts.append(
            {
                "path": canonical,
                "component": _legacy_component(canonical),
                "size_bytes": size,
                "digest": {"algorithm": "sha256", "value": digest},
                "codec": _legacy_codec(canonical, data.get("tokenizer_kind")),
                "codec_version": None,
                "media_type": "application/octet-stream",
                "required": True,
                "metadata": {},
            }
        )

    created_at = data.get("created_at") or fallback_created_at
    if created_at is None:
        raise ValueError("legacy manifest has no created_at and no filesystem fallback")
    validate_utc_timestamp(str(created_at))

    metadata = dict(_mapping(data.get("metadata", {}), "metadata"))
    metadata.setdefault("legacy_manifest_schema_version", source_schema)
    framework = dict(_mapping(data.get("framework", {}), "framework"))
    upgraded: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "checkpoint_id": _legacy_checkpoint_id(data),
        "version": version,
        "created_at": str(created_at),
        "format": checkpoint_format.value,
        "artifacts": artifacts,
        "producer": {"name": "SLAI", "environment": framework},
        "metrics": dict(_mapping(data.get("metrics", {}), "metrics")),
        "metadata": metadata,
        "provenance": {
            "lineage": {"generation": 0},
            "requested_components": [],
            "omitted_components": [],
        },
        "compatibility": {
            "platform_tags": [],
            "required_codecs": {},
            "component_schemas": {},
        },
    }
    if data.get("epoch") is not None:
        upgraded["epoch"] = _integer(data.get("epoch"), "epoch")
    if data.get("step") is not None:
        upgraded["step"] = _integer(data.get("step"), "step")
    return upgraded


def manifest_from_mapping(
    value: Mapping[str, Any],
    *,
    limits: ManifestLimits | None = None,
    fallback_created_at: str | None = None,
) -> CheckpointManifest:
    """Parse, migrate, and validate a manifest mapping."""

    active_limits = limits or ManifestLimits()
    try:
        data: Mapping[str, Any] = _mapping(value, "manifest")
        schema = _integer(data.get("schema_version", 1), "schema_version")
        assert schema is not None
        if schema not in {1, 2, MANIFEST_SCHEMA_VERSION}:
            raise CheckpointManifestVersionError(
                "unsupported checkpoint manifest schema version",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.MANIFEST,
                details={
                    "schema_version": schema,
                    "supported_versions": [1, 2, MANIFEST_SCHEMA_VERSION],
                },
            )
        if schema in {1, 2}:
            data = _upgrade_legacy_mapping(
                data,
                source_schema=schema,
                fallback_created_at=fallback_created_at,
            )
        else:
            _only_keys(data, _V3_TOP_LEVEL_KEYS, "manifest")

        artifacts_raw = _sequence(data.get("artifacts"), "artifacts")
        if len(artifacts_raw) > active_limits.max_artifacts:
            raise CheckpointManifestTooLargeError(
                "manifest exceeds the configured artifact-count limit",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.MANIFEST,
                details={
                    "artifact_count": len(artifacts_raw),
                    "max_artifacts": active_limits.max_artifacts,
                },
            )
        metadata = dict(_mapping(data.get("metadata", {}), "metadata"))
        metadata_size = len(canonical_json_bytes(metadata))
        if metadata_size > active_limits.max_metadata_bytes:
            raise CheckpointManifestTooLargeError(
                "manifest metadata exceeds the configured limit",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.MANIFEST,
                details={
                    "metadata_bytes": metadata_size,
                    "max_metadata_bytes": active_limits.max_metadata_bytes,
                },
            )

        return CheckpointManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            checkpoint_id=str(
                _string(data.get("checkpoint_id"), "checkpoint_id")
            ),
            version=str(_string(data.get("version"), "version")),
            created_at=str(_string(data.get("created_at"), "created_at")),
            checkpoint_format=normalize_format(
                str(_string(data.get("format"), "format"))
            ),
            artifacts=tuple(
                _parse_artifact(item, index)
                for index, item in enumerate(artifacts_raw)
            ),
            producer=_parse_producer(data.get("producer")),
            epoch=_integer(data.get("epoch"), "epoch", optional=True),
            step=_integer(data.get("step"), "step", optional=True),
            metrics=dict(_mapping(data.get("metrics", {}), "metrics")),
            metadata=metadata,
            provenance=_parse_provenance(data.get("provenance", {})),
            compatibility=_parse_compatibility(data.get("compatibility", {})),
        )
    except (CheckpointManifestError, CheckpointManifestVersionError):
        raise
    except (TypeError, ValueError, KeyError) as exc:
        raise CheckpointManifestError(
            f"invalid checkpoint manifest: {exc}",
            operation=CheckpointOperation.LOAD,
            stage=CheckpointStage.MANIFEST,
        ) from exc


def decode_manifest(
    data: bytes,
    *,
    limits: ManifestLimits | None = None,
    fallback_created_at: str | None = None,
) -> CheckpointManifest:
    """Decode strict UTF-8 JSON and return a validated schema-v3 manifest."""

    active_limits = limits or ManifestLimits()
    if not isinstance(data, bytes):
        raise TypeError("decode_manifest requires bytes")
    if len(data) > active_limits.max_bytes:
        raise CheckpointManifestTooLargeError(
            "checkpoint manifest exceeds the configured byte limit",
            operation=CheckpointOperation.LOAD,
            stage=CheckpointStage.MANIFEST,
            details={"size_bytes": len(data), "max_bytes": active_limits.max_bytes},
        )
    try:
        text = data.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CheckpointManifestError(
            f"checkpoint manifest is not valid strict UTF-8 JSON: {exc}",
            operation=CheckpointOperation.LOAD,
            stage=CheckpointStage.MANIFEST,
        ) from exc
    if not isinstance(value, Mapping):
        raise CheckpointManifestError(
            "checkpoint manifest root must be a JSON object",
            operation=CheckpointOperation.LOAD,
            stage=CheckpointStage.MANIFEST,
        )
    return manifest_from_mapping(
        value,
        limits=active_limits,
        fallback_created_at=fallback_created_at,
    )


def build_artifact(
    checkpoint_dir: Path,
    relative_path: str,
    *,
    component: str,
    codec: str | None = None,
    codec_version: str | None = None,
    media_type: str = "application/octet-stream",
    required: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> CheckpointArtifact:
    """Build a manifest entry from an artifact that was actually written."""

    path = resolve_artifact_path(checkpoint_dir, relative_path, must_exist=True)
    return CheckpointArtifact(
        relative_path=validate_relative_path(relative_path),
        component=validate_component_name(component),
        size_bytes=path.stat().st_size,
        digest=ArtifactDigest("sha256", sha256_file(path)),
        codec=codec,
        codec_version=codec_version,
        media_type=media_type,
        required=required,
        metadata=dict(metadata or {}),
    )


def build_manifest(
    *,
    version: str,
    artifacts: Sequence[CheckpointArtifact],
    checkpoint_format: CheckpointFormat | str = CheckpointFormat.COMPOSITE,
    producer: ProducerInfo,
    checkpoint_id: str | None = None,
    created_at: str | None = None,
    epoch: int | None = None,
    step: int | None = None,
    metrics: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    provenance: CheckpointProvenance | None = None,
    compatibility: CompatibilityConstraints | None = None,
) -> CheckpointManifest:
    """Construct a validated manifest from storage-derived artifacts."""

    try:
        return CheckpointManifest(
            checkpoint_id=checkpoint_id or make_checkpoint_id(),
            version=validate_version(version),
            created_at=created_at or utc_now_iso(),
            checkpoint_format=normalize_format(checkpoint_format),
            artifacts=tuple(artifacts),
            producer=producer,
            epoch=epoch,
            step=step,
            metrics=dict(metrics or {}),
            metadata=dict(metadata or {}),
            provenance=provenance or CheckpointProvenance(),
            compatibility=compatibility or CompatibilityConstraints(),
        )
    except ValueError as exc:
        raise CheckpointManifestError(
            f"failed to construct checkpoint manifest: {exc}",
            operation=CheckpointOperation.SAVE,
            stage=CheckpointStage.MANIFEST,
            version=version,
        ) from exc


def validate_manifest_identity(
    manifest: CheckpointManifest,
    checkpoint_dir: Path,
) -> None:
    """Require the published directory name to match ``manifest.version``."""

    actual_version = Path(checkpoint_dir).name
    if actual_version != manifest.version:
        raise CheckpointManifestError(
            "manifest version does not match its checkpoint directory",
            operation=CheckpointOperation.VERIFY,
            stage=CheckpointStage.MANIFEST,
            path=checkpoint_dir,
            version=manifest.version,
            checkpoint_id=manifest.checkpoint_id,
            details={
                "manifest_version": manifest.version,
                "directory_version": actual_version,
            },
        )


def write_manifest(
    checkpoint_dir: Path,
    manifest: CheckpointManifest,
    *,
    overwrite: bool = False,
    verify_artifacts: bool = True,
    exact_files: bool = True,
    durable: bool = True,
    limits: ManifestLimits | None = None,
) -> Path:
    """Verify artifacts and atomically write the manifest as the final staged file."""

    directory = Path(checkpoint_dir)
    manifest_path = directory / MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        raise CheckpointConflictError(
            "checkpoint manifest already exists",
            operation=CheckpointOperation.SAVE,
            stage=CheckpointStage.MANIFEST,
            path=manifest_path,
            version=manifest.version,
            checkpoint_id=manifest.checkpoint_id,
            committed=False,
        )
    if verify_artifacts:
        # A staging directory intentionally has a transaction-specific name;
        # directory/manifest identity is checked after atomic publication.
        assert_manifest_files(
            directory,
            manifest,
            exact_files=exact_files,
            validate_identity=False,
        )
    payload = encode_manifest(manifest, pretty=True, limits=limits)
    atomic_write_bytes(payload, manifest_path, durable=durable)
    return manifest_path


def read_manifest(
    checkpoint_dir: Path,
    *,
    missing_ok: bool = False,
    validate_identity: bool = True,
    limits: ManifestLimits | None = None,
) -> CheckpointManifest | None:
    """Read a committed manifest, applying supported legacy adapters in memory."""

    directory = Path(checkpoint_dir)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        if missing_ok:
            return None
        raise CheckpointManifestError(
            "checkpoint manifest does not exist",
            operation=CheckpointOperation.LOAD,
            stage=CheckpointStage.MANIFEST,
            path=manifest_path,
            version=directory.name,
        )
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise CheckpointManifestError(
            "checkpoint manifest must be a regular, non-symlink file",
            operation=CheckpointOperation.LOAD,
            stage=CheckpointStage.MANIFEST,
            path=manifest_path,
            version=directory.name,
        )
    active_limits = limits or ManifestLimits()
    try:
        data = read_limited_bytes(manifest_path, max_bytes=active_limits.max_bytes)
        fallback = _dt.datetime.fromtimestamp(
            manifest_path.stat().st_mtime, tz=_dt.timezone.utc
        ).isoformat().replace("+00:00", "Z")
        manifest = decode_manifest(
            data,
            limits=active_limits,
            fallback_created_at=fallback,
        )
    except CheckpointManifestError:
        raise
    except CheckpointStorageError as exc:
        if exc.details.get("size_bytes") is not None:
            raise CheckpointManifestTooLargeError(
                exc.message,
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.MANIFEST,
                path=manifest_path,
                version=directory.name,
                details=exc.details,
            ) from exc
        raise CheckpointManifestError(
            "failed to read checkpoint manifest",
            operation=CheckpointOperation.LOAD,
            stage=CheckpointStage.MANIFEST,
            path=manifest_path,
            version=directory.name,
            retryable=exc.retryable,
        ) from exc
    if validate_identity:
        validate_manifest_identity(manifest, directory)
    return manifest


def verify_manifest_files(
    checkpoint_dir: Path,
    manifest: CheckpointManifest,
    *,
    exact_files: bool = True,
    validate_identity: bool = True,
) -> VerificationResult:
    """Compare the manifest with the exact observed file set and digests."""

    started = time.monotonic()
    issues: list[VerificationIssue] = []
    if validate_identity:
        try:
            validate_manifest_identity(manifest, checkpoint_dir)
        except CheckpointManifestError as exc:
            issues.append(
                VerificationIssue(
                    code="identity_mismatch",
                    message=exc.message,
                    expected=manifest.version,
                    actual=Path(checkpoint_dir).name,
                )
            )

    try:
        observed = observe_checkpoint_files(checkpoint_dir, exclude=(MANIFEST_NAME,))
    except (CheckpointPathError, CheckpointStorageError) as exc:
        issues.append(
            VerificationIssue(
                code="unsafe_file",
                message=exc.message,
                actual=str(exc.path) if exc.path is not None else None,
            )
        )
        return VerificationResult(
            status=VerificationStatus.FAILED,
            health=CheckpointHealth.CORRUPT,
            issues=tuple(issues),
            duration_seconds=time.monotonic() - started,
        )

    expected_by_path = {item.relative_path: item for item in manifest.artifacts}
    observed_by_path = {item.relative_path: item for item in observed}

    for relative_path, expected in expected_by_path.items():
        actual = observed_by_path.get(relative_path)
        if actual is None:
            issues.append(
                VerificationIssue(
                    code="missing_artifact",
                    message="manifest-declared artifact is missing",
                    relative_path=relative_path,
                    expected=True,
                    actual=False,
                )
            )
            continue
        if actual.size_bytes != expected.size_bytes:
            issues.append(
                VerificationIssue(
                    code="size_mismatch",
                    message="artifact size differs from the manifest",
                    relative_path=relative_path,
                    expected=expected.size_bytes,
                    actual=actual.size_bytes,
                )
            )
        if actual.digest != expected.digest:
            issues.append(
                VerificationIssue(
                    code="hash_mismatch",
                    message="artifact SHA-256 differs from the manifest",
                    relative_path=relative_path,
                    expected=expected.digest.value,
                    actual=actual.digest.value,
                )
            )

    if exact_files:
        for relative_path in sorted(set(observed_by_path) - set(expected_by_path)):
            issues.append(
                VerificationIssue(
                    code="unexpected_artifact",
                    message="checkpoint contains an undeclared file",
                    relative_path=relative_path,
                    expected=False,
                    actual=True,
                )
            )

    if not issues:
        status = VerificationStatus.PASSED
        health = CheckpointHealth.HEALTHY
    else:
        status = VerificationStatus.FAILED
        codes = {issue.code for issue in issues}
        if codes & {"hash_mismatch", "size_mismatch", "unsafe_file", "identity_mismatch"}:
            health = CheckpointHealth.CORRUPT
        elif "missing_artifact" in codes:
            health = CheckpointHealth.INCOMPLETE
        else:
            health = CheckpointHealth.DEGRADED
    return VerificationResult(
        status=status,
        health=health,
        issues=tuple(issues),
        observed_artifacts=observed,
        duration_seconds=time.monotonic() - started,
    )


def assert_manifest_files(
    checkpoint_dir: Path,
    manifest: CheckpointManifest,
    *,
    exact_files: bool = True,
    validate_identity: bool = True,
) -> VerificationResult:
    """Verify files and raise the most specific structured integrity error."""

    result = verify_manifest_files(
        checkpoint_dir,
        manifest,
        exact_files=exact_files,
        validate_identity=validate_identity,
    )
    if result.ok:
        return result

    issue = result.issues[0]
    context = {
        "operation": CheckpointOperation.VERIFY,
        "stage": CheckpointStage.INTEGRITY,
        "path": Path(checkpoint_dir),
        "version": manifest.version,
        "checkpoint_id": manifest.checkpoint_id,
        "details": {
            "relative_path": issue.relative_path,
            "expected": thaw_json(issue.expected),
            "actual": thaw_json(issue.actual),
            "issues": [
                {
                    "code": item.code,
                    "message": item.message,
                    "relative_path": item.relative_path,
                    "expected": thaw_json(item.expected),
                    "actual": thaw_json(item.actual),
                }
                for item in result.issues
            ],
        },
    }
    if issue.code == "missing_artifact":
        raise CheckpointMissingArtifactError(issue.message, **context)
    if issue.code == "unexpected_artifact":
        raise CheckpointUnexpectedArtifactError(issue.message, **context)
    if issue.code == "size_mismatch":
        context["details"].update(
            {
                "expected_size": thaw_json(issue.expected),
                "actual_size": thaw_json(issue.actual),
            }
        )
        raise CheckpointSizeMismatchError(issue.message, **context)
    if issue.code == "hash_mismatch":
        context["details"].update(
            {
                "expected_hash": thaw_json(issue.expected),
                "actual_hash": thaw_json(issue.actual),
            }
        )
        raise CheckpointHashMismatchError(issue.message, **context)
    raise CheckpointIntegrityError(issue.message, **context)


def read_checkpoint_record(
    checkpoint_dir: Path,
    *,
    verify: bool = False,
    exact_files: bool = True,
    limits: ManifestLimits | None = None,
) -> CheckpointRecord:
    manifest = read_manifest(checkpoint_dir, limits=limits)
    assert manifest is not None
    verification = (
        verify_manifest_files(checkpoint_dir, manifest, exact_files=exact_files)
        if verify
        else None
    )
    return CheckpointRecord(
        manifest=manifest,
        path=Path(checkpoint_dir),
        health=verification.health if verification else CheckpointHealth.UNKNOWN,
        verification=verification,
    )


def looks_like_legacy_checkpoint(path: Path) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and not candidate.is_symlink() and (
        (candidate / LEGACY_TORCH_CHECKPOINT_NAME).is_file()
        or (candidate / NPZ_WEIGHTS_NAME).is_file()
    )


def build_legacy_record(path: Path) -> CheckpointRecord:
    """Build an in-memory record for a pre-manifest SLAI checkpoint."""

    candidate = Path(path)
    if not looks_like_legacy_checkpoint(candidate):
        raise CheckpointManifestError(
            "directory is not a recognized legacy checkpoint",
            operation=CheckpointOperation.LOAD,
            stage=CheckpointStage.DISCOVERY,
            path=candidate,
            version=candidate.name,
        )
    observed = observe_checkpoint_files(candidate, exclude=(MANIFEST_NAME,))
    checkpoint_format = (
        CheckpointFormat.TORCH
        if (candidate / LEGACY_TORCH_CHECKPOINT_NAME).is_file()
        else CheckpointFormat.NPZ
    )
    identity_source = {
        "version": candidate.name,
        "created_at_ns": candidate.stat().st_mtime_ns,
        "files": [
            {
                "path": item.relative_path,
                "size": item.size_bytes,
                "sha256": item.digest.value,
            }
            for item in observed
        ],
    }
    checkpoint_id = _legacy_checkpoint_id(identity_source)
    artifacts = tuple(
        CheckpointArtifact(
            relative_path=item.relative_path,
            component=_legacy_component(item.relative_path),
            size_bytes=item.size_bytes,
            digest=item.digest,
            codec=_legacy_codec(item.relative_path, None),
        )
        for item in observed
    )
    created_at = _dt.datetime.fromtimestamp(
        candidate.stat().st_mtime, tz=_dt.timezone.utc
    ).isoformat().replace("+00:00", "Z")
    manifest = CheckpointManifest(
        checkpoint_id=checkpoint_id,
        version=validate_version(candidate.name),
        created_at=created_at,
        checkpoint_format=checkpoint_format,
        artifacts=artifacts,
        producer=ProducerInfo(name="SLAI"),
        metadata={"legacy_layout": True, "legacy_manifest_schema_version": 0},
    )
    return CheckpointRecord(
        manifest=manifest,
        path=candidate,
        health=CheckpointHealth.UNKNOWN,
    )


def verify_files_against_record(
    checkpoint_dir: Path,
    record: CheckpointManifest | CheckpointRecord,
) -> bool:
    """Backward-compatible verification wrapper."""

    manifest = record.manifest if isinstance(record, CheckpointRecord) else record
    assert_manifest_files(checkpoint_dir, manifest, exact_files=True)
    return True


__all__ = [
    "assert_manifest_files",
    "build_artifact",
    "build_legacy_record",
    "build_manifest",
    "canonical_json_bytes",
    "decode_manifest",
    "encode_manifest",
    "looks_like_legacy_checkpoint",
    "make_checkpoint_id",
    "make_version",
    "manifest_digest",
    "manifest_from_mapping",
    "read_checkpoint_record",
    "read_manifest",
    "validate_manifest_identity",
    "verify_files_against_record",
    "verify_manifest_files",
    "write_manifest",
]