"""Core types and contracts for SLAI checkpointing.

This module contains immutable data models, enums, configuration, result
objects, and structural protocols shared by the checkpointing package.  It has
no dependency on PyTorch, NumPy, concrete storage, codecs, logging, or the
checkpoint manager, keeping it safe to import from every other layer.
"""

from __future__ import annotations

import datetime as _dt
import math
import re

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Protocol, Sequence, TypeAlias, TypeVar, runtime_checkable


MANIFEST_SCHEMA_VERSION = 3
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset({1, 2, 3})

MANIFEST_NAME = "manifest.json"
TORCH_CHECKPOINT_NAME = "checkpoint.pt"
LEGACY_TORCH_CHECKPOINT_NAME = "model_weights.pt"
NPZ_WEIGHTS_NAME = "model_weights.npz"
TOKENIZER_VOCAB_NAME = "tokenizer_vocab.json"
TOKENIZER_STATE_NAME = "tokenizer_state.pt"
TOKENIZER_DIR_NAME = "tokenizer"
METADATA_NAME = "metadata.json"

# Backward-compatible name used by checkpoint_utils.py and CheckpointManager.
SCHEMA_VERSION = MANIFEST_SCHEMA_VERSION

_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,255}$")
_COMPONENT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:-]{0,127}$")
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DEVICE_RE = re.compile(
    r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$", re.IGNORECASE
)

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | Mapping[str, "JSONValue"] | Sequence["JSONValue"]


class CheckpointFormat(str, Enum):
    """Container-level checkpoint representation."""

    COMPOSITE = "composite"
    TORCH = "torch"
    NPZ = "npz"


class StandardComponent(str, Enum):
    """Canonical names for components understood by SLAI.

    Custom agent components remain permitted through validated string names;
    this enum is a shared vocabulary, not a closed registry.
    """

    MODEL = "model"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    SCALER = "scaler"
    TOKENIZER = "tokenizer"
    RNG = "rng"
    AGENT_STATE = "agent_state"
    METRICS = "metrics"
    METADATA = "metadata"
    CHECKPOINT_PAYLOAD = "checkpoint_payload"
    LEGACY_PAYLOAD = "legacy_payload"


class MetricDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class CheckpointHealth(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    INCOMPLETE = "incomplete"
    CORRUPT = "corrupt"
    INCOMPATIBLE = "incompatible"
    QUARANTINED = "quarantined"


class VerificationStatus(str, Enum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"


def utc_now_iso() -> str:
    """Return a canonical RFC 3339 UTC timestamp."""

    return (
        _dt.datetime.now(tz=_dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def validate_utc_timestamp(value: str) -> str:
    """Validate that ``value`` is an offset-aware ISO-8601 timestamp."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("timestamp must be a non-empty ISO-8601 string")
    candidate = value.strip()
    parse_value = candidate[:-1] + "+00:00" if candidate.endswith("Z") else candidate
    try:
        parsed = _dt.datetime.fromisoformat(parse_value)
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include an explicit UTC offset")
    return candidate


def normalize_format(value: str | CheckpointFormat) -> CheckpointFormat:
    """Normalize supported legacy aliases to a canonical format."""

    if isinstance(value, CheckpointFormat):
        return value
    candidate = str(value).strip().lower()
    aliases = {
        "composite": CheckpointFormat.COMPOSITE,
        "torch": CheckpointFormat.TORCH,
        "pt": CheckpointFormat.TORCH,
        "pth": CheckpointFormat.TORCH,
        "pytorch": CheckpointFormat.TORCH,
        "npz": CheckpointFormat.NPZ,
        "numpy": CheckpointFormat.NPZ,
    }
    try:
        return aliases[candidate]
    except KeyError as exc:
        supported = ", ".join(item.value for item in CheckpointFormat)
        raise ValueError(f"unsupported checkpoint format {value!r}; expected one of: {supported}") from exc


def validate_version(value: str, *, allow_latest: bool = False) -> str:
    """Validate a checkpoint directory version without modifying it."""

    if not isinstance(value, str) or not value:
        raise ValueError("checkpoint version must be a non-empty string")
    if value == "latest":
        if allow_latest:
            return value
        raise ValueError("'latest' is a selector and cannot identify a checkpoint directory")
    if not _VERSION_RE.fullmatch(value):
        raise ValueError(
            "checkpoint version must start with an alphanumeric character and "
            "contain only letters, numbers, '.', '_', or '-' (maximum 128 characters)"
        )
    if ".." in value:
        raise ValueError("checkpoint version cannot contain '..'")
    return value


def validate_identifier(value: str, *, field_name: str = "identifier") -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} must be 1-256 characters and use only safe identifier characters"
        )
    return value


def validate_component_name(value: str | StandardComponent) -> str:
    candidate = value.value if isinstance(value, StandardComponent) else value
    if not isinstance(candidate, str) or not _COMPONENT_RE.fullmatch(candidate):
        raise ValueError(
            "component name must begin with a letter and contain only letters, "
            "numbers, '.', '_', ':', or '-'"
        )
    return candidate


def validate_relative_path(value: str, *, allow_manifest: bool = False) -> str:
    """Validate and return a canonical POSIX checkpoint-relative path."""

    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be a non-empty string")
    if "\x00" in value or "\\" in value:
        raise ValueError("artifact path must be a NUL-free POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("/"):
        raise ValueError("artifact path must be relative")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("artifact path cannot contain empty, '.' or '..' components")
    if any(_WINDOWS_DEVICE_RE.fullmatch(part) for part in path.parts):
        raise ValueError("artifact path contains a reserved Windows device name")
    canonical = path.as_posix()
    if not allow_manifest and canonical == MANIFEST_NAME:
        raise ValueError(f"{MANIFEST_NAME!r} is reserved for the checkpoint manifest")
    return canonical


def freeze_json(value: Any, *, _path: str = "$", _depth: int = 0) -> JSONValue:
    """Validate and recursively freeze a JSON-compatible value.

    Unsupported objects are rejected rather than silently stringified.  This
    prevents non-reproducible manifests and accidental leakage through object
    representations.  Codec/manager layers must normalize framework-specific
    scalar types before constructing a manifest.
    """

    if _depth > 64:
        raise ValueError(f"JSON value exceeds maximum nesting at {_path}")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite floating-point value at {_path}")
        return value
    if isinstance(value, Enum):
        return freeze_json(value.value, _path=_path, _depth=_depth + 1)
    if isinstance(value, Mapping):
        frozen: MutableMapping[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"JSON object key at {_path} must be a string")
            frozen[key] = freeze_json(
                item, _path=f"{_path}.{key}", _depth=_depth + 1
            )
        return MappingProxyType(dict(frozen))
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json(item, _path=f"{_path}[{index}]", _depth=_depth + 1)
            for index, item in enumerate(value)
        )
    raise ValueError(f"unsupported non-JSON value at {_path}: {type(value).__name__}")


def thaw_json(value: JSONValue) -> Any:
    """Return mutable JSON-native containers for serialization."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw_json(item) for item in value]
    return value


def _freeze_mapping(value: Mapping[str, Any] | None, *, path: str) -> Mapping[str, JSONValue]:
    frozen = freeze_json(dict(value or {}), _path=path)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True, slots=True)
class ManifestLimits:
    """Resource limits used while parsing untrusted manifest bytes."""

    max_bytes: int = 4 * 1024 * 1024
    max_artifacts: int = 4096
    max_metadata_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("max_bytes", "max_artifacts", "max_metadata_bytes"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.max_metadata_bytes > self.max_bytes:
            raise ValueError("max_metadata_bytes cannot exceed max_bytes")


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Cross-cutting configuration consumed by manager and storage layers."""

    base_dir: Path = Path("src/checkpoints")
    default_format: CheckpointFormat = CheckpointFormat.TORCH
    allow_overwrite: bool = False
    verify_on_load: bool = True
    require_exact_files: bool = True
    durable_writes: bool = True
    lock_timeout_seconds: float = 30.0
    lock_poll_interval_seconds: float = 0.05
    stale_lock_seconds: float = 3600.0
    file_mode: int = 0o600
    directory_mode: int = 0o700
    manifest_limits: ManifestLimits = field(default_factory=ManifestLimits)

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_dir", Path(self.base_dir).expanduser())
        object.__setattr__(self, "default_format", normalize_format(self.default_format))
        if self.lock_timeout_seconds < 0:
            raise ValueError("lock_timeout_seconds cannot be negative")
        if self.lock_poll_interval_seconds <= 0:
            raise ValueError("lock_poll_interval_seconds must be greater than zero")
        if self.stale_lock_seconds <= 0:
            raise ValueError("stale_lock_seconds must be greater than zero")
        for name in ("file_mode", "directory_mode"):
            mode = getattr(self, name)
            if not isinstance(mode, int) or not 0 <= mode <= 0o777:
                raise ValueError(f"{name} must be an integer permission mode between 0 and 0o777")


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    algorithm: str
    value: str

    def __post_init__(self) -> None:
        algorithm = self.algorithm.strip().lower()
        value = self.value.strip().lower()
        if algorithm != "sha256":
            raise ValueError("manifest schema v3 supports only sha256 artifact digests")
        if not _HEX_64_RE.fullmatch(value):
            raise ValueError("sha256 digest must contain exactly 64 lowercase hexadecimal characters")
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "value", value)


@dataclass(frozen=True, slots=True)
class CheckpointFileInfo:
    """Backward-compatible file integrity view used by v1/v2 callers."""

    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError("size_bytes cannot be negative")
        digest = self.sha256.strip().lower()
        if not _HEX_64_RE.fullmatch(digest):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True, slots=True)
class ObservedArtifact:
    relative_path: str
    size_bytes: int
    digest: ArtifactDigest

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", validate_relative_path(self.relative_path))
        if self.size_bytes < 0:
            raise ValueError("size_bytes cannot be negative")


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    """An immutable manifest entry for one checkpoint file."""

    relative_path: str
    component: str
    size_bytes: int
    digest: ArtifactDigest
    codec: str | None = None
    codec_version: str | None = None
    media_type: str = "application/octet-stream"
    required: bool = True
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", validate_relative_path(self.relative_path))
        object.__setattr__(self, "component", validate_component_name(self.component))
        if self.size_bytes < 0:
            raise ValueError("artifact size_bytes cannot be negative")
        if self.codec is not None:
            object.__setattr__(self, "codec", validate_identifier(self.codec, field_name="codec"))
        if self.codec_version is not None and not str(self.codec_version).strip():
            raise ValueError("codec_version cannot be empty")
        if not isinstance(self.media_type, str) or "/" not in self.media_type:
            raise ValueError("media_type must be a non-empty MIME type")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, path="$.artifact.metadata"))

    @property
    def sha256(self) -> str:
        return self.digest.value


@dataclass(frozen=True, slots=True)
class OmittedComponent:
    component: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", validate_component_name(self.component))
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("omitted component reason must be a non-empty string")
        object.__setattr__(self, "reason", self.reason.strip())


@dataclass(frozen=True, slots=True)
class PrimaryMetric:
    name: str
    value: float
    direction: MetricDirection
    step: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_component_name(self.name))
        numeric = float(self.value)
        if not math.isfinite(numeric):
            raise ValueError("primary metric value must be finite")
        object.__setattr__(self, "value", numeric)
        object.__setattr__(self, "direction", MetricDirection(self.direction))
        if self.step is not None and self.step < 0:
            raise ValueError("primary metric step cannot be negative")


@dataclass(frozen=True, slots=True)
class CheckpointLineage:
    parent_checkpoint_id: str | None = None
    restored_from_checkpoint_id: str | None = None
    generation: int = 0

    def __post_init__(self) -> None:
        for name in ("parent_checkpoint_id", "restored_from_checkpoint_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, validate_identifier(value, field_name=name))
        if self.generation < 0:
            raise ValueError("lineage generation cannot be negative")


@dataclass(frozen=True, slots=True)
class CheckpointProvenance:
    agent_id: str | None = None
    run_id: str | None = None
    trace_id: str | None = None
    checkpoint_reason: str | None = None
    code_revision: str | None = None
    config_digest: ArtifactDigest | None = None
    model_signature: ArtifactDigest | None = None
    objective: PrimaryMetric | None = None
    lineage: CheckpointLineage = field(default_factory=CheckpointLineage)
    requested_components: tuple[str, ...] = ()
    omitted_components: tuple[OmittedComponent, ...] = ()

    def __post_init__(self) -> None:
        for name in ("agent_id", "run_id", "trace_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, validate_identifier(value, field_name=name))
        for name in ("checkpoint_reason", "code_revision"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} cannot be empty")
                object.__setattr__(self, name, value.strip())
        requested = tuple(
            validate_component_name(component)
            for component in self.requested_components
        )
        if len(requested) != len(set(requested)):
            raise ValueError("requested_components cannot contain duplicates")
        object.__setattr__(self, "requested_components", requested)

        omitted = tuple(self.omitted_components)
        names = [item.component for item in omitted]
        if len(names) != len(set(names)):
            raise ValueError("omitted_components cannot contain duplicate component names")
        if omitted and not requested:
            raise ValueError(
                "omitted_components requires an explicit requested_components capture plan"
            )
        object.__setattr__(self, "omitted_components", omitted)


@dataclass(frozen=True, slots=True)
class ProducerInfo:
    name: str
    version: str | None = None
    environment: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("producer name must be a non-empty string")
        object.__setattr__(self, "name", self.name.strip())
        if self.version is not None and not str(self.version).strip():
            raise ValueError("producer version cannot be empty")
        object.__setattr__(self, "environment", _freeze_mapping(self.environment, path="$.producer.environment"))


@dataclass(frozen=True, slots=True)
class CompatibilityConstraints:
    """Declarative constraints evaluated by policy/manager before loading."""

    slai_version: str | None = None
    python_version: str | None = None
    platform_tags: tuple[str, ...] = ()
    required_codecs: Mapping[str, str] = field(default_factory=dict)
    component_schemas: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("slai_version", "python_version"):
            value = getattr(self, name)
            if value is not None and not str(value).strip():
                raise ValueError(f"{name} cannot be empty")
        tags = tuple(str(tag).strip() for tag in self.platform_tags)
        if any(not tag for tag in tags):
            raise ValueError("platform_tags cannot contain empty values")
        object.__setattr__(self, "platform_tags", tags)

        codecs: dict[str, str] = {}
        for name, version in self.required_codecs.items():
            safe_name = validate_identifier(name, field_name="codec")
            if not str(version).strip():
                raise ValueError(f"required codec version for {name!r} cannot be empty")
            codecs[safe_name] = str(version).strip()
        object.__setattr__(self, "required_codecs", MappingProxyType(codecs))

        schemas: dict[str, int] = {}
        for component, schema_version in self.component_schemas.items():
            safe_component = validate_component_name(component)
            version_value = int(schema_version)
            if version_value < 1:
                raise ValueError("component schema versions must be positive integers")
            schemas[safe_component] = version_value
        object.__setattr__(self, "component_schemas", MappingProxyType(schemas))


def _digest_dict(value: ArtifactDigest) -> dict[str, Any]:
    return {"algorithm": value.algorithm, "value": value.value}


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """Schema-v3 durable checkpoint manifest.

    ``path`` is intentionally absent: storage locations are runtime concerns and
    must not make otherwise identical manifests host-dependent.
    """

    checkpoint_id: str
    version: str
    created_at: str
    checkpoint_format: CheckpointFormat
    artifacts: tuple[CheckpointArtifact, ...]
    producer: ProducerInfo
    epoch: int | None = None
    step: int | None = None
    metrics: Mapping[str, JSONValue] = field(default_factory=dict)
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    provenance: CheckpointProvenance = field(default_factory=CheckpointProvenance)
    compatibility: CompatibilityConstraints = field(default_factory=CompatibilityConstraints)
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"CheckpointManifest represents schema {MANIFEST_SCHEMA_VERSION}; "
                f"received {self.schema_version}"
            )
        object.__setattr__(self, "checkpoint_id", validate_identifier(self.checkpoint_id, field_name="checkpoint_id"))
        object.__setattr__(self, "version", validate_version(self.version))
        object.__setattr__(self, "created_at", validate_utc_timestamp(self.created_at))
        object.__setattr__(self, "checkpoint_format", normalize_format(self.checkpoint_format))
        if self.epoch is not None and self.epoch < 0:
            raise ValueError("epoch cannot be negative")
        if self.step is not None and self.step < 0:
            raise ValueError("step cannot be negative")

        artifacts = tuple(self.artifacts)
        if not artifacts:
            raise ValueError("a committed checkpoint manifest must contain at least one artifact")
        paths = [artifact.relative_path for artifact in artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("manifest contains duplicate artifact paths")
        saved = {artifact.component for artifact in artifacts}
        omitted = {item.component for item in self.provenance.omitted_components}
        overlap = saved & omitted
        if overlap:
            raise ValueError(
                "components cannot be both saved and omitted: " + ", ".join(sorted(overlap))
            )
        requested = set(self.provenance.requested_components)
        if requested:
            accounted_for = saved | omitted
            if requested != accounted_for:
                missing = sorted(requested - accounted_for)
                unrequested = sorted(accounted_for - requested)
                raise ValueError(
                    "capture plan does not exactly account for requested components; "
                    f"unaccounted={missing}, unrequested={unrequested}"
                )
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "metrics", _freeze_mapping(self.metrics, path="$.metrics"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, path="$.metadata"))

    @property
    def format(self) -> str:
        """Backward-compatible string representation of checkpoint format."""

        return self.checkpoint_format.value

    @property
    def saved_components(self) -> tuple[str, ...]:
        return tuple(sorted({artifact.component for artifact in self.artifacts}))

    @property
    def files(self) -> Mapping[str, CheckpointFileInfo]:
        """Backward-compatible read-only file map."""

        return MappingProxyType(
            {
                artifact.relative_path: CheckpointFileInfo(
                    size_bytes=artifact.size_bytes,
                    sha256=artifact.digest.value,
                )
                for artifact in self.artifacts
            }
        )

    @property
    def tokenizer_kind(self) -> str | None:
        for artifact in self.artifacts:
            if artifact.component == StandardComponent.TOKENIZER.value:
                return artifact.codec
        return None

    @property
    def framework(self) -> Mapping[str, JSONValue]:
        """Backward-compatible view of producer environment information."""

        return self.producer.environment

    def to_json_dict(self) -> dict[str, Any]:
        """Return the canonical schema-v3 JSON object."""

        provenance: dict[str, Any] = {
            "lineage": {
                "generation": self.provenance.lineage.generation,
            },
            "omitted_components": [
                {"component": item.component, "reason": item.reason}
                for item in self.provenance.omitted_components
            ],
            "requested_components": list(self.provenance.requested_components),
        }
        for name in (
            "agent_id",
            "run_id",
            "trace_id",
            "checkpoint_reason",
            "code_revision",
        ):
            value = getattr(self.provenance, name)
            if value is not None:
                provenance[name] = value
        lineage = provenance["lineage"]
        if self.provenance.lineage.parent_checkpoint_id is not None:
            lineage["parent_checkpoint_id"] = self.provenance.lineage.parent_checkpoint_id
        if self.provenance.lineage.restored_from_checkpoint_id is not None:
            lineage["restored_from_checkpoint_id"] = (
                self.provenance.lineage.restored_from_checkpoint_id
            )
        if self.provenance.config_digest is not None:
            provenance["config_digest"] = _digest_dict(self.provenance.config_digest)
        if self.provenance.model_signature is not None:
            provenance["model_signature"] = _digest_dict(self.provenance.model_signature)
        if self.provenance.objective is not None:
            objective = {
                "name": self.provenance.objective.name,
                "value": self.provenance.objective.value,
                "direction": self.provenance.objective.direction.value,
            }
            if self.provenance.objective.step is not None:
                objective["step"] = self.provenance.objective.step
            provenance["objective"] = objective

        producer: dict[str, Any] = {
            "name": self.producer.name,
            "environment": thaw_json(self.producer.environment),
        }
        if self.producer.version is not None:
            producer["version"] = self.producer.version

        compatibility: dict[str, Any] = {
            "platform_tags": list(self.compatibility.platform_tags),
            "required_codecs": dict(self.compatibility.required_codecs),
            "component_schemas": dict(self.compatibility.component_schemas),
        }
        if self.compatibility.slai_version is not None:
            compatibility["slai_version"] = self.compatibility.slai_version
        if self.compatibility.python_version is not None:
            compatibility["python_version"] = self.compatibility.python_version

        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "checkpoint_id": self.checkpoint_id,
            "version": self.version,
            "created_at": self.created_at,
            "format": self.checkpoint_format.value,
            "artifacts": [
                {
                    "path": artifact.relative_path,
                    "component": artifact.component,
                    "size_bytes": artifact.size_bytes,
                    "digest": _digest_dict(artifact.digest),
                    "codec": artifact.codec,
                    "codec_version": artifact.codec_version,
                    "media_type": artifact.media_type,
                    "required": artifact.required,
                    "metadata": thaw_json(artifact.metadata),
                }
                for artifact in self.artifacts
            ],
            "producer": producer,
            "metrics": thaw_json(self.metrics),
            "metadata": thaw_json(self.metadata),
            "provenance": provenance,
            "compatibility": compatibility,
        }
        if self.epoch is not None:
            result["epoch"] = self.epoch
        if self.step is not None:
            result["step"] = self.step
        return result


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    code: str
    message: str
    relative_path: str | None = None
    expected: JSONValue = None
    actual: JSONValue = None

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError("verification issue code and message must be non-empty")
        if self.relative_path is not None:
            object.__setattr__(self, "relative_path", validate_relative_path(self.relative_path))
        object.__setattr__(self, "expected", freeze_json(self.expected, _path="$.issue.expected"))
        object.__setattr__(self, "actual", freeze_json(self.actual, _path="$.issue.actual"))


@dataclass(frozen=True, slots=True)
class VerificationResult:
    status: VerificationStatus
    health: CheckpointHealth
    issues: tuple[VerificationIssue, ...] = ()
    observed_artifacts: tuple[ObservedArtifact, ...] = ()
    verified_at: str = field(default_factory=utc_now_iso)
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", VerificationStatus(self.status))
        object.__setattr__(self, "health", CheckpointHealth(self.health))
        object.__setattr__(self, "issues", tuple(self.issues))
        object.__setattr__(self, "observed_artifacts", tuple(self.observed_artifacts))
        object.__setattr__(self, "verified_at", validate_utc_timestamp(self.verified_at))
        if self.duration_seconds < 0:
            raise ValueError("verification duration_seconds cannot be negative")
        if self.status is VerificationStatus.PASSED and self.issues:
            raise ValueError("a passed verification result cannot contain issues")
        if self.status is VerificationStatus.FAILED and not self.issues:
            raise ValueError("a failed verification result must contain at least one issue")

    @property
    def ok(self) -> bool:
        return self.status is VerificationStatus.PASSED


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """Runtime association between an immutable manifest and storage path."""

    manifest: CheckpointManifest
    path: Path
    health: CheckpointHealth = CheckpointHealth.UNKNOWN
    verification: VerificationResult | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "health", CheckpointHealth(self.health))
        if self.verification is not None and self.health is CheckpointHealth.UNKNOWN:
            object.__setattr__(self, "health", self.verification.health)

    @property
    def checkpoint_id(self) -> str:
        return self.manifest.checkpoint_id

    @property
    def version(self) -> str:
        return self.manifest.version

    @property
    def format(self) -> str:
        return self.manifest.format

    @property
    def created_at(self) -> str:
        return self.manifest.created_at

    @property
    def files(self) -> Mapping[str, CheckpointFileInfo]:
        return self.manifest.files

    @property
    def epoch(self) -> int | None:
        return self.manifest.epoch

    @property
    def step(self) -> int | None:
        return self.manifest.step

    @property
    def metrics(self) -> Mapping[str, JSONValue]:
        return self.manifest.metrics

    @property
    def metadata(self) -> Mapping[str, JSONValue]:
        return self.manifest.metadata

    @property
    def framework(self) -> Mapping[str, JSONValue]:
        return self.manifest.framework

    @property
    def tokenizer_kind(self) -> str | None:
        return self.manifest.tokenizer_kind

    @property
    def schema_version(self) -> int:
        return self.manifest.schema_version

    def to_json_dict(self) -> dict[str, Any]:
        return self.manifest.to_json_dict()


@dataclass(frozen=True, slots=True)
class StagingArea:
    version: str
    transaction_id: str
    path: Path
    final_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", validate_version(self.version))
        object.__setattr__(self, "transaction_id", validate_identifier(self.transaction_id, field_name="transaction_id"))
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "final_path", Path(self.final_path))


@dataclass(frozen=True, slots=True)
class SaveResult:
    record: CheckpointRecord
    committed: bool
    bytes_written: int
    duration_seconds: float
    archive_path: Path | None = None

    def __post_init__(self) -> None:
        if self.bytes_written < 0 or self.duration_seconds < 0:
            raise ValueError("save result sizes and durations cannot be negative")
        if self.archive_path is not None:
            object.__setattr__(self, "archive_path", Path(self.archive_path))


@dataclass(frozen=True, slots=True)
class LoadResult:
    record: CheckpointRecord
    components: Mapping[str, Any]
    loaded_components: tuple[str, ...]
    skipped_components: tuple[str, ...] = ()
    restored_rng: bool = False
    duration_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        object.__setattr__(self, "loaded_components", tuple(validate_component_name(name) for name in self.loaded_components))
        object.__setattr__(self, "skipped_components", tuple(validate_component_name(name) for name in self.skipped_components))
        if self.duration_seconds < 0:
            raise ValueError("load duration_seconds cannot be negative")


@dataclass(frozen=True, slots=True)
class CodecContext:
    checkpoint_id: str
    version: str
    component: str
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_id", validate_identifier(self.checkpoint_id, field_name="checkpoint_id"))
        object.__setattr__(self, "version", validate_version(self.version))
        object.__setattr__(self, "component", validate_component_name(self.component))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, path="$.codec_context.metadata"))


@dataclass(frozen=True, slots=True)
class CodecOutput:
    """A codec-declared file; storage derives its size and digest."""

    path: Path
    media_type: str = "application/octet-stream"
    required: bool = True
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not self.media_type or "/" not in self.media_type:
            raise ValueError("codec output media_type must be a MIME type")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, path="$.codec_output.metadata"))


T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class CheckpointCodec(Protocol[T_co]):
    """Structural contract implemented by codec adapters."""

    @property
    def codec_id(self) -> str: ...

    @property
    def codec_version(self) -> str: ...

    def supports(self, component: str, value: Any | None = None) -> bool: ...

    def encode(self, value: Any, destination: Path, *, context: CodecContext) -> Sequence[CodecOutput]: ...

    def decode(self, source: Path, *, context: CodecContext) -> T_co: ...


@runtime_checkable
class CheckpointStorageAdapter(Protocol):
    """Minimum transactional storage contract required by the manager."""

    @property
    def base_dir(self) -> Path: ...

    def begin(self, version: str, *, allow_overwrite: bool = False) -> StagingArea: ...

    def commit(self, staging: StagingArea, *, allow_overwrite: bool = False) -> Path: ...

    def abort(self, staging: StagingArea) -> None: ...


__all__ = [
    "ArtifactDigest",
    "CheckpointArtifact",
    "CheckpointCodec",
    "CheckpointConfig",
    "CheckpointFileInfo",
    "CheckpointFormat",
    "CheckpointHealth",
    "CheckpointLineage",
    "CheckpointManifest",
    "CheckpointProvenance",
    "CheckpointRecord",
    "CheckpointStorageAdapter",
    "CodecContext",
    "CodecOutput",
    "CompatibilityConstraints",
    "JSONScalar",
    "JSONValue",
    "LEGACY_TORCH_CHECKPOINT_NAME",
    "LoadResult",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA_VERSION",
    "METADATA_NAME",
    "ManifestLimits",
    "MetricDirection",
    "NPZ_WEIGHTS_NAME",
    "ObservedArtifact",
    "OmittedComponent",
    "PrimaryMetric",
    "ProducerInfo",
    "SCHEMA_VERSION",
    "SUPPORTED_MANIFEST_SCHEMA_VERSIONS",
    "SaveResult",
    "StagingArea",
    "StandardComponent",
    "TOKENIZER_DIR_NAME",
    "TOKENIZER_STATE_NAME",
    "TOKENIZER_VOCAB_NAME",
    "TORCH_CHECKPOINT_NAME",
    "VerificationIssue",
    "VerificationResult",
    "VerificationStatus",
    "freeze_json",
    "normalize_format",
    "thaw_json",
    "utc_now_iso",
    "validate_component_name",
    "validate_identifier",
    "validate_relative_path",
    "validate_utc_timestamp",
    "validate_version",
]