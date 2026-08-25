"""Thread-safe deterministic checkpoint codec registry.

The registry never imports plugins, scans entry points, or mutates a global
singleton at import time.  Applications explicitly register codec instances or
construct a fresh default registry through ``checkpoint_codecs.py``.
"""

from __future__ import annotations

import threading

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from ..checkpoint_errors import *
from ..checkpoint_types import *
from .base import is_codec
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]


logger = get_logger("Checkpoint Codec Registry")
printer = PrettyPrinter()


@dataclass(frozen=True, slots=True)
class CodecDescriptor:
    """Immutable registry-facing identity for one codec instance."""

    codec_id: str
    codec_version: str
    declared_components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "codec_id",
            validate_identifier(self.codec_id, field_name="codec_id"),
        )
        if not isinstance(self.codec_version, str) or not self.codec_version.strip():
            raise ValueError("codec_version must be a non-empty string")
        object.__setattr__(self, "codec_version", self.codec_version.strip())
        if isinstance(self.declared_components, (str, bytes)):
            raise TypeError("declared_components must be a sequence")
        components = tuple(dict.fromkeys(
            validate_component_name(component)
            for component in self.declared_components
            )
        )
        object.__setattr__(self, "declared_components", components)
        logger.info("Codec Descriptor initialized with", components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "codec_id": self.codec_id,
            "codec_version": self.codec_version,
            "declared_components": list(self.declared_components),
        }


@dataclass(frozen=True, slots=True)
class CodecRegistrySnapshot:
    """Immutable diagnostic view of registry identities and component defaults."""

    codecs: Mapping[str, CodecDescriptor]
    defaults: Mapping[str, str]
    frozen: bool

    def __post_init__(self) -> None:
        codecs = dict(self.codecs)
        defaults = dict(self.defaults)
        if any(key != value.codec_id for key, value in codecs.items()):
            raise ValueError("codec descriptor keys must match codec identities")
        for component, codec_id in defaults.items():
            validate_component_name(component)
            if codec_id not in codecs:
                raise ValueError("registry default references an unavailable codec")
        if not isinstance(self.frozen, bool):
            raise TypeError("frozen must be a boolean")
        object.__setattr__(self, "codecs", MappingProxyType(codecs))
        object.__setattr__(self, "defaults", MappingProxyType(defaults))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frozen": self.frozen,
            "codecs": {
                codec_id: descriptor.to_dict()
                for codec_id, descriptor in self.codecs.items()
            },
            "defaults": dict(self.defaults),
        }


class CodecRegistry:
    """Explicit, thread-safe registry with auditable resolution semantics."""

    def __init__(self) -> None:
        self._codecs: dict[str, CheckpointCodec[Any]] = {}
        self._defaults: dict[str, str] = {}
        self._frozen = False
        self._lock = threading.RLock()

    @property
    def frozen(self) -> bool:
        with self._lock:
            return self._frozen

    def _assert_mutable(self) -> None:
        if self._frozen:
            raise CheckpointConfigurationError(
                "checkpoint codec registry is frozen",
                stage=CheckpointStage.VALIDATION,
                details={"reason": "registry_frozen"},
            )

    @staticmethod
    def _validate_codec(codec: CheckpointCodec[Any]) -> tuple[str, str]:
        if not is_codec(codec):
            raise TypeError("codec must expose codec_id, codec_version, supports, encode, and decode")
        codec_id = validate_identifier(codec.codec_id, field_name="codec_id")
        codec_version = codec.codec_version
        if not isinstance(codec_version, str) or not codec_version.strip():
            raise ValueError("codec_version must be a non-empty string")
        return codec_id, codec_version.strip()

    def register(self, codec: CheckpointCodec[Any], *,
        default_for: Iterable[str] = (), replace: bool = False) -> CodecDescriptor:
        """Register one codec and optionally assign component defaults.

        Replacement is never implicit.  Assigning a default also requires the
        codec to report support for that component without inspecting a value.
        """

        if not isinstance(replace, bool):
            raise TypeError("replace must be a boolean")
        if isinstance(default_for, (str, bytes)):
            raise TypeError("default_for must be an iterable of component names")
        codec_id, codec_version = self._validate_codec(codec)
        components = tuple(
            dict.fromkeys(validate_component_name(item) for item in default_for)
        )
        for component in components:
            if not codec.supports(component, None):
                raise CheckpointConfigurationError(
                    f"codec {codec_id!r} cannot be the default for "
                    f"unsupported component {component!r}",
                    stage=CheckpointStage.VALIDATION,
                    component=component,
                    details={"codec_id": codec_id},
                )

        with self._lock:
            self._assert_mutable()
            if codec_id in self._codecs and not replace:
                raise CheckpointConfigurationError(
                    f"codec {codec_id!r} is already registered",
                    stage=CheckpointStage.VALIDATION,
                    details={"codec_id": codec_id, "reason": "duplicate_codec"},
                )
            inherited_defaults = tuple(
                component
                for component, default_id in self._defaults.items()
                if default_id == codec_id and component not in components
            )
            unsupported_inherited = tuple(
                component
                for component in inherited_defaults
                if not codec.supports(component, None)
            )
            if unsupported_inherited:
                raise CheckpointConfigurationError(
                    "replacement codec does not support its inherited defaults",
                    stage=CheckpointStage.VALIDATION,
                    details={
                        "codec_id": codec_id,
                        "unsupported_defaults": list(unsupported_inherited),
                        "resolution": "clear those defaults before replacement",
                    },
                )
            for component in components:
                existing = self._defaults.get(component)
                if existing is not None and existing != codec_id and not replace:
                    raise CheckpointConfigurationError(
                        f"component {component!r} already defaults to codec "
                        f"{existing!r}",
                        stage=CheckpointStage.VALIDATION,
                        component=component,
                        details={
                            "existing_codec_id": existing,
                            "requested_codec_id": codec_id,
                        },
                    )
            self._codecs[codec_id] = codec
            for component in components:
                self._defaults[component] = codec_id
            return self._descriptor(codec)

    def unregister(self, codec_id: str, *, missing_ok: bool = False) -> bool:
        """Remove a codec and all defaults that refer to it."""

        safe_id = validate_identifier(codec_id, field_name="codec_id")
        if not isinstance(missing_ok, bool):
            raise TypeError("missing_ok must be a boolean")
        with self._lock:
            self._assert_mutable()
            if safe_id not in self._codecs:
                if missing_ok:
                    return False
                raise CheckpointCodecNotFoundError(
                    f"codec {safe_id!r} is not registered",
                    stage=CheckpointStage.DISCOVERY,
                    details={"codec_id": safe_id},
                )
            del self._codecs[safe_id]
            self._defaults = {
                component: default_id
                for component, default_id in self._defaults.items()
                if default_id != safe_id
            }
            return True

    def set_default(self, component: str, codec_id: str) -> None:
        """Set an explicit default after verifying declared codec support."""

        safe_component = validate_component_name(component)
        safe_id = validate_identifier(codec_id, field_name="codec_id")
        with self._lock:
            self._assert_mutable()
            codec = self._codecs.get(safe_id)
            if codec is None:
                raise CheckpointCodecNotFoundError(
                    f"codec {safe_id!r} is not registered",
                    stage=CheckpointStage.DISCOVERY,
                    component=safe_component,
                    details={"codec_id": safe_id},
                )
            if not codec.supports(safe_component, None):
                raise CheckpointConfigurationError(
                    f"codec {safe_id!r} does not support component "
                    f"{safe_component!r}",
                    stage=CheckpointStage.VALIDATION,
                    component=safe_component,
                    details={"codec_id": safe_id},
                )
            self._defaults[safe_component] = safe_id

    def clear_default(self, component: str, *, missing_ok: bool = True) -> bool:
        """Remove one component default without unregistering its codec."""

        safe_component = validate_component_name(component)
        if not isinstance(missing_ok, bool):
            raise TypeError("missing_ok must be a boolean")
        with self._lock:
            self._assert_mutable()
            if safe_component not in self._defaults:
                if missing_ok:
                    return False
                raise CheckpointConfigurationError(
                    f"component {safe_component!r} has no default codec",
                    stage=CheckpointStage.DISCOVERY,
                    component=safe_component,
                )
            del self._defaults[safe_component]
            return True

    def get(self, codec_id: str) -> CheckpointCodec[Any]:
        """Return one codec by exact identity."""

        safe_id = validate_identifier(codec_id, field_name="codec_id")
        with self._lock:
            codec = self._codecs.get(safe_id)
        if codec is None:
            raise CheckpointCodecNotFoundError(
                f"codec {safe_id!r} is not registered",
                stage=CheckpointStage.DISCOVERY,
                details={
                    "codec_id": safe_id,
                    "available": list(self.available_codec_ids()),
                },
            )
        return codec

    def resolve(self, component: str, value: Any | None = None, *,
                codec_id: str | None = None, preferred: Sequence[str] = ()) -> CheckpointCodec[Any]:
        """Resolve a codec without registration-order-dependent behavior.

        Resolution order is explicit codec, caller preference order, component
        default, then a unique supporting codec.  Multiple remaining candidates
        are reported as ambiguous instead of being selected arbitrarily.
        """

        safe_component = validate_component_name(component)
        if codec_id is not None:
            codec = self.get(codec_id)
            if not codec.supports(safe_component, value):
                raise CheckpointCodecError(
                    f"codec {codec.codec_id!r} does not support component/value",
                    stage=CheckpointStage.VALIDATION,
                    component=safe_component,
                    details={"codec_id": codec.codec_id},
                )
            return codec

        if isinstance(preferred, (str, bytes)):
            raise TypeError("preferred must be a sequence of codec identifiers")
        preferred_ids = tuple(
            dict.fromkeys(
                validate_identifier(item, field_name="preferred_codec_id")
                for item in preferred
            )
        )
        with self._lock:
            codecs = dict(self._codecs)
            default_id = self._defaults.get(safe_component)

        for preferred_id in preferred_ids:
            candidate = codecs.get(preferred_id)
            if candidate is not None and candidate.supports(safe_component, value):
                return candidate

        if default_id is not None:
            candidate = codecs.get(default_id)
            if candidate is not None and candidate.supports(safe_component, value):
                return candidate

        candidates = tuple(
            codec
            for _, codec in sorted(codecs.items())
            if codec.supports(safe_component, value)
        )
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise CheckpointCodecNotFoundError(
                f"no registered codec supports component {safe_component!r}",
                stage=CheckpointStage.DISCOVERY,
                component=safe_component,
                details={"available": sorted(codecs)},
            )
        raise CheckpointCodecError(
            f"codec resolution for component {safe_component!r} is ambiguous",
            stage=CheckpointStage.DISCOVERY,
            component=safe_component,
            details={
                "candidates": [codec.codec_id for codec in candidates],
                "resolution": "set a component default or provide codec_id",
            },
        )

    def freeze(self) -> CodecRegistrySnapshot:
        """Prevent further mutation and return the frozen registry snapshot."""

        with self._lock:
            self._frozen = True
            return self.snapshot()

    def available_codec_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._codecs))

    def required_codecs(
        self,
        codec_ids: Iterable[str] | None = None,
    ) -> Mapping[str, str]:
        """Return exact codec versions suitable for manifest compatibility."""

        if isinstance(codec_ids, (str, bytes)):
            raise TypeError("codec_ids must be an iterable of codec identifiers")
        selected = (
            self.available_codec_ids()
            if codec_ids is None
            else tuple(
                dict.fromkeys(
                    validate_identifier(item, field_name="codec_id")
                    for item in codec_ids
                )
            )
        )
        return MappingProxyType(
            {codec_id: self.get(codec_id).codec_version for codec_id in selected}
        )

    @staticmethod
    def _descriptor(codec: CheckpointCodec[Any]) -> CodecDescriptor:
        declared = getattr(codec, "components", ())
        if not isinstance(declared, (tuple, list)):
            declared = ()
        return CodecDescriptor(
            codec_id=codec.codec_id,
            codec_version=codec.codec_version,
            declared_components=tuple(declared),
        )

    def snapshot(self) -> CodecRegistrySnapshot:
        with self._lock:
            descriptors = {
                codec_id: self._descriptor(codec)
                for codec_id, codec in sorted(self._codecs.items())
            }
            defaults = dict(sorted(self._defaults.items()))
            frozen = self._frozen
        return CodecRegistrySnapshot(descriptors, defaults, frozen)


__all__ = [
    "CodecDescriptor",
    "CodecRegistry",
    "CodecRegistrySnapshot",
]


if __name__ == "__main__":
    print("\n=== Running Checkpoint Codec Registry Comprehensive Self-Test ===\n")
    printer.status("TEST", "Starting registry tests", "info")

    # Import the dummy codec from base (or redefine a minimal one)
    from .base import BaseCheckpointCodec, CodecOutput, write_json_object

    class DummyCodec(BaseCheckpointCodec):
        def _encode(self, value, destination, *, context):
            write_json_object({"value": value}, destination, durable=False)
            return (CodecOutput(path=destination, media_type="application/json"),)

        def _decode(self, source, *, context):
            from .base import read_json_object
            return read_json_object(source, max_bytes=1024)

    registry = CodecRegistry()
    dummy = DummyCodec("dummy", "1.0", ("model", "optimizer"))

    # Register
    desc = registry.register(dummy, default_for=("model",))
    printer.status("REGISTER", f"registered {desc.codec_id} with defaults {desc.declared_components}", "success")

    # Get
    retrieved = registry.get("dummy")
    assert retrieved is dummy

    # Resolve by explicit id
    ctx = CodecContext(checkpoint_id="test", version="v1", component="model")
    resolved = registry.resolve("model", codec_id="dummy")
    assert resolved is dummy

    # Resolve via default
    resolved_default = registry.resolve("model")
    assert resolved_default is dummy

    # Resolve via fallback (no default set for other component, but only one supports)
    resolved_fallback = registry.resolve("optimizer")
    assert resolved_fallback is dummy

    # Set default and clear
    registry.set_default("optimizer", "dummy")
    assert registry.resolve("optimizer") is dummy
    registry.clear_default("optimizer")
    # Now it falls back to the unique candidate
    assert registry.resolve("optimizer") is dummy

    # Snapshot
    snapshot = registry.snapshot()
    assert snapshot.frozen is False
    assert "dummy" in snapshot.codecs
    assert snapshot.defaults.get("model") == "dummy"

    # Freeze
    frozen_snapshot = registry.freeze()
    assert registry.frozen is True
    assert frozen_snapshot.frozen is True

    printer.status("RESOLVE", "resolution and defaults work correctly", "success")
    print("\n=== All registry tests passed ===\n")