"""Lazy, security-conscious PyTorch state codec.

PyTorch is imported only when an encode, decode, or restore operation actually
requires it.  Decoding uses ``weights_only=True`` by default and never falls
back to unrestricted pickle loading unless both the codec configuration and
the individual decode context explicitly allow that trust decision.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from ..checkpoint_errors import *
from ..checkpoint_types import *
from .base import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]


logger = get_logger("Checkpoint Codec Torch")
printer = PrettyPrinter()


_ENVELOPE_MARKER = "slai.checkpoint.torch-state"
_SCHEMA_VERSION = 1
_ENVELOPE_FIELDS = {
    "__format__",
    "schema_version",
    "codec_version",
    "component",
    "state",
}


def _normalize_state_key(value: Any) -> Any:
    """Return a mapping key accepted by PyTorch's weights-only unpickler."""

    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if isinstance(value, tuple):
        return tuple(_normalize_state_key(item) for item in value)
    value_type = type(value)
    raise TypeError(
        "torch state contains a mapping key that is not weights-only compatible: "
        f"{value_type.__module__}.{value_type.__qualname__}"
    )


def _normalize_state_value(
    value: Any,
    torch_module: Any,
    *,
    move_to_cpu: bool,
    seen: set[int] | None = None,
) -> Any:
    if torch_module.is_tensor(value):
        detached = value.detach()
        return detached.cpu() if move_to_cpu else detached
    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise ValueError("cyclic mappings cannot be serialized as torch state")
        seen.add(identity)
        try:
            normalized: dict[Any, Any] = {}
            for key, item in value.items():
                safe_key = _normalize_state_key(key)
                if safe_key in normalized:
                    raise ValueError(
                        "torch state contains duplicate keys after normalization"
                    )
                normalized[safe_key] = _normalize_state_value(
                    item,
                    torch_module,
                    move_to_cpu=move_to_cpu,
                    seen=seen,
                )
            return normalized
        finally:
            seen.remove(identity)
    if isinstance(value, tuple):
        identity = id(value)
        if identity in seen:
            raise ValueError("cyclic tuples cannot be serialized as torch state")
        seen.add(identity)
        try:
            return tuple(
                _normalize_state_value(
                    item,
                    torch_module,
                    move_to_cpu=move_to_cpu,
                    seen=seen,
                )
                for item in value
            )
        finally:
            seen.remove(identity)
    if isinstance(value, list):
        identity = id(value)
        if identity in seen:
            raise ValueError("cyclic lists cannot be serialized as torch state")
        seen.add(identity)
        try:
            return [
                _normalize_state_value(
                    item,
                    torch_module,
                    move_to_cpu=move_to_cpu,
                    seen=seen,
                )
                for item in value
            ]
        finally:
            seen.remove(identity)
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    value_type = type(value)
    if value_type.__module__.startswith("torch") and value_type.__name__ in {
        "device",
        "dtype",
        "layout",
        "memory_format",
    }:
        return value
    raise TypeError(
        "torch state contains a value that is not weights-only compatible: "
        f"{value_type.__module__}.{value_type.__qualname__}"
    )


class TorchCheckpointCodec(BaseCheckpointCodec):
    """Serialize framework state mappings using PyTorch's checkpoint format."""

    def __init__(
        self,
        *,
        components: Sequence[str] = (
            StandardComponent.MODEL.value,
            StandardComponent.OPTIMIZER.value,
            StandardComponent.SCHEDULER.value,
            StandardComponent.SCALER.value,
        ),
        save_on_cpu: bool = True,
        allow_unsafe_pickle: bool = False,
        max_source_bytes: int | None = None,
        allow_legacy: bool = True,
    ) -> None:
        super().__init__("torch", "1", components)
        if not isinstance(save_on_cpu, bool):
            raise TypeError("save_on_cpu must be a boolean")
        if not isinstance(allow_unsafe_pickle, bool):
            raise TypeError("allow_unsafe_pickle must be a boolean")
        if not isinstance(allow_legacy, bool):
            raise TypeError("allow_legacy must be a boolean")
        if max_source_bytes is not None and (
            isinstance(max_source_bytes, bool)
            or not isinstance(max_source_bytes, int)
            or max_source_bytes <= 0
        ):
            raise ValueError("max_source_bytes must be a positive integer")
        self.save_on_cpu = save_on_cpu
        self.allow_unsafe_pickle = allow_unsafe_pickle
        self.max_source_bytes = max_source_bytes
        self.allow_legacy = allow_legacy

    def _supports_value(self, value: Any | None) -> bool:
        return value is None or isinstance(value, Mapping) or callable(
            getattr(value, "state_dict", None)
        )

    @staticmethod
    def _extract_state(value: Any) -> Mapping[str, Any]:
        state = value.state_dict() if callable(getattr(value, "state_dict", None)) else value
        if not isinstance(state, Mapping):
            raise TypeError("torch codec values must be mappings or expose state_dict()")
        if any(not isinstance(key, str) or not key for key in state):
            raise ValueError("torch state keys must be non-empty strings")
        return state

    def _encode(
        self,
        value: Any,
        destination: Path,
        *,
        context: CodecContext,
    ) -> Sequence[CodecOutput]:
        torch_module = require_dependency("torch", codec_id=self.codec_id)
        state = self._extract_state(value)
        state = _normalize_state_value(
            state,
            torch_module,
            move_to_cpu=self.save_on_cpu,
        )
        envelope = {
            "__format__": _ENVELOPE_MARKER,
            "schema_version": _SCHEMA_VERSION,
            "codec_version": self.codec_version,
            "component": context.component,
            "state": state,
        }
        durable = metadata_bool(context, "durable", default=True)
        with atomic_output_path(destination, durable=durable) as temporary:
            torch_module.save(envelope, temporary)
            encoded_size = temporary.stat().st_size
            if (
                self.max_source_bytes is not None
                and encoded_size > self.max_source_bytes
            ):
                raise CheckpointCodecError(
                    "encoded torch checkpoint exceeds the configured source-byte limit",
                    operation=CheckpointOperation.SAVE,
                    stage=CheckpointStage.VALIDATION,
                    path=temporary,
                    details={
                        "actual_bytes": encoded_size,
                        "max_source_bytes": self.max_source_bytes,
                    },
                )
        return (
            CodecOutput(
                path=destination,
                media_type="application/vnd.pytorch",
                metadata={"weights_only_compatible": True},
            ),
        )

    def _torch_load(
        self,
        source: Path,
        *,
        context: CodecContext,
    ) -> Any:
        torch_module = require_dependency("torch", codec_id=self.codec_id)
        map_location = metadata_string(context, "map_location", default="cpu")
        unsafe_requested = metadata_bool(
            context,
            "allow_unsafe_pickle",
            default=False,
        )
        if unsafe_requested and not self.allow_unsafe_pickle:
            raise CheckpointCodecError(
                "unsafe torch pickle loading is disabled by codec configuration",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.VALIDATION,
                path=source,
                details={"codec_id": self.codec_id, "unsafe_requested": True},
            )

        if not unsafe_requested:
            try:
                return torch_module.load(
                    source,
                    map_location=map_location,
                    weights_only=True,
                )
            except TypeError as exc:
                raise CheckpointCodecError(
                    "installed PyTorch does not support safe weights-only loading",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.DESERIALIZATION,
                    path=source,
                    details={
                        "resolution": (
                            "upgrade PyTorch or explicitly configure trusted "
                            "pickle loading"
                        )
                    },
                ) from exc
            except Exception as exc:
                raise CheckpointCodecError(
                    "safe PyTorch weights-only decoding failed",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.DESERIALIZATION,
                    path=source,
                    details={
                        "error_type": type(exc).__name__,
                        "unsafe_fallback_attempted": False,
                    },
                ) from exc

        try:
            return torch_module.load(
                source,
                map_location=map_location,
                weights_only=False,
            )
        except TypeError:
            # This branch is permitted only after the caller and codec both
            # opted into trusted unrestricted pickle loading.
            return torch_module.load(source, map_location=map_location)

    def _decode(self, source: Path, *, context: CodecContext) -> Mapping[str, Any]:
        ensure_bounded_regular_file(source, max_bytes=self.max_source_bytes)
        payload = self._torch_load(source, context=context)
        if not isinstance(payload, Mapping):
            raise CheckpointCodecError(
                "torch checkpoint payload must be a mapping",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        if payload.get("__format__") != _ENVELOPE_MARKER:
            if "__format__" in payload:
                raise CheckpointCodecError(
                    "torch checkpoint has an unexpected format marker",
                    stage=CheckpointStage.DESERIALIZATION,
                    path=source,
                    details={"actual": payload.get("__format__")},
                )
            if not self.allow_legacy:
                raise CheckpointCodecError(
                    "legacy torch payload lacks a codec envelope",
                    stage=CheckpointStage.DESERIALIZATION,
                    path=source,
                )
            # Explicit legacy support: v2.2 saved mapping payloads directly.
            if any(not isinstance(key, str) or not key for key in payload):
                raise CheckpointCodecError(
                    "legacy torch state keys must be non-empty strings",
                    stage=CheckpointStage.DESERIALIZATION,
                    path=source,
                )
            return payload
        if set(payload) != _ENVELOPE_FIELDS:
            raise CheckpointCodecError(
                "torch checkpoint envelope has an invalid field set",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={
                    "missing": sorted(_ENVELOPE_FIELDS - set(payload)),
                    "unknown": sorted(set(payload) - _ENVELOPE_FIELDS),
                },
            )
        if (
            payload.get("schema_version") != _SCHEMA_VERSION
            or payload.get("codec_version") != self.codec_version
        ):
            raise CheckpointCodecError(
                "unsupported torch codec payload version",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={
                    "expected_schema_version": _SCHEMA_VERSION,
                    "actual_schema_version": payload.get("schema_version"),
                    "expected_codec_version": self.codec_version,
                    "actual_codec_version": payload.get("codec_version"),
                },
            )
        if payload.get("component") != context.component:
            raise CheckpointCodecError(
                "torch payload component does not match decode context",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={
                    "expected": context.component,
                    "actual": payload.get("component"),
                },
            )
        state = payload.get("state")
        if not isinstance(state, Mapping):
            raise CheckpointCodecError(
                "torch payload state must be a mapping",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        if any(not isinstance(key, str) or not key for key in state):
            raise CheckpointCodecError(
                "torch payload state keys must be non-empty strings",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        return state

    def restore(
        self,
        target: Any,
        state: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> StateLoadReport:
        """Apply decoded state through a target's ``load_state_dict`` method."""

        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean")
        loader = getattr(target, "load_state_dict", None)
        if not callable(loader):
            raise TypeError("torch state target must expose load_state_dict()")
        if not isinstance(state, Mapping):
            raise TypeError("decoded torch state must be a mapping")
        try:
            result = loader(state, strict=strict)
        except Exception as exc:
            raise CheckpointIncompatibleError(
                "decoded torch state is incompatible with the target",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.DESERIALIZATION,
                details={"error_type": type(exc).__name__, "strict": strict},
            ) from exc
        missing = tuple(
            sorted(str(item) for item in getattr(result, "missing_keys", ()))
        )
        unexpected = tuple(
            sorted(str(item) for item in getattr(result, "unexpected_keys", ()))
        )
        unexpected_set = set(unexpected)
        loaded = tuple(
            sorted(
                key
                for key in state
                if isinstance(key, str) and key not in unexpected_set
            )
        )
        report = StateLoadReport(
            loaded_keys=loaded,
            missing_keys=missing,
            unexpected_keys=unexpected,
        )
        if strict and not report.compatible:
            raise CheckpointIncompatibleError(
                "decoded torch state is incompatible with the target",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.DESERIALIZATION,
                missing_keys=list(report.missing_keys),
                unexpected_keys=list(report.unexpected_keys),
            )
        return report


__all__ = ["TorchCheckpointCodec"]

if __name__ == "__main__":
    print("\n=== Running Torch State Codec Comprehensive Self-Test ===\n")
    import tempfile

    printer.status("TEST", "Starting torch codec tests", "info")

    # Skip if torch not installed
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        printer.status("SKIP", "PyTorch not installed, skipping torch tests", "warning")
        print("\n=== All torch tests skipped ===\n")
        sys.exit(0)

    codec = TorchCheckpointCodec()
    printer.status("CODEC", f"created {codec.codec_id} v{codec.codec_version}", "success")

    # supports
    assert codec.supports("model") is True
    assert codec.supports("model", value={"a": torch.zeros(2)}) is True
    assert codec.supports("unknown") is False

    # 1. Encode/decode a simple state dict
    state = {"linear.weight": torch.randn(3, 4), "linear.bias": torch.randn(3)}
    ctx = CodecContext(checkpoint_id="test", version="v1", component="model")
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        path = Path(tmp.name)
        outputs = codec.encode(state, path, context=ctx)
        assert len(outputs) == 1
        decoded = codec.decode(path, context=ctx)
        # Compare tensors
        for key in state:
            torch.testing.assert_close(decoded[key], state[key])
    printer.status("ROUNDTRIP", "encode/decode of state dict passed", "success")

    # 2. Test restore with a simple model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)

    model = DummyModel()
    # Save current state, change weights, restore, compare
    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    # modify
    with torch.no_grad():
        model.linear.weight += 1
    report = codec.restore(model, original_state, strict=True)
    assert report.compatible
    for k in original_state:
        torch.testing.assert_close(model.state_dict()[k], original_state[k])
    printer.status("RESTORE", "restore into model passed", "success")

    # 3. Error: incompatible state (wrong shape)
    bad_state = {"linear.weight": torch.randn(2, 2)}
    try:
        codec.restore(model, bad_state, strict=True)
        assert False, "Should have raised"
    except CheckpointIncompatibleError as e:
        assert "incompatible" in str(e)
    printer.status("ERRORS", "shape mismatch detection works", "success")

    print("\n=== All torch tests passed ===\n")