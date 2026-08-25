"""Pickle-free NumPy NPZ codec and explicit model-state restoration."""

from __future__ import annotations

import zipfile

from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

from ..checkpoint_errors import *
from ..checkpoint_types import *
from .base import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]


logger = get_logger("Checkpoint Codec Numpy")
printer = PrettyPrinter()


_FORMAT = "slai.checkpoint.numpy-state"
_SCHEMA_VERSION = 1
_METADATA_KEY = "__slai_codec_metadata_utf8"
_SUPPORTED_DTYPE_KINDS = frozenset({"b", "i", "u", "f", "c"})


def _validate_array_key(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or "\x00" in value
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(
            "NPZ array keys must be non-empty, at most 1024 characters, and "
            "must not contain path separators"
        )
    return value


def _to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        result = value
    elif isinstance(value, np.generic):
        result = np.asarray(value)
    elif all(callable(getattr(value, name, None)) for name in ("detach", "cpu", "numpy")):
        result = value.detach().cpu().numpy()
    else:
        result = np.asarray(value)
    if result.dtype.hasobject:
        raise TypeError("object-dtype arrays are not permitted in checkpoint NPZ files")
    if result.dtype.kind not in _SUPPORTED_DTYPE_KINDS:
        raise TypeError(
            f"unsupported checkpoint array dtype {result.dtype}; expected numeric or boolean"
        )
    return np.ascontiguousarray(result)


class NumpyCheckpointCodec(BaseCheckpointCodec):
    """Serialize named numeric arrays as bounded, pickle-free NPZ payloads."""

    def __init__(
        self,
        *,
        components: Sequence[str] = (StandardComponent.MODEL.value,),
        compressed: bool = True,
        max_arrays: int = 100_000,
        max_decoded_bytes: int | None = 16 * 1024 * 1024 * 1024,
        max_source_bytes: int | None = None,
        allow_legacy: bool = True,
    ) -> None:
        super().__init__("numpy", "1", components)
        if not isinstance(compressed, bool):
            raise TypeError("compressed must be a boolean")
        if isinstance(max_arrays, bool) or not isinstance(max_arrays, int) or max_arrays <= 0:
            raise ValueError("max_arrays must be a positive integer")
        for name, value in (
            ("max_decoded_bytes", max_decoded_bytes),
            ("max_source_bytes", max_source_bytes),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        self.compressed = compressed
        self.max_arrays = max_arrays
        self.max_decoded_bytes = max_decoded_bytes
        self.max_source_bytes = max_source_bytes
        if not isinstance(allow_legacy, bool):
            raise TypeError("allow_legacy must be a boolean")
        self.allow_legacy = allow_legacy

    def _supports_value(self, value: Any | None) -> bool:
        return (
            value is None
            or isinstance(value, (Mapping, np.ndarray, np.generic))
            or callable(getattr(value, "state_dict", None))
        )

    @staticmethod
    def _extract_arrays(value: Any) -> dict[str, np.ndarray]:
        state = value.state_dict() if callable(getattr(value, "state_dict", None)) else value
        if isinstance(state, (np.ndarray, np.generic)):
            return {"value": _to_numpy_array(state)}
        if not isinstance(state, Mapping):
            raise TypeError(
                "NumPy codec values must be arrays, mappings, or expose state_dict()"
            )
        arrays: dict[str, np.ndarray] = {}
        for key, item in state.items():
            safe_key = _validate_array_key(key)
            if safe_key == _METADATA_KEY:
                raise ValueError(f"NPZ array key {_METADATA_KEY!r} is reserved")
            if safe_key in arrays:
                raise ValueError("NumPy state contains duplicate array keys")
            arrays[safe_key] = _to_numpy_array(item)
        if not arrays:
            raise ValueError("NumPy codec cannot serialize an empty array mapping")
        return arrays

    def _encode(
        self,
        value: Any,
        destination: Path,
        *,
        context: CodecContext,
    ) -> Sequence[CodecOutput]:
        arrays = self._extract_arrays(value)
        if len(arrays) > self.max_arrays:
            raise CheckpointCodecError(
                "NumPy state contains too many arrays",
                stage=CheckpointStage.VALIDATION,
                details={"array_count": len(arrays), "max_arrays": self.max_arrays},
            )
        decoded_bytes = sum(array.nbytes for array in arrays.values())
        if self.max_decoded_bytes is not None and decoded_bytes > self.max_decoded_bytes:
            raise CheckpointCodecError(
                "NumPy state exceeds the configured decoded-byte limit",
                stage=CheckpointStage.VALIDATION,
                details={
                    "decoded_bytes": decoded_bytes,
                    "max_decoded_bytes": self.max_decoded_bytes,
                },
            )
        durable = metadata_bool(context, "durable", default=True)
        metadata = {
            "format": _FORMAT,
            "schema_version": _SCHEMA_VERSION,
            "codec_version": self.codec_version,
            "component": context.component,
        }
        serialized = {key: arrays[key] for key in sorted(arrays)}
        serialized[_METADATA_KEY] = np.frombuffer(
            encode_json_object(metadata),
            dtype=np.uint8,
        ).copy()
        with atomic_output_path(destination, durable=durable) as temporary:
            with temporary.open("wb") as handle:
                saver = np.savez_compressed if self.compressed else np.savez
                cast(Any, saver)(handle, **serialized)
            encoded_size = temporary.stat().st_size
            if (
                self.max_source_bytes is not None
                and encoded_size > self.max_source_bytes
            ):
                raise CheckpointCodecError(
                    "encoded NPZ exceeds the configured source-byte limit",
                    operation=CheckpointOperation.SAVE,
                    stage=CheckpointStage.VALIDATION,
                    path=temporary,
                    details={
                        "actual_bytes": encoded_size,
                        "max_source_bytes": self.max_source_bytes,
                    },
                )
            self._inspect_archive(temporary, operation=CheckpointOperation.SAVE)
        return (
            CodecOutput(
                path=destination,
                media_type="application/vnd.numpy.npz",
                metadata={
                    "array_count": len(arrays),
                    "decoded_bytes": decoded_bytes,
                    "compressed": self.compressed,
                    "allow_pickle": False,
                },
            ),
        )

    def _inspect_archive(
        self,
        source: Path,
        *,
        operation: CheckpointOperation = CheckpointOperation.LOAD,
    ) -> None:
        try:
            with zipfile.ZipFile(source, "r") as archive:
                members = archive.infolist()
        except (OSError, zipfile.BadZipFile) as exc:
            raise CheckpointCodecError(
                "invalid NPZ ZIP container",
                operation=operation,
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            ) from exc
        if not members or len(members) > self.max_arrays + 1:
            raise CheckpointCodecError(
                "NPZ member count is empty or exceeds the configured limit",
                stage=CheckpointStage.VALIDATION,
                path=source,
                details={
                    "member_count": len(members),
                    "max_arrays": self.max_arrays,
                },
            )
        total = 0
        member_names: set[str] = set()
        for member in members:
            name = member.filename
            if (
                member.is_dir()
                or not name.endswith(".npy")
                or "/" in name
                or "\\" in name
                or "\x00" in name
            ):
                raise CheckpointCodecError(
                    "NPZ archive contains an invalid member path",
                    stage=CheckpointStage.VALIDATION,
                    path=source,
                    details={"member": name},
                )
            if name in member_names:
                raise CheckpointCodecError(
                    "NPZ archive contains duplicate member names",
                    stage=CheckpointStage.VALIDATION,
                    path=source,
                    details={"member": name},
                )
            member_names.add(name)
            _validate_array_key(name[:-4])
            total += member.file_size
        if self.max_decoded_bytes is not None and total > self.max_decoded_bytes:
            raise CheckpointCodecError(
                "NPZ archive exceeds the configured expanded-byte limit",
                stage=CheckpointStage.VALIDATION,
                path=source,
                details={
                    "expanded_bytes": total,
                    "max_decoded_bytes": self.max_decoded_bytes,
                },
            )

    def _decode(self, source: Path, *, context: CodecContext) -> Mapping[str, np.ndarray]:
        ensure_bounded_regular_file(source, max_bytes=self.max_source_bytes)
        self._inspect_archive(source)
        arrays: dict[str, np.ndarray] = {}
        try:
            with np.load(source, allow_pickle=False) as payload:
                if len(payload.files) > self.max_arrays + 1:
                    raise CheckpointCodecError(
                        "NPZ array count exceeds the configured limit",
                        stage=CheckpointStage.VALIDATION,
                        path=source,
                    )
                decoded_bytes = 0
                for key in payload.files:
                    safe_key = _validate_array_key(key)
                    array = payload[key]
                    if array.dtype.hasobject:
                        raise CheckpointCodecError(
                            "object-dtype arrays are not permitted",
                            stage=CheckpointStage.DESERIALIZATION,
                            path=source,
                            details={"array": safe_key},
                        )
                    if array.dtype.kind not in _SUPPORTED_DTYPE_KINDS:
                        raise CheckpointCodecError(
                            "NPZ array has an unsupported dtype",
                            stage=CheckpointStage.DESERIALIZATION,
                            path=source,
                            details={
                                "array": safe_key,
                                "dtype": str(array.dtype),
                            },
                        )
                    decoded_bytes += array.nbytes
                    if (
                        self.max_decoded_bytes is not None
                        and decoded_bytes > self.max_decoded_bytes
                    ):
                        raise CheckpointCodecError(
                            "decoded NPZ arrays exceed the configured byte limit",
                            stage=CheckpointStage.VALIDATION,
                            path=source,
                            details={
                                "decoded_bytes": decoded_bytes,
                                "max_decoded_bytes": self.max_decoded_bytes,
                            },
                        )
                    arrays[safe_key] = np.array(array, copy=True)
        except CheckpointCodecError:
            raise
        except (OSError, ValueError, EOFError) as exc:
            raise CheckpointCodecError(
                "failed to decode pickle-free NPZ arrays",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={"error_type": type(exc).__name__},
            ) from exc
        metadata_array = arrays.pop(_METADATA_KEY, None)
        if metadata_array is None:
            if not self.allow_legacy:
                raise CheckpointCodecError(
                    "legacy NPZ payload lacks a codec metadata envelope",
                    stage=CheckpointStage.DESERIALIZATION,
                    path=source,
                )
            if not arrays:
                raise CheckpointCodecError(
                    "legacy NPZ payload contains no state arrays",
                    stage=CheckpointStage.DESERIALIZATION,
                    path=source,
                )
            return arrays
        if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
            raise CheckpointCodecError(
                "NumPy codec metadata must be a one-dimensional uint8 array",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        metadata = decode_json_object(metadata_array.tobytes(), source=source)
        expected_fields = {"format", "schema_version", "codec_version", "component"}
        if set(metadata) != expected_fields:
            raise CheckpointCodecError(
                "NumPy codec metadata has an invalid field set",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={
                    "missing": sorted(expected_fields - set(metadata)),
                    "unknown": sorted(set(metadata) - expected_fields),
                },
            )
        if metadata["format"] != _FORMAT:
            raise CheckpointCodecError(
                "NumPy payload has an unexpected format marker",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        if (
            metadata["schema_version"] != _SCHEMA_VERSION
            or metadata["codec_version"] != self.codec_version
        ):
            raise CheckpointCodecError(
                "unsupported NumPy codec payload version",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={
                    "schema_version": metadata["schema_version"],
                    "codec_version": metadata["codec_version"],
                },
            )
        if metadata["component"] != context.component:
            raise CheckpointCodecError(
                "NumPy payload component does not match decode context",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={
                    "actual": metadata["component"],
                    "expected": context.component,
                },
            )
        if not arrays:
            raise CheckpointCodecError(
                "NumPy payload contains no state arrays",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        return arrays

    def restore(
        self,
        target: Any,
        arrays: Mapping[str, np.ndarray],
        *,
        strict: bool = True,
        key_prefix: str | None = None,
    ) -> StateLoadReport:
        """Apply arrays to a torch-like target with shape compatibility evidence."""

        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean")
        if key_prefix is not None and (
            not isinstance(key_prefix, str) or not key_prefix
        ):
            raise ValueError("key_prefix must be a non-empty string")
        if not isinstance(arrays, Mapping):
            raise TypeError("decoded NumPy state must be a mapping")
        normalized_arrays: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            safe_name = _validate_array_key(name)
            normalized_arrays[safe_name] = _to_numpy_array(value)
        state_getter = getattr(target, "state_dict", None)
        state_loader = getattr(target, "load_state_dict", None)
        if not callable(state_getter) or not callable(state_loader):
            raise TypeError("NumPy restore target must expose state_dict/load_state_dict")
        torch_module = require_dependency("torch", codec_id=self.codec_id)
        try:
            raw_target_state = state_getter()
            if not isinstance(raw_target_state, Mapping):
                raise CheckpointIncompatibleError(
                    "NumPy restore target state must be a mapping",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.COMPATIBILITY,
                )
            target_state = dict(cast(Mapping[str, Any], raw_target_state))
        except Exception as exc:
            raise CheckpointIncompatibleError(
                "failed to inspect the NumPy restore target",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.COMPATIBILITY,
                details={"error_type": type(exc).__name__},
            ) from exc
        if any(not isinstance(name, str) or not name for name in target_state):
            raise CheckpointIncompatibleError(
                "NumPy restore target state keys must be non-empty strings",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.COMPATIBILITY,
            )
        scoped_targets = {
            name
            for name in target_state
            if key_prefix is None or name.startswith(key_prefix)
        }
        candidate_arrays = {
            name: array
            for name, array in normalized_arrays.items()
            if key_prefix is None or name.startswith(key_prefix)
        }
        loaded: list[str] = []
        unexpected: list[str] = []
        mismatches: list[ShapeMismatch] = []
        for name, array in candidate_arrays.items():
            if name not in scoped_targets:
                unexpected.append(name)
                continue
            target_value = target_state[name]
            checkpoint_shape = tuple(int(item) for item in array.shape)
            try:
                target_shape = tuple(int(item) for item in target_value.shape)
                target_dtype = target_value.dtype
                target_device = target_value.device
            except Exception as exc:
                raise CheckpointIncompatibleError(
                    "NumPy restore target contains a non-tensor state value",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.COMPATIBILITY,
                    details={"key": name, "error_type": type(exc).__name__},
                ) from exc
            if checkpoint_shape != target_shape:
                mismatches.append(
                    ShapeMismatch(
                        name=name,
                        checkpoint_shape=checkpoint_shape,
                        target_shape=target_shape,
                        checkpoint_dtype=str(array.dtype),
                        target_dtype=str(target_dtype),
                    )
                )
                continue
            try:
                target_state[name] = torch_module.as_tensor(
                    array,
                    dtype=target_dtype,
                    device=target_device,
                )
            except Exception as exc:
                raise CheckpointIncompatibleError(
                    "NumPy array could not be converted for the restore target",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.COMPATIBILITY,
                    details={"key": name, "error_type": type(exc).__name__},
                ) from exc
            loaded.append(name)
        mismatch_names = {item.name for item in mismatches}
        missing = sorted(scoped_targets - set(loaded) - mismatch_names)
        report = StateLoadReport(
            loaded_keys=tuple(sorted(loaded)),
            missing_keys=tuple(missing),
            unexpected_keys=tuple(sorted(unexpected)),
            mismatched_keys=tuple(sorted(mismatches, key=lambda item: item.name)),
        )
        if strict and not report.compatible:
            raise CheckpointIncompatibleError(
                "decoded NumPy state is incompatible with the target",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.DESERIALIZATION,
                missing_keys=list(report.missing_keys),
                unexpected_keys=list(report.unexpected_keys),
                mismatched_keys=[item.to_dict() for item in report.mismatched_keys],
            )
        try:
            result = state_loader(target_state, strict=False)
        except Exception as exc:
            raise CheckpointIncompatibleError(
                "converted NumPy state could not be applied to the target",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.DESERIALIZATION,
                details={"error_type": type(exc).__name__},
            ) from exc
        residual_missing = tuple(
            str(item) for item in getattr(result, "missing_keys", ())
        )
        residual_unexpected = tuple(
            str(item) for item in getattr(result, "unexpected_keys", ())
        )
        if residual_missing or residual_unexpected:
            raise CheckpointIncompatibleError(
                "target rejected the complete converted NumPy state",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.DESERIALIZATION,
                missing_keys=list(residual_missing),
                unexpected_keys=list(residual_unexpected),
            )
        return report


__all__ = ["NumpyCheckpointCodec"]

if __name__ == "__main__":
    print("\n=== Running Numpy Checkpoint Codec Comprehensive Self-Test ===\n")
    import tempfile
    
    printer.status("TEST", "Starting Numpy Codec tests", "info")

    # 1. Basic encode/decode with a simple mapping
    codec = NumpyCheckpointCodec()
    printer.status("CODEC", f"created {codec.codec_id} v{codec.codec_version}", "success")

    # supports
    assert codec.supports("model") is True
    assert codec.supports("model", value={"w": np.zeros(3)}) is True
    assert codec.supports("unknown") is False

    test_state = {"weight": np.ones((2, 2)), "bias": np.zeros(2)}
    ctx = CodecContext(checkpoint_id="test", version="v1", component="model")
    with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
        path = Path(tmp.name)
        outputs = codec.encode(test_state, path, context=ctx)
        assert len(outputs) == 1 and outputs[0].path == path
        decoded = codec.decode(path, context=ctx)
        # decoded is a dict of arrays; compare values
        for key in test_state:
            np.testing.assert_array_equal(decoded[key], test_state[key])
    printer.status("ROUNDTRIP", "encode/decode cycle passed", "success")

    # 2. Legacy NPZ without metadata envelope (when allow_legacy=True)
    with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
        path = Path(tmp.name)
        np.savez_compressed(path, legacy=np.array([1, 2, 3]))
        # decode with allow_legacy=True (default)
        decoded_legacy = codec.decode(path, context=ctx)
        assert "legacy" in decoded_legacy
        np.testing.assert_array_equal(decoded_legacy["legacy"], [1, 2, 3])
    printer.status("LEGACY", "legacy NPZ decoding works", "success")

    # 3. Error: too many arrays
    huge_state = {f"k{i}": np.zeros(1) for i in range(codec.max_arrays + 1)}
    try:
        codec.encode(huge_state, Path("/dev/null"), context=ctx)
        assert False, "Should have raised"
    except CheckpointCodecError as e:
        assert "too many arrays" in str(e)
    printer.status("ERRORS", "array limit enforcement works", "success")

    # 4. Basic restore with a dummy target
    class DummyTarget:
        def __init__(self):
            self.state = {}
        def state_dict(self):
            return self.state
        def load_state_dict(self, state, strict=True):
            self.state.update(state)
            class Result:
                missing_keys = []
                unexpected_keys = []
            return Result()

    target = DummyTarget()
    # populate target state with matching shapes
    target.state = {"weight": np.zeros((2, 2)), "bias": np.zeros(2)}
    report = codec.restore(target, decoded, strict=False)
    assert report.compatible
    np.testing.assert_array_equal(target.state["weight"], np.ones((2, 2)))
    printer.status("RESTORE", "restore into dummy target passed", "success")

    print("\n=== All numpy tests passed ===\n")