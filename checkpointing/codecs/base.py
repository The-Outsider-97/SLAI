"""Shared codec contracts, validation, and safe I/O primitives.

Concrete codecs implement only format-specific conversion.  This module owns
the common lifecycle: component/context validation, structured error wrapping,
output containment, atomic single-file publication, strict JSON parsing, and
state-restoration reports.  It deliberately does not select codecs, build
manifests, hash artifacts, or commit checkpoint directories.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import stat
import uuid

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..checkpoint_errors import *
from ..checkpoint_storage import *
from ..checkpoint_types import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]


logger = get_logger("Checkpoint Codec Base")
printer = PrettyPrinter()

@dataclass(frozen=True, slots=True)
class ShapeMismatch:
    """One state entry whose saved and target tensor shapes disagree."""

    name: str
    checkpoint_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    checkpoint_dtype: str | None = None
    target_dtype: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("shape mismatch name must be non-empty")
        for field_name in ("checkpoint_shape", "target_shape"):
            shape = tuple(getattr(self, field_name))
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in shape
            ):
                raise ValueError(f"{field_name} must contain non-negative integers")
            object.__setattr__(self, field_name, shape)
        for field_name in ("checkpoint_dtype", "target_dtype"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field_name} must be a non-empty string or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "checkpoint_shape": list(self.checkpoint_shape),
            "target_shape": list(self.target_shape),
            "checkpoint_dtype": self.checkpoint_dtype,
            "target_dtype": self.target_dtype,
        }


@dataclass(frozen=True, slots=True)
class StateLoadReport:
    """Framework-neutral result of applying decoded state to a target."""

    loaded_keys: tuple[str, ...] = ()
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()
    mismatched_keys: tuple[ShapeMismatch, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("loaded_keys", "missing_keys", "unexpected_keys"):
            values = tuple(str(item) for item in getattr(self, field_name))
            if any(not item for item in values):
                raise ValueError(f"{field_name} cannot contain empty keys")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} cannot contain duplicate keys")
            object.__setattr__(self, field_name, values)
        mismatches = tuple(self.mismatched_keys)
        if any(not isinstance(item, ShapeMismatch) for item in mismatches):
            raise TypeError("mismatched_keys must contain ShapeMismatch values")
        if len({item.name for item in mismatches}) != len(mismatches):
            raise ValueError("mismatched_keys cannot contain duplicate keys")
        object.__setattr__(self, "mismatched_keys", mismatches)
        categories = (
            set(self.loaded_keys),
            set(self.missing_keys),
            set(self.unexpected_keys),
            {item.name for item in self.mismatched_keys},
        )
        for index, left in enumerate(categories):
            for right in categories[index + 1 :]:
                if left.intersection(right):
                    raise ValueError("state load report key categories must be disjoint")

    @property
    def compatible(self) -> bool:
        return not (self.missing_keys or self.unexpected_keys or self.mismatched_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "loaded_keys": list(self.loaded_keys),
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "mismatched_keys": [item.to_dict() for item in self.mismatched_keys],
        }


def require_dependency(module_name: str, *, codec_id: str) -> Any:
    """Import an optional runtime dependency with codec-domain diagnostics."""

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise CheckpointCodecError(
            f"codec {codec_id!r} requires optional dependency {module_name!r}",
            stage=CheckpointStage.VALIDATION,
            details={"codec_id": codec_id, "dependency": module_name},
        ) from exc


def metadata_bool(
    context: CodecContext,
    name: str,
    *,
    default: bool,
) -> bool:
    """Read a strict boolean option from codec context metadata."""

    value = context.metadata.get(name, default)
    if not isinstance(value, bool):
        raise CheckpointCodecError(
            f"codec option {name!r} must be a boolean",
            stage=CheckpointStage.VALIDATION,
            checkpoint_id=context.checkpoint_id,
            version=context.version,
            component=context.component,
        )
    return value


def metadata_string(
    context: CodecContext,
    name: str,
    *,
    default: str | None = None,
) -> str | None:
    """Read a strict optional string option from codec context metadata."""

    value = context.metadata.get(name, default)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CheckpointCodecError(
            f"codec option {name!r} must be a non-empty string",
            stage=CheckpointStage.VALIDATION,
            checkpoint_id=context.checkpoint_id,
            version=context.version,
            component=context.component,
        )
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not permitted: {value}")


def encode_json_object(value: Mapping[str, Any]) -> bytes:
    """Encode a validated mapping as deterministic UTF-8 JSON."""

    frozen = freeze_json(dict(value), _path="$.codec_payload")
    if not isinstance(frozen, Mapping):
        raise TypeError("codec JSON payload must be an object")
    return json.dumps(
        thaw_json(frozen),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_json_object(data: bytes, *, source: Path | None = None) -> dict[str, Any]:
    """Decode strict UTF-8 JSON, rejecting duplicates and non-finite values."""

    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CheckpointCodecError(
            "invalid codec JSON payload",
            operation=CheckpointOperation.LOAD,
            stage=CheckpointStage.DESERIALIZATION,
            path=source,
            details={"reason": str(exc)},
        ) from exc
    if not isinstance(value, dict):
        raise CheckpointCodecError(
            "codec JSON payload must contain an object",
            operation=CheckpointOperation.LOAD,
            stage=CheckpointStage.DESERIALIZATION,
            path=source,
        )
    try:
        frozen = freeze_json(value, _path="$.codec_payload")
    except ValueError as exc:
        raise CheckpointCodecError(
            "codec JSON payload violates structural limits",
            operation=CheckpointOperation.LOAD,
            stage=CheckpointStage.DESERIALIZATION,
            path=source,
            details={"reason": str(exc)},
        ) from exc
    if not isinstance(frozen, Mapping):
        raise CheckpointCodecError("codec JSON payload must contain an object", path=source)
    thawed = thaw_json(frozen)
    if not isinstance(thawed, dict):
        raise CheckpointCodecError("codec JSON payload must contain an object", path=source)
    return thawed


def write_json_object(
    value: Mapping[str, Any],
    destination: Path,
    *,
    durable: bool = True,
    max_bytes: int | None = None,
) -> int:
    """Atomically persist a deterministic JSON object."""

    if max_bytes is not None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
    encoded = encode_json_object(value)
    if max_bytes is not None:
        if len(encoded) > max_bytes:
            raise CheckpointCodecError(
                "encoded codec JSON exceeds the configured byte limit",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
                path=destination,
                details={"actual_bytes": len(encoded), "max_bytes": max_bytes},
            )
    atomic_write_bytes(
        encoded,
        destination,
        durable=durable,
        mode=0o600,
    )
    return len(encoded)


def read_json_object(source: Path, *, max_bytes: int) -> dict[str, Any]:
    """Read a bounded strict JSON object."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    return decode_json_object(
        read_limited_bytes(source, max_bytes=max_bytes),
        source=source,
    )


@contextmanager
def atomic_output_path(destination: Path, *, durable: bool = True, mode: int = 0o600) -> Iterator[Path]:
    """Yield a same-directory temporary path and atomically publish it on success."""

    if not isinstance(durable, bool):
        raise TypeError("durable must be a boolean")
    if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o777:
        raise ValueError("mode must be an integer permission mask between 0 and 0o777")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.is_symlink() or target.is_dir():
        raise CheckpointCodecError(
            "codec destination must be a regular-file path",
            operation=CheckpointOperation.SAVE,
            stage=CheckpointStage.VALIDATION,
            path=target,
        )
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.codec-tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(temporary), flags, mode)
        os.close(descriptor)
        yield temporary
        if temporary.is_symlink():
            raise CheckpointCodecError(
                "codec serializer produced a symbolic link",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.SERIALIZATION,
                path=temporary,
            )
        try:
            file_stat = temporary.stat()
        except OSError as exc:
            raise CheckpointCodecError(
                "codec serializer did not produce its declared output",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.SERIALIZATION,
                path=temporary,
            ) from exc
        if not stat.S_ISREG(file_stat.st_mode):
            raise CheckpointCodecError(
                "codec serializer output is not a regular file",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.SERIALIZATION,
                path=temporary,
            )
        os.chmod(temporary, mode)
        if durable:
            with temporary.open("rb") as handle:
                if os.name != "nt":      # Windows fsync not supported
                    os.fsync(handle.fileno())
        os.replace(temporary, target)
        if durable:
            fsync_directory(target.parent, strict=True)
    finally:
        if temporary.is_symlink() or temporary.is_file():
            temporary.unlink(missing_ok=True)
        elif temporary.is_dir():
            shutil.rmtree(temporary)


def ensure_bounded_regular_file(source: Path, *, max_bytes: int | None = None) -> Path:
    """Validate a non-symlink regular input file and optional byte limit."""

    path = Path(source)
    if path.is_symlink():
        raise CheckpointCodecError(
            "codec input cannot be a symbolic link",
            operation=CheckpointOperation.LOAD,
            stage=CheckpointStage.VALIDATION,
            path=path,
        )
    try:
        file_stat = path.stat()
    except OSError as exc:
        raise CheckpointCodecError(
            "codec input does not exist or cannot be inspected",
            operation=CheckpointOperation.LOAD,
            stage=CheckpointStage.DISCOVERY,
            path=path,
            retryable=False,
        ) from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise CheckpointCodecError(
            "codec input must be a regular file",
            operation=CheckpointOperation.LOAD,
            stage=CheckpointStage.VALIDATION,
            path=path,
        )
    if max_bytes is not None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if file_stat.st_size > max_bytes:
            raise CheckpointCodecError(
                "codec input exceeds the configured byte limit",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.VALIDATION,
                path=path,
                details={"actual_bytes": file_stat.st_size, "max_bytes": max_bytes},
            )
    return path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class BaseCheckpointCodec(ABC):
    """Template implementation of the shared ``CheckpointCodec`` lifecycle."""

    def __init__(
        self,
        codec_id: str,
        codec_version: str,
        components: Sequence[str],
        *,
        multi_file: bool = False,
    ) -> None:
        self._codec_id = validate_identifier(codec_id, field_name="codec_id")
        if not isinstance(codec_version, str) or not codec_version.strip():
            raise ValueError("codec_version must be a non-empty string")
        self._codec_version = codec_version.strip()
        if isinstance(components, (str, bytes)):
            raise TypeError("components must be a sequence of component names")
        normalized = tuple(
            dict.fromkeys(validate_component_name(component) for component in components)
        )
        if not normalized:
            raise ValueError("a codec must support at least one component")
        self._components = normalized
        if not isinstance(multi_file, bool):
            raise TypeError("multi_file must be a boolean")
        self._multi_file = multi_file

    @property
    def codec_id(self) -> str:
        return self._codec_id

    @property
    def codec_version(self) -> str:
        return self._codec_version

    @property
    def components(self) -> tuple[str, ...]:
        return self._components

    @property
    def multi_file(self) -> bool:
        return self._multi_file

    def supports(self, component: str, value: Any | None = None) -> bool:
        try:
            normalized = validate_component_name(component)
        except ValueError:
            return False
        return normalized in self._components and self._supports_value(value)

    def _supports_value(self, value: Any | None) -> bool:
        del value
        return True

    def _validate_context(
        self,
        context: CodecContext,
        *,
        operation: CheckpointOperation,
    ) -> None:
        if not isinstance(context, CodecContext):
            raise TypeError("context must be a CodecContext")
        if context.component not in self._components:
            raise CheckpointCodecError(
                f"codec {self.codec_id!r} does not support component "
                f"{context.component!r}",
                operation=operation,
                stage=CheckpointStage.VALIDATION,
                checkpoint_id=context.checkpoint_id,
                version=context.version,
                component=context.component,
                details={"supported_components": list(self._components)},
            )

    def _contextualize(
        self,
        error: CheckpointError,
        *,
        context: CodecContext,
        path: Path,
        operation: CheckpointOperation,
        stage: CheckpointStage,
    ) -> CheckpointError:
        changes: dict[str, Any] = {}
        for name, value in (
            ("operation", operation.value),
            ("stage", stage.value),
            ("path", path),
            ("version", context.version),
            ("checkpoint_id", context.checkpoint_id),
            ("component", context.component),
        ):
            if getattr(error.context, name) is None:
                changes[name] = value
        return error.with_context(**changes) if changes else error

    def encode(
        self,
        value: Any,
        destination: Path,
        *,
        context: CodecContext,
    ) -> Sequence[CodecOutput]:
        self._validate_context(context, operation=CheckpointOperation.SAVE)
        destination = Path(destination)
        if not self.supports(context.component, value):
            raise CheckpointCodecError(
                f"codec {self.codec_id!r} does not support the supplied value",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
                path=destination,
                version=context.version,
                checkpoint_id=context.checkpoint_id,
                component=context.component,
            )
        try:
            outputs = tuple(self._encode(value, destination, context=context))
            return self._validate_outputs(outputs, destination)
        except CheckpointError as exc:
            raise self._contextualize(
                exc,
                context=context,
                path=destination,
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.SERIALIZATION,
            ) from exc.__cause__
        except Exception as exc:
            raise CheckpointCodecError(
                f"codec {self.codec_id!r} failed to encode component",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.SERIALIZATION,
                path=destination,
                version=context.version,
                checkpoint_id=context.checkpoint_id,
                component=context.component,
                retryable=isinstance(exc, OSError),
                details={"codec_id": self.codec_id, "error_type": type(exc).__name__},
            ) from exc

    def decode(
        self,
        source: Path,
        *,
        context: CodecContext,
    ) -> Any:
        self._validate_context(context, operation=CheckpointOperation.LOAD)
        source = Path(source)
        try:
            if self.multi_file:
                if source.is_symlink() or not source.is_dir():
                    raise CheckpointCodecError(
                        "multi-file codec input must be a non-symlink directory",
                        stage=CheckpointStage.VALIDATION,
                        path=source,
                    )
            else:
                ensure_bounded_regular_file(source)
            return self._decode(source, context=context)
        except CheckpointError as exc:
            raise self._contextualize(
                exc,
                context=context,
                path=source,
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.DESERIALIZATION,
            ) from exc.__cause__
        except Exception as exc:
            raise CheckpointCodecError(
                f"codec {self.codec_id!r} failed to decode component",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                version=context.version,
                checkpoint_id=context.checkpoint_id,
                component=context.component,
                retryable=isinstance(exc, OSError),
                details={"codec_id": self.codec_id, "error_type": type(exc).__name__},
            ) from exc

    def _validate_outputs(
        self,
        outputs: tuple[CodecOutput, ...],
        destination: Path,
    ) -> tuple[CodecOutput, ...]:
        if not outputs:
            raise CheckpointCodecError(
                f"codec {self.codec_id!r} produced no artifacts",
                stage=CheckpointStage.SERIALIZATION,
                path=destination,
            )
        if any(not isinstance(output, CodecOutput) for output in outputs):
            raise TypeError("codec outputs must contain CodecOutput values")

        root = destination.resolve() if self.multi_file else destination.parent.resolve()
        expected_file = None if self.multi_file else destination.resolve()
        seen: set[Path] = set()
        for output in outputs:
            path = output.path
            if path.is_symlink():
                raise CheckpointCodecError(
                    "codec output cannot be a symbolic link",
                    stage=CheckpointStage.SERIALIZATION,
                    path=path,
                )
            resolved = path.resolve(strict=True)
            if not resolved.is_file():
                raise CheckpointCodecError(
                    "codec output must be a regular file",
                    stage=CheckpointStage.SERIALIZATION,
                    path=resolved,
                )
            if self.multi_file:
                if not _is_within(resolved, root):
                    raise CheckpointCodecError(
                        "codec output escapes its destination directory",
                        stage=CheckpointStage.SERIALIZATION,
                        path=resolved,
                    )
            elif resolved != expected_file:
                raise CheckpointCodecError(
                    "single-file codec produced an undeclared output path",
                    stage=CheckpointStage.SERIALIZATION,
                    path=resolved,
                )
            if resolved in seen:
                raise CheckpointCodecError(
                    "codec returned duplicate output paths",
                    stage=CheckpointStage.SERIALIZATION,
                    path=resolved,
                )
            seen.add(resolved)
        return outputs

    @abstractmethod
    def _encode(
        self,
        value: Any,
        destination: Path,
        *,
        context: CodecContext,
    ) -> Sequence[CodecOutput]:
        raise NotImplementedError

    @abstractmethod
    def _decode(self, source: Path, *, context: CodecContext) -> Any:
        raise NotImplementedError


def is_codec(value: Any) -> bool:
    """Return whether an object satisfies the public structural codec protocol."""

    return isinstance(value, CheckpointCodec)


__all__ = [
    "BaseCheckpointCodec",
    "ShapeMismatch",
    "StateLoadReport",
    "atomic_output_path",
    "decode_json_object",
    "encode_json_object",
    "ensure_bounded_regular_file",
    "is_codec",
    "metadata_bool",
    "metadata_string",
    "read_json_object",
    "require_dependency",
    "write_json_object",
]


if __name__ == "__main__":
    print("\n=== Running Checkpoint Codec Base Comprehensive Self-Test ===\n")
    printer.status("TEST", "Starting base codec tests", "info")

    # Dummy concrete implementation for testing the abstract base
    class DummyCodec(BaseCheckpointCodec):
        def _encode(self, value, destination, *, context):
            # Write a simple JSON file for demonstration
            write_json_object({"value": value}, destination, durable=False)
            return (CodecOutput(path=destination, media_type="application/json"),)

        def _decode(self, source, *, context):
            return read_json_object(source, max_bytes=1024)

    # Instantiate with valid components
    codec = DummyCodec("dummy", "1.0", ("test",))
    printer.status("DUMMY", f"created {codec.codec_id} v{codec.codec_version}", "success")

    # Test supports
    assert codec.supports("test") is True
    assert codec.supports("unknown") is False
    printer.status("SUPPORTS", "component support checks passed", "success")

    # Test encode/decode round-trip with temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        path = Path(tmp.name)
        test_value = {"foo": 42, "bar": "baz"}
        ctx = CodecContext(checkpoint_id="test", version="v1", component="test")
        outputs = codec.encode(test_value, path, context=ctx)
        assert len(outputs) == 1 and outputs[0].path == path

        decoded = codec.decode(path, context=ctx)
        assert decoded == {"value": test_value}  # our encode wraps value

    printer.status("ROUNDTRIP", "encode/decode cycle passed", "success")

    # Test JSON utilities
    test_data = {"a": 1, "b": [2, 3]}
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        p = Path(tmp.name)
        write_json_object(test_data, p, durable=False)
        read_data = read_json_object(p, max_bytes=1024)
        assert read_data == test_data
    printer.status("JSON", "JSON I/O utilities passed", "success")

    print("\n=== All base tests passed ===\n")