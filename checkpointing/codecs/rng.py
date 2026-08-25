"""Pickle-free capture, persistence, and explicit restoration of RNG state."""

from __future__ import annotations

import importlib
import random
import numpy as np

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..checkpoint_errors import *
from ..checkpoint_types import *
from .base import *
from .numpy import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]


logger = get_logger("Checkpoint Codec RNG")
printer = PrettyPrinter()


_FORMAT = "slai.checkpoint.rng-state"
_SCHEMA_VERSION = 1
_METADATA_KEY = "__metadata_utf8"


def _copy_uint8(value: np.ndarray, *, field_name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 1:
        raise ValueError(f"{field_name} must be a one-dimensional uint8 array")
    return np.array(array, dtype=np.uint8, copy=True)


@dataclass(frozen=True, slots=True)
class RNGStateSnapshot:
    """Framework-neutral snapshot of Python, NumPy, and optional torch RNGs."""

    python_state: tuple[Any, ...]
    numpy_algorithm: str
    numpy_keys: np.ndarray
    numpy_position: int
    numpy_has_gauss: int
    numpy_cached_gaussian: float
    torch_cpu: np.ndarray | None = None
    torch_cuda: tuple[np.ndarray, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.python_state, tuple):
            raise TypeError("python_state must be a tuple returned by random.getstate()")
        try:
            random.Random().setstate(self.python_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("python_state is not accepted by random.setstate()") from exc
        if not isinstance(self.numpy_algorithm, str) or not self.numpy_algorithm:
            raise ValueError("numpy_algorithm must be non-empty")
        keys = np.asarray(self.numpy_keys)
        if keys.dtype != np.uint32 or keys.ndim != 1:
            raise ValueError("numpy_keys must be a one-dimensional uint32 array")
        object.__setattr__(self, "numpy_keys", np.array(keys, dtype=np.uint32, copy=True))
        for name in ("numpy_position", "numpy_has_gauss"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.numpy_has_gauss not in {0, 1}:
            raise ValueError("numpy_has_gauss must be either 0 or 1")
        cached = float(self.numpy_cached_gaussian)
        if not np.isfinite(cached):
            raise ValueError("numpy_cached_gaussian must be finite")
        object.__setattr__(self, "numpy_cached_gaussian", cached)
        try:
            np.random.RandomState().set_state(
                (
                    self.numpy_algorithm,
                    np.array(keys, dtype=np.uint32, copy=True),
                    self.numpy_position,
                    self.numpy_has_gauss,
                    cached,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "NumPy RNG fields do not form a valid RandomState state"
            ) from exc
        if self.torch_cpu is not None:
            object.__setattr__(
                self,
                "torch_cpu",
                _copy_uint8(self.torch_cpu, field_name="torch_cpu"),
            )
        cuda = tuple(
            _copy_uint8(item, field_name=f"torch_cuda[{index}]")
            for index, item in enumerate(self.torch_cuda)
        )
        object.__setattr__(self, "torch_cuda", cuda)

    @property
    def providers(self) -> tuple[str, ...]:
        values = ["python", "numpy"]
        if self.torch_cpu is not None:
            values.append("torch_cpu")
        if self.torch_cuda:
            values.append("torch_cuda")
        return tuple(values)


@dataclass(frozen=True, slots=True)
class RNGRestoreReport:
    """Providers restored or deliberately skipped during RNG restoration."""

    restored: tuple[str, ...]
    skipped: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        restored = tuple(dict.fromkeys(str(item) for item in self.restored))
        skipped = tuple(dict.fromkeys(str(item) for item in self.skipped))
        if any(not item for item in restored + skipped):
            raise ValueError("RNG restore provider names cannot be empty")
        if set(restored).intersection(skipped):
            raise ValueError("RNG providers cannot be both restored and skipped")
        object.__setattr__(self, "restored", restored)
        object.__setattr__(self, "skipped", skipped)


def _json_tuple(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_tuple(item) for item in value]
    if isinstance(value, list):
        return [_json_tuple(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        f"Python RNG state contains unsupported value type {type(value).__name__}"
    )


def _restore_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_restore_tuple(item) for item in value)
    return value


def _optional_torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


class RNGStateCodec(BaseCheckpointCodec):
    """Capture and store process RNG state without Python pickle."""

    def __init__(
        self,
        *,
        include_torch: bool = True,
        include_cuda: bool = True,
        max_source_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        super().__init__("rng", "1", (StandardComponent.RNG.value,))
        if not isinstance(include_torch, bool) or not isinstance(include_cuda, bool):
            raise TypeError("include_torch and include_cuda must be booleans")
        if (
            isinstance(max_source_bytes, bool)
            or not isinstance(max_source_bytes, int)
            or max_source_bytes <= 0
        ):
            raise ValueError("max_source_bytes must be a positive integer")
        self.include_torch = include_torch
        self.include_cuda = include_cuda
        self.max_source_bytes = max_source_bytes
        self._numpy_codec = NumpyCheckpointCodec(
            components=(StandardComponent.RNG.value,),
            compressed=True,
            max_arrays=4096,
            max_decoded_bytes=max_source_bytes,
            max_source_bytes=max_source_bytes,
        )

    def _supports_value(self, value: Any | None) -> bool:
        return value is None or isinstance(value, RNGStateSnapshot)

    def capture(self) -> RNGStateSnapshot:
        """Capture configured RNG providers without mutating their state."""

        numpy_state = np.random.get_state()
        torch_cpu: np.ndarray | None = None
        torch_cuda: tuple[np.ndarray, ...] = ()
        torch_module = _optional_torch() if self.include_torch else None
        if torch_module is not None:
            torch_cpu = np.array(torch_module.get_rng_state().cpu().numpy(), copy=True)
            if self.include_cuda and torch_module.cuda.is_available():
                torch_cuda = tuple(
                    np.array(item.cpu().numpy(), copy=True)
                    for item in torch_module.cuda.get_rng_state_all()
                )
        return RNGStateSnapshot(
            python_state=random.getstate(),
            numpy_algorithm=str(numpy_state[0]),
            numpy_keys=np.array(numpy_state[1], dtype=np.uint32, copy=True),
            numpy_position=int(numpy_state[2]),
            numpy_has_gauss=int(numpy_state[3]),
            numpy_cached_gaussian=float(numpy_state[4]),
            torch_cpu=torch_cpu,
            torch_cuda=torch_cuda,
        )

    def _arrays(self, snapshot: RNGStateSnapshot) -> dict[str, np.ndarray]:
        metadata = {
            "format": _FORMAT,
            "schema_version": _SCHEMA_VERSION,
            "codec_version": self.codec_version,
            "python_state": _json_tuple(snapshot.python_state),
            "numpy_algorithm": snapshot.numpy_algorithm,
            "numpy_position": snapshot.numpy_position,
            "numpy_has_gauss": snapshot.numpy_has_gauss,
            "numpy_cached_gaussian": snapshot.numpy_cached_gaussian,
            "torch_cpu_present": snapshot.torch_cpu is not None,
            "torch_cuda_count": len(snapshot.torch_cuda),
        }
        arrays = {
            _METADATA_KEY: np.frombuffer(encode_json_object(metadata), dtype=np.uint8).copy(),
            "numpy.keys": np.array(snapshot.numpy_keys, copy=True),
        }
        if snapshot.torch_cpu is not None:
            arrays["torch.cpu"] = np.array(snapshot.torch_cpu, copy=True)
        for index, state in enumerate(snapshot.torch_cuda):
            arrays[f"torch.cuda.{index}"] = np.array(state, copy=True)
        return arrays

    def _encode(
        self,
        value: Any,
        destination: Path,
        *,
        context: CodecContext,
    ) -> Sequence[CodecOutput]:
        snapshot = self.capture() if value is None else value
        if not isinstance(snapshot, RNGStateSnapshot):
            raise TypeError("RNG codec value must be RNGStateSnapshot or None")
        self._numpy_codec.encode(self._arrays(snapshot), destination, context=context)
        return (
            CodecOutput(
                path=destination,
                media_type="application/vnd.numpy.npz",
                metadata={
                    "schema_version": _SCHEMA_VERSION,
                    "providers": list(snapshot.providers),
                    "allow_pickle": False,
                },
            ),
        )

    def _decode(self, source: Path, *, context: CodecContext) -> RNGStateSnapshot:
        arrays = dict(self._numpy_codec.decode(source, context=context))
        metadata_array = arrays.pop(_METADATA_KEY, None)
        if metadata_array is None:
            raise CheckpointCodecError(
                "RNG payload is missing metadata",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
            raise CheckpointCodecError(
                "RNG metadata must be a one-dimensional uint8 array",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        metadata = decode_json_object(metadata_array.tobytes(), source=source)
        expected_metadata = {
            "format",
            "schema_version",
            "codec_version",
            "python_state",
            "numpy_algorithm",
            "numpy_position",
            "numpy_has_gauss",
            "numpy_cached_gaussian",
            "torch_cpu_present",
            "torch_cuda_count",
        }
        if set(metadata) != expected_metadata:
            raise CheckpointCodecError(
                "RNG metadata has an invalid field set",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={
                    "missing": sorted(expected_metadata - set(metadata)),
                    "unknown": sorted(set(metadata) - expected_metadata),
                },
            )
        if metadata.get("format") != _FORMAT:
            raise CheckpointCodecError(
                "RNG payload has an unexpected format marker",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        if metadata.get("schema_version") != _SCHEMA_VERSION:
            raise CheckpointCodecError(
                "unsupported RNG state schema version",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={"actual": metadata.get("schema_version")},
            )
        if metadata.get("codec_version") != self.codec_version:
            raise CheckpointCodecError(
                "unsupported RNG codec payload version",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={"actual": metadata.get("codec_version")},
            )
        cuda_count = metadata.get("torch_cuda_count")
        if isinstance(cuda_count, bool) or not isinstance(cuda_count, int) or cuda_count < 0:
            raise CheckpointCodecError(
                "RNG metadata contains an invalid CUDA state count",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        expected_keys = {"numpy.keys"}
        torch_cpu_present = metadata.get("torch_cpu_present")
        if not isinstance(torch_cpu_present, bool):
            raise CheckpointCodecError(
                "RNG metadata contains an invalid torch CPU presence flag",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        if torch_cpu_present:
            expected_keys.add("torch.cpu")
        expected_keys.update(f"torch.cuda.{index}" for index in range(cuda_count))
        if set(arrays) != expected_keys:
            raise CheckpointCodecError(
                "RNG payload array set does not match its metadata",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={
                    "missing": sorted(expected_keys - set(arrays)),
                    "unexpected": sorted(set(arrays) - expected_keys),
                },
            )
        python_state = _restore_tuple(metadata.get("python_state"))
        if not isinstance(python_state, tuple):
            raise CheckpointCodecError(
                "RNG metadata contains invalid Python state",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        numpy_algorithm = metadata.get("numpy_algorithm")
        if not isinstance(numpy_algorithm, str) or not numpy_algorithm:
            raise CheckpointCodecError(
                "RNG metadata contains an invalid NumPy algorithm",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        numpy_keys = arrays["numpy.keys"]
        if numpy_keys.dtype != np.uint32 or numpy_keys.ndim != 1:
            raise CheckpointCodecError(
                "RNG NumPy key state must be a one-dimensional uint32 array",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        numpy_position = metadata.get("numpy_position")
        if isinstance(numpy_position, bool) or not isinstance(numpy_position, int) or numpy_position < 0:
            raise CheckpointCodecError(
                "RNG metadata contains an invalid NumPy position",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        numpy_has_gauss = metadata.get("numpy_has_gauss")
        if isinstance(numpy_has_gauss, bool) or not isinstance(numpy_has_gauss, int) or numpy_has_gauss < 0:
            raise CheckpointCodecError(
                "RNG metadata contains an invalid NumPy Gaussian flag",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        numpy_cached_gaussian = metadata.get("numpy_cached_gaussian")
        if not isinstance(numpy_cached_gaussian, (int, float)):
            raise CheckpointCodecError(
                "RNG metadata contains an invalid cached Gaussian value",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        numpy_cached_gaussian = float(numpy_cached_gaussian)
        if not np.isfinite(numpy_cached_gaussian):
            raise CheckpointCodecError(
                "RNG metadata contains a non-finite cached Gaussian value",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        return RNGStateSnapshot(
            python_state=python_state,
            numpy_algorithm=numpy_algorithm,
            numpy_keys=numpy_keys,
            numpy_position=numpy_position,
            numpy_has_gauss=numpy_has_gauss,
            numpy_cached_gaussian=numpy_cached_gaussian,
            torch_cpu=arrays.get("torch.cpu"),
            torch_cuda=tuple(arrays[f"torch.cuda.{index}"] for index in range(cuda_count)),
        )

    def restore(
        self,
        snapshot: RNGStateSnapshot,
        *,
        strict: bool = True,
    ) -> RNGRestoreReport:
        """Restore captured providers with compatibility preflight and rollback."""

        if not isinstance(snapshot, RNGStateSnapshot):
            raise TypeError("snapshot must be RNGStateSnapshot")
        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean")
        restored: list[str] = []
        skipped: list[str] = []
        torch_module = _optional_torch() if self.include_torch else None
        if snapshot.torch_cpu is not None and torch_module is None and strict:
            raise CheckpointCodecError(
                "saved torch RNG state cannot be restored because torch is unavailable",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.COMPATIBILITY,
            )
        cuda_available = bool(
            snapshot.torch_cuda
            and self.include_cuda
            and torch_module is not None
            and torch_module.cuda.is_available()
        )
        assert torch_module is not None
        runtime_count = (
            int(torch_module.cuda.device_count()) if cuda_available else 0
        )
        if snapshot.torch_cuda:
            if not cuda_available and strict:
                raise CheckpointCodecError(
                    "saved CUDA RNG state cannot be restored on this runtime",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.COMPATIBILITY,
                    details={
                        "include_cuda": self.include_cuda,
                        "torch_available": torch_module is not None,
                    },
                )
            if cuda_available:
                if runtime_count != len(snapshot.torch_cuda) and strict:
                    raise CheckpointCodecError(
                        "saved CUDA RNG device count differs from the runtime",
                        operation=CheckpointOperation.LOAD,
                        stage=CheckpointStage.COMPATIBILITY,
                        details={
                            "saved_devices": len(snapshot.torch_cuda),
                            "runtime_devices": runtime_count,
                        },
                    )

        try:
            torch_cpu_tensor = (
                torch_module.as_tensor(
                    np.array(snapshot.torch_cpu, copy=True),
                    dtype=torch_module.uint8,
                    device="cpu",
                )
                if snapshot.torch_cpu is not None and torch_module is not None
                else None
            )
            assert torch_module is not None
            cuda_tensors = tuple(
                torch_module.as_tensor(
                    np.array(state, copy=True),
                    dtype=torch_module.uint8,
                    device="cpu",
                )
                for state in snapshot.torch_cuda[:runtime_count]
            ) if cuda_available else ()
            previous_torch_cpu = (
                torch_module.get_rng_state().clone()
                if torch_cpu_tensor is not None
                else None
            )
            previous_cuda = (
                tuple(state.clone() for state in torch_module.cuda.get_rng_state_all())
                if cuda_tensors
                else ()
            )
        except Exception as exc:
            raise CheckpointCodecError(
                "failed to prepare framework RNG state for restoration",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.COMPATIBILITY,
                details={"error_type": type(exc).__name__},
            ) from exc

        previous_python = random.getstate()
        previous_numpy = np.random.get_state()
        python_applied = False
        numpy_applied = False
        torch_cpu_applied = False
        cuda_applied = False
        try:
            python_applied = True
            random.setstate(snapshot.python_state)
            restored.append("python")
            numpy_applied = True
            np.random.set_state(
                (
                    snapshot.numpy_algorithm,
                    np.array(snapshot.numpy_keys, copy=True),
                    snapshot.numpy_position,
                    snapshot.numpy_has_gauss,
                    snapshot.numpy_cached_gaussian,
                )
            )
            restored.append("numpy")

            if snapshot.torch_cpu is not None:
                if torch_cpu_tensor is None:
                    skipped.append("torch_cpu")
                else:
                    torch_cpu_applied = True
                    torch_module.set_rng_state(torch_cpu_tensor)
                    restored.append("torch_cpu")
            if snapshot.torch_cuda:
                if not cuda_tensors:
                    skipped.append("torch_cuda")
                else:
                    cuda_applied = True
                    for index, tensor in enumerate(cuda_tensors):
                        torch_module.cuda.set_rng_state(tensor, device=index)
                    restored.append("torch_cuda")
                    if runtime_count < len(snapshot.torch_cuda):
                        skipped.append("torch_cuda_excess_saved_devices")
                    elif runtime_count > len(snapshot.torch_cuda):
                        skipped.append("torch_cuda_unspecified_runtime_devices")
        except Exception as exc:
            rollback_errors: list[str] = []
            for provider, was_applied, rollback in (
                (
                    "torch_cuda",
                    cuda_applied,
                    lambda: torch_module.cuda.set_rng_state_all(list(previous_cuda)),
                ),
                (
                    "torch_cpu",
                    torch_cpu_applied,
                    lambda: torch_module.set_rng_state(previous_torch_cpu),
                ),
                ("numpy", numpy_applied, lambda: np.random.set_state(previous_numpy)),
                ("python", python_applied, lambda: random.setstate(previous_python)),
            ):
                if not was_applied:
                    continue
                try:
                    rollback()
                except Exception:
                    rollback_errors.append(provider)
            raise CheckpointCodecError(
                "RNG restoration failed",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.RESTORATION,
                details={
                    "error_type": type(exc).__name__,
                    "rolled_back": not rollback_errors,
                    "rollback_failures": rollback_errors,
                },
            ) from exc
        return RNGRestoreReport(tuple(restored), tuple(skipped))


__all__ = ["RNGRestoreReport", "RNGStateCodec", "RNGStateSnapshot"]

if __name__ == "__main__":
    print("\n=== Running RNG State Codec Comprehensive Self-Test ===\n")
    import tempfile

    printer.status("TEST", "Starting RNG state codec tests", "info")
    codec = RNGStateCodec(include_torch=False)  # avoid torch for speed
    printer.status("CODEC", f"created {codec.codec_id} v{codec.codec_version}", "success")

    # 1. Capture, encode, decode, restore – verify random sequence reproduces
    random.seed(42)
    np.random.seed(42)
    initial_python = random.getstate()
    initial_numpy = np.random.get_state()

    snapshot1 = codec.capture()
    ctx = CodecContext(checkpoint_id="test", version="v1", component="rng")
    with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
        path = Path(tmp.name)
        outputs = codec.encode(snapshot1, path, context=ctx)
        assert len(outputs) == 1
        decoded = codec.decode(path, context=ctx)
        # Compare snapshots (fields)
        assert decoded.python_state == snapshot1.python_state
        np.testing.assert_array_equal(decoded.numpy_keys, snapshot1.numpy_keys)
        assert decoded.numpy_position == snapshot1.numpy_position
    printer.status("ROUNDTRIP", "encode/decode of snapshot passed", "success")

    # 2. Restore and verify state changes
    # Change RNG state, then restore to snapshot1, then draw numbers and compare
    random.seed(999)
    np.random.seed(999)
    report = codec.restore(snapshot1, strict=True)
    assert "python" in report.restored and "numpy" in report.restored
    # Now states should be identical to initial
    assert random.getstate() == initial_python
    np.testing.assert_array_equal(np.random.get_state()[1], initial_numpy[1])
    printer.status("RESTORE", "restore reproduced original RNG state", "success")

    # 3. Error: strict restore with incompatible torch state (skip if torch not available)
    try:
        torch_snapshot = RNGStateSnapshot(
            python_state=random.getstate(),
            numpy_algorithm="MT19937",
            numpy_keys=np.array([0], dtype=np.uint32),
            numpy_position=0,
            numpy_has_gauss=0,
            numpy_cached_gaussian=0.0,
            torch_cpu=np.array([1, 2, 3], dtype=np.uint8),
        )
        codec.restore(torch_snapshot, strict=True)
        assert False, "Should have raised because torch not included"
    except CheckpointCodecError as e:
        assert "torch is unavailable" in str(e)
    printer.status("ERRORS", "missing torch handling works", "success")

    print("\n=== All rng state tests passed ===\n")