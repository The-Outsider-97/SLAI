"""
NumPy-aware JSON encoding utilities for the base subsystem.

This module provides a production-ready JSON encoder focused on robust,
configurable serialization of NumPy scalars, arrays, complex numbers,
datetime-like values, masked arrays, and lightweight custom objects. It is
intended for infrastructure, experiment-tracking, model I/O, telemetry, and
other code paths that need deterministic JSON output without re-implementing
NumPy-specific conversion logic.

Design goals:
- safe and explicit serialization of NumPy-specific value types
- configurable handling of arrays, metadata, callables, custom objects, and
  non-finite numeric values
- optional reversible decoding for tagged payloads such as ndarrays and complex
  values
- bounded operational history and summary statistics for observability
- helper-first integration with the shared base error and helper layer
- practical backward compatibility with ``json.JSONEncoder`` usage
"""

from __future__ import annotations

import json
import math
import io

from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any, Deque, Dict, List, Optional, TextIO, Tuple, Union

import numpy as np

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.base_errors import *
from ..utils.base_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Numpy Encoder")
printer = PrettyPrinter()


@dataclass(frozen=True)
class NumpyEncodingRecord:
    """Bounded audit record for one encoding or decoding event."""

    action: str
    success: bool
    timestamp: str
    payload_type: str
    payload_fingerprint: Optional[str] = None
    detail: Optional[str] = None
    size_hint: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return drop_none_values(
            {
                "action": self.action,
                "success": self.success,
                "timestamp": self.timestamp,
                "payload_type": self.payload_type,
                "payload_fingerprint": self.payload_fingerprint,
                "detail": self.detail,
                "size_hint": self.size_hint,
            },
            recursive=True,
            drop_empty=False,
        )


@dataclass(frozen=True)
class NumpyEncoderStats:
    """Summary counters for encoder activity."""

    encode_operations: int
    decode_operations: int
    failures: int
    arrays_serialized: int
    scalars_serialized: int
    complex_values_serialized: int
    datetime_values_serialized: int
    callable_values_serialized: int
    custom_objects_serialized: int
    history_length: int
    tagged_payloads_decoded: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "encode_operations": self.encode_operations,
            "decode_operations": self.decode_operations,
            "failures": self.failures,
            "arrays_serialized": self.arrays_serialized,
            "scalars_serialized": self.scalars_serialized,
            "complex_values_serialized": self.complex_values_serialized,
            "datetime_values_serialized": self.datetime_values_serialized,
            "callable_values_serialized": self.callable_values_serialized,
            "custom_objects_serialized": self.custom_objects_serialized,
            "history_length": self.history_length,
            "tagged_payloads_decoded": self.tagged_payloads_decoded,
        }


class NumpyEncoder(json.JSONEncoder):
    """
    Production-ready JSON encoder for NumPy-centric payloads.

    In addition to the standard ``json.JSONEncoder`` interface, the class
    provides convenience methods for dumps/loads, reversible tagged decoding,
    bounded history, and config-driven behavior.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.global_config = load_global_config()
        self.numpy_config = get_config_section("numpy_encoder") or {}

        self.array_format = ensure_one_of(
            kwargs.pop("array_format", self.numpy_config.get("array_format", "tagged")),
            ["tagged", "list"],
            "array_format",
            config=self.numpy_config,
            error_cls=BaseConfigurationError,
        )
        self.encode_scalar_metadata = coerce_bool(
            kwargs.pop("encode_scalar_metadata", self.numpy_config.get("encode_scalar_metadata", False)),
            default=False,
        )
        self.encode_datetime_as = ensure_one_of(
            kwargs.pop("encode_datetime_as", self.numpy_config.get("encode_datetime_as", "iso")),
            ["iso", "int"],
            "encode_datetime_as",
            config=self.numpy_config,
            error_cls=BaseConfigurationError,
        )
        self.encode_timedelta_as = ensure_one_of(
            kwargs.pop("encode_timedelta_as", self.numpy_config.get("encode_timedelta_as", "int")),
            ["int", "string"],
            "encode_timedelta_as",
            config=self.numpy_config,
            error_cls=BaseConfigurationError,
        )
        self.handle_non_finite = ensure_one_of(
            kwargs.pop("handle_non_finite", self.numpy_config.get("handle_non_finite", "string")),
            ["string", "null", "raise"],
            "handle_non_finite",
            config=self.numpy_config,
            error_cls=BaseConfigurationError,
        )
        self.include_array_dtype = coerce_bool(
            kwargs.pop("include_array_dtype", self.numpy_config.get("include_array_dtype", True)),
            default=True,
        )
        self.include_array_shape = coerce_bool(
            kwargs.pop("include_array_shape", self.numpy_config.get("include_array_shape", True)),
            default=True,
        )
        self.include_array_size = coerce_bool(
            kwargs.pop("include_array_size", self.numpy_config.get("include_array_size", False)),
            default=False,
        )
        self.serialize_callables = coerce_bool(
            kwargs.pop("serialize_callables", self.numpy_config.get("serialize_callables", True)),
            default=True,
        )
        self.serialize_custom_objects = coerce_bool(
            kwargs.pop("serialize_custom_objects", self.numpy_config.get("serialize_custom_objects", True)),
            default=True,
        )
        self.custom_object_strategy = ensure_one_of(
            kwargs.pop("custom_object_strategy", self.numpy_config.get("custom_object_strategy", "dict")),
            ["dict", "repr", "string"],
            "custom_object_strategy",
            config=self.numpy_config,
            error_cls=BaseConfigurationError,
        )
        self.decode_tagged_payloads = coerce_bool(
            kwargs.pop("decode_tagged_payloads", self.numpy_config.get("decode_tagged_payloads", True)),
            default=True,
        )
        self.record_history = coerce_bool(
            kwargs.pop("record_history", self.numpy_config.get("record_history", True)),
            default=True,
        )
        self.history_limit = coerce_int(
            kwargs.pop("history_limit", self.numpy_config.get("history_limit", 200)),
            default=200,
            minimum=1,
        )
        self.max_history_detail_length = coerce_int(
            kwargs.pop("max_history_detail_length", self.numpy_config.get("max_history_detail_length", 240)),
            default=240,
            minimum=40,
        )
        self.max_array_preview_items = coerce_int(
            kwargs.pop("max_array_preview_items", self.numpy_config.get("max_array_preview_items", 25)),
            default=25,
            minimum=1,
        )
        self.strict_numpy_only = coerce_bool(
            kwargs.pop("strict_numpy_only", self.numpy_config.get("strict_numpy_only", False)),
            default=False,
        )
        self.ensure_ascii_default = coerce_bool(
            kwargs.pop("ensure_ascii_default", self.numpy_config.get("ensure_ascii_default", False)),
            default=False,
        )
        self.sort_keys_default = coerce_bool(
            kwargs.pop("sort_keys_default", self.numpy_config.get("sort_keys_default", True)),
            default=True,
        )
        self.indent_default = kwargs.pop("indent_default", self.numpy_config.get("indent_default", None))

        self._history: Deque[NumpyEncodingRecord] = deque(maxlen=self.history_limit)
        self._stats = defaultdict(int)

        json_kwargs = dict(kwargs)
        json_kwargs.setdefault("ensure_ascii", self.ensure_ascii_default)
        super().__init__(*args, **json_kwargs)

        logger.info("Numpy Encoder successfully initialized")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record(self, action: str, payload: Any, *, success: bool, detail: Optional[str] = None, size_hint: Optional[int] = None) -> None:
        if not self.record_history:
            return
        self._history.append(
            NumpyEncodingRecord(
                action=action,
                success=success,
                timestamp=utc_now_iso(),
                payload_type=type(payload).__name__,
                payload_fingerprint=self._safe_fingerprint(payload),
                detail=truncate_string(detail or "", self.max_history_detail_length) if detail else None,
                size_hint=size_hint,
            )
        )

    def _safe_fingerprint(self, payload: Any) -> Optional[str]:
        try:
            return stable_fingerprint(self._preview_payload(payload), length=24)
        except Exception:
            return None

    def _preview_payload(self, payload: Any) -> Any:
        if isinstance(payload, np.ndarray):
            preview = payload.flatten()[: self.max_array_preview_items].tolist()
            return {
                "type": "ndarray",
                "dtype": str(payload.dtype),
                "shape": tuple(int(dim) for dim in payload.shape),
                "preview": preview,
            }
        if isinstance(payload, np.generic):
            try:
                return {"type": type(payload).__name__, "value": payload.item()}
            except Exception:
                return safe_repr(payload)
        return to_json_safe(payload, max_items=self.max_array_preview_items)

    def _normalize_non_finite(self, value: float, *, type_name: str) -> Any:
        if math.isfinite(value):
            return value
        if self.handle_non_finite == "null":
            return None
        if self.handle_non_finite == "string":
            if math.isnan(value):
                return "NaN"
            if value == float("inf"):
                return "Infinity"
            if value == float("-inf"):
                return "-Infinity"
            return str(value)
        raise BaseSerializationError(
            f"Encountered non-finite numeric value during {type_name} serialization.",
            self.numpy_config,
            component="NumpyEncoder",
            operation="default",
            context={"value": safe_repr(value), "type_name": type_name},
        )

    def _encode_numpy_scalar(self, obj: np.generic) -> Any:
        self._stats["scalars_serialized"] += 1

        if isinstance(obj, np.bool_):
            return bool(obj)

        if isinstance(obj, np.integer):
            value = int(obj)
            if self.encode_scalar_metadata:
                return {"__type__": "np_scalar", "dtype": str(obj.dtype), "value": value}
            return value

        if isinstance(obj, np.floating):
            value = self._normalize_non_finite(float(obj), type_name="numpy floating")
            if self.encode_scalar_metadata:
                return {"__type__": "np_scalar", "dtype": str(obj.dtype), "value": value}
            return value

        if isinstance(obj, np.complexfloating):
            self._stats["complex_values_serialized"] += 1
            return {
                "__type__": "complex",
                "dtype": str(obj.dtype),
                "real": self._normalize_non_finite(float(np.real(obj)), type_name="complex real"),
                "imag": self._normalize_non_finite(float(np.imag(obj)), type_name="complex imag"),
            }

        if isinstance(obj, np.datetime64):
            self._stats["datetime_values_serialized"] += 1
            if self.encode_datetime_as == "int":
                return {
                    "__type__": "datetime64",
                    "dtype": str(obj.dtype),
                    "value": int(obj.astype("datetime64[ns]").astype(np.int64)),
                    "unit": "ns",
                }
            return {
                "__type__": "datetime64",
                "dtype": str(obj.dtype),
                "value": np.datetime_as_string(obj, timezone="naive"),
            }

        if isinstance(obj, np.timedelta64):
            if self.encode_timedelta_as == "string":
                return {"__type__": "timedelta64", "dtype": str(obj.dtype), "value": str(obj)}
            return {
                "__type__": "timedelta64",
                "dtype": str(obj.dtype),
                "value": int(obj.astype("timedelta64[ns]").astype(np.int64)),
                "unit": "ns",
            }

        if isinstance(obj, np.void):
            return {"__type__": "void", "dtype": str(obj.dtype), "value": safe_repr(obj)}

        try:
            value = obj.item()
            if isinstance(value, float):
                value = self._normalize_non_finite(value, type_name="generic scalar")
            return value
        except Exception as exc:
            raise BaseSerializationError.wrap(
                exc,
                message="Failed to serialize numpy scalar value.",
                config=self.numpy_config,
                component="NumpyEncoder",
                operation="_encode_numpy_scalar",
                context={"dtype": str(getattr(obj, "dtype", "unknown"))},
            ) from exc

    def _encode_array(self, obj: np.ndarray) -> Any:
        self._stats["arrays_serialized"] += 1

        if np.ma.isMaskedArray(obj):
            data = np.asarray(obj.filled(np.nan))
            mask = np.asarray(np.ma.getmaskarray(obj))
            return {
                "__type__": "masked_ndarray",
                "dtype": str(data.dtype),
                "shape": [int(dim) for dim in data.shape],
                "data": data.tolist(),
                "mask": mask.tolist(),
            }

        if obj.dtype.kind == "f":
            data_payload = np.vectorize(lambda value: self._normalize_non_finite(float(value), type_name="ndarray float"), otypes=[object])(obj).tolist()
        elif obj.dtype.kind == "c":
            data_payload = [
                {
                    "__type__": "complex",
                    "dtype": str(obj.dtype),
                    "real": self._normalize_non_finite(float(np.real(value)), type_name="complex real"),
                    "imag": self._normalize_non_finite(float(np.imag(value)), type_name="complex imag"),
                }
                for value in obj.flatten().tolist()
            ]
            data_payload = np.asarray(data_payload, dtype=object).reshape(obj.shape).tolist()
        else:
            data_payload = obj.tolist()

        if self.array_format == "list":
            return data_payload

        payload: Dict[str, Any] = {
            "__type__": "ndarray",
            "data": data_payload,
        }
        if self.include_array_dtype:
            payload["dtype"] = str(obj.dtype)
        if self.include_array_shape:
            payload["shape"] = [int(dim) for dim in obj.shape]
        if self.include_array_size:
            payload["size"] = int(obj.size)
        return payload

    def _encode_callable(self, obj: Any) -> Any:
        if not self.serialize_callables:
            raise BaseSerializationError(
                "Callable serialization is disabled by configuration.",
                self.numpy_config,
                component="NumpyEncoder",
                operation="default",
                context={"callable": safe_repr(obj)},
            )
        self._stats["callable_values_serialized"] += 1
        return {
            "__type__": "callable",
            "name": getattr(obj, "__name__", type(obj).__name__),
            "module": getattr(obj, "__module__", None),
        }

    def _encode_custom_object(self, obj: Any) -> Any:
        if not self.serialize_custom_objects:
            if self.strict_numpy_only:
                raise BaseSerializationError(
                    "Custom object serialization is disabled in strict numpy-only mode.",
                    self.numpy_config,
                    component="NumpyEncoder",
                    operation="default",
                    context={"object_type": type(obj).__name__},
                )
            return safe_repr(obj)

        self._stats["custom_objects_serialized"] += 1

        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            try:
                return obj.to_dict()
            except Exception:
                pass

        if self.custom_object_strategy == "repr":
            return {"__type__": "object_repr", "class": type(obj).__name__, "repr": safe_repr(obj)}
        if self.custom_object_strategy == "string":
            return str(obj)

        if hasattr(obj, "__dict__"):
            try:
                return dict(vars(obj))
            except Exception:
                return {"__type__": "object_repr", "class": type(obj).__name__, "repr": safe_repr(obj)}

        return {"__type__": "object_repr", "class": type(obj).__name__, "repr": safe_repr(obj)}

    # ------------------------------------------------------------------
    # JSONEncoder API
    # ------------------------------------------------------------------
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return self._encode_array(obj)
        if isinstance(obj, np.generic):
            return self._encode_numpy_scalar(obj)
        if callable(obj):
            return self._encode_callable(obj)
        if self.strict_numpy_only:
            raise BaseSerializationError(
                "Unsupported non-NumPy object encountered in strict numpy-only mode.",
                self.numpy_config,
                component="NumpyEncoder",
                operation="default",
                context={"object_type": type(obj).__name__},
            )
        return self._encode_custom_object(obj)

    # ------------------------------------------------------------------
    # Dump / load helpers
    # ------------------------------------------------------------------
    def dumps(self, obj: Any, **kwargs: Any) -> str:
        options = dict(kwargs)
        options.setdefault("cls", type(self))
        options.setdefault("ensure_ascii", self.ensure_ascii_default)
        options.setdefault("sort_keys", self.sort_keys_default)
        if "indent" not in options and self.indent_default is not None:
            options["indent"] = self.indent_default

        try:
            text = json.dumps(obj, **options)
            self._stats["encode_operations"] += 1
            self._record("dumps", obj, success=True, size_hint=len(text))
            return text
        except Exception as exc:
            self._stats["failures"] += 1
            self._record("dumps", obj, success=False, detail=str(exc))
            raise BaseSerializationError.wrap(
                exc,
                message="Failed to serialize payload with NumpyEncoder.dumps().",
                config=self.numpy_config,
                component="NumpyEncoder",
                operation="dumps",
                context={"payload_type": type(obj).__name__},
            ) from exc

    def dump(self, obj: Any, fp: Union[TextIO, io.TextIOBase], **kwargs: Any) -> None:
        try:
            text = self.dumps(obj, **kwargs)
            fp.write(text)
        except BaseError:
            raise
        except Exception as exc:
            self._stats["failures"] += 1
            raise BaseIOError.wrap(
                exc,
                message="Failed to write serialized payload to file-like object.",
                config=self.numpy_config,
                component="NumpyEncoder",
                operation="dump",
            ) from exc

    def _decode_tagged_mapping(self, obj: Mapping[str, Any]) -> Any:
        type_name = obj.get("__type__")
        if type_name == "complex":
            self._stats["tagged_payloads_decoded"] += 1
            return complex(obj.get("real", 0.0), obj.get("imag", 0.0))

        if type_name == "np_scalar":
            self._stats["tagged_payloads_decoded"] += 1
            return obj.get("value")

        if type_name == "datetime64":
            self._stats["tagged_payloads_decoded"] += 1
            dtype = str(obj.get("dtype", "datetime64[ns]"))
            value = obj.get("value")
            if isinstance(value, str):
                return np.datetime64(value)
            return np.datetime64(int(value), obj.get("unit", "ns")).astype(dtype)

        if type_name == "timedelta64":
            self._stats["tagged_payloads_decoded"] += 1
            dtype = str(obj.get("dtype", "timedelta64[ns]"))
            value = obj.get("value")
            if isinstance(value, str):
                return np.timedelta64(value)
            return np.timedelta64(int(value), obj.get("unit", "ns")).astype(dtype)

        if type_name == "ndarray":
            self._stats["tagged_payloads_decoded"] += 1
            dtype = obj.get("dtype")
            data = obj.get("data")
            return np.array(data, dtype=dtype) if dtype else np.array(data)

        if type_name == "masked_ndarray":
            self._stats["tagged_payloads_decoded"] += 1
            data = np.array(obj.get("data"), dtype=obj.get("dtype"))
            mask = np.array(obj.get("mask"), dtype=bool)
            return np.ma.array(data, mask=mask)

        return dict(obj)

    def loads(self, value: Union[str, bytes, bytearray], **kwargs: Any) -> Any:
        text = ensure_text(value)
        options = dict(kwargs)

        if self.decode_tagged_payloads and "object_hook" not in options:
            options["object_hook"] = self._decode_tagged_mapping

        try:
            payload = json.loads(text, **options)
            self._stats["decode_operations"] += 1
            self._record("loads", text, success=True, size_hint=len(text))
            return payload
        except Exception as exc:
            self._stats["failures"] += 1
            self._record("loads", text, success=False, detail=str(exc), size_hint=len(text))
            raise BaseSerializationError.wrap(
                exc,
                message="Failed to deserialize payload with NumpyEncoder.loads().",
                config=self.numpy_config,
                component="NumpyEncoder",
                operation="loads",
                context={"text_preview": truncate_string(text, 160)},
            ) from exc

    def load(self, fp: Union[TextIO, io.TextIOBase], **kwargs: Any) -> Any:
        try:
            text = fp.read()
        except Exception as exc:
            raise BaseIOError.wrap(
                exc,
                message="Failed to read JSON text from file-like object.",
                config=self.numpy_config,
                component="NumpyEncoder",
                operation="load",
            ) from exc
        return self.loads(text, **kwargs)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def encode_value(self, obj: Any) -> Any:
        """Return the JSON-serializable Python form of a value."""
        return self.default(obj)

    def save_json(self, path: Union[str, Path], obj: Any, **kwargs: Any) -> str:
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.dumps(obj, **kwargs), encoding="utf-8")
            return str(target)
        except BaseError:
            raise
        except Exception as exc:
            raise BaseIOError.wrap(
                exc,
                message="Failed to save JSON payload to disk.",
                config=self.numpy_config,
                component="NumpyEncoder",
                operation="save_json",
                context={"path": str(target)},
            ) from exc

    def load_json(self, path: Union[str, Path], **kwargs: Any) -> Any:
        target = Path(path)
        try:
            text = target.read_text(encoding="utf-8")
        except Exception as exc:
            raise BaseIOError.wrap(
                exc,
                message="Failed to load JSON payload from disk.",
                config=self.numpy_config,
                component="NumpyEncoder",
                operation="load_json",
                context={"path": str(target)},
            ) from exc
        return self.loads(text, **kwargs)

    def recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        count = coerce_int(limit, default=20, minimum=1)
        return [item.to_dict() for item in list(self._history)[-count:]]

    def stats(self) -> Dict[str, Any]:
        return NumpyEncoderStats(
            encode_operations=int(self._stats.get("encode_operations", 0)),
            decode_operations=int(self._stats.get("decode_operations", 0)),
            failures=int(self._stats.get("failures", 0)),
            arrays_serialized=int(self._stats.get("arrays_serialized", 0)),
            scalars_serialized=int(self._stats.get("scalars_serialized", 0)),
            complex_values_serialized=int(self._stats.get("complex_values_serialized", 0)),
            datetime_values_serialized=int(self._stats.get("datetime_values_serialized", 0)),
            callable_values_serialized=int(self._stats.get("callable_values_serialized", 0)),
            custom_objects_serialized=int(self._stats.get("custom_objects_serialized", 0)),
            history_length=len(self._history),
            tagged_payloads_decoded=int(self._stats.get("tagged_payloads_decoded", 0)),
        ).to_dict()

    def __repr__(self) -> str:
        return (
            f"<NumpyEncoder array_format='{self.array_format}' "
            f"decode_tagged_payloads={self.decode_tagged_payloads} "
            f"history_length={len(self._history)}>"
        )


if __name__ == "__main__":
    print("\n=== Running Numpy Encoder ===\n")
    printer.status("TEST", "Numpy Encoder initialized", "info")

    encoder = NumpyEncoder()

    sample_payload = {
        "vector": np.array([1, 2, 3], dtype=np.int32),
        "matrix": np.array([[1.0, np.nan], [np.inf, 4.5]], dtype=np.float64),
        "complex": np.complex64(2 + 3j),
        "datetime": np.datetime64("2026-04-24T12:30:00"),
        "flag": np.bool_(True),
        "callable": len,
    }

    encoded_text = encoder.dumps(sample_payload, indent=2)
    decoded_payload = encoder.loads(encoded_text)

    printer.pretty("ENCODED_TEXT", encoded_text, "success")
    printer.pretty("DECODED_TYPES", {
        "vector": type(decoded_payload["vector"]).__name__,
        "matrix": type(decoded_payload["matrix"]).__name__,
        "complex": type(decoded_payload["complex"]).__name__,
        "datetime": type(decoded_payload["datetime"]).__name__,
    }, "success")
    printer.pretty("ENCODER_STATS", encoder.stats(), "success")
    printer.pretty("RECENT_HISTORY", encoder.recent_history(), "success")

    print("\n=== Test ran successfully ===\n")
