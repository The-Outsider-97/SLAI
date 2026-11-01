"""
Deployment-owned native Autodesk Revit worker boundary for BIMAP.

This module does not parse RVT/RFA bytes in Python.  It executes an explicitly
configured native worker that owns Autodesk Revit/RevitCoreConsole/Design
Automation access.  Two typed adapters are exposed over one hardened process
client:

* ``RevitBackend`` implements the Revit extraction backend expected by
  ``infra.extraction.revit_extractor.RevitDataExtractor``.
* ``RevitConversionBackend`` implements the native backend expected by
  ``infra.conversion.revit_converter.RevitModelConverter``.

The split is deliberate: the extraction and conversion ports expose different
``capabilities`` types and must not be conflated.

Native worker CLI contract
--------------------------
JSON modes::

    <worker> [extra args]
        --mode inspect|extract
        --input <absolute source.rfa|source.rvt>
        --source-format rfa|rvt
        --output <absolute result.json>
        [--datasets elements,properties,quantities,materials]

Binary modes::

    <worker> [extra args]
        --mode convert
        --input <absolute source.rfa|source.rvt>
        --source-format rfa|rvt
        --target-format glb
        --output-stem <safe-stem>
        --output <absolute artifact.glb|artifact.obj>

    <worker> [extra args]
        --mode preview
        --input <absolute source.rfa|source.rvt>
        --source-format rfa|rvt
        --output <absolute preview.png>

The extract result may include a source-specific extension payload::

    {
      "inspection": {...},
      "project": {...},
      "units": [...],
      "datasets": {
        "elements": [...],
        "properties": [...],
        "quantities": [...],
        "materials": [...]
      },
      "counts": {...},
      "ifc_class_counts": {},
      "geometry_summary": {...},
      "extensions": {
        "revit_family": {
          "schema_version": "1.0.0",
          "extractor_version": "...",
          "sections_assessed": [
            "identity", "type_catalog", "parameters", "formulas",
            "materials", "connectors", "nested_components",
            "geometry_metrics", "documentation"
          ],
          "identity": {...},
          "type_catalog": [...],
          "parameters": [...],
          "formulas": [...],
          "materials": [...],
          "connectors": [...],
          "nested_components": [...],
          "geometry_metrics": [...],
          "documentation": [...]
        }
      }
    }

Paths, stdout/stderr, source bytes and evidence payloads are intentionally not
written to BIMAP logs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile

from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, Final

from applications.bimap.app.ports.data_extraction import ( # type: ignore
    DataExtractionCapability,
    DataSourceInspection,
    ExtractedModelData,
    ExtractionDataset,
    ExtractionSourceFormat,
)
from applications.bimap.app.ports.model_conversion import ( # type: ignore
    ConvertedModelArtifact,
    ModelConversionCapability,
    ModelSourceFormat,
    ModelSourceInspection,
    ModelTargetFormat,
)
from applications.bimap.app.utils.app_errors import ( # type: ignore
    AppConfigurationError,
    AppError,
    AppIntegrityError,
    AppPortOperationError,
    AppPortTimeoutError,
    AppPortUnavailableError,
    AppValidationError,
    UnsupportedAppInputError,
)
from logs.logger import PrettyPrinter, get_logger  # type: ignore


logger = get_logger("BIMAP Deployment Revit Backend")
printer = PrettyPrinter()

_COMPONENT: Final[str] = "deployment_revit_backend"

REVIT_EXTRACTOR_EXECUTABLE_ENV: Final[str] = "BIMAP_REVIT_EXTRACTOR_EXECUTABLE"
REVIT_EXTRACTOR_ARGS_ENV: Final[str] = "BIMAP_REVIT_EXTRACTOR_ARGS_JSON"
REVIT_EXTRACTOR_TIMEOUT_ENV: Final[str] = "BIMAP_REVIT_EXTRACTOR_TIMEOUT_SECONDS"
REVIT_EXTRACTOR_MAX_OUTPUT_ENV: Final[str] = "BIMAP_REVIT_EXTRACTOR_MAX_OUTPUT_BYTES"
REVIT_CONVERTER_MAX_OUTPUT_ENV: Final[str] = "BIMAP_REVIT_CONVERTER_MAX_OUTPUT_BYTES"
REVIT_PREVIEW_MAX_OUTPUT_ENV: Final[str] = "BIMAP_REVIT_PREVIEW_MAX_OUTPUT_BYTES"

_DEFAULT_TIMEOUT_SECONDS: Final[int] = 300
_DEFAULT_JSON_MAX_OUTPUT_BYTES: Final[int] = 64 * 1024 * 1024
_DEFAULT_CONVERSION_MAX_OUTPUT_BYTES: Final[int] = 256 * 1024 * 1024
_DEFAULT_PREVIEW_MAX_OUTPUT_BYTES: Final[int] = 16 * 1024 * 1024
_SAFE_STEM: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]+")

_EXTRACTION_FORMATS: Final[frozenset[ExtractionSourceFormat]] = frozenset(
    {ExtractionSourceFormat.RFA, ExtractionSourceFormat.RVT}
)
_MODEL_FORMATS: Final[frozenset[ModelSourceFormat]] = frozenset(
    {ModelSourceFormat.RFA, ModelSourceFormat.RVT}
)
_SUPPORTED_DATASETS: Final[tuple[ExtractionDataset, ...]] = (
    ExtractionDataset.ELEMENTS,
    ExtractionDataset.PROPERTIES,
    ExtractionDataset.QUANTITIES,
    ExtractionDataset.MATERIALS,
)
_SUPPORTED_TARGETS: Final[tuple[ModelTargetFormat, ...]] = (
    ModelTargetFormat.GLB,
)


_REVIT_FAMILY_SECTIONS: Final[frozenset[str]] = frozenset(
    {
        "identity",
        "type_catalog",
        "parameters",
        "formulas",
        "materials",
        "connectors",
        "nested_components",
        "geometry_metrics",
        "documentation",
    }
)
_REVIT_FAMILY_ROW_SECTIONS: Final[tuple[str, ...]] = (
    "type_catalog",
    "parameters",
    "formulas",
    "materials",
    "connectors",
    "nested_components",
    "geometry_metrics",
    "documentation",
)


def _normalize_revit_extensions(value: Mapping[str, Any], *, source_format: ExtractionSourceFormat) -> dict[str, Any]:
    """Validate the source-specific native Revit extension envelope.

    Absence is allowed for backwards compatibility and causes semantic rules to
    report unknown/not-assessed rather than fabricating Revit state.  When the
    envelope is present, its structural contract is strict.
    """
    result = dict(value)
    family = result.get("revit_family")

    if source_format is not ExtractionSourceFormat.RFA:
        if family is not None:
            raise AppIntegrityError(
                "revit_family extension is valid for RFA extraction only.",
                component=_COMPONENT,
                operation="validate_revit_extensions",
                field="extensions.revit_family",
                context={"source_format": source_format.value},
            )
        return result

    if family is None:
        return result
    if not isinstance(family, Mapping):
        raise AppIntegrityError(
            "extensions.revit_family must be a JSON object.",
            component=_COMPONENT,
            operation="validate_revit_extensions",
            field="extensions.revit_family",
            context={"received_type": type(family).__name__},
        )

    normalized = dict(family)
    for name in (
        "schema_version",
        "extractor_version",
        "revit_engine_version",
        "source_revit_version",
    ):
        raw = normalized.get(name)
        if raw is not None and (not isinstance(raw, str) or not raw.strip()):
            raise AppIntegrityError(
                "Revit family extension metadata values must be non-empty strings.",
                component=_COMPONENT,
                operation="validate_revit_extensions",
                field=f"extensions.revit_family.{name}",
            )
        if isinstance(raw, str):
            normalized[name] = raw.strip()

    assessed = normalized.get("sections_assessed")
    if assessed is not None:
        if isinstance(assessed, (str, bytes, bytearray, Mapping)):
            raise AppIntegrityError(
                "extensions.revit_family.sections_assessed must be an array of section names.",
                component=_COMPONENT,
                operation="validate_revit_extensions",
                field="extensions.revit_family.sections_assessed",
            )
        try:
            raw_sections = tuple(assessed)
        except TypeError as exc:
            raise AppIntegrityError(
                "extensions.revit_family.sections_assessed must be iterable.",
                component=_COMPONENT,
                operation="validate_revit_extensions",
                field="extensions.revit_family.sections_assessed",
                cause=exc,
            ) from exc

        section_names: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_sections):
            if not isinstance(item, str) or not item.strip():
                raise AppIntegrityError(
                    "sections_assessed entries must be non-empty strings.",
                    component=_COMPONENT,
                    operation="validate_revit_extensions",
                    field=f"extensions.revit_family.sections_assessed[{index}]",
                )
            section = item.strip()
            if section not in _REVIT_FAMILY_SECTIONS:
                raise AppIntegrityError(
                    "Native Revit worker reported an unknown family section.",
                    component=_COMPONENT,
                    operation="validate_revit_extensions",
                    field=f"extensions.revit_family.sections_assessed[{index}]",
                    context={
                        "received": section,
                        "allowed": tuple(sorted(_REVIT_FAMILY_SECTIONS)),
                    },
                )
            if section not in seen:
                seen.add(section)
                section_names.append(section)
        normalized["sections_assessed"] = section_names

    identity = normalized.get("identity")
    if identity is not None and not isinstance(identity, Mapping):
        raise AppIntegrityError(
            "extensions.revit_family.identity must be a JSON object.",
            component=_COMPONENT,
            operation="validate_revit_extensions",
            field="extensions.revit_family.identity",
            context={"received_type": type(identity).__name__},
        )
    if isinstance(identity, Mapping):
        normalized["identity"] = dict(identity)

    for section in _REVIT_FAMILY_ROW_SECTIONS:
        raw_rows = normalized.get(section)
        if raw_rows is None:
            continue
        normalized[section] = [
            dict(row)
            for row in _mapping_rows(
                raw_rows,
                field=f"extensions.revit_family.{section}",
                operation="validate_revit_extensions",
            )
        ]

    result["revit_family"] = normalized
    return result


def _positive_int(value: Any, *, field: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise AppConfigurationError(
                f"{field} is required.",
                component=_COMPONENT,
                operation="configure",
                field=field,
            )
        return default

    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise AppConfigurationError(
            f"{field} must be a positive integer.",
            component=_COMPONENT,
            operation="configure",
            field=field,
            cause=exc,
        ) from exc

    if parsed <= 0:
        raise AppConfigurationError(
            f"{field} must be greater than zero.",
            component=_COMPONENT,
            operation="configure",
            field=field,
            context={"received": parsed},
        )
    return parsed


def _resolve_executable(value: str) -> Path:
    raw = str(value).strip()
    if not raw:
        raise AppConfigurationError(
            "Native Revit worker executable is not configured.",
            component=_COMPONENT,
            operation="configure",
            field=REVIT_EXTRACTOR_EXECUTABLE_ENV,
        )

    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    discovered = shutil.which(raw)
    if discovered:
        resolved = Path(discovered).resolve()
        if resolved.is_file():
            return resolved

    raise AppConfigurationError(
        "Configured native Revit worker executable does not exist or is not discoverable on PATH.",
        component=_COMPONENT,
        operation="configure",
        field=REVIT_EXTRACTOR_EXECUTABLE_ENV,
    )


def _parse_extra_args(value: str | None) -> tuple[str, ...]:
    if value is None or not value.strip():
        return ()

    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AppConfigurationError(
            f"{REVIT_EXTRACTOR_ARGS_ENV} must contain a JSON array of strings.",
            component=_COMPONENT,
            operation="configure",
            field=REVIT_EXTRACTOR_ARGS_ENV,
            cause=exc,
        ) from exc

    if not isinstance(decoded, list):
        raise AppConfigurationError(
            f"{REVIT_EXTRACTOR_ARGS_ENV} must contain a JSON array.",
            component=_COMPONENT,
            operation="configure",
            field=REVIT_EXTRACTOR_ARGS_ENV,
        )

    result: list[str] = []
    for index, item in enumerate(decoded):
        if not isinstance(item, str) or not item.strip():
            raise AppConfigurationError(
                "Native Revit worker arguments must be non-empty strings.",
                component=_COMPONENT,
                operation="configure",
                field=f"{REVIT_EXTRACTOR_ARGS_ENV}[{index}]",
                context={"received_type": type(item).__name__},
            )
        result.append(item.strip())
    return tuple(result)


def _safe_output_stem(value: str) -> str:
    stem = _SAFE_STEM.sub("-", str(value).strip()).strip("._-")
    return (stem or "revit-model")[:120]


def _mapping(value: Any, *, field: str, operation: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AppIntegrityError(
            "Native Revit worker returned an invalid JSON object shape.",
            component=_COMPONENT,
            operation=operation,
            field=field,
            context={"received_type": type(value).__name__},
        )
    return value


def _mapping_rows(value: Any, *, field: str, operation: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise AppIntegrityError(
            "Native Revit extraction dataset must be a JSON array.",
            component=_COMPONENT,
            operation=operation,
            field=field,
            context={"received_type": type(value).__name__},
        )

    rows: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise AppIntegrityError(
                "Native Revit extraction dataset rows must be JSON objects.",
                component=_COMPONENT,
                operation=operation,
                field=f"{field}[{index}]",
                context={"received_type": type(item).__name__},
            )
        rows.append(dict(item))
    return tuple(rows)


def _validate_source_path(path: Path, *, expected_suffix: str, operation: str) -> Path:
    if not isinstance(path, Path):
        raise UnsupportedAppInputError(
            "Native Revit backend requires pathlib.Path source input.",
            component=_COMPONENT,
            operation=operation,
            field="path",
            context={"received_type": type(path).__name__},
        )

    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AppValidationError(
            "Native Revit source file does not exist or cannot be resolved.",
            component=_COMPONENT,
            operation=operation,
            field="path",
            cause=exc,
        ) from exc

    if not resolved.is_file():
        raise AppValidationError(
            "Native Revit source must be a regular file.",
            component=_COMPONENT,
            operation=operation,
            field="path",
        )

    if resolved.stat().st_size <= 0:
        raise AppValidationError(
            "Native Revit source file cannot be empty.",
            component=_COMPONENT,
            operation=operation,
            field="path",
        )

    if resolved.suffix.casefold() != expected_suffix.casefold():
        raise AppValidationError(
            "Native Revit source extension does not match source_format.",
            component=_COMPONENT,
            operation=operation,
            field="path",
            context={
                "expected_extension": expected_suffix.casefold(),
                "received_extension": resolved.suffix.casefold(),
            },
        )
    return resolved


class _NativeRevitWorker:
    """Hardened process client shared by extraction and conversion adapters."""

    __slots__ = (
        "_executable",
        "_extra_args",
        "_timeout_seconds",
        "_max_json_output_bytes",
        "_max_conversion_output_bytes",
        "_max_preview_output_bytes",
    )

    def __init__(
        self,
        executable: str | Path,
        *,
        extra_args: Sequence[str] = (),
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        max_json_output_bytes: int = _DEFAULT_JSON_MAX_OUTPUT_BYTES,
        max_conversion_output_bytes: int = _DEFAULT_CONVERSION_MAX_OUTPUT_BYTES,
        max_preview_output_bytes: int = _DEFAULT_PREVIEW_MAX_OUTPUT_BYTES,
    ) -> None:
        self._executable = _resolve_executable(str(executable))

        if isinstance(extra_args, (str, bytes, bytearray, Mapping)):
            raise AppConfigurationError(
                "extra_args must be a sequence of command arguments.",
                component=_COMPONENT,
                operation="initialize",
                field="extra_args",
            )

        normalized_args: list[str] = []
        for index, item in enumerate(extra_args):
            if not isinstance(item, str) or not item.strip():
                raise AppConfigurationError(
                    "extra_args must contain non-empty strings only.",
                    component=_COMPONENT,
                    operation="initialize",
                    field=f"extra_args[{index}]",
                )
            normalized_args.append(item.strip())

        self._extra_args = tuple(normalized_args)
        self._timeout_seconds = _positive_int(
            timeout_seconds,
            field="timeout_seconds",
        )
        self._max_json_output_bytes = _positive_int(
            max_json_output_bytes,
            field="max_json_output_bytes",
        )
        self._max_conversion_output_bytes = _positive_int(
            max_conversion_output_bytes,
            field="max_conversion_output_bytes",
        )
        self._max_preview_output_bytes = _positive_int(
            max_preview_output_bytes,
            field="max_preview_output_bytes",
        )

    @classmethod
    def from_env(cls) -> "_NativeRevitWorker":
        executable = os.getenv(REVIT_EXTRACTOR_EXECUTABLE_ENV)
        if executable is None or not executable.strip():
            raise AppConfigurationError(
                "Native Revit processing was requested but no worker executable is configured.",
                component=_COMPONENT,
                operation="from_env",
                field=REVIT_EXTRACTOR_EXECUTABLE_ENV,
            )

        return cls(
            executable.strip(),
            extra_args=_parse_extra_args(os.getenv(REVIT_EXTRACTOR_ARGS_ENV)),
            timeout_seconds=_positive_int(
                os.getenv(REVIT_EXTRACTOR_TIMEOUT_ENV),
                field=REVIT_EXTRACTOR_TIMEOUT_ENV,
                default=_DEFAULT_TIMEOUT_SECONDS,
            ),
            max_json_output_bytes=_positive_int(
                os.getenv(REVIT_EXTRACTOR_MAX_OUTPUT_ENV),
                field=REVIT_EXTRACTOR_MAX_OUTPUT_ENV,
                default=_DEFAULT_JSON_MAX_OUTPUT_BYTES,
            ),
            max_conversion_output_bytes=_positive_int(
                os.getenv(REVIT_CONVERTER_MAX_OUTPUT_ENV),
                field=REVIT_CONVERTER_MAX_OUTPUT_ENV,
                default=_DEFAULT_CONVERSION_MAX_OUTPUT_BYTES,
            ),
            max_preview_output_bytes=_positive_int(
                os.getenv(REVIT_PREVIEW_MAX_OUTPUT_ENV),
                field=REVIT_PREVIEW_MAX_OUTPUT_ENV,
                default=_DEFAULT_PREVIEW_MAX_OUTPUT_BYTES,
            ),
        )

    def _execute(
        self,
        command: list[str],
        *,
        operation: str,
        source_format: str,
    ) -> None:
        try:
            completed = subprocess.run(
                command,
                shell=False,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise AppPortTimeoutError(
                "Native Revit worker exceeded its timeout.",
                component=_COMPONENT,
                operation=operation,
                context={
                    "source_format": source_format,
                    "timeout_seconds": self._timeout_seconds,
                },
                cause=exc,
            ) from exc
        except OSError as exc:
            raise AppPortUnavailableError(
                "Native Revit worker could not be started.",
                component=_COMPONENT,
                operation=operation,
                context={"source_format": source_format},
                cause=exc,
            ) from exc

        if completed.returncode != 0:
            raise AppPortOperationError(
                "Native Revit worker returned a failure status.",
                component=_COMPONENT,
                operation=operation,
                context={
                    "source_format": source_format,
                    "return_code": completed.returncode,
                },
            )

    def run_json(
        self,
        path: Path,
        *,
        source_format: str,
        mode: str,
        datasets: Sequence[str] = (),
    ) -> Mapping[str, Any]:
        operation = f"native_{mode}"
        source_path = _validate_source_path(
            path,
            expected_suffix=f".{source_format}",
            operation=operation,
        )

        with tempfile.TemporaryDirectory(prefix="bimap-revit-native-") as directory_name:
            output_path = Path(directory_name) / "result.json"
            command = [
                str(self._executable),
                *self._extra_args,
                "--mode",
                mode,
                "--input",
                str(source_path),
                "--source-format",
                source_format,
                "--output",
                str(output_path),
            ]
            if datasets:
                command.extend(("--datasets", ",".join(datasets)))

            self._execute(
                command,
                operation=operation,
                source_format=source_format,
            )

            if not output_path.is_file():
                raise AppIntegrityError(
                    "Native Revit worker produced no JSON result.",
                    component=_COMPONENT,
                    operation=operation,
                    field="output",
                )
            output_size = output_path.stat().st_size
            if output_size <= 0:
                raise AppIntegrityError(
                    "Native Revit worker produced an empty JSON result.",
                    component=_COMPONENT,
                    operation=operation,
                    field="output",
                )
            if output_size > self._max_json_output_bytes:
                raise AppIntegrityError(
                    "Native Revit worker JSON result exceeds the configured maximum size.",
                    component=_COMPONENT,
                    operation=operation,
                    field="output",
                    context={
                        "result_size_bytes": output_size,
                        "max_output_bytes": self._max_json_output_bytes,
                    },
                )

            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise AppIntegrityError(
                    "Native Revit worker returned invalid UTF-8 JSON.",
                    component=_COMPONENT,
                    operation=operation,
                    field="output",
                    cause=exc,
                ) from exc

        return _mapping(payload, field="result", operation=operation)

    def run_binary(
        self,
        path: Path,
        *,
        source_format: str,
        mode: str,
        extension: str,
        target_format: str | None = None,
        output_stem: str | None = None,
    ) -> bytes:
        operation = f"native_{mode}"
        source_path = _validate_source_path(
            path,
            expected_suffix=f".{source_format}",
            operation=operation,
        )

        if not extension.startswith("."):
            extension = f".{extension}"

        with tempfile.TemporaryDirectory(prefix="bimap-revit-native-") as directory_name:
            output_path = Path(directory_name) / f"artifact{extension.casefold()}"
            command = [
                str(self._executable),
                *self._extra_args,
                "--mode",
                mode,
                "--input",
                str(source_path),
                "--source-format",
                source_format,
                "--output",
                str(output_path),
            ]
            if target_format is not None:
                command.extend(("--target-format", target_format))
            if output_stem is not None:
                command.extend(("--output-stem", _safe_output_stem(output_stem)))

            self._execute(
                command,
                operation=operation,
                source_format=source_format,
            )

            if not output_path.is_file():
                raise AppIntegrityError(
                    "Native Revit worker produced no binary artifact.",
                    component=_COMPONENT,
                    operation=operation,
                    field="output",
                )
            output_size = output_path.stat().st_size
            if output_size <= 0:
                raise AppIntegrityError(
                    "Native Revit worker produced an empty binary artifact.",
                    component=_COMPONENT,
                    operation=operation,
                    field="output",
                )
            max_output_bytes = (
                self._max_preview_output_bytes
                if mode == "preview"
                else self._max_conversion_output_bytes
            )
            if output_size > max_output_bytes:
                raise AppIntegrityError(
                    "Native Revit worker artifact exceeds the configured maximum size.",
                    component=_COMPONENT,
                    operation=operation,
                    field="output",
                    context={
                        "result_size_bytes": output_size,
                        "max_output_bytes": max_output_bytes,
                    },
                )
            try:
                return output_path.read_bytes()
            except OSError as exc:
                raise AppIntegrityError(
                    "Native Revit worker artifact could not be read.",
                    component=_COMPONENT,
                    operation=operation,
                    field="output",
                    cause=exc,
                ) from exc


class RevitBackend:
    """Native RVT/RFA structured extraction backend."""

    __slots__ = ("_worker", "_capabilities")

    def __init__(
        self,
        executable: str | Path | None = None,
        *,
        extra_args: Sequence[str] = (),
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        max_output_bytes: int = _DEFAULT_JSON_MAX_OUTPUT_BYTES,
        worker: _NativeRevitWorker | None = None,
    ) -> None:
        printer.status("BIMAP", "Initializing native Revit extraction backend", "info")
        if worker is not None and not isinstance(worker, _NativeRevitWorker):
            raise AppConfigurationError(
                "worker must be a _NativeRevitWorker or None.",
                component=_COMPONENT,
                operation="initialize",
                field="worker",
            )
        if worker is None:
            if executable is None:
                raise AppConfigurationError(
                    "Native Revit extraction requires an executable or shared worker.",
                    component=_COMPONENT,
                    operation="initialize",
                    field="executable",
                )
            worker = _NativeRevitWorker(
                executable,
                extra_args=extra_args,
                timeout_seconds=timeout_seconds,
                max_json_output_bytes=max_output_bytes,
            )
        self._worker = worker
        self._capabilities = (
            DataExtractionCapability(
                source_format=ExtractionSourceFormat.RFA,
                extensions=(".rfa",),
                datasets=_SUPPORTED_DATASETS,
            ),
            DataExtractionCapability(
                source_format=ExtractionSourceFormat.RVT,
                extensions=(".rvt",),
                datasets=_SUPPORTED_DATASETS,
            ),
        )

    @classmethod
    def from_env(cls) -> "RevitBackend":
        return cls(worker=_NativeRevitWorker.from_env())

    @property
    def worker(self) -> _NativeRevitWorker:
        return self._worker

    @property
    def capabilities(self) -> tuple[DataExtractionCapability, ...]:
        return self._capabilities

    @staticmethod
    def _source_format(value: ExtractionSourceFormat | str) -> ExtractionSourceFormat:
        source = ExtractionSourceFormat.parse(value)
        if source not in _EXTRACTION_FORMATS:
            raise UnsupportedAppInputError(
                "Native Revit backend supports RVT/RFA sources only.",
                component=_COMPONENT,
                operation="resolve_source_format",
                field="source_format",
                context={"source_format": source.value},
            )
        return source

    @staticmethod
    def _inspection(
        payload: Mapping[str, Any],
        *,
        expected_format: ExtractionSourceFormat,
        operation: str,
    ) -> DataSourceInspection:
        data = _mapping(
            payload.get("inspection"),
            field="inspection",
            operation=operation,
        )
        try:
            inspection = DataSourceInspection(
                source_format=data.get("source_format"),
                schema=data.get("schema"),
                product_count=data.get("product_count"),
                project_name=data.get("project_name"),
            )
        except AppError:
            raise
        except Exception as exc:
            raise AppIntegrityError(
                "Native Revit inspection payload is invalid.",
                component=_COMPONENT,
                operation=operation,
                field="inspection",
                context={"error_type": type(exc).__name__},
                cause=exc,
            ) from exc

        if inspection.source_format is not expected_format:
            raise AppIntegrityError(
                "Native Revit worker reported the wrong source format.",
                component=_COMPONENT,
                operation=operation,
                field="inspection.source_format",
                context={
                    "expected": expected_format.value,
                    "received": inspection.source_format.value,
                },
            )
        return inspection

    def inspect_file(
        self,
        path: Path,
        *,
        source_format: ExtractionSourceFormat,
    ) -> DataSourceInspection:
        source = self._source_format(source_format)
        payload = self._worker.run_json(
            path,
            source_format=source.value,
            mode="inspect",
        )
        return self._inspection(
            payload,
            expected_format=source,
            operation="inspect_file",
        )

    def extract_file(
        self,
        path: Path,
        *,
        source_format: ExtractionSourceFormat,
        datasets: tuple[ExtractionDataset, ...],
    ) -> ExtractedModelData:
        source = self._source_format(source_format)

        selected: list[ExtractionDataset] = []
        for raw in datasets:
            dataset = ExtractionDataset.parse(raw)
            if dataset not in _SUPPORTED_DATASETS:
                raise UnsupportedAppInputError(
                    "Requested Revit extraction dataset is unsupported.",
                    component=_COMPONENT,
                    operation="extract_file",
                    field="datasets",
                    context={"dataset": dataset.value},
                )
            if dataset not in selected:
                selected.append(dataset)

        if not selected:
            raise AppValidationError(
                "At least one Revit extraction dataset is required.",
                component=_COMPONENT,
                operation="extract_file",
                field="datasets",
            )

        payload = self._worker.run_json(
            path,
            source_format=source.value,
            mode="extract",
            datasets=tuple(item.value for item in selected),
        )
        inspection = self._inspection(
            payload,
            expected_format=source,
            operation="extract_file",
        )
        project = _mapping(payload.get("project"), field="project", operation="extract_file")

        raw_units = payload.get("units")
        if not isinstance(raw_units, list):
            raise AppIntegrityError(
                "Native Revit extraction units must be a JSON array.",
                component=_COMPONENT,
                operation="extract_file",
                field="units",
                context={"received_type": type(raw_units).__name__},
            )
        units = _mapping_rows(raw_units, field="units", operation="extract_file")

        raw_dataset_map = _mapping(
            payload.get("datasets"),
            field="datasets",
            operation="extract_file",
        )
        dataset_map: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for dataset in selected:
            if dataset.value not in raw_dataset_map:
                raise AppIntegrityError(
                    "Native Revit worker omitted a requested dataset.",
                    component=_COMPONENT,
                    operation="extract_file",
                    field=f"datasets.{dataset.value}",
                    context={"dataset": dataset.value},
                )
            dataset_map[dataset.value] = _mapping_rows(
                raw_dataset_map[dataset.value],
                field=f"datasets.{dataset.value}",
                operation="extract_file",
            )

        counts = _mapping(payload.get("counts"), field="counts", operation="extract_file")
        ifc_class_counts = _mapping(
            payload.get("ifc_class_counts", {}),
            field="ifc_class_counts",
            operation="extract_file",
        )
        geometry_summary = _mapping(
            payload.get("geometry_summary", {}),
            field="geometry_summary",
            operation="extract_file",
        )
        raw_extensions = _mapping(
            payload.get("extensions", {}),
            field="extensions",
            operation="extract_file",
        )
        extensions = _normalize_revit_extensions(
            raw_extensions,
            source_format=source,
        )

        try:
            result = ExtractedModelData(
                inspection=inspection,
                project=dict(project),
                units=tuple(units),
                datasets=dataset_map,
                counts=dict(counts),
                ifc_class_counts=dict(ifc_class_counts),
                geometry_summary=dict(geometry_summary),
                extensions=dict(extensions),
            )
        except AppError:
            raise
        except Exception as exc:
            raise AppIntegrityError(
                "Native Revit extraction result does not satisfy BIMAP's ExtractedModelData contract.",
                component=_COMPONENT,
                operation="extract_file",
                field="result",
                context={"error_type": type(exc).__name__},
                cause=exc,
            ) from exc

        logger.info(
            {
                "event": "native_revit_extraction_completed",
                "source_format": source.value,
                "dataset_count": len(dataset_map),
                "has_geometry_summary": bool(result.geometry_summary),
                "has_revit_family_extension": "revit_family" in result.extensions,
            }
        )
        return result

    def render_preview_file(self, path: Path, *, source_format: ExtractionSourceFormat) -> bytes | None:
        source = self._source_format(source_format)
        try:
            return self._worker.run_binary(
                path,
                source_format=source.value,
                mode="preview",
                extension=".png",
            )
        except AppPortOperationError:
            logger.info(
                {
                    "event": "native_revit_preview_unavailable",
                    "source_format": source.value,
                }
            )
            return None


class RevitConversionBackend:
    """Native Revit RFA/RVT -> GLB conversion backend for spatial navigation."""

    __slots__ = ("_worker", "_capabilities")

    def __init__(self, worker: _NativeRevitWorker) -> None:
        if not isinstance(worker, _NativeRevitWorker):
            raise AppConfigurationError(
                "worker must be a _NativeRevitWorker.",
                component=_COMPONENT,
                operation="initialize_conversion_backend",
                field="worker",
            )
        self._worker = worker
        self._capabilities = tuple(
            ModelConversionCapability(
                source_format=source,
                extensions=(f".{source.value}",),
                target_formats=_SUPPORTED_TARGETS,
            )
            for source in (ModelSourceFormat.RFA, ModelSourceFormat.RVT)
        )

    @classmethod
    def from_env(cls) -> "RevitConversionBackend":
        return cls(_NativeRevitWorker.from_env())

    @property
    def capabilities(self) -> tuple[ModelConversionCapability, ...]:
        return self._capabilities

    @staticmethod
    def _source(value: ModelSourceFormat | str) -> ModelSourceFormat:
        source = ModelSourceFormat.parse(value)
        if source not in _MODEL_FORMATS:
            raise UnsupportedAppInputError(
                "Native Revit conversion backend supports RFA/RVT sources only.",
                component=_COMPONENT,
                operation="resolve_conversion_source",
                field="source_format",
                context={"source_format": source.value},
            )
        return source

    def inspect_file(self, path: Path, *, source_format: ModelSourceFormat) -> ModelSourceInspection:
        source = self._source(source_format)
        payload = self._worker.run_json(
            path,
            source_format=source.value,
            mode="inspect",
        )
        raw = _mapping(
            payload.get("inspection"),
            field="inspection",
            operation="inspect_conversion_source",
        )
        reported = ModelSourceFormat.parse(raw.get("source_format"))
        if reported is not source:
            raise AppIntegrityError(
                "Native Revit worker reported the wrong conversion source format.",
                component=_COMPONENT,
                operation="inspect_conversion_source",
                field="inspection.source_format",
                context={"expected": source.value, "received": reported.value},
            )
        return ModelSourceInspection(
            source_format=source,
            schema=raw.get("schema"),
            product_count=raw.get("product_count"),
        )

    def convert_file(
        self,
        path: Path,
        *,
        source_format: ModelSourceFormat,
        target_format: ModelTargetFormat,
        output_stem: str,
    ) -> ConvertedModelArtifact:
        source = self._source(source_format)
        target = ModelTargetFormat.parse(target_format)
        if target not in _SUPPORTED_TARGETS:
            raise UnsupportedAppInputError(
                "Native Revit conversion target is unsupported.",
                component=_COMPONENT,
                operation="convert_file",
                field="target_format",
                context={"target_format": target.value},
            )

        stem = _safe_output_stem(output_stem)
        payload = self._worker.run_binary(
            path,
            source_format=source.value,
            mode="convert",
            extension=f".{target.value}",
            target_format=target.value,
            output_stem=stem,
        )
        digest = hashlib.sha256(payload).hexdigest()
        content_type = (
            "model/gltf-binary"
            if target is ModelTargetFormat.GLB
            else "text/plain"
        )
        artifact = ConvertedModelArtifact(
            stream=BytesIO(payload),
            filename=f"{stem}.{target.value}",
            content_type=content_type,
            size_bytes=len(payload),
            content_hash=digest,
            hash_algorithm="sha256",
        )
        logger.info(
            {
                "event": "native_revit_conversion_completed",
                "source_format": source.value,
                "target_format": target.value,
                "size_bytes": artifact.size_bytes,
            }
        )
        return artifact


def create_revit_backends_from_env() -> tuple[RevitBackend, RevitConversionBackend]:
    """Create extraction and conversion adapters over one native worker process client."""
    worker = _NativeRevitWorker.from_env()
    return RevitBackend(worker=worker), RevitConversionBackend(worker)


__all__ = [
    "REVIT_EXTRACTOR_EXECUTABLE_ENV",
    "REVIT_EXTRACTOR_ARGS_ENV",
    "REVIT_EXTRACTOR_TIMEOUT_ENV",
    "REVIT_EXTRACTOR_MAX_OUTPUT_ENV",
    "REVIT_CONVERTER_MAX_OUTPUT_ENV",
    "REVIT_PREVIEW_MAX_OUTPUT_ENV",
    "RevitBackend",
    "RevitConversionBackend",
    "create_revit_backends_from_env",
]


if __name__ == "__main__":
    print(
        "Revit backend adapters require a native Autodesk worker configured via "
        f"{REVIT_EXTRACTOR_EXECUTABLE_ENV}."
    )