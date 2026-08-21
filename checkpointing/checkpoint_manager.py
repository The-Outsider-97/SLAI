"""Production orchestration for SLAI checkpointing.

``CheckpointManager`` coordinates the package's deliberately separate layers:

* codecs serialize and deserialize individual components;
* manifests describe exactly what was committed;
* storage stages and atomically publishes files;
* policy makes deterministic, auditable recovery and retention decisions; and
* observability reports outcomes without becoming business logic.

The manager contains no framework serialization, learned selection heuristic,
global executor, global registry, or import-time logging configuration.  Its
canonical APIs are :meth:`save_components` and :meth:`load_components`.
Compatibility-oriented ``save``/``load`` and format helpers only normalize
arguments before delegating to those canonical paths.
"""

from __future__ import annotations

import base64
import os
import threading
import time

from collections import defaultdict
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .checkpoint_codecs import *
from .checkpoint_errors import *
from .checkpoint_manifest import *
from .checkpoint_observability import *
from .checkpoint_policy import *
from .checkpoint_storage import *
from .checkpoint_types import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]


logger = get_logger("Checkpoint Manager")
printer = PrettyPrinter()

_MANAGER_CODEC_ROOT = "codec_root"
_LEGACY_PAYLOAD_KEYS = {
    "model_state": StandardComponent.MODEL.value,
    "optimizer_state": StandardComponent.OPTIMIZER.value,
    "scheduler_state": StandardComponent.SCHEDULER.value,
    "scaler_state": StandardComponent.SCALER.value,
    "extra_state": StandardComponent.AGENT_STATE.value,
    "rng_state": StandardComponent.RNG.value,
}
_LEGACY_COMPONENT_ALIASES = {
    **_LEGACY_PAYLOAD_KEYS,
    "model_weights": StandardComponent.MODEL.value,
}
_RESTORE_ORDER = {
    StandardComponent.MODEL.value: 0,
    StandardComponent.OPTIMIZER.value: 1,
    StandardComponent.SCHEDULER.value: 2,
    StandardComponent.SCALER.value: 3,
    StandardComponent.TOKENIZER.value: 4,
}
_EXTENSIONS = {
    "torch": ".pt",
    "numpy": ".npz",
    "rng": ".npz",
    "agent-state": ".json",
}


def _normalize_component(value: str) -> str:
    """Normalize only documented v2.2 component aliases."""

    candidate = _LEGACY_COMPONENT_ALIASES.get(value, value)
    try:
        return validate_component_name(candidate)
    except ValueError as exc:
        raise CheckpointConfigurationError(
            str(exc),
            stage=CheckpointStage.VALIDATION,
            component=value if isinstance(value, str) else None,
        ) from exc


def _normalize_format_for_operation(
    value: str | CheckpointFormat,
    operation: CheckpointOperation,
) -> CheckpointFormat:
    try:
        return normalize_format(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointConfigurationError(
            str(exc),
            operation=operation,
            stage=CheckpointStage.VALIDATION,
        ) from exc


def _validate_version_for_operation(
    value: str,
    operation: CheckpointOperation,
) -> str:
    try:
        return validate_version(value)
    except ValueError as exc:
        raise CheckpointConfigurationError(
            str(exc),
            operation=operation,
            stage=CheckpointStage.VALIDATION,
            version=value if isinstance(value, str) else None,
        ) from exc


def _component_segment(component: str) -> str:
    """Return a reversible, cross-platform-safe directory segment."""

    encoded = base64.urlsafe_b64encode(component.encode("utf-8")).decode("ascii")
    # The prefix prevents Windows device names; removing base64 padding avoids
    # trailing dots/spaces and leaves only portable alphanumerics, '-' and '_'.
    return "c-" + encoded.rstrip("=")


def _contextualize(
    error: CheckpointError,
    *,
    operation: CheckpointOperation,
    stage: CheckpointStage | None = None,
    path: Path | None = None,
    version: str | None = None,
    checkpoint_id: str | None = None,
    component: str | None = None,
    committed: bool | None = None,
) -> CheckpointError:
    """Add boundary context without replacing context supplied downstream."""

    changes: dict[str, Any] = {}
    for name, value in (
        ("operation", operation.value),
        ("stage", stage.value if stage is not None else None),
        ("path", path),
        ("version", version),
        ("checkpoint_id", checkpoint_id),
        ("component", component),
        ("committed", committed),
    ):
        if value is not None and getattr(error.context, name) is None:
            changes[name] = value
    return error.with_context(**changes) if changes else error


def _codec_destination(staging: Path, component: str, codec: Any) -> tuple[Path, str]:
    root = PurePosixPath("components", _component_segment(component))
    if bool(getattr(codec, "multi_file", False)):
        relative = root.as_posix()
        return staging.joinpath(*root.parts), relative
    extension = _EXTENSIONS.get(codec.codec_id, ".bin")
    relative = (root / f"payload{extension}").as_posix()
    return staging.joinpath(*PurePosixPath(relative).parts), relative


def _infer_checkpoint_format(codec_ids: set[str], component_count: int) -> CheckpointFormat:
    if component_count == 1 and codec_ids == {"torch"}:
        return CheckpointFormat.TORCH
    if component_count == 1 and codec_ids == {"numpy"}:
        return CheckpointFormat.NPZ
    return CheckpointFormat.COMPOSITE


def _record_size(record: CheckpointRecord) -> int:
    return sum(artifact.size_bytes for artifact in record.manifest.artifacts)


def _is_legacy_manifest(manifest: CheckpointManifest) -> bool:
    return bool(
        manifest.metadata.get("legacy_layout") is True
        or manifest.metadata.get("legacy_manifest_schema_version") in {0, 1, 2}
    )


class CheckpointManager:
    """Coordinate durable, deterministic, component-oriented checkpoints.

    Args:
        base_dir: Local checkpoint root.  Mutually exclusive with ``config``
            and ``storage``.
        config: Cross-cutting filesystem and verification configuration.
        storage: Transactional storage adapter.  Extended operations such as
            listing or archival are capability-checked when invoked.
        codecs: Explicit codec registry.  A new frozen default registry is
            created per manager when omitted.
        policy: Pure deterministic policy engine.
        telemetry: Structured observability coordinator.  The default records
            in-process metrics and sends events to no external sink.
        runtime: Capabilities used for compatibility gates.  The default
            declares only the exact versions in ``codecs``; applications that
            persist SLAI, Python, platform, or component-schema constraints
            must supply the corresponding runtime facts.
        producer: Default manifest producer identity.
        create_archive: Whether successful saves create ``tar.gz`` archives.
        retention_limit: Optional compatibility setting for latest-N cleanup.
            It is implemented through :class:`RetentionRules`.
        allow_legacy: Permit discovery and bounded decoding of recognized
            v2.2 layouts.  New saves always use schema v3.
        executor: Optional caller-owned executor for ``save_async``.
    """

    def __init__(
        self,
        base_dir: str | os.PathLike[str] | None = None,
        *,
        config: CheckpointConfig | None = None,
        storage: CheckpointStorageAdapter | None = None,
        codecs: CodecRegistry | None = None,
        policy: CheckpointPolicy | None = None,
        telemetry: CheckpointTelemetry | None = None,
        runtime: RuntimeCapabilities | None = None,
        producer: ProducerInfo | None = None,
        create_archive: bool = False,
        retention_limit: int | None = None,
        allow_legacy: bool = True,
        executor: Executor | None = None,
    ) -> None:
        if base_dir is not None and (config is not None or storage is not None):
            raise CheckpointConfigurationError(
                "base_dir is mutually exclusive with config and storage",
                stage=CheckpointStage.VALIDATION,
            )
        if storage is not None and not isinstance(storage, CheckpointStorageAdapter):
            raise CheckpointConfigurationError(
                "storage does not implement the transactional checkpoint contract",
                stage=CheckpointStage.VALIDATION,
            )
        if storage is not None:
            if config is None:
                active_config = CheckpointConfig(base_dir=storage.base_dir)
            else:
                configured = config.base_dir.resolve(strict=False)
                supplied = Path(storage.base_dir).resolve(strict=False)
                if configured != supplied:
                    raise CheckpointConfigurationError(
                        "config.base_dir does not match storage.base_dir",
                        stage=CheckpointStage.VALIDATION,
                        details={
                            "config_base_dir": str(configured),
                            "storage_base_dir": str(supplied),
                        },
                    )
                active_config = config
            active_storage = storage
        else:
            active_config = config or CheckpointConfig(
                base_dir=Path(base_dir) if base_dir is not None else Path("src/checkpoints")
            )
            active_storage = FileSystemCheckpointStorage(config=active_config)

        active_codecs = codecs or build_default_codec_registry()
        if not isinstance(active_codecs, CodecRegistry):
            raise CheckpointConfigurationError(
                "codecs must be a CodecRegistry",
                stage=CheckpointStage.VALIDATION,
            )
        if policy is not None and not isinstance(policy, CheckpointPolicy):
            raise CheckpointConfigurationError(
                "policy must be a CheckpointPolicy",
                stage=CheckpointStage.VALIDATION,
            )
        if telemetry is not None and not isinstance(telemetry, CheckpointTelemetry):
            raise CheckpointConfigurationError(
                "telemetry must be CheckpointTelemetry",
                stage=CheckpointStage.VALIDATION,
            )
        if not isinstance(create_archive, bool) or not isinstance(allow_legacy, bool):
            raise CheckpointConfigurationError(
                "create_archive and allow_legacy must be booleans",
                stage=CheckpointStage.VALIDATION,
            )
        if retention_limit is not None and (
            isinstance(retention_limit, bool)
            or not isinstance(retention_limit, int)
            or retention_limit < 1
        ):
            raise CheckpointConfigurationError(
                "retention_limit must be a positive integer",
                stage=CheckpointStage.VALIDATION,
            )
        if executor is not None and not callable(getattr(executor, "submit", None)):
            raise CheckpointConfigurationError(
                "executor must expose submit()",
                stage=CheckpointStage.VALIDATION,
            )

        self.config = active_config
        self.storage = active_storage
        self.codecs = active_codecs
        self.policy = policy or CheckpointPolicy()
        self.telemetry = telemetry or CheckpointTelemetry()
        self.runtime = runtime or RuntimeCapabilities(
            codecs=self.codecs.required_codecs()
        )
        self.producer = producer or ProducerInfo(name="SLAI")
        self.create_archive = create_archive
        self.retention_limit = retention_limit
        self.allow_legacy = allow_legacy
        self._executor = executor
        self._owns_executor = False
        self._executor_lock = threading.Lock()

    @property
    def base_dir(self) -> Path:
        return Path(self.storage.base_dir)

    @property
    def default_format(self) -> CheckpointFormat:
        return self.config.default_format

    @property
    def allow_overwrite(self) -> bool:
        return self.config.allow_overwrite

    def _storage_method(self, name: str) -> Any:
        method = getattr(self.storage, name, None)
        if not callable(method):
            raise CheckpointConfigurationError(
                f"storage adapter does not support {name}()",
                stage=CheckpointStage.VALIDATION,
                path=self.base_dir,
                details={"required_capability": name},
            )
        return method

    def _checkpoint_path(self, version: str) -> Path:
        method = getattr(self.storage, "checkpoint_path", None)
        if callable(method):
            return Path(str(method(version)))
        return resolve_checkpoint_path(self.base_dir, version)

    # ------------------------------------------------------------------
    # Save orchestration
    # ------------------------------------------------------------------
    def save_components(
        self,
        components: Mapping[str, Any],
        *,
        version: str | None = None,
        codec_ids: Mapping[str, str] | None = None,
        preferred_codecs: Mapping[str, Sequence[str]] | None = None,
        codec_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        checkpoint_format: str | CheckpointFormat | None = None,
        epoch: int | None = None,
        step: int | None = None,
        metrics: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        provenance: CheckpointProvenance | None = None,
        compatibility: CompatibilityConstraints | None = None,
        producer: ProducerInfo | None = None,
        archive: bool | None = None,
        overwrite: bool | None = None,
        trace_id: str | None = None,
        apply_retention: bool = True,
    ) -> SaveResult:
        """Serialize, verify, and atomically commit named components.

        All components are encoded into a transaction-local staging directory.
        The manifest is written last, after storage-derived sizes and digests
        are known.  No final checkpoint path is exposed before commit.
        """

        if not isinstance(components, Mapping) or not components:
            raise CheckpointConfigurationError(
                "components must be a non-empty mapping",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
            )
        if producer is not None and not isinstance(producer, ProducerInfo):
            raise CheckpointConfigurationError(
                "producer must be ProducerInfo",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
            )
        if not isinstance(apply_retention, bool):
            raise CheckpointConfigurationError(
                "apply_retention must be a boolean",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
            )
        normalized: dict[str, Any] = {}
        for raw_name, value in components.items():
            if not isinstance(raw_name, str):
                raise CheckpointConfigurationError(
                    "component mapping keys must be strings",
                    operation=CheckpointOperation.SAVE,
                    stage=CheckpointStage.VALIDATION,
                )
            name = _normalize_component(raw_name)
            if name in normalized:
                raise CheckpointConfigurationError(
                    "component aliases produce a duplicate canonical name",
                    operation=CheckpointOperation.SAVE,
                    stage=CheckpointStage.VALIDATION,
                    component=name,
                )
            normalized[name] = value

        explicit_codecs = self._normalize_component_mapping(codec_ids, "codec_ids")
        preferences = self._normalize_component_mapping(
            preferred_codecs, "preferred_codecs"
        )
        per_component_metadata = self._normalize_component_mapping(
            codec_metadata, "codec_metadata"
        )
        unknown_settings = (
            set(explicit_codecs) | set(preferences) | set(per_component_metadata)
        ) - set(normalized)
        if unknown_settings:
            raise CheckpointConfigurationError(
                "codec settings refer to components that are not being saved",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
                details={"unknown_components": sorted(unknown_settings)},
            )

        should_archive = self.create_archive if archive is None else archive
        should_overwrite = self.allow_overwrite if overwrite is None else overwrite
        if not isinstance(should_archive, bool) or not isinstance(should_overwrite, bool):
            raise CheckpointConfigurationError(
                "archive and overwrite must be booleans",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
            )

        version_tag = make_version(version)
        checkpoint_id = make_checkpoint_id()
        active_provenance = self._prepare_provenance(
            provenance,
            saved_components=tuple(sorted(normalized)),
            trace_id=trace_id,
        )
        started = time.monotonic()
        staging = None
        committed = False
        archive_path: Path | None = None
        agent_id = active_provenance.agent_id
        run_id = active_provenance.run_id
        effective_trace = active_provenance.trace_id

        with self.telemetry.operation(
            CheckpointOperation.SAVE,
            stage=CheckpointStage.STAGING,
            version=version_tag,
            checkpoint_id=checkpoint_id,
            agent_id=agent_id,
            run_id=run_id,
            trace_id=effective_trace,
            attributes={"components": sorted(normalized)},
        ) as span:
            try:
                staging = self.storage.begin(
                    version_tag, allow_overwrite=should_overwrite
                )
                artifacts: list[CheckpointArtifact] = []
                used_codecs: dict[str, str] = {}

                for component in sorted(normalized):
                    value = normalized[component]
                    raw_preferred = preferences.get(component, ())
                    if isinstance(raw_preferred, (str, bytes)):
                        raise CheckpointConfigurationError(
                            "preferred codec values must be sequences, not strings",
                            operation=CheckpointOperation.SAVE,
                            stage=CheckpointStage.VALIDATION,
                            component=component,
                        )
                    codec = self.codecs.resolve(
                        component,
                        value,
                        codec_id=explicit_codecs.get(component),
                        preferred=tuple(raw_preferred),
                    )
                    used_codecs[codec.codec_id] = codec.codec_version
                    destination, codec_root = _codec_destination(
                        staging.path, component, codec
                    )
                    destination.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                        mode=self.config.directory_mode,
                    )
                    context_metadata = dict(per_component_metadata.get(component, {}))
                    # Storage synchronizes the complete staged tree once.  A
                    # codec-level fsync would duplicate work without improving
                    # the commit boundary.
                    context_metadata["durable"] = False
                    context = CodecContext(
                        checkpoint_id=checkpoint_id,
                        version=version_tag,
                        component=component,
                        metadata=context_metadata,
                    )
                    outputs = tuple(codec.encode(value, destination, context=context))
                    for output in sorted(outputs, key=lambda item: str(item.path)):
                        try:
                            resolved_output = output.path.resolve(strict=True)
                            staging_root = staging.path.resolve(strict=True)
                            relative_path = resolved_output.relative_to(
                                staging_root
                            ).as_posix()
                        except (OSError, ValueError) as exc:
                            raise CheckpointPathError(
                                "codec output is outside the checkpoint staging area",
                                operation=CheckpointOperation.SAVE,
                                stage=CheckpointStage.SERIALIZATION,
                                path=output.path,
                                version=version_tag,
                                checkpoint_id=checkpoint_id,
                                component=component,
                                committed=False,
                            ) from exc
                        resolved_output.chmod(self.config.file_mode)
                        parent = resolved_output.parent
                        while parent != staging_root:
                            parent.chmod(self.config.directory_mode)
                            parent = parent.parent
                        artifact_metadata = dict(output.metadata)
                        if _MANAGER_CODEC_ROOT in artifact_metadata:
                            raise CheckpointManifestError(
                                f"artifact metadata key {_MANAGER_CODEC_ROOT!r} is reserved",
                                operation=CheckpointOperation.SAVE,
                                stage=CheckpointStage.MANIFEST,
                                component=component,
                                committed=False,
                            )
                        artifact_metadata[_MANAGER_CODEC_ROOT] = codec_root
                        artifacts.append(
                            build_artifact(
                                staging.path,
                                relative_path,
                                component=component,
                                codec=codec.codec_id,
                                codec_version=codec.codec_version,
                                media_type=output.media_type,
                                required=output.required,
                                metadata=artifact_metadata,
                            )
                        )

                inferred_format = _infer_checkpoint_format(
                    set(used_codecs), len(normalized)
                )
                if checkpoint_format is None:
                    active_format = inferred_format
                else:
                    try:
                        active_format = normalize_format(checkpoint_format)
                    except ValueError as exc:
                        raise CheckpointConfigurationError(
                            str(exc),
                            operation=CheckpointOperation.SAVE,
                            stage=CheckpointStage.VALIDATION,
                        ) from exc
                    if (
                        active_format is not inferred_format
                        and active_format is not CheckpointFormat.COMPOSITE
                    ):
                        raise CheckpointConfigurationError(
                            "declared checkpoint format does not match the encoded artifacts",
                            operation=CheckpointOperation.SAVE,
                            stage=CheckpointStage.VALIDATION,
                            details={
                                "declared": active_format.value,
                                "inferred": inferred_format.value,
                                "codecs": sorted(used_codecs),
                                "component_count": len(normalized),
                            },
                        )

                active_compatibility = self._prepare_compatibility(
                    compatibility, used_codecs
                )
                manifest = build_manifest(
                    version=version_tag,
                    checkpoint_id=checkpoint_id,
                    artifacts=tuple(sorted(artifacts, key=lambda item: item.relative_path)),
                    checkpoint_format=active_format,
                    producer=producer or self.producer,
                    epoch=epoch,
                    step=step,
                    metrics=metrics,
                    metadata=metadata,
                    provenance=active_provenance,
                    compatibility=active_compatibility,
                )
                manifest_path = write_manifest(
                    staging.path,
                    manifest,
                    verify_artifacts=True,
                    exact_files=self.config.require_exact_files,
                    durable=False,
                    limits=self.config.manifest_limits,
                )
                manifest_path.chmod(self.config.file_mode)
                manifest_size = manifest_path.stat().st_size
                final_path = self.storage.commit(
                    staging, allow_overwrite=should_overwrite
                )
                committed = True
                span.set_result(committed=True)
                record = read_checkpoint_record(
                    final_path,
                    verify=True,
                    exact_files=self.config.require_exact_files,
                    limits=self.config.manifest_limits,
                )
                if record.verification is None or not record.verification.ok:
                    raise CheckpointSaveError(
                        "committed checkpoint failed post-commit verification",
                        operation=CheckpointOperation.SAVE,
                        stage=CheckpointStage.INTEGRITY,
                        path=final_path,
                        version=version_tag,
                        checkpoint_id=checkpoint_id,
                        committed=True,
                    )

                if should_archive:
                    archive_path = Path(
                        self._storage_method("create_archive")(
                            version_tag, overwrite=should_overwrite
                        )
                    )

                if apply_retention and self.retention_limit is not None:
                    rules = RetentionRules(
                        max_checkpoints=self.retention_limit,
                        minimum_keep=1,
                        keep_latest=1,
                        protected_checkpoint_ids=(checkpoint_id,),
                    )
                    plan = self.plan_retention(rules)
                    self.execute_retention(plan, require_constraints_satisfied=True)

                bytes_written = sum(item.size_bytes for item in artifacts)
                bytes_written += manifest_size
                duration = time.monotonic() - started
                span.set_result(
                    checkpoint_id=checkpoint_id,
                    size_bytes=bytes_written,
                    component_count=len(normalized),
                    health=record.health,
                    committed=True,
                    attributes={"format": record.format},
                )
                return SaveResult(
                    record=record,
                    committed=True,
                    bytes_written=bytes_written,
                    duration_seconds=duration,
                    archive_path=archive_path,
                )
            except Exception as exc:
                abort_error: Exception | None = None
                if staging is not None and not committed:
                    try:
                        self.storage.abort(staging)
                    except Exception as cleanup_exc:  # Preserve the primary failure.
                        abort_error = cleanup_exc
                if isinstance(exc, CheckpointError):
                    if committed and exc.committed is False:
                        raise CheckpointSaveError(
                            "checkpoint was committed but a post-commit operation failed",
                            operation=CheckpointOperation.SAVE,
                            stage=CheckpointStage.CLEANUP,
                            path=self._checkpoint_path(version_tag),
                            version=version_tag,
                            checkpoint_id=checkpoint_id,
                            retryable=exc.retryable,
                            committed=True,
                            details={"post_commit_error": exc.to_dict()},
                        ) from exc
                    contextual = _contextualize(
                        exc,
                        operation=CheckpointOperation.SAVE,
                        stage=CheckpointStage.SERIALIZATION,
                        path=staging.path if staging is not None else self.base_dir,
                        version=version_tag,
                        checkpoint_id=checkpoint_id,
                        committed=committed,
                    )
                    if abort_error is not None and not contextual.details.get(
                        "cleanup_error"
                    ):
                        details = dict(contextual.details)
                        details["cleanup_error"] = (
                            f"{type(abort_error).__name__}: {abort_error}"
                        )
                        contextual = contextual.with_context(details=details)
                    raise contextual from exc.__cause__
                raise CheckpointSaveError(
                    "checkpoint save failed",
                    operation=CheckpointOperation.SAVE,
                    stage=CheckpointStage.SERIALIZATION,
                    path=staging.path if staging is not None else self.base_dir,
                    version=version_tag,
                    checkpoint_id=checkpoint_id,
                    retryable=isinstance(exc, OSError),
                    committed=committed,
                    details={
                        "error_type": type(exc).__name__,
                        **(
                            {
                                "cleanup_error": (
                                    f"{type(abort_error).__name__}: {abort_error}"
                                )
                            }
                            if abort_error is not None
                            else {}
                        ),
                    },
                ) from exc

        raise CheckpointSaveError(
            "checkpoint save completed without producing a result",
            operation=CheckpointOperation.SAVE,
            stage=CheckpointStage.CLEANUP,
            version=version_tag,
            checkpoint_id=checkpoint_id,
            committed=committed,
        )

    @staticmethod
    def _normalize_component_mapping(
        value: Mapping[str, Any] | None,
        field_name: str,
    ) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise CheckpointConfigurationError(
                f"{field_name} must be a mapping",
                stage=CheckpointStage.VALIDATION,
            )
        result: dict[str, Any] = {}
        for raw_name, item in value.items():
            if not isinstance(raw_name, str):
                raise CheckpointConfigurationError(
                    f"{field_name} keys must be strings",
                    stage=CheckpointStage.VALIDATION,
                )
            name = _normalize_component(raw_name)
            if name in result:
                raise CheckpointConfigurationError(
                    f"{field_name} contains duplicate canonical component names",
                    stage=CheckpointStage.VALIDATION,
                    component=name,
                )
            result[name] = item
        return result

    @staticmethod
    def _prepare_provenance(
        provenance: CheckpointProvenance | None,
        *,
        saved_components: tuple[str, ...],
        trace_id: str | None,
    ) -> CheckpointProvenance:
        active = provenance or CheckpointProvenance()
        if not isinstance(active, CheckpointProvenance):
            raise CheckpointConfigurationError(
                "provenance must be CheckpointProvenance",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
            )
        if trace_id is not None and active.trace_id not in {None, trace_id}:
            raise CheckpointConfigurationError(
                "trace_id conflicts with provenance.trace_id",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
            )
        if not active.requested_components:
            if active.omitted_components:
                raise CheckpointConfigurationError(
                    "omitted components require an explicit requested capture plan",
                    operation=CheckpointOperation.SAVE,
                    stage=CheckpointStage.VALIDATION,
                )
            active = replace(active, requested_components=saved_components)
        if trace_id is not None and active.trace_id is None:
            active = replace(active, trace_id=trace_id)
        return active

    @staticmethod
    def _prepare_compatibility(
        compatibility: CompatibilityConstraints | None,
        used_codecs: Mapping[str, str],
    ) -> CompatibilityConstraints:
        active = compatibility or CompatibilityConstraints()
        if not isinstance(active, CompatibilityConstraints):
            raise CheckpointConfigurationError(
                "compatibility must be CompatibilityConstraints",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
            )
        conflicts = {
            codec_id: {
                "declared": active.required_codecs[codec_id],
                "actual": actual,
            }
            for codec_id, actual in used_codecs.items()
            if codec_id in active.required_codecs
            and active.required_codecs[codec_id] != actual
        }
        if conflicts:
            raise CheckpointConfigurationError(
                "compatibility constraints conflict with selected codec versions",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.COMPATIBILITY,
                details={"conflicts": conflicts},
            )
        required = dict(active.required_codecs)
        required.update(used_codecs)
        return replace(active, required_codecs=required)

    def save(
        self,
        model: Any = None,
        tokenizer: Any = None,
        metadata: Mapping[str, Any] | None = None,
        version: str | None = None,
        format: str | CheckpointFormat | None = None,
        *,
        components: Mapping[str, Any] | None = None,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
        epoch: int | None = None,
        current_epoch: int | None = None,
        step: int | None = None,
        metrics: Mapping[str, Any] | None = None,
        extra_state: Mapping[str, Any] | None = None,
        archive: bool | None = None,
        overwrite: bool | None = None,
        save_rng: bool = True,
        save_on_cpu: bool = True,
        provenance: CheckpointProvenance | None = None,
        compatibility: CompatibilityConstraints | None = None,
        trace_id: str | None = None,
    ) -> SaveResult:
        """Compatibility wrapper for v2.2-style training-state saves.

        New code should call :meth:`save_components`.  ``components=`` is
        available here only to ease call-site migration and cannot be combined
        with positional training objects.
        """

        if components is not None:
            if model is not None or any(
                value is not None
                for value in (tokenizer, optimizer, scheduler, scaler, extra_state)
            ):
                raise CheckpointConfigurationError(
                    "components cannot be combined with model/tokenizer/training objects",
                    operation=CheckpointOperation.SAVE,
                    stage=CheckpointStage.VALIDATION,
                )
            return self.save_components(
                components,
                version=version,
                checkpoint_format=format,
                epoch=epoch if epoch is not None else current_epoch,
                step=step,
                metrics=metrics,
                metadata=metadata,
                provenance=provenance,
                compatibility=compatibility,
                archive=archive,
                overwrite=overwrite,
                trace_id=trace_id,
            )
        if model is None:
            raise CheckpointConfigurationError(
                "save requires a model or an explicit components mapping",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
            )
        if not isinstance(save_rng, bool) or not isinstance(save_on_cpu, bool):
            raise CheckpointConfigurationError(
                "save_rng and save_on_cpu must be booleans",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
            )
        active_format = _normalize_format_for_operation(
            format or self.default_format,
            CheckpointOperation.SAVE,
        )
        values: dict[str, Any] = {StandardComponent.MODEL.value: model}
        for name, value in (
            (StandardComponent.TOKENIZER.value, tokenizer),
            (StandardComponent.OPTIMIZER.value, optimizer),
            (StandardComponent.SCHEDULER.value, scheduler),
            (StandardComponent.SCALER.value, scaler),
            (StandardComponent.AGENT_STATE.value, extra_state),
        ):
            if value is not None:
                values[name] = value
        if save_rng and active_format is not CheckpointFormat.NPZ:
            # ``None`` instructs RNGStateCodec to capture current providers.
            values[StandardComponent.RNG.value] = None
        codec_ids: dict[str, str] = {
            StandardComponent.MODEL.value: (
                "numpy" if active_format is CheckpointFormat.NPZ else "torch"
            )
        }
        if active_format is CheckpointFormat.NPZ and any(
            value is not None for value in (optimizer, scheduler, scaler)
        ):
            raise CheckpointConfigurationError(
                "NPZ compatibility saves do not serialize optimizer, scheduler, or scaler; "
                "use torch or save_components()",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
            )
        codec_metadata = {
            name: {"save_on_cpu": save_on_cpu}
            for name in values
            if name
            in {
                StandardComponent.MODEL.value,
                StandardComponent.OPTIMIZER.value,
                StandardComponent.SCHEDULER.value,
                StandardComponent.SCALER.value,
            }
        }
        # A multi-component v2.3 checkpoint is structurally composite even
        # when the primary model representation is torch or NPZ.
        declared_format = active_format if len(values) == 1 else CheckpointFormat.COMPOSITE
        return self.save_components(
            values,
            version=version,
            codec_ids=codec_ids,
            codec_metadata=codec_metadata,
            checkpoint_format=declared_format,
            epoch=epoch if epoch is not None else current_epoch,
            step=step,
            metrics=metrics,
            metadata=metadata,
            provenance=provenance,
            compatibility=compatibility,
            archive=archive,
            overwrite=overwrite,
            trace_id=trace_id,
        )

    def save_torch(self, model: Any, *args: Any, **kwargs: Any) -> SaveResult:
        kwargs["format"] = CheckpointFormat.TORCH
        return self.save(model, *args, **kwargs)

    def save_npz(
        self,
        model: Any,
        tokenizer: Any = None,
        metadata: Mapping[str, Any] | None = None,
        version: str | None = None,
        *,
        metrics: Mapping[str, Any] | None = None,
        epoch: int | None = None,
        step: int | None = None,
        archive: bool | None = None,
        overwrite: bool | None = None,
        compressed: bool = True,
        provenance: CheckpointProvenance | None = None,
        compatibility: CompatibilityConstraints | None = None,
        trace_id: str | None = None,
    ) -> SaveResult:
        if not isinstance(compressed, bool):
            raise CheckpointConfigurationError(
                "compressed must be a boolean",
                operation=CheckpointOperation.SAVE,
                stage=CheckpointStage.VALIDATION,
            )
        values = {StandardComponent.MODEL.value: model}
        if tokenizer is not None:
            values[StandardComponent.TOKENIZER.value] = tokenizer
        return self.save_components(
            values,
            version=version,
            codec_ids={StandardComponent.MODEL.value: "numpy"},
            codec_metadata={
                StandardComponent.MODEL.value: {"compressed": compressed}
            },
            checkpoint_format=(
                CheckpointFormat.NPZ
                if len(values) == 1
                else CheckpointFormat.COMPOSITE
            ),
            epoch=epoch,
            step=step,
            metrics=metrics,
            metadata=metadata,
            provenance=provenance,
            compatibility=compatibility,
            archive=archive,
            overwrite=overwrite,
            trace_id=trace_id,
        )

    def save_async(
        self,
        *args: Any,
        executor: Executor | None = None,
        **kwargs: Any,
    ) -> Future[SaveResult]:
        """Submit ``save`` without creating process-global resources.

        State-bearing objects must remain quiescent until the future completes;
        the manager intentionally does not make an opaque deep copy of models.
        """

        active = executor
        if active is None:
            with self._executor_lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="slai-checkpoint"
                    )
                    self._owns_executor = True
                active = self._executor
        assert active is not None
        return active.submit(self.save, *args, **kwargs)

    # ------------------------------------------------------------------
    # Discovery, selection, and verification
    # ------------------------------------------------------------------
    def read_record(
        self,
        version: str,
        *,
        verify: bool = False,
        allow_legacy: bool | None = None,
    ) -> CheckpointRecord:
        try:
            safe_version = validate_version(version)
        except ValueError as exc:
            raise CheckpointNotFoundError(
                str(exc),
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.VALIDATION,
                version=version if isinstance(version, str) else None,
            ) from exc
        path = self._checkpoint_path(safe_version)
        if not path.is_dir() or path.is_symlink():
            raise CheckpointNotFoundError(
                "checkpoint version does not exist",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.DISCOVERY,
                path=path,
                version=safe_version,
            )
        use_legacy = self.allow_legacy if allow_legacy is None else allow_legacy
        if not isinstance(use_legacy, bool):
            raise CheckpointConfigurationError(
                "allow_legacy must be a boolean",
                stage=CheckpointStage.VALIDATION,
            )
        if (path / MANIFEST_NAME).is_file():
            return read_checkpoint_record(
                path,
                verify=verify,
                exact_files=self.config.require_exact_files,
                limits=self.config.manifest_limits,
            )
        if use_legacy and looks_like_legacy_checkpoint(path):
            record = build_legacy_record(path)
            if verify:
                verification = verify_manifest_files(
                    path,
                    record.manifest,
                    exact_files=self.config.require_exact_files,
                )
                record = CheckpointRecord(
                    manifest=record.manifest,
                    path=record.path,
                    health=verification.health,
                    verification=verification,
                )
            return record
        raise CheckpointManifestError(
            "checkpoint has no manifest and is not a recognized legacy layout",
            operation=CheckpointOperation.LOAD,
            stage=CheckpointStage.MANIFEST,
            path=path,
            version=safe_version,
        )

    def read_manifest(
        self,
        version: str,
        *,
        missing_ok: bool = False,
    ) -> CheckpointRecord | None:
        """Compatibility view returning a record rather than a bare manifest."""

        try:
            return self.read_record(version, verify=False)
        except CheckpointNotFoundError:
            if missing_ok:
                return None
            raise

    def _discover_records(
        self,
        *,
        verify: bool,
        allow_legacy: bool,
        strict: bool,
    ) -> tuple[tuple[CheckpointRecord, ...], tuple[dict[str, Any], ...]]:
        versions = self._storage_method("list_versions")()
        records: list[CheckpointRecord] = []
        failures: list[dict[str, Any]] = []
        for version in versions:
            try:
                records.append(
                    self.read_record(
                        version, verify=verify, allow_legacy=allow_legacy
                    )
                )
            except CheckpointError as exc:
                if strict:
                    raise _contextualize(
                        exc,
                        operation=CheckpointOperation.LIST,
                        stage=CheckpointStage.DISCOVERY,
                        version=version,
                    ) from exc.__cause__
                failures.append(exc.to_dict())
        return tuple(records), tuple(failures)

    def list_records(
        self,
        *,
        verify: bool = False,
        allow_legacy: bool | None = None,
        strict: bool = True,
    ) -> tuple[CheckpointRecord, ...]:
        use_legacy = self.allow_legacy if allow_legacy is None else allow_legacy
        if not all(isinstance(value, bool) for value in (verify, use_legacy, strict)):
            raise CheckpointConfigurationError(
                "verify, allow_legacy, and strict must be booleans",
                operation=CheckpointOperation.LIST,
                stage=CheckpointStage.VALIDATION,
            )
        with self.telemetry.operation(
            CheckpointOperation.LIST,
            stage=CheckpointStage.DISCOVERY,
        ) as span:
            records, failures = self._discover_records(
                verify=verify, allow_legacy=use_legacy, strict=strict
            )
            span.set_result(
                component_count=len(records),
                attributes={"discovery_error_count": len(failures)},
            )
            return records

    def list_checkpoints(self) -> tuple[str, ...]:
        return tuple(record.version for record in self.list_records(strict=True))

    def select_checkpoint(
        self,
        criteria: SelectionCriteria | None = None,
        *,
        runtime: RuntimeCapabilities | None = None,
        verify: bool | None = None,
    ) -> SelectionDecision:
        if criteria is not None and not isinstance(criteria, SelectionCriteria):
            raise CheckpointConfigurationError(
                "criteria must be SelectionCriteria",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.VALIDATION,
            )
        if runtime is not None and not isinstance(runtime, RuntimeCapabilities):
            raise CheckpointConfigurationError(
                "runtime must be RuntimeCapabilities",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.VALIDATION,
            )
        active_criteria = criteria or SelectionCriteria()
        should_verify = (
            active_criteria.require_verified if verify is None else verify
        )
        if not isinstance(should_verify, bool):
            raise CheckpointConfigurationError(
                "verify must be a boolean",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.VALIDATION,
            )
        records, failures = self._discover_records(
            verify=should_verify,
            allow_legacy=self.allow_legacy,
            strict=False,
        )
        decision = self.policy.select(
            records, active_criteria, runtime=runtime or self.runtime
        )
        self._emit_selection_decision(decision, failures=failures)
        return decision

    def _emit_selection_decision(
        self,
        decision: SelectionDecision,
        *,
        failures: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        attributes = decision.to_dict()
        attributes["discovery_errors"] = list(failures)
        selected = decision.selected
        provenance = selected.manifest.provenance if selected is not None else None
        self.telemetry.emit(
            CheckpointEvent(
                kind=CheckpointEventKind.SELECTION_DECIDED,
                severity=(EventSeverity.INFO if selected else EventSeverity.WARNING),
                message=(
                    "checkpoint selection completed"
                    if selected
                    else "checkpoint selection found no eligible recovery point"
                ),
                operation=CheckpointOperation.LOAD.value,
                stage=CheckpointStage.DISCOVERY.value,
                checkpoint_id=selected.checkpoint_id if selected else None,
                version=selected.version if selected else None,
                agent_id=provenance.agent_id if provenance else None,
                run_id=provenance.run_id if provenance else None,
                trace_id=provenance.trace_id if provenance else None,
                success=selected is not None,
                health=selected.health if selected else None,
                attributes=attributes,
            )
        )

    def get_latest_checkpoint(self) -> str | None:
        decision = self.select_checkpoint()
        return decision.selected.version if decision.selected is not None else None

    def verify(
        self,
        version: str,
        *,
        exact_files: bool | None = None,
    ) -> VerificationResult:
        exact = self.config.require_exact_files if exact_files is None else exact_files
        if not isinstance(exact, bool):
            raise CheckpointConfigurationError(
                "exact_files must be a boolean",
                operation=CheckpointOperation.VERIFY,
                stage=CheckpointStage.VALIDATION,
            )
        record = self.read_record(version, verify=False)
        provenance = record.manifest.provenance
        with self.telemetry.operation(
            CheckpointOperation.VERIFY,
            stage=CheckpointStage.INTEGRITY,
            version=record.version,
            checkpoint_id=record.checkpoint_id,
            agent_id=provenance.agent_id,
            run_id=provenance.run_id,
            trace_id=provenance.trace_id,
        ) as span:
            result = verify_manifest_files(
                record.path,
                record.manifest,
                exact_files=exact,
            )
            span.set_result(
                size_bytes=_record_size(record),
                component_count=len(record.manifest.saved_components),
                health=result.health,
            )
            return result

    def verify_checkpoint(self, version: str) -> bool:
        record = self.read_record(version, verify=False)
        assert_manifest_files(
            record.path,
            record.manifest,
            exact_files=self.config.require_exact_files,
        )
        return True

    # ------------------------------------------------------------------
    # Load orchestration
    # ------------------------------------------------------------------
    def load_components(
        self,
        version: str | None = None,
        *,
        components: Sequence[str] | None = None,
        skip_components: Sequence[str] = (),
        targets: Mapping[str, Any] | None = None,
        codec_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        strict: bool = True,
        restore_rng: bool = False,
        verify_integrity: bool | None = None,
        criteria: SelectionCriteria | None = None,
        runtime: RuntimeCapabilities | None = None,
        expected_codecs: Mapping[str, str] | None = None,
        expected_format: str | CheckpointFormat | None = None,
        load_key_prefix: str | None = None,
        trace_id: str | None = None,
    ) -> LoadResult:
        """Select, verify, decode, then explicitly restore components.

        Integrity and compatibility checks and all component decodes complete
        before any target is mutated.  Arbitrary target objects cannot provide
        a general cross-component rollback transaction, so restoration occurs
        in a documented deterministic order with RNG restored last.
        """

        if not isinstance(strict, bool) or not isinstance(restore_rng, bool):
            raise CheckpointConfigurationError(
                "strict and restore_rng must be booleans",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.VALIDATION,
            )
        if criteria is not None and not isinstance(criteria, SelectionCriteria):
            raise CheckpointConfigurationError(
                "criteria must be SelectionCriteria",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.VALIDATION,
            )
        if runtime is not None and not isinstance(runtime, RuntimeCapabilities):
            raise CheckpointConfigurationError(
                "runtime must be RuntimeCapabilities",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.VALIDATION,
            )
        verify = self.config.verify_on_load if verify_integrity is None else verify_integrity
        if not isinstance(verify, bool):
            raise CheckpointConfigurationError(
                "verify_integrity must be a boolean",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.VALIDATION,
            )
        requested = self._normalize_component_sequence(components, "components")
        skipped_by_caller = set(
            self._normalize_component_sequence(skip_components, "skip_components") or ()
        )
        if requested is not None and skipped_by_caller.intersection(requested):
            requested = tuple(name for name in requested if name not in skipped_by_caller)
        target_map = self._normalize_component_mapping(targets, "targets")
        metadata_map = self._normalize_component_mapping(
            codec_metadata, "codec_metadata"
        )
        expected_map = self._normalize_component_mapping(
            expected_codecs, "expected_codecs"
        )
        if load_key_prefix is not None and (
            not isinstance(load_key_prefix, str) or not load_key_prefix
        ):
            raise CheckpointConfigurationError(
                "load_key_prefix must be a non-empty string",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.VALIDATION,
            )

        if version == "latest":
            version = None
        if version is not None:
            try:
                safe_version = validate_version(version)
            except ValueError as exc:
                raise CheckpointNotFoundError(
                    str(exc),
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.VALIDATION,
                    version=version,
                ) from exc
            record = self.read_record(safe_version, verify=verify)
            if verify and record.verification is not None and not record.verification.ok:
                assert_manifest_files(
                    record.path,
                    record.manifest,
                    exact_files=self.config.require_exact_files,
                )
            if criteria is None:
                active_criteria = self._default_load_criteria(
                    verify=verify, version=safe_version
                )
            else:
                active_criteria = criteria
            decision = self.policy.select(
                (record,), active_criteria, runtime=runtime or self.runtime
            )
            self._emit_selection_decision(decision)
            record = decision.require_selected()
        else:
            active_criteria = criteria or self._default_load_criteria(verify=verify)
            decision = self.select_checkpoint(
                active_criteria,
                runtime=runtime,
                verify=verify,
            )
            record = decision.require_selected()

        if expected_format is not None:
            try:
                normalized_format = normalize_format(expected_format)
            except ValueError as exc:
                raise CheckpointConfigurationError(
                    str(exc),
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.VALIDATION,
                ) from exc
            if record.manifest.checkpoint_format is not normalized_format:
                raise CheckpointIncompatibleError(
                    "checkpoint format does not match the requested loader",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.COMPATIBILITY,
                    path=record.path,
                    version=record.version,
                    checkpoint_id=record.checkpoint_id,
                    format=record.format,
                    details={"expected_format": normalized_format.value},
                )

        provenance = record.manifest.provenance
        effective_trace = trace_id or provenance.trace_id
        started = time.monotonic()
        with self.telemetry.operation(
            CheckpointOperation.LOAD,
            stage=CheckpointStage.DESERIALIZATION,
            version=record.version,
            checkpoint_id=record.checkpoint_id,
            agent_id=provenance.agent_id,
            run_id=provenance.run_id,
            trace_id=effective_trace,
        ) as span:
            try:
                decoded, codec_by_component, manifest_selected, unsupported = (
                    self._decode_record_components(
                        record,
                        requested=requested,
                        skipped=skipped_by_caller,
                        codec_metadata=metadata_map,
                        expected_codecs=expected_map,
                    )
                )
                if requested is not None:
                    decoded = {
                        name: value for name, value in decoded.items() if name in requested
                    }
                    codec_by_component = {
                        name: value
                        for name, value in codec_by_component.items()
                        if name in decoded
                    }
                    missing_requested = set(requested) - set(decoded)
                    if missing_requested:
                        raise CheckpointIncompatibleError(
                            "checkpoint does not provide every requested component",
                            operation=CheckpointOperation.LOAD,
                            stage=CheckpointStage.COMPATIBILITY,
                            path=record.path,
                            version=record.version,
                            checkpoint_id=record.checkpoint_id,
                            missing_keys=sorted(missing_requested),
                        )

                unknown_targets = set(target_map) - set(decoded)
                if unknown_targets:
                    raise CheckpointIncompatibleError(
                        "restore targets refer to components that were not decoded",
                        operation=CheckpointOperation.LOAD,
                        stage=CheckpointStage.COMPATIBILITY,
                        path=record.path,
                        version=record.version,
                        checkpoint_id=record.checkpoint_id,
                        missing_keys=sorted(unknown_targets),
                    )
                self._validate_restore_plan(
                    decoded,
                    codec_by_component,
                    target_map,
                    restore_rng=restore_rng,
                )
                self._restore_decoded(
                    decoded,
                    codec_by_component,
                    target_map,
                    strict=strict,
                    restore_rng=restore_rng,
                    load_key_prefix=load_key_prefix,
                )

                manifest_names = set(record.manifest.saved_components)
                loaded_names = tuple(sorted(decoded))
                skippable_names = manifest_names | set(_LEGACY_PAYLOAD_KEYS.values())
                skipped_names = tuple(
                    sorted(
                        (manifest_names - manifest_selected)
                        | (skipped_by_caller & skippable_names)
                        | unsupported
                    )
                )
                duration = time.monotonic() - started
                span.set_result(
                    size_bytes=_record_size(record),
                    component_count=len(loaded_names),
                    health=record.health,
                    committed=True,
                    attributes={"restored_rng": restore_rng and "rng" in decoded},
                )
                return LoadResult(
                    record=record,
                    components=decoded,
                    loaded_components=loaded_names,
                    skipped_components=skipped_names,
                    restored_rng=restore_rng and StandardComponent.RNG.value in decoded,
                    duration_seconds=duration,
                )
            except CheckpointError as exc:
                raise _contextualize(
                    exc,
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.DESERIALIZATION,
                    path=record.path,
                    version=record.version,
                    checkpoint_id=record.checkpoint_id,
                    committed=True,
                ) from exc.__cause__
            except Exception as exc:
                raise CheckpointLoadError(
                    "checkpoint load failed",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.DESERIALIZATION,
                    path=record.path,
                    version=record.version,
                    checkpoint_id=record.checkpoint_id,
                    format=record.format,
                    committed=True,
                    details={"error_type": type(exc).__name__},
                ) from exc

    @staticmethod
    def _normalize_component_sequence(
        value: Sequence[str] | None,
        field_name: str,
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            raise CheckpointConfigurationError(
                f"{field_name} must be a sequence, not a string",
                stage=CheckpointStage.VALIDATION,
            )
        result = tuple(dict.fromkeys(_normalize_component(item) for item in value))
        return result

    @staticmethod
    def _default_load_criteria(
        *, verify: bool, version: str | None = None
    ) -> SelectionCriteria:
        if verify:
            return SelectionCriteria(version=version) if version else SelectionCriteria()
        accepted = (
            CheckpointHealth.UNKNOWN,
            CheckpointHealth.HEALTHY,
            CheckpointHealth.DEGRADED,
        )
        return SelectionCriteria(
            version=version,
            require_verified=False,
            accepted_health=accepted,
        )

    def _decode_record_components(
        self,
        record: CheckpointRecord,
        *,
        requested: tuple[str, ...] | None,
        skipped: set[str],
        codec_metadata: Mapping[str, Mapping[str, Any]],
        expected_codecs: Mapping[str, str],
    ) -> tuple[dict[str, Any], dict[str, Any], set[str], set[str]]:
        grouped: dict[str, list[CheckpointArtifact]] = defaultdict(list)
        for artifact in record.manifest.artifacts:
            grouped[artifact.component].append(artifact)

        legacy_payload = StandardComponent.CHECKPOINT_PAYLOAD.value in grouped
        selected: set[str] = set()
        for component, artifacts in grouped.items():
            if component in skipped:
                continue
            if requested is None:
                # Codec-less legacy metadata/tokenizer artifacts remain
                # integrity-verified but are not guessed into a modern codec.
                if any(item.codec is None for item in artifacts):
                    continue
                selected.add(component)
            elif component in requested:
                selected.add(component)
            elif legacy_payload and component == StandardComponent.CHECKPOINT_PAYLOAD.value:
                if set(requested).intersection(_LEGACY_PAYLOAD_KEYS.values()):
                    selected.add(component)

        unsupported: set[str] = set()
        if requested is None:
            unsupported = {
                component
                for component, artifacts in grouped.items()
                if any(item.codec is None for item in artifacts)
            }
        decoded: dict[str, Any] = {}
        codecs: dict[str, Any] = {}
        # Validate every selected group before performing any decode.
        plans: list[tuple[str, Any, Path, tuple[CheckpointArtifact, ...]]] = []
        for component in sorted(selected):
            artifacts = tuple(sorted(grouped[component], key=lambda item: item.relative_path))
            codec_ids = {item.codec for item in artifacts}
            codec_versions = {item.codec_version for item in artifacts}
            legacy = _is_legacy_manifest(record.manifest)
            if (
                None in codec_ids
                or len(codec_ids) != 1
                or len(codec_versions) != 1
                or (None in codec_versions and not legacy)
            ):
                raise CheckpointIncompatibleError(
                    "component artifacts do not declare one exact codec identity",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.COMPATIBILITY,
                    path=record.path,
                    version=record.version,
                    checkpoint_id=record.checkpoint_id,
                    component=component,
                    details={
                        "codec_ids": sorted(str(item) for item in codec_ids),
                        "codec_versions": sorted(str(item) for item in codec_versions),
                    },
                )
            codec_id = next(iter(codec_ids))
            codec_version = next(iter(codec_versions))
            assert codec_id is not None
            codec = self.codecs.get(codec_id)
            if codec_version is not None and codec.codec_version != codec_version:
                raise CheckpointIncompatibleError(
                    "registered codec version does not match the artifact declaration",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.COMPATIBILITY,
                    path=record.path,
                    version=record.version,
                    checkpoint_id=record.checkpoint_id,
                    component=component,
                    details={
                        "codec_id": codec_id,
                        "required": codec_version,
                        "actual": codec.codec_version,
                    },
                )
            expected = expected_codecs.get(component)
            if expected is not None and expected != codec_id:
                raise CheckpointIncompatibleError(
                    "component codec does not match the requested loader",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.COMPATIBILITY,
                    path=record.path,
                    version=record.version,
                    checkpoint_id=record.checkpoint_id,
                    component=component,
                    details={"expected_codec": expected, "actual_codec": codec_id},
                )
            source = self._component_source(record.path, artifacts, codec)
            plans.append((component, codec, source, artifacts))

        for component, codec, source, _ in plans:
            context = CodecContext(
                checkpoint_id=record.checkpoint_id,
                version=record.version,
                component=component,
                metadata=dict(codec_metadata.get(component, {})),
            )
            decoded[component] = codec.decode(source, context=context)
            codecs[component] = codec

        if (
            _is_legacy_manifest(record.manifest)
            and StandardComponent.CHECKPOINT_PAYLOAD.value in decoded
        ):
            payload = decoded.pop(StandardComponent.CHECKPOINT_PAYLOAD.value)
            payload_codec = codecs.pop(StandardComponent.CHECKPOINT_PAYLOAD.value)
            if not isinstance(payload, Mapping):
                raise CheckpointIncompatibleError(
                    "legacy monolithic checkpoint payload is not a mapping",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.COMPATIBILITY,
                    path=record.path,
                    version=record.version,
                    checkpoint_id=record.checkpoint_id,
                )
            for old_name, component in _LEGACY_PAYLOAD_KEYS.items():
                if old_name in payload and payload[old_name] is not None:
                    decoded[component] = payload[old_name]
                    # v2.2 training-state fields were all carried by torch's
                    # monolithic payload except RNG, whose old representation
                    # is returned but intentionally not applied automatically.
                    codecs[component] = payload_codec if component != "rng" else None
            selected.discard(StandardComponent.CHECKPOINT_PAYLOAD.value)
            selected.update(decoded)
        for component, expected in expected_codecs.items():
            if component not in decoded:
                continue
            actual = getattr(codecs.get(component), "codec_id", None)
            if actual != expected:
                raise CheckpointIncompatibleError(
                    "component codec does not match the requested loader",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.COMPATIBILITY,
                    path=record.path,
                    version=record.version,
                    checkpoint_id=record.checkpoint_id,
                    component=component,
                    details={"expected_codec": expected, "actual_codec": actual},
                )
        return decoded, codecs, selected, unsupported

    @staticmethod
    def _component_source(
        checkpoint_dir: Path,
        artifacts: tuple[CheckpointArtifact, ...],
        codec: Any,
    ) -> Path:
        # Metadata values are JSONValue instances and may be unhashable (for
        # example, a malformed list or object), so do not collect them in a
        # set before validating the value below.
        roots = [item.metadata.get(_MANAGER_CODEC_ROOT) for item in artifacts]
        if None not in roots and len(roots) == 1:
            raw_root = next(iter(roots))
            if not isinstance(raw_root, str):
                raise CheckpointManifestError(
                    "artifact codec_root metadata must be a string",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.MANIFEST,
                    path=checkpoint_dir,
                )
            try:
                relative_root = validate_relative_path(raw_root)
            except ValueError as exc:
                raise CheckpointManifestError(
                    f"invalid artifact codec_root metadata: {exc}",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.MANIFEST,
                    path=checkpoint_dir,
                ) from exc
            source = resolve_artifact_path(checkpoint_dir, relative_root)
        elif bool(getattr(codec, "multi_file", False)):
            parents = [PurePosixPath(item.relative_path).parent for item in artifacts]
            common = PurePosixPath(os.path.commonpath([item.as_posix() for item in parents]))
            source = checkpoint_dir.joinpath(*common.parts)
        elif len(artifacts) == 1:
            source = resolve_artifact_path(
                checkpoint_dir, artifacts[0].relative_path, must_exist=True
            )
        else:
            raise CheckpointManifestError(
                "single-file codec component declares multiple artifacts",
                operation=CheckpointOperation.LOAD,
                stage=CheckpointStage.MANIFEST,
                path=checkpoint_dir,
                component=artifacts[0].component,
            )

        if bool(getattr(codec, "multi_file", False)):
            if source.is_symlink() or not source.is_dir():
                raise CheckpointPathError(
                    "multi-file codec root is not a regular directory",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.INTEGRITY,
                    path=source,
                    component=artifacts[0].component,
                )
            resolved_root = source.resolve(strict=True)
            for artifact in artifacts:
                file_path = resolve_artifact_path(
                    checkpoint_dir, artifact.relative_path, must_exist=True
                ).resolve(strict=True)
                if not file_path.is_relative_to(resolved_root):
                    raise CheckpointPathError(
                        "component artifact lies outside its codec root",
                        operation=CheckpointOperation.LOAD,
                        stage=CheckpointStage.INTEGRITY,
                        path=file_path,
                        component=artifacts[0].component,
                    )
        else:
            if len(artifacts) != 1:
                raise CheckpointManifestError(
                    "single-file codec component declares multiple artifacts",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.MANIFEST,
                    path=checkpoint_dir,
                    component=artifacts[0].component,
                )
            source = resolve_artifact_path(
                checkpoint_dir, artifacts[0].relative_path, must_exist=True
            )
        return source

    @staticmethod
    def _validate_restore_plan(
        decoded: Mapping[str, Any],
        codecs: Mapping[str, Any],
        targets: Mapping[str, Any],
        *,
        restore_rng: bool,
    ) -> None:
        for component in targets:
            codec = codecs.get(component)
            if not isinstance(
                codec,
                (TorchCheckpointCodec, NumpyCheckpointCodec, TokenizerCheckpointCodec),
            ):
                raise CheckpointConfigurationError(
                    "codec protocol defines decoding but not generic target mutation; "
                    "this component has no manager-supported restore adapter",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.COMPATIBILITY,
                    component=component,
                    details={
                        "codec_id": getattr(codec, "codec_id", None),
                        "resolution": "consume the decoded component explicitly",
                    },
                )
        if restore_rng:
            if StandardComponent.RNG.value not in decoded:
                raise CheckpointIncompatibleError(
                    "RNG restoration was requested but no RNG component was decoded",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.COMPATIBILITY,
                    component=StandardComponent.RNG.value,
                )
            if not isinstance(codecs.get(StandardComponent.RNG.value), RNGStateCodec):
                raise CheckpointIncompatibleError(
                    "legacy or custom RNG state has no safe automatic restore adapter",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.COMPATIBILITY,
                    component=StandardComponent.RNG.value,
                )

    @staticmethod
    def _restore_decoded(
        decoded: Mapping[str, Any],
        codecs: Mapping[str, Any],
        targets: Mapping[str, Any],
        *,
        strict: bool,
        restore_rng: bool,
        load_key_prefix: str | None,
    ) -> None:
        ordered = sorted(
            targets,
            key=lambda item: (_RESTORE_ORDER.get(item, 100), item),
        )
        for component in ordered:
            codec = codecs[component]
            payload = decoded[component]
            if isinstance(codec, NumpyCheckpointCodec):
                codec.restore(
                    targets[component],
                    payload,
                    strict=strict,
                    key_prefix=load_key_prefix,
                )
            elif isinstance(codec, TorchCheckpointCodec):
                if load_key_prefix is not None and component == StandardComponent.MODEL.value:
                    if not isinstance(payload, Mapping):
                        raise CheckpointIncompatibleError(
                            "prefix-filtered torch state is not a mapping",
                            stage=CheckpointStage.COMPATIBILITY,
                            component=component,
                        )
                    payload = {
                        name: value
                        for name, value in payload.items()
                        if isinstance(name, str) and name.startswith(load_key_prefix)
                    }
                codec.restore(targets[component], payload, strict=strict)
            elif isinstance(codec, TokenizerCheckpointCodec):
                codec.restore(targets[component], payload)
        if restore_rng:
            rng_codec = codecs[StandardComponent.RNG.value]
            assert isinstance(rng_codec, RNGStateCodec)
            rng_codec.restore(decoded[StandardComponent.RNG.value], strict=strict)

    def load(
        self,
        model: Any = None,
        tokenizer: Any = None,
        version: str | None = None,
        format: str | CheckpointFormat | None = None,
        *,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
        map_location: str = "cpu",
        strict: bool = True,
        restore_rng: bool = False,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_scaler: bool = True,
        verify_integrity: bool | None = None,
        load_components: Sequence[str] | None = None,
        skip_components: Sequence[str] = (),
        load_key_prefix: str | None = None,
        criteria: SelectionCriteria | None = None,
        runtime: RuntimeCapabilities | None = None,
        trace_id: str | None = None,
    ) -> LoadResult:
        """Compatibility wrapper that restores supplied training targets."""

        for name, value in (
            ("load_optimizer", load_optimizer),
            ("load_scheduler", load_scheduler),
            ("load_scaler", load_scaler),
        ):
            if not isinstance(value, bool):
                raise CheckpointConfigurationError(
                    f"{name} must be a boolean",
                    operation=CheckpointOperation.LOAD,
                    stage=CheckpointStage.VALIDATION,
                )
        targets: dict[str, Any] = {}
        for component, target, enabled in (
            (StandardComponent.MODEL.value, model, model is not None),
            (StandardComponent.TOKENIZER.value, tokenizer, tokenizer is not None),
            (
                StandardComponent.OPTIMIZER.value,
                optimizer,
                optimizer is not None and load_optimizer,
            ),
            (
                StandardComponent.SCHEDULER.value,
                scheduler,
                scheduler is not None and load_scheduler,
            ),
            (
                StandardComponent.SCALER.value,
                scaler,
                scaler is not None and load_scaler,
            ),
        ):
            if enabled:
                targets[component] = target
        metadata = {
            name: {"map_location": map_location}
            for name in (
                StandardComponent.MODEL.value,
                StandardComponent.OPTIMIZER.value,
                StandardComponent.SCHEDULER.value,
                StandardComponent.SCALER.value,
            )
        }
        expected_codecs: dict[str, str] = {}
        expected_manifest_format: CheckpointFormat | None = None
        if format is not None:
            normalized = _normalize_format_for_operation(
                format,
                CheckpointOperation.LOAD,
            )
            if normalized is CheckpointFormat.TORCH:
                expected_codecs[StandardComponent.MODEL.value] = "torch"
            elif normalized is CheckpointFormat.NPZ:
                expected_codecs[StandardComponent.MODEL.value] = "numpy"
            else:
                expected_manifest_format = CheckpointFormat.COMPOSITE
        return self.load_components(
            version,
            components=load_components,
            skip_components=skip_components,
            targets=targets,
            codec_metadata=metadata,
            strict=strict,
            restore_rng=restore_rng,
            verify_integrity=verify_integrity,
            criteria=criteria,
            runtime=runtime,
            expected_codecs=expected_codecs,
            expected_format=expected_manifest_format,
            load_key_prefix=load_key_prefix,
            trace_id=trace_id,
        )

    def load_torch(self, model: Any, *args: Any, **kwargs: Any) -> LoadResult:
        kwargs["format"] = CheckpointFormat.TORCH
        return self.load(model, *args, **kwargs)

    def load_npz(self, model: Any, *args: Any, **kwargs: Any) -> LoadResult:
        kwargs["format"] = CheckpointFormat.NPZ
        return self.load(model, *args, **kwargs)

    # ------------------------------------------------------------------
    # Retention, archival, recovery, and lifecycle
    # ------------------------------------------------------------------
    def plan_retention(self, rules: RetentionRules) -> RetentionPlan:
        if not isinstance(rules, RetentionRules):
            raise CheckpointConfigurationError(
                "rules must be RetentionRules",
                operation=CheckpointOperation.RETAIN,
                stage=CheckpointStage.VALIDATION,
            )
        records, failures = self._discover_records(
            verify=True,
            allow_legacy=self.allow_legacy,
            strict=False,
        )
        plan = self.policy.plan_retention(records, rules)
        if failures:
            plan = replace(
                plan,
                constraints_satisfied=False,
                reasons=plan.reasons
                + (
                    PolicyReason(
                        "checkpoint_discovery_failed",
                        "retention cannot safely account for every checkpoint directory",
                        {"discovery_error_count": len(failures)},
                    ),
                ),
            )
        attributes = plan.to_dict()
        attributes["discovery_errors"] = list(failures)
        self.telemetry.emit(
            CheckpointEvent(
                kind=CheckpointEventKind.RETENTION_PLANNED,
                severity=(
                    EventSeverity.INFO
                    if plan.constraints_satisfied
                    else EventSeverity.WARNING
                ),
                message="checkpoint retention plan created",
                operation=CheckpointOperation.RETAIN.value,
                stage=CheckpointStage.CLEANUP.value,
                success=plan.constraints_satisfied,
                attributes=attributes,
            )
        )
        return plan

    def plan_rollback(
        self,
        current: str | CheckpointRecord,
        rules: RollbackRules | None = None,
        *,
        runtime: RuntimeCapabilities | None = None,
    ) -> RollbackPlan:
        """Create, but never execute, an evidence-backed rollback plan."""

        if rules is not None and not isinstance(rules, RollbackRules):
            raise CheckpointConfigurationError(
                "rules must be RollbackRules",
                operation=CheckpointOperation.RESTORE,
                stage=CheckpointStage.VALIDATION,
            )
        if runtime is not None and not isinstance(runtime, RuntimeCapabilities):
            raise CheckpointConfigurationError(
                "runtime must be RuntimeCapabilities",
                operation=CheckpointOperation.RESTORE,
                stage=CheckpointStage.VALIDATION,
            )
        active_rules = rules or RollbackRules()
        if isinstance(current, str):
            current_record = self.read_record(
                current,
                verify=active_rules.criteria.require_verified,
            )
        elif isinstance(current, CheckpointRecord):
            current_record = current
        else:
            raise CheckpointConfigurationError(
                "current must be a checkpoint version or CheckpointRecord",
                operation=CheckpointOperation.RESTORE,
                stage=CheckpointStage.VALIDATION,
            )
        records, failures = self._discover_records(
            verify=active_rules.criteria.require_verified,
            allow_legacy=self.allow_legacy,
            strict=False,
        )
        plan = self.policy.plan_rollback(
            current_record,
            records,
            active_rules,
            runtime=runtime or self.runtime,
        )
        attributes = plan.to_dict()
        attributes["discovery_errors"] = list(failures)
        provenance = current_record.manifest.provenance
        self.telemetry.emit(
            CheckpointEvent(
                kind=CheckpointEventKind.ROLLBACK_PLANNED,
                severity=(EventSeverity.INFO if plan.possible else EventSeverity.WARNING),
                message=(
                    "checkpoint rollback target planned"
                    if plan.possible
                    else "no eligible checkpoint rollback target was found"
                ),
                operation=CheckpointOperation.RESTORE.value,
                stage=CheckpointStage.DISCOVERY.value,
                checkpoint_id=(
                    plan.target.checkpoint_id if plan.target is not None else None
                ),
                version=plan.target.version if plan.target is not None else None,
                agent_id=provenance.agent_id,
                run_id=provenance.run_id,
                trace_id=provenance.trace_id,
                success=plan.possible,
                health=plan.target.health if plan.target is not None else None,
                attributes=attributes,
            )
        )
        return plan

    def execute_retention(
        self,
        plan: RetentionPlan,
        *,
        require_constraints_satisfied: bool = True,
        delete_archives: bool = True,
    ) -> tuple[str, ...]:
        """Execute only a previously audited retention plan.

        Each version is re-read and matched by checkpoint identity immediately
        before deletion, preventing a stale plan from deleting a replacement.
        """

        if not isinstance(plan, RetentionPlan):
            raise CheckpointConfigurationError(
                "plan must be RetentionPlan",
                operation=CheckpointOperation.RETAIN,
                stage=CheckpointStage.VALIDATION,
            )
        if not isinstance(require_constraints_satisfied, bool) or not isinstance(
            delete_archives, bool
        ):
            raise CheckpointConfigurationError(
                "retention execution flags must be booleans",
                operation=CheckpointOperation.RETAIN,
                stage=CheckpointStage.VALIDATION,
            )
        if require_constraints_satisfied and not plan.constraints_satisfied:
            raise CheckpointRetentionError(
                "retention plan cannot satisfy its declared constraints",
                operation=CheckpointOperation.RETAIN,
                stage=CheckpointStage.VALIDATION,
                committed=False,
                details=plan.to_dict(),
            )
        deleted: list[str] = []
        with self.telemetry.operation(
            CheckpointOperation.RETAIN,
            stage=CheckpointStage.CLEANUP,
            attributes={"delete_count": len(plan.delete)},
        ) as span:
            try:
                for planned in plan.delete:
                    current = self.read_record(planned.version, verify=False)
                    if current.checkpoint_id != planned.checkpoint_id:
                        raise CheckpointRetentionError(
                            "retention plan is stale; checkpoint identity changed",
                            operation=CheckpointOperation.RETAIN,
                            stage=CheckpointStage.VALIDATION,
                            path=current.path,
                            version=current.version,
                            checkpoint_id=current.checkpoint_id,
                            committed=bool(deleted),
                            details={
                                "planned_checkpoint_id": planned.checkpoint_id,
                                "deleted_versions": list(deleted),
                            },
                        )
                    self.delete_checkpoint(
                        current.version,
                        missing_ok=False,
                        delete_archive=delete_archives,
                    )
                    deleted.append(current.version)
                span.set_result(
                    component_count=len(deleted),
                    committed=bool(deleted),
                )
                return tuple(deleted)
            except CheckpointError as exc:
                raise _contextualize(
                    exc,
                    operation=CheckpointOperation.RETAIN,
                    stage=CheckpointStage.CLEANUP,
                    committed=bool(deleted),
                ) from exc.__cause__

    def cleanup_old_checkpoints(self, keep: int | None = None) -> tuple[str, ...]:
        limit = self.retention_limit if keep is None else keep
        if limit is None:
            return ()
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise CheckpointConfigurationError(
                "keep must be a positive integer",
                operation=CheckpointOperation.RETAIN,
                stage=CheckpointStage.VALIDATION,
            )
        plan = self.plan_retention(
            RetentionRules(
                max_checkpoints=limit,
                minimum_keep=1,
                keep_latest=1,
            )
        )
        return self.execute_retention(plan)

    def archive_checkpoint(
        self, version: str, *, overwrite: bool = False
    ) -> Path:
        if not isinstance(overwrite, bool):
            raise CheckpointConfigurationError(
                "overwrite must be a boolean",
                operation=CheckpointOperation.ARCHIVE,
                stage=CheckpointStage.VALIDATION,
            )
        safe_version = _validate_version_for_operation(
            version, CheckpointOperation.ARCHIVE
        )
        with self.telemetry.operation(
            CheckpointOperation.ARCHIVE,
            stage=CheckpointStage.ARCHIVAL,
            version=safe_version,
        ) as span:
            path = Path(
                self._storage_method("create_archive")(
                    safe_version, overwrite=overwrite
                )
            )
            span.set_result(committed=True)
            return path

    def restore_archive(
        self,
        version: str,
        *,
        allow_overwrite: bool = False,
        require_digest: bool = True,
        max_members: int = 100_000,
        max_total_bytes: int | None = None,
    ) -> CheckpointRecord:
        if not isinstance(allow_overwrite, bool) or not isinstance(
            require_digest, bool
        ):
            raise CheckpointConfigurationError(
                "allow_overwrite and require_digest must be booleans",
                operation=CheckpointOperation.RESTORE,
                stage=CheckpointStage.VALIDATION,
            )
        if (
            isinstance(max_members, bool)
            or not isinstance(max_members, int)
            or max_members < 1
            or (
                max_total_bytes is not None
                and (
                    isinstance(max_total_bytes, bool)
                    or not isinstance(max_total_bytes, int)
                    or max_total_bytes < 0
                )
            )
        ):
            raise CheckpointConfigurationError(
                "archive extraction limits are invalid",
                operation=CheckpointOperation.RESTORE,
                stage=CheckpointStage.VALIDATION,
            )
        safe_version = _validate_version_for_operation(
            version, CheckpointOperation.RESTORE
        )
        with self.telemetry.operation(
            CheckpointOperation.RESTORE,
            stage=CheckpointStage.RESTORATION,
            version=safe_version,
        ) as span:
            published = False
            path = self.base_dir
            try:
                path = Path(
                    self._storage_method("restore_archive")(
                        safe_version,
                        allow_overwrite=allow_overwrite,
                        require_digest=require_digest,
                        max_members=max_members,
                        max_total_bytes=max_total_bytes,
                    )
                )
                published = True
                record = read_checkpoint_record(
                    path,
                    verify=True,
                    exact_files=self.config.require_exact_files,
                    limits=self.config.manifest_limits,
                )
                if record.verification is None or not record.verification.ok:
                    raise CheckpointLoadError(
                        "restored archive failed checkpoint verification",
                        operation=CheckpointOperation.RESTORE,
                        stage=CheckpointStage.INTEGRITY,
                        path=path,
                        version=safe_version,
                        checkpoint_id=record.checkpoint_id,
                        committed=True,
                    )
                span.set_result(
                    checkpoint_id=record.checkpoint_id,
                    size_bytes=_record_size(record),
                    component_count=len(record.manifest.saved_components),
                    health=record.health,
                    committed=True,
                )
                return record
            except CheckpointError as exc:
                raise _contextualize(
                    exc,
                    operation=CheckpointOperation.RESTORE,
                    stage=CheckpointStage.RESTORATION,
                    path=path,
                    version=safe_version,
                    committed=published,
                ) from exc.__cause__

    def delete_checkpoint(
        self,
        version: str,
        *,
        missing_ok: bool = False,
        delete_archive: bool = True,
    ) -> bool:
        if not isinstance(missing_ok, bool) or not isinstance(delete_archive, bool):
            raise CheckpointConfigurationError(
                "missing_ok and delete_archive must be booleans",
                operation=CheckpointOperation.DELETE,
                stage=CheckpointStage.VALIDATION,
            )
        try:
            safe_version = validate_version(version)
        except ValueError as exc:
            raise CheckpointNotFoundError(
                str(exc),
                operation=CheckpointOperation.DELETE,
                stage=CheckpointStage.VALIDATION,
                version=version if isinstance(version, str) else None,
            ) from exc
        deleted = False
        with self.telemetry.operation(
            CheckpointOperation.DELETE,
            stage=CheckpointStage.CLEANUP,
            version=safe_version,
        ) as span:
            try:
                deleted = bool(
                    self._storage_method("delete")(
                        safe_version, missing_ok=missing_ok
                    )
                )
                if delete_archive:
                    archive_method = getattr(self.storage, "archive_path", None)
                    if callable(archive_method):
                        archive_value = archive_method(safe_version)
                        if not isinstance(archive_value, (str, os.PathLike)):
                            raise CheckpointPathError(
                                "storage returned an invalid archive path",
                                operation=CheckpointOperation.DELETE,
                                stage=CheckpointStage.VALIDATION,
                                version=safe_version,
                                committed=deleted,
                            )
                        archive = Path(archive_value)
                        sidecar = archive.with_name(f"{archive.name}.sha256")
                        root = self.base_dir.resolve(strict=True)
                        for candidate in (archive, sidecar):
                            if candidate.parent.resolve(strict=False) != root:
                                raise CheckpointPathError(
                                    "storage returned an archive path outside base_dir",
                                    operation=CheckpointOperation.DELETE,
                                    stage=CheckpointStage.VALIDATION,
                                    path=candidate,
                                    version=safe_version,
                                    committed=deleted,
                                )
                            if candidate.is_symlink():
                                raise CheckpointPathError(
                                    "refusing to delete a symbolic-link archive",
                                    operation=CheckpointOperation.DELETE,
                                    stage=CheckpointStage.VALIDATION,
                                    path=candidate,
                                    version=safe_version,
                                    committed=deleted,
                                )
                            if candidate.exists():
                                if not candidate.is_file():
                                    raise CheckpointPathError(
                                        "archive path is not a regular file",
                                        operation=CheckpointOperation.DELETE,
                                        stage=CheckpointStage.VALIDATION,
                                        path=candidate,
                                        version=safe_version,
                                        committed=deleted,
                                    )
                                candidate.unlink()
                span.set_result(committed=deleted)
                return deleted
            except CheckpointError:
                raise
            except OSError as exc:
                raise CheckpointStorageError(
                    "failed to remove checkpoint archive files",
                    operation=CheckpointOperation.DELETE,
                    stage=CheckpointStage.CLEANUP,
                    path=self.base_dir,
                    version=safe_version,
                    retryable=True,
                    committed=deleted,
                    details={"error_type": type(exc).__name__},
                ) from exc
        return deleted

    def recover_incomplete_transactions(self) -> tuple[Path, ...]:
        with self.telemetry.operation(
            CheckpointOperation.RESTORE,
            stage=CheckpointStage.RESTORATION,
        ) as span:
            recovered = tuple(
                Path(item)
                for item in self._storage_method(
                    "recover_incomplete_transactions"
                )()
            )
            span.set_result(component_count=len(recovered), committed=bool(recovered))
            return recovered

    def health_report(
        self,
        *,
        emit: bool = True,
        trace_id: str | None = None,
    ) -> CheckpointHealthReport:
        if not isinstance(emit, bool):
            raise CheckpointConfigurationError(
                "emit must be a boolean",
                operation=CheckpointOperation.VERIFY,
                stage=CheckpointStage.VALIDATION,
            )
        records, failures = self._discover_records(
            verify=True,
            allow_legacy=self.allow_legacy,
            strict=False,
        )
        report = build_health_report(records)
        if failures:
            report = replace(
                report,
                overall_health=CheckpointHealth.CORRUPT,
                findings=report.findings
                + (
                    HealthFinding(
                        code="checkpoint_discovery_failed",
                        severity=EventSeverity.ERROR,
                        message=(
                            f"{len(failures)} checkpoint director"
                            f"{'y' if len(failures) == 1 else 'ies'} could not be "
                            "read as a valid checkpoint"
                        ),
                    ),
                ),
            )
        if emit:
            self.telemetry.emit_health_report(report, trace_id=trace_id)
        return report

    def close(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        """Release a lazily created manager-owned async executor."""

        if not isinstance(wait, bool) or not isinstance(cancel_futures, bool):
            raise TypeError("wait and cancel_futures must be booleans")
        with self._executor_lock:
            executor = self._executor if self._owns_executor else None
            if self._owns_executor:
                self._executor = None
                self._owns_executor = False
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> "CheckpointManager":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = ["CheckpointManager"]

if __name__ == "__main__":
    print("\n=== Running Checkpoint Manager Comprehensive Self-Test ===\n")
    import tempfile

    printer.status("TEST", "Starting Checkpoint Manager tests", "info")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(base_dir=tmpdir)
        # Register a dummy codec if needed, but default registry already has some
        # Save a simple component (agent_state)
        result = manager.save_components(
            {"agent_state": {"test": 1}},
            version="v1",
        )
        assert result.committed
        # Load it back
        load_result = manager.load_components("v1", components=["agent_state"])
        assert load_result.components["agent_state"] == {"test": 1}
    printer.status("SMOKE", "save/load cycle with agent_state passed", "success")

    print("\n=== All manager tests passed ===\n")