"""
SLAI Reader Agent

Production-ready orchestration layer for the Reader subsystem.

Responsibilities
----------------
- coordinate Reader parsing, optional conservative recovery, conversion, and merge flows
- keep subsystem-specific parsing/recovery/conversion policy inside ``src/agents/reader``
- load only agent-level orchestration settings from ``agents_config.yaml``
- use ``SharedMemory`` as the agent state/cache/checkpoint boundary
- preserve BaseAgent lifecycle, metrics, retry, shared-memory audit, and issue handling
- return deterministic, JSON-safe Reader task results without fabricating document content

Capability engines (all opt-in; defaults leave existing behavior unchanged)
---------------------------------------------------------------------------
- ``detect`` / ``validate`` modes: content-based format detection and integrity validation
- ``compress`` / ``decompress`` modes: packaging, lossless optimization and safe extraction
- ``salvage`` mode, or ``salvage: true`` on a task: recover a damaged file that failed to parse; a recovered
  document carries ``metadata.salvage`` and a ``salvaged:<level>`` warning, and the run is ``partial``
- ``durable: true`` on a convert/merge task: journal each finished artifact so ``resume`` (or the same task
  again) skips completed work whose output is still intact; ``jobs`` lists or inspects journals
- ``verify: true`` / ``detect: true``: validate each written artifact / annotate parsed documents
"""
from __future__ import annotations

__version__ = "2.4.0"

import asyncio
import contextlib
import threading
import time
import uuid

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Awaitable, Dict, Iterable, List, Mapping, TypeVar, Optional, Sequence, Tuple, Generic

from .base_agent import BaseAgent
from .base.utils.main_config_loader import get_config_section
from .reader import *
from .reader.utils.reader_error import *
from .reader.utils.reader_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Reader Agent")
printer = PrettyPrinter()


T = TypeVar("T")

@dataclass(frozen=True)
class ReaderAgentTask:
    """Normalized ReaderAgent task contract used internally by the pipeline."""

    instruction: str
    files: List[str]
    output_format: str
    output_dir: str
    mode: str = "pipeline"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    recover: bool = False
    merge: bool = False
    fail_fast: bool = False
    use_cache: bool = True
    write_checkpoints: bool = True
    include_documents: bool = False
    include_content: bool = False
    output_filename: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    detect: bool = False
    verify: bool = False
    salvage: bool = False
    durable: bool = False
    job_id: Optional[str] = None
    deep: bool = False
    compress_method: str = "auto"
    package_format: Optional[str] = None
    lossy: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


T = TypeVar("T")
StageItem = TypeVar("StageItem", bound=Mapping[str, Any])


@dataclass(frozen=True, slots=True)
class ReaderStageError:
    """Normalized failure returned by one Reader pipeline stage."""

    stage: str
    code: str
    message: str
    retryable: bool
    recoverable: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        safe_details = json_safe(dict(self.details), redact=True)
        if not isinstance(safe_details, dict):
            safe_details = {"payload": safe_details}

        # Keep the original error envelope at the top level for compatibility,
        # while also exposing the normalized typed fields.
        return {
            **safe_details,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "recoverable": self.recoverable,
            "details": safe_details,
        }


@dataclass(frozen=True, slots=True)
class ReaderStageBatch(Generic[StageItem]):
    """Typed result of one Reader pipeline stage."""

    stage: str
    items: Tuple[Mapping[str, Any], ...] = ()
    errors: Tuple[ReaderStageError, ...] = ()

    @property
    def failed_count(self) -> int:
        return len(self.errors)

    @property
    def success_count(self) -> int:
        return len(self.items)

    @classmethod
    def empty(cls, stage: str) -> "ReaderStageBatch[Any]":
        # Construct through ``cls``: calling the subscripted alias makes typing assign
        # ``__orig_class__``, which raises TypeError on frozen+slots dataclasses (Python 3.10).
        return cls(stage=stage, items=(), errors=())

    def error_dicts(self) -> List[Dict[str, Any]]:
        return [error.to_dict() for error in self.errors]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "errors": self.error_dicts(),
        }


@dataclass(frozen=True, slots=True)
class ReaderAgentStageResult:
    """Normalized stage result for Reader parse/recover/convert operations."""

    stage: str = ""
    items: Tuple[Mapping[str, Any], ...] = ()
    errors: Tuple[Mapping[str, Any], ...] = ()

    @property
    def failed_count(self) -> int:
        return len(self.errors)

    @property
    def success_count(self) -> int:
        return len(self.items)

    def error_dicts(self) -> List[Dict[str, Any]]:
        return [dict(error) for error in self.errors]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "errors": self.error_dicts(),
        }


class ReaderAgent(BaseAgent):
    """Top-level Reader orchestration agent.

    The agent intentionally avoids ``ReaderMemory`` for orchestration. All state,
    checkpoint records, cache entries, and events are written through
    ``self.shared_memory`` so the Reader participates in the same runtime fabric
    as the other SLAI agents. The single exception is the durable job journal
    (``JobEngine``): it is created lazily and only when a task asks for ``durable``
    execution, because a journal that must survive a crash cannot live in process memory.
    """

    STREAM_COMPRESSION_METHODS = frozenset({"gz", "gzip", "bz2", "bzip2", "xz", "lzma"})
    UTILITY_MODES = frozenset({"detect", "validate", "compress", "decompress", "salvage", "jobs"})
    SUPPORTED_MODES = {
        "pipeline", "parse", "read", "recover", "convert", "merge", "capabilities", "describe",
        "detect", "validate", "compress", "decompress", "salvage", "jobs", "resume",
    }
    CACHE_VERSION = "reader_agent.v3"
    RECOVERY_CACHE_VERSION = ("reader_agent.recovery.semantic_flags.v2")

    def __init__(self, shared_memory, agent_factory, config: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory)
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory

        self.agent_config: Dict[str, Any] = dict(get_config_section("reader_agent") or {})
        if config:
            self.agent_config.update(dict(config))

        self._load_reader_agent_config()
        self._validate_reader_agent_config()

        self.parser = ParserEngine()
        self.recovery = RecoveryEngine()
        self.converter = ConversionEngine()
        # Foundations and capability engines. None of these touch disk until an operation needs it.
        self.detector = DetectionEngine()
        self.integrity = IntegrityEngine(detector=self.detector)
        self.compressor = CompressionEngine(integrity=self.integrity)
        self.salvager = SalvageEngine(integrity=self.integrity)
        self._job_engine: Optional[JobEngine] = None
        self._job_lock = threading.Lock()
        self._job_context: Dict[str, Dict[str, Any]] = {}

        self.retry_config = RetryConfig(
            max_attempts=self.retry_max_attempts,
            base_delay_seconds=self.retry_base_delay_seconds,
            max_delay_seconds=self.retry_max_delay_seconds,
            backoff_factor=self.retry_backoff_factor,
            jitter_ratio=self.retry_jitter_ratio,
        )

        self.register_known_issue_handler("ReaderValidationError", self._handle_reader_known_issue, replace=True)
        self.register_known_issue_handler("ReaderBatchError", self._handle_reader_known_issue, replace=True)
        self.register_known_issue_handler("ReaderTaskExecutionError", self._handle_reader_known_issue, replace=True)

        self._publish_reader_event("initialized", {"settings": self._settings_snapshot()})
        logger.info(
            "Reader Agent initialized | output_dir=%s default_format=%s max_concurrency=%s shared_cache=%s",
            self.output_dir,
            self.default_output_format,
            self.max_concurrency,
            self.use_shared_cache,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _cfg(self, key: str, default: Any) -> Any:
        return self.agent_config.get(key, default)

    def _load_reader_agent_config(self) -> None:
        self.enabled = coerce_bool(self._cfg("enabled", True), True)
        self.output_dir = str(self._cfg("output_dir", "output/reader") or "output/reader")
        self.default_output_format = normalize_output_format(
            self._cfg("default_output_format", "txt"),
            supported_formats=DEFAULT_OUTPUT_FORMATS,
            default="txt",
        )
        self.max_concurrency = coerce_int(self._cfg("max_concurrency", 4), 4, minimum=1, maximum=64)
        self.fail_fast = coerce_bool(self._cfg("fail_fast", False), False)
        self.auto_recover_low_quality = coerce_bool(self._cfg("auto_recover_low_quality", True), True)
        self.low_quality_threshold = coerce_float(self._cfg("low_quality_threshold", 0.55), 0.55, minimum=0.0, maximum=1.0)
        self.dedupe_files = coerce_bool(self._cfg("dedupe_files", True), True)
        self.include_documents_by_default = coerce_bool(self._cfg("include_documents_by_default", False), False)
        self.include_content_by_default = coerce_bool(self._cfg("include_content_by_default", False), False)
        self.max_result_documents = coerce_int(self._cfg("max_result_documents", 50), 50, minimum=1, maximum=5000)

        # Opt-in capability defaults. Engine policy (limits, formats) lives in reader_config.yaml.
        self.detect_files = coerce_bool(self._cfg("detect_files", False), False)
        self.verify_outputs = coerce_bool(self._cfg("verify_outputs", False), False)
        self.salvage_on_parse_failure = coerce_bool(self._cfg("salvage_on_parse_failure", False), False)
        self.durable_jobs = coerce_bool(self._cfg("durable_jobs", False), False)

        self.use_shared_cache = coerce_bool(self._cfg("use_shared_cache", self._cfg("enable_cache", True)), True)
        self.cache_ttl_seconds = coerce_int(self._cfg("cache_ttl_seconds", 0), 0, minimum=0)
        self.cache_key_prefix = str(self._cfg("cache_key_prefix", "reader_agent.cache") or "reader_agent.cache")
        self.cache_parse_results = coerce_bool(self._cfg("cache_parse_results", True), True)
        self.cache_recovery_results = coerce_bool(self._cfg("cache_recovery_results", True), True)

        self.write_checkpoints = coerce_bool(self._cfg("write_checkpoints", True), True)
        self.checkpoint_key_prefix = str(self._cfg("checkpoint_key_prefix", "reader_agent.checkpoint") or "reader_agent.checkpoint")
        self.checkpoint_index_key = str(self._cfg("checkpoint_index_key", "reader_agent.checkpoints") or "reader_agent.checkpoints")
        self.max_checkpoint_index_entries = coerce_int(
            self._cfg("max_checkpoint_index_entries", 250),
            250,
            minimum=1,
            maximum=10000,
        )
        self.redact_checkpoints = coerce_bool(self._cfg("redact_checkpoints", True), True)

        self.publish_events = coerce_bool(self._cfg("publish_events", True), True)
        self.event_channel = str(self._cfg("event_channel", "reader.events") or "reader.events")
        self.result_key_prefix = str(self._cfg("result_key_prefix", "reader_agent.result") or "reader_agent.result")
        self.summary_key_prefix = str(self._cfg("summary_key_prefix", "reader_agent.summary") or "reader_agent.summary")
        self.error_key_prefix = str(self._cfg("error_key_prefix", "reader_agent.error") or "reader_agent.error")

        self.raise_on_pipeline_error = coerce_bool(self._cfg("raise_on_pipeline_error", False), False)
        self.include_debug_errors = coerce_bool(self._cfg("include_debug_errors", False), False)
        self.include_traceback_errors = coerce_bool(self._cfg("include_traceback_errors", False), False)

        self.retry_max_attempts = coerce_int(self._cfg("retry_max_attempts", 2), 2, minimum=1, maximum=10)
        self.retry_base_delay_seconds = coerce_float(self._cfg("retry_base_delay_seconds", 0.15), 0.15, minimum=0.0, maximum=60.0)
        self.retry_max_delay_seconds = coerce_float(self._cfg("retry_max_delay_seconds", 2.0), 2.0, minimum=0.0, maximum=300.0)
        self.retry_backoff_factor = coerce_float(self._cfg("retry_backoff_factor", 2.0), 2.0, minimum=1.0, maximum=10.0)
        self.retry_jitter_ratio = coerce_float(self._cfg("retry_jitter_ratio", 0.10), 0.10, minimum=0.0, maximum=1.0)

    def _validate_reader_agent_config(self) -> None:
        if not self.enabled:
            logger.warning("Reader Agent is configured as disabled; direct calls will still return a structured error.")
        if not self.output_dir.strip():
            raise ReaderConfigurationError("reader_agent.output_dir cannot be empty")
        if self.max_concurrency < 1:
            raise ReaderConfigurationError("reader_agent.max_concurrency must be >= 1", {"max_concurrency": self.max_concurrency})
        if self.max_result_documents < 1:
            raise ReaderConfigurationError("reader_agent.max_result_documents must be >= 1")

    def _settings_snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "output_dir": self.output_dir,
            "default_output_format": self.default_output_format,
            "max_concurrency": self.max_concurrency,
            "fail_fast": self.fail_fast,
            "auto_recover_low_quality": self.auto_recover_low_quality,
            "low_quality_threshold": self.low_quality_threshold,
            "dedupe_files": self.dedupe_files,
            "use_shared_cache": self.use_shared_cache,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "write_checkpoints": self.write_checkpoints,
            "publish_events": self.publish_events,
            "raise_on_pipeline_error": self.raise_on_pipeline_error,
            "detect_files": self.detect_files,
            "verify_outputs": self.verify_outputs,
            "salvage_on_parse_failure": self.salvage_on_parse_failure,
            "durable_jobs": self.durable_jobs,
        }

    # ------------------------------------------------------------------
    # BaseAgent integration
    # ------------------------------------------------------------------

    def perform_task(self, task_data: Any) -> Dict[str, Any]:
        """Execute a Reader task through parse/recover/convert/merge orchestration."""

        if not self.enabled:
            error = ReaderConfigurationError("Reader Agent is disabled by configuration")
            return self._error_result(error, operation="reader_task")

        try:
            if isinstance(task_data, Mapping):
                operation = str(task_data.get("operation", "") or "").strip().lower()
                if operation in {"capabilities", "describe"}:
                    return build_success_result(operation="reader_capabilities", payload=self.capabilities())

            task = self._normalize_task(task_data)
            result = self._run_coroutine_sync(self._execute_reader_task(task))
            self._store_latest_result(task.run_id, result)
            return result
        except ReaderError as exc:
            self._record_reader_error(exc, task_data)
            if self.raise_on_pipeline_error:
                raise
            return self._error_result(exc, operation="reader_task")
        except Exception as exc:
            error = normalize_reader_error(
                exc,
                message="Reader Agent task execution failed",
                context={"task_preview": truncate_text(stable_json_dumps(json_safe(task_data)), 600)},
                stage=ReaderErrorStage.RUNTIME,
            )
            self._record_reader_error(error, task_data)
            if self.raise_on_pipeline_error:
                raise error from exc
            return self._error_result(error, operation="reader_task")

    def predict(self, state: Any, context: Any = None) -> Dict[str, Any]:
        """Compatibility route for BaseAgent dispatchers that call ``predict``."""

        task = state if isinstance(state, Mapping) else {"instruction": str(state), "context": context}
        return self.perform_task(task)

    def act(self, task_data: Any, context: Any = None) -> Dict[str, Any]:
        """Compatibility route for action-style orchestration."""

        task = task_data if isinstance(task_data, Mapping) else {"instruction": str(task_data), "context": context}
        return self.perform_task(task)

    def extract_performance_metrics(self, result: Any) -> Dict[str, float]:
        metrics = super().extract_performance_metrics(result)
        if isinstance(result, Mapping):
            metadata = result.get("metadata", {})
            if isinstance(metadata, Mapping):
                for key in ("latency_ms", "throughput", "success_rate", "cache_hit_rate", "documents_processed"):
                    value = metadata.get(key)
                    if isinstance(value, (int, float)):
                        metrics[key] = float(value)
        return metrics

    # ------------------------------------------------------------------
    # Task normalization and planning
    # ------------------------------------------------------------------

    def _unwrap_payload(self, task_data: Any) -> Mapping[str, Any]:
        if not isinstance(task_data, Mapping):
            raise ReaderValidationError("ReaderAgent task_data must be a mapping", {"type": type(task_data).__name__})

        payload: Mapping[str, Any] = task_data
        for key in ("task_data", "input_data", "payload"):
            nested = task_data.get(key)
            if isinstance(nested, Mapping) and not any(candidate in task_data for candidate in ("files", "file", "path", "paths")):
                payload = nested
                break
        return payload

    def _normalize_task(self, task_data: Any) -> ReaderAgentTask:
        payload = self._unwrap_payload(task_data)
        instruction = normalize_instruction(payload.get("instruction", payload.get("query", payload.get("message", ""))))

        raw_files = payload.get("files", None)
        if raw_files is None:
            raw_files = payload.get("paths", None)
        if raw_files is None:
            raw_files = payload.get("file", payload.get("path", []))
        requested_mode = str(payload.get("operation", "") or "").strip().lower()
        if requested_mode not in self.SUPPORTED_MODES:
            requested_mode = str(payload.get("mode", "pipeline") or "pipeline").strip().lower()
        if requested_mode in {"jobs", "resume"} and not raw_files:
            files: List[str] = []  # these modes work from a job id, not from input files
        else:
            files = normalize_files(raw_files)
        if self.dedupe_files:
            files = dedupe_preserve_order(files)

        operation = str(payload.get("operation", "") or "").strip().lower()
        mode = operation if operation in self.SUPPORTED_MODES else str(payload.get("mode", "pipeline") or "pipeline").strip().lower()
        if mode not in self.SUPPORTED_MODES:
            mode = "pipeline"
        if mode == "describe":
            mode = "capabilities"

        explicit_format = payload.get("output_format", payload.get("target_format", payload.get("format", None)))
        requested_format = normalize_output_format(
            explicit_format or extract_requested_format(instruction, self.converter.supported_formats(), default=self.default_output_format),
            supported_formats=self.converter.supported_formats(),
            default=self.default_output_format,
        )

        output_dir = str(payload.get("output_dir", self.output_dir) or self.output_dir)
        run_id = safe_filename(payload.get("run_id", uuid.uuid4().hex), fallback="reader_run")

        recover = self._bool_or_default(payload.get("recover", payload.get("recovery", None)), instruction_requests_recovery(instruction))
        merge = self._bool_or_default(payload.get("merge", None), instruction_requests_merge(instruction, file_count=len(files)))
        if mode == "recover":
            recover = True
            merge = False
        elif mode == "merge":
            merge = True
        elif mode in {"parse", "read", "convert"}:
            merge = False
        elif mode in self.UTILITY_MODES:
            recover = False
            merge = False

        fail_fast = self._bool_or_default(payload.get("fail_fast", None), self.fail_fast)
        use_cache = self._bool_or_default(payload.get("use_cache", payload.get("enable_cache", None)), self.use_shared_cache)
        write_checkpoints = self._bool_or_default(payload.get("write_checkpoints", None), self.write_checkpoints)
        include_documents = self._bool_or_default(payload.get("include_documents", None), self.include_documents_by_default)
        include_content = self._bool_or_default(payload.get("include_content", None), self.include_content_by_default)

        detect = self._bool_or_default(payload.get("detect", None), self.detect_files)
        verify = self._bool_or_default(payload.get("verify", None), self.verify_outputs)
        salvage = self._bool_or_default(payload.get("salvage", None), self.salvage_on_parse_failure)
        durable = self._bool_or_default(payload.get("durable", None), self.durable_jobs) or mode == "resume"
        raw_job_id = payload.get("job_id", None)
        job_id = str(raw_job_id).strip() if raw_job_id not in (None, "") else None
        if mode in {"resume"} and not job_id:
            raise ReaderValidationError("resume mode requires a job_id", {"mode": mode})
        deep = self._bool_or_default(payload.get("deep", None), False)
        lossy = self._bool_or_default(payload.get("lossy", None), False)
        compress_method = str(payload.get("compress_method", payload.get("method", "auto")) or "auto").strip().lower()
        raw_package_format = payload.get("package_format", None)
        package_format = str(raw_package_format).strip().lower() if raw_package_format else None

        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), Mapping) else {}
        return ReaderAgentTask(
            instruction=instruction,
            files=files,
            output_format=requested_format,
            output_dir=output_dir,
            mode=mode,
            run_id=run_id,
            recover=recover,
            merge=merge,
            fail_fast=fail_fast,
            use_cache=use_cache,
            write_checkpoints=write_checkpoints,
            include_documents=include_documents,
            include_content=include_content,
            output_filename=str(payload.get("output_filename", payload.get("filename", "")) or "") or None,
            metadata=dict(metadata),
            detect=detect,
            verify=verify,
            salvage=salvage,
            durable=durable,
            job_id=job_id,
            deep=deep,
            compress_method=compress_method,
            package_format=package_format,
            lossy=lossy,
        )

    @staticmethod
    def _bool_or_default(value: Any, default: bool) -> bool:
        if value is None:
            return bool(default)
        return coerce_bool(value, bool(default))

    def build_plan(self, task_data: Mapping[str, Any] | ReaderAgentTask) -> List[Dict[str, Any]]:
        task = task_data if isinstance(task_data, ReaderAgentTask) else self._normalize_task(task_data)
        if task.mode in self.UTILITY_MODES and task.mode != "jobs":
            return [{"action": task.mode, "files": list(task.files)}]
        if task.mode == "jobs":
            return [{"action": "jobs", "job_id": task.job_id}]
        if task.mode == "resume":
            return [{"action": "resume", "job_id": task.job_id}]
        if task.mode in {"parse", "read"}:
            return [{"action": "parse", "files": list(task.files)}]
        if task.mode == "recover":
            return [{"action": "parse", "files": list(task.files)}, {"action": "recover"}]
        if task.mode == "merge":
            return [{"action": "parse", "files": list(task.files)}, {"action": "recover", "condition": "if_requested_or_low_quality"}, {"action": "merge", "output": task.output_format}]
        if task.mode == "convert":
            return [{"action": "parse", "files": list(task.files)}, {"action": "recover", "condition": "if_requested_or_low_quality"}, {"action": "convert", "target": task.output_format}]
        return build_basic_reader_plan(
            task.instruction,
            task.files,
            supported_output_formats=self.converter.supported_formats(),
            default_output_format=task.output_format,
        )

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    async def _execute_reader_task(self, task: ReaderAgentTask) -> Dict[str, Any]:
        timer = OperationTimer(operation="reader_task")
        self._publish_reader_event(
            "started",
            {
                "run_id": task.run_id,
                "mode": task.mode,
                "file_count": len(task.files),
            },
        )

        if task.mode in self.UTILITY_MODES:
            return await self._execute_utility_mode(task, timer)

        if task.mode == "resume":
            task = await asyncio.to_thread(self._task_from_job, task)

        context: Optional[Dict[str, Any]] = None
        extra_warnings: List[str] = []
        if task.durable:
            task, context, shortcut, extra_warnings = await asyncio.to_thread(self._job_begin, task)
            if shortcut is not None:
                return shortcut
        try:
            result = await self._run_pipeline(task, timer, job_context=context)
        except BaseException as exc:
            if context is not None:
                await asyncio.to_thread(self._job_interrupt, task, exc)
            raise
        if extra_warnings:
            result["warnings"] = dedupe_preserve_order([*result.get("warnings", []), *extra_warnings])
        return result

    async def _run_pipeline(
        self,
        task: ReaderAgentTask,
        timer: OperationTimer,
        *,
        job_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        reused_artifacts: List[Dict[str, Any]] = list(job_context["reused_artifacts"]) if job_context else []
        plan = self.build_plan(task)
        if task.write_checkpoints:
            self._write_checkpoint(task.run_id, "plan",
                                   {"task": task.to_dict(), "plan": plan},
            )

        # 1. Parse
        parse_started = time.perf_counter()
        parsed_stage = await self._parse_files(task)

        self._profile_hot_path(
            "reader.parse",
            parse_started,
            run_id=task.run_id,
            files=len(task.files),
            failed=parsed_stage.failed_count,
        )

        if parsed_stage.failed_count and task.fail_fast:
            raise ReaderBatchError(
                "Reader parse stage failed",
                {
                    "stage": "parse",
                    "errors": parsed_stage.error_dicts(),
                },
                failed_count=parsed_stage.failed_count,
                total_count=len(task.files),
            )

        parsed_docs = parsed_stage.items

        if task.write_checkpoints:
            self._write_checkpoint(
                task.run_id,
                "parse",
                parsed_stage.to_dict()
                | {"documents": self._summaries(parsed_docs)},
            )

        # 2. Low-level recovery followed by optional semantic recovery.
        working_docs: Sequence[Mapping[str, Any]] = parsed_docs
        recovery_stage = ReaderStageBatch[Dict[str, Any]].empty("recover")
        semantic_attempted = 0
        semantic_applied = 0

        should_recover = self._should_run_recovery(task, parsed_docs)

        if should_recover and parsed_docs:
            recovery_started = time.perf_counter()
            recovery_stage = await self._recover_documents(task, parsed_docs)

            self._profile_hot_path(
                "reader.recovery",
                recovery_started,
                run_id=task.run_id,
                documents=len(parsed_docs),
                failed=recovery_stage.failed_count,
            )

            semantic_attempted, semantic_applied = self._semantic_recovery_usage(recovery_stage.items)

            # SemanticRecovery is internal to RecoveryEngine. Consequently,
            # duration_ms represents the containing recovery-stage wall time.
            self._profile_hot_path(
                "reader.semantic_recovery",
                recovery_started,
                run_id=task.run_id,
                documents=semantic_attempted,
                applied=semantic_applied,
                measurement_scope="recovery_stage",
            )

            if recovery_stage.failed_count and task.fail_fast:
                raise ReaderBatchError(
                    "Reader recovery stage failed",
                    {
                        "stage": "recovery",
                        "errors": recovery_stage.error_dicts(),
                    },
                    failed_count=recovery_stage.failed_count,
                    total_count=len(parsed_docs),
                )

            working_docs = self._merge_recovered_with_originals(
                parsed_docs,
                recovery_stage.items,
            )

            if task.write_checkpoints:
                self._write_checkpoint(
                    task.run_id,
                    "recover",
                    recovery_stage.to_dict()
                    | {
                        "documents": self._summaries(working_docs),
                        "semantic_attempted": semantic_attempted,
                        "semantic_applied": semantic_applied,
                    },
                )

        # 3. Conversion or merge
        artifacts: List[Dict[str, Any]] = []
        merge_artifact: Optional[Dict[str, Any]] = None
        conversion_stage = ReaderStageBatch[Dict[str, Any]].empty(
            "convert"
        )

        if (
            task.mode not in {"parse", "read", "recover"}
            and working_docs
        ):
            if task.merge:
                merge_artifact = await self._merge_documents(
                    task,
                    working_docs,
                )
                artifacts = [merge_artifact]

                if task.write_checkpoints:
                    self._write_checkpoint(
                        task.run_id,
                        "merge",
                        {"artifact": merge_artifact},
                    )
            else:
                conversion_started = time.perf_counter()
                conversion_stage = await self._convert_documents(
                    task,
                    working_docs,
                )

                self._profile_hot_path(
                    "reader.conversion",
                    conversion_started,
                    run_id=task.run_id,
                    documents=len(working_docs),
                    failed=conversion_stage.failed_count,
                    output_format=task.output_format,
                )

                if conversion_stage.failed_count and task.fail_fast:
                    raise ReaderBatchError(
                        "Reader conversion stage failed",
                        {
                            "stage": "conversion",
                            "errors": conversion_stage.error_dicts(),
                        },
                        failed_count=conversion_stage.failed_count,
                        total_count=len(working_docs),
                    )

                artifacts = [dict(item) for item in conversion_stage.items]

                if task.write_checkpoints:
                    self._write_checkpoint(
                        task.run_id,
                        "convert",
                        conversion_stage.to_dict()
                        | {"artifacts": artifacts},
                    )

        if reused_artifacts:
            # Finished in an earlier run of this job and still intact: reported, not redone.
            artifacts = [*reused_artifacts, *artifacts]
            if task.merge and merge_artifact is None:
                merge_artifact = reused_artifacts[0]

        job_summary: Optional[Dict[str, Any]] = None
        if job_context is not None:
            job_summary = await asyncio.to_thread(self._job_finish, task, parsed_stage)

        stage_errors = (
            parsed_stage.errors
            + recovery_stage.errors
            + conversion_stage.errors
        )
        serialized_stage_errors = [
            error.to_dict() if hasattr(error, 'to_dict') else error # type: ignore
            for error in stage_errors
        ]

        warnings = self._collect_warnings(
            parsed_docs,
            working_docs,
            artifacts,
            serialized_stage_errors,
        )

        completed = timer.stop()
        elapsed = completed.elapsed_seconds
        processed = len(working_docs)
        # Conversion/merge runs only succeed for artifacts they actually produced;
        # parse/read/recover runs succeed per processed document.
        produces_artifacts = task.mode not in {"parse", "read", "recover"}
        success_count = (
            len(artifacts)
            if artifacts or produces_artifacts
            else len(working_docs)
        )
        failed_count = len(stage_errors)

        status = (
            "ok"
            if failed_count == 0
            else "partial"
            if success_count > 0
            else "error"
        )

        # A recovered document is known to be incomplete (or text-only), so the run is not a clean success.
        salvage_notes = [
            {"source": doc.get("source"), **dict((doc.get("metadata") or {}).get("salvage") or {})}
            for doc in working_docs
            if isinstance(doc, Mapping) and isinstance((doc.get("metadata") or {}).get("salvage"), Mapping)
        ]
        if status == "ok" and any(note.get("level") != "full" for note in salvage_notes):
            status = "partial"

        payload: Dict[str, Any] = {
            "run_id": task.run_id,
            "plan": plan,
            "parsed": parsed_stage.to_dict(),
            "recovery": recovery_stage.to_dict(),
            "conversion": conversion_stage.to_dict(),
            "artifacts": artifacts,
        }

        if merge_artifact is not None:
            payload["merge"] = merge_artifact

        if job_summary is not None:
            payload["job"] = job_summary

        if salvage_notes:
            payload["salvage"] = salvage_notes

        if task.include_documents:
            payload["documents"] = self._sanitize_documents(
                working_docs,
                include_content=task.include_content,
            )

        metadata = {
            "mode": task.mode,
            "file_count": len(task.files),
            "documents_processed": processed,
            "output_format": task.output_format,
            "output_dir": task.output_dir,
            "semantic_recovery_attempted": semantic_attempted,
            "semantic_recovery_applied": semantic_applied,
            "reused_items": len(reused_artifacts),
            "latency_ms": round(elapsed * 1000.0, 3),
            "throughput": (
                round(processed / elapsed, 6)
                if elapsed > 0 and processed
                else 0.0
            ),
            "success_rate": round(
                success_count
                / max(1, success_count + failed_count),
                6,
            ),
            "cache_hit_rate": self._cache_hit_rate([*parsed_docs, *working_docs]),
            "settings": self._settings_snapshot(),
        }

        result = {
            "status": status,
            "operation": "reader_task",
            "payload": json_safe(payload),
            "metadata": json_safe(metadata),
            "warnings": dedupe_preserve_order(warnings),
            "timing": completed.to_dict(),
        }

        if serialized_stage_errors:
            result["errors"] = serialized_stage_errors

        if task.write_checkpoints:
            self._write_checkpoint(task.run_id, "complete", result)

        self._publish_reader_event(
            "completed",
            {
                "run_id": task.run_id,
                "status": status,
                "metadata": metadata,
            },
        )
        return result

    @staticmethod
    def _adapt_engine_result(result: Any, *, operation: str) -> Dict[str, Any]:
        """Normalize one engine result at the ReaderAgent boundary."""
    
        if not isinstance(result, Mapping):
            raise ReaderTaskExecutionError(
                "Reader engine returned an unsupported result type",
                {
                    "operation": operation,
                    "type": type(result).__name__,
                },
            )
    
        adapted = json_safe(dict(result), redact=False)
        if not isinstance(adapted, dict):
            raise ReaderTaskExecutionError(
                "Reader engine result could not be normalized to a mapping",
                {
                    "operation": operation,
                    "type": type(adapted).__name__,
                },
            )
    
        status = str(adapted.get("status", "")).strip().lower()
        if not status:
            raise ReaderTaskExecutionError(
                "Reader engine result is missing its status field",
                {
                    "operation": operation,
                    "keys": sorted(adapted),
                },
            )
    
        return adapted
    
    
    @staticmethod
    def _mark_cache_hit(result: Mapping[str, Any]) -> Dict[str, Any]:
        """Return a copy marked as originating from the shared Reader cache."""
    
        cached_result = dict(result)
        raw_metadata = cached_result.get("metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        metadata["cached"] = True
        cached_result["metadata"] = metadata
        return cached_result

    async def _parse_files(self, task: ReaderAgentTask) -> ReaderStageBatch[Dict[str, Any]]:
        async def worker(file_path: str) -> Dict[str, Any]:
            return await self._parse_one(task, file_path)
    
        results = await async_bounded_map(
            task.files,
            worker,
            max_concurrency=self.max_concurrency,
            operation_name="reader_parse",
            return_exceptions=True,
        )
        return self._split_stage_results(results, operation="parse")
    
    
    async def _parse_one(self, task: ReaderAgentTask, file_path: str) -> Dict[str, Any]:
        cache_key = self._shared_parse_cache_key(file_path)
    
        if task.use_cache and self.cache_parse_results:
            cached = self._shared_get(cache_key)
            if isinstance(cached, Mapping):
                adapted = self._adapt_engine_result(
                    cached,
                    operation="parser.parse",
                )
                return await self._annotate_detection(task, file_path, self._mark_cache_hit(adapted))
    
        try:
            raw_result = await retry_async(
                lambda: asyncio.to_thread(self.parser.parse, file_path),
                config=self.retry_config,
                operation_name="parser.parse",
            )
        except ReaderError as exc:
            if not task.salvage or not self._is_salvageable_error(exc):
                raise
            # Recovered content is deliberately not cached: it must never be served later as if it were a clean parse.
            return await self._annotate_detection(task, file_path, await self._salvage_and_parse(task, file_path, exc))
        result = self._adapt_engine_result(raw_result, operation="parser.parse")
    
        if (
            task.use_cache
            and self.cache_parse_results
            and str(result.get("status", "")).lower() == "ok"
        ):
            self._shared_set(cache_key, result, tags=("reader", "reader_parse_cache"))
    
        return await self._annotate_detection(task, file_path, result)

    def _should_run_recovery(self, task: ReaderAgentTask, parsed_docs: Sequence[Mapping[str, Any]]) -> bool:
        if task.mode == "recover" or task.recover:
            return True
        if not self.auto_recover_low_quality:
            return False
        for doc in parsed_docs:
            try:
                cost = estimate_processing_cost(doc)
            except ReaderError:
                continue
            if float(cost.get("quality_score", 1.0) or 1.0) < self.low_quality_threshold or bool(cost.get("requires_attention")):
                return True
        return False

    @staticmethod
    def _semantic_recovery_usage(documents: Sequence[Mapping[str, Any]]) -> Tuple[int, int]:
        """Return semantic-recovery attempted and applied counts."""
    
        attempted = 0
        applied = 0
    
        for document in documents:
            raw_recovery = document.get("recovery", {})
            if not isinstance(raw_recovery, Mapping):
                continue
    
            raw_metadata = raw_recovery.get("metadata", {})
            metadata = (
                raw_metadata
                if isinstance(raw_metadata, Mapping)
                else {}
            )
    
            semantic_was_attempted = (metadata.get("semantic_attempted", metadata.get("used_semantic")) is True)
            semantic_was_applied = (
                metadata.get("semantic_applied") is True
                or str(raw_recovery.get("strategy", "")).lower()
                == "semantic"
            )
    
            if semantic_was_attempted:
                attempted += 1
            if semantic_was_applied:
                applied += 1
    
        return attempted, applied

    async def _recover_documents(self, task: ReaderAgentTask, parsed_docs: Sequence[Mapping[str, Any]]) -> ReaderStageBatch[Dict[str, Any]]:
        async def worker(doc: Mapping[str, Any]) -> Dict[str, Any]:
            return await self._recover_one(task, doc)
    
        results = await async_bounded_map(
            list(parsed_docs),
            worker,
            max_concurrency=self.max_concurrency,
            operation_name="reader_recovery",
            return_exceptions=True,
        )
        return self._split_stage_results(results, operation="recover")
    
    
    async def _recover_one(self, task: ReaderAgentTask, parsed_doc: Mapping[str, Any]) -> Dict[str, Any]:
        doc = validate_parsed_document(parsed_doc)
        source = str(doc.get("source", "unknown"))
        content = str(doc.get("content", ""))
        cache_key = self._shared_recovery_cache_key(source, content)
    
        if task.use_cache and self.cache_recovery_results:
            cached = self._shared_get(cache_key)
            if isinstance(cached, Mapping):
                adapted = self._adapt_engine_result(cached, operation="recovery.recover_and_apply")
                return self._mark_cache_hit(adapted)
    
        raw_result = await retry_async(
            lambda: asyncio.to_thread(self.recovery.recover_and_apply, doc),
            config=self.retry_config,
            operation_name="recovery.recover_and_apply",
        )
        result = self._adapt_engine_result(raw_result, operation="recovery.recover_and_apply")
    
        if (
            task.use_cache
            and self.cache_recovery_results
            and str(result.get("status", "")).lower() == "ok"
        ):
            self._shared_set(cache_key, result, tags=("reader", "reader_recovery_cache"))
    
        return result

    async def _convert_documents(self, task: ReaderAgentTask, docs: Sequence[Mapping[str, Any]]) -> ReaderStageBatch[Dict[str, Any]]:
        async def worker(doc: Mapping[str, Any]) -> Dict[str, Any]:
            return await self._convert_one(task, doc)
    
        results = await async_bounded_map(
            list(docs),
            worker,
            max_concurrency=self.max_concurrency,
            operation_name="reader_conversion",
            return_exceptions=True,
        )
        return self._split_stage_results(results, operation="convert")
    
    async def _convert_one(self, task: ReaderAgentTask, parsed_doc: Mapping[str, Any]) -> Dict[str, Any]:
        raw_result = await retry_async(
            lambda: asyncio.to_thread(
                self.converter.convert,
                dict(parsed_doc),
                task.output_format,
                task.output_dir,
            ),
            config=self.retry_config,
            operation_name="conversion.convert",
        )
        artifact = self._adapt_engine_result(
            raw_result,
            operation="conversion.convert",
        )
        if task.verify or self._job_context.get(task.run_id):
            candidates = [str(parsed_doc.get("source", "")), str(parsed_doc.get("resolved_source", ""))]
            artifact = await asyncio.to_thread(self._post_artifact, task, candidates, artifact, target_format=task.output_format)
        return artifact
    
    
    async def _merge_documents(self, task: ReaderAgentTask, docs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        docs_as_dicts = [dict(doc) for doc in docs]
    
        raw_result = await retry_async(
            lambda: asyncio.to_thread(
                self.converter.merge,
                docs_as_dicts,
                task.output_format,
                task.output_dir,
                task.output_filename or "merged",
            ),
            config=self.retry_config,
            operation_name="conversion.merge",
        )
    
        # Normalise raw_result to a dict if necessary
        if not isinstance(raw_result, Mapping):
            # Assume it's a string path or a simple value
            raw_result = {
                "status": "ok",
                "output_path": str(raw_result),
            }
    
        artifact = self._adapt_engine_result(raw_result, operation="conversion.merge")
        if task.verify or self._job_context.get(task.run_id):
            artifact = await asyncio.to_thread(self._post_artifact, task, [], artifact, target_format=task.output_format)
        return artifact

    # ------------------------------------------------------------------
    # Capability engines: utility modes
    # ------------------------------------------------------------------

    def _jobs(self) -> JobEngine:
        """The durable job journal, created on first use so idle agents never touch the disk."""

        with self._job_lock:
            if self._job_engine is None:
                self._job_engine = JobEngine(integrity=self.integrity)
            return self._job_engine

    async def _run_files(self, task: ReaderAgentTask, func: Any, *, operation: str) -> List[Any]:
        async def worker(file_path: str) -> Any:
            return await asyncio.to_thread(func, file_path)

        return await async_bounded_map(
            task.files,
            worker,
            max_concurrency=self.max_concurrency,
            operation_name=f"reader_{operation}",
            return_exceptions=True,
        )

    async def _execute_utility_mode(self, task: ReaderAgentTask, timer: OperationTimer) -> Dict[str, Any]:
        """Run detect / validate / compress / decompress / salvage / jobs, which bypass the parse pipeline."""

        mode = task.mode
        plan = self.build_plan(task)
        if mode != "jobs" and not task.files:
            raise ReaderValidationError(f"Reader '{mode}' mode requires at least one file", {"mode": mode})

        results: List[Any]
        if mode == "jobs":
            jobs = self._jobs()
            if task.job_id:
                results = [build_success_result(operation="job", payload={"job": await asyncio.to_thread(jobs.get_job, task.job_id)})]
            else:
                listing = await asyncio.to_thread(jobs.list_jobs, limit=self.max_result_documents)
                results = [build_success_result(operation="jobs", payload={"jobs": listing, "count": len(listing)})]
        elif mode == "detect":
            results = await self._run_files(task, self.detector.detect, operation=mode)
        elif mode == "validate":
            results = await self._run_files(task, lambda f: self.integrity.validate_file(f, deep=task.deep), operation=mode)
            # An invalid file is a finding, not an engine failure: keep the report instead of an error envelope.
            results = [
                {**item, "status": "invalid"} if isinstance(item, Mapping) and item.get("status") == "error" and "error" not in item else item
                for item in results
            ]
        elif mode == "salvage":
            results = await self._run_files(task, lambda f: self.salvager.salvage(f), operation=mode)
        elif mode == "decompress":
            results = await self._run_files(task, lambda f: self.compressor.expand(f, task.output_dir), operation=mode)
        else:  # compress
            per_file = task.compress_method == "optimize" or task.compress_method in self.STREAM_COMPRESSION_METHODS
            if per_file:
                results = await self._run_files(
                    task,
                    lambda f: self.compressor.compress([f], task.output_dir, method=task.compress_method, lossy=task.lossy),
                    operation=mode,
                )
            else:
                try:
                    results = [await asyncio.to_thread(
                        self.compressor.compress, list(task.files), task.output_dir,
                        method=task.compress_method, package_format=task.package_format, lossy=task.lossy,
                        name=task.output_filename,
                    )]
                except (ReaderError, OSError) as exc:
                    results = [exc]

        stage = self._split_stage_results(results, operation=mode)
        invalid = sum(1 for item in stage.items if item.get("status") == "invalid")
        degraded = [
            item for item in stage.items
            if isinstance(item.get("payload"), Mapping)
            and (item["payload"].get("report", {}) or {}).get("level") in {"partial", "text_only"}
        ]
        failed = stage.failed_count + invalid
        succeeded = len(stage.items) - invalid
        status = "ok" if failed == 0 and not degraded else ("partial" if succeeded > 0 else "error")

        artifacts: List[Dict[str, Any]] = []
        for item in stage.items:
            payload = item.get("payload") if isinstance(item.get("payload"), Mapping) else {}
            assert payload is not None
            output = payload.get("output_path") or payload.get("repaired_path") or payload.get("output_dir")
            if output:
                artifacts.append({"operation": mode, "output_path": output, "report": payload.get("report")})

        completed = timer.stop()
        result: Dict[str, Any] = {
            "status": status,
            "operation": "reader_task",
            "payload": json_safe({"run_id": task.run_id, "plan": plan, "results": list(stage.items), "artifacts": artifacts}),
            "metadata": json_safe({
                "mode": mode,
                "file_count": len(task.files),
                "succeeded": succeeded,
                "failed": failed,
                "latency_ms": round(completed.elapsed_seconds * 1000.0, 3),
                "settings": self._settings_snapshot(),
            }),
            "warnings": dedupe_preserve_order(self._collect_warnings(stage.items)),
            "timing": completed.to_dict(),
        }
        if stage.errors:
            result["errors"] = [error.to_dict() for error in stage.errors]
        if task.write_checkpoints:
            self._write_checkpoint(task.run_id, "complete", result)
        self._publish_reader_event("completed", {"run_id": task.run_id, "status": status, "metadata": result["metadata"]})
        return result

    # ------------------------------------------------------------------
    # Capability engines: durable jobs
    # ------------------------------------------------------------------

    def _task_from_job(self, task: ReaderAgentTask) -> ReaderAgentTask:
        """Rebuild the original task from a job's stored parameters (resume mode)."""

        job = self._jobs().get_job(str(task.job_id))
        if job.get("kind") != "reader":
            raise JobError("Job was not created by the Reader agent", {"job_id": task.job_id, "kind": job.get("kind")})
        params = dict(job.get("params") or {})
        return self._normalize_task({**params, "run_id": task.run_id, "durable": True, "job_id": task.job_id})

    def _job_begin(self, task: ReaderAgentTask) -> Tuple[ReaderAgentTask, Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
        """Create or resume the task's job.

        Returns ``(task_to_run, job_context, shortcut_result, warnings)``. ``task_to_run`` only lists the files that
        still need work; ``shortcut_result`` is set when the job already completed and its artifacts are intact.
        """

        if task.mode in {"parse", "read", "recover"}:
            logger.warning("durable=true ignored for mode '%s': it produces no artifacts to journal", task.mode)
            return task, None, None, ["durable_ignored_for_mode"]

        jobs = self._jobs()
        files = list(task.files)
        if task.merge:
            items: List[Dict[str, Any]] = [{"item_id": "merge", "key": f"merge:{stable_hash(files)[:16]}", "data": {"files": files}}]
        else:
            items = [{"key": file_path} for file_path in files]
        params = {key: value for key, value in task.to_dict().items() if key not in {"run_id", "durable", "job_id"}}
        identity = {
            "files": files, "mode": task.mode, "format": task.output_format, "dir": task.output_dir, "merge": task.merge,
            "recover": task.recover, "salvage": task.salvage, "filename": task.output_filename, "instruction": task.instruction,
        }

        if task.job_id:
            try:
                job = jobs.get_job(task.job_id)
            except JobNotFoundError:
                job = jobs.create_job("reader", items, params=params, job_id=task.job_id)["payload"]["job"]
        else:
            job = jobs.create_job("reader", items, params=params, idempotency_key=identity)["payload"]["job"]
        job_id = job["job_id"]

        reusable = jobs.reusable_items(job_id) if job["counts"]["done"] else {}
        if job["status"] == "completed":
            if len(reusable) == job["counts"]["done"] and job["counts"]["failed"] == 0:
                return task, None, self._job_shortcut(task, job, reusable), ["job_already_completed"]
            job = jobs.reopen_job(job_id, "recorded artifacts changed or went missing")

        for item in job["items"].values():
            if (item["status"] == "done" and item["item_id"] not in reusable) or item["status"] == "failed":
                jobs.reset_item(job_id, item["item_id"])
        jobs.start_job(job_id, owner=task.run_id)
        job = jobs.get_job(job_id)

        by_source: Dict[str, str] = {}
        for item in job["items"].values():
            if item["item_id"] != "merge":
                by_source[str(expand_path(item["key"]))] = item["item_id"]
        done_ids = {item_id for item_id in reusable if job["items"][item_id]["status"] == "done"}
        if task.merge:
            remaining = [] if "merge" in done_ids else files
        else:
            remaining = [file_path for file_path in files if by_source.get(str(expand_path(file_path))) not in done_ids]
        context = {
            "job_id": job_id,
            "by_source": by_source,
            "merge": task.merge,
            "reused_artifacts": [dict(job["items"][item_id]["result"] or {}) for item_id in job["item_order"] if item_id in done_ids],
            "reused_count": len(done_ids),
            "resumed": job["attempts"] > 1,
            "journal_failures": 0,
        }
        self._job_context[task.run_id] = context
        self._publish_reader_event("job_started", {"run_id": task.run_id, "job_id": job_id, "reused": len(done_ids), "remaining": len(remaining)})
        return replace(task, files=remaining, job_id=job_id), context, None, (["job_resumed"] if context["resumed"] else [])

    def _job_shortcut(self, task: ReaderAgentTask, job: Mapping[str, Any], reusable: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
        """Result for a task whose job already completed with intact artifacts: nothing is reprocessed."""

        artifacts = [dict(reusable[item_id]["result"] or {}) for item_id in job["item_order"] if item_id in reusable]
        payload = {
            "run_id": task.run_id,
            "plan": [{"action": "job_reuse", "job_id": job["job_id"]}],
            "parsed": ReaderStageBatch[Dict[str, Any]].empty("parse").to_dict(),
            "recovery": ReaderStageBatch[Dict[str, Any]].empty("recover").to_dict(),
            "conversion": ReaderStageBatch[Dict[str, Any]].empty("convert").to_dict(),
            "artifacts": artifacts,
            "job": {"job_id": job["job_id"], "status": job["status"], "counts": job["counts"], "reused_items": len(artifacts)},
        }
        if task.merge and artifacts:
            payload["merge"] = artifacts[0]
        return {
            "status": "ok",
            "operation": "reader_task",
            "payload": json_safe(payload),
            "metadata": json_safe({"mode": task.mode, "file_count": len(task.files), "reused_job": True,
                                   "output_format": task.output_format, "output_dir": task.output_dir, "settings": self._settings_snapshot()}),
            "warnings": ["job_already_completed"],
        }

    def _job_record_success(self, task: ReaderAgentTask, candidates: Sequence[str], artifact: Mapping[str, Any]) -> List[str]:
        """Journal one finished artifact. A journal failure is reported, never allowed to lose the artifact."""

        context = self._job_context.get(task.run_id)
        if not context:
            return []
        item_id = "merge" if context["merge"] else next((context["by_source"][c] for c in candidates if c in context["by_source"]), None)
        if item_id is None:
            return []
        try:
            self._jobs().complete_item(context["job_id"], item_id, dict(artifact), artifacts=[str(artifact["output_path"])])
            return []
        except (ReaderError, OSError, KeyError) as exc:
            context["journal_failures"] += 1
            logger.warning("Job %s: failed journaling item %s: %s", context["job_id"], item_id, exc)
            return ["job_journal_failed"]

    def _job_finish(self, task: ReaderAgentTask, parsed_stage: ReaderStageBatch[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Close the job: unfinished items become failed (with their parse error when known), then derive the status."""

        context = self._job_context.pop(task.run_id, None)
        if not context:
            return None
        jobs = self._jobs()
        job_id = context["job_id"]
        try:
            job = jobs.get_job(job_id)
            parse_errors: Dict[str, str] = {}
            for error in parsed_stage.errors:
                index = error.details.get("index") if isinstance(error.details, Mapping) else None
                if isinstance(index, int) and 0 <= index < len(task.files):
                    parse_errors[str(expand_path(task.files[index]))] = error.message
            for item in job["items"].values():
                if item["status"] in {"pending", "running"}:
                    reason = parse_errors.get(str(expand_path(item["key"])), "not processed: see the run's errors") if item["item_id"] != "merge" else "merge did not complete"
                    jobs.fail_item(job_id, item["item_id"], {"message": reason})
            job = jobs.finish_job(job_id) if jobs.get_job(job_id)["status"] == "running" else jobs.get_job(job_id)
        except (ReaderError, OSError) as exc:
            logger.warning("Job %s: could not be finalized: %s", job_id, exc)
            return {"job_id": job_id, "status": "unknown", "finalize_error": str(exc)}
        return {
            "job_id": job_id, "status": job["status"], "counts": job["counts"], "attempts": job["attempts"],
            "reused_items": context["reused_count"], "resumed": context["resumed"], "journal_failures": context["journal_failures"],
        }

    def _job_interrupt(self, task: ReaderAgentTask, exc: BaseException) -> None:
        """A task that raised leaves its job resumable instead of 'running' until the stale timeout."""

        context = self._job_context.pop(task.run_id, None)
        if not context:
            return
        try:
            jobs = self._jobs()
            jobs.interrupt_job(context["job_id"], f"{type(exc).__name__}: {exc}"[:300])
        except Exception as inner:  # never mask the original failure
            logger.warning("Job %s: could not mark interrupted: %s", context["job_id"], inner)

    # ------------------------------------------------------------------
    # Capability engines: parse-time salvage, detection and output verification
    # ------------------------------------------------------------------

    @staticmethod
    def _is_salvageable_error(exc: BaseException) -> bool:
        """Container damage is salvageable; empty content, decode problems and unsupported types are not."""

        if isinstance(exc, (EmptyContentError, DecodeFailureError, UnsupportedFormatError)):
            return False
        return isinstance(exc, (ParseFailureError, CorruptContentError, MalformedReaderDocumentError))

    async def _salvage_and_parse(self, task: ReaderAgentTask, file_path: str, original: ReaderError) -> Dict[str, Any]:
        try:
            salvaged = await asyncio.to_thread(self.salvager.salvage, file_path)
        except ReaderError as exc:
            logger.warning("Salvage of %s failed (%s); reporting the original parse error", file_path, exc)
            raise original from exc
        payload = salvaged["payload"]
        repaired = payload.get("repaired_path")
        if not repaired:
            raise original  # the file validates as intact, so the parse failure is not container damage
        parsed = self._adapt_engine_result(await asyncio.to_thread(self.parser.parse, repaired), operation="parser.parse")
        report = payload["report"]
        path = expand_path(file_path)
        parsed.update({
            "source": str(path),
            "resolved_source": str(resolve_path(path)),
            "filename": path.name,
            "stem": path.stem,
            "extension": normalize_extension(path.suffix, include_dot=True),
        })
        metadata = dict(parsed.get("metadata") or {})
        metadata.pop("cached", None)
        metadata["salvage"] = {
            "level": report["level"],
            "strategy": report["strategy"],
            "repaired_path": repaired,
            "recovered_count": report["recovered_count"],
            "lost_count": report["lost_count"],
            "lost_parts": report["lost_parts"][:20],
            "original_error": {"code": getattr(original, "code", None), "message": str(original)},
        }
        parsed["metadata"] = metadata
        parsed["warnings"] = dedupe_preserve_order([*(parsed.get("warnings") or []), f"salvaged:{report['level']}"])
        return parsed

    async def _annotate_detection(self, task: ReaderAgentTask, file_path: str, result: Dict[str, Any]) -> Dict[str, Any]:
        if not task.detect:
            return result
        try:
            detection = (await asyncio.to_thread(self.detector.detect, file_path))["payload"]
        except ReaderError as exc:
            logger.warning("Detection of %s failed: %s", file_path, exc)
            return result
        annotated = dict(result)
        metadata = dict(annotated.get("metadata") or {})
        metadata["detection"] = {
            key: detection.get(key)
            for key in ("format_id", "family", "mime", "detected_by", "confidence", "extension_mismatch", "damaged", "encrypted", "warnings")
        }
        warnings = list(annotated.get("warnings") or [])
        if detection.get("extension_mismatch"):
            warnings.append(f"extension_mismatch:{detection.get('content_format') or detection.get('format_id')}")
        if detection.get("damaged"):
            warnings.append("container_damaged")
        annotated["metadata"] = metadata
        annotated["warnings"] = dedupe_preserve_order(warnings)
        return annotated

    def _post_artifact(self, task: ReaderAgentTask, candidates: Sequence[str], artifact: Dict[str, Any], *, target_format: str) -> Dict[str, Any]:
        """Verify a freshly written artifact (when asked) and journal it (when durable)."""

        output_path = artifact.get("output_path")
        warnings: List[str] = []
        if output_path and task.verify:
            report = self.integrity.verify_output(output_path, target_format)["payload"]
            if not report["valid"]:
                failed = [check for check in report["checks"] if not check["ok"] and check["severity"] == "error"]
                raise IntegrityError(str(output_path), "Reader output failed integrity verification", checks=failed,
                                     context={"left_in_place_for_inspection": str(output_path)})
            artifact["verification"] = {"integrity_status": report["integrity_status"], "sha256": report["hashes"].get("sha256"),
                                        "size_bytes": report["hashes"].get("size_bytes")}
        if output_path:
            warnings += self._job_record_success(task, candidates, artifact)
        if warnings:
            artifact["warnings"] = dedupe_preserve_order([*(artifact.get("warnings") or []), *warnings])
        return artifact

    # ------------------------------------------------------------------
    # Stage, cache, checkpoint, and event helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _merge_recovered_with_originals(
        original_docs: Sequence[Mapping[str, Any]],
        recovered_docs: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep original parse order while replacing only successfully recovered docs."""

        recovered_by_source = {str(doc.get("source", "unknown")): dict(doc) for doc in recovered_docs if isinstance(doc, Mapping)}
        merged: List[Dict[str, Any]] = []
        for original in original_docs:
            source = str(original.get("source", "unknown"))
            merged.append(recovered_by_source.get(source, dict(original)))
        return merged

    def _stage_error_from_payload(self, payload: Mapping[str, Any], *, operation: str, index: int) -> ReaderStageError:
        safe_payload = json_safe(dict(payload), redact=True)
        if not isinstance(safe_payload, dict):
            safe_payload = {"payload": safe_payload}

        # Preserve both the stage location and the complete engine envelope.
        details = {"operation": operation, "index": index, **safe_payload}
        nested_error = safe_payload.get("error")
        descriptor = (
            nested_error
            if isinstance(nested_error, Mapping)
            else safe_payload
        )
        raw_policy = descriptor.get("policy", {})
        policy = raw_policy if isinstance(raw_policy, Mapping) else {}

        return ReaderStageError(
            stage=str(descriptor.get("stage") or operation),
            code=str(
                descriptor.get("code")
                or "READER_STAGE_FAILURE"
            ),
            message=str(
                descriptor.get("public_message")
                or descriptor.get("message")
                or "Reader stage failed"
            ),
            retryable=policy.get("retryable") is True,
            recoverable=policy.get("recoverable") is True,
            details=details,
        )


    def _split_stage_results(self, results: Sequence[Any], *, operation: str) -> ReaderStageBatch[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        errors: List[ReaderStageError] = []

        for index, item in enumerate(results):
            if isinstance(item, ReaderError):
                errors.append(
                    self._stage_error_from_payload(
                        item.to_public_dict(),
                        operation=operation,
                        index=index,
                    )
                )
                continue

            if isinstance(item, BaseException):
                errors.append(
                    self._stage_error_from_payload(
                        reader_error_payload(
                            item,
                            include_debug=self.include_debug_errors,
                            include_traceback=self.include_traceback_errors,
                        ),
                        operation=operation,
                        index=index,
                    )
                )
                continue

            if (
                isinstance(item, Mapping)
                and str(item.get("status", "")).lower() == "error"
            ):
                # Keep status, operation, error, metadata, timing and any other
                # outer fields supplied by the engine.
                errors.append(
                    self._stage_error_from_payload(
                        {
                            "operation": operation,
                            "index": index,
                            **json_safe(dict(item), redact=True),
                        },
                        operation=operation,
                        index=index,
                    )
                )
                continue

            if isinstance(item, Mapping):
                try:
                    items.append(self._adapt_engine_result(item, operation=operation))
                except ReaderError as exc:
                    errors.append(
                        self._stage_error_from_payload(
                            exc.to_public_dict(),
                            operation=operation,
                            index=index,
                        )
                    )
                continue

            unsupported = ReaderTaskExecutionError(
                "Reader stage returned an unsupported result type",
                {
                    "operation": operation,
                    "index": index,
                    "type": type(item).__name__,
                },
            )
            errors.append(
                self._stage_error_from_payload(
                    unsupported.to_public_dict(),
                    operation=operation,
                    index=index,
                )
            )

        return ReaderStageBatch(
            stage=operation,
            items=tuple(items),
            errors=tuple(errors),
        )

    def _shared_parse_cache_key(self, file_path: str) -> str:
        return f"{self.cache_key_prefix}:parse:{parse_cache_key(file_path)}:{self.CACHE_VERSION}"

    def _shared_recovery_cache_key(self, source: str, content: str) -> str:
        policy = {
            'agent': self.CACHE_VERSION,
            'recovery_contract': self.RECOVERY_CACHE_VERSION,
        }
        key = recovery_cache_key(source, content, policy=policy)
        return f"{self.cache_key_prefix}:recovery:{key}"

    def _shared_get(self, key: str) -> Any:
        if not self.use_shared_cache:
            return None
        with contextlib.suppress(Exception):
            return self.shared_memory.get(key)
        return None

    def _shared_set(self, key: str, value: Any, *, tags: Iterable[str] = ()) -> None:
        if not self.use_shared_cache:
            return
        ttl = None if self.cache_ttl_seconds <= 0 else self.cache_ttl_seconds
        with contextlib.suppress(Exception):
            self.shared_memory.set(key, json_safe(value, redact=False), ttl=ttl, tags=list(tags), metadata={"agent": self.name})

    def _write_checkpoint(self, run_id: str, step: str, payload: Mapping[str, Any]) -> None:
        if not self.write_checkpoints:
            return
        key = f"{self.checkpoint_key_prefix}:{run_id}:{safe_filename(step, fallback='step')}"
        record = {
            "run_id": run_id,
            "step": step,
            "timestamp": time.time(),
            "payload": json_safe(payload, redact=self.redact_checkpoints),
        }
        with contextlib.suppress(Exception):
            self.shared_memory.set(key, record, tags=["reader", "reader_checkpoint"], metadata={"agent": self.name, "step": step})
            index = self.shared_memory.get(self.checkpoint_index_key) or []
            if not isinstance(index, list):
                index = []
            index.append({"run_id": run_id, "step": step, "key": key, "timestamp": record["timestamp"]})
            self.shared_memory.set(self.checkpoint_index_key, index[-self.max_checkpoint_index_entries:])

    def _publish_reader_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if not self.publish_events:
            return
        event = {"event_type": event_type, "timestamp": time.time(), "agent": self.name, "payload": json_safe(payload, redact=True)}
        with contextlib.suppress(Exception):
            if hasattr(self.shared_memory, "publish"):
                self.shared_memory.publish(self.event_channel, event)
            if hasattr(self.shared_memory, "notify"):
                self.shared_memory.notify(self.event_channel, event)


    def _profile_hot_path(self, path: str, started: float, **extra: Any) -> None:
        duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        payload = {"path": path, "duration_ms": round(duration_ms, 4), **{k: v for k, v in extra.items() if v is not None}}
        self._publish_reader_event("hot_path_profile", payload)

    def _store_latest_result(self, run_id: str, result: Mapping[str, Any]) -> None:
        with contextlib.suppress(Exception):
            self.shared_memory.set(f"{self.result_key_prefix}:{run_id}", json_safe(result, redact=False), tags=["reader", "reader_result"])
            summary = {
                "run_id": run_id,
                "status": result.get("status"),
                "operation": result.get("operation"),
                "metadata": result.get("metadata", {}),
                "warnings": result.get("warnings", []),
            }
            self.shared_memory.set(f"{self.summary_key_prefix}:{run_id}", json_safe(summary, redact=True), tags=["reader", "reader_summary"])

    def _record_reader_error(self, exc: BaseException, task_data: Any) -> None:
        payload = reader_error_payload(exc, include_debug=self.include_debug_errors, include_traceback=self.include_traceback_errors)
        payload["task_preview"] = truncate_text(stable_json_dumps(json_safe(task_data, redact=True)), 600)
        with contextlib.suppress(Exception):
            key = f"{self.error_key_prefix}:{uuid.uuid4().hex}"
            self.shared_memory.set(key, payload, tags=["reader", "reader_error"])
        self._publish_reader_event("error", payload)

    # ------------------------------------------------------------------
    # Result shaping and public utilities
    # ------------------------------------------------------------------

    def _summaries(self, docs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for doc in list(docs)[: self.max_result_documents]:
            with contextlib.suppress(Exception):
                summaries.append(summarize_parsed_document(doc, include_hash=True))
        return summaries

    def _sanitize_documents(self, docs: Sequence[Mapping[str, Any]], *, include_content: bool) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for doc in list(docs)[: self.max_result_documents]:
            safe_doc = dict(doc)
            if not include_content:
                content = str(safe_doc.get("content", ""))
                safe_doc["content_preview"] = truncate_text(content, 500) if content else ""
                safe_doc.pop("content", None)
            output.append(json_safe(safe_doc, redact=False))
        return output

    @staticmethod
    def _cache_hit_rate(docs: Sequence[Mapping[str, Any]]) -> float:
        if not docs:
            return 0.0
        hits = 0
        for doc in docs:
            metadata = doc.get("metadata", {}) if isinstance(doc, Mapping) else {}
            if isinstance(metadata, Mapping) and metadata.get("cached"):
                hits += 1
        return round(hits / max(1, len(docs)), 6)

    @staticmethod
    def _collect_warnings(*groups: Any) -> List[str]:
        warnings: List[str] = []
        for group in groups:
            if isinstance(group, Mapping):
                warnings.extend(str(item) for item in group.get("warnings", []) or [])
                continue
            if isinstance(group, Sequence) and not isinstance(group, (str, bytes, bytearray)):
                for item in group:
                    if isinstance(item, Mapping):
                        warnings.extend(str(entry) for entry in item.get("warnings", []) or [])
        return dedupe_preserve_order(warnings)

    def _error_result(self, exc: BaseException, *, operation: str) -> Dict[str, Any]:
        result = build_error_result(
            exc,
            operation=operation,
            include_debug=self.include_debug_errors,
            include_traceback=self.include_traceback_errors,
        )
        self._publish_reader_event("failed", result)
        return result

    def capabilities(self) -> Dict[str, Any]:
        return {
            "agent": "ReaderAgent",
            "version": __version__,
            "modes": sorted(self.SUPPORTED_MODES),
            "settings": self._settings_snapshot(),
            "parser": self.parser.capabilities(),
            "recovery": self.recovery.capabilities(),
            "conversion": self.converter.describe(),
            "detection": self.detector.capabilities(),
            "integrity": self.integrity.capabilities(),
            "compression": self.compressor.capabilities(),
            "salvage": self.salvager.capabilities(),
            # The job store is created lazily, so describing it must not create it.
            "jobs": {"enabled_by_default": self.durable_jobs, "modes": ["resume", "jobs"], "task_flags": ["durable", "job_id"]},
        }

    def summarize(self, task_data: Mapping[str, Any]) -> Dict[str, Any]:
        task = self._normalize_task(task_data)
        return {
            "task": task.to_dict(),
            "plan": self.build_plan(task),
            "capabilities": self.capabilities(),
        }

    def _handle_reader_known_issue(self, _agent: Any, task_data: Any, error_info: Mapping[str, Any], _issue_handler: Any = None) -> Dict[str, Any]:
        """Local BaseAgent known-issue recovery boundary for Reader failures."""

        message = str(error_info.get("error_message", ""))
        if "max_concurrency" in message.lower():
            # Copy task_data and ensure it's a mutable dict
            if isinstance(task_data, Mapping):
                fallback_task = dict(task_data)
            else:
                fallback_task = {"instruction": str(task_data)}
            # Set max_concurrency to 1 – the value is an integer, acceptable for the agent
            fallback_task["max_concurrency"] = 1   # type: ignore[assignment]
            try:
                result = self.perform_task(fallback_task)
                return {"recovered": True, "result": result, "strategy": "reader_concurrency_fallback"}
            except Exception as exc:
                return {"status": "failed", "reason": str(exc), "strategy": "reader_concurrency_fallback"}
        return {"status": "failed", "reason": "No Reader-specific known issue fallback matched."}

    # ------------------------------------------------------------------
    # Async bridge
    # ------------------------------------------------------------------

    @staticmethod
    def _run_coroutine_sync(coro: Awaitable[T]) -> T:
        """Run an awaitable synchronously, handling both running loop and no loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop – we can run the coroutine
            return asyncio.run(coro)   # type: ignore[arg-type]  # asyncio.run accepts Awaitable
        else:
            # Already in an event loop – we can't block, so we create a new thread
            import threading
            result_holder: Dict[str, Any] = {}
            def runner():
                try:
                    result_holder["result"] = asyncio.run(coro) # type: ignore
                except BaseException as e:
                    result_holder["error"] = e
            t = threading.Thread(target=runner, daemon=True)
            t.start()
            t.join()
            if "error" in result_holder:
                raise result_holder["error"]
            return result_holder["result"]


if __name__ == "__main__":
    print("\n=== Running Reader Agent ===\n")
    printer.status("TEST", "Reader Agent initialized", "info")
    import tempfile

    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()

    with tempfile.TemporaryDirectory(prefix="reader_agent_test_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        file_one = tmp_path / "alpha.txt"
        file_two = tmp_path / "beta.txt"
        output_dir = tmp_path / "out"

        file_one.write_text("Architecture organizes light, memory, and threshold.\n", encoding="utf-8")
        file_two.write_text("Reader recovery removes\x00 detectable corruption without fabricating facts.\n", encoding="utf-8")

        agent = ReaderAgent(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config={
                "output_dir": str(output_dir),
                "default_output_format": "txt",
                "max_concurrency": 2,
                "use_shared_cache": True,
                "write_checkpoints": True,
                "publish_events": True,
                "include_documents_by_default": True,
                "include_content_by_default": False,
            },
        )

        result = agent.execute(
            {
                "instruction": "recover and merge to md",
                "files": [str(file_one), str(file_two)],
                "output_dir": str(output_dir),
                "output_format": "md",
                "include_documents": True,
            }
        )
        assert result["status"] in {"ok", "partial"}, result
        assert result["payload"]["parsed"]["success_count"] == 2, result
        assert result["payload"].get("merge", {}).get("status") == "ok", result
        assert Path(result["payload"]["merge"]["output_path"]).exists(), result

        parse_only = agent.perform_task({"operation": "parse", "files": str(file_one), "include_documents": True})
        assert parse_only["status"] == "ok", parse_only
        assert parse_only["payload"]["parsed"]["success_count"] == 1, parse_only

        caps = agent.perform_task({"operation": "capabilities", "files": str(file_one)})
        assert caps["status"] == "ok", caps
        assert caps["payload"]["agent"] == "ReaderAgent", caps

    print("\n=== Test ran successfully ===\n")
