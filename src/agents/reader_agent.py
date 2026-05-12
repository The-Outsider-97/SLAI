from __future__ import annotations

__version__ = "2.2.0"

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

Configuration boundary
----------------------
``ReaderAgent`` reads the ``reader_agent`` section from
``src/agents/base/configs/agents_config.yaml`` through the global config loader.
It does not load or copy the Reader subsystem config directly. Internal engines
remain responsible for their own subsystem settings through their existing
config loader.
"""

import asyncio
import contextlib
import threading
import time
import uuid

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Coroutine, Dict, Iterable, List, Mapping, TypeVar, Optional, Sequence, Tuple

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

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True)
class ReaderAgentStageResult:
    """Container for one pipeline stage's successful items and normalized errors."""

    items: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]

    @property
    def failed_count(self) -> int:
        return len(self.errors)

    @property
    def success_count(self) -> int:
        return len(self.items)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success_count": self.success_count,
            "failed_count": self.failed_count,
            "errors": list(self.errors),
        }


class ReaderAgent(BaseAgent):
    """Top-level Reader orchestration agent.

    The agent intentionally avoids ``ReaderMemory``. All orchestration state,
    checkpoint records, cache entries, and events are written through
    ``self.shared_memory`` so the Reader participates in the same runtime fabric
    as the other SLAI agents.
    """

    SUPPORTED_MODES = {"pipeline", "parse", "read", "recover", "convert", "merge", "capabilities", "describe"}
    CACHE_VERSION = "reader_agent.v2"

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

        fail_fast = self._bool_or_default(payload.get("fail_fast", None), self.fail_fast)
        use_cache = self._bool_or_default(payload.get("use_cache", payload.get("enable_cache", None)), self.use_shared_cache)
        write_checkpoints = self._bool_or_default(payload.get("write_checkpoints", None), self.write_checkpoints)
        include_documents = self._bool_or_default(payload.get("include_documents", None), self.include_documents_by_default)
        include_content = self._bool_or_default(payload.get("include_content", None), self.include_content_by_default)

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
        )

    @staticmethod
    def _bool_or_default(value: Any, default: bool) -> bool:
        if value is None:
            return bool(default)
        return coerce_bool(value, bool(default))

    def build_plan(self, task_data: Mapping[str, Any] | ReaderAgentTask) -> List[Dict[str, Any]]:
        task = task_data if isinstance(task_data, ReaderAgentTask) else self._normalize_task(task_data)
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
        self._publish_reader_event("started", {"run_id": task.run_id, "mode": task.mode, "file_count": len(task.files)})

        plan = self.build_plan(task)
        if task.write_checkpoints:
            self._write_checkpoint(task.run_id, "plan", {"task": task.to_dict(), "plan": plan})

        parsed_stage = await self._parse_files(task)
        if parsed_stage.failed_count and task.fail_fast:
            raise ReaderBatchError("Reader parse stage failed", {"stage": "parse", "errors": parsed_stage.errors}, failed_count=parsed_stage.failed_count, total_count=len(task.files))
        parsed_docs = parsed_stage.items
        if task.write_checkpoints:
            self._write_checkpoint(task.run_id, "parse", parsed_stage.to_dict() | {"documents": self._summaries(parsed_docs)})

        working_docs = parsed_docs
        recovery_stage = ReaderAgentStageResult(items=[], errors=[])
        should_recover = self._should_run_recovery(task, parsed_docs)
        if should_recover and parsed_docs:
            recovery_stage = await self._recover_documents(task, parsed_docs)
            if recovery_stage.failed_count and task.fail_fast:
                raise ReaderBatchError("Reader recovery stage failed", {"stage": "recovery", "errors": recovery_stage.errors}, failed_count=recovery_stage.failed_count, total_count=len(parsed_docs))
            working_docs = self._merge_recovered_with_originals(parsed_docs, recovery_stage.items)
            if task.write_checkpoints:
                self._write_checkpoint(task.run_id, "recover", recovery_stage.to_dict() | {"documents": self._summaries(working_docs)})

        artifacts: List[Dict[str, Any]] = []
        merge_artifact: Optional[Dict[str, Any]] = None
        conversion_stage = ReaderAgentStageResult(items=[], errors=[])

        if task.mode not in {"parse", "read", "recover"} and working_docs:
            if task.merge:
                merge_artifact = await self._merge_documents(task, working_docs)
                artifacts = [merge_artifact]
                if task.write_checkpoints:
                    self._write_checkpoint(task.run_id, "merge", {"artifact": merge_artifact})
            else:
                conversion_stage = await self._convert_documents(task, working_docs)
                if conversion_stage.failed_count and task.fail_fast:
                    raise ReaderBatchError("Reader conversion stage failed", {"stage": "conversion", "errors": conversion_stage.errors}, failed_count=conversion_stage.failed_count, total_count=len(working_docs))
                artifacts = conversion_stage.items
                if task.write_checkpoints:
                    self._write_checkpoint(task.run_id, "convert", conversion_stage.to_dict() | {"artifacts": artifacts})

        stage_errors = parsed_stage.errors + recovery_stage.errors + conversion_stage.errors
        warnings = self._collect_warnings(parsed_docs, working_docs, artifacts, stage_errors)
        completed = timer.stop()
        elapsed = completed.elapsed_seconds
        processed = len(working_docs)
        success_count = len(artifacts) if artifacts else len(working_docs)
        failed_count = len(stage_errors)
        status = "ok" if failed_count == 0 else ("partial" if success_count > 0 else "error")

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
        if task.include_documents:
            payload["documents"] = self._sanitize_documents(working_docs, include_content=task.include_content)

        metadata = {
            "mode": task.mode,
            "file_count": len(task.files),
            "documents_processed": processed,
            "output_format": task.output_format,
            "output_dir": task.output_dir,
            "latency_ms": round(elapsed * 1000.0, 3),
            "throughput": round(processed / elapsed, 6) if elapsed > 0 and processed else 0.0,
            "success_rate": round(success_count / max(1, success_count + failed_count), 6),
            "cache_hit_rate": self._cache_hit_rate(parsed_docs + working_docs),
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
        if stage_errors:
            result["errors"] = stage_errors

        if task.write_checkpoints:
            self._write_checkpoint(task.run_id, "complete", result)
        self._publish_reader_event("completed", {"run_id": task.run_id, "status": status, "metadata": metadata})
        return result

    async def _parse_files(self, task: ReaderAgentTask) -> ReaderAgentStageResult:
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
                result = dict(cached)
                result.setdefault("metadata", {})["cached"] = True
                return json_safe(result)

        result = await retry_async(
            lambda: asyncio.to_thread(self.parser.parse, file_path),
            config=self.retry_config,
            operation_name="parser.parse",
        )
        if task.use_cache and self.cache_parse_results and isinstance(result, Mapping) and result.get("status") == "ok":
            self._shared_set(cache_key, result, tags=("reader", "reader_parse_cache"))
        return dict(result)

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

    async def _recover_documents(self, task: ReaderAgentTask, parsed_docs: Sequence[Mapping[str, Any]]) -> ReaderAgentStageResult:
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
                result = dict(cached)
                result.setdefault("metadata", {})["cached"] = True
                return json_safe(result)

        result = await retry_async(
            lambda: asyncio.to_thread(self.recovery.recover_and_apply, doc),
            config=self.retry_config,
            operation_name="recovery.recover_and_apply",
        )
        if task.use_cache and self.cache_recovery_results and isinstance(result, Mapping) and result.get("status") == "ok":
            self._shared_set(cache_key, result, tags=("reader", "reader_recovery_cache"))
        return dict(result)

    async def _convert_documents(self, task: ReaderAgentTask, docs: Sequence[Mapping[str, Any]]) -> ReaderAgentStageResult:
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
        return await retry_async(
            lambda: asyncio.to_thread(self.converter.convert, dict(parsed_doc), task.output_format, task.output_dir),
            config=self.retry_config,
            operation_name="conversion.convert",
        )

    async def _merge_documents(self, task: ReaderAgentTask, docs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        docs_as_dicts = [dict(doc) for doc in docs]   # Convert Mapping → dict
        return await retry_async(
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

    def _split_stage_results(self, results: Sequence[Any], *, operation: str) -> ReaderAgentStageResult:
        items: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        for index, item in enumerate(results):
            if isinstance(item, ReaderError):
                errors.append(item.to_public_dict())
            elif isinstance(item, BaseException):
                errors.append(reader_error_payload(item, include_debug=self.include_debug_errors, include_traceback=self.include_traceback_errors))
            elif isinstance(item, Mapping) and str(item.get("status", "")).lower() == "error":
                error_payload = item.get("error")
                if error_payload is None:
                    error_dict: Dict[str, Any] = {"message": "unknown Reader error"}
                elif isinstance(error_payload, Mapping):
                    # Convert all keys to strings (handle dict[bytes, bytes] etc.)
                    error_dict = {str(k): v for k, v in error_payload.items()}
                else:
                    error_dict = {"message": str(error_payload)}
                errors.append(error_dict)
            elif isinstance(item, Mapping):
                items.append(dict(item))
            else:
                errors.append(
                    ReaderTaskExecutionError(
                        "Reader stage returned an unsupported result type",
                        {"operation": operation, "index": index, "type": type(item).__name__},
                    ).to_public_dict()
                )
        return ReaderAgentStageResult(items=items, errors=errors)

    def _shared_parse_cache_key(self, file_path: str) -> str:
        return f"{self.cache_key_prefix}:parse:{parse_cache_key(file_path)}:{self.CACHE_VERSION}"

    def _shared_recovery_cache_key(self, source: str, content: str) -> str:
        return f"{self.cache_key_prefix}:recovery:{recovery_cache_key(source, content, policy={'agent': self.CACHE_VERSION})}"

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
