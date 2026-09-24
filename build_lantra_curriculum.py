"""Build a deterministic, agent-enriched curriculum for SLAI LANTRA.

This entry point is intentionally offline. Knowledge, Reasoning, and optional
Perception agents are used once to construct provenance-rich JSONL artifacts;
``train_lantra.py`` then consumes those artifacts without calling agents inside
its optimizer loop.

Default output:
    data/processed/lantra/agent_enriched/

The emitted JSONL files use only LANTRA's existing seven task schemas. No new
``task`` enum is introduced.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import shutil
import threading
import time

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, cast

from src.training.curriculum_builder import LantraCurriculumBuilder
from src.training.enrichment_contracts import *
from src.training.knowledge_adapter import KnowledgeAdapter
from src.training.perception_adapter import PerceptionAdapter
from src.training.reasoning_adapter import ReasoningAdapter
from src.training.source_adapter import CanonicalLantraSourceAdapter
from src.utils.configuration import bind_config
from logs.logger import get_logger, PrettyPrinter


LOGGER = get_logger("LANTRA Curriculum Builder")
printer = PrettyPrinter()

knowledge = None
reasoning = None
perception = None

DEFAULT_CONFIG = Path("src/training/configs/lantra_curriculum.yaml")
_CONFIG = bind_config(DEFAULT_CONFIG)

DEFAULT_PROGRESS_INTERVAL = 30.0
DEFAULT_PROGRESS_EVERY_FILES = 10
LARGE_SOURCE_BYTES = 256 * 1024 * 1024
LARGE_DOCUMENT_CHARS = 256 * 1024 * 1024


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    runtime_metadata: Dict[str, Any] = {}
    result = None
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _human_bytes(value: int) -> str:
    size = float(max(0, int(value)))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _emit_progress(status: str, message: str, **fields: Any) -> None:
    """Emit an immediately visible, line-oriented progress event.

    ``print(..., flush=True)`` is intentional here. The curriculum builder can
    spend hours inside third-party PDF extraction or SLAI agent calls, and the
    console progress stream must remain visible even when the application's
    logging level filters INFO messages.
    """

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    suffix = " | ".join(
        f"{key}={value}"
        for key, value in fields.items()
        if value is not None and value != ""
    )
    line = f"[{timestamp}] [LANTRA][{status}] {message}"
    if suffix:
        line += " | " + suffix
    print(line, flush=True)


class _ProgressStage:
    """Context manager that reports start/end and periodic liveness heartbeats."""

    def __init__(
        self,
        name: str,
        *,
        heartbeat_seconds: float,
        detail: Optional[str] = None,
    ) -> None:
        self.name = name
        self.heartbeat_seconds = max(0.0, float(heartbeat_seconds))
        self.detail = detail
        self.started = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_ProgressStage":
        self.started = time.perf_counter()
        _emit_progress("START", self.name, detail=self.detail)
        if self.heartbeat_seconds > 0:
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"lantra-progress-{self.name}",
                daemon=True,
            )
            self._thread.start()
        return self

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            _emit_progress(
                "WORKING",
                self.name,
                elapsed=_format_duration(time.perf_counter() - self.started),
                detail=self.detail,
            )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        elapsed = _format_duration(time.perf_counter() - self.started)
        if exc_type is None:
            _emit_progress("DONE", self.name, elapsed=elapsed)
        elif exc_type is KeyboardInterrupt:
            _emit_progress("CANCELLED", self.name, elapsed=elapsed)
        else:
            _emit_progress(
                "FAILED",
                self.name,
                elapsed=elapsed,
                error=f"{getattr(exc_type, '__name__', exc_type)}: {exc}",
            )
        return False


class _ProgressSequence(Sequence[Any]):
    """Non-mutating sequence wrapper that reports iteration progress.

    The wrapped items and their order are unchanged. This is used at existing
    adapter boundaries so progress can be shown without reimplementing corpus
    extraction, hashing, deduplication, or segmentation logic.
    """

    def __init__(
        self,
        items: Sequence[Any],
        *,
        operation: str,
        every: int,
        describe: Optional[Callable[[Any], str]] = None,
        measure: Optional[Callable[[Any], int]] = None,
        large_threshold: Optional[int] = None,
        measure_name: str = "size",
        measure_format: Optional[Callable[[int], str]] = None,
    ) -> None:
        self._items = items
        self.operation = operation
        self.every = max(0, int(every))
        self.describe = describe or (lambda item: str(item))
        self.measure = measure
        self.large_threshold = large_threshold
        self.measure_name = measure_name
        self.measure_format = measure_format or (lambda value: str(value))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: Any) -> Any:
        return self._items[index]

    def __iter__(self) -> Iterator[Any]:
        total = len(self._items)
        started = time.perf_counter()
        for index, item in enumerate(self._items, 1):
            item_started = time.perf_counter()
            measurement: Optional[int] = None
            if self.measure is not None:
                try:
                    measurement = int(self.measure(item))
                except Exception:
                    measurement = None

            if (
                measurement is not None
                and self.large_threshold is not None
                and measurement >= self.large_threshold
            ):
                _emit_progress(
                    "LARGE",
                    f"{self.operation}: starting large item",
                    item=f"{index}/{total}",
                    source=self.describe(item),
                    **{self.measure_name: self.measure_format(measurement)},
                )

            yield item

            item_elapsed = time.perf_counter() - item_started
            if self.every and (index % self.every == 0 or index == total):
                _emit_progress(
                    "PROGRESS",
                    self.operation,
                    completed=f"{index}/{total}",
                    percent=f"{(index / max(1, total)) * 100:.1f}%",
                    elapsed=_format_duration(time.perf_counter() - started),
                    last=self.describe(item),
                    last_time=_format_duration(item_elapsed),
                )
            elif item_elapsed >= 60.0:
                _emit_progress(
                    "SLOW",
                    f"{self.operation}: slow item completed",
                    item=f"{index}/{total}",
                    source=self.describe(item),
                    item_time=_format_duration(item_elapsed),
                )


def _safe_file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _safe_document_chars(document: Any) -> int:
    try:
        return len(str(getattr(document, "text", "") or ""))
    except Exception:
        return 0


def _describe_document(document: Any) -> str:
    return str(
        getattr(document, "source_path", None)
        or getattr(document, "document_id", None)
        or "<document>"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> CurriculumConfig:
    try:
        raw = _CONFIG.load(path)
    except FileNotFoundError as exc:
        raise CurriculumError(f"Curriculum configuration file does not exist: {path}") from exc
    section = raw.get("lantra_curriculum", raw)
    if not isinstance(section, Mapping):
        raise CurriculumError("lantra_curriculum configuration must be a YAML mapping.")
    return CurriculumConfig.from_mapping(section)


def _runtime_file_inventory(config: CurriculumConfig) -> List[Dict[str, str]]:
    """Fingerprint enrichment code/config/resources that can change generated data."""

    roots = [
        Path("build_lantra_curriculum.py"),
        Path("src/training"),
        Path("src/agents/knowledge_agent.py"),
        Path("src/agents/reasoning_agent.py"),
        Path("src/agents/knowledge"),
        Path("src/agents/reasoning"),
        Path("src/agents/base/configs/agents_config.yaml"),
    ]
    if config.enable_perception:
        roots.extend([
            Path("src/agents/perception_agent.py"),
            Path("src/agents/perception"),
        ])

    allowed = {".py", ".yaml", ".yml", ".json", ".db", ".ttl"}
    files: Dict[str, Path] = {}
    for root in roots:
        if root.is_file():
            files[str(root.resolve())] = root
        elif root.is_dir():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in allowed:
                    files[str(path.resolve())] = path

    inventory: List[Dict[str, str]] = []
    for key, path in sorted(files.items()):
        try:
            digest = _sha256_file(path)
        except OSError:
            continue
        inventory.append({"path": str(path), "sha256": digest})
    return inventory


def _external_runtime_inventory(knowledge_agent: Any) -> List[Dict[str, str]]:
    """Include the actual configured ontology DB when it lives outside package roots."""

    paths: List[Path] = []
    ontology = getattr(knowledge_agent, "ontology_manager", None)
    db_path = getattr(ontology, "db_path", None)
    if db_path:
        paths.append(Path(str(db_path)))
    output: List[Dict[str, str]] = []
    for path in paths:
        try:
            if path.is_file():
                output.append({"path": str(path.resolve()), "sha256": _sha256_file(path)})
        except OSError:
            continue
    return output


def _agent_metadata(agent: Any) -> Dict[str, Any]:
    module_name = type(agent).__module__
    version = None
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", None)
    except Exception:
        pass
    return {
        "class": f"{module_name}.{type(agent).__name__}",
        "version": version,
        "name": getattr(agent, "name", None),
    }


def _build_fingerprint(
    *,
    source_fingerprint: str,
    config: CurriculumConfig,
    runtime_inventory: Sequence[Mapping[str, str]],
    agent_runtime: Optional[Mapping[str, Any]] = None,
) -> str:
    return sha256_payload(
        {
            "source_fingerprint": source_fingerprint,
            "config": config.to_dict(),
            "runtime_files": list(runtime_inventory),
            "agent_runtime": dict(agent_runtime or {}),
        }
    )


def _existing_build_is_reusable(output_dir: Path, build_fingerprint: str) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(manifest, Mapping):
        return False
    if manifest.get("schema") != MANIFEST_SCHEMA:
        return False
    if manifest.get("build_fingerprint") != build_fingerprint:
        return False
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, Sequence):
        return False
    for item in artifacts:
        if not isinstance(item, Mapping):
            return False
        path = output_dir / str(item.get("file", ""))
        try:
            if not path.is_file() or _sha256_file(path) != str(item.get("sha256", "")):
                return False
        except OSError:
            return False
    return True


def _load_existing_result(output_dir: Path) -> CurriculumBuildResult:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return CurriculumBuildResult(
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
        manifest=manifest,
        reused=True,
    )


_PREPARED_CACHE_SCHEMA = "slai.lantra.prepared-corpus.v1"


def _prepared_corpus_fingerprint(
    source_fingerprint: str,
    config: CurriculumConfig,
) -> str:
    """Fingerprint only the inputs that affect extraction, splitting, and segmentation."""
    return sha256_payload(
        {
            "source_fingerprint": source_fingerprint,
            "seed": config.seed,
            "validation_fraction": config.validation_fraction,
            "test_fraction": config.test_fraction,
            "min_document_chars": config.min_document_chars,
            "segment_min_chars": config.segment_min_chars,
            "segment_chunk_chars": config.segment_chunk_chars,
            "max_documents": config.max_documents,
            "max_segments": config.max_segments,
            "max_segments_per_document": config.max_segments_per_document,
        }
    )


def _prepared_cache_dir(config: CurriculumConfig, cache_fingerprint: str) -> Path:
    output_dir = Path(config.output_dir)
    return output_dir.parent / ".lantra_prepared_cache" / cache_fingerprint


def _source_document_payload(document: SourceDocument) -> Dict[str, Any]:
    return {
        "document_id": document.document_id,
        "source_path": document.source_path,
        "source_type": document.source_type,
        "source_sha256": document.source_sha256,
        "normalized_text_sha256": document.normalized_text_sha256,
        "text": document.text,
        "title": document.title,
        "extractor": document.extractor,
        "logical_index": document.logical_index,
        "metadata": dict(document.metadata),
        "split": document.split,
    }


def _source_segment_payload(segment: SourceSegment) -> Dict[str, Any]:
    return {
        "segment_id": segment.segment_id,
        "document_id": segment.document_id,
        "split": segment.split,
        "segment_index": segment.segment_index,
        "text": segment.text,
        "normalized_text_sha256": segment.normalized_text_sha256,
        "title": segment.title,
        "source_path": segment.source_path,
    }


def _source_document_from_payload(value: Mapping[str, Any]) -> SourceDocument:
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise CurriculumError("Prepared corpus document metadata must be a mapping.")
    split_value = value.get("split")
    return SourceDocument(
        document_id=str(value["document_id"]),
        source_path=str(value["source_path"]),
        source_type=str(value["source_type"]),
        source_sha256=str(value["source_sha256"]),
        normalized_text_sha256=str(value["normalized_text_sha256"]),
        text=str(value["text"]),
        title=(None if value.get("title") is None else str(value.get("title"))),
        extractor=str(value["extractor"]),
        logical_index=int(value["logical_index"]),
        metadata=dict(metadata),
        split=(None if split_value is None else str(split_value)),
    )


def _source_segment_from_payload(value: Mapping[str, Any]) -> SourceSegment:
    return SourceSegment(
        segment_id=str(value["segment_id"]),
        document_id=str(value["document_id"]),
        split=str(value["split"]),
        segment_index=int(value["segment_index"]),
        text=str(value["text"]),
        normalized_text_sha256=str(value["normalized_text_sha256"]),
        title=(None if value.get("title") is None else str(value.get("title"))),
        source_path=(
            None if value.get("source_path") is None else str(value.get("source_path"))
        ),
    )


def _write_gzip_jsonl(path: Path, items: Iterable[Mapping[str, Any]]) -> None:
    """Write deterministic gzip-compressed JSONL and durably flush it to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the actual filesystem descriptor writable until after the gzip
    # stream has been finalized. This is required for reliable fsync()
    # behaviour on Windows.
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as compressed_handle:
            for item in items:
                line = (json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                compressed_handle.write(line.encode("utf-8"))

        # GzipFile has now emitted its footer, while raw_handle is still writable and valid.
        raw_handle.flush()
        os.fsync(raw_handle.fileno())


def _save_prepared_corpus(
    cache_dir: Path,
    *,
    cache_fingerprint: str,
    source_fingerprint: str,
    documents: Sequence[SourceDocument],
    segments: Sequence[SourceSegment],
) -> None:
    """Persist extraction/split/segmentation output for restart-safe reuse."""
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = cache_dir.parent / (
        f".{cache_dir.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=False)

    try:
        documents_path = staging / "documents.jsonl.gz"
        segments_path = staging / "segments.jsonl.gz"
        _write_gzip_jsonl(
            documents_path,
            (_source_document_payload(document) for document in documents),
        )
        _write_gzip_jsonl(
            segments_path,
            (_source_segment_payload(segment) for segment in segments),
        )
        manifest = {
            "schema": _PREPARED_CACHE_SCHEMA,
            "cache_fingerprint": cache_fingerprint,
            "source_fingerprint": source_fingerprint,
            "documents": len(documents),
            "segments": len(segments),
            "documents_sha256": _sha256_file(documents_path),
            "segments_sha256": _sha256_file(segments_path),
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        os.replace(staging, cache_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_prepared_corpus(
    cache_dir: Path,
    *,
    cache_fingerprint: str,
    source_fingerprint: str,
) -> Optional[Tuple[List[SourceDocument], List[SourceSegment]]]:
    manifest_path = cache_dir / "manifest.json"
    documents_path = cache_dir / "documents.jsonl.gz"
    segments_path = cache_dir / "segments.jsonl.gz"
    if not (manifest_path.is_file() and documents_path.is_file() and segments_path.is_file()):
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            return None
        if manifest.get("schema") != _PREPARED_CACHE_SCHEMA:
            return None
        if manifest.get("cache_fingerprint") != cache_fingerprint:
            return None
        if manifest.get("source_fingerprint") != source_fingerprint:
            return None
        if _sha256_file(documents_path) != str(manifest.get("documents_sha256", "")):
            return None
        if _sha256_file(segments_path) != str(manifest.get("segments_sha256", "")):
            return None

        documents: List[SourceDocument] = []
        with gzip.open(documents_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise CurriculumError(
                        f"Prepared document cache line {line_number} is not a mapping."
                    )
                documents.append(_source_document_from_payload(raw))

        segments: List[SourceSegment] = []
        with gzip.open(segments_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise CurriculumError(
                        f"Prepared segment cache line {line_number} is not a mapping."
                    )
                segments.append(_source_segment_from_payload(raw))

        if len(documents) != int(manifest.get("documents", -1)):
            return None
        if len(segments) != int(manifest.get("segments", -1)):
            return None
        return documents, segments
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, CurriculumError):
        return None


def _apply_cli_overrides(config: CurriculumConfig, args: argparse.Namespace) -> CurriculumConfig:
    overrides: Dict[str, Any] = {}
    if args.source:
        overrides["source_paths"] = tuple(args.source)
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    if args.max_documents is not None:
        overrides["max_documents"] = int(args.max_documents)
    if args.max_segments is not None:
        overrides["max_segments"] = int(args.max_segments)
    if args.max_retrieval_pairs is not None:
        overrides["max_retrieval_pairs"] = int(args.max_retrieval_pairs)
    if args.no_reasoning:
        overrides["enable_reasoning"] = False
    if args.enable_perception:
        overrides["enable_perception"] = True
    if args.perception_checkpoint_version:
        overrides["perception_checkpoint_version"] = args.perception_checkpoint_version
    if args.force:
        overrides["reuse_if_unchanged"] = False
    return config.with_overrides(**overrides)


def _create_agents(config: CurriculumConfig) -> Tuple[Any, Any, Any, Optional[Any], Optional[Any]]:
    from src.agents.agent_factory import AgentFactory
    from src.agents.collaborative.shared_memory import SharedMemory

    if not config.enable_knowledge:
        raise CurriculumError(
            "This curriculum version requires KnowledgeAgent because phases 2A/2B and the "
            "grounding input for phase 2C are knowledge-backed."
        )

    memory = SharedMemory()
    factory = AgentFactory()
    knowledge_agent = factory.create("knowledge", shared_memory=memory)
    reasoning_agent = (
        factory.create("reasoning", shared_memory=memory)
        if config.enable_reasoning
        else None
    )
    perception_agent = (
        factory.create("perception", shared_memory=memory)
        if config.enable_perception
        else None
    )
    return memory, factory, knowledge_agent, reasoning_agent, perception_agent


def _shutdown_runtime(memory: Any, factory: Any) -> None:
    if factory is not None:
        release = getattr(factory, "release", None)
        active = getattr(factory, "get_active_agent_types", None)
        if callable(release) and callable(active):
            try:
                names = list(cast(Iterable[str], active()))
            except Exception:
                names = []
            for name in reversed(names):
                try:
                    release(name)
                except Exception as exc:
                    LOGGER.debug("AgentFactory release failed for %s: %s", name, exc)
        for method_name in ("close", "shutdown", "stop"):
            method = getattr(factory, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
                break
    if memory is not None:
        close = getattr(memory, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def build(
    config: CurriculumConfig,
    *,
    dry_run: bool = False,
    progress_interval: float = DEFAULT_PROGRESS_INTERVAL,
    progress_every_files: int = DEFAULT_PROGRESS_EVERY_FILES,
    resume: bool = True,
    checkpoint_every: int = 25,
) -> CurriculumBuildResult | Dict[str, Any]:
    """Build LANTRA curriculum artifacts with explicit long-running progress.

    The progress instrumentation is observational only: it does not change
    source order, extraction, deduplication, split assignment, segmentation,
    agent behavior, curriculum acceptance policy, or output schemas.
    """

    progress_interval = max(0.0, float(progress_interval))
    progress_every_files = max(0, int(progress_every_files))

    # These values are populated by the pipeline stages below.  Initialize
    # them here so they remain definitely assigned on every control-flow path.
    source_adapter = CanonicalLantraSourceAdapter(config)
    files = []
    source_inventory = []
    source_fingerprint = ""
    runtime_inventory = {}
    extracted_documents = []
    documents = []
    segments = []

    _emit_progress(
        "INFO",
        "Curriculum pipeline starting",
        pid=os.getpid(),
        dry_run=dry_run,
        output_dir=config.output_dir,
        reasoning=config.enable_reasoning,
        perception=config.enable_perception,
        max_documents=("unlimited" if config.max_documents <= 0 else config.max_documents),
        max_segments=("unlimited" if config.max_segments <= 0 else config.max_segments),
        max_retrieval_pairs=(
            "unlimited"
            if config.max_retrieval_pairs <= 0
            else config.max_retrieval_pairs
        ),
    )

    with _ProgressStage(
        "Discover corpus sources",
        heartbeat_seconds=progress_interval,
    ):
        files = source_adapter.discover_files()

    if not files:
        raise CurriculumError(
            "No LANTRA raw/document corpus files were found. Check data/library or --source."
        )

    total_source_bytes = sum(_safe_file_size(path) for path in files)
    large_files = [
        path for path in files if _safe_file_size(path) >= LARGE_SOURCE_BYTES
    ]
    _emit_progress(
        "INFO",
        "Corpus sources discovered",
        files=f"{len(files):,}",
        source_size=_human_bytes(total_source_bytes),
        large_sources=len(large_files),
    )
    for path in sorted(large_files, key=_safe_file_size, reverse=True)[:10]:
        _emit_progress(
            "INFO",
            "Large corpus source",
            source=str(path),
            size=_human_bytes(_safe_file_size(path)),
        )

    inventory_files = _ProgressSequence(
        files,
        operation="Hash source files",
        every=progress_every_files,
        describe=lambda item: str(item),
        measure=_safe_file_size,
        large_threshold=LARGE_SOURCE_BYTES,
        measure_name="size",
        measure_format=_human_bytes,
    )
    with _ProgressStage(
        "Build source inventory",
        heartbeat_seconds=progress_interval,
        detail=f"hashing {len(files):,} source files",
    ):
        source_inventory = source_adapter.inventory_files(inventory_files)
        source_fingerprint = source_adapter.source_fingerprint(source_inventory)

    _emit_progress(
        "INFO",
        "Source inventory ready",
        files=f"{len(source_inventory):,}",
        fingerprint=source_fingerprint[:16],
    )

    with _ProgressStage(
        "Fingerprint curriculum runtime",
        heartbeat_seconds=progress_interval,
    ):
        runtime_inventory = _runtime_file_inventory(config)

    _emit_progress(
        "INFO",
        "Runtime inventory ready",
        files=f"{len(runtime_inventory):,}",
    )

    extraction_files = _ProgressSequence(
        files,
        operation="Extract source documents",
        every=progress_every_files,
        describe=lambda item: str(item),
        measure=_safe_file_size,
        large_threshold=LARGE_SOURCE_BYTES,
        measure_name="size",
        measure_format=_human_bytes,
    )

    if dry_run:
        with _ProgressStage(
            "Extract canonical documents",
            heartbeat_seconds=progress_interval,
            detail=f"{len(files):,} source files",
        ):
            extracted_documents = source_adapter.extract_documents(
                extraction_files,
                inventory=source_inventory,
            )

        _emit_progress(
            "INFO",
            "Canonical extraction complete",
            documents=f"{len(extracted_documents):,}",
        )

        with _ProgressStage(
            "Assign document splits",
            heartbeat_seconds=progress_interval,
        ):
            documents = source_adapter.assign_document_splits(extracted_documents)

        document_split_counts: Dict[str, int] = {}
        for document in documents:
            split = str(getattr(document, "split", "unknown"))
            document_split_counts[split] = document_split_counts.get(split, 0) + 1
        _emit_progress(
            "INFO",
            "Document splits assigned",
            train=document_split_counts.get("train", 0),
            validation=document_split_counts.get("validation", 0),
            test=document_split_counts.get("test", 0),
        )

        segmentation_documents = _ProgressSequence(
            documents,
            operation="Segment source documents",
            every=progress_every_files,
            describe=_describe_document,
            measure=_safe_document_chars,
            large_threshold=LARGE_DOCUMENT_CHARS,
            measure_name="characters",
            measure_format=lambda value: f"{value:,}",
        )
        with _ProgressStage(
            "Segment canonical documents",
            heartbeat_seconds=progress_interval,
            detail=(
                f"documents={len(documents):,}, "
                f"chunk_chars={config.segment_chunk_chars:,}, "
                f"max_segments={config.max_segments if config.max_segments > 0 else 'unlimited'}"
            ),
        ):
            segments = source_adapter.segment_documents(segmentation_documents)

        split_counts = source_adapter.segment_counts_by_split(segments)
        _emit_progress(
            "INFO",
            "Segmentation complete",
            segments=f"{len(segments):,}",
            train=split_counts.get("train", 0),
            validation=split_counts.get("validation", 0),
            test=split_counts.get("test", 0),
        )
        return {
            "source_files": len(files),
            "documents": len(documents),
            "segments": len(segments),
            "segments_by_split": split_counts,
            "source_fingerprint": source_fingerprint,
            "output_dir": config.output_dir,
        }

    memory = factory = knowledge_agent = reasoning_agent = perception_agent = None
    knowledge = reasoning = perception = None
    runtime_metadata: Dict[str, Any] = {}
    result = None
    perception_state: Optional[Dict[str, Any]] = None
    final_fingerprint = ""
    reusable = False
    try:
        with _ProgressStage(
            "Initialize SLAI agent runtime",
            heartbeat_seconds=progress_interval,
            detail=(
                "KnowledgeAgent"
                + (" + ReasoningAgent" if config.enable_reasoning else "")
                + (" + PerceptionAgent" if config.enable_perception else "")
            ),
        ):
            (
                memory,
                factory,
                knowledge_agent,
                reasoning_agent,
                perception_agent,
            ) = _create_agents(config)

        _emit_progress(
            "INFO",
            "Agent runtime initialized",
            knowledge=type(knowledge_agent).__name__,
            reasoning=(
                type(reasoning_agent).__name__
                if reasoning_agent is not None
                else "disabled"
            ),
            perception=(
                type(perception_agent).__name__
                if perception_agent is not None
                else "disabled"
            ),
        )

        with _ProgressStage("Prepare curriculum agent adapters", heartbeat_seconds=progress_interval):
            knowledge = KnowledgeAdapter(knowledge_agent, config)
            reasoning = (
                ReasoningAdapter(reasoning_agent, config)
                if reasoning_agent is not None
                else None
            )
            perception = None
            if perception_agent is not None:
                perception = PerceptionAdapter(perception_agent, config)
                perception_state = perception.prepare()

        with _ProgressStage("Finalize build fingerprint", heartbeat_seconds=progress_interval):
            runtime_inventory = list(runtime_inventory) + _external_runtime_inventory(
                knowledge_agent
            )
            runtime_metadata: Dict[str, Any] = {
                "files": runtime_inventory,
                "knowledge_agent": _agent_metadata(knowledge_agent),
                "reasoning_agent": (
                    _agent_metadata(reasoning_agent)
                    if reasoning_agent is not None
                    else None
                ),
                "perception_agent": (
                    _agent_metadata(perception_agent)
                    if perception_agent is not None
                    else None
                ),
                "perception_checkpoint": perception_state,
            }
            final_fingerprint = _build_fingerprint(
                source_fingerprint=source_fingerprint,
                config=config,
                runtime_inventory=runtime_inventory,
                agent_runtime={
                    "knowledge": runtime_metadata["knowledge_agent"],
                    "reasoning": runtime_metadata["reasoning_agent"],
                    "perception": runtime_metadata["perception_agent"],
                    "perception_checkpoint": perception_state,
                },
            )

        _emit_progress(
            "INFO",
            "Build fingerprint ready",
            fingerprint=final_fingerprint[:16],
            runtime_files=f"{len(runtime_inventory):,}",
        )

        output_dir = Path(config.output_dir)
        if config.reuse_if_unchanged:
            with _ProgressStage(
                "Validate reusable curriculum cache",
                heartbeat_seconds=progress_interval,
                detail=str(output_dir),
            ):
                reusable = _existing_build_is_reusable(output_dir, final_fingerprint)
            if reusable:
                _emit_progress(
                    "CACHE",
                    "Reusable curriculum build found; source extraction skipped",
                    output_dir=output_dir,
                )
                return _load_existing_result(output_dir)
            _emit_progress("CACHE", "No reusable curriculum build matched", output_dir=output_dir)
        else:
            _emit_progress("CACHE", "Reusable-build check disabled", reason="reuse_if_unchanged=false")

        # Extraction/split/segmentation is deterministic and expensive. Persist
        # this prepared corpus independently from phase-level curriculum state so
        # a later Phase 2A/2B/2C failure does not require re-reading 1,000+ files.
        prepared_fingerprint = _prepared_corpus_fingerprint(
            source_fingerprint,
            config,
        )
        prepared_cache_dir = _prepared_cache_dir(
            config,
            prepared_fingerprint,
        )
        prepared = (
            _load_prepared_corpus(
                prepared_cache_dir,
                cache_fingerprint=prepared_fingerprint,
                source_fingerprint=source_fingerprint,
            )
            if resume
            else None
        )

        if prepared is not None:
            documents, segments = prepared
            split_counts = source_adapter.segment_counts_by_split(segments)
            _emit_progress(
                "CACHE",
                "Prepared corpus restored",
                cache=prepared_cache_dir,
                documents=f"{len(documents):,}",
                segments=f"{len(segments):,}",
                train=split_counts.get("train", 0),
                validation=split_counts.get("validation", 0),
                test=split_counts.get("test", 0),
            )
        else:
            with _ProgressStage(
                "Extract canonical documents",
                heartbeat_seconds=progress_interval,
                detail=f"{len(files):,} source files",
            ):
                extracted_documents = source_adapter.extract_documents(
                    extraction_files,
                    inventory=source_inventory,
                )

            _emit_progress(
                "INFO",
                "Canonical extraction complete",
                documents=f"{len(extracted_documents):,}",
            )

            with _ProgressStage(
                "Assign document splits",
                heartbeat_seconds=progress_interval,
            ):
                documents = source_adapter.assign_document_splits(extracted_documents)

            document_split_counts: Dict[str, int] = {}
            for document in documents:
                split = str(getattr(document, "split", "unknown"))
                document_split_counts[split] = document_split_counts.get(split, 0) + 1
            _emit_progress(
                "INFO",
                "Document splits assigned",
                train=document_split_counts.get("train", 0),
                validation=document_split_counts.get("validation", 0),
                test=document_split_counts.get("test", 0),
            )

            segmentation_documents = _ProgressSequence(
                documents,
                operation="Segment source documents",
                every=progress_every_files,
                describe=_describe_document,
                measure=_safe_document_chars,
                large_threshold=LARGE_DOCUMENT_CHARS,
                measure_name="characters",
                measure_format=lambda value: f"{value:,}",
            )
            with _ProgressStage(
                "Segment canonical documents",
                heartbeat_seconds=progress_interval,
                detail=(
                    f"documents={len(documents):,}, "
                    f"chunk_chars={config.segment_chunk_chars:,}, "
                    f"max_segments={config.max_segments if config.max_segments > 0 else 'unlimited'}"
                ),
            ):
                segments = source_adapter.segment_documents(segmentation_documents)

            split_counts = source_adapter.segment_counts_by_split(segments)
            _emit_progress(
                "INFO",
                "Segmentation complete",
                segments=f"{len(segments):,}",
                train=split_counts.get("train", 0),
                validation=split_counts.get("validation", 0),
                test=split_counts.get("test", 0),
            )

            with _ProgressStage(
                "Persist prepared corpus cache",
                heartbeat_seconds=progress_interval,
                detail=str(prepared_cache_dir),
            ):
                _save_prepared_corpus(
                    prepared_cache_dir,
                    cache_fingerprint=prepared_fingerprint,
                    source_fingerprint=source_fingerprint,
                    documents=documents,
                    segments=segments,
                )
            _emit_progress(
                "CACHE",
                "Prepared corpus cache saved",
                cache=prepared_cache_dir,
                documents=f"{len(documents):,}",
                segments=f"{len(segments):,}",
            )

        assert knowledge is not None
        builder = LantraCurriculumBuilder(
            config,
            knowledge=knowledge,
            reasoning=reasoning,
            perception=perception,
            runtime_metadata=runtime_metadata,
            resume=resume,
            checkpoint_every=checkpoint_every,
        )
        with _ProgressStage(
            "Construct agent-enriched curriculum",
            heartbeat_seconds=progress_interval,
            detail=(
                f"{len(segments):,} segments; "
                "knowledge indexing -> ontology facts -> phase 2A -> "
                "phase 2B retrieval -> phase 2C reasoning -> artifacts"
            ),
        ):
            result = builder.build(
                documents,
                segments,
                source_inventory=source_inventory,
                source_fingerprint=source_fingerprint,
                build_fingerprint=final_fingerprint,
            )

        assert result is not None
        artifacts = result.manifest.get("artifacts", [])
        coverage = result.manifest.get("coverage", {})
        _emit_progress(
            "DONE",
            "Curriculum build complete",
            output_dir=result.output_dir,
            artifacts=len(artifacts) if isinstance(artifacts, Sequence) else "unknown",
            active_tasks=(
                ",".join(coverage.get("active_tasks", []))
                if isinstance(coverage, Mapping)
                else "unknown"
            ),
        )
        return result
    finally:
        if factory is not None or memory is not None:
            started = time.perf_counter()
            _emit_progress("START", "Shutdown SLAI agent runtime")
            _shutdown_runtime(memory, factory)
            _emit_progress(
                "DONE",
                "Shutdown SLAI agent runtime",
                elapsed=_format_duration(time.perf_counter() - started),
            )

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build source-grounded LANTRA curriculum artifacts using SLAI Knowledge, "
            "Reasoning, and optional frozen Perception agents."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Curriculum YAML path.")
    parser.add_argument("--source", action="append", default=[], help="Additional/override raw corpus path; repeatable.")
    parser.add_argument("--output-dir", default=None, help="Override curriculum output directory.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--max-retrieval-pairs", type=int, default=None)
    parser.add_argument("--no-reasoning", action="store_true", help="Disable phase 2C reasoning validation/inference.")
    parser.add_argument("--enable-perception", action="store_true", help="Enable frozen Perception semantic hardness filtering.")
    parser.add_argument(
        "--perception-checkpoint-version",
        default=None,
        help="SLAI PerceptionAgent checkpoint version passed to restore_checkpoint().",
    )
    parser.add_argument("--force", action="store_true", help="Ignore matching cached curriculum artifacts and rebuild.")
    parser.add_argument("--dry-run", action="store_true", help="Extract/split/segment sources and print counts without creating agents or writing output.")
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=DEFAULT_PROGRESS_INTERVAL,
        help=(
            "Seconds between liveness heartbeats during long stages; "
            "0 disables heartbeats (default: 30)."
        ),
    )
    parser.add_argument(
        "--progress-every-files",
        type=int,
        default=DEFAULT_PROGRESS_EVERY_FILES,
        help=(
            "Emit file/document progress after every N items; "
            "0 disables item-count progress (default: 10)."
        ),
    )
    parser.add_argument("--no-resume", action="store_true", help=(
        "Ignore resumable curriculum work state "
        "and rebuild intermediate phases."
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=25, help=(
            "Persist resumable curriculum state "
            "after every N processed phase items "
            "(default: 25)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.progress_interval < 0:
            raise CurriculumError("--progress-interval must be >= 0.")
        if args.progress_every_files < 0:
            raise CurriculumError("--progress-every-files must be >= 0.")
        if args.checkpoint_every <= 0:
            raise CurriculumError("--checkpoint-every must be > 0.")

        _emit_progress("INFO", "Loading curriculum configuration", config=args.config)
        config = _apply_cli_overrides(_load_config(Path(args.config)), args)
        result = build(config, dry_run=bool(args.dry_run),
            progress_interval=float(args.progress_interval),
            progress_every_files=int(args.progress_every_files),
            resume=not bool(args.no_resume or args.force),
            checkpoint_every=int(args.checkpoint_every),
        )
        if isinstance(result, CurriculumBuildResult):
            summary = {
                "status": "reused" if result.reused else "built",
                "output_dir": result.output_dir,
                "manifest": result.manifest_path,
                "records": result.manifest.get("records", {}),
                "coverage": result.manifest.get("coverage", {}),
            }
        else:
            summary = {"status": "dry_run", **result}
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except CurriculumError as exc:
        LOGGER.error("LANTRA curriculum build failed: %s", exc)
        _emit_progress("ERROR", "LANTRA curriculum build failed", error=str(exc))
        return 2
    except KeyboardInterrupt:
        _emit_progress("CANCELLED", "LANTRA curriculum build interrupted by user")
        return 130
    except Exception as exc:
        LOGGER.exception("Unexpected LANTRA curriculum failure")
        _emit_progress(
            "ERROR",
            "Unexpected LANTRA curriculum failure",
            error=f"{type(exc).__name__}: {exc}",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
