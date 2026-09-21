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
import hashlib
import importlib
import json
import sys

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.training.curriculum_builder import LantraCurriculumBuilder
from src.training.enrichment_contracts import (
    CurriculumBuildResult,
    CurriculumConfig,
    CurriculumError,
    MANIFEST_SCHEMA,
    sha256_payload,
)
from src.training.knowledge_adapter import KnowledgeAdapter
from src.training.perception_adapter import PerceptionAdapter
from src.training.reasoning_adapter import ReasoningAdapter
from src.training.source_adapter import CanonicalLantraSourceAdapter
from src.utils.configuration import bind_config
from logs.logger import get_logger, PrettyPrinter


LOGGER = get_logger("LANTRA Curriculum Builder")
printer = PrettyPrinter()

DEFAULT_CONFIG = Path("src/training/configs/lantra_curriculum.yaml")
_CONFIG = bind_config(DEFAULT_CONFIG)


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
                names = list(active())
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


def build(config: CurriculumConfig, *, dry_run: bool = False) -> CurriculumBuildResult | Dict[str, Any]:
    source_adapter = CanonicalLantraSourceAdapter(config)
    files = source_adapter.discover_files()
    if not files:
        raise CurriculumError(
            "No LANTRA raw/document corpus files were found. Check data/library or --source."
        )
    source_inventory = source_adapter.inventory_files(files)
    source_fingerprint = source_adapter.source_fingerprint(source_inventory)
    runtime_inventory = _runtime_file_inventory(config)

    if dry_run:
        documents = source_adapter.assign_document_splits(
            source_adapter.extract_documents(files, inventory=source_inventory)
        )
        segments = source_adapter.segment_documents(documents)
        return {
            "source_files": len(files),
            "documents": len(documents),
            "segments": len(segments),
            "segments_by_split": source_adapter.segment_counts_by_split(segments),
            "source_fingerprint": source_fingerprint,
            "output_dir": config.output_dir,
        }

    memory = factory = knowledge_agent = reasoning_agent = perception_agent = None
    try:
        memory, factory, knowledge_agent, reasoning_agent, perception_agent = _create_agents(config)
        knowledge = KnowledgeAdapter(knowledge_agent, config)
        reasoning = ReasoningAdapter(reasoning_agent, config) if reasoning_agent is not None else None
        perception = None
        perception_state: Optional[Dict[str, Any]] = None
        if perception_agent is not None:
            perception = PerceptionAdapter(perception_agent, config)
            perception_state = perception.prepare()

        runtime_inventory = list(runtime_inventory) + _external_runtime_inventory(knowledge_agent)
        runtime_metadata: Dict[str, Any] = {
            "files": runtime_inventory,
            "knowledge_agent": _agent_metadata(knowledge_agent),
            "reasoning_agent": _agent_metadata(reasoning_agent) if reasoning_agent is not None else None,
            "perception_agent": _agent_metadata(perception_agent) if perception_agent is not None else None,
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
        output_dir = Path(config.output_dir)
        if config.reuse_if_unchanged and _existing_build_is_reusable(output_dir, final_fingerprint):
            return _load_existing_result(output_dir)

        # Parse/chunk only after the cache decision. Agent construction is cheaper
        # than re-extracting a large PDF/EPUB library and also exposes external
        # ontology/checkpoint state needed for a correct cache fingerprint.
        documents = source_adapter.assign_document_splits(
            source_adapter.extract_documents(files, inventory=source_inventory)
        )
        segments = source_adapter.segment_documents(documents)

        builder = LantraCurriculumBuilder(
            config,
            knowledge=knowledge,
            reasoning=reasoning,
            perception=perception,
            runtime_metadata=runtime_metadata,
        )
        return builder.build(
            documents,
            segments,
            source_inventory=source_inventory,
            source_fingerprint=source_fingerprint,
            build_fingerprint=final_fingerprint,
        )
    finally:
        _shutdown_runtime(memory, factory)


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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        config = _apply_cli_overrides(_load_config(Path(args.config)), args)
        result = build(config, dry_run=bool(args.dry_run))
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
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        LOGGER.exception("Unexpected LANTRA curriculum failure")
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
