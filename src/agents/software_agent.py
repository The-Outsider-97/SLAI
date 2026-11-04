"""
SLAI Software Agent. Top-level orchestration facade for the Software engineering subsystem.

Implements:
- Aho, Lam, Sethi & Ullman (2006), Compilers: Principles, Techniques, and Tools, 2nd ed.
- Tarjan (1972), Depth-First Search and Linear Graph Algorithms.
- Grove, DeFouw, Dean & Chambers (1997), Call Graph Construction in Object-Oriented Languages.
- Barrett et al. (1996), A Monotonic Superclass Linearization for Dylan.
- McCabe (1976), A Complexity Measure.
- Campbell (2018), Cognitive Complexity: A New Way of Measuring Understandability.
- Martin (2002), Agile Software Development: Principles, Patterns, and Practices.
- Bohner & Arnold (1996), Software Change Impact Analysis.
- Fowler (2018), Refactoring: Improving the Design of Existing Code, 2nd ed.

SoftwareAgent understands, analyses and proposes transformations of software artifacts (syntax trees, symbols,
dependency and call graphs, class hierarchies, rule-based static analysis, dead code, change impact, architecture,
interface compatibility, refactorings, patches and API migrations). It never executes anything and never applies
a change: every patch is a proposal that must still pass verification, safety, execution and evaluation.
"""

from __future__ import annotations

__version__ = "2.4.0"

import uuid

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .base_agent import BaseAgent
from .base.utils.base_errors import *
from .base.utils.base_helpers import *
from .base.utils.main_config_loader import *
from .software import *
from .software.utils.software_helpers import opt_bool, opt_float, opt_int, opt_strings, require_text
from logs.logger import PrettyPrinter, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Software Agent")
printer = PrettyPrinter()


class SoftwareOperation(str, Enum):
    """Procedures exposed by the Software subsystem."""

    SUMMARY = "summary"
    OUTLINE = "outline"
    SYMBOL = "symbol"
    SEARCH = "search"
    REFERENCES = "references"
    DEPENDENCY_GRAPH = "dependency_graph"
    CALL_GRAPH = "call_graph"
    CLASS_HIERARCHY = "class_hierarchy"
    STATIC_ANALYSIS = "static_analysis"
    DEAD_CODE = "dead_code"
    CHANGE_IMPACT = "change_impact"
    COMPATIBILITY = "compatibility"
    ARCHITECTURE = "architecture"
    RENAME = "rename"
    REMOVE_UNUSED_IMPORTS = "remove_unused_imports"
    REMOVE_DEAD_CODE = "remove_dead_code"
    MIGRATE_API = "migrate_api"
    APPLY_DIFF = "apply_diff"
    PROPOSAL = "proposal"


_STATE_KEYS = ("files", "snapshot", "base_patch_id")


def _coerce_operation(raw: object) -> SoftwareOperation:
    if isinstance(raw, SoftwareOperation):
        return raw
    try:
        return SoftwareOperation(str(raw).strip().lower())
    except (TypeError, ValueError) as exc:
        raise UnsupportedSoftwareOperationError(
            "unsupported SoftwareAgent operation",
            context={"operation": str(raw), "supported": [item.value for item in SoftwareOperation]},
            cause=exc,
        ) from exc


def _mapping(value: object, field_name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SoftwareRequestError(f"{field_name} must be a mapping", context={"field": field_name, "type": type(value).__name__})
    return dict(value)


def _strings(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, Sequence) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise SoftwareRequestError(f"{field_name} must be a sequence of non-empty strings", context={"field": field_name})
    return tuple(dict.fromkeys(item.strip() for item in value))


@dataclass(frozen=True, slots=True)
class SoftwareSource:
    """One repository state: file contents, a remembered snapshot, or the result of an earlier patch.

    ``changes`` overlays path → new content (``None`` deletes) on top of the chosen state.
    """

    files: Mapping[str, object] | None = None
    snapshot: str | None = None
    base_patch_id: str | None = None
    changes: Mapping[str, object] | None = None
    source_roots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        given = [name for name in _STATE_KEYS if getattr(self, name) is not None]
        if len(given) != 1:
            raise SoftwareRequestError(
                "a repository state needs exactly one of files, snapshot or base_patch_id",
                context={"given": given},
            )
        for name in ("files", "changes"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _mapping(value, name))
        for name in ("snapshot", "base_patch_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_text(value, name, error_cls=SoftwareRequestError))
        object.__setattr__(self, "source_roots", _strings(self.source_roots, "source_roots"))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object], label: str = "source") -> "SoftwareSource":
        if not isinstance(payload, Mapping):
            raise SoftwareRequestError(f"{label} must be a mapping", context={"field": label})
        return cls(
            files=payload.get("files"),  # type: ignore[arg-type]
            snapshot=payload.get("snapshot"),  # type: ignore[arg-type]
            base_patch_id=payload.get("base_patch_id"),  # type: ignore[arg-type]
            changes=payload.get("changes"),  # type: ignore[arg-type]
            source_roots=payload.get("source_roots") or (),  # type: ignore[arg-type]
        )

    def describe(self) -> dict[str, object]:
        """Bounded description for shared-memory request records (never the file contents)."""

        return {
            "files": None if self.files is None else len(self.files),
            "snapshot": self.snapshot,
            "base_patch_id": self.base_patch_id,
            "changes": None if self.changes is None else len(self.changes),
            "source_roots": list(self.source_roots),
        }


@dataclass(frozen=True, slots=True)
class SoftwareAgentRequest:
    """Normalized agent-facing request.

    ``source`` is the repository state under analysis (top-level ``files`` | ``snapshot`` | ``base_patch_id``
    with optional ``changes``/``source_roots``); ``baseline`` is a second state of the same shape used by
    change-impact and compatibility checks. ``parameters`` are the operation's arguments and ``options`` are
    per-request analyzer overrides.
    """

    operation: SoftwareOperation
    source: SoftwareSource | None = None
    baseline: SoftwareSource | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)
    options: Mapping[str, object] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    metadata: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)
    prior_request_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _coerce_operation(self.operation))
        for name in ("source", "baseline"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, SoftwareSource):
                raise SoftwareRequestError(f"{name} must be a SoftwareSource or null", context={"field": name})

        request_id = str(self.request_id or "").strip()
        if not request_id:
            raise SoftwareRequestError("request_id must be a non-empty string")
        object.__setattr__(self, "request_id", request_id)

        object.__setattr__(self, "parameters", _mapping(self.parameters, "parameters"))
        object.__setattr__(self, "options", _mapping(self.options, "options"))

        metadata: dict[str, str | int | float | bool | None] = {}
        for raw_key, value in _mapping(self.metadata, "metadata").items():
            key = str(raw_key).strip()
            if not key:
                raise SoftwareRequestError("metadata keys must be non-empty strings")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise SoftwareRequestError("metadata values must be scalar", context={"key": key, "type": type(value).__name__})
            metadata[key] = value  # type: ignore[assignment]
        object.__setattr__(self, "metadata", metadata)

        prior = _strings(self.prior_request_ids, "prior_request_ids")
        object.__setattr__(self, "prior_request_ids", prior)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "SoftwareAgentRequest":
        if not isinstance(payload, Mapping):
            raise SoftwareRequestError("SoftwareAgent request must be a mapping")
        if payload.get("operation") is None:
            raise SoftwareRequestError("SoftwareAgent request requires operation")
        has_state = any(key in payload for key in (*_STATE_KEYS, "changes"))
        baseline = payload.get("baseline")
        return cls(
            operation=_coerce_operation(payload["operation"]),
            source=SoftwareSource.from_mapping(payload) if has_state else None,
            baseline=None if baseline is None else SoftwareSource.from_mapping(baseline, "baseline"),  # type: ignore[arg-type]
            parameters=_mapping(payload.get("parameters"), "parameters"),
            options=_mapping(payload.get("options"), "options"),
            request_id=str(payload.get("request_id") or uuid.uuid4().hex),
            metadata=_mapping(payload.get("metadata"), "metadata"),  # type: ignore[arg-type]
            prior_request_ids=_strings(payload.get("prior_request_ids"), "prior_request_ids"),
        )


@dataclass(frozen=True, slots=True)
class SoftwareAgentOutcome:
    """Execution envelope. Reaching it means the analysis completed; findings never imply failure."""

    request_id: str
    operation: SoftwareOperation
    result: Mapping[str, object]
    snapshot_digest: str | None = None
    proposal: PatchProposal | None = None
    prior_references: tuple[Mapping[str, object], ...] = ()
    elapsed_seconds: float = 0.0

    @property
    def execution_success(self) -> bool:
        return True

    @property
    def proposal_applied(self) -> bool:
        return False if self.proposal is None else bool(self.proposal.applied)

    @property
    def requires_gates(self) -> tuple[str, ...]:
        return () if self.proposal is None else tuple(self.proposal.requires_gates)

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_success": True,
            "request_id": self.request_id,
            "operation": self.operation.value,
            "snapshot_digest": self.snapshot_digest,
            "patch_id": None if self.proposal is None else self.proposal.patch_id,
            "proposal_applied": self.proposal_applied,
            "requires_gates": list(self.requires_gates),
            "result": dict(self.result),
            "prior_references": [dict(item) for item in self.prior_references],
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        }


@dataclass(frozen=True, slots=True)
class _Dispatch:
    payload: Mapping[str, object]
    snapshot_digest: str | None = None
    proposal: PatchProposal | None = None


class SoftwareAgent(BaseAgent):
    """Production orchestration facade over ``src.agents.software``.

    The facade owns no analysis logic and no memory of its own. Program facts are computed by the
    subsystem's handler chain; cross-agent state is published through BaseAgent's injected SharedMemory.
    """

    CHECKPOINTING_SUPPORTED = False

    def __init__(
        self,
        shared_memory: object,
        agent_factory: object,
        config: Mapping[str, object] | None = None,
        *,
        checkpoint_manager: object | None = None,
    ) -> None:
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            checkpoint_manager=checkpoint_manager,
        )

        self.agent_config: dict[str, object] = dict(get_config_section("software_agent") or {})
        if config is not None:
            if not isinstance(config, Mapping):
                raise BaseConfigurationError("SoftwareAgent config override must be a mapping", component=self.name)
            self.agent_config.update(dict(config))

        self._operation_handlers: dict[SoftwareOperation, Callable[[SoftwareAgentRequest], _Dispatch]] = {
            SoftwareOperation.SUMMARY: self._op_summary,
            SoftwareOperation.OUTLINE: self._op_outline,
            SoftwareOperation.SYMBOL: self._op_symbol,
            SoftwareOperation.SEARCH: self._op_search,
            SoftwareOperation.REFERENCES: self._op_references,
            SoftwareOperation.DEPENDENCY_GRAPH: self._op_dependency_graph,
            SoftwareOperation.CALL_GRAPH: self._op_call_graph,
            SoftwareOperation.CLASS_HIERARCHY: self._op_class_hierarchy,
            SoftwareOperation.STATIC_ANALYSIS: self._op_static_analysis,
            SoftwareOperation.DEAD_CODE: self._op_dead_code,
            SoftwareOperation.CHANGE_IMPACT: self._op_change_impact,
            SoftwareOperation.COMPATIBILITY: self._op_compatibility,
            SoftwareOperation.ARCHITECTURE: self._op_architecture,
            SoftwareOperation.RENAME: self._op_rename,
            SoftwareOperation.REMOVE_UNUSED_IMPORTS: self._op_remove_unused_imports,
            SoftwareOperation.REMOVE_DEAD_CODE: self._op_remove_dead_code,
            SoftwareOperation.MIGRATE_API: self._op_migrate_api,
            SoftwareOperation.APPLY_DIFF: self._op_apply_diff,
            SoftwareOperation.PROPOSAL: self._op_proposal,
        }
        self._load_agent_config()
        self._validate_agent_config()

        # The handler chain is built bottom-up; each layer only talks to the one below it.
        self._models = SoftwareModels()
        self._graphs = SoftwareGraphs(self._models)
        self._semantics = SoftwareSemantics(self._graphs)
        self._analysis = SoftwareAnalysis(self._semantics)
        self._refactoring = SoftwareRefactoring(self._analysis)

        logger.info(
            "Software Agent initialized | operations=%d | languages=%s | shared_memory=%s",
            len(self.allowed_operations),
            ",".join(self._models.languages()),
            self.publish_shared_memory,
        )
        self._publish_event("initialized", {"agent_id": self.agent_id})

    # ------------------------------------------------------------------
    # Agent-level configuration: agents_config.yaml only
    # ------------------------------------------------------------------

    def _cfg(self, key: str, default: object) -> object:
        return self.agent_config.get(key, default)

    def _load_agent_config(self) -> None:
        self.enabled = coerce_bool(self._cfg("enabled", True), True)
        self.publish_shared_memory = coerce_bool(self._cfg("publish_shared_memory", True), True)
        self.fail_on_shared_memory_error = coerce_bool(self._cfg("fail_on_shared_memory_error", False), False)
        self.publish_proposal_diffs = coerce_bool(self._cfg("publish_proposal_diffs", True), True)
        self.shared_memory_ttl_seconds = coerce_int(self._cfg("shared_memory_ttl_seconds", 604800), 604800, minimum=0, maximum=31_536_000)
        self.max_shared_diff_chars = coerce_int(self._cfg("max_shared_diff_chars", 65_536), 65_536, minimum=0, maximum=16_777_216)
        self.max_shared_list_items = coerce_int(self._cfg("max_shared_list_items", 128), 128, minimum=1, maximum=100_000)
        self.max_prior_references = coerce_int(self._cfg("max_prior_references", 32), 32, minimum=0, maximum=1_000)

        self.request_key_prefix = str(self._cfg("request_key_prefix", "software_agent.request") or "").strip()
        self.result_key_prefix = str(self._cfg("result_key_prefix", "software_agent.result") or "").strip()
        self.summary_key_prefix = str(self._cfg("summary_key_prefix", "software_agent.summary") or "").strip()
        self.proposal_key_prefix = str(self._cfg("proposal_key_prefix", "software_agent.proposal") or "").strip()
        self.error_key_prefix = str(self._cfg("error_key_prefix", "software_agent.error") or "").strip()
        self.event_key_prefix = str(self._cfg("event_key_prefix", "software_agent.event") or "").strip()
        self.latest_result_key = str(self._cfg("latest_result_key", "software_agent.latest") or "").strip()

        configured = self._cfg("allowed_operations", None)
        if configured is None:
            self._configured_operations: set[str] | None = None
        elif isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
            self._configured_operations = {str(item).strip().lower() for item in configured if str(item).strip()}
        else:
            raise BaseConfigurationError("software_agent.allowed_operations must be a sequence of operation names", component=self.name)

    def _validate_agent_config(self) -> None:
        for field_name in (
            "request_key_prefix",
            "result_key_prefix",
            "summary_key_prefix",
            "proposal_key_prefix",
            "error_key_prefix",
            "event_key_prefix",
            "latest_result_key",
        ):
            if not getattr(self, field_name):
                raise BaseConfigurationError(
                    f"software_agent.{field_name} must be a non-empty string",
                    component=self.name,
                    context={"field": field_name},
                )
        supported = {item.value for item in self._operation_handlers}
        if self._configured_operations is None:
            self.allowed_operations = frozenset(self._operation_handlers)
        else:
            unknown = self._configured_operations - supported
            if unknown:
                raise BaseConfigurationError(
                    "software_agent.allowed_operations contains unsupported operations",
                    component=self.name,
                    context={"unknown": sorted(unknown)},
                )
            self.allowed_operations = frozenset(SoftwareOperation(name) for name in self._configured_operations)
        if self.publish_shared_memory:
            missing = [name for name in ("get", "set") if not callable(getattr(self.shared_memory, name, None))]
            if missing:
                raise BaseConfigurationError(
                    "SharedMemory does not satisfy SoftwareAgent's get/set contract",
                    component=self.name,
                    context={"missing_methods": missing},
                )

    # ------------------------------------------------------------------
    # Public facade API
    # ------------------------------------------------------------------

    def analyze(self, request: SoftwareAgentRequest | Mapping[str, object]) -> SoftwareAgentOutcome:
        """Run one request through the Software subsystem; patches are proposed, never applied."""

        if not self.enabled:
            raise BaseConfigurationError("SoftwareAgent is disabled by configuration", component=self.name, operation="analyze")
        normalized = request if isinstance(request, SoftwareAgentRequest) else SoftwareAgentRequest.from_mapping(request)
        if normalized.operation not in self.allowed_operations:
            raise UnsupportedSoftwareOperationError(
                "operation is not enabled for SoftwareAgent",
                context={"operation": normalized.operation.value, "allowed": sorted(item.value for item in self.allowed_operations)},
            )
        prior_references = self._load_prior_references(normalized.prior_request_ids)

        logger.info("Software request received | id=%s | operation=%s", normalized.request_id, normalized.operation.value)
        self._shared_set(
            self._key(self.request_key_prefix, normalized.request_id),
            self._request_metadata(normalized, prior_references),
            tags=("software", "request", normalized.operation.value),
        )
        self._publish_event("request_received", {"request_id": normalized.request_id, "operation": normalized.operation.value})

        started = monotonic_seconds()
        try:
            dispatched = self._operation_handlers[normalized.operation](normalized)
            outcome = SoftwareAgentOutcome(
                request_id=normalized.request_id,
                operation=normalized.operation,
                result=dispatched.payload,
                snapshot_digest=dispatched.snapshot_digest,
                proposal=dispatched.proposal,
                prior_references=prior_references,
                elapsed_seconds=monotonic_seconds() - started,
            )
            self._publish_outcome(normalized, outcome)
            logger.info(
                "Software request completed | id=%s | operation=%s | patch=%s | elapsed=%.3fs",
                normalized.request_id,
                normalized.operation.value,
                outcome.proposal.patch_id if outcome.proposal else None,
                outcome.elapsed_seconds,
            )
            return outcome
        except SoftwareError as exc:
            self._publish_error(normalized, exc)
            logger.warning(
                "Software request failed operationally | id=%s | operation=%s | error=%s",
                normalized.request_id,
                normalized.operation.value,
                type(exc).__name__,
            )
            raise
        except Exception as exc:
            error = SoftwareError(
                "SoftwareAgent orchestration failed",
                operation=normalized.operation.value,
                context={"request_id": normalized.request_id},
                cause=exc,
            )
            self._publish_error(normalized, error)
            logger.exception("SoftwareAgent orchestration failure | id=%s | operation=%s", normalized.request_id, normalized.operation.value)
            raise error from exc

    def perform_task(self, task_data: object) -> dict[str, object]:
        """BaseAgent task entry point."""

        if isinstance(task_data, SoftwareAgentRequest):
            request = task_data
        elif isinstance(task_data, Mapping):
            request = SoftwareAgentRequest.from_mapping(task_data)
        else:
            raise SoftwareRequestError("SoftwareAgent task_data must be a request mapping or SoftwareAgentRequest")
        return self.analyze(request).to_dict()

    def predict(self, state: object, context: object = None) -> dict[str, object]:
        """Compatibility route for BaseAgent/factory predict-style dispatch."""

        if isinstance(state, Mapping):
            payload = dict(state)
            if context is not None and "context" not in payload:
                payload["context"] = context
            return self.perform_task(payload)
        raise SoftwareRequestError("SoftwareAgent.predict requires a request mapping")

    def act(self, task_data: object, context: object = None) -> dict[str, object]:
        """Compatibility route for BaseAgent/factory action-style dispatch."""

        return self.predict(task_data, context=context)

    def get_software_result(self, request_id: str) -> Mapping[str, object] | None:
        """Return an Agent-published result from SharedMemory, if present."""

        if not self.publish_shared_memory:
            return None
        key = self._key(self.result_key_prefix, str(request_id).strip())
        try:
            value = self.shared_memory.get(key)
        except Exception as exc:
            self._handle_shared_memory_error("get", key, exc)
            return None
        return value if isinstance(value, Mapping) else None

    def capabilities(self) -> dict[str, object]:
        return {
            "agent": self.name,
            "operations": tuple(sorted(item.value for item in self.allowed_operations)),
            "languages": self._models.languages(),
            "proposals_only": True,
            "executes_commands": False,
            "handoff_gates": HANDOFF_GATES,
            "shared_memory": self.publish_shared_memory,
        }

    # ------------------------------------------------------------------
    # Subsystem dispatch -- orchestration only
    # ------------------------------------------------------------------

    def _repository(self, source: SoftwareSource | None, label: str = "source") -> RepositoryModel:
        if source is None:
            raise SoftwareRequestError(f"{label} requires files, snapshot or base_patch_id", context={"field": label})
        snapshot = self._models.snapshot(
            files=source.files,  # type: ignore[arg-type]
            digest=source.snapshot,
            patch_id=source.base_patch_id,
            changes=source.changes,  # type: ignore[arg-type]
            source_roots=source.source_roots or None,
        )
        return self._models.repository(snapshot)

    def _subject(self, request: SoftwareAgentRequest) -> RepositoryModel:
        return self._repository(request.source)

    def _before_after(self, request: SoftwareAgentRequest, *, baseline_required: bool) -> tuple[RepositoryModel | None, RepositoryModel]:
        """``(before, after)`` from an explicit baseline, or from a recorded proposal via ``patch_id``."""

        patch_id = _optional_text(request.parameters, "patch_id")
        if patch_id is not None:
            if request.source is not None or request.baseline is not None:
                raise SoftwareRequestError("patch_id cannot be combined with source or baseline")
            proposal = self._refactoring.proposal(patch_id)
            before = self._models.repository(self._models.snapshot(digest=proposal.base_snapshot_digest))
            after = self._models.repository(self._models.snapshot(patch_id=patch_id))
            return before, after
        after = self._subject(request)
        if request.baseline is None:
            if baseline_required:
                raise SoftwareRequestError("operation requires a baseline (or a patch_id)")
            return None, after
        return self._repository(request.baseline, "baseline"), after

    @staticmethod
    def _parent(request: SoftwareAgentRequest) -> str | None:
        return None if request.source is None else request.source.base_patch_id

    @staticmethod
    def _read(repo: RepositoryModel, payload: Mapping[str, object]) -> _Dispatch:
        return _Dispatch(payload, snapshot_digest=repo.snapshot.digest)

    @staticmethod
    def _patch(result: PatchResult) -> _Dispatch:
        proposal = result.proposal
        return _Dispatch({"proposal": proposal.summary(), "diff": proposal.diff}, snapshot_digest=proposal.base_snapshot_digest, proposal=proposal)

    # -- read-only analyses ----------------------------------------------------------------------

    def _op_summary(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        return self._read(
            repo,
            {
                "snapshot": repo.snapshot.summary(),
                "graphs": self._graphs.summary(repo),
                "index": self._semantics.index(repo).stats(),
            },
        )

    def _op_outline(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        return self._read(repo, self._semantics.outline(repo, paths=_strings(request.parameters.get("paths"), "paths") or None))

    def _op_symbol(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        return self._read(repo, self._semantics.symbol_info(repo, _required_text(request.parameters, "symbol")))

    def _op_search(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        p = request.parameters
        found = self._semantics.find(
            repo,
            text=_optional_text(p, "text"),
            grep=opt_bool(p, "grep", False),
            mode=_optional_text(p, "mode") or "substring",
            kinds=_strings(p.get("kinds"), "kinds"),
            decorator=_optional_text(p, "decorator"),
            base_class=_optional_text(p, "base_class"),
            calls=_optional_text(p, "calls"),
            path_glob=_optional_text(p, "path_glob"),
            include_private=opt_bool(p, "include_private", True),
            regex=opt_bool(p, "regex", False),
            ignore_case=opt_bool(p, "ignore_case", False),
            limit=opt_int(p, "limit", 100, minimum=1, maximum=1000),
        )
        return self._read(repo, found)

    def _op_references(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        p = request.parameters
        line = p.get("line")
        found = self._semantics.references(
            repo,
            symbol=_optional_text(p, "symbol"),
            path=_optional_text(p, "path"),
            line=None if line is None else opt_int(p, "line", 1, minimum=1),
            column=opt_int(p, "column", 0, minimum=0),
            include_heuristic=opt_bool(p, "include_heuristic", False),
        )
        return self._read(repo, found)

    def _op_dependency_graph(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        p = request.parameters
        module, target = _optional_text(p, "module"), _optional_text(p, "target")
        limit = opt_int(p, "limit", 500, minimum=1, maximum=10_000)
        if target is not None:
            if module is None:
                raise SoftwareRequestError("parameter 'module' is required with 'target'", context={"parameter": "module"})
            return self._read(repo, self._graphs.path(repo, module, target, graph="dependency"))
        if module is not None:
            neighbours = self._graphs.dependency_neighbours(
                repo,
                module,
                direction=_optional_text(p, "direction") or "dependencies",
                transitive=opt_bool(p, "transitive", False),
                limit=limit,
            )
            return self._read(repo, neighbours)
        cycles = self._graphs.cycles(repo, graph="dependency", load_time_only=opt_bool(p, "load_time_only", False), limit=min(limit, 500))
        return self._read(repo, {"cycles": cycles, "summary": self._graphs.summary(repo)["dependency"]})

    def _op_call_graph(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        p = request.parameters
        symbol, target = _optional_text(p, "symbol"), _optional_text(p, "target")
        limit = opt_int(p, "limit", 500, minimum=1, maximum=10_000)
        if target is not None:
            if symbol is None:
                raise SoftwareRequestError("parameter 'symbol' is required with 'target'", context={"parameter": "symbol"})
            source, destination = self._semantics.resolve_symbol(repo, symbol), self._semantics.resolve_symbol(repo, target)
            return self._read(repo, self._graphs.path(repo, source, destination, graph="call"))
        if symbol is not None:
            neighbours = self._graphs.call_neighbours(
                repo,
                self._semantics.resolve_symbol(repo, symbol),
                direction=_optional_text(p, "direction") or "callees",
                transitive=opt_bool(p, "transitive", False),
                exact_only=opt_bool(p, "exact_only", False),
                limit=limit,
            )
            return self._read(repo, neighbours)
        cycles = self._graphs.cycles(repo, graph="call", limit=min(limit, 500))
        return self._read(repo, {"cycles": cycles, "summary": self._graphs.summary(repo)["call"]})

    def _op_class_hierarchy(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        return self._read(repo, self._semantics.hierarchy(repo, scope=_strings(request.parameters.get("scope"), "scope") or None).to_dict())

    def _op_static_analysis(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        p = request.parameters
        report = self._analysis.static(
            repo,
            paths=_strings(p.get("paths"), "paths") or None,
            rules=_strings(p.get("rules"), "rules") or None,
            options=request.options or None,
        )
        return self._read(repo, report.to_dict())

    def _op_dead_code(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        report = self._analysis.dead_code(repo, scope=_strings(request.parameters.get("scope"), "scope"), options=request.options or None)
        return self._read(repo, report.to_dict())

    def _op_change_impact(self, request: SoftwareAgentRequest) -> _Dispatch:
        before, after = self._before_after(request, baseline_required=False)
        p = request.parameters
        report = self._analysis.impact(after, symbols=_strings(p.get("symbols"), "symbols"), paths=_strings(p.get("paths"), "paths"), baseline=before)
        return self._read(after, report.to_dict())

    def _op_compatibility(self, request: SoftwareAgentRequest) -> _Dispatch:
        before, after = self._before_after(request, baseline_required=True)
        p = request.parameters
        report = self._semantics.compatibility(before, after, include_tests=opt_bool(p, "include_tests", False), scope=_strings(p.get("scope"), "scope"))  # type: ignore[arg-type]
        return self._read(after, report.to_dict())

    def _op_architecture(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        p = request.parameters
        report = self._analysis.architecture(
            repo,
            layers=_mappings(p.get("layers"), "layers"),
            forbidden=_mappings(p.get("forbidden"), "forbidden"),
            options=request.options or None,
        )
        return self._read(repo, report.to_dict())

    # -- patch proposals ------------------------------------------------------------------------------

    def _op_rename(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        p = request.parameters
        return self._patch(
            self._refactoring.rename(
                repo,
                _required_text(p, "symbol"),
                _required_text(p, "new_name"),
                include_heuristic=opt_bool(p, "include_heuristic", False),
                parent_patch_id=self._parent(request),
            )
        )

    def _op_remove_unused_imports(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        paths = _strings(request.parameters.get("paths"), "paths") or None
        return self._patch(self._refactoring.remove_unused_imports(repo, paths=paths, parent_patch_id=self._parent(request)))

    def _op_remove_dead_code(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        p = request.parameters
        return self._patch(
            self._refactoring.remove_dead_code(
                repo,
                min_confidence=None if p.get("min_confidence") is None else opt_float(p, "min_confidence", 0.0, minimum=0.0, maximum=1.0),
                symbols=_strings(p.get("symbols"), "symbols") or None,
                scope=_strings(p.get("scope"), "scope"),
                options=request.options or None,
                parent_patch_id=self._parent(request),
            )
        )

    def _op_migrate_api(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        rules = _mappings(request.parameters.get("rules"), "rules")
        if not rules:
            raise SoftwareRequestError("parameter 'rules' must be a non-empty sequence of rule mappings", context={"parameter": "rules"})
        return self._patch(self._refactoring.migrate(repo, rules, parent_patch_id=self._parent(request)))

    def _op_apply_diff(self, request: SoftwareAgentRequest) -> _Dispatch:
        repo = self._subject(request)
        p = request.parameters
        return self._patch(
            self._refactoring.from_diff(
                repo,
                _required_text(p, "diff"),
                intent=_optional_text(p, "intent") or "apply externally supplied patch",
                parent_patch_id=self._parent(request),
            )
        )

    def _op_proposal(self, request: SoftwareAgentRequest) -> _Dispatch:
        p = request.parameters
        proposal = self._refactoring.proposal(_required_text(p, "patch_id"))
        payload: dict[str, object] = {"proposal": proposal.summary()}
        if opt_bool(p, "include_diff", True):
            payload["diff"] = proposal.diff
        if opt_bool(p, "include_reverse_diff", False):
            payload["reverse_diff"] = proposal.reverse_diff
        if opt_bool(p, "include_lineage", True):
            payload["lineage"] = [entry.to_dict() for entry in self._refactoring.lineage(proposal.patch_id)]
        return _Dispatch(payload, snapshot_digest=proposal.base_snapshot_digest, proposal=proposal)

    # ------------------------------------------------------------------
    # SharedMemory coordination
    # ------------------------------------------------------------------

    @staticmethod
    def _key(prefix: str, request_id: str) -> str:
        request_id = str(request_id).strip()
        if not request_id:
            raise SoftwareRequestError("request_id must be a non-empty string")
        return f"{prefix}:{request_id}"

    def _shared_set(self, key: str, value: object, *, tags: Sequence[str] = ()) -> None:
        if not self.publish_shared_memory:
            return
        ttl = None if self.shared_memory_ttl_seconds <= 0 else self.shared_memory_ttl_seconds
        try:
            payload = to_json_safe(value, max_depth=12, max_items=max(1, self.max_shared_list_items), max_string_length=max(8_192, self.max_shared_diff_chars))
            self.shared_memory.set(
                key,
                payload,
                ttl=ttl,
                tags=list(tags),
                metadata={"agent": self.name, "agent_id": self.agent_id, "schema": "software_agent.v1"},
            )
        except Exception as exc:
            self._handle_shared_memory_error("set", key, exc)

    def _handle_shared_memory_error(self, operation: str, key: str, exc: BaseException) -> None:
        if self.fail_on_shared_memory_error:
            raise BaseStateError(
                "SoftwareAgent shared-memory operation failed",
                component=self.name,
                operation=f"shared_memory.{operation}",
                context={"key": key, "error_type": type(exc).__name__},
                cause=exc,
            ) from exc
        logger.warning("SoftwareAgent shared-memory %s degraded | key=%s | error=%s", operation, key, type(exc).__name__)

    def _load_prior_references(self, request_ids: Sequence[str]) -> tuple[Mapping[str, object], ...]:
        if not self.publish_shared_memory or self.max_prior_references <= 0:
            return ()
        references: list[Mapping[str, object]] = []
        for request_id in tuple(request_ids)[: self.max_prior_references]:
            key = self._key(self.summary_key_prefix, request_id)
            try:
                value = self.shared_memory.get(key)
            except Exception as exc:
                self._handle_shared_memory_error("get", key, exc)
                continue
            if isinstance(value, Mapping):
                references.append(dict(value))
        return tuple(references)

    def _parameter_preview(self, parameters: Mapping[str, object]) -> dict[str, object]:
        """Scalars verbatim, long text and containers reduced to their size (sources and diffs stay out)."""

        preview: dict[str, object] = {}
        for key, value in list(parameters.items())[: self.max_shared_list_items]:
            if value is None or isinstance(value, (bool, int, float)):
                preview[str(key)] = value
            elif isinstance(value, str):
                preview[str(key)] = value if len(value) <= 200 else f"<{len(value)} chars>"
            elif isinstance(value, Mapping):
                preview[str(key)] = f"<{len(value)} keys>"
            elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
                preview[str(key)] = f"<{len(value)} items>"
            else:
                preview[str(key)] = f"<{type(value).__name__}>"
        return preview

    def _request_metadata(self, request: SoftwareAgentRequest, prior_references: Sequence[Mapping[str, object]]) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": "software_agent.request.v1",
            "request_id": request.request_id,
            "operation": request.operation.value,
            "source": None if request.source is None else request.source.describe(),
            "baseline": None if request.baseline is None else request.baseline.describe(),
            "parameters": self._parameter_preview(request.parameters),
            "option_keys": sorted(str(key) for key in request.options)[: self.max_shared_list_items],
            "prior_request_ids": request.prior_request_ids[: self.max_prior_references],
            "prior_reference_count": len(prior_references),
            "metadata": redact_mapping(dict(request.metadata)),
        }
        payload["fingerprint"] = stable_fingerprint(payload)
        return payload

    def _shared_result_payload(self, outcome: SoftwareAgentOutcome) -> dict[str, object]:
        full = outcome.to_dict()
        result = dict(outcome.result)
        for key in ("diff", "reverse_diff"):
            text = result.get(key)
            if not isinstance(text, str):
                continue
            if not self.publish_proposal_diffs:
                result.pop(key)
                result[f"{key}_omitted"] = True
            elif len(text) > self.max_shared_diff_chars:
                result[key] = text[: self.max_shared_diff_chars]
                result[f"{key}_truncated_chars"] = len(text) - self.max_shared_diff_chars
        for key, value in list(result.items()):
            if isinstance(value, list) and len(value) > self.max_shared_list_items:
                result[key] = value[: self.max_shared_list_items]
                result[f"{key}_truncated"] = len(value) - self.max_shared_list_items
        full["result"] = result
        full["prior_references"] = list(outcome.prior_references)[: self.max_prior_references]
        full["fingerprint"] = stable_fingerprint({"operation": outcome.operation.value, "snapshot": outcome.snapshot_digest, "result": result})
        return full

    def _result_headline(self, outcome: SoftwareAgentOutcome) -> dict[str, object]:
        """Compact counts/scalars of the result, so consumers can triage without loading it."""

        headline: dict[str, object] = {}
        for key, value in outcome.result.items():
            if isinstance(value, bool) or (isinstance(value, (int, float)) and not isinstance(value, bool)):
                headline[key] = value
            elif isinstance(value, (list, tuple)):
                headline[f"{key}_count"] = len(value)
            if len(headline) >= 32:
                break
        return headline

    def _publish_outcome(self, request: SoftwareAgentRequest, outcome: SoftwareAgentOutcome) -> None:
        if not self.publish_shared_memory:
            return
        shared = self._shared_result_payload(outcome)
        proposal = outcome.proposal
        summary: dict[str, object] = {
            "schema": "software_agent.summary.v1",
            "request_id": request.request_id,
            "operation": request.operation.value,
            "execution_success": True,
            "snapshot_digest": outcome.snapshot_digest,
            "patch_id": None if proposal is None else proposal.patch_id,
            "proposal_applied": outcome.proposal_applied,
            "headline": self._result_headline(outcome),
            "elapsed_seconds": round(outcome.elapsed_seconds, 6),
            "fingerprint": shared.get("fingerprint"),
        }
        self._shared_set(self._key(self.result_key_prefix, request.request_id), shared, tags=("software", "result", request.operation.value))
        self._shared_set(self._key(self.summary_key_prefix, request.request_id), summary, tags=("software", "summary", request.operation.value))
        self._shared_set(self.latest_result_key, summary, tags=("software", "latest"))
        if proposal is not None and request.operation is not SoftwareOperation.PROPOSAL:
            # Downstream gates (verification, safety, execution, evaluation) discover proposals through this key.
            self._shared_set(
                self._key(self.proposal_key_prefix, proposal.patch_id),
                {**proposal.summary(), "request_id": request.request_id, "requires_gates": list(outcome.requires_gates)},
                tags=("software", "proposal", proposal.kind.value),
            )
        self._publish_event(
            "completed",
            {"request_id": request.request_id, "operation": request.operation.value, "patch_id": None if proposal is None else proposal.patch_id},
        )

    def _publish_error(self, request: SoftwareAgentRequest, error: BaseException) -> None:
        payload: dict[str, object] = {
            "schema": "software_agent.error.v1",
            "request_id": request.request_id,
            "operation": request.operation.value,
            "execution_success": False,
            "error_type": type(error).__name__,
            "message": str(error)[:1000],
        }
        if isinstance(error, SoftwareError):
            to_dict = getattr(error, "to_dict", None)
            if callable(to_dict):
                try:
                    serialized = to_dict()
                except Exception:
                    serialized = None
                if isinstance(serialized, Mapping):
                    payload["error"] = dict(serialized)
        self._shared_set(self._key(self.error_key_prefix, request.request_id), payload, tags=("software", "error", request.operation.value))
        self._publish_event("failed", {"request_id": request.request_id, "operation": request.operation.value, "error_type": type(error).__name__})

    def _publish_event(self, event_type: str, payload: Mapping[str, object]) -> None:
        if not self.publish_shared_memory:
            return
        event = {"schema": "software_agent.event.v1", "event_type": str(event_type), "agent": self.name, **dict(payload)}
        request_id = str(payload.get("request_id") or self.agent_id)
        self._shared_set(
            f"{self._key(self.event_key_prefix, request_id)}:{str(event_type).strip()}",
            event,
            tags=("software", "event", str(event_type)),
        )

    # ------------------------------------------------------------------
    # BaseAgent metrics integration
    # ------------------------------------------------------------------

    def extract_performance_metrics(self, result: object) -> dict[str, float]:
        """Expose orchestration and analysis telemetry without scoring correctness."""

        if not isinstance(result, Mapping):
            return {}
        payload = result.get("result")
        payload = payload if isinstance(payload, Mapping) else {}
        proposal = payload.get("proposal")
        metrics: dict[str, float] = {
            "software.execution_success": 1.0 if result.get("execution_success") is True else 0.0,
            "software.elapsed_seconds": _finite(result.get("elapsed_seconds")),
            "software.proposal_emitted": 1.0 if isinstance(proposal, Mapping) else 0.0,
        }
        if isinstance(proposal, Mapping):
            stats = proposal.get("stats")
            stats = stats if isinstance(stats, Mapping) else {}
            metrics.update(
                {
                    "software.proposal_confidence": _finite(proposal.get("confidence")),
                    "software.proposal_files": float(len(proposal.get("files") or ())),
                    "software.proposal_additions": _finite(stats.get("additions")),
                    "software.proposal_deletions": _finite(stats.get("deletions")),
                    "software.proposal_unresolved_sites": _finite(proposal.get("unresolved_sites")),
                    "software.proposal_risks": float(len(proposal.get("risks") or ())),
                    "software.proposal_warnings": float(len(proposal.get("warnings") or ())),
                }
            )
        for key in ("findings", "items", "hits", "impacted_symbols", "layer_violations", "module_cycles", "changes"):
            value = payload.get(key)
            if isinstance(value, (list, tuple)):
                metrics[f"software.{key}"] = float(len(value))
        for key in ("risk_score", "breaking"):
            if isinstance(payload.get(key), (int, float)) and not isinstance(payload.get(key), bool):
                metrics[f"software.{key}"] = _finite(payload.get(key))
        return metrics


def _finite(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if number == number and abs(number) != float("inf") else 0.0


def _optional_text(parameters: Mapping[str, object], key: str) -> str | None:
    value = parameters.get(key)
    return None if value is None else require_text(value, key, error_cls=SoftwareRequestError)


def _required_text(parameters: Mapping[str, object], key: str) -> str:
    if parameters.get(key) is None:
        raise SoftwareRequestError(f"parameter '{key}' is required", context={"parameter": key})
    return require_text(parameters[key], key, error_cls=SoftwareRequestError)


def _mappings(value: object, field_name: str) -> list[Mapping[str, object]] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or any(not isinstance(item, Mapping) for item in value):
        raise SoftwareRequestError(f"{field_name} must be a sequence of mappings", context={"field": field_name})
    return [dict(item) for item in value]


__all__ = [
    "SoftwareAgent",
    "SoftwareAgentOutcome",
    "SoftwareAgentRequest",
    "SoftwareOperation",
    "SoftwareSource",
]


if __name__ == "__main__":
    print("\n=== Running Software Agent ===\n")
    printer.status("TEST", "Software Agent initialized", "info")
    from .agent_factory import AgentFactory
    from .collaborative.shared_memory import SharedMemory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()
    config = {"publish_shared_memory": False}

    agent = SoftwareAgent(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
    printer.status("START", agent, "info")

    files = {
        "app/__init__.py": "from .core import compute\n__all__ = ['compute']\n",
        "app/core.py": "import os\n\n\ndef compute(a):\n    return a + 1\n\n\ndef user():\n    return compute(2)\n",
    }
    smoke = agent.perform_task(
        {
            "request_id": "software-agent-smoke",
            "operation": "rename",
            "files": files,
            "parameters": {"symbol": "app.core.compute", "new_name": "calculate"},
        }
    )
    printer.status("TASK #1", smoke["patch_id"], "success" if smoke["execution_success"] and not smoke["proposal_applied"] else "error")
