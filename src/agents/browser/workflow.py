from __future__ import annotations

"""
Production-grade workflow compiler and validator for the browser subsystem.

This module is intentionally not a browser action executor. It does not own a
driver, does not call Selenium, and does not duplicate BrowserFunctions,
DoClick, DoType, DoScroll, DoNavigate, DoCopyCutPaste, DoDragAndDrop, security,
content handling, or memory execution logic.

Workflow owns the layer before execution:

- accepting concise or structured workflow definitions;
- resolving safe variables into step parameters;
- normalizing aliases into canonical browser actions;
- validating step shape, required fields, URLs, selectors, dependencies, and
  bounded repeat expansions;
- compiling definitions into executable BrowserFunctions-compatible steps;
- producing dry-run summaries for logs, tests, user review, and security checks.

The compiled output is meant to be passed to BrowserFunctions.execute_workflow().
The legacy WorkFlow.normalize(...) method is retained for older BrowserAgent
code that expects a list of {"action": ..., "params": ...} dictionaries.

Local imports are intentionally direct. They are not wrapped in try/except so
packaging or path problems fail clearly during development and deployment.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union, Callable

from .utils.config_loader import load_global_config, get_config_section
from .utils.browser_errors import *
from .utils.Browser_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Workflow")
printer = PrettyPrinter


WORKFLOW_SCHEMA_VERSION = "1.0"
WORKFLOW_ACTION = "workflow"

DEFAULT_SUPPORTED_ACTIONS: Tuple[str, ...] = (
    "navigate",
    "search",
    "click",
    "type",
    "scroll",
    "copy",
    "cut",
    "paste",
    "extract",
    "screenshot",
    "page_state",
    "drag_and_drop",
    "back",
    "forward",
    "refresh",
    "current_url",
    "history",
    "clear",
    "press_key",
)

DEFAULT_ALIASES: Dict[str, str] = {
    "go_to_url": "navigate",
    "open_url": "navigate",
    "navigate_to": "navigate",
    "url": "navigate",
    "go_back": "back",
    "browser_back": "back",
    "go_forward": "forward",
    "browser_forward": "forward",
    "refresh_page": "refresh",
    "reload": "refresh",
    "get_current_url": "current_url",
    "current": "current_url",
    "get_navigation_history": "history",
    "navigation_history": "history",
    "google": "search",
    "query": "search",
    "extract_page": "extract",
    "extract_page_content": "extract",
    "get_dom": "extract",
    "page_content": "extract",
    "take_screenshot": "screenshot",
    "screenshot_page": "screenshot",
    "get_page_state": "page_state",
    "click_element": "click",
    "do_click": "click",
    "press": "click",
    "do_type": "type",
    "type_text": "type",
    "enter_text": "type",
    "input_text": "type",
    "type_element": "type",
    "clear_text": "clear",
    "scroll_element": "scroll",
    "scroll_to_element": "scroll",
    "scroll_element_into_view": "scroll",
    "scroll_direction": "scroll",
    "copy_element": "copy",
    "cut_element": "cut",
    "paste_element": "paste",
    "clipboard_copy": "copy",
    "clipboard_cut": "cut",
    "clipboard_paste": "paste",
    "drag_element": "drag_and_drop",
    "drop_element": "drag_and_drop",
    "drag_to_element": "drag_and_drop",
    "drag_by_offset": "drag_and_drop",
    "do_drag_and_drop": "drag_and_drop",
    "drag_drop": "drag_and_drop",
}

DEFAULT_PARAM_ALIASES: Dict[str, Dict[str, str]] = {
    "navigate": {"target": "url", "href": "url", "link": "url"},
    "search": {"text": "query", "q": "query"},
    "click": {"target": "selector", "element": "selector"},
    "type": {"target": "selector", "input": "text", "value": "text", "raw_input": "text"},
    "scroll": {"target": "selector"},
    "copy": {"target": "selector", "element": "selector"},
    "cut": {"target": "selector", "element": "selector"},
    "paste": {"target": "selector", "element": "selector", "value": "text"},
    "drag_and_drop": {
        "source": "source_selector",
        "from": "source_selector",
        "target": "target_selector",
        "to": "target_selector",
        "x": "offset_x",
        "y": "offset_y",
    },
    "clear": {"target": "selector", "element": "selector"},
    "press_key": {"key_name": "key", "value": "key"},
}

DEFAULT_REQUIRED_PARAMS: Dict[str, Tuple[str, ...]] = {
    "navigate": ("url",),
    "search": ("query",),
    "click": ("selector",),
    "type": ("selector", "text"),
    "copy": ("selector",),
    "cut": ("selector",),
    "paste": ("selector",),
    "clear": ("selector",),
    "press_key": ("selector", "key"),
    "drag_and_drop": ("source_selector",),
}

SELECTOR_PARAM_NAMES: Tuple[str, ...] = ("selector", "source_selector", "target_selector", "container_selector")
URL_PARAM_NAMES: Tuple[str, ...] = ("url", "engine_url")

CONTROL_KEYS = {
    "id", "name", "description", "action", "task", "tool", "function", "params",
    "policy", "security", "metadata", "depends_on", "requires", "enabled", "optional",
    "condition", "if", "then", "else", "repeat", "repeat_until", "max_iterations", "steps",
    "variables", "defaults", "schema_version", "version",
}


class WorkflowIssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class WorkflowIssueCode(str, Enum):
    INVALID_WORKFLOW = "invalid_workflow"
    INVALID_STEP = "invalid_step"
    UNSUPPORTED_ACTION = "unsupported_action"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_PARAMS = "invalid_params"
    INVALID_SELECTOR = "invalid_selector"
    INVALID_URL = "invalid_url"
    DUPLICATE_STEP_ID = "duplicate_step_id"
    UNKNOWN_DEPENDENCY = "unknown_dependency"
    CYCLIC_DEPENDENCY = "cyclic_dependency"
    UNRESOLVED_VARIABLE = "unresolved_variable"
    DISABLED_STEP = "disabled_step"
    LOOP_LIMIT_EXCEEDED = "loop_limit_exceeded"
    CONDITION_NOT_COMPILED = "condition_not_compiled"


@dataclass(frozen=True)
class WorkflowOptions:
    """Config-backed policy for workflow compilation and validation."""

    enabled: bool = True
    schema_version: str = WORKFLOW_SCHEMA_VERSION
    allow_empty_workflow: bool = False
    max_steps: int = 100
    max_repeat_iterations: int = 10
    strict_actions: bool = True
    strict_params: bool = True
    strict_variables: bool = True
    validate_selectors: bool = True
    validate_urls: bool = True
    normalize_aliases: bool = True
    include_disabled_steps: bool = False
    compile_disabled_steps: bool = False
    include_original_step: bool = True
    include_metadata: bool = True
    preserve_unknown_step_fields: bool = True
    default_stop_on_error: bool = True
    default_optional: bool = False
    supported_actions: Tuple[str, ...] = DEFAULT_SUPPORTED_ACTIONS
    aliases: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ALIASES))
    param_aliases: Dict[str, Dict[str, str]] = field(default_factory=lambda: {key: dict(value) for key, value in DEFAULT_PARAM_ALIASES.items()})
    required_params: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_REQUIRED_PARAMS))

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "WorkflowOptions":
        cfg = dict(config or {})
        aliases = dict(DEFAULT_ALIASES)
        aliases.update({str(k).strip().lower(): str(v).strip().lower() for k, v in dict(cfg.get("aliases") or {}).items()})

        param_aliases = {key: dict(value) for key, value in DEFAULT_PARAM_ALIASES.items()}
        for action, mapping in dict(cfg.get("param_aliases") or {}).items():
            normalized_action = normalize_workflow_name(action)
            param_aliases.setdefault(normalized_action, {})
            param_aliases[normalized_action].update({str(k).strip(): str(v).strip() for k, v in dict(mapping or {}).items()})

        required_params: Dict[str, Tuple[str, ...]] = dict(DEFAULT_REQUIRED_PARAMS)
        for action, params in dict(cfg.get("required_params") or {}).items():
            required_params[normalize_workflow_name(action)] = tuple(str(item).strip() for item in ensure_list(params) if str(item).strip())

        supported = ensure_list(cfg.get("supported_actions", DEFAULT_SUPPORTED_ACTIONS))
        supported_actions = tuple(normalize_workflow_name(item) for item in supported if normalize_workflow_name(item)) or DEFAULT_SUPPORTED_ACTIONS

        return cls(
            enabled=coerce_bool(cfg.get("enabled", True), default=True),
            schema_version=str(cfg.get("schema_version") or WORKFLOW_SCHEMA_VERSION),
            allow_empty_workflow=coerce_bool(cfg.get("allow_empty_workflow", False), default=False),
            max_steps=coerce_int(cfg.get("max_steps", 100), default=100, minimum=1),
            max_repeat_iterations=coerce_int(cfg.get("max_repeat_iterations", 10), default=10, minimum=1),
            strict_actions=coerce_bool(cfg.get("strict_actions", True), default=True),
            strict_params=coerce_bool(cfg.get("strict_params", True), default=True),
            strict_variables=coerce_bool(cfg.get("strict_variables", True), default=True),
            validate_selectors=coerce_bool(cfg.get("validate_selectors", True), default=True),
            validate_urls=coerce_bool(cfg.get("validate_urls", True), default=True),
            normalize_aliases=coerce_bool(cfg.get("normalize_aliases", True), default=True),
            include_disabled_steps=coerce_bool(cfg.get("include_disabled_steps", False), default=False),
            compile_disabled_steps=coerce_bool(cfg.get("compile_disabled_steps", False), default=False),
            include_original_step=coerce_bool(cfg.get("include_original_step", True), default=True),
            include_metadata=coerce_bool(cfg.get("include_metadata", True), default=True),
            preserve_unknown_step_fields=coerce_bool(cfg.get("preserve_unknown_step_fields", True), default=True),
            default_stop_on_error=coerce_bool(cfg.get("default_stop_on_error", True), default=True),
            default_optional=coerce_bool(cfg.get("default_optional", False), default=False),
            supported_actions=supported_actions,
            aliases=aliases,
            param_aliases=param_aliases,
            required_params=required_params,
        )


@dataclass(frozen=True)
class WorkflowValidationIssue:
    """A static validation issue found while compiling a workflow."""

    severity: str
    code: str
    message: str
    step_index: Optional[int] = None
    step_id: Optional[str] = None
    action: Optional[str] = None
    field: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.context is None:
            # Use object.__setattr__ because dataclass is frozen
            object.__setattr__(self, 'context', {})

    @property
    def is_error(self) -> bool:
        return self.severity == WorkflowIssueSeverity.ERROR.value

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self))
        return redact_mapping(payload) if redact else payload


@dataclass(frozen=True)
class WorkflowStep:
    """Normalized workflow step before final executable compilation."""

    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    index: int = 0
    depends_on: Tuple[str, ...] = ()
    optional: bool = False
    enabled: bool = True
    policy: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    original: Dict[str, Any] = field(default_factory=dict)

    def to_executable_step(self, *, include_metadata: bool = True, include_original: bool = True) -> Dict[str, Any]:
        step = {"action": self.action, "params": dict(self.params)}
        workflow_meta = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "index": self.index,
            "depends_on": list(self.depends_on),
            "optional": self.optional,
            "enabled": self.enabled,
            "policy": safe_serialize(self.policy),
            "security": safe_serialize(self.security),
            "metadata": safe_serialize(self.metadata),
        }
        workflow_meta = prune_none(workflow_meta)
        if include_metadata and workflow_meta:
            step["_workflow"] = workflow_meta
        if include_original and self.original:
            step["_original"] = safe_serialize(self.original)
        return prune_none(step)

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self))
        return redact_mapping(payload) if redact else payload


@dataclass(frozen=True)
class WorkflowDefinition:
    """Structured workflow definition accepted by WorkFlow.compile."""

    steps: Tuple[Any, ...]
    name: str = "workflow"
    schema_version: str = WORKFLOW_SCHEMA_VERSION
    description: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    @classmethod
    def from_input(cls, workflow: Any, *, variables: Optional[Mapping[str, Any]] = None) -> "WorkflowDefinition":
        if isinstance(workflow, WorkflowDefinition):
            merged_variables = merge_dicts(workflow.variables, dict(variables or {}), deep=True)
            return cls(
                steps=workflow.steps,
                name=workflow.name,
                schema_version=workflow.schema_version,
                description=workflow.description,
                variables=merged_variables,
                defaults=dict(workflow.defaults),
                policy=dict(workflow.policy),
                security=dict(workflow.security),
                metadata=dict(workflow.metadata),
                source=workflow.source,
            )

        if isinstance(workflow, Mapping):
            raw_steps = workflow.get("steps", workflow.get("workflow", []))
            merged_variables = merge_dicts(dict(workflow.get("variables") or {}), dict(variables or {}), deep=True)
            return cls(
                steps=tuple(ensure_list(raw_steps)),
                name=str(workflow.get("name") or workflow.get("id") or "workflow"),
                schema_version=str(workflow.get("schema_version") or workflow.get("version") or WORKFLOW_SCHEMA_VERSION),
                description=str(workflow.get("description") or ""),
                variables=merged_variables,
                defaults=dict(workflow.get("defaults") or {}),
                policy=dict(workflow.get("policy") or {}),
                security=dict(workflow.get("security") or {}),
                metadata=dict(workflow.get("metadata") or {}),
                source=workflow.get("source"),
            )

        if isinstance(workflow, Sequence) and not isinstance(workflow, (str, bytes, bytearray)):
            return cls(steps=tuple(workflow), variables=dict(variables or {}))

        raise WorkflowValidationError(
            "Workflow must be a mapping or a sequence of steps",
            context={"workflow_type": type(workflow).__name__},
        )

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self))
        return redact_mapping(payload) if redact else payload


@dataclass(frozen=True)
class CompiledWorkflow:
    """Compiled BrowserFunctions-compatible workflow."""

    name: str
    steps: Tuple[Dict[str, Any], ...]
    normalized_steps: Tuple[WorkflowStep, ...] = ()
    issues: Tuple[WorkflowValidationIssue, ...] = ()
    schema_version: str = WORKFLOW_SCHEMA_VERSION
    description: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)
    policy: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    correlation_id: str = field(default_factory=lambda: new_correlation_id("wf"))
    created_at: str = field(default_factory=utc_now_iso)

    @property
    def errors(self) -> Tuple[WorkflowValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.is_error)

    @property
    def warnings(self) -> Tuple[WorkflowValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == WorkflowIssueSeverity.WARNING.value)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self, *, redact: bool = True, include_normalized: bool = True) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "description": self.description,
            "schema_version": self.schema_version,
            "status": self.status,
            "valid": self.valid,
            "steps": list(self.steps),
            "issues": [issue.to_dict(redact=redact) for issue in self.issues],
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "policy": safe_serialize(self.policy),
            "security": safe_serialize(self.security),
            "metadata": safe_serialize(self.metadata),
            "correlation_id": self.correlation_id,
            "created_at": self.created_at,
        }
        if include_normalized:
            payload["normalized_steps"] = [step.to_dict(redact=redact) for step in self.normalized_steps]
        payload = prune_none(payload)
        return redact_mapping(payload) if redact else payload


@dataclass(frozen=True)
class WorkflowDryRun:
    """Dry-run report for a workflow definition."""

    workflow: CompiledWorkflow
    summary: Dict[str, Any] = field(default_factory=dict)
    executable_preview: Tuple[Dict[str, Any], ...] = ()
    would_execute: bool = False

    def to_result(self, *, redact: bool = True) -> Dict[str, Any]:
        status = "success" if self.workflow.valid else "error"
        message = "Workflow dry run completed" if self.workflow.valid else "Workflow dry run found validation errors"
        return {
            "status": status,
            "action": "workflow_dry_run",
            "message": message,
            "would_execute": self.would_execute,
            "summary": redact_mapping(self.summary) if redact else self.summary,
            "workflow": self.workflow.to_dict(redact=redact, include_normalized=True),
            "executable_preview": list(self.executable_preview),
        }


def normalize_workflow_name(value: Any) -> str:
    """Normalize a workflow action, alias, or field name."""

    return normalize_whitespace(value).lower().replace("-", "_").replace(" ", "_").strip("_")


def _issue(
    severity: Union[str, WorkflowIssueSeverity],
    code: Union[str, WorkflowIssueCode],
    message: str,
    *,
    step_index: Optional[int] = None,
    step_id: Optional[str] = None,
    action: Optional[str] = None,
    field: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> WorkflowValidationIssue:
    severity_value = severity.value if isinstance(severity, WorkflowIssueSeverity) else str(severity)
    code_value = code.value if isinstance(code, WorkflowIssueCode) else str(code)
    return WorkflowValidationIssue(
        severity=severity_value,
        code=code_value,
        message=message,
        step_index=step_index,
        step_id=step_id,
        action=action,
        field=field,
        context=safe_serialize(dict(context or {})),
    )


class WorkFlow:
    """Driver-agnostic workflow compiler and validator.

    ``WorkFlow`` is intentionally a pre-execution utility. It compiles authorable
    workflow definitions into simple executable step lists consumed by
    BrowserFunctions.execute_workflow().
    """

    SUPPORTED_ACTIONS = set(DEFAULT_SUPPORTED_ACTIONS)

    def __init__(
        self,
        *,
        supported_actions: Optional[Iterable[str]] = None,
        aliases: Optional[Mapping[str, str]] = None,
        config: Optional[Mapping[str, Any]] = None,
    ):
        self.config = load_global_config()
        self.workflow_config = merge_dicts(
            get_config_section("workflow") or {},
            get_config_section("browser_workflow") or {},
            dict(config or {}),
            deep=True,
        )
        self.options = WorkflowOptions.from_config(self.workflow_config)

        if supported_actions is not None:
            resolved_supported = tuple(normalize_workflow_name(action) for action in supported_actions if normalize_workflow_name(action))
            self.options = WorkflowOptions.from_config(
                merge_dicts(asdict(self.options), {"supported_actions": list(resolved_supported)}, deep=True)
            )

        if aliases:
            merged_aliases = dict(self.options.aliases)
            merged_aliases.update({normalize_workflow_name(k): normalize_workflow_name(v) for k, v in aliases.items()})
            self.options = WorkflowOptions.from_config(merge_dicts(asdict(self.options), {"aliases": merged_aliases}, deep=True))

        self.supported_actions: Set[str] = set(self.options.supported_actions)
        self.aliases: Dict[str, str] = dict(self.options.aliases)
        logger.info("Workflow compiler initialized.")

    def normalize(
        self,
        workflow_script: Union[List[Dict[str, Any]], Mapping[str, Any], WorkflowDefinition],
        *,
        variables: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Backwards-compatible normalization returning executable steps only."""
        compiled = self.compile(workflow_script, variables=variables, raise_on_error=True)
        return [dict(step) for step in compiled.steps]

    def validate(
        self,
        workflow: Union[Sequence[Any], Mapping[str, Any], WorkflowDefinition],
        *,
        variables: Optional[Mapping[str, Any]] = None,
    ) -> List[WorkflowValidationIssue]:
        """Validate a workflow definition without returning executable steps."""

        return list(self.compile(workflow, variables=variables, raise_on_error=False).issues)

    def compile(
        self,
        workflow: Union[Sequence[Any], Mapping[str, Any], WorkflowDefinition],
        *,
        variables: Optional[Mapping[str, Any]] = None,
        raise_on_error: bool = False,
    ) -> CompiledWorkflow:
        """Compile a workflow definition into BrowserFunctions-compatible steps."""

        definition = WorkflowDefinition.from_input(workflow, variables=variables)
        issues: List[WorkflowValidationIssue] = []
        normalized_steps: List[WorkflowStep] = []

        if not self.options.enabled:
            issues.append(_issue(WorkflowIssueSeverity.ERROR, WorkflowIssueCode.INVALID_WORKFLOW, "Workflow support is disabled"))

        if not definition.steps and not self.options.allow_empty_workflow:
            issues.append(_issue(WorkflowIssueSeverity.ERROR, WorkflowIssueCode.INVALID_WORKFLOW, "Workflow must contain at least one step"))

        compiled_variables = merge_dicts(definition.variables, dict(variables or {}), deep=True)

        for index, raw_step in enumerate(definition.steps):
            normalized_steps.extend(self._normalize_step(raw_step, index=index, definition=definition, variables=compiled_variables, issues=issues))

        if len(normalized_steps) > self.options.max_steps:
            issues.append(
                _issue(
                    WorkflowIssueSeverity.ERROR,
                    WorkflowIssueCode.INVALID_WORKFLOW,
                    f"Workflow has {len(normalized_steps)} compiled steps, exceeding max_steps={self.options.max_steps}",
                    context={"max_steps": self.options.max_steps, "compiled_steps": len(normalized_steps)},
                )
            )
            normalized_steps = normalized_steps[: self.options.max_steps]

        self._validate_step_ids_and_dependencies(normalized_steps, issues)
        self._validate_steps(normalized_steps, issues)

        executable_steps = tuple(
            step.to_executable_step(include_metadata=self.options.include_metadata, include_original=self.options.include_original_step)
            for step in normalized_steps
            if step.enabled or self.options.compile_disabled_steps
        )

        status = "success" if not any(issue.is_error for issue in issues) else "error"
        compiled = CompiledWorkflow(
            name=definition.name,
            description=definition.description,
            schema_version=definition.schema_version,
            steps=executable_steps,
            normalized_steps=tuple(normalized_steps),
            issues=tuple(issues),
            variables=safe_serialize(compiled_variables),
            policy=safe_serialize(definition.policy),
            security=safe_serialize(definition.security),
            metadata=merge_dicts(
                definition.metadata,
                {
                    "source": definition.source,
                    "compiled_step_count": len(executable_steps),
                    "normalized_step_count": len(normalized_steps),
                    "fingerprint": stable_hash(
                        {"name": definition.name, "steps": executable_steps, "variables": safe_serialize(compiled_variables)},
                        length=20,
                    ),
                },
                deep=True,
            ),
            status=status,
        )

        if raise_on_error and not compiled.valid:
            raise WorkflowValidationError(
                "Workflow validation failed",
                context={"name": definition.name, "issues": [issue.to_dict() for issue in compiled.errors]},
            )

        return compiled

    def dry_run(
        self,
        workflow: Union[Sequence[Any], Mapping[str, Any], WorkflowDefinition],
        *,
        variables: Optional[Mapping[str, Any]] = None,
        max_preview_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compile and summarize a workflow without executing browser actions."""

        compiled = self.compile(workflow, variables=variables, raise_on_error=False)
        preview_count = max_preview_steps if max_preview_steps is not None else min(20, len(compiled.steps))
        summary = {
            "name": compiled.name,
            "valid": compiled.valid,
            "status": compiled.status,
            "steps_total": len(compiled.steps),
            "normalized_steps_total": len(compiled.normalized_steps),
            "errors": len(compiled.errors),
            "warnings": len(compiled.warnings),
            "actions": self._summarize_actions(compiled.normalized_steps),
            "would_execute": compiled.valid,
            "fingerprint": stable_hash([step for step in compiled.steps], length=20),
        }
        dry_run = WorkflowDryRun(
            workflow=compiled,
            summary=summary,
            executable_preview=tuple(compiled.steps[: max(0, preview_count)]),
            would_execute=compiled.valid,
        )
        return dry_run.to_result(redact=True)

    def _normalize_step(
        self,
        raw_step: Any,
        *,
        index: int,
        definition: WorkflowDefinition,
        variables: Mapping[str, Any],
        issues: List[WorkflowValidationIssue],
    ) -> List[WorkflowStep]:
        if isinstance(raw_step, WorkflowStep):
            return [raw_step]

        if not isinstance(raw_step, Mapping):
            issues.append(
                _issue(
                    WorkflowIssueSeverity.ERROR,
                    WorkflowIssueCode.INVALID_STEP,
                    "Workflow step must be a mapping",
                    step_index=index,
                    context={"step_type": type(raw_step).__name__},
                )
            )
            return []

        raw = dict(raw_step)

        if coerce_bool(raw.get("enabled", True), default=True) is False and not self.options.include_disabled_steps:
            issues.append(
                _issue(
                    WorkflowIssueSeverity.INFO,
                    WorkflowIssueCode.DISABLED_STEP,
                    "Workflow step is disabled and will not be compiled",
                    step_index=index,
                    step_id=raw.get("id"),
                )
            )
            return []

        if "repeat" in raw:
            return self._normalize_repeat_step(raw, index=index, definition=definition, variables=variables, issues=issues)

        if "if" in raw and ("then" in raw or "else" in raw):
            return self._normalize_conditional_step(raw, index=index, definition=definition, variables=variables, issues=issues)

        action, params = self._extract_action_and_params(raw, issues=issues, index=index)
        canonical_action = self._canonical_action(action)
        params = self._normalize_params(canonical_action, params, defaults=definition.defaults)
        params = self._resolve_variables(params, variables=variables, issues=issues, step_index=index, action=canonical_action)

        step_id = raw.get("id") or f"step_{index + 1:03d}_{canonical_action or 'unknown'}"
        depends_on = raw.get("depends_on", raw.get("requires", ()))
        depends_on_tuple = tuple(str(item).strip() for item in ensure_list(depends_on) if str(item).strip())

        metadata: Dict[str, Any] = dict(raw.get("metadata") or {})
        if self.options.preserve_unknown_step_fields:
            extras = {key: value for key, value in raw.items() if key not in CONTROL_KEYS and key != action}
            if extras:
                metadata["extras"] = safe_serialize(extras)

        return [
            WorkflowStep(
                id=str(step_id),
                name=raw.get("name"),
                description=raw.get("description"),
                index=index,
                action=canonical_action,
                params=params,
                depends_on=depends_on_tuple,
                optional=coerce_bool(raw.get("optional", self.options.default_optional), default=self.options.default_optional),
                enabled=coerce_bool(raw.get("enabled", True), default=True),
                policy=merge_dicts(definition.policy, dict(raw.get("policy") or {}), deep=True),
                security=merge_dicts(definition.security, dict(raw.get("security") or {}), deep=True),
                metadata=metadata,
                original=safe_serialize(raw),
            )
        ]

    def _extract_action_and_params(self, raw: Mapping[str, Any], *, issues: List[WorkflowValidationIssue], index: int) -> Tuple[str, Dict[str, Any]]:
        for key in ("action", "task", "tool", "function"):
            if raw.get(key):
                action = normalize_workflow_name(raw.get(key))
                params = dict(raw.get("params") or {})
                implicit = {k: v for k, v in raw.items() if k not in CONTROL_KEYS}
                params.update(implicit)
                return action, params

        action_keys = [key for key in raw.keys() if normalize_workflow_name(key) in self.supported_actions or normalize_workflow_name(key) in self.aliases]
        if len(action_keys) == 1:
            action_key = action_keys[0]
            action = normalize_workflow_name(action_key)
            value = raw[action_key]
            params = self._params_from_concise_value(action, value)
            params.update(dict(raw.get("params") or {}))
            implicit = {k: v for k, v in raw.items() if k not in CONTROL_KEYS and k != action_key}
            params.update(implicit)
            return action, params

        issues.append(
            _issue(
                WorkflowIssueSeverity.ERROR,
                WorkflowIssueCode.MISSING_REQUIRED_FIELD,
                "Workflow step is missing an action/task/tool/function field",
                step_index=index,
                context={"step_keys": sorted(str(key) for key in raw.keys())},
            )
        )
        return "", {}

    def _params_from_concise_value(self, action: str, value: Any) -> Dict[str, Any]:
        canonical = self._canonical_action(action)
        if isinstance(value, Mapping):
            return dict(value)
        if canonical == "navigate":
            return {"url": value}
        if canonical == "search":
            return {"query": value}
        if canonical in {"click", "copy", "cut", "paste", "clear", "scroll"}:
            return {"selector": value}
        if canonical == "type":
            return {"text": value}
        if canonical == "press_key":
            return {"key": value}
        return {"value": value}

    def _normalize_repeat_step(self, raw: Mapping[str, Any], *, index: int, definition: WorkflowDefinition, variables: Mapping[str, Any], issues: List[WorkflowValidationIssue]) -> List[WorkflowStep]:
        repeat = raw.get("repeat")
        repeat_cfg = dict(repeat if isinstance(repeat, Mapping) else {"times": repeat})
        times = coerce_int(repeat_cfg.get("times", raw.get("times", 1)), default=1, minimum=0)
        if times > self.options.max_repeat_iterations:
            issues.append(
                _issue(
                    WorkflowIssueSeverity.ERROR,
                    WorkflowIssueCode.LOOP_LIMIT_EXCEEDED,
                    f"Repeat count {times} exceeds max_repeat_iterations={self.options.max_repeat_iterations}",
                    step_index=index,
                    context={"times": times, "max_repeat_iterations": self.options.max_repeat_iterations},
                )
            )
            times = self.options.max_repeat_iterations

        steps = ensure_list(repeat_cfg.get("steps", raw.get("steps", [])))
        normalized: List[WorkflowStep] = []
        for iteration in range(times):
            iteration_variables = merge_dicts(dict(variables), {"iteration": iteration, "iteration_number": iteration + 1}, deep=True)
            for nested_index, nested_step in enumerate(steps):
                expanded = self._normalize_step(
                    nested_step,
                    index=index * 1000 + iteration * 100 + nested_index,
                    definition=definition,
                    variables=iteration_variables,
                    issues=issues,
                )
                for step in expanded:
                    normalized.append(
                        WorkflowStep(
                            action=step.action,
                            params=step.params,
                            id=f"{raw.get('id') or f'repeat_{index + 1}'}_{iteration + 1}_{step.id}",
                            name=step.name,
                            description=step.description,
                            index=len(normalized),
                            depends_on=step.depends_on,
                            optional=step.optional,
                            enabled=step.enabled,
                            policy=merge_dicts(step.policy, {"repeat_iteration": iteration + 1}, deep=True),
                            security=step.security,
                            metadata=merge_dicts(step.metadata, {"repeat": {"index": index, "iteration": iteration + 1}}, deep=True),
                            original=step.original,
                        )
                    )
        return normalized

    def _normalize_conditional_step(self, raw: Mapping[str, Any], *, index: int, definition: WorkflowDefinition, variables: Mapping[str, Any], issues: List[WorkflowValidationIssue]) -> List[WorkflowStep]:
        condition = raw.get("if")
        decision = self._evaluate_static_condition(condition, variables)
        if decision is None:
            issues.append(
                _issue(
                    WorkflowIssueSeverity.WARNING,
                    WorkflowIssueCode.CONDITION_NOT_COMPILED,
                    "Conditional step could not be statically evaluated; no branch was compiled",
                    step_index=index,
                    step_id=raw.get("id"),
                    context={"condition": safe_serialize(condition)},
                )
            )
            return []

        branch_key = "then" if decision else "else"
        branch_steps = ensure_list(raw.get(branch_key, []))
        normalized: List[WorkflowStep] = []
        for nested_index, nested_step in enumerate(branch_steps):
            normalized.extend(
                self._normalize_step(nested_step, index=index * 1000 + nested_index, definition=definition, variables=variables, issues=issues)
            )
        return normalized

    def _evaluate_static_condition(self, condition: Any, variables: Mapping[str, Any]) -> Optional[bool]:
        if isinstance(condition, bool):
            return condition
        if not isinstance(condition, Mapping):
            return None
        if "variable" in condition:
            value = self._resolve_variable_path(str(condition.get("variable")), variables)
            if value is _MissingVariable:
                return None
            if "equals" in condition:
                return value == condition.get("equals")
            if "not_equals" in condition:
                return value != condition.get("not_equals")
            if "truthy" in condition:
                return bool(value) is coerce_bool(condition.get("truthy"), default=True)
            return bool(value)
        return None

    def _canonical_action(self, action: Any) -> str:
        normalized = normalize_workflow_name(action)
        if self.options.normalize_aliases:
            return self.aliases.get(normalized, normalized)
        return normalized

    def _normalize_params(self, action: str, params: Mapping[str, Any], *, defaults: Mapping[str, Any]) -> Dict[str, Any]:
        action_defaults = dict(defaults.get(action) or {}) if isinstance(defaults.get(action), Mapping) else {}
        merged = merge_dicts(action_defaults, dict(params or {}), deep=True)
        aliases = self.options.param_aliases.get(action, {})
        normalized: Dict[str, Any] = {}
        for key, value in merged.items():
            normalized_key = aliases.get(str(key), str(key))
            normalized[normalized_key] = value
        return normalized

    def _resolve_variables(self, value: Any, *, variables: Mapping[str, Any], issues: List[WorkflowValidationIssue], step_index: int, action: str) -> Any:
        if isinstance(value, str):
            return self._resolve_template_string(value, variables=variables, issues=issues, step_index=step_index, action=action)
        if isinstance(value, Mapping):
            return {key: self._resolve_variables(item, variables=variables, issues=issues, step_index=step_index, action=action) for key, item in value.items()}
        if isinstance(value, list):
            return [self._resolve_variables(item, variables=variables, issues=issues, step_index=step_index, action=action) for item in value]
        if isinstance(value, tuple):
            return tuple(self._resolve_variables(item, variables=variables, issues=issues, step_index=step_index, action=action) for item in value)
        return value

    def _resolve_template_string(self, text: str, *, variables: Mapping[str, Any], issues: List[WorkflowValidationIssue], step_index: int, action: str) -> Any:
        stripped = text.strip()
        if stripped.startswith("{{") and stripped.endswith("}}") and stripped.count("{{") == 1 and stripped.count("}}") == 1:
            key = stripped[2:-2].strip()
            value = self._resolve_variable_path(key, variables)
            if value is _MissingVariable:
                self._record_unresolved_variable(key, issues, step_index, action)
                return text
            return value

        output = text
        start = 0
        while True:
            open_index = output.find("{{", start)
            if open_index < 0:
                break
            close_index = output.find("}}", open_index + 2)
            if close_index < 0:
                break
            key = output[open_index + 2 : close_index].strip()
            value = self._resolve_variable_path(key, variables)
            if value is _MissingVariable:
                self._record_unresolved_variable(key, issues, step_index, action)
                start = close_index + 2
                continue
            output = output[:open_index] + str(value) + output[close_index + 2 :]
            start = open_index + len(str(value))
        return output

    def _resolve_variable_path(self, path: str, variables: Mapping[str, Any]) -> Any:
        current: Any = variables
        for part in str(path).split("."):
            part = part.strip()
            if not part:
                return _MissingVariable
            if isinstance(current, Mapping) and part in current:
                current = current[part]
                continue
            if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
                try:
                    current = current[int(part)]
                    continue
                except Exception:
                    return _MissingVariable
            return _MissingVariable
        return current

    def _record_unresolved_variable(self, key: str, issues: List[WorkflowValidationIssue], step_index: int, action: str) -> None:
        issues.append(
            _issue(
                WorkflowIssueSeverity.ERROR if self.options.strict_variables else WorkflowIssueSeverity.WARNING,
                WorkflowIssueCode.UNRESOLVED_VARIABLE,
                f"Unresolved workflow variable: {key}",
                step_index=step_index,
                action=action,
                field=key,
            )
        )

    def _validate_steps(self, steps: Sequence[WorkflowStep], issues: List[WorkflowValidationIssue]) -> None:
        for step in steps:
            if not step.action:
                issues.append(_issue(WorkflowIssueSeverity.ERROR, WorkflowIssueCode.MISSING_REQUIRED_FIELD, "Workflow step action is required", step_index=step.index, step_id=step.id))
                continue

            if self.options.strict_actions and step.action not in self.supported_actions:
                issues.append(
                    _issue(
                        WorkflowIssueSeverity.ERROR,
                        WorkflowIssueCode.UNSUPPORTED_ACTION,
                        f"Unsupported workflow action: {step.action}",
                        step_index=step.index,
                        step_id=step.id,
                        action=step.action,
                        context={"supported_actions": sorted(self.supported_actions)},
                    )
                )

            if not isinstance(step.params, Mapping):
                issues.append(_issue(WorkflowIssueSeverity.ERROR, WorkflowIssueCode.INVALID_PARAMS, "Workflow step params must be a mapping", step_index=step.index, step_id=step.id, action=step.action))
                continue

            self._validate_required_params(step, issues)
            if self.options.validate_urls:
                self._validate_url_params(step, issues)
            if self.options.validate_selectors:
                self._validate_selector_params(step, issues)

    def _validate_required_params(self, step: WorkflowStep, issues: List[WorkflowValidationIssue]) -> None:
        required = self.options.required_params.get(step.action, ())
        for field_name in required:
            if field_name not in step.params or step.params.get(field_name) in (None, ""):
                issues.append(
                    _issue(
                        WorkflowIssueSeverity.ERROR if self.options.strict_params else WorkflowIssueSeverity.WARNING,
                        WorkflowIssueCode.MISSING_REQUIRED_FIELD,
                        f"Workflow step '{step.action}' is missing required param '{field_name}'",
                        step_index=step.index,
                        step_id=step.id,
                        action=step.action,
                        field=field_name,
                    )
                )

        if step.action == "drag_and_drop":
            has_target = bool(step.params.get("target_selector"))
            has_offset = step.params.get("offset_x") is not None or step.params.get("offset_y") is not None
            if not has_target and not has_offset:
                issues.append(
                    _issue(
                        WorkflowIssueSeverity.ERROR,
                        WorkflowIssueCode.MISSING_REQUIRED_FIELD,
                        "drag_and_drop requires either target_selector or offset_x/offset_y",
                        step_index=step.index,
                        step_id=step.id,
                        action=step.action,
                        field="target_selector",
                    )
                )

    def _validate_url_params(self, step: WorkflowStep, issues: List[WorkflowValidationIssue]) -> None:
        for field_name in URL_PARAM_NAMES:
            if field_name not in step.params or step.params.get(field_name) in (None, ""):
                continue
            value = step.params[field_name]
            try:
                normalized = normalize_url(str(value))
                validate_url(normalized, field_name=field_name)
                step.params[field_name] = normalized
            except Exception as exc:
                issues.append(
                    _issue(
                        WorkflowIssueSeverity.ERROR,
                        WorkflowIssueCode.INVALID_URL,
                        f"Invalid URL in workflow step param '{field_name}'",
                        step_index=step.index,
                        step_id=step.id,
                        action=step.action,
                        field=field_name,
                        context={"value": value, "error": str(exc)},
                    )
                )

    def _validate_selector_params(self, step: WorkflowStep, issues: List[WorkflowValidationIssue]) -> None:
        for field_name in SELECTOR_PARAM_NAMES:
            if field_name not in step.params or step.params.get(field_name) in (None, ""):
                continue
            value = step.params[field_name]
            try:
                validate_css_selector(str(value), field_name=field_name)
            except Exception as exc:
                issues.append(
                    _issue(
                        WorkflowIssueSeverity.ERROR,
                        WorkflowIssueCode.INVALID_SELECTOR,
                        f"Invalid CSS selector in workflow step param '{field_name}'",
                        step_index=step.index,
                        step_id=step.id,
                        action=step.action,
                        field=field_name,
                        context={"value": value, "error": str(exc)},
                    )
                )

    def _validate_step_ids_and_dependencies(self, steps: Sequence[WorkflowStep], issues: List[WorkflowValidationIssue]) -> None:
        seen: Dict[str, WorkflowStep] = {}
        for step in steps:
            if not step.id:
                continue
            if step.id in seen:
                issues.append(_issue(WorkflowIssueSeverity.ERROR, WorkflowIssueCode.DUPLICATE_STEP_ID, f"Duplicate workflow step id: {step.id}", step_index=step.index, step_id=step.id))
            seen[step.id] = step

        for step in steps:
            for dependency in step.depends_on:
                if dependency not in seen:
                    issues.append(
                        _issue(
                            WorkflowIssueSeverity.ERROR,
                            WorkflowIssueCode.UNKNOWN_DEPENDENCY,
                            f"Workflow step depends on unknown step id: {dependency}",
                            step_index=step.index,
                            step_id=step.id,
                            field="depends_on",
                            context={"dependency": dependency},
                        )
                    )

        self._detect_dependency_cycles(steps, issues)

    def _detect_dependency_cycles(self, steps: Sequence[WorkflowStep], issues: List[WorkflowValidationIssue]) -> None:
        graph = {step.id: set(step.depends_on) for step in steps if step.id}
        visiting: Set[str] = set()
        visited: Set[str] = set()

        def visit(node: str, stack: List[str]) -> None:
            if node in visited:
                return
            if node in visiting:
                issues.append(_issue(WorkflowIssueSeverity.ERROR, WorkflowIssueCode.CYCLIC_DEPENDENCY, "Workflow dependency cycle detected", step_id=node, context={"cycle": [*stack, node]}))
                return
            visiting.add(node)
            for dependency in graph.get(node, set()):
                if dependency in graph:
                    visit(dependency, [*stack, node])
            visiting.remove(node)
            visited.add(node)

        for node in graph:
            visit(node, [])

    def _summarize_actions(self, steps: Sequence[WorkflowStep]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for step in steps:
            counts[step.action] = counts.get(step.action, 0) + 1
        return counts


class _MissingVariableType:
    pass


_MissingVariable = _MissingVariableType()


if __name__ == "__main__":
    print("\n=== Running Workflow ===\n")
    printer.status("TEST", "Workflow initialized", "info")

    workflow = WorkFlow(supported_actions=["navigate", "click", "type", "scroll", "extract", "drag_and_drop"])

    definition = {
        "name": "example_login_flow",
        "description": "Compile a browser workflow without executing it.",
        "variables": {
            "base_url": "https://example.com",
            "email": "agent@example.com",
            "selectors": {"login": "#login", "email": "input[name='email']"},
        },
        "steps": [
            {"id": "open", "navigate": "{{base_url}}/login"},
            {"id": "click_login", "click": "{{selectors.login}}", "depends_on": ["open"]},
            {
                "id": "type_email",
                "action": "type",
                "params": {"selector": "{{selectors.email}}", "text": "{{email}}"},
                "depends_on": ["click_login"],
            },
            {"id": "scroll_down", "action": "scroll", "params": {"mode": "direction", "direction": "down", "amount": 300}},
            {"id": "extract", "action": "extract", "params": {"preview_only": True}},
        ],
    }

    compiled = workflow.compile(definition, raise_on_error=True)
    assert compiled.valid, compiled.to_dict()
    assert len(compiled.steps) == 5
    assert compiled.steps[0]["action"] == "navigate"
    assert compiled.steps[0]["params"]["url"] == "https://example.com/login"
    assert compiled.steps[2]["params"]["text"] == "agent@example.com"

    normalized = workflow.normalize(definition["steps"], variables=definition.get("variables"))
    assert isinstance(normalized, list)
    assert normalized[0]["action"] == "navigate"

    invalid = workflow.compile([{"action": "click", "params": {}}])
    assert not invalid.valid
    assert invalid.errors

    repeated = workflow.compile(
        {
            "name": "repeat_example",
            "steps": [
                {
                    "id": "repeat_scroll",
                    "repeat": {"times": 2, "steps": [{"action": "scroll", "params": {"mode": "direction", "direction": "down"}}]},
                }
            ],
        }
    )
    assert repeated.valid
    assert len(repeated.steps) == 2

    dry_run = workflow.dry_run(definition)
    assert dry_run["status"] == "success"
    assert dry_run["would_execute"] is True

    print("\n=== Test ran successfully ===\n")
