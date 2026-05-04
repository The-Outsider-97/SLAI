from __future__ import annotations

"""
Production-grade task contract validation for the collaborative agent subsystem.

This module owns the validation boundary for collaborative task payloads. It
keeps orchestration, policy decisions, routing strategy, registry discovery, and
reliability state transitions in their dedicated modules while providing a
stable contract layer for payload shape, required fields, type checks, field
constraints, defaults, aliases, and custom validators.

Responsibilities
----------------
- Preserve the original public API: ``ContractValidationResult``,
  ``TaskContract``, ``TaskContractRegistry.register_contract()``, ``get()``,
  ``validate()``, and ``list_contracts()`` remain available and compatible.
- Validate payloads before routing or execution using deterministic contract
  rules.
- Support both programmatic contracts and config-backed contracts from
  ``collaborative_config.yaml``.
- Use collaborative helpers for normalization, redaction, JSON-safe telemetry,
  audit events, result payloads, identifiers, timestamps, and shared-memory
  publishing.
- Use collaboration errors at registry/contract boundary failures instead of
  leaking unstructured exceptions.

Design principles
-----------------
1. Stable contracts: existing dataclass names and method names are retained.
2. Direct local imports: project-local config, error, and helper imports remain
   explicit and unwrapped.
3. No routing ownership: validation can decide whether a payload is valid, but
   routing and fallback behavior remain in TaskRouter/CollaborationManager.
4. Config-backed behavior: task contract tuning belongs in
   collaborative_config.yaml under ``task_contracts``.
5. Defensive diagnostics: validation output is JSON-safe, redacted by default,
   and suitable for logs, shared memory, reports, and tests.
"""

import re
import threading

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.collaboration_error import *
from .utils.collaborative_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Task Contracts")
printer = PrettyPrinter()

Validator = Callable[[Dict[str, Any]], Tuple[bool, Optional[str]]]
FieldTransform = Callable[[Any], Any]


class ContractIssueSeverity(str, Enum):
    """Severity for validation findings."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ContractFieldOperator(str, Enum):
    """Supported field-level assertion operators for config-backed rules."""

    EXISTS = "exists"
    MISSING = "missing"
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    REGEX = "regex"
    LENGTH_MIN = "length_min"
    LENGTH_MAX = "length_max"
    TRUTHY = "truthy"
    FALSY = "falsy"


_TYPE_ALIASES: Dict[str, Tuple[type, ...]] = {
    "any": (object,),
    "object": (dict,),
    "mapping": (dict,),
    "dict": (dict,),
    "json": (dict, list, str, int, float, bool, type(None)),
    "list": (list,),
    "array": (list,),
    "tuple": (tuple,),
    "sequence": (list, tuple),
    "str": (str,),
    "string": (str,),
    "text": (str,),
    "int": (int,),
    "integer": (int,),
    "float": (float,),
    "number": (int, float),
    "numeric": (int, float),
    "bool": (bool,),
    "boolean": (bool,),
    "none": (type(None),),
    "null": (type(None),),
    "path": (str, Path),
}

_DEFAULT_TASK_CONTRACT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "load_configured_contracts": True,
    "fail_open_without_contract": True,
    "allow_contract_override": True,
    "max_contracts": 1000,
    "audit_enabled": True,
    "audit_key": "collaboration:task_contract_events",
    "audit_max_events": 1000,
    "include_payload_snapshot": False,
    "redact_payload_snapshots": True,
    "default_allow_unknown_fields": True,
    "default_coerce_types": False,
    "default_strict_types": True,
    "validation_error_raises": False,
    "configured_contracts": [],
}

_MISSING = object()


@dataclass(frozen=True)
class ContractValidationIssue:
    """Single structured validation finding."""

    field: Optional[str]
    message: str
    code: str = "validation_error"
    severity: ContractIssueSeverity = ContractIssueSeverity.ERROR
    expected: Any = None
    actual: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(
            {
                "field": self.field,
                "message": self.message,
                "code": self.code,
                "severity": self.severity.value if isinstance(self.severity, ContractIssueSeverity) else str(self.severity),
                "expected": json_safe(self.expected),
                "actual": json_safe(self.actual),
                "metadata": json_safe(self.metadata),
            },
            drop_empty=True,
        )
        return redact_mapping(payload) if redact else payload


@dataclass(frozen=True)
class FieldConstraint:
    """Declarative assertion for one field value.

    Config examples::

        {"operator": "gte", "value": 0.0}
        {"operator": "regex", "value": "^[a-z0-9_]+$"}
        {"operator": "in", "values": ["train", "eval"]}
    """

    operator: ContractFieldOperator
    value: Any = None
    values: Tuple[Any, ...] = ()
    message: Optional[str] = None
    code: str = "field_constraint_failed"
    severity: ContractIssueSeverity = ContractIssueSeverity.ERROR
    case_sensitive: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FieldConstraint":
        data = ensure_mapping(payload, field_name="field_constraint")
        raw_values = data.get("values", ())
        if raw_values is None:
            values: Tuple[Any, ...] = ()
        elif isinstance(raw_values, (str, bytes)):
            values = (raw_values,)
        elif isinstance(raw_values, Iterable):
            values = tuple(raw_values)
        else:
            values = (raw_values,)
        return cls(
            operator=normalize_constraint_operator(data.get("operator", data.get("op", "exists"))),
            value=data.get("value"),
            values=values,
            message=str(data["message"]).strip() if data.get("message") is not None else None,
            code=str(data.get("code", "field_constraint_failed")),
            severity=normalize_issue_severity(data.get("severity", ContractIssueSeverity.ERROR.value)),
            case_sensitive=coerce_bool(data.get("case_sensitive"), default=True),
            metadata=normalize_metadata(data.get("metadata"), drop_none=True),
        )

    def evaluate(self, field_name: str, value: Any, exists: bool) -> Optional[ContractValidationIssue]:
        op = self.operator
        matched = True
        expected: Any = self.value if self.values == () else list(self.values)

        if op == ContractFieldOperator.EXISTS:
            matched = exists
        elif op == ContractFieldOperator.MISSING:
            matched = not exists
        elif not exists:
            matched = False
        elif op == ContractFieldOperator.EQ:
            matched = _compare_values(value, self.value, case_sensitive=self.case_sensitive) == 0
        elif op == ContractFieldOperator.NE:
            matched = _compare_values(value, self.value, case_sensitive=self.case_sensitive) != 0
        elif op == ContractFieldOperator.GT:
            matched = _numeric_compare(value, self.value, ">")
        elif op == ContractFieldOperator.GTE:
            matched = _numeric_compare(value, self.value, ">=")
        elif op == ContractFieldOperator.LT:
            matched = _numeric_compare(value, self.value, "<")
        elif op == ContractFieldOperator.LTE:
            matched = _numeric_compare(value, self.value, "<=")
        elif op == ContractFieldOperator.IN:
            matched = _value_in(value, self.values or (self.value,), case_sensitive=self.case_sensitive)
        elif op == ContractFieldOperator.NOT_IN:
            matched = not _value_in(value, self.values or (self.value,), case_sensitive=self.case_sensitive)
        elif op == ContractFieldOperator.CONTAINS:
            matched = _contains(value, self.value, case_sensitive=self.case_sensitive)
        elif op == ContractFieldOperator.NOT_CONTAINS:
            matched = not _contains(value, self.value, case_sensitive=self.case_sensitive)
        elif op == ContractFieldOperator.REGEX:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            matched = re.search(str(self.value or ""), str(value), flags=flags) is not None
        elif op == ContractFieldOperator.LENGTH_MIN:
            matched = _length(value) >= coerce_int(self.value, default=0, minimum=0)
        elif op == ContractFieldOperator.LENGTH_MAX:
            matched = _length(value) <= coerce_int(self.value, default=0, minimum=0)
        elif op == ContractFieldOperator.TRUTHY:
            matched = bool(value)
        elif op == ContractFieldOperator.FALSY:
            matched = not bool(value)

        if matched:
            return None
        return ContractValidationIssue(
            field=field_name,
            message=self.message or f"field '{field_name}' failed constraint '{op.value}'",
            code=self.code,
            severity=self.severity,
            expected=expected,
            actual=value if exists else "<missing>",
            metadata=self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return redact_mapping(
            prune_none(
                {
                    "operator": self.operator.value,
                    "value": json_safe(self.value),
                    "values": [json_safe(item) for item in self.values],
                    "message": self.message,
                    "code": self.code,
                    "severity": self.severity.value,
                    "case_sensitive": self.case_sensitive,
                    "metadata": self.metadata,
                },
                drop_empty=True,
            )
        )


@dataclass
class FieldRule:
    """Production field rule used by ``TaskContract``.

    The older ``TaskContract.field_types`` map is still supported. ``FieldRule``
    adds field-level defaults, aliases, nullability, allowed values, scalar and
    collection bounds, regex patterns, item type checks, nested object checks,
    and declarative constraints.
    """

    name: str
    types: Tuple[type, ...] = (object,)
    required: bool = False
    nullable: bool = False
    default: Any = _MISSING
    aliases: Tuple[str, ...] = ()
    choices: Tuple[Any, ...] = ()
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[Union[str, Pattern[str]]] = None
    item_types: Tuple[type, ...] = ()
    allow_empty: bool = True
    coerce: bool = False
    transform: Optional[FieldTransform] = None
    constraints: Tuple[FieldConstraint, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = require_non_empty_string(self.name, "field.name")
        self.types = normalize_type_spec(self.types or (object,))
        self.required = coerce_bool(self.required, default=False)
        self.nullable = coerce_bool(self.nullable, default=False)
        self.aliases = tuple(normalize_field_name(item) for item in ensure_list(self.aliases) if normalize_field_name(item))
        self.choices = tuple(self.choices or ())
        self.min_value = None if self.min_value is None else coerce_float(self.min_value)
        self.max_value = None if self.max_value is None else coerce_float(self.max_value)
        self.min_length = None if self.min_length is None else coerce_int(self.min_length, minimum=0)
        self.max_length = None if self.max_length is None else coerce_int(self.max_length, minimum=0)
        self.item_types = normalize_type_spec(self.item_types) if self.item_types else ()
        self.allow_empty = coerce_bool(self.allow_empty, default=True)
        self.coerce = coerce_bool(self.coerce, default=False)
        self.constraints = tuple(
            constraint if isinstance(constraint, FieldConstraint) else FieldConstraint.from_mapping(constraint)
            for constraint in ensure_list(self.constraints)
        )
        self.metadata = normalize_metadata(self.metadata, drop_none=True)

    @classmethod
    def from_mapping(cls, name: str, payload: Mapping[str, Any]) -> "FieldRule":
        data = ensure_mapping(payload, field_name=f"field_rule.{name}")
        constraints = data.get("constraints", data.get("validators", ()))
        if isinstance(constraints, Mapping):
            constraints = [constraints]
        return cls(
            name=str(data.get("name", name)),
            types=normalize_type_spec(data.get("types", data.get("type", "any"))),
            required=coerce_bool(data.get("required"), default=False),
            nullable=coerce_bool(data.get("nullable"), default=False),
            default=data.get("default", _MISSING),
            aliases=tuple(ensure_list(data.get("aliases", ()) or ())),
            choices=tuple(ensure_list(data.get("choices", data.get("allowed_values", ())) or ())),
            min_value=None if data.get("min_value") is None else coerce_float(data.get("min_value")),
            max_value=None if data.get("max_value") is None else coerce_float(data.get("max_value")),
            min_length=None if data.get("min_length") is None else coerce_int(data.get("min_length"), minimum=0),
            max_length=None if data.get("max_length") is None else coerce_int(data.get("max_length"), minimum=0),
            pattern=data.get("pattern"),
            item_types=normalize_type_spec(data.get("item_types", data.get("items_type", ()))) if data.get("item_types", data.get("items_type", None)) is not None else (),
            allow_empty=coerce_bool(data.get("allow_empty"), default=True),
            coerce=coerce_bool(data.get("coerce"), default=False),
            constraints=tuple(
                item if isinstance(item, FieldConstraint) else FieldConstraint.from_mapping(item)
                for item in ensure_list(constraints)
            ),
            metadata=normalize_metadata(data.get("metadata"), drop_none=True),
        )

    def resolve_value(self, payload: Dict[str, Any]) -> Tuple[bool, Any, Optional[str]]:
        """Resolve the field from its canonical name or aliases."""

        if self.name in payload:
            return True, payload[self.name], self.name
        for alias in self.aliases:
            if alias in payload:
                return True, payload[alias], alias
        if self.default is not _MISSING:
            return True, self.default, None
        return False, None, None

    def validate_value(self, payload: Dict[str, Any], *, strict_types: bool = True) -> Tuple[List[ContractValidationIssue], Any, bool]:
        exists, value, source_name = self.resolve_value(payload)
        issues: List[ContractValidationIssue] = []

        if not exists:
            if self.required:
                issues.append(
                    ContractValidationIssue(
                        field=self.name,
                        message=f"missing required field '{self.name}'",
                        code="missing_required_field",
                        expected="present",
                        actual="<missing>",
                    )
                )
            return issues, None, False

        if source_name is not None and source_name != self.name:
            payload[self.name] = value

        if value is None:
            if self.nullable:
                return issues, None, True
            issues.append(
                ContractValidationIssue(
                    field=self.name,
                    message=f"field '{self.name}' cannot be null",
                    code="null_not_allowed",
                    expected=[type_.__name__ for type_ in self.types],
                    actual=None,
                )
            )
            return issues, value, True

        if self.coerce:
            value = coerce_value_for_types(value, self.types)
            payload[self.name] = value

        if strict_types and not _matches_type(value, self.types):
            issues.append(
                ContractValidationIssue(
                    field=self.name,
                    message=f"field '{self.name}' expected [{', '.join(t.__name__ for t in self.types)}] but got [{type(value).__name__}]",
                    code="invalid_type",
                    expected=[type_.__name__ for type_ in self.types],
                    actual=type(value).__name__,
                )
            )
            return issues, value, True

        if not self.allow_empty and value in ("", [], {}, ()): 
            issues.append(
                ContractValidationIssue(
                    field=self.name,
                    message=f"field '{self.name}' cannot be empty",
                    code="empty_not_allowed",
                    expected="non-empty value",
                    actual=value,
                )
            )

        if self.choices and not _value_in(value, self.choices, case_sensitive=True):
            issues.append(
                ContractValidationIssue(
                    field=self.name,
                    message=f"field '{self.name}' must be one of {list(self.choices)}",
                    code="invalid_choice",
                    expected=list(self.choices),
                    actual=value,
                )
            )

        if self.min_value is not None and not _numeric_compare(value, self.min_value, ">="):
            issues.append(ContractValidationIssue(self.name, f"field '{self.name}' must be >= {self.min_value}", "min_value", expected=self.min_value, actual=value))
        if self.max_value is not None and not _numeric_compare(value, self.max_value, "<="):
            issues.append(ContractValidationIssue(self.name, f"field '{self.name}' must be <= {self.max_value}", "max_value", expected=self.max_value, actual=value))
        if self.min_length is not None and _length(value) < self.min_length:
            issues.append(ContractValidationIssue(self.name, f"field '{self.name}' length must be >= {self.min_length}", "min_length", expected=self.min_length, actual=_length(value)))
        if self.max_length is not None and _length(value) > self.max_length:
            issues.append(ContractValidationIssue(self.name, f"field '{self.name}' length must be <= {self.max_length}", "max_length", expected=self.max_length, actual=_length(value)))

        if self.pattern is not None:
            if isinstance(self.pattern, re.Pattern):
                pattern_str = self.pattern.pattern
            else:
                pattern_str = str(self.pattern)
            if re.search(pattern_str, str(value)) is None:
                issues.append(
                    ContractValidationIssue(
                        field=self.name,
                        message=f"field '{self.name}' does not match required pattern",
                        code="pattern_mismatch",
                        expected=pattern_str,
                        actual=value,
                    )
                )

        if self.item_types and isinstance(value, (list, tuple, set, frozenset)):
            for index, item in enumerate(value):
                if not _matches_type(item, self.item_types):
                    issues.append(
                        ContractValidationIssue(
                            field=f"{self.name}[{index}]",
                            message=f"item {index} in field '{self.name}' expected [{', '.join(t.__name__ for t in self.item_types)}] but got [{type(item).__name__}]",
                            code="invalid_item_type",
                            expected=[type_.__name__ for type_ in self.item_types],
                            actual=type(item).__name__,
                        )
                    )

        if self.transform is not None and not issues:
            try:
                payload[self.name] = self.transform(value)
                value = payload[self.name]
            except Exception as exc:
                issues.append(
                    ContractValidationIssue(
                        field=self.name,
                        message=f"field '{self.name}' transform failed: {exc}",
                        code="transform_failed",
                        actual=type(exc).__name__,
                    )
                )

        for constraint in self.constraints:
            issue = constraint.evaluate(self.name, value, exists=True)
            if issue is not None:
                issues.append(issue)

        return issues, value, True

    def to_dict(self) -> Dict[str, Any]:
        return redact_mapping(
            prune_none(
                {
                    "name": self.name,
                    "types": [type_.__name__ for type_ in self.types],
                    "required": self.required,
                    "nullable": self.nullable,
                    "has_default": self.default is not _MISSING,
                    "aliases": list(self.aliases),
                    "choices": [json_safe(item) for item in self.choices],
                    "min_value": self.min_value,
                    "max_value": self.max_value,
                    "min_length": self.min_length,
                    "max_length": self.max_length,
                    "pattern": self.pattern.pattern if isinstance(self.pattern, re.Pattern) else self.pattern,
                    "item_types": [type_.__name__ for type_ in self.item_types],
                    "allow_empty": self.allow_empty,
                    "coerce": self.coerce,
                    "constraints": [constraint.to_dict() for constraint in self.constraints],
                    "metadata": self.metadata,
                },
                drop_empty=True,
            )
        )


@dataclass
class ContractValidationResult:
    """Validation result with backward-compatible ``valid`` and ``errors`` fields."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)
    normalized_payload: Dict[str, Any] = field(default_factory=dict)
    task_type: Optional[str] = None
    contract_version: Optional[Union[str, int, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: generate_correlation_id("contract"))
    validated_at: float = field(default_factory=epoch_seconds)
    validated_at_utc: str = field(default_factory=utc_timestamp)
    duration_ms: float = 0.0

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def to_dict(self, *, include_payload: bool = True, redact: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "issues": list(self.issues),
            "normalized_payload": self.normalized_payload if include_payload else None,
            "task_type": self.task_type,
            "contract_version": self.contract_version,
            "metadata": json_safe(self.metadata),
            "correlation_id": self.correlation_id,
            "validated_at": self.validated_at,
            "validated_at_utc": self.validated_at_utc,
            "duration_ms": self.duration_ms,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }
        payload = prune_none(payload, drop_empty=True)
        return redact_mapping(payload) if redact else payload

    def to_result(self, *, action: str = "task_contract_validation") -> Dict[str, Any]:
        if self.valid:
            return success_result(action=action, message="Task contract validation passed", data=self.to_dict(), correlation_id=self.correlation_id)
        return error_result(
            action=action,
            message="Task contract validation failed",
            error={"errors": self.errors, "warnings": self.warnings, "task_type": self.task_type},
            data=self.to_dict(),
            correlation_id=self.correlation_id,
        )

    def raise_for_errors(self) -> None:
        if self.valid:
            return
        raise _contract_error(
            "Task contract validation failed.",
            context=self.to_dict(include_payload=False),
            severity="medium",
        )


@dataclass
class TaskContract:
    task_type: str
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, tuple[type, ...]] = field(default_factory=dict)
    allow_unknown_fields: bool = True
    validators: List[Validator] = field(default_factory=list)
    field_rules: Dict[str, FieldRule] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)
    coerce_types: bool = False
    strict_types: bool = True
    deprecated: bool = False
    version: Union[str, int, float] = 1
    description: str = ""
    tags: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.task_type = normalize_task_type(self.task_type)
        self.required_fields = [normalize_field_name(item) for item in ensure_list(self.required_fields) if normalize_field_name(item)]
        self.optional_fields = [normalize_field_name(item) for item in ensure_list(self.optional_fields) if normalize_field_name(item)]
        self.field_types = {
            normalize_field_name(key): normalize_type_spec(value)
            for key, value in (self.field_types or {}).items()
            if normalize_field_name(key)
        }
        self.allow_unknown_fields = coerce_bool(self.allow_unknown_fields, default=True)
        self.coerce_types = coerce_bool(self.coerce_types, default=False)
        self.strict_types = coerce_bool(self.strict_types, default=True)
        self.deprecated = coerce_bool(self.deprecated, default=False)
        self.description = str(self.description or "")
        self.tags = normalize_tags(self.tags)
        self.metadata = normalize_metadata(self.metadata, drop_none=True)
        self.examples = [normalize_task_payload(example, allow_none=True) for example in ensure_list(self.examples)]
        self.aliases = {normalize_field_name(k): normalize_field_name(v) for k, v in (self.aliases or {}).items() if normalize_field_name(k) and normalize_field_name(v)}
        self.defaults = dict(self.defaults or {})
        self.field_rules = self._build_field_rules(self.field_rules)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TaskContract":
        data = ensure_mapping(payload, field_name="task_contract")
        field_types = parse_field_types(data.get("field_types", data.get("types", {})))
        fields_payload = data.get("fields", data.get("field_rules", {}))
        field_rules: Dict[str, FieldRule] = {}
        if isinstance(fields_payload, Mapping):
            for field_name, field_data in fields_payload.items():
                field_rules[normalize_field_name(field_name)] = FieldRule.from_mapping(str(field_name), ensure_mapping(field_data, field_name=f"fields.{field_name}"))
        elif isinstance(fields_payload, Sequence) and not isinstance(fields_payload, (str, bytes)):
            for field_data in fields_payload:
                item = ensure_mapping(field_data, field_name="field_rule")
                field_name = require_non_empty_string(item.get("name"), "field.name")
                field_rules[normalize_field_name(field_name)] = FieldRule.from_mapping(field_name, item)

        return cls(
            task_type=require_non_empty_string(data.get("task_type", data.get("name")), "task_type"),
            required_fields=[str(item) for item in ensure_list(data.get("required_fields", []))],
            optional_fields=[str(item) for item in ensure_list(data.get("optional_fields", []))],
            field_types=field_types,
            allow_unknown_fields=coerce_bool(data.get("allow_unknown_fields"), default=True),
            validators=[],
            field_rules=field_rules,
            aliases={str(k): str(v) for k, v in ensure_mapping(data.get("aliases", {}), field_name="aliases", allow_none=True).items()},
            defaults=ensure_mapping(data.get("defaults", {}), field_name="defaults", allow_none=True),
            coerce_types=coerce_bool(data.get("coerce_types"), default=False),
            strict_types=coerce_bool(data.get("strict_types"), default=True),
            deprecated=coerce_bool(data.get("deprecated"), default=False),
            version=data.get("version", 1),
            description=str(data.get("description", "")),
            tags=tuple(ensure_list(data.get("tags", ()) or ())),
            metadata=normalize_metadata(data.get("metadata"), drop_none=True),
            examples=[ensure_mapping(item, field_name="example") for item in ensure_list(data.get("examples", [])) if isinstance(item, Mapping)],
        )

    def _build_field_rules(self, configured_rules: Optional[Mapping[str, Any]]) -> Dict[str, FieldRule]:
        rules: Dict[str, FieldRule] = {}
        for field_name, rule in (configured_rules or {}).items():
            canonical = normalize_field_name(field_name)
            if isinstance(rule, FieldRule):
                rules[canonical] = rule
            elif isinstance(rule, Mapping):
                rules[canonical] = FieldRule.from_mapping(canonical, rule)
        for field_name in self.required_fields:
            if field_name not in rules:
                rules[field_name] = FieldRule(
                    name=field_name,
                    types=self.field_types.get(field_name, (object,)),
                    required=True,
                    default=self.defaults.get(field_name, _MISSING),
                    coerce=self.coerce_types,
                    aliases=tuple(alias for alias, canonical in self.aliases.items() if canonical == field_name),
                )
            else:
                rules[field_name].required = True
        for field_name in self.optional_fields:
            if field_name not in rules:
                rules[field_name] = FieldRule(
                    name=field_name,
                    types=self.field_types.get(field_name, (object,)),
                    required=False,
                    default=self.defaults.get(field_name, _MISSING),
                    coerce=self.coerce_types,
                    aliases=tuple(alias for alias, canonical in self.aliases.items() if canonical == field_name),
                )
        for field_name, types in self.field_types.items():
            if field_name not in rules:
                rules[field_name] = FieldRule(
                    name=field_name,
                    types=types,
                    required=field_name in self.required_fields,
                    default=self.defaults.get(field_name, _MISSING),
                    coerce=self.coerce_types,
                    aliases=tuple(alias for alias, canonical in self.aliases.items() if canonical == field_name),
                )
        for field_name, default in self.defaults.items():
            canonical = normalize_field_name(field_name)
            if canonical not in rules:
                rules[canonical] = FieldRule(name=canonical, default=default, coerce=self.coerce_types)
        return OrderedDict(sorted(rules.items(), key=lambda item: item[0]))

    @property
    def known_fields(self) -> set[str]:
        return set(self.required_fields) | set(self.optional_fields) | set(self.field_types.keys()) | set(self.field_rules.keys())

    def validate(self, payload: Dict[str, Any]) -> ContractValidationResult:
        start_ms = monotonic_ms()
        correlation_id = generate_correlation_id("contract")
        issues: List[ContractValidationIssue] = []
        warnings: List[str] = []

        if payload is None:
            payload_dict: Dict[str, Any] = {}
        else:
            try:
                payload_dict = ensure_mapping(payload, field_name="payload")
            except Exception as exc:
                issue = ContractValidationIssue(
                    field=None,
                    message=f"payload must be a mapping-like object: {exc}",
                    code="invalid_payload_type",
                    expected="mapping",
                    actual=type(payload).__name__,
                )
                return self._build_result(
                    issues=[issue],
                    warnings=warnings,
                    normalized_payload={},
                    started_ms=start_ms,
                    correlation_id=correlation_id,
                )

        normalized_payload = normalize_task_payload(payload_dict, allow_none=True)
        self._apply_aliases(normalized_payload)
        self._apply_defaults(normalized_payload)

        if self.deprecated:
            warnings.append(f"contract for task_type '{self.task_type}' is deprecated")

        for _, rule in self.field_rules.items():
            rule_issues, _, _ = rule.validate_value(normalized_payload, strict_types=self.strict_types)
            issues.extend(rule_issues)

        # Preserve the original field_types behavior even for fields not covered
        # by richer FieldRule instances.
        for field_name, expected_types in self.field_types.items():
            if field_name not in normalized_payload:
                continue
            if self.coerce_types:
                normalized_payload[field_name] = coerce_value_for_types(normalized_payload[field_name], expected_types)
            if self.strict_types and not _matches_type(normalized_payload[field_name], expected_types):
                issues.append(
                    ContractValidationIssue(
                        field=field_name,
                        message=f"field '{field_name}' expected [{', '.join(t.__name__ for t in expected_types)}] but got [{type(normalized_payload[field_name]).__name__}]",
                        code="invalid_type",
                        expected=[type_.__name__ for type_ in expected_types],
                        actual=type(normalized_payload[field_name]).__name__,
                    )
                )

        if not self.allow_unknown_fields and self.known_fields:
            allowed = self.known_fields | set(self.aliases.keys())
            for field_name in normalized_payload.keys():
                if field_name not in allowed:
                    issues.append(
                        ContractValidationIssue(
                            field=field_name,
                            message=f"unknown field '{field_name}' is not allowed",
                            code="unknown_field",
                            expected=sorted(self.known_fields),
                            actual=field_name,
                        )
                    )

        if not any(issue.code == "missing_required_field" for issue in issues):
            for validator in self.validators:
                try:
                    is_valid, message = validator(dict(normalized_payload))
                    if not is_valid:
                        issues.append(
                            ContractValidationIssue(
                                field=None,
                                message=message or f"custom validator failed for task '{self.task_type}'",
                                code="custom_validator_failed",
                            )
                        )
                except Exception as exc:
                    issues.append(
                        ContractValidationIssue(
                            field=None,
                            message=f"custom validator raised {type(exc).__name__}: {exc}",
                            code="custom_validator_exception",
                            actual=type(exc).__name__,
                        )
                    )

        return self._build_result(
            issues=issues,
            warnings=warnings,
            normalized_payload=normalized_payload,
            started_ms=start_ms,
            correlation_id=correlation_id,
        )

    def _apply_aliases(self, payload: Dict[str, Any]) -> None:
        for alias, canonical in self.aliases.items():
            if alias in payload and canonical not in payload:
                payload[canonical] = payload[alias]

    def _apply_defaults(self, payload: Dict[str, Any]) -> None:
        for field_name, default in self.defaults.items():
            canonical = normalize_field_name(field_name)
            payload.setdefault(canonical, default)
        for field_name, rule in self.field_rules.items():
            if field_name not in payload and rule.default is not _MISSING:
                payload[field_name] = rule.default

    def _build_result(
        self,
        *,
        issues: List[ContractValidationIssue],
        warnings: List[str],
        normalized_payload: Dict[str, Any],
        started_ms: float,
        correlation_id: str,
    ) -> ContractValidationResult:
        error_issues = [issue for issue in issues if issue.severity == ContractIssueSeverity.ERROR]
        warning_issues = [issue for issue in issues if issue.severity == ContractIssueSeverity.WARNING]
        errors = [issue.message for issue in error_issues]
        warning_messages = list(warnings) + [issue.message for issue in warning_issues]
        return ContractValidationResult(
            valid=not errors,
            errors=errors,
            warnings=warning_messages,
            issues=[issue.to_dict() for issue in issues],
            normalized_payload=redact_mapping(normalized_payload),
            task_type=self.task_type,
            contract_version=self.version,
            metadata={
                "contract_hash": stable_hash(self.to_dict(redact=False), length=16),
                "known_fields": sorted(self.known_fields),
                "deprecated": self.deprecated,
            },
            correlation_id=correlation_id,
            duration_ms=elapsed_ms(started_ms),
        )

    def validate_or_raise(self, payload: Dict[str, Any]) -> ContractValidationResult:
        result = self.validate(payload)
        result.raise_for_errors()
        return result

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = {
            "task_type": self.task_type,
            "required_fields": list(self.required_fields),
            "optional_fields": list(self.optional_fields),
            "field_types": {
                key: [value.__name__ for value in values]
                for key, values in self.field_types.items()
            },
            "allow_unknown_fields": self.allow_unknown_fields,
            "validators_count": len(self.validators),
            "field_rules": {key: rule.to_dict() for key, rule in self.field_rules.items()},
            "aliases": dict(self.aliases),
            "defaults": json_safe(self.defaults),
            "coerce_types": self.coerce_types,
            "strict_types": self.strict_types,
            "deprecated": self.deprecated,
            "version": self.version,
            "description": self.description,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "examples": self.examples,
        }
        payload = prune_none(payload, drop_empty=True)
        return redact_mapping(payload) if redact else payload


class TaskContractRegistry:
    """Thread-safe registry for task contracts."""

    def __init__(self, shared_memory: Optional[Any] = None, load_configured: Optional[bool] = None):
        self._lock = threading.RLock()
        self._contracts: Dict[str, TaskContract] = {}
        self.shared_memory = shared_memory
        self.config = load_global_config()
        self.contract_config = get_config_section("task_contracts") or {}
        merged_config = {**_DEFAULT_TASK_CONTRACT_CONFIG, **dict(self.contract_config or {})}

        self.enabled = coerce_bool(merged_config.get("enabled"), default=True)
        self.fail_open_without_contract = coerce_bool(merged_config.get("fail_open_without_contract"), default=True)
        self.allow_contract_override = coerce_bool(merged_config.get("allow_contract_override"), default=True)
        self.max_contracts = coerce_int(merged_config.get("max_contracts"), default=1000, minimum=1)
        self.audit_enabled = coerce_bool(merged_config.get("audit_enabled"), default=True)
        self.audit_key = str(merged_config.get("audit_key", "collaboration:task_contract_events"))
        self.audit_max_events = coerce_int(merged_config.get("audit_max_events"), default=1000, minimum=1)
        self.include_payload_snapshot = coerce_bool(merged_config.get("include_payload_snapshot"), default=False)
        self.redact_payload_snapshots = coerce_bool(merged_config.get("redact_payload_snapshots"), default=True)
        self.default_allow_unknown_fields = coerce_bool(merged_config.get("default_allow_unknown_fields"), default=True)
        self.default_coerce_types = coerce_bool(merged_config.get("default_coerce_types"), default=False)
        self.default_strict_types = coerce_bool(merged_config.get("default_strict_types"), default=True)
        self.validation_error_raises = coerce_bool(merged_config.get("validation_error_raises"), default=False)

        should_load = self.enabled and coerce_bool(
            load_configured if load_configured is not None else merged_config.get("load_configured_contracts"),
            default=True,
        )
        if should_load:
            self.load_configured_contracts(merged_config)

        logger.info("Task Contract Registry initialized with %s contract(s)", len(self._contracts))

    def register(self, contract: TaskContract) -> None:
        if not isinstance(contract, TaskContract):
            raise _contract_error(
                "register() requires a TaskContract instance.",
                context={"received_type": type(contract).__name__},
                severity="medium",
            )
        with self._lock:
            if contract.task_type in self._contracts and not self.allow_contract_override:
                raise _contract_error(
                    f"contract for task_type '{contract.task_type}' is already registered.",
                    context={"task_type": contract.task_type},
                    severity="medium",
                )
            if contract.task_type not in self._contracts and len(self._contracts) >= self.max_contracts:
                raise _contract_error(
                    "maximum number of task contracts exceeded.",
                    context={"max_contracts": self.max_contracts, "task_type": contract.task_type},
                    severity="high",
                )
            replaced = contract.task_type in self._contracts
            self._contracts[contract.task_type] = contract
            self._record_event(
                "contract_replaced" if replaced else "contract_registered",
                f"Task contract {'replaced' if replaced else 'registered'} for '{contract.task_type}'.",
                metadata={"task_type": contract.task_type, "version": contract.version, "contract": contract.to_dict()},
            )

    def register_contract(
        self,
        task_type: str,
        required_fields: Optional[List[str]] = None,
        optional_fields: Optional[List[str]] = None,
        field_types: Optional[Dict[str, tuple[type, ...]]] = None,
        allow_unknown_fields: bool = True,
        validators: Optional[List[Validator]] = None,
        **kwargs: Any,
    ) -> TaskContract:
        contract = TaskContract(
            task_type=task_type,
            required_fields=required_fields or [],
            optional_fields=optional_fields or [],
            field_types=field_types or {},
            allow_unknown_fields=allow_unknown_fields,
            validators=validators or [],
            field_rules=kwargs.get("field_rules", {}),
            aliases=kwargs.get("aliases", {}),
            defaults=kwargs.get("defaults", {}),
            coerce_types=kwargs.get("coerce_types", self.default_coerce_types),
            strict_types=kwargs.get("strict_types", self.default_strict_types),
            deprecated=kwargs.get("deprecated", False),
            version=kwargs.get("version", 1),
            description=kwargs.get("description", ""),
            tags=tuple(kwargs.get("tags", ()) or ()),
            metadata=normalize_metadata(kwargs.get("metadata"), drop_none=True),
            examples=list(kwargs.get("examples", []) or []),
        )
        self.register(contract)
        return contract

    def load_configured_contracts(self, config: Optional[Mapping[str, Any]] = None) -> int:
        source = dict(config or self.contract_config or {})
        raw_contracts = source.get("configured_contracts", source.get("contracts", []))
        loaded = 0
        if isinstance(raw_contracts, Mapping):
            iterable = []
            for task_type, payload in raw_contracts.items():
                data = ensure_mapping(payload, field_name=f"contract.{task_type}")
                data.setdefault("task_type", task_type)
                iterable.append(data)
        else:
            iterable = ensure_list(raw_contracts)
        for item in iterable:
            try:
                contract = TaskContract.from_mapping(ensure_mapping(item, field_name="configured_contract"))
                if "allow_unknown_fields" not in item:
                    contract.allow_unknown_fields = self.default_allow_unknown_fields
                if "coerce_types" not in item:
                    contract.coerce_types = self.default_coerce_types
                if "strict_types" not in item:
                    contract.strict_types = self.default_strict_types
                self.register(contract)
                loaded += 1
            except Exception as exc:
                self._record_event(
                    "contract_config_failed",
                    "Failed to load configured task contract.",
                    severity="error",
                    error=exc,
                    metadata={"contract": json_safe(item)},
                )
                if not self.fail_open_without_contract:
                    raise _contract_error(
                        "Failed to load configured task contract.",
                        context={"contract": json_safe(item)},
                        cause=exc,
                        severity="high",
                    ) from exc
        return loaded

    def get(self, task_type: str) -> Optional[TaskContract]:
        return self._contracts.get(normalize_task_type(task_type))

    def has_contract(self, task_type: str) -> bool:
        return self.get(task_type) is not None

    def validate(self, task_type: str, payload: Dict[str, Any]) -> ContractValidationResult:
        start_ms = monotonic_ms()
        normalized_task_type = normalize_task_type(task_type)
        if not self.enabled:
            return ContractValidationResult(
                valid=True,
                warnings=["task contract validation is disabled"],
                normalized_payload=normalize_task_payload(payload, allow_none=True, redact=True),
                task_type=normalized_task_type,
                metadata={"validation_disabled": True},
                duration_ms=elapsed_ms(start_ms),
            )

        contract = self.get(normalized_task_type)
        if contract is None:
            valid = self.fail_open_without_contract
            message = f"no task contract registered for task_type '{normalized_task_type}'"
            result = ContractValidationResult(
                valid=valid,
                errors=[] if valid else [message],
                warnings=[message] if valid else [],
                normalized_payload=normalize_task_payload(payload, allow_none=True, redact=True),
                task_type=normalized_task_type,
                metadata={"contract_missing": True, "fail_open_without_contract": self.fail_open_without_contract},
                duration_ms=elapsed_ms(start_ms),
            )
            self._record_validation(normalized_task_type, result, payload)
            if self.validation_error_raises and not result.valid:
                result.raise_for_errors()
            return result

        result = contract.validate(payload)
        self._record_validation(normalized_task_type, result, payload)
        if self.validation_error_raises and not result.valid:
            result.raise_for_errors()
        return result

    def validate_or_raise(self, task_type: str, payload: Dict[str, Any]) -> ContractValidationResult:
        result = self.validate(task_type, payload)
        result.raise_for_errors()
        return result

    def validate_envelope(self, envelope: Any) -> ContractValidationResult:
        if hasattr(envelope, "task_type") and hasattr(envelope, "payload"):
            return self.validate(str(envelope.task_type), dict(envelope.payload))
        data = ensure_mapping(envelope, field_name="task_envelope")
        task_type = data.get("task_type")
        payload = data.get("payload", data.get("task_data", data))
        return self.validate(str(task_type), ensure_mapping(payload, field_name="payload", allow_none=True))

    def unregister(self, task_type: str) -> bool:
        normalized = normalize_task_type(task_type)
        with self._lock:
            existed = normalized in self._contracts
            if existed:
                del self._contracts[normalized]
                self._record_event("contract_unregistered", f"Task contract unregistered for '{normalized}'.", metadata={"task_type": normalized})
            return existed

    def clear(self) -> None:
        with self._lock:
            count = len(self._contracts)
            self._contracts.clear()
            self._record_event("contracts_cleared", "Task contract registry cleared.", metadata={"count": count})

    def list_contracts(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {task_type: contract.to_dict() for task_type, contract in self._contracts.items()}

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            tags: Dict[str, int] = {}
            for contract in self._contracts.values():
                for tag in contract.tags:
                    tags[tag] = tags.get(tag, 0) + 1
            return redact_mapping(
                {
                    "enabled": self.enabled,
                    "contract_count": len(self._contracts),
                    "task_types": sorted(self._contracts.keys()),
                    "tags": tags,
                    "audit_enabled": self.audit_enabled,
                    "fail_open_without_contract": self.fail_open_without_contract,
                    "default_allow_unknown_fields": self.default_allow_unknown_fields,
                    "default_coerce_types": self.default_coerce_types,
                    "default_strict_types": self.default_strict_types,
                }
            )

    def export_contracts(self, *, redact: bool = True) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "contracts": self.list_contracts() if redact else {task_type: contract.to_dict(redact=False) for task_type, contract in self._contracts.items()},
            "exported_at": epoch_seconds(),
            "exported_at_utc": utc_timestamp(),
        }

    def evaluate_result(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate(task_type, payload).to_result()

    def _record_validation(self, task_type: str, result: ContractValidationResult, original_payload: Any) -> None:
        if not self.audit_enabled:
            return
        metadata: Dict[str, Any] = {
            "task_type": task_type,
            "valid": result.valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "duration_ms": result.duration_ms,
        }
        if self.include_payload_snapshot:
            snapshot = json_safe(original_payload)
            metadata["payload"] = redact_mapping(snapshot) if self.redact_payload_snapshots and isinstance(snapshot, Mapping) else snapshot
        self._record_event(
            "contract_validation_passed" if result.valid else "contract_validation_failed",
            f"Task contract validation {'passed' if result.valid else 'failed'} for '{task_type}'.",
            severity="info" if result.valid else "warning",
            metadata=metadata,
            correlation_id=result.correlation_id,
        )

    def _record_event(
        self,
        event_type: str,
        message: str,
        *,
        severity: str = "info",
        error: Optional[BaseException] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        event = build_audit_event(
            event_type,
            message,
            severity=severity,
            component="task_contracts",
            correlation_id=correlation_id,
            error=error,
            metadata=metadata,
        )
        if self.shared_memory is not None:
            append_audit_event(self.shared_memory, event, key=self.audit_key, max_events=self.audit_max_events)
        return event


# ---------------------------------------------------------------------------
# Validator factories
# ---------------------------------------------------------------------------
def make_required_one_of_validator(fields: Sequence[str], *, message: Optional[str] = None) -> Validator:
    normalized_fields = [normalize_field_name(field_name) for field_name in fields]

    def _validator(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if any(field_name in payload and payload.get(field_name) not in (None, "", [], {}) for field_name in normalized_fields):
            return True, None
        return False, message or f"at least one of {normalized_fields} is required"

    return _validator


def make_mutually_exclusive_validator(fields: Sequence[str], *, message: Optional[str] = None) -> Validator:
    normalized_fields = [normalize_field_name(field_name) for field_name in fields]

    def _validator(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        present = [field_name for field_name in normalized_fields if field_name in payload and payload.get(field_name) not in (None, "", [], {})]
        if len(present) <= 1:
            return True, None
        return False, message or f"fields {present} are mutually exclusive"

    return _validator


def make_numeric_range_validator(field_name: str, *, minimum: Optional[float] = None, maximum: Optional[float] = None) -> Validator:
    field = normalize_field_name(field_name)

    def _validator(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if field not in payload:
            return True, None
        try:
            value = float(payload[field])
        except Exception:
            return False, f"field '{field}' must be numeric"
        if minimum is not None and value < minimum:
            return False, f"field '{field}' must be >= {minimum}"
        if maximum is not None and value > maximum:
            return False, f"field '{field}' must be <= {maximum}"
        return True, None

    return _validator


def make_regex_validator(field_name: str, pattern: str, *, message: Optional[str] = None, flags: int = 0) -> Validator:
    field = normalize_field_name(field_name)
    compiled = re.compile(pattern, flags=flags)

    def _validator(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if field not in payload:
            return True, None
        if compiled.search(str(payload[field])):
            return True, None
        return False, message or f"field '{field}' does not match required pattern"

    return _validator


def make_allowed_values_validator(field_name: str, values: Sequence[Any], *, case_sensitive: bool = True) -> Validator:
    field = normalize_field_name(field_name)
    allowed = tuple(values)

    def _validator(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        if field not in payload:
            return True, None
        if _value_in(payload[field], allowed, case_sensitive=case_sensitive):
            return True, None
        return False, f"field '{field}' must be one of {list(allowed)}"

    return _validator


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def normalize_field_name(value: Any) -> str:
    return str(value or "").strip()


def normalize_issue_severity(value: Any) -> ContractIssueSeverity:
    if isinstance(value, ContractIssueSeverity):
        return value
    lowered = str(value or ContractIssueSeverity.ERROR.value).strip().lower()
    for severity in ContractIssueSeverity:
        if lowered == severity.value:
            return severity
    return ContractIssueSeverity.ERROR


def normalize_constraint_operator(value: Any) -> ContractFieldOperator:
    if isinstance(value, ContractFieldOperator):
        return value
    lowered = str(value or ContractFieldOperator.EXISTS.value).strip().lower()
    aliases = {
        "equals": ContractFieldOperator.EQ,
        "not_equals": ContractFieldOperator.NE,
        "min": ContractFieldOperator.GTE,
        "max": ContractFieldOperator.LTE,
        "not in": ContractFieldOperator.NOT_IN,
        "not_contains": ContractFieldOperator.NOT_CONTAINS,
        "length_gte": ContractFieldOperator.LENGTH_MIN,
        "length_lte": ContractFieldOperator.LENGTH_MAX,
    }
    if lowered in aliases:
        return aliases[lowered]
    for operator in ContractFieldOperator:
        if lowered == operator.value:
            return operator
    raise _contract_error("Unsupported field constraint operator.", context={"operator": value}, severity="medium")


def parse_field_types(field_types: Any) -> Dict[str, Tuple[type, ...]]:
    if field_types is None:
        return {}
    source = ensure_mapping(field_types, field_name="field_types", allow_none=True)
    return {normalize_field_name(key): normalize_type_spec(value) for key, value in source.items() if normalize_field_name(key)}


def normalize_type_spec(value: Any) -> Tuple[type, ...]:
    if value is None or value == ():
        return (object,)
    if isinstance(value, type):
        return (value,)
    if isinstance(value, tuple) and all(isinstance(item, type) for item in value):
        return value
    if isinstance(value, list) and all(isinstance(item, type) for item in value):
        return tuple(value)
    if isinstance(value, str):
        parts = [part.strip().lower() for part in re.split(r"[,|]", value) if part.strip()]
    elif isinstance(value, Iterable):
        parts = list(value)
    else:
        parts = [value]

    types: List[type] = []
    for item in parts:
        if isinstance(item, type):
            types.append(item)
            continue
        key = str(item).strip().lower()
        if key not in _TYPE_ALIASES:
            raise _contract_error("Unsupported field type specifier.", context={"type_specifier": item}, severity="medium")
        types.extend(_TYPE_ALIASES[key])
    deduped: List[type] = []
    for type_ in types:
        if type_ not in deduped:
            deduped.append(type_)
    return tuple(deduped or (object,))


def coerce_value_for_types(value: Any, types: Tuple[type, ...]) -> Any:
    if value is None:
        return None
    if _matches_type(value, types):
        return value
    if str in types:
        return str(value)
    if bool in types:
        return coerce_bool(value, default=False)
    if int in types and not isinstance(value, bool):
        try:
            return int(float(value))
        except Exception:
            return value
    if float in types:
        try:
            return float(value)
        except Exception:
            return value
    if dict in types and isinstance(value, str):
        loaded = json_loads(value, default=value)
        return loaded
    if list in types:
        return ensure_list(value)
    return value


def _matches_type(value: Any, types: Tuple[type, ...]) -> bool:
    if object in types:
        return True
    if bool not in types and isinstance(value, bool) and (int in types or float in types):
        return False
    return isinstance(value, types)


def _compare_values(left: Any, right: Any, *, case_sensitive: bool = True) -> int:
    if isinstance(left, str) or isinstance(right, str):
        l_text = str(left)
        r_text = str(right)
        if not case_sensitive:
            l_text = l_text.lower()
            r_text = r_text.lower()
        return 0 if l_text == r_text else -1
    return 0 if left == right else -1


def _numeric_compare(left: Any, right: Any, op: str) -> bool:
    try:
        lhs = float(left)
        rhs = float(right)
    except Exception:
        return False
    if op == ">":
        return lhs > rhs
    if op == ">=":
        return lhs >= rhs
    if op == "<":
        return lhs < rhs
    if op == "<=":
        return lhs <= rhs
    return False


def _value_in(value: Any, values: Iterable[Any], *, case_sensitive: bool = True) -> bool:
    if case_sensitive:
        return value in tuple(values)
    value_text = str(value).lower()
    return value_text in {str(item).lower() for item in values}


def _contains(container: Any, needle: Any, *, case_sensitive: bool = True) -> bool:
    if isinstance(container, Mapping):
        return needle in container
    if isinstance(container, (list, tuple, set, frozenset)):
        return needle in container
    haystack = str(container)
    needle_text = str(needle)
    if not case_sensitive:
        haystack = haystack.lower()
        needle_text = needle_text.lower()
    return needle_text in haystack


def _length(value: Any) -> int:
    try:
        return len(value)
    except Exception:
        return len(str(value))


def _contract_error(
    message: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
    cause: Optional[BaseException] = None,
    severity: str = "medium",
) -> CollaborationError: # type: ignore
    return CollaborationError(
        CollaborationErrorType.DELEGATION_FAILURE, # type: ignore
        message,
        severity=severity,
        context=normalize_metadata(context, drop_none=True),
        remediation_guidance="Review task contract definitions, configured field rules, and payload schema compatibility.",
        cause=cause,
        retryable=False,
        component="task_contracts",
    ) # type: ignore


if __name__ == "__main__":
    print("\n=== Running Task Contracts ===\n")
    printer.status("TEST", "Task Contracts initialized", "info")

    class _Memory:
        def __init__(self):
            self.store: Dict[str, Any] = {}

        def get(self, key: str, default: Any = None) -> Any:
            return self.store.get(key, default)

        def set(self, key: str, value: Any, **kwargs: Any) -> None:
            self.store[key] = value

        def append(self, key: str, value: Any, **kwargs: Any) -> None:
            current = self.store.get(key, [])
            if not isinstance(current, list):
                current = [current]
            current.append(value)
            self.store[key] = current

    memory = _Memory()
    registry = TaskContractRegistry(shared_memory=memory, load_configured=False)

    contract = registry.register_contract(
        "TranslateAndSummarize",
        required_fields=["text", "target_language"],
        optional_fields=["summary_length", "risk_score"],
        field_types={"text": (str,), "target_language": (str,), "summary_length": (int,), "risk_score": (int, float)},
        allow_unknown_fields=False,
        validators=[make_numeric_range_validator("risk_score", minimum=0.0, maximum=1.0)],
        field_rules={
            "text": FieldRule(name="text", types=(str,), required=True, min_length=1, max_length=5000, allow_empty=False),
            "target_language": FieldRule(name="target_language", types=(str,), required=True, choices=("en", "es", "fr", "de")),
            "summary_length": FieldRule(name="summary_length", types=(int,), required=False, default=5, min_value=1, max_value=20, coerce=True),
        },
        aliases={"language": "target_language"},
        defaults={"summary_length": 5},
        coerce_types=True,
        version="1.0.0",
        description="Translate text and produce a bounded summary.",
        tags=("translation", "summarization"),
    )

    assert contract.task_type == "TranslateAndSummarize"
    assert registry.has_contract("TranslateAndSummarize") is True

    valid_payload = {"text": "Hello world", "language": "es", "summary_length": "3", "risk_score": 0.2}
    result = registry.validate("TranslateAndSummarize", valid_payload)
    assert result.valid, result.to_dict()
    assert result.normalized_payload["target_language"] == "es"
    assert result.normalized_payload["summary_length"] == 3

    invalid_payload = {"text": "", "target_language": "it", "risk_score": 1.5, "unexpected": True}
    invalid = registry.validate("TranslateAndSummarize", invalid_payload)
    assert invalid.valid is False
    assert any("unknown field" in error for error in invalid.errors), invalid.errors
    assert any("target_language" in error for error in invalid.errors), invalid.errors

    missing = registry.validate("UnknownTask", {"payload": True})
    assert missing.valid is True
    assert missing.warnings

    one_of_contract = registry.register_contract(
        "Lookup",
        required_fields=[],
        allow_unknown_fields=True,
        validators=[make_required_one_of_validator(["query", "id"]), make_mutually_exclusive_validator(["query", "id"])],
    )
    assert one_of_contract.validate({"query": "abc"}).valid is True
    assert one_of_contract.validate({}).valid is False
    assert one_of_contract.validate({"query": "abc", "id": "123"}).valid is False

    config_contract = TaskContract.from_mapping(
        {
            "task_type": "AnalyzeData",
            "allow_unknown_fields": False,
            "fields": {
                "dataset": {"type": "dict", "required": True},
                "mode": {"type": "str", "required": False, "default": "fast", "choices": ["fast", "full"]},
            },
        }
    )
    registry.register(config_contract)
    config_result = registry.validate("AnalyzeData", {"dataset": {"rows": 3}})
    assert config_result.valid is True
    assert config_result.normalized_payload["mode"] == "fast"

    contracts = registry.list_contracts()
    assert "TranslateAndSummarize" in contracts and "AnalyzeData" in contracts

    summary = registry.summary()
    assert summary["contract_count"] >= 3

    exported = registry.export_contracts()
    assert "contracts" in exported and "summary" in exported

    assert registry.unregister("Lookup") is True
    assert registry.has_contract("Lookup") is False

    audit_events = memory.get("collaboration:task_contract_events", [])
    assert isinstance(audit_events, list) and audit_events

    printer.status("TEST", "Task contract registration, validation, config loading, audit, and lifecycle checks passed", "success")
    print("\n=== Test ran successfully ===\n")
