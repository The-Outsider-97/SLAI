"""Safe, bounded, declarative requirement assessment (NO eval/exec).

Rule data describes an applicability expression ("when") and one assertion
("assert"). This is a conservative v1 checker, NOT a general legal-compliance
engine, optimizer, arbitrary program runner, or substitute for VerificationAgent.
Domain-specific methods must use explicitly reviewed external adapters.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .contracts import (AccessContext, Lifecycle, MissingInput, RecordDraft, RecordRevision,
                        RuleEvaluation, RuleOutcome, Scope, UnsupportedEvaluation,
                        ValidationFailure, digest_json, utc_now)
from .repository import KnowledgeRepository


@dataclass(frozen=True, slots=True)
class UnitDefinition:
    dimension: str
    to_base: Decimal


DEFAULT_UNITS: dict[str, UnitDefinition] = {
    "1": UnitDefinition("dimensionless", Decimal("1")),
    "m": UnitDefinition("length", Decimal("1")),
    "cm": UnitDefinition("length", Decimal("0.01")),
    "mm": UnitDefinition("length", Decimal("0.001")),
    "km": UnitDefinition("length", Decimal("1000")),
    "s": UnitDefinition("time", Decimal("1")),
    "min": UnitDefinition("time", Decimal("60")),
    "h": UnitDefinition("time", Decimal("3600")),
    "kg": UnitDefinition("mass", Decimal("1")),
    "g": UnitDefinition("mass", Decimal("0.001")),
    "rad": UnitDefinition("angle", Decimal("1")),
}


@dataclass(frozen=True, slots=True)
class Number:
    amount: Decimal
    dimension: str


def _decimal(x: Any) -> Decimal:
    if isinstance(x, bool) or not isinstance(x, (int, float, Decimal, str)):
        raise ValueError("Expected finite numerical quantity")
    try:
        output = Decimal(str(x))
    except InvalidOperation as exc:
        raise ValueError("Malformed numerical quantity") from exc
    if not output.is_finite() or len(str(output)) > 256:
        raise ValueError("Non-finite or oversized quantity")
    return output


class DeclarativeRuleEvaluator:
    """Deterministic interpreter limited by AST size and depth (data-only rules)."""

    VERSION = "slai.rule_eval.v1"

    def __init__(self, units: Mapping[str, UnitDefinition] | None = None,
                 max_nodes: int = 128, max_depth: int = 16):
        self.units = dict(DEFAULT_UNITS)
        if units:
            for name, spec in units.items():
                if not isinstance(name, str) or not isinstance(spec, UnitDefinition):
                    raise ValidationFailure("Unit extension must use UnitDefinition")
                if not spec.to_base.is_finite() or spec.to_base <= 0:
                    raise ValidationFailure("Unit scale must be finite and positive")
                self.units[name] = spec
        self.max_nodes = max_nodes
        self.max_depth = max_depth

    def _quantity(self, item: Any, default_unit: str = "1") -> Any:
        if isinstance(item, Mapping) and "value" in item:
            unit = item.get("unit", default_unit)
            value = item["value"]
        else:
            unit, value = default_unit, item
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and unit == "text":
            if len(value) > 2000:
                raise ValueError("String exceeds size limit")
            return value
        if isinstance(value, str) and unit == "1":
            try:
                return Number(_decimal(value), "dimensionless")
            except ValueError:
                if len(value) > 2000:
                    raise ValueError("String exceeds size limit")
                return value
        if unit not in self.units:
            raise UnsupportedEvaluation(f"Unit {unit!r} is not registered")
        spec = self.units[unit]
        return Number(_decimal(value) * spec.to_base, spec.dimension)

    def _term(self, expression: Any, inputs: Mapping[str, Any],
              input_units: Mapping[str, str]) -> Any:
        if not isinstance(expression, Mapping):
            raise UnsupportedEvaluation("Operands must be explicit {field} or {const} objects")
        if set(expression) == {"field"}:
            name = expression["field"]
            if not isinstance(name, str) or len(name) > 128 or not name:
                raise ValueError("Invalid field identifier")
            if name not in inputs or inputs[name] is None:
                raise MissingInput({name})
            return self._quantity(inputs[name], input_units.get(name, "1"))
        if "const" in expression and set(expression).issubset({"const", "unit"}):
            return self._quantity({"value": expression["const"], "unit": expression.get("unit", "1")})
        raise UnsupportedEvaluation("Unsupported term form")

    def _compare(self, operation: str, left: Any, right: Any) -> bool:
        if isinstance(left, Number) and isinstance(right, Number):
            if left.dimension != right.dimension:
                raise ValueError("Dimension mismatch")
            lhs, rhs = left.amount, right.amount
        elif isinstance(left, str) and isinstance(right, str) and operation in {"eq", "ne"}:
            lhs, rhs = left, right
        elif isinstance(left, bool) and isinstance(right, bool) and operation in {"eq", "ne"}:
            lhs, rhs = left, right
        else:
            raise ValueError("Incompatible compared types")
        return {"eq": lambda: lhs == rhs, "ne": lambda: lhs != rhs,
                "lt": lambda: lhs < rhs, "le": lambda: lhs <= rhs,
                "gt": lambda: lhs > rhs, "ge": lambda: lhs >= rhs}[operation]()

    def _eval(self, expr: Any, inputs: Mapping[str, Any],
              units: Mapping[str, str], depth: int, counter: list[int]) -> bool:
        if depth > self.max_depth:
            raise UnsupportedEvaluation("Rule expression nesting limit exceeded")
        counter[0] += 1
        if counter[0] > self.max_nodes:
            raise UnsupportedEvaluation("Rule expression node budget exceeded")
        if not isinstance(expr, dict) or set(expr) - {"op", "args", "left", "right", "arg", "field"}:
            raise UnsupportedEvaluation("Rule expression must be a recognized data object")
        op = expr.get("op")
        if op == "exists":
            if set(expr) != {"op", "field"} or not isinstance(expr["field"], str):
                raise UnsupportedEvaluation("exists requires a field")
            return expr["field"] in inputs and inputs[expr["field"]] is not None
        if op in {"all", "any"}:
            if set(expr) != {"op", "args"} or not isinstance(expr["args"], list) or not expr["args"]:
                raise UnsupportedEvaluation("all/any requires a nonempty list")
            missing: set[str] = set()
            outcomes = []
            for child in expr["args"]:
                try:
                    outcomes.append(self._eval(child, inputs, units, depth + 1, counter))
                except MissingInput as exc:
                    missing.update(exc.fields)
            if op == "all" and False in outcomes:
                return False
            if op == "any" and True in outcomes:
                return True
            if missing:
                raise MissingInput(missing)
            return all(outcomes) if op == "all" else any(outcomes)
        if op == "not":
            if set(expr) != {"op", "arg"}:
                raise UnsupportedEvaluation("not requires arg")
            return not self._eval(expr["arg"], inputs, units, depth + 1, counter)
        if op in {"eq", "ne", "lt", "le", "gt", "ge"}:
            if set(expr) != {"op", "left", "right"}:
                raise UnsupportedEvaluation("Comparison requires left/right")
            missing: set[str] = set()
            args = []
            for item in (expr["left"], expr["right"]):
                try:
                    args.append(self._term(item, inputs, units))
                except MissingInput as exc:
                    missing.update(exc.fields)
            if missing:
                raise MissingInput(missing)
            return self._compare(op, args[0], args[1])
        raise UnsupportedEvaluation(f"Unsupported operation: {op!r}")

    def validate_rule(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(spec, dict):
            raise ValidationFailure("Rule specification must be an object")
        if spec.get("schema") != self.VERSION:
            raise ValidationFailure("Unsupported declarative-rule schema")
        if "assert" not in spec:
            raise ValidationFailure("Rule assertion is required")
        if set(spec) - {"schema", "when", "assert", "input_units", "description", "modality"}:
            raise ValidationFailure("Unknown rule specification fields")
        if not isinstance(spec.get("input_units", {}), dict):
            raise ValidationFailure("input_units must be an object")
        for name, unit in spec.get("input_units", {}).items():
            if not isinstance(name, str) or not name or unit not in self.units and unit != "text":
                raise ValidationFailure("Unknown input field or unit")
        # Validate structural expression restrictions WITHOUT evaluating user data.
        self._validate_structure(spec["assert"])
        if "when" in spec:
            self._validate_structure(spec["when"])
        return dict(spec)

    def _validate_structure(self, expr: Any) -> None:
        """A finite symbolic dummy environment surfaces unrecognized operators."""
        count = [0]
        def walk(node: Any, depth: int) -> None:
            count[0] += 1
            if depth > self.max_depth or count[0] > self.max_nodes:
                raise ValidationFailure("Expression complexity budget exceeded")
            if not isinstance(node, dict) or "op" not in node:
                raise ValidationFailure("Every expression requires an operator")
            op = node["op"]
            if op in {"all", "any"}:
                if set(node) != {"op", "args"} or not isinstance(node["args"], list) or not node["args"]:
                    raise ValidationFailure("Malformed logical expression")
                for c in node["args"]:
                    walk(c, depth + 1)
            elif op == "not":
                if set(node) != {"op", "arg"}:
                    raise ValidationFailure("Malformed negation")
                walk(node["arg"], depth + 1)
            elif op == "exists":
                if set(node) != {"op", "field"} or not isinstance(node["field"], str):
                    raise ValidationFailure("Malformed existence test")
            elif op in {"eq", "ne", "lt", "le", "gt", "ge"}:
                if set(node) != {"op", "left", "right"}:
                    raise ValidationFailure("Malformed comparison")
                for term in (node["left"], node["right"]):
                    if not isinstance(term, dict) or not (set(term) == {"field"} or
                            "const" in term and set(term).issubset({"const", "unit"})):
                        raise ValidationFailure("Malformed comparison operand")
                    if "field" in term and (not isinstance(term["field"], str) or not term["field"]):
                        raise ValidationFailure("Malformed field identifier")
            else:
                raise ValidationFailure(f"Unsupported operator: {op!r}")
        walk(expr, 0)

    def evaluate(self, *, rule_id: str, rule_revision: int, spec: Mapping[str, Any],
                 inputs: Mapping[str, Any]) -> RuleEvaluation:
        now = utc_now()
        def outcome(status: RuleOutcome, msg: str, missing: tuple[str, ...] = ()) -> RuleEvaluation:
            try:
                input_digest = digest_json(dict(inputs))
            except (TypeError, ValueError, OverflowError):
                # Malformed inputs are never serialized or echoed to clients.
                input_digest = ""
            return RuleEvaluation(rule_id=rule_id, rule_revision=rule_revision,
                                  outcome=status, evaluator=self.VERSION,
                                  message=msg, missing_inputs=missing,
                                  input_digest=input_digest, evaluated_at=now)
        try:
            self.validate_rule(spec)
            if not isinstance(inputs, dict):
                raise ValueError("Inputs must be a finite JSON-compatible object")
            units = spec.get("input_units", {})
            if "when" in spec:
                if not self._eval(spec["when"], inputs, units, 0, [0]):
                    return outcome(RuleOutcome.NOT_APPLICABLE, "Rule applicability condition is false")
            result = self._eval(spec["assert"], inputs, units, 0, [0])
            if result:
                return outcome(RuleOutcome.SATISFIED, "Declarative predicate is satisfied")
            return outcome(RuleOutcome.VIOLATED, "Declarative predicate is false")
        except MissingInput as exc:
            return outcome(RuleOutcome.UNDETERMINED,
                           "Required input missing; no compliance conclusion", tuple(sorted(exc.fields)))
        except (UnsupportedEvaluation, ValidationFailure) as exc:
            return outcome(RuleOutcome.UNSUPPORTED, str(exc))
        except (ValueError, TypeError, ArithmeticError, OverflowError, RecursionError) as exc:
            return outcome(RuleOutcome.EVALUATION_ERROR, str(exc))


class RulesRegistry:
    """Store reviewed declarative rules, link sources and persist evaluations."""

    def __init__(self, repository: KnowledgeRepository, evaluator: DeclarativeRuleEvaluator | None = None):
        self.repository = repository
        self.evaluator = evaluator or DeclarativeRuleEvaluator()

    def register_rule(self, *, spec: dict[str, Any], access: AccessContext,
                      source_requirement_id: str | None = None,
                      scope: Scope = Scope.GLOBAL,
                      project_id: str | None = None,
                      organization_id: str | None = None,
                      record_id: str | None = None) -> RecordRevision:
        self.evaluator.validate_rule(spec)
        if source_requirement_id:
            requirement = self.repository.get(source_requirement_id, access)
            if requirement is None or requirement.draft.kind != "requirement":
                raise ValidationFailure("Accessible source requirement record is required")
        else:
            requirement = None
        draft = RecordDraft(kind="design_rule", scope=scope, project_id=project_id,
                            organization_id=organization_id, payload={"rule_spec": spec,
                             "source_requirement_id": source_requirement_id,
                             "review_status": "candidate"},
                            source_ids=requirement.draft.source_ids if requirement else (),
                            status=Lifecycle.DRAFT)
        record = self.repository.create(draft, access, record_id=record_id,
                                        reason="declarative rule candidate registered")
        if source_requirement_id:
            self.repository.link(record.record_id, source_requirement_id, access)
        return record

    def evaluate(self, record_id: str, *, inputs: dict[str, Any],
                 access: AccessContext | None) -> RuleEvaluation:
        record = self.repository.get(record_id, access)
        if record is None or record.draft.kind != "design_rule":
            raise ValidationFailure("Accessible design rule required")
        if record.draft.status not in {Lifecycle.ACTIVE, Lifecycle.REVIEWED}:
            return RuleEvaluation(rule_id=record_id, rule_revision=record.revision,
                                  outcome=RuleOutcome.UNSUPPORTED,
                                  evaluator=self.evaluator.VERSION,
                                  message="Rule has not been reviewed/activated")
        return self.evaluator.evaluate(rule_id=record_id, rule_revision=record.revision,
                                       spec=record.draft.payload["rule_spec"], inputs=inputs)

    def record_evaluation(self, evaluation: RuleEvaluation, access: AccessContext,
                          *, record_id: str | None = None) -> RecordRevision:
        rule = self.repository.get(evaluation.rule_id, access)
        if rule is None or rule.draft.kind != "design_rule":
            raise ValidationFailure("Rule not accessible")
        if evaluation.rule_revision != rule.revision:
            raise ValidationFailure("Evaluation was made against a stale rule revision")
        draft = RecordDraft(kind="verification_outcome", scope=rule.draft.scope,
                            project_id=rule.draft.project_id,
                            organization_id=rule.draft.organization_id,
                            status=Lifecycle.DRAFT,
                            payload={"rule_id": evaluation.rule_id,
                                     "rule_revision": evaluation.rule_revision,
                                     "outcome": evaluation.outcome.value,
                                     "evaluator": evaluation.evaluator,
                                     "message": evaluation.message,
                                     "missing_inputs": list(evaluation.missing_inputs),
                                     "input_digest": evaluation.input_digest,
                                     "evaluated_at": evaluation.evaluated_at,
                                     "compliance_certification": False},
                            source_ids=rule.draft.source_ids)
        result = self.repository.create(draft, access, record_id=record_id,
                                        reason="scope-bounded declarative rule assessment")
        self.repository.link(result.record_id, evaluation.rule_id, access)
        return result
