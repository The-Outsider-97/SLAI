"""Backend-neutral SAT/SMT terms, requests, capabilities, and responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence, runtime_checkable

from ..utils.verification_errors import MalformedSpecificationError


class SortKind(str, Enum):
    BOOL = "bool"
    INT = "int"
    REAL = "real"


@dataclass(frozen=True, slots=True)
class Sort:
    kind: SortKind

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SortKind):
            try:
                object.__setattr__(self, "kind", SortKind(self.kind))
            except (TypeError, ValueError) as exc:
                raise MalformedSpecificationError("invalid solver sort") from exc


BOOL_SORT = Sort(SortKind.BOOL)
INT_SORT = Sort(SortKind.INT)
REAL_SORT = Sort(SortKind.REAL)


class TermOp(str, Enum):
    SYMBOL = "symbol"
    BOOL_VAL = "bool_val"
    INT_VAL = "int_val"
    REAL_VAL = "real_val"
    NOT = "not"
    AND = "and"
    OR = "or"
    XOR = "xor"
    IMPLIES = "implies"
    EQ = "eq"
    DISTINCT = "distinct"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    NEG = "neg"
    ITE = "ite"
    FORALL = "forall"
    EXISTS = "exists"


_BOOLEAN_NARY = {TermOp.AND, TermOp.OR, TermOp.XOR}
_COMPARISONS = {TermOp.LT, TermOp.LE, TermOp.GT, TermOp.GE}
_ARITHMETIC = {TermOp.ADD, TermOp.SUB, TermOp.MUL, TermOp.DIV}
_QUANTIFIERS = {TermOp.FORALL, TermOp.EXISTS}


@dataclass(frozen=True, slots=True)
class Term:
    """Small solver-neutral expression tree.

    It intentionally covers only the Boolean/integer/real fragment required by
    this subsystem.  Backend-specific native terms never escape backend modules.
    """

    op: TermOp
    sort: Sort
    args: tuple["Term", ...] = ()
    name: str | None = None
    value: bool | int | str | None = None
    bound_variables: tuple["Term", ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.op, TermOp):
            try:
                object.__setattr__(self, "op", TermOp(self.op))
            except (TypeError, ValueError) as exc:
                raise MalformedSpecificationError("invalid term operator") from exc
        if not isinstance(self.sort, Sort):
            raise MalformedSpecificationError("term sort must be a Sort")
        args = tuple(self.args)
        bound_variables = tuple(self.bound_variables)
        if any(not isinstance(arg, Term) for arg in args + bound_variables):
            raise MalformedSpecificationError("term arguments and bound variables must be Term objects")
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "bound_variables", bound_variables)
        self._validate_shape()

    def _validate_shape(self) -> None:
        op = self.op
        args = self.args
        if op is TermOp.SYMBOL:
            name = str(self.name or "").strip()
            if not name or args or self.value is not None or self.bound_variables:
                raise MalformedSpecificationError("symbol terms require only a non-empty name")
            object.__setattr__(self, "name", name)
            return
        if op is TermOp.BOOL_VAL:
            if self.sort != BOOL_SORT or not isinstance(self.value, bool) or args or self.bound_variables:
                raise MalformedSpecificationError("invalid Boolean literal term")
            return
        if op is TermOp.INT_VAL:
            if self.sort != INT_SORT or isinstance(self.value, bool) or not isinstance(self.value, int) or args or self.bound_variables:
                raise MalformedSpecificationError("invalid integer literal term")
            return
        if op is TermOp.REAL_VAL:
            if self.sort != REAL_SORT or not isinstance(self.value, str) or args or self.bound_variables:
                raise MalformedSpecificationError("invalid real literal term")
            if "/" in self.value:
                pieces = self.value.split("/")
                try:
                    valid_rational = len(pieces) == 2 and int(pieces[1]) != 0
                    if valid_rational:
                        int(pieces[0])
                except ValueError:
                    valid_rational = False
                if not valid_rational:
                    raise MalformedSpecificationError("invalid rational real literal")
            else:
                try:
                    parsed = Decimal(self.value)
                except InvalidOperation as exc:
                    raise MalformedSpecificationError("invalid real literal") from exc
                if not parsed.is_finite():
                    raise MalformedSpecificationError("real literal must be finite")
            return
        if self.name is not None or self.value is not None:
            raise MalformedSpecificationError("compound terms cannot carry name/value fields")

        if op is TermOp.NOT:
            self._require_arity(1)
            self._require_bool(args[0])
            self._require_sort(BOOL_SORT)
        elif op in _BOOLEAN_NARY:
            if len(args) < 2:
                raise MalformedSpecificationError(f"{op.value} requires at least two arguments")
            for arg in args:
                self._require_bool(arg)
            self._require_sort(BOOL_SORT)
        elif op is TermOp.IMPLIES:
            self._require_arity(2)
            self._require_bool(args[0]); self._require_bool(args[1]); self._require_sort(BOOL_SORT)
        elif op in {TermOp.EQ, TermOp.DISTINCT}:
            if len(args) < 2:
                raise MalformedSpecificationError(f"{op.value} requires at least two arguments")
            first_sort = args[0].sort
            if any(arg.sort != first_sort for arg in args):
                raise MalformedSpecificationError(f"{op.value} arguments must have identical sorts")
            self._require_sort(BOOL_SORT)
        elif op in _COMPARISONS:
            self._require_arity(2)
            self._require_numeric_same_sort(args)
            self._require_sort(BOOL_SORT)
        elif op in _ARITHMETIC:
            if len(args) < 2:
                raise MalformedSpecificationError(f"{op.value} requires at least two arguments")
            self._require_numeric_same_sort(args)
            if self.sort != args[0].sort:
                raise MalformedSpecificationError(f"{op.value} result sort must match operand sort")
        elif op is TermOp.NEG:
            self._require_arity(1)
            if args[0].sort.kind not in {SortKind.INT, SortKind.REAL} or self.sort != args[0].sort:
                raise MalformedSpecificationError("neg requires a numeric operand and matching result sort")
        elif op is TermOp.ITE:
            self._require_arity(3)
            self._require_bool(args[0])
            if args[1].sort != args[2].sort or self.sort != args[1].sort:
                raise MalformedSpecificationError("ite branches/result must have identical sorts")
        elif op in _QUANTIFIERS:
            self._require_arity(1)
            self._require_bool(args[0])
            self._require_sort(BOOL_SORT)
            if not self.bound_variables:
                raise MalformedSpecificationError("quantifier requires at least one bound variable")
            names: set[str] = set()
            for variable in self.bound_variables:
                if variable.op is not TermOp.SYMBOL:
                    raise MalformedSpecificationError("quantifier bound variables must be symbols")
                key = variable.name or ""
                if key in names:
                    raise MalformedSpecificationError("duplicate quantified variable name")
                names.add(key)
        else:  # pragma: no cover - exhaustive defensive guard
            raise MalformedSpecificationError(f"unsupported term operator {op.value!r}")

        if op not in _QUANTIFIERS and self.bound_variables:
            raise MalformedSpecificationError("only quantified terms may carry bound variables")

    def _require_arity(self, expected: int) -> None:
        if len(self.args) != expected:
            raise MalformedSpecificationError(
                f"{self.op.value} requires exactly {expected} argument(s)",
                context={"observed": len(self.args)},
            )

    @staticmethod
    def _require_bool(term: "Term") -> None:
        if term.sort != BOOL_SORT:
            raise MalformedSpecificationError("Boolean operator received a non-Boolean term")

    def _require_sort(self, expected: Sort) -> None:
        if self.sort != expected:
            raise MalformedSpecificationError(
                f"{self.op.value} result must have sort {expected.kind.value}"
            )

    @staticmethod
    def _require_numeric_same_sort(args: Sequence["Term"]) -> None:
        if not args or args[0].sort.kind not in {SortKind.INT, SortKind.REAL}:
            raise MalformedSpecificationError("numeric operator requires integer or real operands")
        if any(arg.sort != args[0].sort for arg in args):
            raise MalformedSpecificationError("numeric operands must have identical sorts")

    @property
    def contains_quantifier(self) -> bool:
        return self.op in _QUANTIFIERS or any(arg.contains_quantifier for arg in self.args)


def symbol(name: str, sort: Sort) -> Term:
    return Term(TermOp.SYMBOL, sort, name=name)


def bool_val(value: bool) -> Term:
    if not isinstance(value, bool):
        raise MalformedSpecificationError("bool_val requires bool")
    return Term(TermOp.BOOL_VAL, BOOL_SORT, value=value)


def int_val(value: int) -> Term:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MalformedSpecificationError("int_val requires int")
    return Term(TermOp.INT_VAL, INT_SORT, value=value)


def real_val(value: int | float | str | Decimal | Fraction) -> Term:
    if isinstance(value, bool):
        raise MalformedSpecificationError("real_val does not accept bool")
    if isinstance(value, Fraction):
        text = f"{value.numerator}/{value.denominator}"
    elif isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise MalformedSpecificationError("real_val requires a finite value")
        text = repr(value)
    else:
        text = str(value).strip()
    if not text:
        raise MalformedSpecificationError("real_val requires a value")
    # Decimal cannot parse a rational string, but Z3 can. Validate both forms.
    if "/" in text:
        parts = text.split("/")
        if len(parts) != 2:
            raise MalformedSpecificationError("invalid rational real literal")
        try:
            numerator = int(parts[0]); denominator = int(parts[1])
        except ValueError as exc:
            raise MalformedSpecificationError("invalid rational real literal") from exc
        if denominator == 0:
            raise MalformedSpecificationError("real literal denominator cannot be zero")
        text = f"{numerator}/{denominator}"
    else:
        try:
            decimal = Decimal(text)
        except InvalidOperation as exc:
            raise MalformedSpecificationError("invalid real literal") from exc
        if not decimal.is_finite():
            raise MalformedSpecificationError("real literal must be finite")
    return Term(TermOp.REAL_VAL, REAL_SORT, value=text)


def apply(op: TermOp, *args: Term) -> Term:
    try:
        normalized_op = op if isinstance(op, TermOp) else TermOp(op)
    except (TypeError, ValueError) as exc:
        raise MalformedSpecificationError("invalid solver operator") from exc
    tuple_args = tuple(args)
    if normalized_op in {
        TermOp.SYMBOL,
        TermOp.BOOL_VAL,
        TermOp.INT_VAL,
        TermOp.REAL_VAL,
        TermOp.FORALL,
        TermOp.EXISTS,
    }:
        raise MalformedSpecificationError(f"use the dedicated builder for {normalized_op.value}")
    result_sort = BOOL_SORT
    if normalized_op in _ARITHMETIC or normalized_op is TermOp.NEG:
        if not tuple_args:
            raise MalformedSpecificationError(f"{normalized_op.value} requires operands")
        result_sort = tuple_args[0].sort
    elif normalized_op is TermOp.ITE:
        if len(tuple_args) < 2:
            raise MalformedSpecificationError("ite requires three arguments")
        result_sort = tuple_args[1].sort
    return Term(normalized_op, result_sort, args=tuple_args)


def forall(variables: Sequence[Term], body: Term) -> Term:
    return Term(TermOp.FORALL, BOOL_SORT, args=(body,), bound_variables=tuple(variables))


def exists(variables: Sequence[Term], body: Term) -> Term:
    return Term(TermOp.EXISTS, BOOL_SORT, args=(body,), bound_variables=tuple(variables))


def collect_free_symbols(term: Term) -> tuple[Term, ...]:
    """Return free symbols in deterministic first-occurrence order."""
    if not isinstance(term, Term):
        raise MalformedSpecificationError("collect_free_symbols requires a Term")
    ordered: list[Term] = []
    seen: set[tuple[str, SortKind]] = set()

    def visit(node: Term, bound: frozenset[tuple[str, SortKind]]) -> None:
        if node.op is TermOp.SYMBOL:
            key = (node.name or "", node.sort.kind)
            if key not in bound and key not in seen:
                seen.add(key)
                ordered.append(node)
            return
        local_bound = bound
        if node.op in _QUANTIFIERS:
            local_bound = bound | frozenset(
                (item.name or "", item.sort.kind) for item in node.bound_variables
            )
        for child in node.args:
            visit(child, local_bound)

    visit(term, frozenset())
    return tuple(ordered)


@dataclass(frozen=True, slots=True)
class NamedConstraint:
    name: str
    term: Term

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise MalformedSpecificationError("constraint name must be non-empty")
        if not isinstance(self.term, Term) or self.term.sort != BOOL_SORT:
            raise MalformedSpecificationError("constraints must contain Boolean terms")
        object.__setattr__(self, "name", name)


class SolverStatus(str, Enum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SolverCapabilities:
    sorts: frozenset[SortKind]
    operators: frozenset[TermOp]
    quantifiers: bool = False
    models: bool = False
    unsat_cores: bool = False
    assumptions: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "sorts", frozenset(SortKind(item) for item in self.sorts))
        object.__setattr__(self, "operators", frozenset(TermOp(item) for item in self.operators))


@dataclass(frozen=True, slots=True)
class SolverRequest:
    constraints: tuple[NamedConstraint, ...]
    assumptions: tuple[NamedConstraint, ...] = ()
    timeout_seconds: float | None = None
    produce_model: bool = True
    produce_unsat_core: bool = False

    def __post_init__(self) -> None:
        constraints = tuple(self.constraints)
        assumptions = tuple(self.assumptions)
        if any(not isinstance(item, NamedConstraint) for item in constraints + assumptions):
            raise MalformedSpecificationError("solver requests require NamedConstraint objects")
        names = [item.name for item in constraints + assumptions]
        if len(names) != len(set(names)):
            raise MalformedSpecificationError("solver constraint/assumption names must be unique")

        symbol_sorts: dict[str, SortKind] = {}

        def validate_symbol_sorts(term: Term) -> None:
            if term.op is TermOp.SYMBOL:
                name = term.name or ""
                previous = symbol_sorts.get(name)
                if previous is not None and previous is not term.sort.kind:
                    raise MalformedSpecificationError(
                        "a solver request cannot reuse a symbol name with different sorts",
                        context={
                            "symbol": name,
                            "first_sort": previous.value,
                            "second_sort": term.sort.kind.value,
                        },
                    )
                symbol_sorts[name] = term.sort.kind
            for variable in term.bound_variables:
                validate_symbol_sorts(variable)
            for child in term.args:
                validate_symbol_sorts(child)

        for named in constraints + assumptions:
            validate_symbol_sorts(named.term)
        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)):
                raise MalformedSpecificationError("solver timeout_seconds must be numeric or null")
            if float(self.timeout_seconds) <= 0:
                raise MalformedSpecificationError("solver timeout_seconds must be greater than zero")
            object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))
        if not isinstance(self.produce_model, bool) or not isinstance(self.produce_unsat_core, bool):
            raise MalformedSpecificationError("solver evidence flags must be boolean")
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "assumptions", assumptions)

    @property
    def quantified(self) -> bool:
        return any(item.term.contains_quantifier for item in self.constraints + self.assumptions)

    @property
    def required_sorts(self) -> frozenset[SortKind]:
        sorts: set[SortKind] = set()

        def walk(term: Term) -> None:
            sorts.add(term.sort.kind)
            for variable in term.bound_variables:
                sorts.add(variable.sort.kind)
            for child in term.args:
                walk(child)

        for constraint in self.constraints + self.assumptions:
            walk(constraint.term)
        return frozenset(sorts)

    @property
    def required_operators(self) -> frozenset[TermOp]:
        operators: set[TermOp] = set()

        def walk(term: Term) -> None:
            operators.add(term.op)
            for child in term.args:
                walk(child)

        for constraint in self.constraints + self.assumptions:
            walk(constraint.term)
        return frozenset(operators)


@dataclass(frozen=True, slots=True)
class SolverResponse:
    status: SolverStatus
    backend: str
    model: Mapping[str, str] = field(default_factory=dict)
    unsat_core: tuple[str, ...] = ()
    reason_unknown: str | None = None
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.status, SolverStatus):
            object.__setattr__(self, "status", SolverStatus(self.status))
        backend = str(self.backend).strip().lower()
        if not backend:
            raise MalformedSpecificationError("solver response backend must be non-empty")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "model", MappingProxyType(dict(self.model)))
        object.__setattr__(self, "unsat_core", tuple(str(item) for item in self.unsat_core))
        if self.elapsed_seconds < 0:
            raise MalformedSpecificationError("solver elapsed_seconds must be non-negative")
        if self.status is SolverStatus.UNKNOWN and not self.reason_unknown:
            raise MalformedSpecificationError("UNKNOWN solver responses require reason_unknown")
        if self.status is not SolverStatus.UNKNOWN and self.reason_unknown is not None:
            raise MalformedSpecificationError("only UNKNOWN solver responses may carry reason_unknown")
        if self.status is SolverStatus.SAT and self.unsat_core:
            raise MalformedSpecificationError("SAT responses cannot carry an unsat core")
        if self.status is SolverStatus.UNSAT and self.model:
            raise MalformedSpecificationError("UNSAT responses cannot carry a model")


@runtime_checkable
class SolverBackend(Protocol):
    """Small solver-independent boundary implemented by optional backends."""

    @property
    def name(self) -> str:
        ...

    @property
    def available(self) -> bool:
        ...

    @property
    def capabilities(self) -> SolverCapabilities:
        ...

    def check(self, request: SolverRequest) -> SolverResponse:
        ...


__all__ = [
    "BOOL_SORT",
    "INT_SORT",
    "REAL_SORT",
    "NamedConstraint",
    "SolverBackend",
    "SolverCapabilities",
    "SolverRequest",
    "SolverResponse",
    "SolverStatus",
    "Sort",
    "SortKind",
    "Term",
    "TermOp",
    "apply",
    "bool_val",
    "collect_free_symbols",
    "exists",
    "forall",
    "int_val",
    "real_val",
    "symbol",
]
