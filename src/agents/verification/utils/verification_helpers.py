"""Small deterministic helpers shared by Verification subsystem modules.

The module deliberately reuses SLAI's tuning helper primitives for timestamps
and stable fingerprints instead of reimplementing generic serialization or
hashing.  Functions here add only Verification-specific semantics.
"""

from __future__ import annotations

from typing import Iterable, TypedDict

from src.tuning.utils.tuning_helpers import stable_fingerprint # type: ignore
from ..solving.solver import Term, TermOp, collect_free_symbols
from .verification_errors import MalformedSpecificationError
from ..verification_result import VerificationResult


class TermStatistics(TypedDict):
    """Structural statistics for a backend-neutral verification term."""

    node_count: int
    max_depth: int
    quantifier_count: int
    free_symbol_count: int


def normalize_tags(tags: Iterable[str] | str | None) -> tuple[str, ...]:
    """Return unique, sorted, non-empty tags.

    Sorting gives memory queries and serialized records deterministic tag order.
    """

    if tags is None:
        return ()
    values = (tags,) if isinstance(tags, str) else tags
    normalized: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if not value:
            raise MalformedSpecificationError("verification tags must be non-empty strings")
        normalized.add(value)
    return tuple(sorted(normalized))


def term_statistics(term: Term) -> TermStatistics:
    """Return bounded structural facts about a validated solver-neutral term."""

    if not isinstance(term, Term):
        raise MalformedSpecificationError("term_statistics requires a Term")

    node_count = 0
    max_depth = 0
    quantifier_count = 0

    def visit(node: Term, depth: int) -> None:
        nonlocal node_count, max_depth, quantifier_count
        node_count += 1
        max_depth = max(max_depth, depth)
        if node.op in {TermOp.FORALL, TermOp.EXISTS}:
            quantifier_count += 1
        for variable in node.bound_variables:
            visit(variable, depth + 1)
        for child in node.args:
            visit(child, depth + 1)

    visit(term, 1)
    return {
        "node_count": node_count,
        "max_depth": max_depth,
        "quantifier_count": quantifier_count,
        "free_symbol_count": len(collect_free_symbols(term)),
    }


def _term_payload(term: Term) -> dict[str, object]:
    return {
        "op": term.op.value,
        "sort": term.sort.kind.value,
        "name": term.name,
        "value": term.value,
        "bound_variables": [_term_payload(item) for item in term.bound_variables],
        "args": [_term_payload(item) for item in term.args],
    }


def term_fingerprint(term: Term) -> str:
    """Return a deterministic fingerprint for a solver-neutral formal term."""

    if not isinstance(term, Term):
        raise MalformedSpecificationError("term_fingerprint requires a Term")
    return stable_fingerprint(_term_payload(term))


def result_fingerprint(result: VerificationResult[object]) -> str:
    """Return a deterministic fingerprint of formal result evidence.

    Runtime memory metadata such as insertion time is intentionally excluded;
    only :meth:`VerificationResult.to_dict` evidence participates.
    """

    if not isinstance(result, VerificationResult):
        raise MalformedSpecificationError("result_fingerprint requires VerificationResult")
    return stable_fingerprint(result.to_dict())


__all__ = [
    "TermStatistics",
    "normalize_tags",
    "result_fingerprint",
    "term_fingerprint",
    "term_statistics",
]
