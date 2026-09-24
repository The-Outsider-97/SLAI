"""Minimal abstract-interpretation primitives for formal verification.

The module implements only the verification mechanics justified by the
subsystem's scope: ordered abstract domains, joins, bounded ascending fixpoint
iteration, optional caller-supplied widening, and explicit checks that a
candidate is an inductive/post-fixpoint approximation.

The caller remains responsible for defining a sound abstraction and transfer
semantics.  This module does not infer abstractions, synthesize widenings, or
claim completeness when an iteration bound is exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Protocol, TypeVar, runtime_checkable


from ..utils.verification_errors import *
from ..utils.verification_helpers import *
from logs.logger import get_logger # pyright: ignore[reportMissingImports]


logger = get_logger("Verification Abstraction")


TAbstract = TypeVar("TAbstract")


@runtime_checkable
class AbstractDomain(Protocol[TAbstract]):
    """Minimum join-semilattice operations needed by this subsystem."""

    def bottom(self) -> TAbstract:
        ...

    def join(self, left: TAbstract, right: TAbstract) -> TAbstract:
        ...

    def leq(self, left: TAbstract, right: TAbstract) -> bool:
        ...


AbstractTransformer = Callable[[TAbstract], TAbstract]
WideningOperator = Callable[[TAbstract, TAbstract], TAbstract]


@dataclass(frozen=True, slots=True)
class FixpointResult(Generic[TAbstract]):
    """Bounded outcome of an ascending abstract fixpoint computation."""

    value: TAbstract
    converged: bool
    iterations: int
    used_widening: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        require_non_negative_int(self.iterations, "fixpoint iterations")
        if not isinstance(self.converged, bool):
            raise MalformedSpecificationError("fixpoint converged flag must be boolean")
        if not isinstance(self.used_widening, bool):
            raise MalformedSpecificationError("fixpoint used_widening flag must be boolean")
        if self.converged and self.reason is not None:
            raise MalformedSpecificationError(
                "a converged fixpoint result must not carry a failure reason"
            )
        if not self.converged and not str(self.reason or "").strip():
            raise MalformedSpecificationError(
                "a non-converged fixpoint result must preserve a reason"
            )


@dataclass(frozen=True, slots=True)
class AbstractInvariantCheck(Generic[TAbstract]):
    """Result of checking an abstract candidate for inductive preservation.

    ``post_fixpoint`` records whether ``transfer(candidate) <= candidate``.
    ``initial_included`` is ``None`` when no initial abstract state was supplied.
    Only when an initial state is supplied and both conditions hold is
    ``inductive`` true.
    """

    candidate: TAbstract
    post_fixpoint: bool
    initial_included: bool | None

    def __post_init__(self) -> None:
        if not isinstance(self.post_fixpoint, bool):
            raise MalformedSpecificationError("post_fixpoint must be boolean")
        if self.initial_included is not None and not isinstance(self.initial_included, bool):
            raise MalformedSpecificationError("initial_included must be boolean or null")

    @property
    def inductive(self) -> bool:
        """Whether initialization and preservation were both established."""

        return self.initial_included is True and self.post_fixpoint


def _require_domain(domain: AbstractDomain[TAbstract]) -> AbstractDomain[TAbstract]:
    if not isinstance(domain, AbstractDomain):
        raise MalformedSpecificationError("domain must implement AbstractDomain")
    return domain


def _domain_bottom(domain: AbstractDomain[TAbstract]) -> TAbstract:
    try:
        return domain.bottom()
    except VerificationError:
        raise
    except Exception as exc:
        raise AbstractionError("abstract domain bottom() failed", cause=exc) from exc


def _domain_leq(
    domain: AbstractDomain[TAbstract],
    left: TAbstract,
    right: TAbstract,
    *,
    operation: str,
) -> bool:
    try:
        result = domain.leq(left, right)
    except VerificationError:
        raise
    except Exception as exc:
        raise AbstractionError(
            "abstract domain ordering check failed",
            context={"operation": operation},
            cause=exc,
        ) from exc
    if not isinstance(result, bool):
        raise AbstractionContractError(
            "abstract domain leq() must return bool",
            context={"operation": operation, "result_type": type(result).__name__},
        )
    return result


def _domain_join(
    domain: AbstractDomain[TAbstract],
    left: TAbstract,
    right: TAbstract,
    *,
    operation: str,
) -> TAbstract:
    try:
        joined = domain.join(left, right)
    except VerificationError:
        raise
    except Exception as exc:
        raise AbstractionError(
            "abstract domain join() failed",
            context={"operation": operation},
            cause=exc,
        ) from exc

    if not _domain_leq(domain, left, joined, operation=f"{operation}:left<=join"):
        raise AbstractionContractError(
            "abstract join must upper-bound its left operand",
            context={"operation": operation},
        )
    if not _domain_leq(domain, right, joined, operation=f"{operation}:right<=join"):
        raise AbstractionContractError(
            "abstract join must upper-bound its right operand",
            context={"operation": operation},
        )
    return joined


def _apply_transfer(
    transfer: AbstractTransformer[TAbstract],
    value: TAbstract,
    *,
    operation: str,
) -> TAbstract:
    try:
        return transfer(value)
    except VerificationError:
        raise
    except Exception as exc:
        raise AbstractionError(
            "abstract transfer function failed",
            context={"operation": operation},
            cause=exc,
        ) from exc


def join_abstract_values(
    domain: AbstractDomain[TAbstract],
    values: Iterable[TAbstract],
    *,
    initial: TAbstract | None = None,
) -> TAbstract:
    """Join a finite sequence of abstract values with contract checking.

    ``bottom()`` is used when ``initial`` is omitted.  The helper is deliberately
    finite and does not attempt a fixpoint computation.
    """

    _require_domain(domain)
    current = _domain_bottom(domain) if initial is None else initial
    for index, value in enumerate(values):
        current = _domain_join(
            domain,
            current,
            value,
            operation=f"join_abstract_values[{index}]",
        )
    return current


def check_abstract_invariant(
    domain: AbstractDomain[TAbstract],
    transfer: AbstractTransformer[TAbstract],
    candidate: TAbstract,
    *,
    initial: TAbstract | None = None,
) -> AbstractInvariantCheck[TAbstract]:
    """Check initialization and inductive preservation of an abstract candidate.

    The preservation condition is the post-fixpoint test
    ``transfer(candidate) <= candidate``.  If ``initial`` is supplied, the
    initialization condition ``initial <= candidate`` is checked as well.

    This function checks the caller-provided abstract semantics; it does not
    establish that those semantics are a sound abstraction of any concrete
    program on its own.
    """

    _require_domain(domain)
    require_callable(transfer, "transfer")

    transferred = _apply_transfer(transfer, candidate, operation="invariant-preservation")
    post_fixpoint = _domain_leq(
        domain,
        transferred,
        candidate,
        operation="transfer(candidate)<=candidate",
    )
    initial_included = (
        None
        if initial is None
        else _domain_leq(domain, initial, candidate, operation="initial<=candidate")
    )

    logger.debug(
        "Abstract invariant checked | post_fixpoint=%s | initial_included=%s",
        post_fixpoint,
        initial_included,
    )
    return AbstractInvariantCheck(
        candidate=candidate,
        post_fixpoint=post_fixpoint,
        initial_included=initial_included,
    )


def compute_post_fixpoint(
    domain: AbstractDomain[TAbstract],
    transfer: AbstractTransformer[TAbstract],
    *,
    initial: TAbstract | None = None,
    max_iterations: int = 256,
    widening_after: int | None = None,
    widening: WideningOperator[TAbstract] | None = None,
) -> FixpointResult[TAbstract]:
    """Compute a bounded ascending post-fixpoint approximation.

    Iteration uses ``current join transfer(current)``.  A supplied widening may
    be applied after ``widening_after`` iterations.  Exhausting the iteration
    bound returns ``converged=False`` rather than pretending a proof exists.
    """

    _require_domain(domain)
    require_callable(transfer, "transfer")
    max_iterations = require_positive_int(max_iterations, "max_iterations")

    if widening_after is not None:
        widening_after = require_non_negative_int(widening_after, "widening_after")
        if widening is None:
            raise MalformedSpecificationError(
                "widening_after requires a widening operator"
            )
    if widening is not None:
        require_callable(widening, "widening")

    current = _domain_bottom(domain) if initial is None else initial
    if initial is not None:
        bottom = _domain_bottom(domain)
        if not _domain_leq(domain, bottom, initial, operation="bottom<=initial"):
            raise AbstractionContractError(
                "abstract domain bottom() must be <= the supplied initial value"
            )

    logger.debug(
        "Starting abstract fixpoint computation | max_iterations=%d | widening_after=%s",
        max_iterations,
        widening_after,
    )

    used_widening = False
    for iteration in range(1, max_iterations + 1):
        transferred = _apply_transfer(transfer, current, operation=f"iteration-{iteration}")
        candidate = _domain_join(
            domain,
            current,
            transferred,
            operation=f"iteration-{iteration}",
        )

        if _domain_leq(domain, candidate, current, operation="candidate<=current"):
            logger.debug(
                "Abstract fixpoint converged | iterations=%d | widening=%s",
                iteration - 1,
                used_widening,
            )
            return FixpointResult(
                value=current,
                converged=True,
                iterations=iteration - 1,
                used_widening=used_widening,
            )

        next_value = candidate
        if widening is not None and widening_after is not None and iteration > widening_after:
            try:
                next_value = widening(current, candidate)
            except VerificationError:
                raise
            except Exception as exc:
                raise AbstractionError("abstract widening operator failed", context={"iteration": iteration}, cause=exc) from exc
            used_widening = True
            if not _domain_leq(domain, candidate, next_value, operation="candidate<=widened"):
                raise AbstractionContractError(
                    "widening must upper-bound the ascending candidate",
                    context={"iteration": iteration},
                )

        if not _domain_leq(domain, current, next_value, operation="current<=next"):
            raise AbstractionContractError("abstract iteration must remain ascending", context={"iteration": iteration})
        current = next_value

    logger.warning(
        "Abstract fixpoint bound exhausted without convergence | iterations=%d | widening=%s",
        max_iterations,
        used_widening,
    )
    return FixpointResult(
        value=current,
        converged=False,
        iterations=max_iterations,
        used_widening=used_widening,
        reason="maximum abstract-interpretation iterations exhausted before stabilization",
    )


__all__ = [
    "AbstractDomain",
    "AbstractInvariantCheck",
    "AbstractTransformer",
    "FixpointResult",
    "WideningOperator",
    "check_abstract_invariant",
    "compute_post_fixpoint",
    "join_abstract_values",
]
