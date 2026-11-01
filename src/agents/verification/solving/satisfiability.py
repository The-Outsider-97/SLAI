"""SAT/SMT-backed satisfiability and validity orchestration."""

from __future__ import annotations

from .backends import BackendRegistry, default_backend_registry
from .solver import *
from ..verification_result import VerificationProvenance, VerificationResult
from ..verification_types import *
from ..utils.verification_errors import *


class SatisfiabilityVerifier:
    """Backend-neutral satisfiability/validity service."""

    def __init__(
        self,
        settings: SolverSettings | None = None,
        registry: BackendRegistry | None = None,
    ) -> None:
        self.settings = settings or SolverSettings()
        self.registry = registry or default_backend_registry()

    def _backend(self, backend_name: str | None) -> SolverBackend:
        if backend_name is not None:
            return self.registry.create(backend_name)
        return self.registry.select(self.settings.preferred_backends)

    @staticmethod
    def _unsupported_reason(request: SolverRequest, backend: SolverBackend) -> str | None:
        capabilities = backend.capabilities
        missing_sorts = sorted(
            (item.value for item in request.required_sorts - capabilities.sorts)
        )
        if missing_sorts:
            return f"backend does not support required sorts: {', '.join(missing_sorts)}"
        missing_ops = sorted(
            (item.value for item in request.required_operators - capabilities.operators)
        )
        if missing_ops:
            return f"backend does not support required operators: {', '.join(missing_ops)}"
        if request.quantified and not capabilities.quantifiers:
            return "backend does not support quantified formulas"
        if request.assumptions and not capabilities.assumptions:
            return "backend does not support solver assumptions"
        return None

    def check(
        self,
        request: SolverRequest,
        *,
        backend_name: str | None = None,
        property_name: str = "constraint_system",
    ) -> VerificationResult[object]:
        if not isinstance(request, SolverRequest):
            raise MalformedSpecificationError("check requires a SolverRequest")
        backend = self._backend(backend_name)
        unsupported = self._unsupported_reason(request, backend)
        assumptions = tuple(item.name for item in request.assumptions)
        method = (
            VerificationMethod.SAT
            if not request.quantified and request.required_sorts.issubset({BOOL_SORT.kind})
            else VerificationMethod.SMT
        )
        if unsupported is not None:
            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                property_name=property_name,
                summary="Solver capabilities are insufficient for this formal request.",
                provenance=VerificationProvenance(
                    method=method,
                    scope=VerificationScope.COMPLETE,
                    backend=backend.name,
                    quantified=request.quantified,
                ),
                assumptions=assumptions,
                limitations=(unsupported,),
                unknown_reason=unsupported,
            )
        if request.produce_model and not backend.capabilities.models:
            raise UnsupportedVerificationError(
                "requested model evidence is unsupported by the selected backend",
                context={"backend": backend.name},
            )
        if request.produce_unsat_core and not backend.capabilities.unsat_cores:
            raise UnsupportedVerificationError(
                "requested unsat-core evidence is unsupported by the selected backend",
                context={"backend": backend.name},
            )

        response = backend.check(request)
        provenance = VerificationProvenance(
            method=method,
            scope=VerificationScope.COMPLETE,
            backend=response.backend,
            elapsed_seconds=response.elapsed_seconds,
            quantified=request.quantified,
        )
        if response.status is SolverStatus.SAT:
            return VerificationResult(
                status=VerificationStatus.SATISFIABLE,
                property_name=property_name,
                summary="The supplied constraints are satisfiable.",
                provenance=provenance,
                assumptions=assumptions,
                model=response.model,
            )
        if response.status is SolverStatus.UNSAT:
            return VerificationResult(
                status=VerificationStatus.UNSATISFIABLE,
                property_name=property_name,
                summary="The supplied constraints are unsatisfiable.",
                provenance=provenance,
                assumptions=assumptions,
                unsat_core=response.unsat_core,
            )
        return VerificationResult(
            status=VerificationStatus.UNKNOWN,
            property_name=property_name,
            summary="The solver did not establish satisfiability or unsatisfiability.",
            provenance=provenance,
            assumptions=assumptions,
            unknown_reason=response.reason_unknown or "solver returned unknown",
            limitations=(response.reason_unknown or "solver returned unknown",),
        )

    def prove(
        self,
        property_term: Term,
        *,
        assumptions: tuple[NamedConstraint, ...] = (),
        backend_name: str | None = None,
        property_name: str = "property",
        timeout_seconds: float | None = None,
    ) -> VerificationResult[object]:
        """Establish validity by testing ``assumptions ∧ ¬property`` for UNSAT."""
        if not isinstance(property_term, Term) or property_term.sort != BOOL_SORT:
            raise MalformedSpecificationError("prove requires a Boolean property term")
        assumption_tuple = tuple(assumptions)
        reserved = {item.name for item in assumption_tuple}
        property_constraint_name = "__negated_property__"
        suffix = 0
        while property_constraint_name in reserved:
            suffix += 1
            property_constraint_name = f"__negated_property_{suffix}__"
        negated = apply(TermOp.NOT, property_term)
        request = SolverRequest(
            constraints=(NamedConstraint(property_constraint_name, negated),),
            assumptions=assumption_tuple,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else self.settings.timeout_seconds,
            produce_model=self.settings.produce_models,
            produce_unsat_core=self.settings.produce_unsat_cores,
        )
        raw = self.check(request, backend_name=backend_name, property_name=property_name)
        if raw.status is VerificationStatus.UNSATISFIABLE:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                property_name=property_name,
                summary="The property is valid under the supplied assumptions: its negation is unsatisfiable.",
                provenance=raw.provenance,
                assumptions=raw.assumptions,
                unsat_core=raw.unsat_core,
            )
        if raw.status is VerificationStatus.SATISFIABLE:
            return VerificationResult(
                status=VerificationStatus.REFUTED,
                property_name=property_name,
                summary="A model satisfies the assumptions and the negation of the property.",
                provenance=raw.provenance,
                assumptions=raw.assumptions,
                model=raw.model,
            )
        return VerificationResult(
            status=VerificationStatus.UNKNOWN,
            property_name=property_name,
            summary="Validity could not be established or refuted by the selected solver.",
            provenance=raw.provenance,
            assumptions=raw.assumptions,
            limitations=raw.limitations,
            unknown_reason=raw.unknown_reason or "validity check unresolved",
        )


__all__ = ["SatisfiabilityVerifier"]
