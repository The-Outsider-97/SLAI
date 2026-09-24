"""Core closed types and resource settings for the Verification subsystem."""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence, cast

from .utils.verification_errors import MalformedSpecificationError


class VerificationStatus(str, Enum):
    """Semantically distinct outcomes produced by formal verification procedures."""

    VERIFIED = "verified"
    REFUTED = "refuted"
    SATISFIABLE = "satisfiable"
    UNSATISFIABLE = "unsatisfiable"
    UNKNOWN = "unknown"
    BOUNDED = "bounded"

    @property
    def is_conclusive(self) -> bool:
        """Whether the status establishes/refutes a proposition within its semantics."""
        return self in {
            VerificationStatus.VERIFIED,
            VerificationStatus.REFUTED,
            VerificationStatus.SATISFIABLE,
            VerificationStatus.UNSATISFIABLE,
        }


class VerificationScope(str, Enum):
    """Scope under which a result was obtained."""

    COMPLETE = "complete"
    BOUNDED = "bounded"
    RESOURCE_LIMITED = "resource_limited"
    ABSTRACT = "abstract"


class VerificationMethod(str, Enum):
    """Verification mechanism used to obtain a result."""

    CONTRACT = "contract"
    INVARIANT = "invariant"
    ABSTRACT_INTERPRETATION = "abstract_interpretation"
    SAT = "sat"
    SMT = "smt"
    MODEL_CHECKING = "model_checking"


def _positive_int(value: object, name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise MalformedSpecificationError(
            f"{name} must be a positive integer" + (" or null" if allow_none else ""),
            context={"field": name, "value": repr(value)},
        )
    if value <= 0:
        raise MalformedSpecificationError(f"{name} must be greater than zero", context={"field": name, "value": value})
    return value


def _positive_float(value: object, name: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MalformedSpecificationError(
            f"{name} must be a positive number" + (" or null" if allow_none else ""),
            context={"field": name, "value": repr(value)},
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise MalformedSpecificationError(
            f"{name} must be finite and greater than zero",
            context={"field": name, "value": repr(value)},
        )
    return numeric


@dataclass(frozen=True, slots=True)
class ResourceBounds:
    """Hard limits for explicit-state verification.

    ``max_depth=None`` means there is no *semantic* depth bound.  The search is
    still protected by state, transition, and wall-time limits.
    """

    max_states: int = 10_000
    max_transitions: int = 50_000
    max_depth: int | None = None
    timeout_seconds: float | None = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_states", _positive_int(self.max_states, "max_states"))
        object.__setattr__(self, "max_transitions", _positive_int(self.max_transitions, "max_transitions"))
        if self.max_depth is not None:
            if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int):
                raise MalformedSpecificationError(
                    "max_depth must be a non-negative integer or null",
                    context={"field": "max_depth", "value": repr(self.max_depth)},
                )
            if self.max_depth < 0:
                raise MalformedSpecificationError(
                    "max_depth must be non-negative",
                    context={"field": "max_depth", "value": self.max_depth},
                )
        if self.timeout_seconds is not None:
            object.__setattr__(self, "timeout_seconds", _positive_float(self.timeout_seconds, "timeout_seconds"))


@dataclass(frozen=True, slots=True)
class SolverSettings:
    """Declarative solver preferences; no backend is imported by this object."""

    preferred_backends: tuple[str, ...] = ("z3",)
    timeout_seconds: float | None = 10.0
    produce_models: bool = True
    produce_unsat_cores: bool = False

    def __post_init__(self) -> None:
        names = tuple(str(name).strip().lower() for name in self.preferred_backends)
        if not names or any(not name for name in names):
            raise MalformedSpecificationError("preferred_backends must contain at least one non-empty backend name")
        if len(set(names)) != len(names):
            raise MalformedSpecificationError("preferred_backends must not contain duplicates", context={"preferred_backends": names})
        object.__setattr__(self, "preferred_backends", names)
        if self.timeout_seconds is not None:
            object.__setattr__(self, "timeout_seconds", _positive_float(self.timeout_seconds, "solver.timeout_seconds"))
        if not isinstance(self.produce_models, bool):
            raise MalformedSpecificationError("produce_models must be boolean")
        if not isinstance(self.produce_unsat_cores, bool):
            raise MalformedSpecificationError("produce_unsat_cores must be boolean")


@dataclass(frozen=True, slots=True)
class AbstractionSettings:
    """Bounds for monotone abstract fixpoint iteration."""

    max_iterations: int = 256
    widening_after: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_iterations", _positive_int(self.max_iterations, "abstract_interpretation.max_iterations"))
        if self.widening_after is not None:
            if isinstance(self.widening_after, bool) or not isinstance(self.widening_after, int):
                raise MalformedSpecificationError("abstract_interpretation.widening_after must be a non-negative integer or null")
            if self.widening_after < 0:
                raise MalformedSpecificationError("abstract_interpretation.widening_after must be non-negative")




@dataclass(frozen=True, slots=True)
class MemorySettings:
    """Bounds for transient Verification-specific result history."""

    max_entries: int = 1_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_entries", _positive_int(self.max_entries, "memory.max_entries"))


@dataclass(frozen=True, slots=True)
class VerificationSettings:
    """Typed view over a mapping loaded by SLAI's existing config loader.

    This class intentionally does not read files.  Callers should use the existing
    SLAI configuration mechanism and pass the resulting mapping to
    :meth:`from_mapping`.
    """

    solver: SolverSettings = field(default_factory=SolverSettings)
    model_checking: ResourceBounds = field(default_factory=ResourceBounds)
    abstract_interpretation: AbstractionSettings = field(default_factory=AbstractionSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> "VerificationSettings":
        if not isinstance(config, Mapping):
            raise MalformedSpecificationError("verification configuration must be a mapping")

        solver_raw = config.get("solver", {})
        model_raw = config.get("model_checking", {})
        abstract_raw = config.get("abstract_interpretation", {})
        memory_raw = config.get("memory", {})
        for name, raw in (
            ("solver", solver_raw),
            ("model_checking", model_raw),
            ("abstract_interpretation", abstract_raw),
            ("memory", memory_raw),
        ):
            if not isinstance(raw, Mapping):
                raise MalformedSpecificationError(
                    f"configuration section {name!r} must be a mapping"
                )

            solver_raw = cast(Mapping[str, object], solver_raw)
        preferred = solver_raw.get("preferred_backends", ("z3",))
        if isinstance(preferred, str):
            preferred_backends: Sequence[object] = (preferred,)
        elif isinstance(preferred, Sequence):
            preferred_backends = preferred
        else:
            raise MalformedSpecificationError("solver.preferred_backends must be a sequence")

        solver = SolverSettings(
            preferred_backends=tuple(str(item) for item in preferred_backends),
            timeout_seconds=solver_raw.get("timeout_seconds", 10.0),  # type: ignore[arg-type]
            produce_models=solver_raw.get("produce_models", True),  # type: ignore[arg-type]
            produce_unsat_cores=solver_raw.get("produce_unsat_cores", False),  # type: ignore[arg-type]
        )
        model = ResourceBounds(
            max_states=model_raw.get("max_states", 10_000),  # type: ignore[arg-type]
            max_transitions=model_raw.get("max_transitions", 50_000),  # type: ignore[arg-type]
            max_depth=model_raw.get("max_depth", None),  # type: ignore[arg-type]
            timeout_seconds=model_raw.get("timeout_seconds", 10.0),  # type: ignore[arg-type]
        )
        abstraction = AbstractionSettings(
            max_iterations=abstract_raw.get("max_iterations", 256),  # type: ignore[arg-type]
            widening_after=abstract_raw.get("widening_after", None),  # type: ignore[arg-type]
        )
        memory = MemorySettings(
            max_entries=memory_raw.get("max_entries", 1_000),  # type: ignore[arg-type]
        )
        return cls(
            solver=solver,
            model_checking=model,
            abstract_interpretation=abstraction,
            memory=memory,
        )


__all__ = [
    "AbstractionSettings",
    "MemorySettings",
    "ResourceBounds",
    "SolverSettings",
    "VerificationMethod",
    "VerificationScope",
    "VerificationSettings",
    "VerificationStatus",
]
