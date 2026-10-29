"""Animation-domain exception taxonomy; no import-time fallback.

The Animation Agent must be able to distinguish "the caller asked for
something impossible" from "the caller asked for something we don't
support" from "the scene changed underneath us" — without relying on
string matching.  Every animation-domain failure therefore derives from
:class:`AnimationError`, which in turn derives from the shared
:class:`~src.agents.base.utils.base_errors.BaseError`.  That gives us
deterministic codes, structured context, and serialisable metadata
consistent with the rest of the Base Agent stack.

Error code layout (``ANI-1xxx``, mirroring the base stack's ``BAS-1xxx``
layout with 100-unit spacing per category):

=================================== ========  ==========================================
Class                               Code      Purpose
=================================== ========  ==========================================
AnimationError                      ANI-1000  Domain root — not raised directly.
AnimationConfigurationError         ANI-1100  Invalid/contradictory configuration.
AnimationRevisionConflictError      ANI-1200  Optimistic-lock (CAS) failure.
AnimationValidationError            ANI-1300  Invalid payload, track, or keyframe.
AnimationUnsupportedOperationError  ANI-1400  Operation/provider not supported.
AnimationStateError                 ANI-1500  Runtime precondition not satisfied.
AnimationIndexError                 ANI-1600  Track/keyframe index out of range.
==================================  ========  ==========================================

Two compatibility guarantees matter for incremental migration:

* Every subclass that replaces a historical ``raise ValueError(...)``
  also inherits :class:`ValueError`; every subclass replacing a
  historical ``raise IndexError(...)`` also inherits :class:`IndexError`;
  every subclass replacing a historical ``raise RuntimeError(...)``
  also inherits :class:`RuntimeError`.  Existing ``except`` clauses
  keep working while callers migrate to ``except AnimationError``.
* The base ``AnimationError.__init__`` sets the class ``default_code``
  *only as a default*; callers may override ``code=`` when they need a
  finer-grained sub-code, but they can never silently lose the
  ``ANI-`` prefix by omitting it.

The module imports nothing at call time and performs no I/O.
"""

from __future__ import annotations

from typing import Any

from ...base.utils.base_errors import BaseError, BaseErrorType


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
class AnimationError(BaseError):
    """Root of the animation exception hierarchy.

    Not raised directly by first-party code; callers should catch
    :class:`AnimationError` to handle *any* animation-domain failure
    uniformly, or one of the concrete subclasses for finer dispatch.

    Attributes
    ----------
    error_type
        Inherited default :data:`BaseErrorType.RUNTIME`; subclasses
        narrow this where a more specific domain fits.
    default_code
        ``"ANI-1000"`` — the taxonomy root.
    default_severity
        ``"medium"``.  Concrete subclasses raise or lower this according
        to operational impact.
    default_category
        ``"animation"`` — kept as the namespace prefix for downstream
        categories.
    """

    error_type = BaseErrorType.RUNTIME
    default_code = "ANI-1000"
    default_severity = "medium"
    default_category = "animation"
    default_retryable = False

    def __init__(self, message: str, **kwargs: Any) -> None:
        # Honour an explicit ``code`` from the caller; otherwise pin the
        # class-level ``default_code``.  We cannot rely on BaseError's
        # ``_ERROR_CODE_MAP`` fallback here because that map only knows
        # the base ``BAS-xxxx`` codes — passing our ANI code explicitly
        # bypasses the lookup and prevents accidental BAS downgrade.
        kwargs["code"] = kwargs.get("code") or self.default_code
        super().__init__(message, **kwargs)


# ---------------------------------------------------------------------------
# 1100 — Configuration
# ---------------------------------------------------------------------------
class AnimationConfigurationError(AnimationError, ValueError):
    """Invalid or contradictory animation configuration.

    Raised by constructors and config bindings (e.g. non-positive
    ``AnimationMemory.capacity``, malformed config sections).  Inherits
    :class:`ValueError` so pre-existing ``except ValueError`` handlers
    continue to trap it during incremental migration.
    """

    error_type = BaseErrorType.CONFIGURATION
    default_code = "ANI-1100"
    default_severity = "high"
    default_category = "animation.configuration"


# ---------------------------------------------------------------------------
# 1200 — Optimistic-lock / CAS
# ---------------------------------------------------------------------------
class AnimationRevisionConflictError(AnimationError):
    """A caller-supplied ``expected_revision`` does not match the scene.

    Always recoverable by re-reading the scene revision and retrying the
    request, so it is flagged retryable by default.  Deliberately *not*
    a :class:`ValueError`: the caller's payload may be perfectly valid,
    only its view of the world is stale.
    """

    error_type = BaseErrorType.STATE
    default_code = "ANI-1200"
    default_severity = "high"
    default_category = "animation.revision"
    default_retryable = True


# ---------------------------------------------------------------------------
# 1300 — Validation
# ---------------------------------------------------------------------------
class AnimationValidationError(AnimationError, ValueError):
    """Structurally or semantically invalid animation input.

    Raised by validators for payload shape, scalar finiteness,
    quaternion shape/magnitude, keyframe layout, and the scalar
    pre-conditions documented in
    :mod:`src.agents.animations.utils.animation_helpers`.
    Inherits :class:`ValueError` for compatibility with the legacy
    ``raise ValueError(...)`` sites in ``animation_commands`` and
    ``interpolation``.
    """

    error_type = BaseErrorType.VALIDATION
    default_code = "ANI-1300"
    default_severity = "medium"
    default_category = "animation.validation"


# ---------------------------------------------------------------------------
# 1400 — Unsupported operation / provider
# ---------------------------------------------------------------------------
class AnimationUnsupportedOperationError(AnimationError):
    """The requested operation, provider method, or edit is unsupported.

    This is the deliberate "we refuse" signal — distinct from
    :class:`AnimationStateError` (state isn't ready) and
    :class:`AnimationValidationError` (input is wrong).  Never
    retryable: retrying the same unsupported request will fail the
    same way.
    """

    error_type = BaseErrorType.RUNTIME
    default_code = "ANI-1400"
    default_severity = "medium"
    default_category = "animation.operation"


# ---------------------------------------------------------------------------
# 1500 — Runtime state
# ---------------------------------------------------------------------------
class AnimationStateError(AnimationError, RuntimeError):
    """A required precondition of the animation runtime is not met.

    Canonical examples: editing a scene that carries no animation,
    evaluating a scene whose provider does not implement ``evaluate``,
    or addressing a scene that was never bound to the agent.  Also a
    :class:`RuntimeError` for callers that guard with ``except
    RuntimeError``.
    """

    error_type = BaseErrorType.STATE
    default_code = "ANI-1500"
    default_severity = "high"
    default_category = "animation.state"


# ---------------------------------------------------------------------------
# 1600 — Index
# ---------------------------------------------------------------------------
class AnimationIndexError(AnimationError, IndexError):
    """Track or keyframe index is outside the valid range.

    Also an :class:`IndexError` so call sites that already guard list
    indexing with ``except IndexError`` continue to work.  Raised
    eagerly *before* any mutation so failed commands cannot leave the
    caller's list half-edited.
    """

    error_type = BaseErrorType.VALIDATION
    default_code = "ANI-1600"
    default_severity = "medium"
    default_category = "animation.index"


__all__ = [
    "AnimationError",
    "AnimationConfigurationError",
    "AnimationRevisionConflictError",
    "AnimationValidationError",
    "AnimationUnsupportedOperationError",
    "AnimationStateError",
    "AnimationIndexError",
]