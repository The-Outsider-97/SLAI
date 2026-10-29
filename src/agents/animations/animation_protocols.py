"""Minimal, application-independent contracts for animation scene providers.

SGEO is an existing implementation.  The protocols below are structural
contracts only: they say what attributes and methods a provider must
expose, not how it must be implemented.  In particular, they make **no
claim about thread-safety** — that is a per-provider guarantee that a
caller must verify, typically via the presence of a reentrant edit lock
(see :class:`AnimationEditLock`).

Protocol layering
-----------------
The surface is decomposed into orthogonal protocols so that a provider
can implement exactly what it needs and callers can accept the weakest
sufficient interface:

``EvaluatedAnimationScene``
    The read-only result of one evaluation — a detached snapshot with a
    revision stamp and a time value.  This is the *value* type returned
    by evaluation, not a provider.

``ReadOnlyAnimationSceneProvider``
    The minimum a scene must expose to support evaluation.  A provider
    implementing only this protocol can be evaluated but not edited.

``VersionedEvaluatorProvider``
    An optional extension on top of the read-only provider: a method
    that evaluates under an explicit revision check, atomically.  When
    present, :func:`evaluate_animation` uses it to eliminate the
    "check-then-evaluate" race that a plain ``revision`` + ``evaluate``
    sequence would otherwise contain.

``AnimationSceneProvider``
    The mutable extension: adds ``document_id``, ``metadata``, and
    ``set_animation``.  This is what the editing commands require.

``AnimationEditLock``
    A minimal runtime-checkable context-manager contract for the
    provider-owned edit lock.  SGEO exposes its reentrant
    ``_mutation_lock``; other providers may expose a public
    ``animation_edit_lock``.  When neither is present, editing fails
    closed rather than offering an unsafe pseudo-CAS (see
    :func:`animation_commands._edit_lock`).

Runtime checks
--------------
The protocols here are **not** marked ``@runtime_checkable`` because
they contain non-method members (``revision: int``, ``time: float``,
``metadata: Mapping``).  ``runtime_checkable`` only verifies method
presence for such protocols, which yields false positives.  Instead,
the module exports a set of :class:`typing.TypeGuard` predicates that
check both method presence **and** attribute type:

* :func:`is_evaluated_scene`
* :func:`is_read_only_animation_scene_provider`
* :func:`is_versioned_evaluator_provider`
* :func:`is_animation_scene_provider`
* :func:`get_animation_edit_lock`

Each predicate returns a narrowed type on success, so callers can write
``if is_animation_scene_provider(x): x.set_animation(...)`` with
static-type-checker support.  These predicates are the *canonical*
implementation of the duck-typing checks; ad-hoc ``hasattr`` chains
elsewhere should be replaced with them as files are touched, so the
contract lives in exactly one place.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Any, Protocol, TypeGuard, runtime_checkable


# ---------------------------------------------------------------------------
# Value type
# ---------------------------------------------------------------------------
class EvaluatedAnimationScene(Protocol):
    """Detached scene state returned by read-only evaluation.

    Attributes
    ----------
    time
        Evaluated timestamp in seconds.  Matches the time passed to
        :meth:`ReadOnlyAnimationSceneProvider.evaluate` unless the
        provider snaps or clamps for its own reasons; the exact
        relationship is provider-defined.
    revision
        Scene revision captured *at evaluation time*, not read before
        or after.  This is what makes the returned snapshot safe to
        reason about against an ``expected_revision``: the caller
        compares against the snapshot's stamp, not against a
        potentially-stale live value.
    """

    time: float
    revision: int

    def ids(self, *, include_hidden: bool = False) -> tuple[str, ...]:
        """Return object identifiers present in the evaluated snapshot.

        ``include_hidden=False`` (the default) is expected to exclude
        objects the provider considers invisible at this time; the
        definition of "hidden" is provider-defined.
        """
        ...


# ---------------------------------------------------------------------------
# Provider protocols
# ---------------------------------------------------------------------------
class ReadOnlyAnimationSceneProvider(Protocol):
    """Read-only scene-provider interface, independent of the file format.

    A provider implementing only this protocol supports evaluation but
    not editing.  This is the minimum required by
    :func:`animation_evaluation.evaluate_animation`.
    """

    revision: int

    def evaluate(self, time: float) -> EvaluatedAnimationScene:
        """Return a detached snapshot of the scene at ``time``.

        The provider is responsible for:

        * validating that ``time`` is within the provider's supported
          range (raises the provider's own error on out-of-range),
        * ensuring the returned snapshot does not alias live scene state,
        * stamping the snapshot with the *revision observed during*
          evaluation, not before or after.

        The caller does not pre-validate ``time`` beyond finiteness and
        non-negativity — semantic range checks belong to the provider.
        """
        ...


class VersionedEvaluatorProvider(ReadOnlyAnimationSceneProvider, Protocol):
    """Optional extension: atomic revision-checked evaluation.

    When a provider implements ``evaluate_at_revision``, the evaluation
    facade uses it in preference to ``evaluate`` *when the caller
    supplied* ``expected_revision``.  The method must fail atomically
    (raise a conflict error) if the live revision differs from
    ``expected_revision`` at evaluation time — this is what closes the
    check-then-evaluate race that the plain ``evaluate`` + ``revision``
    sequence cannot close.

    Providers that do not implement this method remain fully supported:
    the evaluation facade falls back to plain ``evaluate`` and then
    compares the snapshot's revision stamp against ``expected_revision``.
    That fallback is correct, but a concurrent mutation between
    ``evaluate`` and the revision check would be observable — the
    versioned method eliminates that window.
    """

    def evaluate_at_revision(
        self,
        time: float,
        expected_revision: int,
    ) -> EvaluatedAnimationScene:
        """Evaluate at ``time``, failing if the live revision is stale."""
        ...


class AnimationSceneProvider(ReadOnlyAnimationSceneProvider, Protocol):
    """Existing SGEO-compatible mutable animation interface.

    Implementations **must**:

    * validate ``animation`` against the provider's schema (SGEO's
      ``validate_animation`` is the reference implementation),
    * atomically replace the animation metadata,
    * atomically advance ``revision``,
    * return the new revision as a strictly positive integer greater
      than the previous revision.

    Safe read-modify-write edits additionally require an
    :class:`AnimationEditLock` (see :func:`get_animation_edit_lock`).
    A provider that exposes ``set_animation`` but no edit lock is
    treated as unsafe for mutation by :mod:`animation_commands`.

    Notes
    -----
    ``metadata`` carries the authoritative ``slai.animation.v1`` document
    under the key exported by
    :data:`src.utils.spatial_formats.sgeo.animation.ANIMATION_KEY`.
    Callers must not mutate ``metadata`` or any value reachable from it
    directly; all mutation goes through ``set_animation``.
    """

    document_id: str
    metadata: Mapping[str, Any]

    def set_animation(self, animation: dict[str, Any] | None) -> int:
        """Replace the animation document atomically; return the new revision.

        Passing ``None`` clears the animation.  The provider is
        responsible for validating ``animation``, updating ``metadata``,
        and advancing ``revision`` in a single logical operation —
        intermediate states must never be observable.
        """
        ...


# ---------------------------------------------------------------------------
# Edit lock
# ---------------------------------------------------------------------------
@runtime_checkable
class AnimationEditLock(Protocol):
    """Reentrant context manager serialising animation edits.

    Only the two context-manager dunders are required, which is why
    this protocol *can* be marked :func:`runtime_checkable` — every
    member is a method, so ``isinstance(obj, AnimationEditLock)``
    correctly distinguishes context managers from arbitrary objects.

    Reentrancy is required because :meth:`AnimationSceneProvider.set_animation`
    may itself acquire the lock; a non-reentrant lock would deadlock on
    the second acquisition.
    """

    def __enter__(self) -> Any: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None: ...


# ---------------------------------------------------------------------------
# Runtime predicates
# ---------------------------------------------------------------------------
def is_evaluated_scene(obj: object) -> TypeGuard[EvaluatedAnimationScene]:
    """Return ``True`` iff ``obj`` structurally matches ``EvaluatedAnimationScene``.

    Checks presence **and type** of each member:

    * ``time`` is an :class:`int` or :class:`float` (``bool`` rejected —
      a boolean timestamp is virtually always a bug),
    * ``revision`` is a plain :class:`int` (``bool`` rejected for the
      same reason),
    * ``ids`` is callable.

    The predicate does not invoke ``ids`` and does not validate the
    returned tuple; it only certifies structural conformance.
    """
    if type(getattr(obj, "time", None)) not in (int, float):
        return False
    if type(getattr(obj, "revision", None)) is not int:
        return False
    if not callable(getattr(obj, "ids", None)):
        return False
    return True


def is_read_only_animation_scene_provider(
    obj: object,
) -> TypeGuard[ReadOnlyAnimationSceneProvider]:
    """Return ``True`` iff ``obj`` can serve as a read-only provider.

    Requires ``revision: int`` and ``evaluate: Callable``.  ``bool``
    revisions are rejected: the provider contract specifies a positive
    integer, and ``True``/``False`` passing structural checks would mask
    provider bugs.
    """
    if type(getattr(obj, "revision", None)) is not int:
        return False
    if not callable(getattr(obj, "evaluate", None)):
        return False
    return True


def is_versioned_evaluator_provider(obj: object) -> TypeGuard[VersionedEvaluatorProvider]:
    """Return ``True`` iff ``obj`` supports atomic revision-checked evaluation.

    Implies :func:`is_read_only_animation_scene_provider` and additionally
    requires ``evaluate_at_revision`` to be callable.  Used by
    :func:`animation_evaluation.evaluate_animation` to decide whether
    the atomic path is available.
    """
    return (
        is_read_only_animation_scene_provider(obj)
        and callable(getattr(obj, "evaluate_at_revision", None))
    )


def is_animation_scene_provider(obj: object) -> TypeGuard[AnimationSceneProvider]:
    """Return ``True`` iff ``obj`` can serve as a mutable provider.

    Implies :func:`is_read_only_animation_scene_provider` and additionally
    requires:

    * ``document_id: str``,
    * ``metadata: Mapping``,
    * ``set_animation: Callable``.

    The predicate does **not** verify thread-safety; that is a separate
    concern checked by :func:`get_animation_edit_lock` at edit time.
    """
    if not is_read_only_animation_scene_provider(obj):
        return False
    if not isinstance(getattr(obj, "document_id", None), str):
        return False
    if not isinstance(getattr(obj, "metadata", None), Mapping):
        return False
    if not callable(getattr(obj, "set_animation", None)):
        return False
    return True


def get_animation_edit_lock(obj: object) -> AnimationEditLock | None:
    """Return the provider's reentrant edit lock, or ``None`` if absent.

    The lookup order is:

    1. ``animation_edit_lock`` — the public, non-SGEO name documented
       for third-party providers,
    2. ``_mutation_lock`` — SGEO's existing reentrant lock, retained
       for compatibility.

    If both are present, ``animation_edit_lock`` wins — a provider that
    exposes the public name is signalling that this is the supported
    entry point, and any hidden fallback should be treated as an
    implementation detail.

    Returns ``None`` when neither attribute holds an object that
    satisfies :class:`AnimationEditLock` (i.e. it is not a context
    manager).  Callers that require an edit lock (e.g.
    :mod:`animation_commands`) should treat ``None`` as a fail-closed
    condition and refuse the operation.
    """
    for name in ("animation_edit_lock", "_mutation_lock"):
        candidate = getattr(obj, name, None)
        if isinstance(candidate, AnimationEditLock):
            return candidate
    return None


__all__ = [
    "EvaluatedAnimationScene",
    "ReadOnlyAnimationSceneProvider",
    "VersionedEvaluatorProvider",
    "AnimationSceneProvider",
    "AnimationEditLock",
    "is_evaluated_scene",
    "is_read_only_animation_scene_provider",
    "is_versioned_evaluator_provider",
    "is_animation_scene_provider",
    "get_animation_edit_lock",
]