"""Validated animation request envelopes and canonical result shapes.

This module is the *boundary contract* between the AnimationAgent facade
and its callers.  Three concerns are separated:

1. **Request envelope validation.**  ``AnimationRequest`` validates its
   own field types and value domains at construction time so that the
   command/evaluation dispatchers below only ever receive well-formed
   inputs.  Per-operation *payload* semantics (keyframe shape, track
   indices, animation-document structure) are deliberately **not**
   validated here — those checks belong to the layer that knows what the
   operation means, and duplicating them across layers would produce two
   sources of truth that can drift.

2. **Known-operation catalogues.**  ``EVALUATION_OPERATIONS``,
   ``EDIT_OPERATIONS``, and ``KNOWN_OPERATIONS`` are exported as
   ``frozenset``s so that downstream code, tests, and introspection tools
   can reason about the supported surface without hard-coding string
   literals.  The request envelope does **not** gate on these sets:
   unknown operations pass through to the dispatcher, which raises
   ``AnimationUnsupportedOperationError``.  This preserves the two-layer
   design ("is the request well-formed?" vs "is the operation
   supported?") and keeps the door open for downstream extensions that
   subclass or decorate ``AnimationAgent``.

3. **Canonical result shapes.**  ``EvaluationResult`` and ``EditResult``
   describe the dictionaries returned by ``AnimationAgent.perform_task``
   so that callers, serialisers, and typed clients have a single
   authoritative reference rather than scattered ``dict[str, Any]``
   annotations.  The agent continues to return *plain dicts* matching
   these shapes — ``to_dict()`` produces an identical structure — so the
   typed wrappers serve documentation, testing, and a future migration to
   a structured-return API without altering the wire format.

Error handling
--------------
Every validation failure raises :class:`AnimationValidationError`, which
inherits :class:`ValueError` for backward compatibility with the
historical ``raise ValueError(...)`` call sites.  ``AnimationRequest``
itself never raises :class:`AnimationUnsupportedOperationError` — that
error is reserved for the dispatcher.

Immutability
------------
``AnimationRequest`` is a frozen dataclass and additionally wraps its
``payload`` mapping in :class:`types.MappingProxyType` so that
``frozen=True`` is not cosmetic: the top-level key set cannot be mutated
after construction.  *Nested* mutable values inside the payload (e.g.
lists of keyframe dictionaries) are **not** deep-frozen — this mirrors
the SGEO contract, which ``deepcopy``\\ s payloads when staging edits, so
the caller's nested data cannot be mutated through the request even
though the request exposes it by reference.  Callers that require
end-to-end deep immutability should pass ``MappingProxyType``-wrapped
nested structures, or serialise through :meth:`AnimationRequest.to_dict`
(which returns a fresh, mutable copy of the top-level mapping).

Note that :class:`MappingProxyType` is not picklable.  Callers that
serialise requests (for logging, cross-process dispatch, or audit trails)
should call :meth:`AnimationRequest.to_dict`, which returns plain Python
types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Mapping
from uuid import uuid4

from .utils.animation_errors import AnimationValidationError
from .utils.animation_helpers import *


# ---------------------------------------------------------------------------
# Operation catalogues
# ---------------------------------------------------------------------------
#: Operations dispatched to :func:`animation_evaluation.evaluate_animation`.
EVALUATION_OPERATIONS: Final[frozenset[str]] = frozenset({"evaluate"})

#: Operations dispatched to :func:`animation_commands.update_animation`.
EDIT_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "set_animation",
        "clear_animation",
        "add_keyframe",
        "update_keyframe",
        "delete_keyframe",
    }
)

#: Union of all built-in operations.  The request envelope does **not**
#: gate on this set; the dispatcher does.  Exporting it lets callers
#: enumerate the built-in surface without string literals.
KNOWN_OPERATIONS: Final[frozenset[str]] = EVALUATION_OPERATIONS | EDIT_OPERATIONS


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AnimationRequest:
    """Immutable, validated request envelope.

    Parameters
    ----------
    operation
        Non-empty string naming the operation to dispatch.  Leading and
        trailing whitespace is stripped before storage so that
        ``" evaluate "`` and ``"evaluate"`` are equivalent.  Membership
        in :data:`KNOWN_OPERATIONS` is *not* enforced here — unsupported
        operations surface as
        :class:`~.animation_errors.AnimationUnsupportedOperationError`
        at dispatch time.
    scene_id
        ``None`` when the operation does not target a specific scene,
        otherwise a non-empty string naming the scene to operate on.
        Whitespace is stripped before storage.  When provided,
        ``expected_revision`` is strongly recommended but not required.
    expected_revision
        Optional positive integer used for optimistic-concurrency control
        (compare-and-swap).  If supplied, the dispatcher raises
        :class:`~.animation_errors.AnimationRevisionConflictError` when
        the live scene revision does not match.  ``None`` disables the
        check (callers must serialise externally, or accept
        last-writer-wins semantics).
    payload
        Operation-specific arguments.  Validated only for the fact that
        it is a mapping — per-operation schema checks belong to the
        command/evaluation layer.  Wrapped in
        :class:`types.MappingProxyType` at construction so that the
        top-level mapping cannot be mutated through the request object.
    request_id
        Non-empty string identifier used to correlate responses with
        requests.  Defaults to a fresh :func:`uuid.uuid4().hex` (32 hex
        characters); callers may supply their own as long as it is a
        non-empty string.
    dry_run
        When ``True``, edit operations must not mutate the scene; the
        dispatcher returns the revision that *would* result and a
        ``would_change`` flag.  Ignored by ``"evaluate"``.

    Notes
    -----
    All validation happens in :meth:`__post_init__`.  The constructor
    never raises :class:`AnimationUnsupportedOperationError` — that is
    the dispatcher's responsibility.

    See Also
    --------
    EvaluationResult, EditResult
    """

    operation: str
    scene_id: str | None = None
    expected_revision: int | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid4().hex)
    dry_run: bool = False

    def __post_init__(self) -> None:
        # ``operation``: non-empty string; stripped for storage.
        ensure_non_empty_string(self.operation, "operation", strip=True)
        object.__setattr__(self, "operation", self.operation.strip())

        # ``scene_id``: ``None`` or non-empty string; stripped for storage.
        if self.scene_id is not None:
            ensure_non_empty_string(self.scene_id, "scene_id", strip=True)
            object.__setattr__(self, "scene_id", self.scene_id.strip())

        # ``expected_revision``: ``None`` or positive int (``bool`` rejected).
        if self.expected_revision is not None:
            ensure_positive_integer(self.expected_revision, "expected_revision")

        # ``payload``: mapping; frozen into a ``MappingProxyType``.
        if not isinstance(self.payload, Mapping):
            raise AnimationValidationError(
                f"payload must be a mapping, got {type(self.payload).__name__}",
                context={
                    "field": "payload",
                    "received_type": type(self.payload).__name__,
                },
            )
        if not isinstance(self.payload, MappingProxyType):
            # Copy into a fresh dict *first* so that a caller-supplied
            # mutable mapping cannot be mutated through this request.
            object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

        # ``request_id``: non-empty string; opaque, so not stripped.
        ensure_non_empty_string(self.request_id, "request_id", strip=False)

        # ``dry_run``: strict ``bool``.
        if not isinstance(self.dry_run, bool):
            raise AnimationValidationError(
                f"dry_run must be bool, got {type(self.dry_run).__name__}",
                context={
                    "field": "dry_run",
                    "received_type": type(self.dry_run).__name__,
                },
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a plain-``dict`` representation of this request.

        The result is a shallow dictionary whose ``payload`` entry is a
        plain :class:`dict` copy (nested values are *shared* by reference
        — callers needing full isolation should ``copy.deepcopy`` the
        returned structure themselves).  This method exists primarily
        for serialisation (``json.dumps``, logging, ``pickle``) where
        :class:`MappingProxyType` would be rejected.
        """
        return {
            "operation": self.operation,
            "scene_id": self.scene_id,
            "expected_revision": self.expected_revision,
            "payload": dict(self.payload),
            "request_id": self.request_id,
            "dry_run": self.dry_run,
        }

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        ignore_unknown: bool = False,
    ) -> "AnimationRequest":
        """Build a request from a mapping (e.g. parsed JSON / YAML).

        Parameters
        ----------
        data
            Mapping whose keys mirror the dataclass field names.
        ignore_unknown
            When ``False`` (default), unknown keys cause
            :class:`AnimationValidationError` — this surfaces typos in
            serialised requests early.  When ``True``, unknown keys are
            silently dropped; useful for forward-compatibility where the
            sender may include fields this version does not yet know.

        Raises
        ------
        AnimationValidationError
            If ``data`` is not a mapping, if the required ``operation``
            key is missing, or if ``ignore_unknown=False`` and ``data``
            contains keys outside the dataclass field set.
        """
        if not isinstance(data, Mapping):
            raise AnimationValidationError(
                f"request data must be a mapping, got {type(data).__name__}",
                context={"received_type": type(data).__name__},
            )
        known = AnimationRequest.__dataclass_fields__
        if "operation" not in data:
            raise AnimationValidationError(
                "request data is missing required 'operation' field",
                context={"keys": list(data.keys())},
            )
        if not ignore_unknown:
            unknown = [k for k in data if k not in known]
            if unknown:
                raise AnimationValidationError(
                    f"request data contains unknown fields: {unknown!r}",
                    context={"unknown": unknown, "known": list(known)},
                )
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Canonical result shape for ``"evaluate"`` operations.

    Mirrors the dictionary currently returned by
    :meth:`AnimationAgent.perform_task` for evaluation requests::

        {
            "status": "ok",
            "request_id": ...,
            "revision": ...,
            "time": ...,
            "objects": [...],
        }

    ``object_ids`` is stored as a tuple for hashability; :meth:`to_dict`
    serialises it back to a list to match the historical wire format.
    The agent continues to return plain dicts — this dataclass exists so
    that callers, tests, and future structured-return APIs share a
    single authoritative shape definition.
    """

    request_id: str
    scene_id: str
    revision: int
    time: float
    object_ids: tuple[str, ...]
    status: str = "ok"

    def __post_init__(self) -> None:
        ensure_non_empty_string(self.request_id, "request_id", strip=False)
        ensure_non_empty_string(self.scene_id, "scene_id", strip=True)
        ensure_non_negative_integer(self.revision, "revision")
        ensure_finite_scalar(self.time, "time")
        ensure_non_empty_string(self.status, "status", strip=False)
        if not isinstance(self.object_ids, tuple):
            raise AnimationValidationError(
                f"object_ids must be a tuple, got {type(self.object_ids).__name__}",
                context={
                    "field": "object_ids",
                    "received_type": type(self.object_ids).__name__,
                },
            )
        for index, object_id in enumerate(self.object_ids):
            ensure_non_empty_string(object_id, f"object_ids[{index}]", strip=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "request_id": self.request_id,
            "revision": self.revision,
            "time": self.time,
            "objects": list(self.object_ids),
        }


@dataclass(frozen=True, slots=True)
class EditResult:
    """Canonical result shape for edit operations.

    Covers ``set_animation``, ``clear_animation``, ``add_keyframe``,
    ``update_keyframe``, and ``delete_keyframe``.  The wire shape is::

        {
            "status": "ok",
            "request_id": ...,
            "revision": ...,
            # Optional, only on dry-run:
            "would_change": True | False,
        }

    ``would_change`` is ``None`` for committed edits and a strict
    ``bool`` for dry runs.  The :attr:`is_dry_run` property reflects this
    invariant directly, so callers never need to guess whether a
    ``None`` value means "not applicable" or "unknown" — and
    :meth:`to_dict` omits the key entirely for committed edits, matching
    the historical wire format produced by ``update_animation``.
    """

    request_id: str
    revision: int
    would_change: bool | None = None
    status: str = "ok"

    def __post_init__(self) -> None:
        ensure_non_empty_string(self.request_id, "request_id", strip=False)
        ensure_non_negative_integer(self.revision, "revision")
        if self.would_change is not None and not isinstance(self.would_change, bool):
            raise AnimationValidationError(
                f"would_change must be bool or None, got "
                f"{type(self.would_change).__name__}",
                context={
                    "field": "would_change",
                    "received_type": type(self.would_change).__name__,
                },
            )
        ensure_non_empty_string(self.status, "status", strip=False)

    @property
    def is_dry_run(self) -> bool:
        """``True`` iff this result was produced by a dry-run request.

        Formally: ``is_dry_run ⇔ would_change is not None``.
        """
        return self.would_change is not None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": self.status,
            "request_id": self.request_id,
            "revision": self.revision,
        }
        if self.would_change is not None:
            data["would_change"] = self.would_change
        return data


__all__ = [
    "EVALUATION_OPERATIONS",
    "EDIT_OPERATIONS",
    "KNOWN_OPERATIONS",
    "AnimationRequest",
    "EvaluationResult",
    "EditResult",
]