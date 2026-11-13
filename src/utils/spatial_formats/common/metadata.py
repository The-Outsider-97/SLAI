"""Validated metadata and attributable claims; none are executable.

Metadata keys are namespaced (``namespace:name``) or, for SLAI-owned
annotations, prefixed with ``slai.``. Claims additionally carry a
semantic ``kind`` (observation / measurement / derivation / inference /
assertion) so downstream consumers can distinguish authoritative facts
from uncertain hypotheses without inventing parallel metadata formats.
"""

from typing import cast

from .errors import ValidationError, ResourceLimitError
from .helpers import json_safe, finite, MAX_STRING
from .ids import require_id

MAX_PREDICATE = 255
MAX_EVIDENCE_ITEMS = 64
MAX_PROVENANCE_DIGEST = 64

_CLAIM_KINDS = frozenset({"observation", "measurement", "derivation", "inference", "assertion"})
_PROVENANCE_FIELDS = frozenset({"source", "adapter", "adapter_version", "revision", "timestamp", "digest"})


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _require_metadata_key(key: str) -> None:
    if key.startswith("slai."):
        if len(key) == len("slai."):
            raise ValidationError("empty slai namespace key", context={"key": key})
        return
    namespace, sep, name = key.partition(":")
    if not sep or not namespace or not name or ":" in name:
        raise ValidationError("metadata keys must be namespaced as namespace:name", context={"key": key})


def require_metadata(value) -> dict:
    if not isinstance(value, dict):
        raise ValidationError("metadata must be a JSON object")
    safe = json_safe(value)
    for key in safe:
        _require_metadata_key(key)
    return cast(dict, safe)


def merge_metadata(*sources, on_conflict: str = "error") -> dict:
    """Merge one or more namespaced metadata dictionaries.

    ``on_conflict`` controls differing values for the same key:

    * ``"error"`` (default) — raise :class:`ValidationError`.
    * ``"first"`` — earlier sources win.
    * ``"last"``  — later sources win.

    All sources are validated with :func:`require_metadata`; the return
    value is a detached plain dict.
    """
    if on_conflict not in {"error", "first", "last"}:
        raise ValueError("on_conflict must be 'error', 'first' or 'last'")
    result: dict = {}
    for source in sources:
        safe = require_metadata(source)
        for key, value in safe.items():
            if key in result and result[key] != value:
                if on_conflict == "error":
                    raise ValidationError(
                        "metadata conflict on key " + repr(key),
                        context={"key": key},
                    )
                if on_conflict == "first":
                    continue
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

def _require_evidence(evidence):
    if isinstance(evidence, str):
        if not evidence or len(evidence) > MAX_STRING:
            raise ValidationError(
                "claim evidence must be a nonempty bounded identifier",
                context={"length": len(evidence)},
            )
        return evidence
    if isinstance(evidence, (list, tuple)):
        if not 1 <= len(evidence) <= MAX_EVIDENCE_ITEMS:
            raise ValidationError(
                "claim evidence list must have 1.." + str(MAX_EVIDENCE_ITEMS) + " entries",
                context={"length": len(evidence)},
            )
        return [_require_evidence(item) for item in evidence]
    raise ValidationError(
        "claim evidence must be a string or list of strings",
        context={"type": type(evidence).__name__},
    )


def require_provenance(provenance) -> dict:
    """Validate a structured provenance record.

    Recognised fields: ``source``, ``adapter``, ``adapter_version``,
    ``revision``, ``timestamp``, ``digest``. All are optional; unknown
    fields are rejected so that future additions become a schema change
    rather than silent drift.
    """
    if not isinstance(provenance, dict):
        raise ValidationError("provenance must be an object")
    extra = set(provenance) - _PROVENANCE_FIELDS
    if extra:
        raise ValidationError(
            "unsupported provenance fields: " + ", ".join(sorted(extra)),
            context={"fields": sorted(extra)},
        )
    for name in ("source", "adapter", "adapter_version", "timestamp"):
        if name in provenance:
            value = provenance[name]
            if not isinstance(value, str) or not value:
                raise ValidationError(
                    "provenance." + name + " must be a nonempty string",
                    context={"field": name},
                )
    if "revision" in provenance:
        revision = provenance["revision"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValidationError(
                "provenance.revision must be a nonnegative integer",
                context={"revision": revision},
            )
    if "digest" in provenance:
        value = provenance["digest"]
        if (
            not isinstance(value, str)
            or len(value) != MAX_PROVENANCE_DIGEST
            or any(ch not in "0123456789abcdef" for ch in value)
        ):
            raise ValidationError(
                "provenance.digest must be a lowercase hex SHA-256",
            )
    return cast(dict, json_safe(provenance))


def require_claim(claim) -> dict:
    if not isinstance(claim, dict):
        raise ValidationError("claim must be an object")
    subject_value = claim.get("subject")
    if not isinstance(subject_value, str):
        raise ValidationError("claim subject must be a string")
    subject = require_id(subject_value)
    predicate = claim.get("predicate")
    if (
        not isinstance(predicate, str)
        or not predicate
        or len(predicate) > MAX_PREDICATE
    ):
        raise ValidationError(
            "claim requires a bounded nonempty predicate",
            context={"predicate": predicate if isinstance(predicate, str) else None},
        )
    confidence = finite(claim.get("confidence", 1.0), "confidence", limit=1)
    if confidence < 0:
        raise ValidationError(
            "confidence must be between 0 and 1",
            context={"confidence": confidence},
        )
    kind = claim.get("kind", "assertion")
    if kind not in _CLAIM_KINDS:
        raise ValidationError(
            "unsupported claim kind: " + str(kind),
            context={"kind": kind, "allowed": sorted(_CLAIM_KINDS)},
        )

    result = dict(claim)
    result["subject"] = subject
    result["predicate"] = predicate
    result["kind"] = kind
    result["confidence"] = confidence
    if "evidence" in claim:
        result["evidence"] = _require_evidence(claim["evidence"])
    if "provenance" in claim:
        result["provenance"] = require_provenance(claim["provenance"])
    return cast(dict, json_safe(result))


def require_claim_list(values, *, max_items: int = 4096) -> list[dict]:
    if not isinstance(values, (list, tuple)):
        raise ValidationError("claim list must be an array")
    if len(values) > max_items:
        raise ResourceLimitError(
            "claim list exceeds budget",
            context={"length": len(values), "limit": max_items},
        )
    return [require_claim(claim) for claim in values]


__all__ = [
    "require_metadata",
    "require_claim",
    "require_claim_list",
    "require_provenance",
    "merge_metadata",
]
