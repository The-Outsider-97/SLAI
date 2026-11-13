"""SDRW linetypes: named dash patterns referenced by layers and entity styles.

A pattern is a list of numbers in *drawing units*, DXF style: a positive number
is a dash, a negative number is a gap and ``0`` is a dot. The pattern repeats
along the stroke. ``continuous`` is built in (solid) and cannot be redefined;
an absent linetype means "by layer".
"""
from __future__ import annotations

from ..common import ValidationError, finite

BUILTIN_LINETYPE = "continuous"
_RESERVED = frozenset({"continuous", "bylayer", "byblock"})
MAX_LINETYPES = 1_000
MAX_PATTERN_ELEMENTS = 32
MAX_PATTERN_VALUE = 1e9


def validate_linetype(record) -> dict:
    """Validate one registry record ``{name, pattern[, description]}``."""
    if not isinstance(record, dict) or set(record) - {"name", "description", "pattern"}:
        raise ValidationError("invalid linetype record")
    name = record.get("name")
    if not isinstance(name, str) or not name or len(name) > 255:
        raise ValidationError("invalid linetype name")
    if name.lower() in _RESERVED:
        raise ValidationError("linetype name is reserved: " + name)
    if "description" in record and (not isinstance(record["description"], str) or len(record["description"]) > 255):
        raise ValidationError("invalid linetype description")
    pattern = record.get("pattern")
    if not isinstance(pattern, list) or not 1 <= len(pattern) <= MAX_PATTERN_ELEMENTS:
        raise ValidationError("linetype pattern requires 1..%d elements" % MAX_PATTERN_ELEMENTS)
    values = [finite(v, "linetype element", limit=MAX_PATTERN_VALUE) for v in pattern]
    if not any(v >= 0 for v in values):
        raise ValidationError("linetype pattern needs at least one dash or dot")
    if sum(abs(v) for v in values) <= 0:
        raise ValidationError("linetype pattern has zero length")
    return record


def check_linetype_ref(name, known) -> str:
    """Validate a reference. ``known`` is the set of defined names (None skips the lookup)."""
    from ..common.errors import ReferenceError
    if not isinstance(name, str) or not name or len(name) > 255:
        raise ValidationError("invalid linetype reference")
    lowered = name.lower()
    if lowered in ("continuous", "bylayer"):
        return name
    if lowered == "byblock":
        raise ValidationError("BYBLOCK linetypes are not supported")
    if known is not None and name not in known:
        raise ReferenceError("reference to unknown linetype: " + name, context={"linetype": name})
    return name


def dash_array(record, *, scale: float = 1.0):
    """SVG ``stroke-dasharray`` numbers for a pattern (dots become tiny dashes)."""
    values = [abs(v) * scale for v in record["pattern"]]
    values = [v if v > 0 else 1e-6 * max(scale, 1e-12) for v in values]
    return values if len(values) % 2 == 0 else values * 2
