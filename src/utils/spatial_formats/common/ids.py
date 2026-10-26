"""Creation and validation of immutable spatial identifiers."""
from uuid import UUID, uuid4
from .errors import ValidationError

def new_id() -> str:
    return str(uuid4())

def require_id(value: str) -> str:
    if not isinstance(value, str) or len(value) > 64:
        raise ValidationError("object ID must be a UUID string")
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValidationError("invalid UUID") from exc
    if str(parsed) != value.lower():
        raise ValidationError("UUID must use canonical hyphenated notation")
    return str(parsed)

def assert_unique(values) -> None:
    ids = [require_id(value) for value in values]
    if len(ids) != len(set(ids)):
        raise ValidationError("duplicate object identifier")
