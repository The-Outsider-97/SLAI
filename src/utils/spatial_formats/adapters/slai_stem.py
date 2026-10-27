"""Narrow stable quantity conversions; no eagerly imported STEM scientific stack."""
from ..common.units import convert_area,convert_length,convert_volume,require_unit
from ..common.helpers import finite
from ..common.errors import ValidationError

def quantity(value,unit,kind="length"):
    """Build a validated scalar quantity DTO at the file boundary."""
    require_unit(unit)
    if kind not in ("length", "area", "volume"):
        raise ValidationError("unsupported physical quantity")
    return {"value":finite(value),"unit":unit,"dimension":kind}

def convert_quantity(record,target_unit):
    if not isinstance(record,dict):
        raise ValidationError("quantity must be a mapping")
    kind=record.get("dimension")
    if not isinstance(kind,str):
        raise ValidationError("unknown quantity dimension")
    conversion={"length":convert_length,"area":convert_area,"volume":convert_volume}.get(kind)
    if conversion is None:
        raise ValidationError("unknown quantity dimension")
    value=finite(record.get("value"))
    unit=record.get("unit")
    if not isinstance(unit,str):
        raise ValidationError("quantity unit must be a string")
    return quantity(conversion(value,unit,target_unit),target_unit,kind)
