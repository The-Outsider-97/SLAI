"""Deterministic, declarative bounded parameters for primitive SGEO families, with named family types.

A family is ``{id, name, parameters, generator[, types]}``. ``parameters`` is the typed table (see ``parameters.py``); ``types`` maps a
type name to the parameter values that type fixes (``{"D-800": {"width": 0.8}}``). Value precedence, lowest to highest:

1. the parameter's ``default`` (or its formula, which is never overridable),
2. the instance's named type, when it has one,
3. the instance's own overrides.

A parameter whose ``scope`` is ``"type"`` may be set by a type but not overridden on an instance. A family without ``types`` behaves exactly
as before, so existing documents are unchanged.
"""
from __future__ import annotations
from ..common import ValidationError, require_id
from .parameters import evaluate_parameters

_GENERATORS={"builtin.box_v1","builtin.sphere_v1","builtin.cylinder_v1"}
MAX_FAMILY_TYPES=256
MAX_TYPE_NAME=80
_FAMILY_FIELDS={"id","name","parameters","generator","types"}


def validate_type_name(name):
    if not isinstance(name,str) or not name.strip() or name!=name.strip() or len(name)>MAX_TYPE_NAME or any(ord(c)<32 for c in name):
        raise ValidationError("invalid family type name",context={"type":name if isinstance(name,str) else None})
    return name


def validate_family(family):
    if not isinstance(family,dict) or set(family)-_FAMILY_FIELDS:
        raise ValidationError("invalid family fields")
    require_id(family.get("id"))
    if not isinstance(family.get("name"),str) or not family["name"] or len(family["name"])>255:
        raise ValidationError("invalid family name")
    parameters=family.get("parameters")
    if not isinstance(parameters,dict) or len(parameters)>128:
        raise ValidationError("invalid family parameter table")
    generator=family.get("generator")
    if not isinstance(generator,dict) or set(generator)!={"kind"} or not isinstance(generator.get("kind"),str) or generator["kind"] not in _GENERATORS:
        raise ValidationError("untrusted or unsupported family generator")
    resolve_parameters(family)
    types=family.get("types")
    if types is not None:
        if not isinstance(types,dict) or len(types)>MAX_FAMILY_TYPES:
            raise ValidationError("family types must be a bounded object",context={"family":family["id"]})
        seen=set()
        for name,values in types.items():
            validate_type_name(name)
            if name.lower() in seen:
                raise ValidationError("duplicate family type name (names are case-insensitive)",context={"type":name})
            seen.add(name.lower())
            if not isinstance(values,dict):
                raise ValidationError("a family type maps parameter names to values",context={"type":name})
            try:
                _build(family,resolve_parameters(family,None,name))
            except ValidationError as exc:
                raise type(exc)("family type '%s': %s"%(name,exc.args[0] if exc.args else exc),context={**(exc.context or {}),"type":name}) from exc
    return family


def resolve_parameters(family, overrides=None, type_name=None):
    """Evaluate a bounded typed parameter DAG without partial writes: defaults < the named type < ``overrides``."""
    if not isinstance(family,dict) or not isinstance(family.get("parameters"),dict):
        raise ValidationError("family parameters must be an object")
    merged={}
    if type_name is not None:
        types=family.get("types") or {}
        if type_name not in types:
            raise ValidationError("unknown family type: %s"%type_name,context={"type":type_name,"known":sorted(types)})
        merged.update(types[type_name])
    if overrides:
        if not isinstance(overrides,dict):
            raise ValidationError("invalid parameter overrides")
        for name in overrides:
            if family["parameters"].get(name,{}).get("scope")=="type":
                raise ValidationError("%s is a type parameter: change it in the family type, not on the instance"%name,
                                      context={"parameter":name})
        merged.update(overrides)
    return evaluate_parameters(family["parameters"],merged)


def _build(family, p):
    kind=family["generator"]["kind"]
    def positive(name):
        value=p.get(name)
        if family["parameters"].get(name, {}).get("type") != "length" or value is None or value<=0:
            raise ValidationError("generator requires positive parameter "+name,context={"parameter":name})
        return value
    if kind=="builtin.box_v1":
        result={"kind":"box","center":[0,0,0],
                "size":[positive("width"),positive("depth"),positive("height")]}
    elif kind=="builtin.sphere_v1":
        result={"kind":"sphere","center":[0,0,0],"radius":positive("radius")}
    elif kind=="builtin.cylinder_v1":
        result={"kind":"cylinder","center":[0,0,0],"radius":positive("radius"),
                "height":positive("height")}
    else:
        raise ValidationError("unrecognized geometry generator")
    from .geometry import validate_geometry
    return validate_geometry(result)


def generate_geometry(family,overrides=None,type_name=None):
    """Only allowlisted deterministic builtins; arbitrary expressions are never evaluated."""
    validate_family(family)
    return _build(family,resolve_parameters(family,overrides,type_name))
