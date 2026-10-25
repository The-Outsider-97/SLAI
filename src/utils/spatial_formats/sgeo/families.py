"""Deterministic, declarative bounded parameters for primitive SGEO families."""
from __future__ import annotations
from ..common import ValidationError, require_id, finite, vector
from .parameters import evaluate_parameters

_TYPES={"length","area","volume","angle","number","integer","boolean"}
_GENERATORS={"builtin.box_v1","builtin.sphere_v1","builtin.cylinder_v1"}

def validate_family(family):
    if not isinstance(family,dict) or set(family)-{"id","name","parameters","generator"}:
        raise ValidationError("invalid family fields")
    require_id(family.get("id"))
    if not isinstance(family.get("name"),str) or not family["name"] or len(family["name"])>255:
        raise ValidationError("invalid family name")
    parameters=family.get("parameters")
    if not isinstance(parameters,dict) or len(parameters)>128:
        raise ValidationError("invalid family parameter table")
    for name,spec in parameters.items():
        if not isinstance(name,str) or not name.isidentifier() or len(name)>64:
            raise ValidationError("invalid parameter name")
        if not isinstance(spec,dict) or set(spec)-{"type","default","min","max","expression","read_only","description"} or spec.get("type") not in _TYPES:
            raise ValidationError("invalid family parameter specification")
        if "expression" in spec:
            continue
        if spec["type"]=="boolean":
            if type(spec.get("default")) is not bool:
                raise ValidationError("boolean parameter default must be Boolean")
            continue
        value=finite(spec.get("default"),name)
        if spec["type"]=="integer" and value!=int(value):
            raise ValidationError("integer parameter must have an integer default")
        low=finite(spec.get("min",-1e12),name+".min")
        high=finite(spec.get("max",1e12),name+".max")
        if low>high or not low<=value<=high:
            raise ValidationError("family default violates parameter bounds")
        if spec["type"]=="length" and low<0:
            raise ValidationError("length parameter minimum must be nonnegative")
    generator=family.get("generator")
    if not isinstance(generator,dict) or set(generator)!={"kind"} or generator["kind"] not in _GENERATORS:
        raise ValidationError("untrusted or unsupported family generator")
    resolve_parameters(family)
    return family

def resolve_parameters(family, overrides=None):
    """Evaluate a bounded typed parameter DAG without partial writes."""
    if not isinstance(family,dict) or not isinstance(family.get("parameters"),dict):
        raise ValidationError("family parameters must be an object")
    return evaluate_parameters(family["parameters"],overrides)

def generate_geometry(family,overrides=None):
    """Only allowlisted deterministic builtins; arbitrary expressions are never evaluated."""
    validate_family(family)
    p=resolve_parameters(family,overrides)
    kind=family["generator"]["kind"]
    def positive(name):
        value=p.get(name)
        if value is None or value<=0:
            raise ValidationError("generator requires positive parameter "+name)
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
