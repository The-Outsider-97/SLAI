"""Shared strict conversion policy, capability and evidence contracts."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from ..common.errors import UnsupportedFeatureError, ValidationError
from ..common.helpers import json_safe
from ..verification.loss_report import ConversionReport

class Policy(str, Enum):
    STRICT="strict"
    LOSSLESS_WHERE_SUPPORTED="lossless_where_supported"
    ALLOW_APPROXIMATION="allow_approximation"
    BEST_EFFORT="best_effort"

@dataclass(frozen=True)
class Capabilities:
    format: str
    backend: str
    available: bool
    import_versions: tuple[str, ...]
    export_versions: tuple[str, ...]
    entities: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    def as_dict(self): return json_safe(self.__dict__)

@dataclass(frozen=True)
class ConversionResult:
    document: object
    report: ConversionReport

def enforce_policy(policy: Policy | str, *, losses=(), approximations=()):
    try: setting=Policy(policy)
    except ValueError as exc: raise ValidationError("unrecognized conversion policy") from exc
    if setting in (Policy.STRICT, Policy.LOSSLESS_WHERE_SUPPORTED) and (losses or approximations):
        raise UnsupportedFeatureError("conversion would lose information: "+", ".join((*losses,*approximations)))
    if setting==Policy.ALLOW_APPROXIMATION and losses:
        raise UnsupportedFeatureError("conversion would discard unsupported semantics: "+", ".join(losses))
    return setting
