"""Explicit STEP boundary: a B-rep-capable kernel is required."""
from ..common.errors import UnsupportedFeatureError

def import_step(*args,**kwargs):
    raise UnsupportedFeatureError("STEP conversion requires a validated OCCT-class B-rep adapter")

def export_step(*args,**kwargs):
    raise UnsupportedFeatureError("SGEO mesh and analytic primitives are not a STEP B-rep implementation")
