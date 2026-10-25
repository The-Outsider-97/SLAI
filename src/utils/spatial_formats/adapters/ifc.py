"""Explicit IFC boundary: refusing lossy or unverified BIM conversion is required."""
from ..common.errors import UnsupportedFeatureError

def import_ifc(*args,**kwargs):
    raise UnsupportedFeatureError("IFC semantic conversion is not implemented or verified for SGEO v0.1")

def export_ifc(*args,**kwargs):
    raise UnsupportedFeatureError("SGEO primitives do not preserve IFC entities, property sets and placements")
