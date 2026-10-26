"""OCCT-backed STEP precision-geometry boundary with exact source retention.

Import via OCP when installed. Preserve original STEP bytes separately from a
derived tessellated preview rather than falsely claiming that a triangle mesh
round-trips exact CAD topology. STEP export supports only basic analytic boxes
with a real CAD kernel and validates output by reopening it.
"""
from __future__ import annotations
from pathlib import Path
from ..common.errors import DependencyUnavailableError, UnsupportedFeatureError, ValidationError, ResourceLimitError
from ..sgeo.scene import Scene
from ..common.helpers import digest
from .contracts import Capabilities

_MAX_STEP=64*1024*1024

def _occt():
    try:
        from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.BRepCheck import BRepCheck_Analyzer
    except ImportError as exc:
        raise DependencyUnavailableError("STEP conversion requires compatible OCP/Open CASCADE bindings") from exc
    return STEPControl_Reader,STEPControl_Writer,STEPControl_AsIs,IFSelect_RetDone,BRepPrimAPI_MakeBox,BRepCheck_Analyzer

def capabilities():
    try: _occt();available=True
    except DependencyUnavailableError: available=False
    return Capabilities("step","OCP (Open CASCADE)",available,("AP203","AP214","AP242") if available else (),
                        ("AP214",) if available else (),("B-rep-source","analytic-box"),
                        ("Original STEP data retained as source artifact; preview tessellation not implemented",
                         "Mesh-only and assembly STEP export unsupported"))

def validate_step(path):
    source=Path(path)
    if not source.is_file() or source.is_symlink() or not 0<source.stat().st_size<=_MAX_STEP:
        raise ResourceLimitError("STEP file is absent or exceeds input budget")
    with source.open("rb") as f:
        header=f.read(4096)
    if b"ISO-10303-21" not in header or b"HEADER;" not in header:
        raise ValidationError("STEP Part 21 header missing")
    return source

def import_step(path):
    """Validate exact CAD shape and return provenance rather than silently tessellating."""
    reader_cls,_,_,done,_,analyzer_cls=_occt()
    source=validate_step(path)
    reader=reader_cls()
    if reader.ReadFile(str(source))!=done:
        raise ValidationError("Open CASCADE could not read STEP file")
    count=reader.TransferRoots()
    if count<=0:
        raise ValidationError("STEP contains no transferred CAD shapes")
    shape=reader.OneShape()
    if shape.IsNull() or not analyzer_cls(shape).IsValid():
        raise ValidationError("imported STEP B-rep is invalid")
    return {"source_sha256":digest(source.read_bytes()),"source_path":str(source.resolve()),
            "transferred_roots":int(count),"exact_brep_available":True}

def export_step(scene,path,*,overwrite=False):
    """Write a real STEP solid only for a single axis-aligned box, using OCCT."""
    _,writer_cls,as_is,done,make_box,analyzer=_occt()
    scene.validate()
    if len(scene.objects)!=1 or scene.families:
        raise UnsupportedFeatureError("STEP export supports only one nonparametric axis-aligned box")
    obj=scene.objects[0]
    geometry=obj.get("geometry")
    from ..common.transforms import IDENTITY
    if not isinstance(geometry,dict) or geometry.get("kind")!="box" or tuple(obj.get("transform",IDENTITY))!=IDENTITY:
        raise UnsupportedFeatureError("STEP export supports only an untransformed analytic box")
    if scene.unit!="m" or obj.get("metadata") or scene.metadata:
        raise UnsupportedFeatureError("STEP export requires metre units and no unrepresentable semantic metadata")
    sx,sy,sz=geometry["size"]
    # OCCT STEP translators traditionally operate in millimetres; explicit
    # scaling is necessary and round-trip unit handling must be independently tested.
    from OCP.gp import gp_Pnt
    x,y,z=geometry["center"]
    box=make_box(gp_Pnt((x-sx/2)*1000,(y-sy/2)*1000,(z-sz/2)*1000),sx*1000,sy*1000,sz*1000).Shape()
    if not analyzer(box).IsValid():
        raise ValidationError("generated STEP B-rep failed geometry validation")
    writer=writer_cls()
    if writer.Transfer(box,as_is)!=done:
        raise ValidationError("STEP shape transfer failed")
    destination=Path(path)
    if destination.exists() and not overwrite:
        raise ValidationError("STEP destination already exists")
    if writer.Write(str(destination))!=done:
        raise ValidationError("STEP export failed")
    validate_step(destination)
    return destination
