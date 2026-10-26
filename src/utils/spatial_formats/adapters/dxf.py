"""DXF subset through ezdxf, explicitly optional and restricted to supported entities."""
from ..common import ValidationError, DependencyUnavailableError, UnsupportedFeatureError
from ..sdrw import Drawing
from .contracts import Capabilities
from ..common.errors import ResourceLimitError
from pathlib import Path

_MAX_DXF_BYTES=64*1024*1024

def capabilities():
    try: backend=_ezdxf();available=True
    except DependencyUnavailableError: available=False
    return Capabilities("dxf","ezdxf",available,("R2010",) if available else (),
                        ("R2010",) if available else (),
                        ("POINT","LINE","LWPOLYLINE","CIRCLE","ARC"),
                        ("Only model-space planar entities", "Bulges, OCS extrusion and elevated entities refused",
                         "No block, annotation or DIMENSION preservation"))


def _ezdxf():
    try: import ezdxf
    except ImportError as exc: raise DependencyUnavailableError("install ezdxf for DXF exchange") from exc
    return ezdxf

def export_dxf(drawing:Drawing, path):
    ezdxf=_ezdxf()
    drawing.validate()
    doc=ezdxf.new("R2010")
    msp=doc.modelspace()
    for layer in drawing.layers:
        if layer["name"]!="0": doc.layers.new(layer["name"])
    for entity in drawing.entities:
        kind,g=entity["kind"],entity["geometry"]
        attrs={"layer":entity.get("layer","0")}
        if kind=="point": msp.add_point((*g["position"],0),dxfattribs=attrs)
        elif kind=="line": msp.add_line((*g["start"],0),(*g["end"],0),dxfattribs=attrs)
        elif kind=="polyline": msp.add_lwpolyline(g["points"],close=g.get("closed",False),dxfattribs=attrs)
        elif kind=="circle": msp.add_circle((*g["center"],0),g["radius"],dxfattribs=attrs)
        elif kind=="arc":
            from math import degrees
            msp.add_arc((*g["center"],0),g["radius"],degrees(g["start_angle"]),degrees(g["end_angle"]),dxfattribs=attrs)
    doc.saveas(str(path))

def import_dxf(path, *, unit="mm"):
    ezdxf=_ezdxf()
    source=Path(path)
    if not source.is_file() or source.is_symlink() or not 0<source.stat().st_size<=_MAX_DXF_BYTES:
        raise ResourceLimitError("invalid or oversized DXF file")
    doc=ezdxf.readfile(str(path))
    if any(True for _ in doc.layouts if _.name.lower() not in ("model",)):
        # Paper-space is not part of SDRW v0.1 and must not vanish silently.
        raise UnsupportedFeatureError("DXF paper-space layouts are not supported")
    drawing=Drawing(unit=unit)
    from math import radians
    for entity in doc.modelspace():
        kind=entity.dxftype()
        # Reject 3D/projected entities: SDRW only represents planar coordinates.
        if kind == "LWPOLYLINE":
            if abs(float(entity.dxf.elevation)) > 1e-9:
                raise UnsupportedFeatureError("elevated DXF polylines require projection")
            points=entity.get_points("xyb")
            if any(abs(float(point[2]))>1e-12 for point in points):
                raise UnsupportedFeatureError("DXF polyline bulge arcs require exact curve decomposition")
        elif kind in {"POINT","LINE","CIRCLE","ARC"}:
            values = ([entity.dxf.start,entity.dxf.end] if kind=="LINE" else
                      [entity.dxf.location] if kind=="POINT" else [entity.dxf.center])
            if any(abs(float(p[2]))>1e-9 for p in values):
                raise UnsupportedFeatureError("3D/elevated DXF entity cannot be silently projected")

        if kind not in {"POINT","LINE","LWPOLYLINE","CIRCLE","ARC"}:
            raise UnsupportedFeatureError("unsupported DXF entity: "+kind)
        layer=entity.dxf.layer
        if layer not in {x["name"] for x in drawing.layers}: drawing.add_layer(layer)
        if kind=="POINT":
            drawing.add("point",{"position":list(entity.dxf.location)[:2]},layer=layer)
        elif kind=="LINE":
            drawing.add("line",{"start":list(entity.dxf.start)[:2],"end":list(entity.dxf.end)[:2]},layer=layer)
        elif kind=="LWPOLYLINE":
            drawing.add("polyline",{"points":[list(p[:2]) for p in entity.get_points()],"closed":bool(entity.closed)},layer=layer)
        elif kind=="CIRCLE":
            drawing.add("circle",{"center":list(entity.dxf.center)[:2],"radius":entity.dxf.radius},layer=layer)
        else:
            drawing.add("arc",{"center":list(entity.dxf.center)[:2],"radius":entity.dxf.radius,
                              "start_angle":radians(entity.dxf.start_angle),"end_angle":radians(entity.dxf.end_angle if entity.dxf.end_angle>entity.dxf.start_angle else entity.dxf.end_angle+360)},layer=layer)
    return drawing
