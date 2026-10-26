"""DXF subset through ezdxf, explicitly optional and restricted to supported entities."""
from ..common import ValidationError, DependencyUnavailableError, UnsupportedFeatureError
from ..sdrw import Drawing
from .contracts import Capabilities
from ..common.errors import ResourceLimitError
from pathlib import Path

_UNIT_CODES={"in":1,"ft":2,"mm":4,"cm":5,"m":6,"km":7}
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
    doc.units=_UNIT_CODES[drawing.unit]
    msp=doc.modelspace()
    for layer in drawing.layers:
        if layer["name"]!="0": doc.layers.new(layer["name"])
        if not layer.get("visible",True): doc.layers.get(layer["name"]).off()
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

def import_dxf(path, *, unit=None):
    ezdxf=_ezdxf()
    source=Path(path)
    if not source.is_file() or source.is_symlink() or not 0<source.stat().st_size<=_MAX_DXF_BYTES:
        raise ResourceLimitError("invalid or oversized DXF file")
    doc=ezdxf.readfile(str(path))
    if any(len(layout)>0 for layout in doc.layouts if layout.name.lower() != "model"):
        # Paper-space is not part of SDRW v0.1 and must not vanish silently.
        raise UnsupportedFeatureError("DXF paper-space layouts are not supported")
    declared=next((name for name,code in _UNIT_CODES.items() if code==doc.units),None)
    if unit is None:
        if declared is None:
            raise UnsupportedFeatureError("DXF units require an explicit supported unit override")
        unit=declared
    drawing=Drawing(unit=unit,layers=[{"name":layer.dxf.name,"visible":not layer.is_off() and not layer.is_frozen()}
                                    for layer in doc.layers if layer.dxf.name != "Defpoints"])

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

        if tuple(entity.dxf.get("extrusion",(0,0,1))) != (0,0,1):
            raise UnsupportedFeatureError("DXF OCS extrusion requires explicit conversion")
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
