"""Safe SVG 1.1 subset exchange for native SDRW primitives.

Unsupported SVG features fail closed. No external resources, embedded scripts, CSS,
text, use-elements or general path mini-language are interpreted.
"""
from __future__ import annotations
import io
import math
import xml.etree.ElementTree as ET
from ..common import ValidationError, UnsupportedFeatureError, finite
from ..sdrw import Drawing

NS="http://www.w3.org/2000/svg"
MAX_SVG_BYTES=8*1024*1024

def export_svg(drawing: Drawing) -> str:
    drawing.validate()
    root=ET.Element("svg",{"xmlns":NS,"version":"1.1","data-unit":drawing.unit})
    for entity in drawing.entities:
        kind,g=entity["kind"],entity["geometry"]
        attrs={"data-id":entity["id"],"data-layer":entity.get("layer","0"),"data-kind":kind}
        if kind=="point":
            tag="circle";attrs.update({"cx":str(g["position"][0]),"cy":str(g["position"][1]),"r":"0.01"})
        elif kind=="line":
            tag="line";attrs.update({"x1":str(g["start"][0]),"y1":str(g["start"][1]),"x2":str(g["end"][0]),"y2":str(g["end"][1])})
        elif kind=="polyline":
            tag="polygon" if g.get("closed") else "polyline"
            attrs["points"]=" ".join(f"{p[0]},{p[1]}" for p in g["points"])
        elif kind=="circle":
            tag="circle";attrs.update({"cx":str(g["center"][0]),"cy":str(g["center"][1]),"r":str(g["radius"])})
        elif kind=="arc":
            tag="path"
            start,end=g["start_angle"],g["end_angle"];cx,cy=g["center"];r=g["radius"]
            # Full sweeps require two arcs; unsupported until an explicit segment policy exists.
            if end-start>=2*math.pi-1e-10:
                raise UnsupportedFeatureError("full-sweep arc requires decomposition")
            ax,ay=cx+r*math.cos(start),cy+r*math.sin(start)
            bx,by=cx+r*math.cos(end),cy+r*math.sin(end)
            large=int(end-start>math.pi)
            attrs["d"]=f"M {ax} {ay} A {r} {r} 0 {large} 1 {bx} {by}"
        ET.SubElement(root,tag,attrs)
    return ET.tostring(root,encoding="unicode",short_empty_elements=True)

def import_svg(raw: str|bytes, *, unit="mm") -> Drawing:
    data=raw.encode("utf-8") if isinstance(raw,str) else raw
    if not isinstance(data,bytes) or len(data)>MAX_SVG_BYTES:
        raise ValidationError("SVG input exceeds byte limit")
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise UnsupportedFeatureError("SVG document declarations are prohibited")
    try:
        root=ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValidationError("invalid SVG XML") from exc
    if root.tag not in (f"{{{NS}}}svg","svg"):
        raise ValidationError("SVG root element required")
    drawing=Drawing(unit=unit)
    tag_names={"line","circle","polyline","polygon"}
    for child in root:
        tag=child.tag.rsplit("}",1)[-1]
        if tag not in tag_names:
            raise UnsupportedFeatureError("unsupported SVG element: "+tag)
        if any(key.lower().startswith("on") or key in {"href","style","transform"} or key.endswith("}href") for key in child.attrib):
            raise UnsupportedFeatureError("SVG event handlers, references, styles and transforms are not allowed")
        layer=child.attrib.get("data-layer","0")
        if layer not in {x["name"] for x in drawing.layers}:
            drawing.add_layer(layer)
        def n(key):
            try:
                return finite(float(child.attrib[key]),key)
            except (KeyError,ValueError,TypeError) as exc:
                raise ValidationError("invalid SVG numeric attribute: "+key) from exc
        if tag=="circle" and child.attrib.get("data-kind")=="point":
            drawing.add("point",{"position":[n("cx"),n("cy")]},layer=layer)
        elif tag=="line":
            drawing.add("line",{"start":[n("x1"),n("y1")],"end":[n("x2"),n("y2")]},layer=layer)
        elif tag=="circle":
            drawing.add("circle",{"center":[n("cx"),n("cy")],"radius":n("r")},layer=layer)
        else:
            points=[]
            source=child.attrib.get("points","").replace(","," ").split()
            if len(source)%2 or len(source)>200000:
                raise ValidationError("invalid SVG point coordinates")
            for i in range(0,len(source),2):
                try: points.append([finite(float(source[i])),finite(float(source[i+1]))])
                except ValueError as exc: raise ValidationError("invalid SVG points") from exc
            drawing.add("polyline",{"points":points,"closed":tag=="polygon"},layer=layer)
    return drawing
