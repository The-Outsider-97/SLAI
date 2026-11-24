"""Safe SVG 1.1 subset exchange for native SDRW primitives.

Unsupported SVG features fail closed. No external resources, embedded scripts, CSS,
text, use-elements or general path mini-language are interpreted.
"""
from __future__ import annotations
import math
import xml.etree.ElementTree as ET
from ..common import ValidationError, UnsupportedFeatureError, finite
from ..common.ids import is_valid_id, new_id
from ..sdrw import Drawing
from ..sdrw.annotations import dimension_primitives, hatch_lines, loop_polygon, text_layout
from ..sdrw.linetypes import dash_array
from ..sdrw.operations import expand_insert
from ..sdrw import curves as _curves

NS="http://www.w3.org/2000/svg"
MAX_SVG_BYTES=8*1024*1024

def _num(value) -> str:
    return repr(float(value))


def _arc_attrs(cx, cy, r, start, end, attrs):
    """``(tag, attrs)`` for an arc; a full sweep is exactly a circle (SVG arcs cannot start and end at one point)."""
    if end-start>=2*math.pi-1e-10:
        attrs.update({"cx":str(cx),"cy":str(cy),"r":str(r),"data-kind":"circle"})
        return "circle",attrs
    ax,ay=cx+r*math.cos(start),cy+r*math.sin(start)
    bx,by=cx+r*math.cos(end),cy+r*math.sin(end)
    large=int(end-start>math.pi)
    attrs["d"]=f"M {ax!r} {ay!r} A {r!r} {r!r} 0 {large} 1 {bx!r} {by!r}"
    # Exact parameters travel alongside the display path so reimport is lossless.
    attrs.update({"data-cx":repr(cx),"data-cy":repr(cy),"data-r":repr(r),
                  "data-start":repr(start),"data-end":repr(end)})
    return "path",attrs


def _spline_path_tokens(g):
    """Display path data of a spline, derived only from its exact parameters: cubic Béziers (``C``) up to degree 3; above, a polyline (``L``) within ``svg_spline_tolerance`` of the curve (guaranteed: SVG has no primitive for it)."""
    cubic=_curves.cubic_segments(g)
    if cubic is not None:
        tokens=["M",f"{cubic[0][0][0]!r},{cubic[0][0][1]!r}"]
        for seg in cubic:
            tokens.append("C")
            tokens.extend(f"{q[0]!r},{q[1]!r}" for q in seg[1:])
        return tokens
    pts=_curves.flatten_spline(g,_curves.svg_spline_tolerance(g))
    return ["M",f"{pts[0][0]!r},{pts[0][1]!r}"]+sum((["L",f"{q[0]!r},{q[1]!r}"] for q in pts[1:]),[])


def _ellipse_transform(cx,cy,rotation) -> str:
    return "rotate(%r %r %r)"%(math.degrees(rotation),cx,cy)


def _presentation(entity, layers, linetypes):
    """SVG presentation attributes from the effective style (layer defaults, then the entity's)."""
    style={}
    layer=layers.get(entity.get("layer","0"),{})
    for key in ("stroke","stroke_width","linetype"):
        if key in layer:
            style[key]=layer[key]
    style.update(entity.get("style") or {})
    out={}
    if "stroke" in style:
        out["stroke"]=style["stroke"]
    if "stroke_width" in style:
        out["stroke-width"]=_num(style["stroke_width"])
    if "fill" in style:
        out["fill"]=style["fill"]
    record=linetypes.get(style.get("linetype"))
    if record is not None:
        out["stroke-dasharray"]=" ".join(_num(v) for v in dash_array(record,scale=style.get("linetype_scale",1.0)))
    return out


def _text_elements(root, g, attrs, paint):
    """One ``<text>`` per text entity, in the drawing's own y-up orientation (see the loss report)."""
    layout=text_layout(g)
    h=g["height"]
    theta=g.get("rotation",0.0)
    c,sn=math.cos(theta),math.sin(theta)
    halign,valign=g.get("halign","left"),g.get("valign","baseline")
    y0={"baseline":-(layout["height"]-h),"bottom":0.0,"middle":-layout["height"]/2,"top":-layout["height"]}[valign]
    first=-(y0+layout["height"]-h)               # local y (downwards) of the first baseline
    pitch=h*g.get("line_spacing",1.0)
    px,py=g["position"]
    node_attrs={**attrs,"data-kind":attrs.get("data-kind","text"),
                "transform":"matrix(%s %s %s %s %s %s)"%tuple(_num(v) for v in (c,sn,sn,-c,px,py)),
                "font-size":_num(h),"text-anchor":{"left":"start","center":"middle","right":"end"}[halign],
                "xml:space":"preserve"}
    if "font" in g:
        node_attrs["font-family"]=g["font"]
    node_attrs.update({k:v for k,v in paint.items() if k=="fill" or (k=="stroke" and "fill" not in paint)})
    if "stroke" in node_attrs:                       # text is painted by the stroke colour
        node_attrs["fill"]=node_attrs.pop("stroke")
    node_attrs={("{http://www.w3.org/XML/1998/namespace}space" if k=="xml:space" else k):v for k,v in node_attrs.items()}
    if len(layout["lines"])==1:
        node=ET.SubElement(root,"text",{**node_attrs,"x":"0","y":_num(first)})
        node.text=layout["lines"][0]
        return
    node=ET.SubElement(root,"text",node_attrs)
    for i,line in enumerate(layout["lines"]):
        tspan=ET.SubElement(node,"tspan",{"x":"0","y":_num(first+i*pitch)})
        tspan.text=line


def _path_data(polygons) -> str:
    return " ".join("M "+" L ".join(f"{x!r},{y!r}" for x,y in poly)+" Z" for poly in polygons)


def _emit(root, entity, ident, layers, linetypes):
    kind,g=entity["kind"],entity["geometry"]
    paint=_presentation(entity,layers,linetypes)
    attrs={**ident,"data-layer":entity.get("layer","0"),"data-kind":kind}
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
        tag,attrs=_arc_attrs(*g["center"],g["radius"],g["start_angle"],g["end_angle"],attrs)
    elif kind=="ellipse":
        cx,cy=g["center"]
        rot=g.get("rotation",0.0)
        tag="ellipse";attrs.update({"cx":str(cx),"cy":str(cy),"rx":str(g["rx"]),"ry":str(g["ry"])})
        if rot:      # the same turn in the file's own axes: a viewer that flips y flips this too, so the ellipse stays consistent with the lines
            attrs.update({"data-rotation":repr(float(rot)),"transform":_ellipse_transform(cx,cy,rot)})
    elif kind=="spline":
        tag="path"
        attrs.update({"d":" ".join(_spline_path_tokens(g)),"data-degree":str(g["degree"]),
                      "data-knots":" ".join(repr(float(k)) for k in g["knots"]),
                      "data-points":" ".join(f"{p[0]!r},{p[1]!r}" for p in g["control_points"])})
        paint={**paint,"fill":"none"}
    elif kind=="text":
        _text_elements(root,g,attrs,paint)
        return
    elif kind=="hatch":
        polygons=[loop_polygon(loop) for loop in g["boundary"]]
        if g.get("pattern","solid")=="solid":
            tag="path";attrs.update({"d":_path_data(polygons),"fill-rule":"evenodd"})
            fill=paint.get("fill",paint.get("stroke"))
            if fill:
                paint={"fill":fill}
            else:
                paint={}
            paint.pop("stroke-width",None)
        else:
            tag="path";attrs["d"]=" ".join(f"M {a[0]!r},{a[1]!r} L {b[0]!r},{b[1]!r}" for a,b in hatch_lines(g)) or "M 0,0"
            paint={k:v for k,v in paint.items() if k!="fill"};paint["fill"]="none"
    elif kind=="dimension":
        marker={"data-dimension":ident.get("data-id",ident.get("data-insert",""))}
        dim_attrs={k:v for k,v in attrs.items() if k not in ("data-id",)}
        dim_attrs.update(marker)
        for prim in dimension_primitives(g):
            if prim["kind"]=="line":
                ET.SubElement(root,"line",{**dim_attrs,"x1":str(prim["start"][0]),"y1":str(prim["start"][1]),
                                           "x2":str(prim["end"][0]),"y2":str(prim["end"][1]),**paint})
            elif prim["kind"]=="arc":
                t,a=_arc_attrs(*prim["center"],prim["radius"],prim["start_angle"],prim["end_angle"],dict(dim_attrs))
                ET.SubElement(root,t,{**a,**paint})
            else:
                _text_elements(root,prim,dict(dim_attrs),paint)
        return
    else:  # insert is expanded by the caller
        raise UnsupportedFeatureError("SVG export cannot draw entity kind: "+kind)
    ET.SubElement(root,tag,{**attrs,**paint})


def export_svg(drawing: Drawing) -> str:
    """Export the drawing as an SVG-subset string.

    Points, lines, polylines, circles, arcs, ellipses (``<ellipse>`` with a rotation) and splines (a cubic-Bézier ``<path>``) round-trip. Text, hatches, dimensions and inserts are drawn
    for viewing only (inserts are expanded, dimensions become lines and text): such files cannot be
    re-imported. Coordinates are written as-is (y-up), so a viewer that flips y shows the drawing upright.
    """
    drawing.validate()
    root=ET.Element("svg",{"xmlns":NS,"version":"1.1","data-unit":drawing.unit})
    blocks={b["name"]:b for b in drawing.blocks}
    layers={l["name"]:l for l in drawing.layers}
    linetypes={r["name"]:r for r in drawing.linetypes}
    for entity in drawing.entities:
        if entity["kind"]=="insert":
            for part in expand_insert(entity,blocks,recursive=True):
                _emit(root,part,{"data-insert":entity["id"]},layers,linetypes)
        else:
            _emit(root,entity,{"data-id":entity["id"]},layers,linetypes)
    return ET.tostring(root,encoding="unicode",short_empty_elements=True)


_ARC_ATTRS=("data-cx","data-cy","data-r","data-start","data-end")

def _arc_from_attributes(attrib,number):
    """Rebuild an SDRW arc from the exact data-* parameters written by export_svg.

    The display ``d`` attribute is validated against those parameters (same
    endpoints and flags) so a file cannot show one arc while claiming another.
    """
    if any(key not in attrib for key in _ARC_ATTRS) or "d" not in attrib:
        raise UnsupportedFeatureError("SVG path arcs require exported data-cx/cy/r/start/end attributes")
    cx,cy,r,start,end=(number(key) for key in _ARC_ATTRS)
    if r<=0 or not 0<end-start<2*math.pi-1e-10:
        raise ValidationError("invalid SVG arc parameters")
    tokens=attrib["d"].replace(","," ").split()
    if len(tokens)!=11 or tokens[0]!="M" or tokens[3]!="A" or tokens[6]!="0" or tokens[8]!="1":
        raise UnsupportedFeatureError("SVG path data is not an exported single arc")
    try:
        ax,ay,rx,ry,large,bx,by=(float(tokens[i]) for i in (1,2,4,5,7,9,10))
    except ValueError as exc:
        raise ValidationError("invalid SVG arc path numbers") from exc
    tolerance=1e-6*max(1.0,abs(cx),abs(cy),r)
    expected=(cx+r*math.cos(start),cy+r*math.sin(start),cx+r*math.cos(end),cy+r*math.sin(end))
    if (abs(rx-r)>tolerance or abs(ry-r)>tolerance or int(large)!=int(end-start>math.pi)
            or any(abs(a-b)>tolerance for a,b in zip((ax,ay,bx,by),expected))):
        raise ValidationError("SVG arc path disagrees with its exact parameters")
    return {"center":[cx,cy],"radius":r,"start_angle":start,"end_angle":end}

def _numbers(text,label):
    try:
        return [finite(float(t),label) for t in text.replace(","," ").split()]
    except (ValueError,TypeError) as exc:
        raise ValidationError("invalid SVG "+label) from exc


def _ellipse_from_attributes(attrib,number):
    """An SDRW ellipse from ``<ellipse>``. A rotation must be declared by ``data-rotation`` *and* shown by exactly the matching ``transform`` (nothing else is accepted)."""
    cx,cy,rx,ry=number("cx"),number("cy"),number("rx"),number("ry")
    if rx<=0 or ry<=0:
        raise ValidationError("invalid SVG ellipse radii")
    rot=0.0
    if "data-rotation" in attrib:
        rot=finite(float(attrib["data-rotation"]),"data-rotation",limit=1e8)
    expected=_ellipse_transform(cx,cy,rot) if rot else None
    shown=attrib.get("transform")
    if (shown is None)!=(expected is None):
        raise UnsupportedFeatureError("SVG ellipse transform is not an exported rotation")
    if shown is not None:
        inner=shown.strip()
        if not (inner.startswith("rotate(") and inner.endswith(")")):
            raise UnsupportedFeatureError("SVG ellipse transform is not an exported rotation")
        got=_numbers(inner[7:-1],"ellipse transform")
        want=[math.degrees(rot),cx,cy]
        if len(got)!=3 or any(abs(a-b)>1e-6*max(1.0,abs(b)) for a,b in zip(got,want)):
            raise ValidationError("SVG ellipse transform disagrees with its exact rotation")
    return {"center":[cx,cy],"rx":rx,"ry":ry,"rotation":rot}


def _spline_from_attributes(attrib):
    """An SDRW spline from the exact ``data-degree/knots/points`` written by export_svg; the display ``d`` must be what those parameters produce."""
    for key in ("data-degree","data-knots","data-points","d"):
        if key not in attrib:
            raise UnsupportedFeatureError("SVG spline paths require exported data-degree/knots/points attributes")
    try:
        degree=int(attrib["data-degree"])
    except ValueError as exc:
        raise ValidationError("invalid SVG spline degree") from exc
    flat=_numbers(attrib["data-points"],"spline points")
    if len(flat)%2:
        raise ValidationError("invalid SVG spline points")
    g={"degree":degree,"control_points":[[flat[i],flat[i+1]] for i in range(0,len(flat),2)],"knots":_numbers(attrib["data-knots"],"spline knots")}
    _curves.validate_spline(g)
    want=_spline_path_tokens(g)
    got=attrib["d"].replace(","," ").split()
    flat_want=[t for tok in want for t in tok.replace(","," ").split()]
    if len(got)!=len(flat_want):
        raise ValidationError("SVG spline path disagrees with its exact parameters")
    scale=max(1.0,*(abs(v) for p in g["control_points"] for v in p))
    for a,b in zip(got,flat_want):
        if a in ("M","C","L") or b in ("M","C","L"):
            if a!=b:
                raise ValidationError("SVG spline path disagrees with its exact parameters")
        elif abs(float(a)-float(b))>1e-6*scale:
            raise ValidationError("SVG spline path disagrees with its exact parameters")
    return g

def import_svg(raw: str|bytes, *, unit=None, preserve_ids: bool=False) -> Drawing:
    data=raw.encode("utf-8") if isinstance(raw,str) else raw
    if not isinstance(data,bytes) or len(data)>MAX_SVG_BYTES:
        raise ValidationError("SVG input exceeds byte limit")
    try:
        text=data.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValidationError("SVG must be UTF-8") from exc
    if "\x00" in text:
        raise ValidationError("SVG must be UTF-8 without NUL bytes")
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise UnsupportedFeatureError("SVG document declarations are prohibited")
    try:
        root=ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValidationError("invalid SVG XML") from exc
    if root.tag not in (f"{{{NS}}}svg","svg"):
        raise ValidationError("SVG root element required")
    if set(root.attrib)-{"version","data-unit"}:
        raise UnsupportedFeatureError("SVG root attributes require explicit viewport/style conversion")
    if root.text and root.text.strip():
        raise UnsupportedFeatureError("SVG text is unsupported")
    drawing=Drawing(unit=unit if unit is not None else root.attrib.get("data-unit","mm"))
    tag_names={"line","circle","polyline","polygon","path","ellipse"}
    used_ids=set()
    def add(kind,geometry,layer,child):
        """Add one entity; with ``preserve_ids`` keep a valid, unique ``data-id`` written by export_svg."""
        ident=child.attrib.get("data-id") if preserve_ids else None
        if not isinstance(ident,str) or not is_valid_id(ident) or ident in used_ids:
            ident=new_id()
        used_ids.add(ident)
        drawing.add(kind,geometry,layer=layer,create_layer=True,entity_id=ident)
    for child in root:
        tag=child.tag.rsplit("}",1)[-1]
        if child.tag not in (tag,f"{{{NS}}}{tag}") or len(child) or (child.text and child.text.strip()) or (child.tail and child.tail.strip()):
            raise UnsupportedFeatureError("SVG nested content or foreign namespaces are unsupported")
        if tag not in tag_names:
            raise UnsupportedFeatureError("unsupported SVG element: "+tag)
        if any(key.lower().startswith("on") or key in {"href","style"} or (key=="transform" and tag!="ellipse") or key.endswith("}href") for key in child.attrib):
            raise UnsupportedFeatureError("SVG event handlers, references, styles and transforms are not allowed")
        layer=child.attrib.get("data-layer","0")
        def n(key):
            try:
                return finite(float(child.attrib[key]),key)
            except (KeyError,ValueError,TypeError) as exc:
                raise ValidationError("invalid SVG numeric attribute: "+key) from exc
        if tag=="path":
            kind=child.attrib.get("data-kind")
            if kind=="spline":
                add("spline",_spline_from_attributes(child.attrib),layer,child)
            elif kind!="arc":
                raise UnsupportedFeatureError("general SVG path data is unsupported; only exported arcs and splines are read")
            else:
                add("arc",_arc_from_attributes(child.attrib,n),layer,child)
        elif tag=="ellipse":
            add("ellipse",_ellipse_from_attributes(child.attrib,n),layer,child)
        elif tag=="circle" and child.attrib.get("data-kind")=="point":
            add("point",{"position":[n("cx"),n("cy")]},layer,child)
        elif tag=="line":
            add("line",{"start":[n("x1"),n("y1")],"end":[n("x2"),n("y2")]},layer,child)
        elif tag=="circle":
            add("circle",{"center":[n("cx"),n("cy")],"radius":n("r")},layer,child)
        else:
            points=[]
            source=child.attrib.get("points","").replace(","," ").split()
            if len(source)%2 or len(source)>200000:
                raise ValidationError("invalid SVG point coordinates")
            for i in range(0,len(source),2):
                try: points.append([finite(float(source[i])),finite(float(source[i+1]))])
                except ValueError as exc: raise ValidationError("invalid SVG points") from exc
            add("polyline",{"points":points,"closed":tag=="polygon"},layer,child)
    return drawing
