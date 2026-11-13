"""Real-backend unit fidelity plus adversarial SVG/GLB preflight checks."""
import json
import pytest
from src.utils.spatial_formats import Drawing, Scene
from src.utils.spatial_formats.adapters import dxf, gltf, svg
from src.utils.spatial_formats.common import ValidationError, UnsupportedFeatureError, ResourceLimitError
from src.utils.spatial_formats.common.transforms import IDENTITY


def test_dxf_units_and_hidden_empty_layers(tmp_path):
    pytest.importorskip("ezdxf")
    drawing=Drawing(unit="ft")
    drawing.add_layer("Hidden",visible=False)
    drawing.add("line",{"start":[0,0],"end":[3,0]})
    path=tmp_path/"feet.dxf"
    dxf.export_dxf(drawing,path)
    result=dxf.import_dxf(path)
    assert result.unit=="ft"
    assert {l["name"]:l.get("visible",True) for l in result.layers}["Hidden"] is False
    assert result.entities[0]["geometry"]["end"]==[3,0]


def test_glb_millimetres_scale_parent_translation_and_vertices():
    pytest.importorskip("trimesh")
    scene=Scene(unit="mm")
    parent_matrix=list(IDENTITY);parent_matrix[3]=2000
    parent=scene.add_object(transform=parent_matrix)
    local=list(IDENTITY);local[7]=3000
    scene.add_object(parent=parent,transform=local,geometry={"kind":"mesh",
        "vertices":[[0,0,0],[1000,0,0],[0,1000,0]],"faces":[[0,1,2]]})
    result=gltf.import_glb_scene(gltf.export_glb_scene(scene))
    assert result.unit=="m"
    bounds=list(result.bounds_by_object().values())
    assert len(bounds)==1
    assert bounds[0][0]==pytest.approx((2,3,0))
    assert bounds[0][1]==pytest.approx((3,4,0))


def test_svg_native_unit_roundtrip():
    drawing=Drawing(unit="m")
    drawing.add("circle",{"center":[2,3],"radius":1})
    restored=svg.import_svg(svg.export_svg(drawing))
    assert restored.unit=="m"
    assert restored.entities[0]["geometry"]==drawing.entities[0]["geometry"]


@pytest.mark.parametrize("raw",[
    '<svg transform="scale(10)"><circle cx="0" cy="0" r="1"/></svg>',
    '<svg viewBox="0 0 10 10"/>',
    '<svg><circle cx="0" cy="0" r="1"><script>bad()</script></circle></svg>',
    '<svg xmlns:x="urn:foreign"><x:circle cx="0" cy="0" r="1"/></svg>',
    '<!DOCTYPE svg [<!ENTITY x "unsafe">]><svg/>',
    '<!DOCTYPE svg [<!ENTITY x "unsafe">]><svg/>'.encode('utf-16'),
])
def test_svg_rejects_unsupported_root_and_nested_semantics(raw):
    with pytest.raises((ValidationError,UnsupportedFeatureError)):
        svg.import_svg(raw)


def glb(model,binary=b'\0'*36):
    payload=json.dumps(model).encode();payload+=b' '*((-len(payload))%4)
    chunks=len(payload).to_bytes(4,'little')+b'JSON'+payload
    if binary:chunks+=len(binary).to_bytes(4,'little')+b'BIN\0'+binary
    return b'glTF'+(2).to_bytes(4,'little')+(12+len(chunks)).to_bytes(4,'little')+chunks


def minimal():
    return {"asset":{"version":"2.0"},"buffers":[{"byteLength":36}],
        "bufferViews":[{"buffer":0,"byteLength":36}],
        "accessors":[{"bufferView":0,"componentType":5126,"type":"VEC3","count":3}],
        "meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],
        "nodes":[{"mesh":0}],"scenes":[{"nodes":[0]}],"scene":0}


@pytest.mark.parametrize("case",["count","offset","reference","cycle","parent","stride","table","sparse"])
def test_glb_preflight_rejects_before_backend(case,monkeypatch):
    model=minimal()
    if case=="count":model['accessors'][0]['count']=1_000_000_000
    if case=="offset":model['accessors'][0]['byteOffset']=100
    if case=="reference":model['meshes'][0]['primitives'][0]['attributes']['POSITION']=2
    if case=="cycle":model['nodes'][0]['children']=[0]
    if case=="parent":model['nodes'] += [{'children':[0]},{'children':[0]}]
    if case=="stride":model['bufferViews'][0]['byteStride']=8
    if case=="table":model['accessors']={}
    if case=="sparse":model['accessors'][0]['sparse']={}
    def unexpected():pytest.fail("backend loaded before preflight rejection")
    monkeypatch.setattr(gltf,'_trimesh',unexpected)
    with pytest.raises((ValidationError,UnsupportedFeatureError,ResourceLimitError)):
        gltf.import_glb_scene(glb(model))


def test_glb_rejects_float32_triangle_collapse():
    pytest.importorskip("trimesh")
    mesh={"kind":"mesh","vertices":[[1e9,0,0],[1e9+1,0,0],[1e9,1,0]],"faces":[[0,1,2]]}
    with pytest.raises(ValidationError,match="degenerate"):
        gltf.export_glb(mesh)
