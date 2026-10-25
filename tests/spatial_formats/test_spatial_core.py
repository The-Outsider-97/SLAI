"""CPU-only security and geometry regression tests for SGEO/SDRW internals."""
import json
import math
import struct
from pathlib import Path
import pytest

from src.utils.spatial_formats import Drawing, Scene, load_package, write_package
from src.utils.spatial_formats.common import ValidationError, ConflictError
from src.utils.spatial_formats.adapters.svg import export_svg, import_svg
from src.utils.spatial_formats.adapters.stl import export_binary_stl, import_binary_stl
from src.utils.spatial_formats.sgeo.geometry import validate_geometry
from src.utils.spatial_formats.sgeo.families import validate_family, generate_geometry
from src.utils.spatial_formats.common.ids import new_id
from src.utils.spatial_formats.common.transforms import IDENTITY, compose, invert_transform, transform_point

def test_drawing_archive_roundtrip(tmp_path):
    drawing=Drawing(name="Plan",unit="mm")
    drawing.add("line",{"start":[0,0],"end":[100,0]})
    path=tmp_path/"plan.sdrw"
    hexdigest=write_package(path,drawing.manifest(),drawing.as_dict())
    assert len(hexdigest)==64
    copy=Drawing.from_package(load_package(path))
    assert copy.entities==drawing.entities
    with pytest.raises(ConflictError):
        write_package(path,drawing.manifest(),drawing.as_dict())
    assert write_package(path,drawing.manifest(),drawing.as_dict(),overwrite=True)==hexdigest

def test_scene_archive_roundtrip(tmp_path):
    scene=Scene(name="test")
    oid=scene.add_object(geometry={"kind":"box","center":[0,0,0],"size":[1,2,3]})
    assert scene.world_bounds(oid)==((-0.5,-1.0,-1.5),(0.5,1.0,1.5))
    path=tmp_path/"sample.sgeo"
    write_package(path,scene.manifest(),scene.as_dict())
    assert Scene.from_package(load_package(path)).objects==scene.objects

def test_transform_inverse_and_composition():
    affine=(2,0,0,4, 0,3,0,5, 0,0,4,6, 0,0,0,1)
    assert transform_point(invert_transform(affine),transform_point(affine,(1,2,3)))==pytest.approx((1,2,3))
    assert compose(affine,invert_transform(affine))==pytest.approx(IDENTITY)

def test_cycle_rejected():
    a,b=new_id(),new_id()
    objects=[{"id":a,"parent":b},{"id":b,"parent":a}]
    with pytest.raises(ValidationError):
        Scene(objects=objects)

def test_orphan_rejected():
    with pytest.raises(ValidationError):
        Scene(objects=[{"id":new_id(),"parent":new_id()}])

def test_svg_line_roundtrip():
    drawing=Drawing()
    drawing.add("line",{"start":[0,1],"end":[4,5]})
    loaded=import_svg(export_svg(drawing))
    assert loaded.entities[0]["geometry"]==drawing.entities[0]["geometry"]

def test_svg_point_roundtrip():
    drawing=Drawing()
    drawing.add("point",{"position":[2,3]})
    loaded=import_svg(export_svg(drawing))
    assert loaded.entities[0]["kind"]=="point"

def test_mesh_stl_roundtrip():
    geo={"kind":"mesh","vertices":[[0,0,0],[1,0,0],[0,1,0]],"faces":[[0,1,2]]}
    result=import_binary_stl(export_binary_stl(geo))
    assert result["faces"]==[[0,1,2]]

def test_mesh_rejects_nan_and_degenerate_triangles():
    with pytest.raises(ValidationError):
        validate_geometry({"kind":"mesh","vertices":[[0,0,0],[float("nan"),0,0],[0,1,0]],"faces":[[0,1,2]]})
    with pytest.raises(ValidationError):
        validate_geometry({"kind":"mesh","vertices":[[0,0,0],[1,0,0],[2,0,0]],"faces":[[0,1,2]]})

def test_family_generator():
    fam={"id":new_id(),"name":"Block",
         "parameters":{k:{"type":"length","default":1.0,"min":0.1,"max":5.0} for k in ("width","depth","height")},
         "generator":{"kind":"builtin.box_v1"}}
    validate_family(fam)
    assert generate_geometry(fam,{"width":2.0})["size"]==[2.0,1.0,1.0]
    with pytest.raises(ValidationError):
        generate_geometry(fam,{"width":-3})

def test_corrupt_archive_rejected(tmp_path):
    scene=Scene()
    target=tmp_path/"valid.sgeo"
    write_package(target,scene.manifest(),scene.as_dict())
    damaged=tmp_path/"bad.sgeo"
    data=bytearray(target.read_bytes());data[len(data)//2]^=0xFF
    damaged.write_bytes(data)
    with pytest.raises(Exception):
        load_package(damaged)

def test_metadata_namespace_required():
    with pytest.raises(ValidationError):
        Scene(metadata={"unqualified":1})
