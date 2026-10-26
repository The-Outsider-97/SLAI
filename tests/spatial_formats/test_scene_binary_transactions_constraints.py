"""Integration tests for transactional scenes, binary archives and constraints."""
import pytest
from zipfile import ZipFile
from src.utils.spatial_formats import Scene, Drawing, load_package, write_package
from src.utils.spatial_formats.common import ConflictError, ValidationError, IntegrityError
from src.utils.spatial_formats.common.ids import new_id
from src.utils.spatial_formats.sgeo.transactions import apply_scene_operations, objects_within_bounds
from src.utils.spatial_formats.sdrw.constraints import check_constraints
from src.utils.spatial_formats.sdrw.entities import make_entity
from src.utils.spatial_formats.container.scene_buffers import externalize_scene,hydrate_scene

MESH={"kind":"mesh","vertices":[[0,0,0],[1,0,0],[0,1,0]],"faces":[[0,1,2]]}

def test_mesh_asset_serialization_and_hydration(tmp_path):
    scene=Scene()
    a=scene.add_object(name="A",geometry=MESH)
    b=scene.add_object(name="B",geometry=MESH)
    path=tmp_path/"indexed.sgeo"
    write_package(path,scene.manifest(),scene.as_dict())
    with ZipFile(path) as archive:
        paths=archive.namelist()
        assert len([name for name in paths if name.startswith("geometry/")])==2
        assert b"mesh_buffer" in archive.read("scene.json")
        assert len([name for name in paths if name=="scene.json"])==1
    loaded=Scene.from_package(load_package(path))
    assert loaded.objects==scene.objects
    assert loaded.objects[0]["geometry"]==loaded.objects[1]["geometry"]

def test_missing_or_tampered_binary_resource_refused():
    scene=Scene()
    scene.add_object(geometry=MESH)
    serialized,assets=externalize_scene(scene.as_dict())
    with pytest.raises(IntegrityError):
        hydrate_scene(serialized,{})
    for key in assets:
        tampered=dict(assets)
        tampered[key]=b"bad"
        with pytest.raises(IntegrityError):
            hydrate_scene(serialized,tampered)

def test_scene_batch_commit_and_conflict():
    scene=Scene()
    parent=scene.add_object(name="start",geometry={"kind":"box","center":[0,0,0],"size":[1,1,1]})
    old=scene.revision
    ids=apply_scene_operations(scene,[{"op":"rename","id":parent,"value":"changed"},
                                      {"op":"add","name":"other","parent":parent,"geometry":MESH}],
                               expected_revision=old)
    assert scene.revision==old+1
    assert scene.objects[0]["name"]=="changed"
    assert len(ids)==2
    with pytest.raises(ConflictError):
        apply_scene_operations(scene,[{"op":"rename","id":parent,"value":"oops"}],
                               expected_revision=old)

def test_scene_batch_rollback_on_invalid_reparent():
    scene=Scene()
    a=scene.add_object(name="A",geometry=MESH)
    b=scene.add_object(name="B",parent=a,geometry=MESH)
    original=scene.as_dict()
    with pytest.raises(ValidationError):
        apply_scene_operations(scene,[{"op":"rename","id":a,"value":"renamed"},
                                      {"op":"reparent","id":a,"value":b}])
    assert scene.as_dict()==original

def test_query_broad_phase_only():
    scene=Scene()
    oid=scene.add_object(geometry={"kind":"sphere","center":[0,0,0],"radius":1})
    assert objects_within_bounds(scene,(-0.5,-0.5,-0.5),(0.5,0.5,0.5))==[oid]
    assert objects_within_bounds(scene,(20,20,20),(21,21,21))==[]

def test_geometric_constraints_residuals():
    a=make_entity("line",{"start":[0,0],"end":[3,0]})
    b=make_entity("line",{"start":[0,0],"end":[0,2]})
    p=make_entity("point",{"position":[0,0]})
    q=make_entity("point",{"position":[3,4]})
    result=check_constraints([a,b,p,q],[
        {"kind":"perpendicular","entities":[a["id"],b["id"]]},
        {"kind":"horizontal","entities":[a["id"]]},
        {"kind":"distance","entities":[p["id"],q["id"]],"value":5},
        {"kind":"equal_length","entities":[a["id"],b["id"]]},
    ])
    assert [x.satisfied for x in result]==[True,True,True,False]
    assert result[3].residual==pytest.approx(1)

def test_geometric_constraints_reject_broken_references():
    a=make_entity("line",{"start":[0,0],"end":[1,0]})
    with pytest.raises(ValidationError):
        check_constraints([a],[{"kind":"parallel","entities":[a["id"],new_id()]}])
