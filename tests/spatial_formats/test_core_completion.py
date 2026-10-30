"""Regression coverage for atomic edits, family lifecycle, and package boundaries."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
from zipfile import ZipFile, ZIP_STORED

import pytest

from src.utils.spatial_formats import Drawing, Scene, load_package, write_document
from src.utils.spatial_formats.common import ConflictError, ValidationError, ResourceLimitError, new_id
from src.utils.spatial_formats.common.helpers import finite, parse_json
from src.utils.spatial_formats.common.transforms import IDENTITY, compose, invert_transform
from src.utils.spatial_formats.sdrw import apply_drawing_operations
from src.utils.spatial_formats.sgeo import apply_scene_operations, objects_within_bounds
from src.utils.spatial_formats.sgeo.parameters import evaluate_parameters, evaluate_expression
from src.utils.spatial_formats.verification import validate_document


def family():
    return {"id":new_id(), "name":"Block", "generator":{"kind":"builtin.box_v1"},
            "parameters": {"width":{"type":"length","default":2},
                           "depth":{"type":"length","default":3},
                           "height":{"type":"length","expression":"width * 2"}}}


def test_drawing_batch_rollback_identity_and_layers():
    drawing=Drawing()
    geometry={"start":[0,0],"end":[2,0]}
    oid=drawing.add("line",geometry)
    geometry["end"][0]=100
    assert drawing.entities[0]["geometry"]["end"]==[2,0]
    before=drawing.as_dict()
    with pytest.raises(ValidationError):
        apply_drawing_operations(drawing,[{"op":"update","id":oid,"value":{"name":"changed"}},
                                          {"op":"remove_layer","name":"0"}])
    assert drawing.as_dict()==before
    apply_drawing_operations(drawing,[{"op":"rename_layer","name":"0","value":"Walls"},
                                     {"op":"transform","id":oid,"value":[2,0,1,0,2,3,0,0,1]}],
                             expected_revision=before["revision"])
    assert drawing.revision==before["revision"]+1
    assert drawing.entities[0]["id"]==oid
    assert drawing.entities[0]["layer"]=="Walls"
    assert drawing.entities[0]["geometry"]=={"start":[1,3],"end":[5,3]}
    with pytest.raises(ConflictError):
        apply_drawing_operations(drawing,[{"op":"remove","id":oid}],expected_revision=before["revision"])


def test_drawing_competing_optimistic_edits():
    drawing=Drawing()
    revision=drawing.revision
    def attempt(index):
        try:
            apply_drawing_operations(drawing,[{"op":"add_layer","name":str(index)}],expected_revision=revision)
            return True
        except ConflictError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(attempt,[1,2]))==[False,True]
    assert drawing.revision==revision+1


def test_family_lifecycle_regenerates_atomically(tmp_path):
    scene=Scene()
    definition=family()
    fid=definition["id"]
    result=apply_scene_operations(scene,[{"op":"add_family","value":definition},
                                        {"op":"instantiate","family_id":fid}])
    oid=result[1]
    assert scene.revision==2
    assert scene.objects[0]["geometry"]["size"]==[2,3,4]
    definition["parameters"]["width"]["default"]=5
    apply_scene_operations(scene,[{"op":"update_family","id":fid,"value":definition}])
    assert scene.objects[0]["geometry"]["size"]==[5,3,10]
    before=scene.as_dict()
    bad=deepcopy(definition)
    bad["parameters"]["width"]["default"]=-1
    with pytest.raises(ValidationError):
        apply_scene_operations(scene,[{"op":"update_family","id":fid,"value":bad}])
    with pytest.raises(ValidationError):
        apply_scene_operations(scene,[{"op":"remove_family","id":fid}])
    assert scene.as_dict()==before
    path=tmp_path/"family.sgeo"
    write_document(path,scene)
    assert Scene.from_package(load_package(path)).as_dict()==scene.as_dict()
    apply_scene_operations(scene,[{"op":"remove","id":oid},{"op":"remove_family","id":fid}])
    assert scene.objects==scene.families==[]


def test_family_geometry_cannot_become_stale():
    scene=Scene(families=[family()])
    oid=apply_scene_operations(scene,[{"op":"instantiate","family_id":scene.families[0]["id"]}])[0]
    before=scene.as_dict()
    with pytest.raises(ValidationError,match="does not match"):
        apply_scene_operations(scene,[{"op":"geometry","id":oid,"value":{"kind":"sphere","center":[0,0,0],"radius":3}}])
    assert scene.as_dict()==before


@pytest.mark.parametrize("spec",[
    {"type":[]}, {"type":"number","default":1,"min":"0"},
    {"type":"number","default":1,"min":2,"max":0},
    {"type":"number","default":1,"read_only":"false"},
    {"type":"boolean","default":True,"max":1},
    {"type":"length","expression":"length(1)","min":-1},
    {"type":"number","expression":"1","default":1},
    {"type":"number","default":1,"unknown":True},
    {"type":"area","default":-1}, {"type":"volume","default":-1},
])
def test_invalid_parameter_declarations(spec):
    with pytest.raises(ValidationError):
        evaluate_parameters({"value":spec})


def test_function_named_parameter_dependency_and_clamp():
    assert evaluate_parameters({"result":{"type":"number","expression":"max(min, 3)"},
                                "min":{"type":"number","default":4}})["result"]==4
    with pytest.raises(ValidationError,match="minimum"):
        evaluate_expression("clamp(1, 4, 2)",{})


def test_small_uniform_scale_and_bounded_composition():
    matrix=list(IDENTITY)
    for index in (0,5,10): matrix[index]=1e-6
    assert compose(matrix,invert_transform(matrix))==pytest.approx(IDENTITY)
    matrix[0]=matrix[5]=matrix[10]=1e8
    with pytest.raises(ValidationError): compose(matrix,matrix)


def test_hierarchy_batch_bounds():
    scene=Scene()
    matrix=list(IDENTITY);matrix[3]=10
    root=scene.add_object(transform=matrix)
    child=scene.add_object(parent=root,geometry={"kind":"box","center":[0,0,0],"size":[2,2,2]})
    assert scene.world_transform(child)[3]==10
    assert scene.bounds_by_object()=={child:((9,-1,-1),(11,1,1))}
    assert objects_within_bounds(scene,[8,-2,-2],[12,2,2])==[child]
    assert objects_within_bounds(scene,[-1,-1,-1],[1,1,1])==[]


@pytest.mark.parametrize("model",[Drawing,Scene])
def test_schema_required_and_unknown_fields(model):
    document=model()
    manifest={**document.manifest(),"parts":[{"path":"scene.json" if model is Scene else "drawing.json","size":0,"sha256":"0"*64}]}
    missing=document.as_dict();del missing["name"]
    extra={**document.as_dict(),"surprise":1}
    for payload in (missing,extra):
        with pytest.raises(ValidationError):validate_document(manifest,payload)


def test_bounded_json_and_large_integer_errors():
    with pytest.raises(ValidationError):finite(10**10000)
    with pytest.raises(ValidationError):parse_json(b'1'+b'0'*5000)
    with pytest.raises(ResourceLimitError):parse_json(b'['*2000+b'0'+b']'*2000)


def test_compressible_metadata_roundtrip_and_digest(tmp_path):
    drawing=Drawing(metadata={"test:repetition":"a"*100_000})
    path=tmp_path/"drawing.sdrw"
    digest=write_document(path,drawing)
    assert digest==sha256(path.read_bytes()).hexdigest()
    assert load_package(path).document==drawing.as_dict()
    from src.utils.spatial_formats.container.reader import MAX_RATIO
    with ZipFile(path) as archive:
        for info in archive.infolist():
            assert info.file_size <= MAX_RATIO * max(1, info.compress_size)


def test_writer_stores_content_to_respect_ratio_limit(tmp_path, monkeypatch):
    from src.utils.spatial_formats.container import reader, writer
    # Exercise fallback independently of the production ratio policy and zlib
    # version. A ratio of one requires repetitive content to be stored.
    monkeypatch.setattr(reader, "MAX_RATIO", 1)
    monkeypatch.setattr(writer, "MAX_RATIO", 1)
    drawing = Drawing(metadata={"test:repetition": "a" * 100_000})
    path = tmp_path / "stored.sdrw"
    digest = write_document(path, drawing)
    assert digest == sha256(path.read_bytes()).hexdigest()
    assert load_package(path).document == drawing.as_dict()
    with ZipFile(path) as archive:
        assert archive.getinfo("drawing.json").compress_type == ZIP_STORED


@pytest.mark.parametrize("raw, context", [
    (b'{"key": 1, "key": 2}', {"key": "key"}),
    (b'{"value": NaN}', {"token": "NaN"}),
])
def test_json_hook_errors_preserve_context(raw, context):
    with pytest.raises(ValidationError) as caught:
        parse_json(raw)
    assert caught.value.context == context


def test_no_partial_output_when_atomic_publication_unavailable(tmp_path, monkeypatch):
    from src.utils.spatial_formats.container import writer

    def unavailable(*args, **kwargs):
        raise OSError("atomic publication unavailable")

    # Windows legitimately falls back to a no-clobber rename when hard links
    # are unsupported. Disable *both* publication mechanisms to exercise the
    # intended failure/cleanup invariant on every supported operating system.
    monkeypatch.setattr(writer.os, "link", unavailable)
    monkeypatch.setattr(writer.os, "rename", unavailable)
    path = tmp_path / "new.sdrw"
    with pytest.raises(ConflictError, match="atomic"):
        write_document(path, Drawing())
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_repeated_binary_references_fail_before_mesh_allocation(monkeypatch):
    from src.utils.spatial_formats.container import scene_buffers
    from src.utils.spatial_formats.container.buffers import encode_mesh_buffers
    mesh={"kind":"mesh","vertices":[[0,0,0],[1,0,0],[0,1,0]],"faces":[[0,1,2]]}
    descriptors,assets=encode_mesh_buffers(mesh)
    # Large descriptors need no payload allocation to demonstrate preflight.
    descriptors["vertices"]["shape"]=[500000,3]
    document={"objects":[{"geometry":{"kind":"mesh_buffer","buffers":descriptors}}]}
    def unexpected(*args):pytest.fail("mesh decoder ran before allocation preflight")
    monkeypatch.setattr(scene_buffers,"decode_mesh_buffers",unexpected)
    with pytest.raises(ResourceLimitError):scene_buffers.hydrate_scene(document,assets)
