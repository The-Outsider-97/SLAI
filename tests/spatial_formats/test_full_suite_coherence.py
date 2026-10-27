"""Repository coherence and persistence regressions for all historical failures.

These tests deliberately assert the public behavior rather than merely checking
that an implementation file contains a particular symbol.
"""
from __future__ import annotations
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile

import pytest

from src.utils.spatial_formats import Scene, load_package, write_package
from src.utils.spatial_formats.common import ConflictError, IntegrityError, ValidationError
from src.utils.spatial_formats.container import reader
from src.utils.spatial_formats.sgeo.transactions import apply_scene_operations

MESH={"kind":"mesh","vertices":[[0,0,0],[1,0,0],[0,1,0]],"faces":[[0,1,2]]}

def test_active_runtime_has_coherent_scene_and_binary_reader():
    """Catch mixed checkouts, source shadowing and outdated source behavior."""
    assert hasattr(Scene(),"_mutation_lock")
    assert "hydrate_scene" in inspect.getsource(reader._load_stream)
    assert "src.utils.spatial_formats" in Scene.__module__

def test_binary_mesh_archive_reads_as_native_geometry(tmp_path):
    scene=Scene()
    scene.add_object(geometry=MESH)
    scene.add_object(geometry=MESH)
    target=tmp_path/"indexed.sgeo"
    write_package(target,scene.manifest(),scene.as_dict())
    with ZipFile(target) as archive:
        stored=json.loads(archive.read("scene.json"))
        names={name for name in archive.namelist() if name.startswith("geometry/")}
        assert len(names)==2
        assert all(o["geometry"]["kind"]=="mesh_buffer" for o in stored["objects"])
        assert stored["objects"][0]["geometry"]["buffers"]==stored["objects"][1]["geometry"]["buffers"]
    package=load_package(target)
    assert all(o["geometry"]["kind"]=="mesh" for o in package.document["objects"])
    assert Scene.from_package(package).objects==scene.objects

def test_transaction_revision_conflict_and_rollback_are_atomic():
    scene=Scene()
    first=scene.add_object(geometry=MESH)
    child=scene.add_object(parent=first,geometry=MESH)
    original=scene.as_dict()
    with pytest.raises(ValidationError):
        apply_scene_operations(scene,[
            {"op":"rename","id":first,"value":"changed"},
            {"op":"reparent","id":first,"value":child},
        ])
    assert scene.as_dict()==original
    with pytest.raises(ConflictError):
        apply_scene_operations(scene,[{"op":"rename","id":first,"value":"wrong"}],
                               expected_revision=scene.revision-1)
    assert scene.as_dict()==original

def test_no_untrusted_scene_assets_even_empty(tmp_path):
    scene=Scene()
    target=tmp_path/"malicious.sgeo"
    for assets in ({}, {"geometry/fake.bin":b"bad"}):
        with pytest.raises(ValidationError):
            write_package(target,scene.manifest(),scene.as_dict(),assets=assets)
        assert not target.exists()

def test_parallel_scene_additions_maintain_unique_ids_and_revisions():
    scene=Scene()
    count=24
    def add(_):
        return scene.add_object(geometry=MESH)
    with ThreadPoolExecutor(max_workers=6) as executor:
        results=list(executor.map(add,range(count)))
    assert len(set(results))==count
    assert len(scene.objects)==count
    assert scene.revision==count+1
    scene.validate()
