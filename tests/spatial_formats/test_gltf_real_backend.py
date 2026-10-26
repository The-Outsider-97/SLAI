"""Real GLB backend regressions: no mocks, no fabricated success."""
from __future__ import annotations
import json
import pytest
from src.utils.spatial_formats.adapters import gltf
from src.utils.spatial_formats.common.errors import UnsupportedFeatureError

pytestmark=pytest.mark.skipif(not gltf.capabilities().available, reason="real trimesh backend is required")

def test_glb_mesh_coordinates_roundtrip_z_up():
    geometry={"kind":"mesh","vertices":[[1,2,3],[4,2,3],[1,6,3]],"faces":[[0,1,2]]}
    result=gltf.import_glb_scene(gltf.export_glb(geometry))
    meshes=[obj["geometry"] for obj in result.objects if obj.get("geometry") is not None]
    assert len(meshes)==1
    actual=meshes[0]["vertices"]
    assert sorted(tuple(round(float(x),5) for x in point) for point in actual)==sorted(tuple(point) for point in geometry["vertices"])

def test_glb_external_buffer_is_rejected():
    import trimesh
    model=trimesh.Scene(trimesh.creation.box())
    data=bytearray(model.export(file_type="glb"))
    start=20
    length=int.from_bytes(data[12:16],"little")
    payload=json.loads(data[start:start+length].decode("utf-8"))
    payload["buffers"][0]["uri"]="https://example.invalid/external.bin"
    raw=json.dumps(payload,separators=(",",":")).encode("utf-8")
    raw+=b" " *((4-len(raw)%4)%4)
    original_end=start+length
    binary=data[original_end:]
    modified=bytearray(b"glTF"+(2).to_bytes(4,"little")+
        (12+8+len(raw)+len(binary)).to_bytes(4,"little")+
        len(raw).to_bytes(4,"little")+b"JSON"+raw+binary)
    with pytest.raises(UnsupportedFeatureError):
        gltf.import_glb_scene(bytes(modified))

def test_glb_y_up_to_z_up_basis_is_proper():
    from src.utils.spatial_formats.common.transforms import det3
    assert det3(gltf.Y_UP_TO_Z)==pytest.approx(1)
    assert gltf._map_vertices([[2,3,4]],to_z=True).tolist()==[[2,-4,3]]
    assert gltf._map_vertices([[2,-4,3]],to_z=False).tolist()==[[2,3,4]]
