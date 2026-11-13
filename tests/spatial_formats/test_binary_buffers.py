"""Content-addressed, length-checked numeric geometry buffer regressions."""
import pytest
from src.utils.spatial_formats.container.buffers import (encode_buffer,decode_buffer,encode_mesh_buffers,decode_mesh_buffers)
from src.utils.spatial_formats.common.errors import ValidationError,IntegrityError,ResourceLimitError

def test_finite_f64_roundtrip():
    info,raw=encode_buffer([1.25,2.5,3.75],dtype="f64",shape=(1,3))
    assert decode_buffer(info.to_dict(),raw)==[1.25,2.5,3.75]
    assert info.path.endswith(info.sha256+".bin")

def test_content_address_is_deterministic():
    a,x=encode_buffer([2,4,6],dtype="u32",shape=(3,))
    b,y=encode_buffer([2,4,6],dtype="u32",shape=(3,))
    assert a==b and x==y

def test_binary_mesh_roundtrip():
    mesh={"kind":"mesh","vertices":[[0,0,0],[2,0,0],[0,3,0]],"faces":[[0,1,2]]}
    descriptors,assets=encode_mesh_buffers(mesh)
    assert decode_mesh_buffers(descriptors,assets)==mesh

def test_digest_corruption_rejected():
    desc,raw=encode_buffer([1,2,3],dtype="u32",shape=(3,))
    with pytest.raises(IntegrityError):
        decode_buffer(desc.to_dict(),raw[:-1]+bytes([raw[-1]^1]))

def test_nonfinite_binary_rejected():
    with pytest.raises(ValidationError):
        encode_buffer([float("nan")],dtype="f64",shape=(1,))
    with pytest.raises(ResourceLimitError):
        encode_buffer([],dtype="f64",shape=(10000000,))

def test_shape_does_not_silently_change():
    desc,raw=encode_buffer([1,2,3],dtype="u32",shape=(3,))
    descriptor=desc.to_dict()
    descriptor["shape"]=[4]
    with pytest.raises(IntegrityError):
        decode_buffer(descriptor,raw)
