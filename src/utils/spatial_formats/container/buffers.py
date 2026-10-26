"""Versioned, content-addressed little-endian numeric geometry buffers.

Buffers are inert bytes; no pickles, memory-mapped arbitrary files or object
deserialization. Descriptors are explicit and validated before allocation.
"""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from struct import Struct
from ..common.errors import ValidationError, ResourceLimitError, IntegrityError

_DTYPES={"f32":("f",4),"f64":("d",8),"u32":("I",4)}
_MAX_ELEMENTS=5_000_000
_MAX_BYTES=64*1024*1024

@dataclass(frozen=True)
class BufferDescriptor:
    path: str
    sha256: str
    dtype: str
    shape: tuple[int,...]
    byte_order: str = "little"

    def to_dict(self):
        return {"path":self.path,"sha256":self.sha256,"dtype":self.dtype,
                "shape":list(self.shape),"byte_order":self.byte_order}

def validate_descriptor(value)->BufferDescriptor:
    if not isinstance(value,dict):
        raise ValidationError("buffer descriptor must be an object")
    if set(value)!={"path","sha256","dtype","shape","byte_order"}:
        raise ValidationError("invalid buffer descriptor fields")
    shape=value["shape"]
    if not isinstance(shape,list) or not 1<=len(shape)<=4 or any(type(n) is not int or n<1 for n in shape):
        raise ValidationError("invalid array shape")
    if prod(shape)>_MAX_ELEMENTS:
        raise ResourceLimitError("numeric array element budget exceeded")
    dtype=value["dtype"]
    if not isinstance(dtype,str) or dtype not in _DTYPES or value["byte_order"]!="little":
        raise ValidationError("unsupported binary element format")
    path=value["path"]
    digest=value["sha256"]
    if not isinstance(digest,str) or len(digest)!=64 or any(x not in "0123456789abcdef" for x in digest):
        raise ValidationError("invalid buffer SHA256")
    if path!=f"geometry/sha256-{digest}.bin":
        raise ValidationError("content-addressed buffer path mismatch")
    if prod(shape)*_DTYPES[dtype][1]>_MAX_BYTES:
        raise ResourceLimitError("numeric array size exceeds budget")
    return BufferDescriptor(path,digest,dtype,tuple(shape))

def encode_buffer(values, *, dtype: str, shape: tuple[int,...]):
    if not isinstance(dtype,str) or dtype not in _DTYPES:
        raise ValidationError("unsupported binary dtype")
    if not isinstance(shape,tuple) or len(shape)<1 or len(shape)>4 or any(type(n) is not int or n<1 for n in shape):
        raise ValidationError("invalid binary array shape")
    count=prod(shape)
    if count>_MAX_ELEMENTS or count*_DTYPES[dtype][1]>_MAX_BYTES:
        raise ResourceLimitError("numeric array exceeds budget")
    if not isinstance(values,(list,tuple)) or len(values)!=count:
        raise ValidationError("numeric array must be flat and match declared shape")
    from math import isfinite
    for v in values:
        if isinstance(v,bool) or not isinstance(v,(int,float)) or abs(v)>1.7976931348623157e308 or not isfinite(v):
            raise ValidationError("invalid numeric array element")
        if dtype=="u32" and (type(v) is not int or not 0<=v<=4294967295):
            raise ValidationError("u32 elements must be unsigned integers")
    fmt,width=_DTYPES[dtype]
    element=Struct("<"+fmt)
    raw=bytearray(count*width)
    try:
        for index,v in enumerate(values):
            element.pack_into(raw,index*width,v)
    except (OverflowError,ValueError) as exc:
        raise ValidationError("numeric element out of binary format range") from exc
    result=bytes(raw)
    content_digest=sha256(result).hexdigest()
    descriptor=BufferDescriptor(f"geometry/sha256-{content_digest}.bin",content_digest,dtype,shape)
    return descriptor,result

def decode_buffer(descriptor: dict, raw: bytes)->list:
    info=validate_descriptor(descriptor)
    _,width=_DTYPES[info.dtype]
    expected=prod(info.shape)*width
    if not isinstance(raw,bytes) or len(raw)!=expected:
        raise IntegrityError("binary buffer byte length does not match array shape")
    if sha256(raw).hexdigest()!=info.sha256:
        raise IntegrityError("binary buffer digest mismatch")
    parser=Struct("<"+_DTYPES[info.dtype][0])
    result=[v[0] for v in parser.iter_unpack(raw)]
    from math import isfinite
    if info.dtype!="u32" and any(not isfinite(x) for x in result):
        raise ValidationError("nonfinite coordinate in binary geometry")
    return result

def encode_mesh_buffers(mesh:dict):
    """Return two content-addressed parts and associated type/shape descriptors."""
    from ..sgeo.geometry import validate_geometry
    validate_geometry(mesh)
    if mesh["kind"]!="mesh":
        raise ValidationError("binary mesh encoding requires indexed triangle mesh")
    vertices=[float(value) for row in mesh["vertices"] for value in row]
    faces=[int(value) for face in mesh["faces"] for value in face]
    v,vr=encode_buffer(vertices,dtype="f64",shape=(len(mesh["vertices"]),3))
    f,fr=encode_buffer(faces,dtype="u32",shape=(len(mesh["faces"]),3))
    return {"vertices":v.to_dict(),"faces":f.to_dict()}, {v.path:vr,f.path:fr}

def decode_mesh_buffers(descriptors: dict, assets:dict[str,bytes]):
    if not isinstance(descriptors,dict) or set(descriptors)!={"vertices","faces"}:
        raise ValidationError("invalid binary mesh descriptor")
    v,f=(validate_descriptor(descriptors[name]) for name in ("vertices","faces"))
    if v.dtype not in {"f64","f32"} or f.dtype!="u32" or v.shape[-1]!=3 or f.shape[-1]!=3 or len(v.shape)!=2 or len(f.shape)!=2:
        raise ValidationError("invalid indexed geometry buffer shapes")
    try:
        vertices=decode_buffer(descriptors["vertices"],assets[v.path])
        indices=decode_buffer(descriptors["faces"],assets[f.path])
    except KeyError as exc:
        raise IntegrityError("missing referenced geometry resource") from exc
    geo={"kind":"mesh","vertices":[vertices[i:i+3] for i in range(0,len(vertices),3)],
         "faces":[indices[i:i+3] for i in range(0,len(indices),3)]}
    from ..sgeo.geometry import validate_geometry
    return validate_geometry(geo)
