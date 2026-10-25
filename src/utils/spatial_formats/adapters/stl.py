"""Binary STL triangle-surface interchange; labels, scale and semantics are lost."""
from __future__ import annotations
from struct import Struct
from math import sqrt
from ..common import ValidationError, ResourceLimitError, UnsupportedFeatureError
from ..sgeo.geometry import validate_geometry

TRIANGLE=Struct("<12fH")
COUNT=Struct("<I")
MAX_FACES=2_000_000

def export_binary_stl(geometry:dict) -> bytes:
    validate_geometry(geometry)
    if geometry["kind"]!="mesh":
        raise UnsupportedFeatureError("STL only represents triangle meshes")
    faces,vertices=geometry["faces"],geometry["vertices"]
    if len(faces)>MAX_FACES:
        raise ResourceLimitError("too many triangles for STL")
    result=bytearray(b"SLAI SGEO mesh".ljust(80,b" ")+COUNT.pack(len(faces)))
    for indices in faces:
        a,b,c=(vertices[i] for i in indices)
        u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)]
        normal=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
        length=sqrt(sum(n*n for n in normal))
        n=[x/length for x in normal]
        try: result.extend(TRIANGLE.pack(*(n+list(a)+list(b)+list(c)+[0])))
        except (OverflowError,ValueError) as exc:
            raise ValidationError("STL single precision coordinate overflow") from exc
    return bytes(result)

def import_binary_stl(data: bytes) -> dict:
    if not isinstance(data,bytes) or len(data)<84:
        raise ValidationError("invalid binary STL payload")
    count=COUNT.unpack_from(data,80)[0]
    if count>min(MAX_FACES,333333) or len(data)!=84+50*count:
        raise ResourceLimitError("STL size/count mismatch or triangle budget exceeded")
    vertices=[];faces=[]
    for index in range(count):
        unpacked=TRIANGLE.unpack_from(data,84+50*index)
        base=len(vertices)
        vertices.extend([list(unpacked[3:6]),list(unpacked[6:9]),list(unpacked[9:12])])
        faces.append([base,base+1,base+2])
    return validate_geometry({"kind":"mesh","vertices":vertices,"faces":faces})
