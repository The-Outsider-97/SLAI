"""Controlled DTO bridge to the existing Spatial Agent, imported only on demand."""
from ..common.errors import ValidationError
from ..common.helpers import vector
from ..sgeo.scene import Scene

def spatial_records(scene:Scene)->list[dict]:
    """Return JSON-safe DTOs; do not mutate SpatialMemory or import agent runtime."""
    scene.validate()
    results=[]
    for obj in scene.objects:
        geometry=obj.get("geometry")
        if geometry is None: continue
        lo,hi=scene.world_bounds(obj["id"])
        results.append({"entity_id":obj["id"],"frame_id":"world",
                        "geometry_kind":geometry["kind"],"position":[(a+b)/2 for a,b in zip(lo,hi)],
                        "bounds":{"lower":list(lo),"upper":list(hi),"frame_id":"world"},
                        "metadata":{"slai.source_document":scene.document_id,
                                    "slai.source_revision":scene.revision}})
    return results

def to_agent_bundle(scene:Scene):
    """Hydrate actual Spatial types after caller opts into loading the agent runtime."""
    from src.agents.spatial.spatial_types import SpatialEntity, SpatialBounds, SpatialBundle, GeometryKind
    items=[]
    for record in spatial_records(scene):
        enum=GeometryKind(record["geometry_kind"]) if record["geometry_kind"] in GeometryKind._value2member_map_ else GeometryKind.UNKNOWN
        bounds=record["bounds"]
        items.append(SpatialEntity(entity_id=record["entity_id"],position=tuple(record["position"]),
                                   geometry_kind=enum,bounds=SpatialBounds(tuple(bounds["lower"]),tuple(bounds["upper"])),
                                   metadata=record["metadata"]))
    return SpatialBundle(entities=tuple(items))
