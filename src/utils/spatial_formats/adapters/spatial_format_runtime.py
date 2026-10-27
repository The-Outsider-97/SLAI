"""Explicit composition boundary for optional SLAI spatial-format services.

Nothing initializes an agent merely because ``spatial_formats`` was imported.
Applications opt in to individual integrations by calling a factory. Each
factory imports the relevant agent only after the explicit request, and the
Spatial session uses its own memory to avoid mixing documents or revisions.
"""

from __future__ import annotations

from .slai_stem import STEMQuantityAdapter, STEMUnitService
from .slai_spatial import SpatialSceneAdapter, spatial_reference, to_agent_bundle, to_drawing_bundle
from .slai_provenance import ProvenanceArtifactAdapter
from ..sgeo.scene import Scene
from ..sdrw.drawing import Drawing


def build_stem_adapter(unit_system: STEMUnitService | None = None) -> STEMQuantityAdapter:
    """Connect the native spatial formats to SLAI's real STEM ``UnitSystem``.

    A caller may provide its configured unit system, avoiding extra registry
    instances. Without one, this explicit factory constructs the actual STEM
    service using its existing configuration loader. Importing the package or
    the adapter does not trigger that scientific stack.
    """

    if unit_system is None:
        from src.agents.stem.units.unit_system import UnitSystem

        unit_system = UnitSystem()
    return STEMQuantityAdapter(unit_system)


def build_spatial_adapter(scene: Scene | Drawing, *, object_ids=None, sha256: str | None = None,
                          uri: str | None = None) -> SpatialSceneAdapter:
    """Create a document-revision-scoped bridge to real Spatial services.

    The scene/drawing is converted to world/model-space SI entities. No global
    SpatialMemory is touched. Recreate the session after document modifications.
    """
    from src.agents.spatial.spatial_memory import SpatialMemory
    from src.agents.spatial.spatial_compute import SpatialCompute
    from src.agents.spatial.spatial_queries import SpatialQueries

    reference = spatial_reference(scene, sha256=sha256, uri=uri, object_ids=object_ids)
    bundle = (to_agent_bundle(scene, object_ids=object_ids, unit="m")
              if isinstance(scene, Scene)
              else to_drawing_bundle(scene, entity_ids=object_ids, unit="m"))
    memory = SpatialMemory()
    for entity in bundle.entities:
        memory.upsert_entity(entity)
    return SpatialSceneAdapter(
        reference=reference,
        entity_ids=frozenset(entity.entity_id for entity in bundle.entities),
        memory=memory,
        compute=SpatialCompute(memory=memory),
        queries=SpatialQueries(memory=memory),
    )


def build_provenance_adapter(*, store=None) -> ProvenanceArtifactAdapter:
    """Connect spatial artifacts to real persistent Provenance infrastructure.

    The caller may pass an already configured ProvenanceStore, including a
    ``persist=False`` store for controlled testing. The default uses SLAI's
    normal Provenance configuration. Nothing is recorded until an adapter
    method is explicitly called.
    """
    from src.agents.provenance.provenance_store import ProvenanceStore
    from src.agents.provenance.provenance_lineage import ProvenanceLineage

    selected_store = store if store is not None else ProvenanceStore()
    return ProvenanceArtifactAdapter(selected_store, ProvenanceLineage(store=selected_store))


__all__ = ["build_stem_adapter", "build_spatial_adapter", "build_provenance_adapter"]
