"""Real Spatial/Provenance subsystem integration, with no fake agent services."""

from __future__ import annotations

import subprocess
import sys

import pytest

from src.utils.spatial_formats.adapters.slai_provenance import make_observation
from src.utils.spatial_formats.adapters.spatial_format_runtime import (
    build_provenance_adapter,
    build_spatial_adapter,
)
from src.utils.spatial_formats.common.errors import ValidationError
from src.utils.spatial_formats.sdrw.drawing import Drawing
from src.utils.spatial_formats.sgeo.scene import Scene


def _move_x(distance: float) -> list[float]:
    return [1, 0, 0, distance, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def _three_boxes() -> tuple[Scene, tuple[str, str, str]]:
    scene = Scene(name="Spatial bridge test", unit="cm")
    bounds = {"kind": "box", "center": [0, 0, 0], "size": [100, 100, 100]}
    first = scene.add_object(geometry=bounds)
    second = scene.add_object(geometry=bounds, transform=_move_x(200))
    overlapping = scene.add_object(geometry=bounds, transform=_move_x(50))
    return scene, (first, second, overlapping)


def test_real_spatial_session_uses_isolated_si_memory():
    from src.agents.spatial.spatial_compute import SpatialCompute
    from src.agents.spatial.spatial_memory import SpatialMemory
    from src.agents.spatial.spatial_queries import SpatialQueries

    scene, (first, second, _) = _three_boxes()
    adapter = build_spatial_adapter(scene)
    assert isinstance(adapter.memory, SpatialMemory)
    assert isinstance(adapter.compute, SpatialCompute)
    assert isinstance(adapter.queries, SpatialQueries)
    assert adapter.reference["document_id"] == scene.document_id
    assert adapter.reference["units"] == "cm"
    assert adapter.entity(first).bounds.lower == (-0.5, -0.5, -0.5)
    assert adapter.distance(first, second) == pytest.approx(2.0)
    assert adapter.distance(first, second, unit="cm") == pytest.approx(200.0)


def test_real_spatial_queries_and_broad_phase():
    scene, (first, second, overlapping) = _three_boxes()
    adapter = build_spatial_adapter(scene)
    assert adapter.nearest((0, 0, 0), unit="cm").entity_ids == (first,)
    assert adapter.nearest((0, 0, 0), unit="m", k=2).entity_ids == (first, overlapping)
    assert adapter.within_radius((0, 0, 0), 80, unit="cm").entity_ids == (first, overlapping)
    assert adapter.intersecting_bounds(first) == (overlapping,)
    assert adapter.centre_relation(second, first, "right_of")
    assert not adapter.centre_relation(first, second, "right_of")


def test_spatial_selection_is_strict_and_document_sessions_do_not_share_state():
    scene, (first, second, _) = _three_boxes()
    selected = build_spatial_adapter(scene, object_ids=[first])
    assert selected.entity_ids == {first}
    with pytest.raises(ValidationError):
        selected.entity(second)
    assert len(selected.memory.list_entities()) == 1
    other = build_spatial_adapter(scene)
    assert len(other.memory.list_entities()) == 3
    assert selected.memory is not other.memory
    with pytest.raises(ValidationError, match="unknown spatial reference selection"):
        build_spatial_adapter(scene, object_ids=["00000000-0000-0000-0000-000000000000"])


def test_spatial_rejects_unsupported_topology_and_invalid_units():
    scene, (first, second, _) = _three_boxes()
    adapter = build_spatial_adapter(scene)
    with pytest.raises(ValidationError, match="centre relation"):
        adapter.centre_relation(first, second, "contains")
    with pytest.raises(ValidationError):
        adapter.distance(first, second, unit="parsec")
    with pytest.raises(ValidationError):
        adapter.nearest((0, 0, 0), unit="invalid")


def test_spatial_session_is_a_snapshot_not_live_scene_state():
    scene, (first, second, _) = _three_boxes()
    adapter = build_spatial_adapter(scene)
    original = adapter.distance(first, second)
    scene.add_object(geometry={"kind": "box", "center": [0, 0, 0], "size": [1, 1, 1]})
    assert adapter.distance(first, second) == original
    assert len(adapter.entity_ids) == 3


def test_real_spatial_agent_supports_sdrw_2d_centres_without_geometry_duplication():
    drawing = Drawing(name="2D query test", unit="mm")
    first = drawing.add("line", {"start": [0, 0], "end": [1000, 0]})
    second = drawing.add("line", {"start": [2000, 0], "end": [3000, 0]})
    bridge = build_spatial_adapter(drawing)
    assert bridge.reference["format_id"] == "slai.sdrw"
    assert bridge.entity(first).dimension == 2
    assert bridge.entity(first).bounds.lower == (0.0, 0.0)
    assert bridge.distance(first, second) == pytest.approx(2.0)
    assert bridge.nearest((400, 0), unit="mm").entity_ids == (first,)
    assert bridge.intersecting_bounds(first) == ()


def test_real_provenance_store_persists_spatial_artifact_and_retries_in_memory():
    from src.agents.provenance.provenance_store import ProvenanceStore
    from src.agents.provenance.provenance_lineage import ProvenanceLineage

    store = ProvenanceStore(persist=False)
    bridge = build_provenance_adapter(store=store)
    assert isinstance(bridge.lineage, ProvenanceLineage)
    scene, _ = _three_boxes()
    content_sha256 = "a" * 64
    saved = bridge.record_scene(scene, sha256=content_sha256)
    assert saved["digest"] == content_sha256
    assert saved["metadata"]["document_id"] == scene.document_id
    assert not saved["metadata"]["digest_verified_by_adapter"]
    assert bridge.record_scene(scene, sha256=content_sha256) == saved
    assert store.get_entity(saved["entity_id"]) == saved
    with pytest.raises(ValidationError, match="different revision or digest"):
        bridge.record_scene(scene, sha256="b" * 64, artifact_id=saved["entity_id"])


def test_real_provenance_claim_is_a_claim_not_an_authenticated_fact():
    from src.agents.provenance.provenance_store import ProvenanceStore

    store = ProvenanceStore(persist=False)
    bridge = build_provenance_adapter(store=store)
    scene, (first, _, _) = _three_boxes()
    claim = make_observation(first, "adjacent_to", evidence=f"sgeo:{scene.document_id}",
                             confidence=0.65, agent="Spatial")
    saved = bridge.record_observation(claim)
    assert saved["entity_type"] == "spatial_observation"
    assert saved["metadata"]["claim"]["confidence"] == pytest.approx(0.65)
    assert saved["metadata"]["independently_verified"] is False
    assert bridge.record_observation(claim) == saved
    assert store.get_entity(saved["entity_id"]) == saved


def test_real_provenance_records_sdrw_artifact_with_distinct_format_id():
    from src.agents.provenance.provenance_store import ProvenanceStore

    drawing = Drawing(name="2D design", unit="mm")
    drawing.add("line", {"start": [0, 0], "end": [100, 0]})
    bridge = build_provenance_adapter(store=ProvenanceStore(persist=False))
    result = bridge.record_document(drawing, sha256="f" * 64)
    assert result["metadata"]["format_id"] == "slai.sdrw"
    assert result["entity_id"].startswith("sdrw:")
    assert bridge.record_document(drawing, sha256="f" * 64) == result


def test_provenance_agent_durably_restores_artifact_and_lineage(tmp_path):
    from src.agents.provenance.provenance_store import ProvenanceStore

    path = tmp_path / "provenance.json"
    original = build_provenance_adapter(store=ProvenanceStore(path, persist=True))
    scene, _ = _three_boxes()
    record = original.record_document(scene, sha256="b" * 64)
    original.record_derivation(record["entity_id"], ("source:survey",),
                               transformation="imported", agent_id="agent:spatial")
    assert path.is_file()

    restored = build_provenance_adapter(store=ProvenanceStore(path, persist=True))
    assert restored.store.get_entity(record["entity_id"]) == record
    assert len(restored.lineage_for(record["entity_id"])) == 1


def test_provenance_lineage_uses_real_dag_cycle_detection():
    from src.agents.provenance.provenance_store import ProvenanceStore
    from src.agents.provenance.utils.provenance_errors import ProvenanceGraphError

    bridge = build_provenance_adapter(store=ProvenanceStore(persist=False))
    one = bridge.record_derivation("model:v2", ("model:v1",), transformation="edited")
    assert "model:v1" in one["parent_artifact_ids"]
    assert len(bridge.lineage_for("model:v2")) == 1
    with pytest.raises(ProvenanceGraphError):
        bridge.record_derivation("model:v1", ("model:v2",), transformation="edited")


def test_provenance_invalid_digest_rejected_before_persistence():
    from src.agents.provenance.provenance_store import ProvenanceStore

    store = ProvenanceStore(persist=False)
    scene, _ = _three_boxes()
    with pytest.raises(ValidationError):
        build_provenance_adapter(store=store).record_scene(scene, sha256="NOT-A-DIGEST")
    assert store.list_entities() == []


def test_importing_format_adapters_does_not_import_spatial_or_provenance_agent():
    statement = (
        "import sys; "
        "import src.utils.spatial_formats.adapters.slai_spatial; "
        "import src.utils.spatial_formats.adapters.slai_provenance; "
        "import src.utils.spatial_formats.adapters.spatial_format_runtime; "
        "assert 'src.agents.spatial' not in sys.modules; "
        "assert 'src.agents.provenance' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", statement], check=True, timeout=25)
