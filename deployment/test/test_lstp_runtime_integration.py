from __future__ import annotations

from types import SimpleNamespace

import pytest

lstp = pytest.importorskip("lstp")

from src.integrations.lstp_adapter import (  # noqa: E402
    canonical_lattice_to_packet,
    compact_lattice_to_packet,
    packet_from_frame,
    packet_from_language_response,
    packet_to_mapping,
    validate_json_packet,
)


def _frame() -> SimpleNamespace:
    return SimpleNamespace(
        intent="open",
        entities={"door": "front"},
        confidence=0.9,
        act_type=SimpleNamespace(name="DIRECTIVE"),
        propositional_content="open the front door",
    )


def test_language_frame_projects_without_execution_authority() -> None:
    packet = packet_from_frame(
        _frame(),
        source_text="Open the front door",
        packet_id="p1",
        thread_id="t1",
    )
    assert packet.octad.pragmatics.type == "request"
    assert packet.octad.pragmatics.speech_act == "command"
    assert packet.octad.pragmatics.goal == "open"
    assert packet.octad.permissions.mode is None
    assert packet.octad.permissions.scope == ()
    assert packet.octad.evidence[0].source_type == "user"

    mapping = packet_to_mapping(packet)
    assert mapping["permissions"] == {}
    assert validate_json_packet(lstp.canonical_dumps(packet)) == packet


def test_language_response_boundary_uses_trace_and_session_identity() -> None:
    response = SimpleNamespace(
        frame=_frame(),
        trace_id="lang-abc",
        session_id="session-1",
        status="success",
    )
    packet = packet_from_language_response(
        response,
        source_text="Open the front door",
    )
    assert packet.packet_id == "lang-abc"
    assert packet.octad.context.thread_id == "session-1"
    assert packet.audit["slai_trace_id"] == "lang-abc"


def test_canonical_lattice_ingress_adds_only_transport_envelope() -> None:
    source = (
        '[π=(TYPE=INFORM)|A=()|R=()|C=(THREAD="t1")|κ=1|'
        'Π=()|E=()|Ω=(FORMAT=NL)]'
    )
    packet = canonical_lattice_to_packet(source, packet_id="p2")
    assert packet.packet_id == "p2"
    assert packet.octad.context.thread_id == "t1"
    assert packet.octad.permissions.mode is None


def test_compact_lattice_ingress_preserves_explicit_preview_request() -> None:
    packet = compact_lattice_to_packet(
        '!draft @email {mode=PREVIEW, scope=["urn:email:draft:1"]} -> NL',
        packet_id="p3",
        thread_id="t1",
    )
    assert packet.octad.permissions.mode == "PREVIEW"
    assert packet.octad.permissions.scope == ("urn:email:draft:1",)
