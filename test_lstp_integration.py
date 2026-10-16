from __future__ import annotations

from dataclasses import replace

from lstp import canonical_dumps, canonical_loads, validate_packet
from run_lstp import frame_to_lstp, process_text_to_lstp
from src.agents.language.utils.linguistic_frame import (
    LinguisticFrame,
    SpeechActType,
)


def _directive_frame() -> LinguisticFrame:
    return LinguisticFrame(
        intent="open_door",
        entities={"target": "front door", "room": "lab"},
        sentiment=0.0,
        modality="deontic",
        confidence=0.92,
        act_type=SpeechActType.DIRECTIVE,
        propositional_content="open the front door",
        illocutionary_force="request",
    )


def test_language_frame_maps_to_canonical_lstp_without_authority() -> None:
    packet = frame_to_lstp(
        _directive_frame(),
        packet_id="pkt-1",
        thread_id="session-1",
    )

    assert validate_packet(packet).valid
    assert packet.octad.pragmatics.type == "request"
    assert packet.octad.pragmatics.speech_act == "command"
    assert packet.octad.pragmatics.goal == "slai.intent.open_door"
    assert packet.octad.confidence == 0.92

    assert packet.octad.permissions.mode is None
    assert packet.octad.permissions.scope == ()
    assert packet.octad.permissions.forbid == ()

    assert [atom.id for atom in packet.octad.atoms] == ["a0", "a1", "a2"]
    assert [atom.role for atom in packet.octad.atoms] == [
        "slai.entity.room",
        "slai.entity.target",
        "slai.proposition.content",
    ]


def test_slai_lstp_adapter_is_strict_canonical_json_round_trip() -> None:
    packet = frame_to_lstp(
        _directive_frame(),
        packet_id="pkt-1",
        thread_id="session-1",
    )
    encoded = canonical_dumps(packet)
    restored = canonical_loads(encoded, require_canonical_bytes=True)
    assert restored == packet


def test_language_frame_entity_order_is_deterministic() -> None:
    frame = _directive_frame()
    reversed_entities = dict(reversed(list(frame.entities.items())))
    reordered = replace(frame, entities=reversed_entities)

    left = frame_to_lstp(frame, packet_id="pkt-1", thread_id="session-1")
    right = frame_to_lstp(reordered, packet_id="pkt-1", thread_id="session-1")

    assert left == right
    assert canonical_dumps(left) == canonical_dumps(right)


def test_real_language_agent_to_lstp_smoke() -> None:
    packet = process_text_to_lstp(
        "Summarize the quarterly report.",
        session_id="lstp-smoke",
        packet_id="lstp-smoke-1",
    )
    assert validate_packet(packet).valid
    assert packet.protocol_version == "0.1"
    assert packet.octad.context.thread_id == "lstp-smoke"
    assert packet.octad.permissions.mode is None
    assert packet.octad.permissions.scope == ()
    assert packet.octad.evidence[0].source_type == "model"
