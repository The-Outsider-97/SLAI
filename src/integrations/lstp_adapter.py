"""SLAI v2.3 <-> LSTP v0.1 semantic boundary.

The adapter projects an already-produced SLAI language frame into a canonical
LSTP packet. It does not replace NLU/LANTRA and never infers execution authority
from natural language.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from lstp import (
    Atom,
    Context,
    EvidenceItem,
    Octad,
    Output,
    PacketEnvelope,
    Permissions,
    Pragmatics,
    canonical_dumps,
    canonical_loads,
    compile_canonical_lattice,
)
from lstp.models import JSONValue
from lstp.packet.compiler import CompilerOptions, compile_lattice

_QUALIFIED_IDENTIFIER = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*$"
)

_ACT_MAPPING = {
    "ASSERTIVE": ("inform", "statement"),
    "DIRECTIVE": ("request", "command"),
    "COMMISSIVE": ("assert", "statement"),
    "EXPRESSIVE": ("inform", "statement"),
    "DECLARATION": ("assert", "statement"),
}


class LSTPIntegrationError(RuntimeError):
    """SLAI data cannot be represented without guessing LSTP semantics."""


def _json_value(value: Any, *, path: str) -> JSONValue:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LSTPIntegrationError(f"non-finite value at {path}")
        return value
    if isinstance(value, Mapping):
        output: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LSTPIntegrationError(f"non-string mapping key at {path}")
            output[key] = _json_value(item, path=f"{path}.{key}")
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise LSTPIntegrationError(
        f"unsupported SLAI value type at {path}: {type(value).__name__}"
    )


def _act_name(frame: Any) -> str:
    act = getattr(frame, "act_type", None)
    name = getattr(act, "name", None)
    if isinstance(name, str) and name:
        return name.upper()
    raw = str(getattr(act, "value", act) or "").strip()
    return raw.rsplit(".", 1)[-1].upper()


def _pragmatics(frame: Any) -> Pragmatics:
    act_name = _act_name(frame)
    pragmatic_type, speech_act = _ACT_MAPPING.get(
        act_name,
        ("inform", "statement"),
    )
    intent = str(getattr(frame, "intent", "") or "").strip()
    goal = intent if _QUALIFIED_IDENTIFIER.fullmatch(intent) else None
    extensions: Mapping[str, JSONValue] = {}
    if intent and goal is None:
        extensions = {"slai": {"intent": intent}}
    return Pragmatics(
        pragmatic_type,
        speech_act=speech_act,
        goal=goal,
        extensions=extensions,
    )


def _atoms(frame: Any) -> tuple[Atom, ...]:
    atoms: list[Atom] = []
    entities = getattr(frame, "entities", {}) or {}
    if not isinstance(entities, Mapping):
        raise LSTPIntegrationError("SLAI frame entities must be a mapping")

    for name in sorted(entities, key=str):
        if not isinstance(name, str):
            raise LSTPIntegrationError("SLAI entity names must be strings")
        value = _json_value(entities[name], path=f"$.entities.{name}")
        atoms.append(
            Atom(
                f"a{len(atoms)}",
                "entity",
                value=value,
                role="ENTITY",
                attributes={"slai_entity_type": name},
            )
        )

    proposition = str(
        getattr(frame, "propositional_content", "") or ""
    ).strip()
    if proposition:
        atoms.append(
            Atom(
                f"a{len(atoms)}",
                "proposition",
                value=proposition,
                role="CONTENT",
            )
        )
    return tuple(atoms)


def packet_from_frame(
    frame: Any,
    *,
    source_text: str,
    packet_id: str,
    thread_id: str,
    output_format: str = "NL",
    permissions: Permissions | None = None,
    carrier_metadata: Mapping[str, JSONValue] | None = None,
    audit_metadata: Mapping[str, JSONValue] | None = None,
) -> PacketEnvelope:
    """Project one SLAI LinguisticFrame-like value to canonical LSTP.

    Permissions default to an empty request. This function never derives write,
    execute, or commit authority from text, intent, modality, or speech act.
    """

    if not packet_id:
        raise LSTPIntegrationError("packet_id is required")
    if not thread_id:
        raise LSTPIntegrationError("thread_id is required")

    confidence = float(getattr(frame, "confidence", 0.0) or 0.0)
    if not math.isfinite(confidence):
        raise LSTPIntegrationError("frame confidence must be finite")
    confidence = min(1.0, max(0.0, confidence))

    evidence = (
        (EvidenceItem("e0", "user", source_ref=source_text),)
        if source_text
        else ()
    )
    carrier: dict[str, JSONValue] = {
        "source": "slai_language_agent",
    }
    if carrier_metadata:
        carrier.update(carrier_metadata)

    return PacketEnvelope(
        Octad(
            _pragmatics(frame),
            _atoms(frame),
            (),
            Context(thread_id),
            confidence,
            permissions or Permissions(),
            evidence,
            Output(output_format),
        ),
        packet_id,
        "0.1",
        carrier=carrier,
        audit=dict(audit_metadata or {}),
    )


def packet_from_language_response(
    response: Any,
    *,
    source_text: str,
    packet_id: str | None = None,
    thread_id: str | None = None,
) -> PacketEnvelope:
    """Convert an existing LanguageAgentResponse-like result to LSTP."""

    frame = getattr(response, "frame", None)
    if frame is None:
        raise LSTPIntegrationError("language response has no semantic frame")
    resolved_packet_id = packet_id or str(
        getattr(response, "trace_id", "") or ""
    )
    resolved_thread_id = thread_id or str(
        getattr(response, "session_id", "") or resolved_packet_id
    )
    return packet_from_frame(
        frame,
        source_text=source_text,
        packet_id=resolved_packet_id,
        thread_id=resolved_thread_id,
        carrier_metadata={"slai_response_status": str(getattr(response, "status", ""))},
        audit_metadata={"slai_trace_id": str(getattr(response, "trace_id", ""))},
    )


def packet_to_mapping(packet: PacketEnvelope) -> dict[str, Any]:
    """Return the canonical packet as ordinary JSON-compatible Python data."""

    return json.loads(canonical_dumps(packet).decode("utf-8"))


def validate_json_packet(source: str | bytes) -> PacketEnvelope:
    """Validate one incoming canonical JSON packet."""

    return canonical_loads(source)


def canonical_lattice_to_packet(
    source: str,
    *,
    packet_id: str,
) -> PacketEnvelope:
    """Validate one canonical Lattice Octad and add the SLAI transport envelope."""

    document = compile_canonical_lattice(source)
    if len(document.packets) != 1:
        raise LSTPIntegrationError(
            "SLAI canonical-Lattice ingress accepts exactly one packet"
        )
    packet = document.packets[0]
    return PacketEnvelope(
        packet.octad,
        packet_id,
        "0.1",
        carrier={
            "source": "canonical_lattice",
            **({"label": packet.label} if packet.label else {}),
        },
        audit={},
    )


def compact_lattice_to_packet(
    source: str,
    *,
    packet_id: str,
    thread_id: str,
) -> PacketEnvelope:
    """Compile the compact Lattice authoring carrier into canonical v0.1."""

    return compile_lattice(
        source,
        options=CompilerOptions(
            packet_id=packet_id,
            thread_id=thread_id,
        ),
    )


__all__ = [
    "LSTPIntegrationError",
    "canonical_lattice_to_packet",
    "compact_lattice_to_packet",
    "packet_from_frame",
    "packet_from_language_response",
    "packet_to_mapping",
    "validate_json_packet",
]
