"""SLAI v2.3 -> LSTP v0.1 integration boundary.

The adapter converts the real LanguageAgent LinguisticFrame into a canonical
LSTP packet. Language semantics never grant execution authority: canonical
permissions remain empty and host authorization is a separate boundary.
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from logs.logger import PrettyPrinter, get_logger
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
    validate_packet,
)
from src.agents.agent_factory import AgentFactory
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.language.utils.linguistic_frame import LinguisticFrame, SpeechActType
from src.agents.language_agent import LanguageAgent

logger = get_logger("LSTP Integration")
printer = PrettyPrinter()

_IDENTIFIER_PART = re.compile(r"[^A-Za-z0-9_-]+")
_SPEECH_MAP: dict[SpeechActType, tuple[str, str]] = {
    SpeechActType.ASSERTIVE: ("assert", "statement"),
    SpeechActType.DIRECTIVE: ("request", "command"),
    SpeechActType.COMMISSIVE: ("inform", "slai.commissive"),
    SpeechActType.EXPRESSIVE: ("inform", "slai.expressive"),
    SpeechActType.DECLARATION: ("inform", "slai.declaration"),
}


class LSTPIntegrationError(ValueError):
    """Raised when SLAI semantic output cannot be represented losslessly."""


def _identifier_part(value: str, *, fallback: str) -> str:
    normalized = _IDENTIFIER_PART.sub("_", str(value).strip()).strip("_-")
    if not normalized:
        normalized = fallback
    if not (normalized[0].isalpha() or normalized[0] == "_"):
        normalized = "_" + normalized
    return normalized[:96]


def _qualified_goal(intent: str) -> str:
    return "slai.intent." + _identifier_part(intent, fallback="unknown")


def _entity_role(name: str) -> str:
    return "slai.entity." + _identifier_part(name, fallback="entity")


def _json_value(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, Enum):
        return _json_value(value.value, path=path)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value), path=path)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LSTPIntegrationError(
                    f"non-string entity mapping key at {path}"
                )
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise LSTPIntegrationError(
        f"unsupported SLAI semantic value at {path}: {type(value).__name__}"
    )


def frame_to_lstp(
    frame: LinguisticFrame,
    *,
    packet_id: str,
    thread_id: str,
    output_format: str = "JSON",
) -> PacketEnvelope:
    """Convert one LanguageAgent frame to a semantically validated LSTP packet."""

    if not packet_id.strip():
        raise LSTPIntegrationError("packet_id must be non-empty")
    if not thread_id.strip():
        raise LSTPIntegrationError("thread_id must be non-empty")

    pragmatic_type, speech_act = _SPEECH_MAP.get(
        frame.act_type,
        ("inform", "slai.unknown"),
    )

    atoms: list[Atom] = []
    for name in sorted(frame.entities or {}):
        value = _json_value(frame.entities[name], path=f"$.entities.{name}")
        atoms.append(
            Atom(
                f"a{len(atoms)}",
                "entity",
                value=value,
                role=_entity_role(name),
            )
        )

    if frame.propositional_content:
        atoms.append(
            Atom(
                f"a{len(atoms)}",
                "proposition",
                value=frame.propositional_content,
                role="slai.proposition.content",
            )
        )

    octad = Octad(
        pragmatics=Pragmatics(
            type=pragmatic_type,
            speech_act=speech_act,
            goal=_qualified_goal(frame.intent),
        ),
        atoms=tuple(atoms),
        relations=(),
        context=Context(thread_id=thread_id),
        confidence=float(frame.confidence),
        permissions=Permissions(),
        evidence=(
            EvidenceItem(
                "e0",
                "model",
                source_ref="slai.language_agent",
            ),
        ),
        output=Output(format=output_format),
    )
    packet = PacketEnvelope(
        octad=octad,
        packet_id=packet_id,
        protocol_version="0.1",
        carrier={"profile": "slai.language-frame"},
        audit={"source": "SLAI.LanguageAgent"},
    )
    validate_packet(packet).raise_for_errors()
    return packet


def process_text_to_lstp(
    text: str,
    *,
    session_id: str,
    packet_id: str,
    language_agent: LanguageAgent | None = None,
) -> PacketEnvelope:
    """Run the real SLAI LanguageAgent and convert its frame to LSTP."""

    source = str(text or "").strip()
    if not source:
        raise LSTPIntegrationError("text must be non-empty")

    agent = language_agent
    if agent is None:
        shared_memory = SharedMemory()
        factory = AgentFactory()
        agent = LanguageAgent(
            shared_memory=shared_memory,
            agent_factory=factory,
        )

    response = agent.process(source, session_id=session_id)
    if response.frame is None:
        raise LSTPIntegrationError(
            "LanguageAgent returned no LinguisticFrame"
        )
    return frame_to_lstp(
        response.frame,
        packet_id=packet_id,
        thread_id=session_id,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SLAI language understanding through canonical LSTP v0.1."
    )
    parser.add_argument("text", nargs="?", help="Natural-language input.")
    parser.add_argument(
        "--session-id",
        default=None,
        help="Stable SLAI/LSTP thread identifier.",
    )
    parser.add_argument(
        "--packet-id",
        default=None,
        help="Explicit LSTP envelope packet id.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.text:
        raise SystemExit("text is required")

    session_id = args.session_id or f"session-{uuid.uuid4().hex}"
    packet_id = args.packet_id or f"pkt-{uuid.uuid4().hex}"

    logger.info("Processing SLAI language input through LSTP")
    packet = process_text_to_lstp(
        args.text,
        session_id=session_id,
        packet_id=packet_id,
    )
    rendered = canonical_dumps(packet).decode("utf-8")
    print(rendered)
    printer.status("SUCCESS", "SLAI -> LSTP packet validated", "success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
