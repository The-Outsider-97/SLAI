"""Paired synthetic examples validated against the installed LSTP implementation."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass

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
)

RESOURCES = (
    ("spadra", "roof_connection_C4", "schematic design"),
    ("spadra", "wall_floor_joint_D2", "detailed design"),
    ("spadra", "parametric_stair", "concept design"),
    ("spadra", "curtain_wall_family", "technical design"),
    ("slai_writer", "contents_table", "document editing"),
    ("slai_writer", "footer_layout", "document editing"),
    ("slai_writer", "table_styles", "quality review"),
    ("documaster", "pdf_annotations", "quality review"),
    ("documaster", "batch_conversion", "workflow review"),
    ("scical", "molecule_viewer", "scientific visualization"),
    ("scical", "periodic_table", "dataset review"),
    ("sreader", "article_extract", "research review"),
    ("splanner", "milestone_plan", "planning review"),
    ("cadcon", "sdrw_drawing", "CAD review"),
    ("cadcon", "sgeo_modifier", "geometry review"),
    ("slai", "agent_router", "system review"),
    ("slai", "language_tokenizer", "language review"),
    ("slai", "memory_retrieval", "data review"),
    ("slai", "authorization_policy", "security review"),
    ("slai", "evaluation_pipeline", "testing review"),
)

ACTIONS = ("inspect", "summarize", "compare", "verify", "describe")
MODES = ("RO", "SUGGEST", "PREVIEW")
FORMATS = ("NL", "TABLE", "JSON")
STAGES = (
    "concept review",
    "schematic review",
    "technical review",
    "implementation review",
    "verification review",
    "release review",
)

NL_PATTERNS = (
    (
        "In project {project}, during {stage}, {action} resource {resource}. "
        "You may {mode_description}. Present the result as {output_format}. "
        "The source identifier is {source}."
    ),
    (
        "Task for {project}: {action} {resource} at the {stage} stage. "
        "Authorization: {mode_description}. Required format: {output_format}. "
        "Source reference: {source}."
    ),
    (
        "Source {source} concerns {resource} in {project} ({stage}). "
        "Please {action} it; {mode_description}. Return {output_format}."
    ),
    (
        "For {stage} in {project}, use {source} to {action} {resource}. "
        "The permission is to {mode_description}. Output must use {output_format}."
    ),
)

MODE_DESCRIPTIONS = {
    "RO": "read the target only, without suggesting or making changes",
    "SUGGEST": "read the target and suggest improvements, without making changes",
    "PREVIEW": "read the target and simulate proposed changes, without saving changes",
}


@dataclass(frozen=True)
class Case:
    case_id: str
    natural_language: str
    ordinary_json: str
    canonical_lstp: str
    gold: dict[str, str]

    def as_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "natural_language": self.natural_language,
            "ordinary_json": self.ordinary_json,
            "canonical_lstp": self.canonical_lstp,
            "gold": self.gold,
        }


def generate_cases(count: int = 1000, seed: int = 20261117) -> list[Case]:
    """Build deterministic semantically paired inputs, without any API requests."""
    designs = list(itertools.product(RESOURCES, ACTIONS, MODES, FORMATS, STAGES))
    if not 1 <= count <= len(designs):
        raise ValueError(f"case count must be between 1 and {len(designs)}")

    random.Random(seed).shuffle(designs)
    cases: list[Case] = []
    for index, (
        (project, resource, _),
        action,
        mode,
        output_format,
        stage,
    ) in enumerate(designs[:count]):
        case_id = f"case-{index + 1:05d}"
        source = f"urn:lstp:benchmark:{case_id}"
        gold = {
            "action": action,
            "resource": resource,
            "project": project,
            "stage": stage,
            "permission_mode": mode,
            "output_format": output_format,
            "source": source,
        }

        natural_language = NL_PATTERNS[index % len(NL_PATTERNS)].format(
            project=project,
            stage=stage,
            action=action,
            resource=resource,
            mode_description=MODE_DESCRIPTIONS[mode],
            output_format=output_format,
            source=source,
        )
        ordinary_json = json.dumps(
            {
                "action": action,
                "target_resource": resource,
                "project": project,
                "stage": stage,
                "requested_permission": mode,
                "output_format": output_format,
                "source_identifier": source,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        packet = PacketEnvelope(
            octad=Octad(
                pragmatics=Pragmatics("request", speech_act="command", goal=f"slai.{action}"),
                atoms=(
                    Atom("a0", "resource", value=resource, role="TARGET"),
                    Atom("a1", "concept", value=action, role="ACTION"),
                ),
                relations=(),
                context=Context(
                    thread_id=f"benchmark_{case_id}",
                    bindings={"project": project, "stage": stage},
                ),
                confidence=1.0,
                permissions=Permissions(mode=mode, scope=(resource,)),
                evidence=(EvidenceItem("e0", "user", source_ref=source),),
                output=Output(output_format),
            ),
            packet_id=f"lstp_{case_id}",
            protocol_version="0.1",
        )
        wire = canonical_dumps(packet)
        decoded = canonical_loads(wire, require_canonical_bytes=True)
        if not decoded.semantically_equals(packet):
            raise AssertionError(f"LSTP round-trip changed meaning for {case_id}")
        cases.append(Case(case_id, natural_language, ordinary_json, wire.decode(), gold))
    return cases


def corpus_sha256(cases: list[Case]) -> str:
    data = "".join(
        json.dumps(case.as_record(), ensure_ascii=False, sort_keys=True) + "\n" for case in cases
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
