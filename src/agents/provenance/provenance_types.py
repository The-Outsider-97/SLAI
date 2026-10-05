"""
Provenance Types is the domain model of the entire subsystem.
Types encode facts about derivation, not credibility scores or operational telemetry.

sources:
- W3C PROV-DM (2013) — primary source.
- Cheney, Chiticariu & Tan (2009), Provenance in Databases: Why, How, and Where.
- Buneman, Khanna & Tan (2001), Why and Where: A Characterization of Data Provenance.
"""

from __future__ import annotations

from typing import Any, Optional

from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Provenance Types")
printer = PrettyPrinter()


class ProvenanceEntity:
    pass


class ProvenanceActivity:
    pass


class ProvenanceAgentRef:
    pass


class ProvenanceRelation:
    pass


class ProvenanceAttribute:
    pass


class ProvenanceBundle:
    pass


class ProvenanceDocument:
    pass


class DerivationRecord:
    pass


class UsageRecord:
    pass


class GenerationRecord:
    pass


class AssociationRecord:
    pass


class AttributionRecord:
    pass


class CustodyRecord:
    pass


class SourceRecord:
    pass


class LineageRecord:
    pass


class CheckpointRecord:
    pass


class ProvenanceTypes:
    pass


__all__ = ["ProvenanceTypes"]

if __name__ == "__main__":
    pass