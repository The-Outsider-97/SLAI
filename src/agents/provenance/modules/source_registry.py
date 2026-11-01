"""Stable source identity registry for SLAI provenance.

Inspired by FAIR persistent identification, Software Heritage content identity,
and Buneman-style origin provenance.  This registry identifies a source; it
never assigns trust, reliability, quality, or credibility scores.
"""
from __future__ import annotations

__version__ = "2.3.0"

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.provenance_errors import *
from ..utils.provenance_helpers import *
from ..provenance_store import ProvenanceStore
from ..provenance_types import SourceRecord
from logs.logger import PrettyPrinter, configure_logging, get_logger  # pyright: ignore[reportMissingImports]


logger = get_logger("Source Registry")
printer = PrettyPrinter()

_PROHIBITED_SCORING_FIELDS = frozenset({
    "trust_score", "reliability_score", "quality_score", "credibility_score",
    "trust", "reliability", "credibility", "quality_rating",
})


def _find_scoring_fields(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text.lower() in _PROHIBITED_SCORING_FIELDS:
                found.append(path)
            found.extend(_find_scoring_fields(item, path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_find_scoring_fields(item, f"{prefix}[{index}]"))
    return found


class SourceRegistry:
    """Register and resolve stable source identities without judging them."""

    def __init__(self, store: Optional[ProvenanceStore] = None) -> None:
        self.config = load_global_config()
        self.source_registry_config = get_config_section("source_registry", config=self.config, default={})
        self.store = store or ProvenanceStore()

    def register_source(self, source_id: str, source_info: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(source_info, Mapping):
            raise ProvenanceValidationError("source_info must be a mapping", context={"type": type(source_info).__name__})
        forbidden = sorted(set(_find_scoring_fields(source_info)))
        if forbidden:
            raise ProvenanceSourceError(
                "source registry does not accept trust/quality scoring fields",
                context={"fields": forbidden, "source_id": source_id},
            )

        info = dict(source_info)
        source_type = str(info.pop("source_type", info.pop("type", "unknown")))
        locator = info.pop("locator", None) or info.pop("uri", None) or info.pop("url", None) or info.pop("path", None)
        digest = info.pop("digest", None) or info.pop("content_digest", None) or info.pop("hash", None)
        external_identifier = info.pop("external_identifier", None) or info.pop("doi", None) or info.pop("external_id", None)
        registered_at = info.pop("registered_at", None) or info.pop("timestamp", None) or get_current_timestamp()
        supplied_metadata = info.pop("metadata", {})
        content = info.pop("content", None)
        compute_locator_digest = bool(info.pop("compute_digest", False))

        if digest is None and content is not None:
            digest = content_digest(content)
        if digest is None and compute_locator_digest and locator:
            candidate = Path(str(locator)).expanduser()
            if candidate.exists() and candidate.is_file():
                digest = content_digest(candidate.resolve())

        metadata = normalize_metadata(supplied_metadata if isinstance(supplied_metadata, Mapping) else {})
        if info:
            metadata = {**metadata, **normalize_metadata(info)}

        record = SourceRecord(
            source_id=require_identifier(source_id, field_name="source_id"),
            source_type=source_type,
            locator=str(locator) if locator is not None else None,
            digest=str(digest) if digest is not None else None,
            external_identifier=str(external_identifier) if external_identifier is not None else None,
            registered_at=registered_at,
            metadata=metadata,
        )
        return self.store.save_source_record(record)

    def get_source(self, source_id: str) -> dict[str, Any]:
        result = self.store.get_source(source_id, strict=True)
        assert result is not None
        return result

    def list_sources(self) -> list[dict[str, Any]]:
        return self.store.list_sources()


__all__ = ["SourceRegistry"]

if __name__ == "__main__":
    configure_logging()
    printer.status("SMOKE", "SourceRegistry module loaded", "success")
