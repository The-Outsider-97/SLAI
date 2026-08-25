"""Shared helper primitives for the Knowledge agent subsystem.

This module centralizes dependency-light mechanics reused by multiple
Knowledge components.

It intentionally does not own configuration loading, governance policy,
persistence, lifecycle management, action execution, ontology semantics,
or synchronization/conflict-resolution behavior.

The helpers defined here are dependency-direction leaves. They must not
import KnowledgeCache, KnowledgeMemory, KnowledgeMonitor,
KnowledgeSynchronizer, KnowledgeOrchestrator, OntologyManager, Governor,
PerformAction, or components from the runtime and validation packages.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_ACTION_DIRECTIVE_PATTERN = re.compile(
    r"action:(\w+):(.+)",
    re.IGNORECASE,
)


def match_action_directive(value: Any) -> Optional[Tuple[str, str]]:
    """Match an action directive at the beginning of a string.

    Returns the normalized action type and stripped payload when ``value``
    starts with a valid action directive. Non-string values and non-matching
    strings return ``None``.
    """
    if not isinstance(value, str):
        return None

    match = _ACTION_DIRECTIVE_PATTERN.match(value)
    if match is None:
        return None

    return match.group(1).lower(), match.group(2).strip()


def find_action_directives(value: Any) -> List[Tuple[str, str]]:
    """Return action directives found in a string in encounter order."""
    if not isinstance(value, str):
        return []

    return [
        (match.group(1).lower(), match.group(2).strip())
        for match in _ACTION_DIRECTIVE_PATTERN.finditer(value)
    ]


def config_relative_candidates(
    path: Path,
    config_file: Path,
) -> Tuple[Path, Path, Path]:
    """Build the standard Knowledge config-relative path candidates.

    Candidate ordering intentionally matches the existing Knowledge
    component lookup convention.
    """
    config_dir = config_file.resolve().parent

    return (
        config_dir / path,
        config_dir.parent / path,
        config_dir.parent.parent / path,
    )


def first_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    """Return the first existing candidate without changing its order."""
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def parse_date(value: Optional[str]) -> Optional[datetime]:
    """Parse a date string using common academic formats.
    
    Supports '%Y-%m-%d', '%Y/%m/%d', and '%Y'. Returns UTC datetime or None.
    """
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y"):
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None

def is_valid_doi(doi: str) -> bool:
    """Basic DOI validation: starts with '10.' and contains a slash."""
    doi = doi.strip()
    return bool(doi and doi.startswith("10.") and "/" in doi)

def extract_domain(url: str) -> str:
    """Return the lowercase netloc (domain) from a URL, or empty string."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.netloc.lower().strip()

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """Serialize to JSON with default=str, sorting keys, and error resilience."""
    options = {"default": str, "sort_keys": True}
    options.update(kwargs)
    return json.dumps(obj, **options)

def safe_hash(data: Dict[str, Any]) -> str:
    """Compute SHA‑256 hex digest of a dictionary using deterministic JSON serialization."""
    return hashlib.sha256(safe_json_dumps(data).encode("utf-8")).hexdigest()


__all__ = [
    "config_relative_candidates",
    "find_action_directives",
    "first_existing_path",
    "match_action_directive",
    "parse_date",
    "is_valid_doi",
    "extract_domain",
    "safe_json_dumps",
    "safe_hash"
]