"""Offline test-only performance and relation-mapping probes."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
import tempfile
import time
from typing import Mapping
from urllib.parse import urlparse
from assurance import (AccessContext, RecordDraft, RevisionConflict, Scope, ValidationFailure)
from assurance.repository import KnowledgeRepository

@dataclass(frozen=True, slots=True)
class LoadProbeResult:
    worker_count: int
    read_operations: int
    successful_conflicting_writes: int
    correctly_rejected_conflicts: int
    errors: tuple[str, ...]
    elapsed_ms: float
    integrity_ok: bool


def run_load_probe(*, worker_count: int = 8, read_operations: int = 40) -> LoadProbeResult:
    if not isinstance(worker_count, int) or not 2 <= worker_count <= 32:
        raise ValidationFailure("worker_count must be 2..32")
    if not isinstance(read_operations, int) or not 1 <= read_operations <= 500:
        raise ValidationFailure("read_operations must be 1..500")
    with tempfile.TemporaryDirectory() as temp:
        store = KnowledgeRepository(Path(temp)/"load-probe.db")
        user = AccessContext("probe", project_ids=frozenset({"fixture"}), can_write=True)
        draft = RecordDraft("project_requirement", Scope.PROJECT, {"statement": "fixture"},
                            project_id="fixture")
        baseline = store.create(draft, user)
        start = time.perf_counter()
        errors = []
        successes = 0
        conflicts = 0

        def write(index: int) -> str:
            try:
                store.revise(baseline.record_id, baseline.revision,
                             replace(draft, payload={"statement": f"writer {index}"}),
                             user, reason="contention test")
                return "success"
            except RevisionConflict:
                return "conflict"
            except Exception:
                return "unexpected_error"

        def read() -> str:
            try:
                seen = store.query(user, project_id="fixture")
                return "read_ok" if len(seen) == 1 else "read_error"
            except Exception:
                return "read_error"

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            tasks = [pool.submit(write, n) for n in range(worker_count)]
            tasks += [pool.submit(read) for _ in range(read_operations)]
            for future in tasks:
                result = future.result()
                if result == "success":
                    successes += 1
                elif result == "conflict":
                    conflicts += 1
                elif result not in {"read_ok"}:
                    errors.append(result)
        with store._connect() as db:
            integrity = db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        return LoadProbeResult(worker_count, read_operations, successes,
                               conflicts, tuple(errors),
                               (time.perf_counter()-start)*1000, integrity)

RELATIONS = frozenset({"depends_on", "supported_by", "refers_to", "supersedes", "implements"})


@dataclass(frozen=True, slots=True)
class RelationAlignment:
    source_relation: str
    external_iri: str
    mapping_kind: str  # candidate_equivalence / broader / narrower
    reviewer_evidence_id: str


def validate_alignments(mappings: Mapping[str, RelationAlignment]) -> tuple[RelationAlignment, ...]:
    """Requires a reviewed evidence reference; a mapping is not self-verifying."""
    result = []
    for name, alignment in mappings.items():
        if name not in RELATIONS or alignment.source_relation != name:
            raise ValidationFailure("Unknown source relationship")
        if alignment.mapping_kind not in {"candidate_equivalence", "broader", "narrower"}:
            raise ValidationFailure("Unsupported semantic alignment type")
        parsed = urlparse(alignment.external_iri)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValidationFailure("External concept must be an absolute HTTP(S) IRI")
        if not alignment.reviewer_evidence_id or len(alignment.reviewer_evidence_id) > 128:
            raise ValidationFailure("Reviewer evidence reference required")
        result.append(alignment)
    return tuple(sorted(result, key=lambda row: row.source_relation))