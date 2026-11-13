"""Application-independent façade for the three P0/P1 assurance capabilities.

This is an opt-in shared service, NOT a replacement Knowledge Agent.
Authoritative caller identity is an external prerequisite. In production,
applications should build AccessContext from their established auth/session
boundary; source documents must never supply their own project permission sets.
"""
from __future__ import annotations
from .standards import PublisherUpdateReview
from .project import ProjectAdoptionService, HistoricalLineageService, KnowledgePromotionService
from .rules import AssessmentService
from .repository import KnowledgeOutbox, OutboxDelivery, KnowledgeBackupService
from .contracts import AuthorizationDenied, ValidationFailure, Scope, Lifecycle, digest_json, normalize_time, validate_id, utc_now
from typing import Callable, Sequence
import time
import shutil
import sqlite3
import json
import re
import math
from enum import Enum
from dataclasses import dataclass
from collections import Counter

from pathlib import Path
from typing import Any, Mapping

from .contracts import AccessContext, RecordDraft, RecordRevision, RuleEvaluation
from .project import ProjectKnowledgeService
from .repository import KnowledgeRepository
from .rules import RulesRegistry
from .standards import StandardsRegistry
from ..utils.config_loader import get_config_section, load_global_config
from logs.logger import PrettyPrinter, configure_logging, get_logger # pyright: ignore[reportMissingImports]


logger = get_logger("Assurance Gateway")
printer = PrettyPrinter()


class KnowledgeAssuranceGateway:
    """Single entry point for canonical knowledge records, standards and rules."""

    def __init__(self, sqlite_path: str | Path, *, provenance_agent: Any = None,
                 verification_agent: Any = None):
        self.repository = KnowledgeRepository(sqlite_path)
        self.standards = StandardsRegistry(self.repository)
        self.rules = RulesRegistry(self.repository)
        self.projects = ProjectKnowledgeService(self.repository)
        self.assessments = AssessmentService(self.repository, self.rules)
        self.adoptions = ProjectAdoptionService(self.repository)
        self.lineage = HistoricalLineageService(self.repository)
        self.promotions = KnowledgePromotionService(self.repository)
        self.outbox = KnowledgeOutbox(self.repository)
        self.retrieval = ScopedLexicalRetriever(self.repository)
        self.sufficiency = KnowledgeSufficiencyService(self.repository)
        self.diagnostics = AssuranceDiagnostics(self.repository, self.outbox)
        self.backup = KnowledgeBackupService(self.repository)
        self.publisher_updates = PublisherUpdateReview(self.repository)
        # Explicit peer agents remain owners of their own functions.
        self.provenance_agent = provenance_agent
        self.verification_agent = verification_agent

        logger.info(f"Knowledge Assurance Gateway initialized")

    def put_knowledge(self, draft: RecordDraft, access: AccessContext, *, record_id: str | None = None) -> RecordRevision:
        return self.repository.create(draft, access, record_id=record_id)

    def revise_knowledge(self, record_id: str, expected_revision: int,
                         draft: RecordDraft, access: AccessContext,
                         *, reason: str) -> RecordRevision:
        return self.repository.revise(record_id, expected_revision, draft, access, reason=reason)

    def find_knowledge(self, access: AccessContext | None, *, kind: str | None = None,
                       query: str | None = None, as_known: str | None = None,
                       as_valid: str | None = None, limit: int = 100) -> list[RecordRevision]:
        return self.repository.query(access, kind=kind, text=query,
                                     as_known=as_known, as_valid=as_valid, limit=limit)

    def resolve_applicable_requirements(self, access: AccessContext | None,
                                        context: Mapping[str, Any]) -> list[dict[str, Any]]:
        return self.standards.applicable_requirements(access, context=context)

    def evaluate_rule(self, record_id: str, inputs: dict[str, Any],
                      access: AccessContext | None) -> RuleEvaluation:
        return self.rules.evaluate(record_id, inputs=inputs, access=access)

    def register_provenance_source(self, source_id: str, source_info: Mapping[str, Any]) -> dict[str, Any]:
        """Delegate identity/digest registration to the existing Provenance Agent."""
        if self.provenance_agent is None:
            raise RuntimeError("ProvenanceAgent not attached; cannot register authoritative source reference")
        return self.provenance_agent.register_source(source_id, source_info)

    def delegate_formal_verification(self, request: Any) -> Any:
        """No implicit assertion-to-proof conversion or compliance certification."""
        if self.verification_agent is None:
            raise RuntimeError("VerificationAgent not attached; cannot verify formal property")
        return self.verification_agent.verify(request)


__all__ = ["KnowledgeAssuranceGateway"]

_TOKEN = re.compile(r"(?u)\b\w+\b")


def _scoped_tokens(text: str) -> list[str]:
    return [x.casefold() for x in _TOKEN.findall(text)][:8192]


def _searchable_text(record: RecordRevision) -> str:
    # Search only plaintext metadata fields; never include internal evidence,
    # source access tokens, confidential attestation bodies or raw source text.
    allowed = {
        "statement", "title", "publisher", "standard_id", "edition",
        "candidate_paraphrase", "clause_locator", "description", "rationale",
    }
    fields = []
    for name, value in record.draft.payload.items():
        if name == "candidate_paraphrase" and record.draft.payload.get("content_rights") not in {
            "public_domain", "authorized_extract", "licensed_content"
        }:
            continue  # Missing explicit AI-processing rights => metadata-only.
        if name in allowed and isinstance(value, str):
            fields.append(value[:4096])
    return " ".join(fields)


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    record_id: str
    revision: int
    kind: str
    scope: str
    score: float
    source_ids: tuple[str, ...]
    evidence_count: int
    record_digest: str


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query_digest: str
    hits: tuple[RetrievalHit, ...]
    scanned: int
    truncated: bool
    corpus_watermark: str
    as_known: str | None
    as_valid: str | None
    project_id: str | None
    engine: str = "slai.assurance.bounded_bm25.v1"


class ScopedLexicalRetriever:
    def __init__(self, repository: KnowledgeRepository, *, max_candidates: int = 3000):
        if not isinstance(max_candidates, int) or not 1 <= max_candidates <= 10000:
            raise ValidationFailure("max_candidates must be 1..10000")
        self.repository = repository
        self.max_candidates = max_candidates

    def search(self, query: str, access: AccessContext | None, *,
               project_id: str | None = None, kind: str | None = None,
               as_known: str | None = None, as_valid: str | None = None,
               k: int = 10) -> RetrievalResult:
        if not isinstance(query, str) or not query.strip() or len(query) > 512:
            raise ValidationFailure("Nonempty query of at most 512 characters required")
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 100:
            raise ValidationFailure("k must be 1..100")
        if project_id is not None:
            validate_id(project_id, "project_id")
            if access is None or project_id not in access.project_ids:
                raise AuthorizationDenied(access.principal_id if access is not None else "anonymous", "knowledge_access", details={"reason": "Unverified project context"})
        as_known = normalize_time(as_known, "as_known")
        as_valid = normalize_time(as_valid, "as_valid")
        words = _scoped_tokens(query)
        if not words:
            raise ValidationFailure("Query must contain indexable tokens")
        documents: list[tuple[RecordRevision, Counter[str], int]] = []
        offset = 0
        # Scope restriction is performed by the SQL query, not by a ranker.
        with self.repository._connect() as db:
            db.execute("BEGIN")  # Stable WAL read snapshot across pagination.
            try:
                while offset < self.max_candidates:
                    page_size = min(500, self.max_candidates - offset)
                    page = self.repository.query(access, kind=kind, as_known=as_known,
                                                 as_valid=as_valid, limit=page_size, offset=offset,
                                                 scope_context_project=project_id or "", _connection=db)
                    for record in page:
                        tokens = _scoped_tokens(_searchable_text(record))
                        documents.append((record, Counter(tokens), len(tokens)))
                    offset += len(page)
                    if len(page) < page_size:
                        break
                db.commit()
            except Exception:
                db.rollback()
                raise
        truncated = offset >= self.max_candidates
        scanned = len(documents)
        terms = tuple(sorted(set(words)))
        df = {term: sum(1 for _, tf, _ in documents if term in tf) for term in terms}
        average_length = sum(length for _, _, length in documents) / max(scanned, 1)
        scores = []
        for record, counts, length in documents:
            score = 0.0
            for word in words:
                freq = counts.get(word, 0)
                if not freq:
                    continue
                idf = math.log1p((scanned - df[word] + 0.5) / (df[word] + 0.5))
                denominator = freq + 1.2 * (0.25 + 0.75 * length / max(average_length, 1))
                score += idf * freq * 2.2 / denominator
            if score > 0:
                scores.append((score, record))
        scores.sort(key=lambda r: (-r[0], r[1].record_id, -r[1].revision))
        hits = tuple(RetrievalHit(
            record_id=record.record_id, revision=record.revision,
            kind=record.draft.kind, scope=record.draft.scope.value,
            score=round(score, 9), source_ids=record.draft.source_ids,
            evidence_count=len(record.draft.evidence), record_digest=record.digest,
        ) for score, record in scores[:k])
        # Watermark includes only authorized candidates; unlike a database-wide
        # MAX(revision), it does not reveal hidden project modifications.
        watermark = digest_json([(r.record_id, r.revision, r.digest)
                                 for r, _, _ in documents])
        return RetrievalResult(
            query_digest=digest_json({"query": query, "kind": kind,
                                     "project_id": project_id,
                                     "as_known": as_known, "as_valid": as_valid}),
            hits=hits, scanned=scanned, truncated=truncated,
            corpus_watermark=watermark, as_known=as_known,
            as_valid=as_valid, project_id=project_id,
        )

class Sufficiency(str, Enum):
    SUFFICIENT = "sufficient_for_requested_evidence"
    NOT_FOUND = "not_found"
    INCOMPLETE = "incomplete"
    UNVERIFIED = "unverified"
    CONFLICTED = "conflicted"
    INAPPLICABLE = "inapplicable"


@dataclass(frozen=True, slots=True)
class SufficiencyReport:
    state: Sufficiency
    considered_record_ids: tuple[str, ...]
    missing: tuple[str, ...]
    cautions: tuple[str, ...]
    request_context: tuple[str, ...] = ()


class KnowledgeSufficiencyService:
    REQUIRED_DIMENSIONS = frozenset({"authenticity", "edition", "integrity",
                                      "currency", "applicability", "interpretation"})

    def __init__(self, repository: KnowledgeRepository):
        self.repository = repository

    def inspect(self, results: RetrievalResult, access: AccessContext | None, *,
                required_dimensions: tuple[str, ...] = (),
                require_source: bool = True,
                strict_lifecycle: bool = True,
                conflict: bool = False) -> SufficiencyReport:
        if set(required_dimensions) - self.REQUIRED_DIMENSIONS:
            raise ValidationFailure("Unknown evidence dimension")
        if conflict:
            return SufficiencyReport(Sufficiency.CONFLICTED, (),
                                     ("conflicting_authority",),
                                     ("A human reviewer must decide applicable precedence",))
        if not results.hits:
            return SufficiencyReport(Sufficiency.NOT_FOUND, (), ("no_authorized_candidates",),
                                     ("No answer should be synthesized from absent evidence",))
        missing: set[str] = set()
        cautions: set[str] = set()
        seen = []
        for hit in results.hits:
            record = self.repository.get(hit.record_id, access, as_known=results.as_known,
                                         as_valid=results.as_valid)
            if record is None or record.revision != hit.revision:
                missing.add("retrieval_revision_not_reproducible")
                continue
            seen.append(record.record_id)
            if require_source and not record.draft.source_ids:
                missing.add("source_evidence")
            if strict_lifecycle and record.draft.status not in {Lifecycle.ACTIVE, Lifecycle.REVIEWED}:
                missing.add("independent_review_or_activation")
            if record.draft.kind in {"standard_edition", "requirement"}:
                verification: dict[str, Any] = record.draft.payload.get("verification", {})
                if not isinstance(verification, dict):
                    verification = {}
                for dimension in required_dimensions:
                    entry = verification.get(dimension)
                    if not isinstance(entry, dict) or entry.get("verdict") != "supported":
                        missing.add(f"verified:{dimension}")
                licence = record.draft.payload.get("licence", "unknown")
                if licence in {"metadata_only", "unknown", None}:
                    cautions.add("source_content_or_AI_processing_rights_not_established")
        if results.truncated:
            missing.add("unscanned_candidates")
        if missing:
            if any(m.startswith("verified:") for m in missing):
                state = Sufficiency.UNVERIFIED
            else:
                state = Sufficiency.INCOMPLETE
        else:
            state = Sufficiency.SUFFICIENT
        cautions.add("Evidence sufficiency is not legal applicability or compliance certification")
        return SufficiencyReport(state, tuple(seen), tuple(sorted(missing)), tuple(sorted(cautions)),
                                 tuple(sorted(m.split(":")[-1] for m in missing
                                              if m.startswith("verified:"))))

@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkCase:
    case_id: str
    query: str
    principal_key: str
    relevant_ids: tuple[str, ...]
    forbidden_ids: tuple[str, ...] = ()
    project_id: str | None = None
    kind: str | None = None
    language: str = "und"
    domain: str = "general"
    as_known: str | None = None
    as_valid: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkResult:
    evaluated: int
    recall_at_k: float
    precision_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    p95_latency_ms: float
    leak_count: int
    truncated_count: int
    by_group: dict[str, dict[str, float]]
    case_results: tuple[dict[str, Any], ...]


def load_cases(path: str | Path, *, max_cases: int = 5000) -> list[RetrievalBenchmarkCase]:
    path = Path(path)
    result = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            if len(line) > 16384:
                raise ValidationFailure(f"Oversized benchmark case at line {lineno}")
            try:
                row = json.loads(line)
                assert isinstance(row, dict)
                result.append(RetrievalBenchmarkCase(**row))
            except (TypeError, AssertionError, ValueError) as exc:
                raise ValidationFailure(f"Malformed benchmark fixture at line {lineno}") from exc
            if len(result) > max_cases:
                raise ValidationFailure("Too many benchmark cases")
    if not result:
        raise ValidationFailure("No benchmark cases")
    return result


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def run_benchmark(retriever: ScopedLexicalRetriever,
                  cases: Sequence[RetrievalBenchmarkCase],
                  identities: Mapping[str, AccessContext | None], *,
                  k: int = 10) -> RetrievalBenchmarkResult:
    if not cases or not 1 <= k <= 100:
        raise ValidationFailure("Nonempty cases and k=1..100 required")
    results: list[dict[str, Any]] = []
    latency: list[float] = []
    groups: dict[str, list[float]] = {}
    leaks = 0
    truncated = 0
    for case in cases:
        validate_id(case.case_id, "case_id")
        if case.principal_key not in identities:
            raise ValidationFailure("Unknown benchmark identity; never construct permissions from fixture rows")
        if not case.relevant_ids:
            raise ValidationFailure("Each case must have at least one relevant ID")
        start = time.perf_counter()
        found = retriever.search(case.query, identities[case.principal_key],
                                 project_id=case.project_id, kind=case.kind,
                                 as_known=case.as_known, as_valid=case.as_valid, k=k)
        elapsed = (time.perf_counter() - start) * 1000
        latency.append(elapsed)
        names = [hit.record_id for hit in found.hits]
        relevant = set(case.relevant_ids)
        forbidden = set(case.forbidden_ids)
        leak = bool(forbidden.intersection(names))
        leaks += int(leak)
        truncated += int(found.truncated)
        correct = len(relevant.intersection(names))
        recall = correct / len(relevant)
        precision = correct / k
        reciprocal = next((1.0 / (i + 1) for i, rid in enumerate(names) if rid in relevant), 0.0)
        dcg = sum(1.0 / math.log2(i + 2) for i, rid in enumerate(names) if rid in relevant)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        ndcg = dcg / ideal if ideal else 0.0
        result = {"case_id": case.case_id, "recall_at_k": recall,
                  "precision_at_k": precision, "mrr_at_k": reciprocal,
                  "ndcg_at_k": ndcg, "latency_ms": elapsed,
                  "leak": leak, "truncated": found.truncated}
        results.append(result)
        for group in (f"language:{case.language}", f"domain:{case.domain}"):
            groups.setdefault(group, []).append(recall)
    mean = lambda field: sum(row[field] for row in results) / len(results)
    return RetrievalBenchmarkResult(
        evaluated=len(results), recall_at_k=mean("recall_at_k"),
        precision_at_k=mean("precision_at_k"), mrr_at_k=mean("mrr_at_k"),
        ndcg_at_k=mean("ndcg_at_k"), p95_latency_ms=_percentile_95(latency),
        leak_count=leaks, truncated_count=truncated,
        by_group={key: {"count": len(data), "recall_at_k": sum(data)/len(data)}
                  for key, data in sorted(groups.items())},
        case_results=tuple(results))

@dataclass(frozen=True, slots=True)
class DispatchReport:
    leased: int
    acknowledged: int
    failed: int
    failure_codes: tuple[str, ...]


class OutboxDispatcher:
    """Receivers MUST deduplicate by event_id and enforce own authorization."""

    def __init__(self, outbox: KnowledgeOutbox,
                 publish: Callable[[OutboxDelivery], None],
                 on_failure: Callable[[str, str], None] | None = None):
        self.outbox = outbox
        self.publish = publish
        self.on_failure = on_failure  # Optional adapter for the existing Handler Agent.

    def run_once(self, *, batch_size: int = 25) -> DispatchReport:
        events = self.outbox.lease(batch_size=batch_size)
        acked = 0
        failures = []
        for event in events:
            try:
                self.publish(event)
                if self.outbox.ack(event.event_id, event.lease_token):
                    acked += 1
                else:
                    failures.append("lease_lost")
            except Exception:
                # No raw exception body or sensitive event payload in telemetry.
                code = "consumer_exception"
                self.outbox.fail(event.event_id, event.lease_token, error_code=code)
                failures.append(code)
                if self.on_failure:
                    try:
                        self.on_failure(event.event_id, code)
                    except Exception:
                        pass  # Handler failure cannot suppress outbox retry state.
        return DispatchReport(len(events), acked, len(failures), tuple(failures))



@dataclass(frozen=True, slots=True)
class OperationalHealth:
    status: str
    sqlite_integrity: str
    outbox_pending: int
    outbox_dead_letter: int
    stale_assessments: int
    database_bytes: int
    free_disk_bytes: int
    inspected_at: str
    warnings: tuple[str, ...]


class AssuranceDiagnostics:
    """Performs LIVE storage checks; does not infer health from successful imports."""
    def __init__(self, repository: KnowledgeRepository, outbox: KnowledgeOutbox):
        self.repository, self.outbox = repository, outbox

    def inspect(self, *, dead_letter_limit: int = 0,
                min_free_bytes: int = 20_000_000) -> OperationalHealth:
        warnings = []
        with self.repository._connect() as db:
            integrity = db.execute("PRAGMA quick_check").fetchone()[0]
            stale_count = db.execute("""SELECT COUNT(DISTINCT assessment_id)
                FROM assessment_stale_events""").fetchone()[0]
        counts = self.outbox.status()
        disk = shutil.disk_usage(self.repository.path.parent)
        if integrity != "ok":
            warnings.append("sqlite_integrity_error")
        if counts["dead_letter"] > dead_letter_limit:
            warnings.append("outbox_dead_letter")
        if disk.free < min_free_bytes:
            warnings.append("low_disk_space")
        if counts["pending"] > 10000:
            warnings.append("outbox_backlog")
        status = "unhealthy" if integrity != "ok" else ("degraded" if warnings else "healthy")
        if warnings:
            logger.warning("Knowledge assurance storage diagnostics: %s", ",".join(warnings))
        return OperationalHealth(status, integrity, counts["pending"],
                                 counts["dead_letter"], stale_count,
                                 self.repository.path.stat().st_size, disk.free,
                                 utc_now(), tuple(warnings))

# Compatibility name only; all features are owned by the original gateway.
EnhancedKnowledgeGateway = KnowledgeAssuranceGateway
