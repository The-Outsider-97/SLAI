"""Unified, resumable SLAI-assisted LANTRA corpus collector.

Run from the SLAI repository root with::

    py -m slai_scraper

The collector writes only TITLE + blank line + CONTENT into data/library/*.txt.
Retrieval metadata, source identifiers, hashes, adaptation statistics, and resume
state live outside the training files.

SLAI participation is intentionally concrete rather than ceremonial:
* PlanningAgent may reorder the next source/topic work batch.
* ReasoningAgent resolves the document's broad topical domain from evidence.
* QualityAgent gates structurally valid candidate documents.
* KnowledgeAgent indexes accepted documents for shared collection context.

Network retrieval follows the same stdlib/API pattern used by the current SLAI
topic collectors; BrowserAgent is not instantiated merely to duplicate working
HTTP retrieval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import signal
import time
import unicodedata

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.training.corpus_dedup import (
    NearDuplicateIndex,
    atomic_write_json,
    collision_safe_path,
    meaningful_words,
    normalized_sha256,
    sanitize_windows_filename,
)
from src.utils.configuration import bind_config


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "src" / "training" / "configs" / "lantra_pipeline.yaml"
_CONFIG = bind_config(DEFAULT_CONFIG)
LOGGER = logging.getLogger("slai_scraper")

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKISOURCE_API = "https://en.wikisource.org/w/api.php"
GUTENDEX_API = "https://gutendex.com/books/"

TOPICS: tuple[str, ...] = (
    "world history", "archaeology", "geography", "law", "political science",
    "economics", "sociology", "psychology", "philosophy", "religion",
    "linguistics", "literature", "art history", "architecture", "education",
    "mathematics", "physics", "astronomy", "chemistry", "biology", "ecology",
    "geology", "medicine", "computer science", "artificial intelligence",
    "engineering", "technology", "agriculture", "botany", "zoology",
)

TOPIC_SIGNALS: dict[str, tuple[str, ...]] = {
    "history": ("history", "historical", "ancient", "medieval", "empire", "war", "century"),
    "society": ("society", "social", "culture", "religion", "psychology", "education", "language"),
    "politics_law": ("politics", "government", "law", "legal", "constitution", "policy", "state"),
    "economics": ("economy", "economic", "trade", "finance", "market", "industry"),
    "humanities": ("philosophy", "literature", "art", "architecture", "humanities", "poetry", "novel"),
    "natural_sciences": ("physics", "astronomy", "chemistry", "biology", "ecology", "geology", "botany", "zoology"),
    "technology_engineering": ("computer", "technology", "engineering", "software", "machine", "artificial intelligence"),
    "geography": ("geography", "country", "region", "river", "mountain", "climate", "population"),
}

SOURCE_ORDER = ("wikipedia", "wikisource", "gutenberg")
_STOP = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _pipeline_config(path: Path) -> dict[str, Any]:
    loaded = _CONFIG.load(path)
    section = loaded.get("lantra_pipeline", {})
    if not isinstance(section, Mapping):
        raise ValueError("lantra_pipeline configuration must be a mapping.")
    return dict(section)


def _historical_roots(config: Mapping[str, Any]) -> list[Path]:
    cleaner = config.get("cleaner", {})
    if not isinstance(cleaner, Mapping):
        return []
    return [_resolve(str(value)) for value in cleaner.get("historical_corpora", ())]


def discover_txt(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for root in paths:
        if not root.exists():
            continue
        iterator = [root] if root.is_file() else root.rglob("*.txt")
        for path in iterator:
            try:
                if not path.is_file():
                    continue
                key = os.path.normcase(str(path.resolve()))
            except OSError:
                continue
            if key in seen:
                continue
            seen.add(key)
            result.append(path)
    return sorted(result, key=lambda item: os.path.normcase(str(item)))


def clean_visible_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^\s*(?:Jump to navigation|Jump to search)\s*$", "", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def title_from_text(title: str) -> str:
    value = clean_visible_text(title).split("\n", 1)[0].strip()
    if len(value) < 3 or len(value) > 220:
        raise ValueError("Source did not provide a usable title.")
    return value


def quality_score(text: str) -> float:
    words = meaningful_words(text)
    if not words:
        return 0.0
    unique_ratio = len(set(words)) / max(1, len(words))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    linkish = sum(1 for line in lines if line.startswith(("http://", "https://", "*", "|")))
    link_ratio = linkish / max(1, len(lines))
    average_word_length = sum(len(word) for word in words) / len(words)
    repetition = 1.0 - min(1.0, unique_ratio * 4.0)
    score = (
        0.35 * min(1.0, len(words) / 500.0)
        + 0.25 * min(1.0, unique_ratio / 0.22)
        + 0.20 * min(1.0, average_word_length / 5.0)
        + 0.20 * (1.0 - min(1.0, link_ratio * 4.0))
        - 0.20 * repetition
    )
    return max(0.0, min(1.0, score))


def looks_english(text: str) -> bool:
    words = meaningful_words(text)
    if len(words) < 20:
        return False
    common = {
        "the", "of", "and", "to", "in", "a", "is", "that", "for", "as",
        "with", "was", "on", "by", "from", "are", "this", "be", "or", "an",
    }
    sample = words[:1500]
    common_hits = sum(1 for word in sample if word in common)
    latin = sum(1 for char in text[:20000] if char.isascii() and char.isalpha())
    alphabetic = sum(1 for char in text[:20000] if char.isalpha())
    return common_hits >= max(3, len(sample) // 80) and latin / max(1, alphabetic) >= 0.80


@dataclass(frozen=True)
class Candidate:
    source: str
    source_key: str
    title: str
    content: str
    topic_hint: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceJob:
    source: str
    topic: str
    cursor: int = 0

    @property
    def key(self) -> str:
        return f"{self.source}:{self.topic}:{self.cursor}"


class RequestLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.minimum_interval = 60.0 / max(1, int(requests_per_minute))
        self.last_request = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_request
        delay = self.minimum_interval - elapsed
        if delay > 0:
            time.sleep(delay)
        self.last_request = time.monotonic()


class HttpClient:
    def __init__(
        self,
        *,
        timeout: float,
        retry_limit: int,
        backoff: float,
        requests_per_minute: int,
        user_agent: str,
    ) -> None:
        self.timeout = float(timeout)
        self.retry_limit = max(0, int(retry_limit))
        self.backoff = max(0.1, float(backoff))
        self.user_agent = str(user_agent)
        self.limiter = RequestLimiter(requests_per_minute)

    def get_bytes(self, url: str) -> bytes:
        last_error: BaseException | None = None
        for attempt in range(self.retry_limit + 1):
            self.limiter.wait()
            request = Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/json,text/plain;q=0.9,*/*;q=0.5",
                },
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return response.read()
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if isinstance(exc, HTTPError) and exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                    break
                if attempt < self.retry_limit:
                    time.sleep(self.backoff * (2 ** attempt))
        assert last_error is not None
        raise last_error

    def get_json(self, base: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        url = base + ("&" if "?" in base else "?") + urlencode(
            {key: value for key, value in params.items() if value is not None}
        )
        payload = json.loads(self.get_bytes(url).decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Expected a JSON object response.")
        return payload


def _mediawiki_candidates(
    client: HttpClient,
    *,
    api: str,
    source: str,
    topic: str,
    cursor: int,
    limit: int = 8,
) -> list[Candidate]:
    search = client.get_json(
        api,
        {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "list": "search",
            "srsearch": topic,
            "srlimit": limit,
            "sroffset": max(0, int(cursor)),
            "srnamespace": 0,
        },
    )
    query = search.get("query", {})
    rows = query.get("search", []) if isinstance(query, Mapping) else []
    pageids = [str(row.get("pageid")) for row in rows if isinstance(row, Mapping) and row.get("pageid")]
    if not pageids:
        return []

    pages_payload = client.get_json(
        api,
        {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "pageids": "|".join(pageids),
            "prop": "extracts|info",
            "explaintext": 1,
            "exsectionformat": "plain",
            "inprop": "url",
        },
    )
    page_query = pages_payload.get("query", {})
    pages = page_query.get("pages", []) if isinstance(page_query, Mapping) else []
    result: list[Candidate] = []
    for page in pages:
        if not isinstance(page, Mapping):
            continue
        title = str(page.get("title") or "").strip()
        extract = clean_visible_text(str(page.get("extract") or ""))
        pageid = page.get("pageid")
        if title and extract and pageid is not None:
            result.append(
                Candidate(
                    source=source,
                    source_key=f"{source}:page:{pageid}",
                    title=title,
                    content=extract,
                    topic_hint=topic,
                    metadata={"pageid": str(pageid)},
                )
            )
    return result


def _select_gutenberg_text_url(formats: Mapping[str, Any]) -> str | None:
    preferred: list[tuple[int, str]] = []
    for mime, value in formats.items():
        if not isinstance(value, str) or not value.startswith(("http://", "https://")):
            continue
        key = str(mime).casefold()
        if "text/plain" not in key:
            continue
        priority = 0 if "utf-8" in key else 1
        preferred.append((priority, value))
    if not preferred:
        return None
    return sorted(preferred)[0][1]


def _gutenberg_candidates(
    client: HttpClient,
    *,
    topic: str,
    cursor: int,
    limit: int = 5,
) -> list[Candidate]:
    page = max(1, int(cursor) // 32 + 1)
    payload = client.get_json(
        GUTENDEX_API,
        {"search": topic, "languages": "en", "page": page},
    )
    rows = payload.get("results", [])
    if not isinstance(rows, Sequence):
        return []
    result: list[Candidate] = []
    for row in rows[:limit]:
        if not isinstance(row, Mapping):
            continue
        book_id = row.get("id")
        title = str(row.get("title") or "").strip()
        formats = row.get("formats", {})
        if book_id is None or not title or not isinstance(formats, Mapping):
            continue
        text_url = _select_gutenberg_text_url(formats)
        if not text_url:
            continue
        body = client.get_bytes(text_url).decode("utf-8", errors="replace")
        body = clean_visible_text(body)
        if body:
            result.append(
                Candidate(
                    source="gutenberg",
                    source_key=f"gutenberg:book:{book_id}",
                    title=title,
                    content=body,
                    topic_hint=topic,
                    metadata={"book_id": str(book_id)},
                )
            )
    return result


class SLAITeam:
    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.memory = self.factory = None
        self.planning = self.reasoning = self.quality = self.knowledge = None
        self.Task = self.TaskType = self.ResourceProfile = None
        if not self.enabled:
            return
        from src.agents.agent_factory import AgentFactory
        from src.agents.collaborative.shared_memory import SharedMemory
        from src.agents.planning.planning_types import ResourceProfile, Task, TaskType

        self.Task, self.TaskType, self.ResourceProfile = Task, TaskType, ResourceProfile
        self.memory = SharedMemory()
        self.factory = AgentFactory()
        self.planning = self.factory.create(
            "planning",
            shared_memory=self.memory,
            config={"plan_history_window": 64, "execution_history_window": 64},
        )
        self.reasoning = self.factory.create("reasoning", shared_memory=self.memory)
        self.quality = self.factory.create(
            "quality",
            shared_memory=self.memory,
            config={"enabled": True, "auto_route_via_workflow": False},
        )
        self.knowledge = self.factory.create(
            "knowledge",
            shared_memory=self.memory,
            config={
                "source": "slai_scraper",
                "directory_path": "",
                "retrieval_mode": "tfidf",
                "bias_detection_enabled": False,
                "use_ontology_expansion": False,
            },
        )
        for agent, methods in (
            (self.planning, ("generate_plan",)),
            (self.reasoning, ("add_fact", "add_rule", "forward_chaining", "forget_by_subject")),
            (self.quality, ("evaluate_batch",)),
            (self.knowledge, ("add_document",)),
        ):
            missing = [name for name in methods if not callable(getattr(agent, name, None))]
            if missing:
                raise RuntimeError(f"{type(agent).__name__} lacks required methods: {missing}")

        def topic_rule(facts: Mapping[Any, float]) -> Mapping[Any, float]:
            inferred: dict[Any, float] = {}
            for fact, confidence in list(facts.items()):
                if not isinstance(fact, tuple) or len(fact) != 3:
                    continue
                subject, predicate, domain = fact
                if subject != "scraper_candidate" or predicate != "topic_signal":
                    continue
                inferred[("scraper_candidate", "selected_topic", domain)] = max(
                    float(confidence),
                    inferred.get(("scraper_candidate", "selected_topic", domain), 0.0),
                )
            return inferred

        self.reasoning.add_rule(topic_rule, rule_name="slai_scraper_topic_evidence", weight=1.0)
        LOGGER.info("AGENTS | Planning + Reasoning + Quality + Knowledge active")

    def order_jobs(self, jobs: Sequence[SourceJob]) -> list[SourceJob]:
        if not self.enabled or len(jobs) < 2:
            return list(jobs)
        assert self.Task is not None and self.TaskType is not None and self.ResourceProfile is not None
        try:
            deadline = time.time() + 3600
            tasks = [
                self.Task(
                    name=f"scrape_{index}",
                    id=f"scrape_{index}",
                    task_type=self.TaskType.PRIMITIVE,
                    preconditions=[],
                    effects=[],
                    resource_requirements=self.ResourceProfile(gpu=0, ram=0.02),
                    duration=10,
                    deadline=deadline,
                    dependencies=[],
                )
                for index, _ in enumerate(jobs)
            ]
            goal = self.Task(
                name="slai_scraper_collection_batch",
                task_type=self.TaskType.ABSTRACT,
                methods=[tasks],
                resource_requirements=self.ResourceProfile(gpu=0, ram=0),
                duration=10 * len(tasks),
                deadline=deadline,
            )
            plan = self.planning.generate_plan(goal)
            if not plan:
                return list(jobs)
            mapping = {f"scrape_{index}": job for index, job in enumerate(jobs)}
            ordered = [mapping[item.id] for item in plan if getattr(item, "id", None) in mapping]
            return ordered if len(ordered) == len(jobs) else list(jobs)
        except Exception as exc:
            LOGGER.warning("PlanningAgent advisory ordering failed: %s", exc)
            return list(jobs)

    def classify_topic(self, title: str, text: str, fallback: str) -> str:
        combined = f"{title}\n{text[:12000]}".casefold()
        candidates: list[tuple[str, float]] = []
        for domain, signals in TOPIC_SIGNALS.items():
            hits = sum(combined.count(signal.casefold()) for signal in signals)
            if hits:
                candidates.append((domain, min(1.0, 0.30 + math.log1p(hits) / 5.0)))
        if not candidates:
            return fallback
        candidates.sort(key=lambda item: (-item[1], item[0]))
        if not self.enabled:
            return candidates[0][0]

        subject = "scraper_candidate"
        self.reasoning.forget_by_subject(subject)
        try:
            for domain, confidence in candidates[:4]:
                self.reasoning.add_fact((subject, "topic_signal", domain), confidence=confidence, publish=False)
            self.reasoning.forward_chaining(max_iterations=2)
            scores = {
                domain: float(self.reasoning.knowledge_base.get((subject, "selected_topic", domain), 0.0))
                for domain, _ in candidates
            }
            chosen = max(scores, key=scores.get)
            return chosen if scores[chosen] > 0 else candidates[0][0]
        finally:
            self.reasoning.forget_by_subject(subject)

    def quality_accepts(
        self,
        *,
        candidate: Candidate,
        topic: str,
        deterministic_score: float,
    ) -> tuple[bool, Mapping[str, Any]]:
        if not self.enabled:
            return True, {"verdict": "accepted_without_agents"}
        record = {
            "id": candidate.source_key,
            "title": candidate.title,
            "text": candidate.content,
            "source_type": candidate.source,
            "topic": topic,
            "word_count": len(meaningful_words(candidate.content)),
            "deterministic_quality_score": deterministic_score,
        }
        schema = {
            "required_fields": ["id", "title", "text", "source_type", "topic", "word_count"],
            "fields": {
                "id": {"type": "str", "required": True},
                "title": {"type": "str", "required": True},
                "text": {"type": "str", "required": True},
                "source_type": {"type": "str", "required": True},
                "topic": {"type": "str", "required": True},
                "word_count": {"type": "int", "required": True},
            },
        }
        result = self.quality.evaluate_batch(
            [record],
            dataset_id="lantra_general_corpus",
            source_id=candidate.source,
            batch_id=hashlib.sha256(candidate.source_key.encode()).hexdigest()[:16],
            schema=schema,
            context={"use_case": "language_model_training_corpus"},
        )
        verdict = str(result.get("verdict", result.get("decision", "warn"))).casefold()
        blocked = verdict in {"reject", "rejected", "block", "blocked", "fail", "failed"}
        return not blocked, result

    def index_accepted(self, candidate: Candidate, topic: str) -> None:
        if self.enabled:
            self.knowledge.add_document(
                candidate.content,
                doc_id=candidate.source_key,
                metadata={"title": candidate.title, "topic": topic, "source": candidate.source},
            )

    def close(self) -> None:
        if self.factory is not None:
            release = getattr(self.factory, "release", None)
            active = getattr(self.factory, "get_active_agent_types", None)
            if callable(release) and callable(active):
                try:
                    for name in reversed(list(active())):
                        release(name)
                except Exception as exc:
                    LOGGER.debug("Agent shutdown warning: %s", exc)
        if self.memory is not None:
            close = getattr(self.memory, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    LOGGER.debug("SharedMemory shutdown warning: %s", exc)


def _new_state(target_bytes: int) -> dict[str, Any]:
    return {
        "schema": "slai.lantra.scraper-state.v1",
        "campaign_started_at": utc_now(),
        "updated_at": utc_now(),
        "target_bytes": int(target_bytes),
        "accepted_documents": 0,
        "accepted_bytes": 0,
        "accepted_words": 0,
        "rejected_documents": 0,
        "duplicates": 0,
        "recoverable_failures": 0,
        "processed_source_keys": [],
        "source_stats": {},
        "topic_stats": {},
        "cursors": {},
        "distribution": {},
    }


def _load_state(path: Path, target_bytes: int, *, reset: bool) -> dict[str, Any]:
    if reset or not path.is_file():
        state = _new_state(target_bytes)
        atomic_write_json(path, state)
        return state
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != "slai.lantra.scraper-state.v1":
        raise ValueError(f"Invalid scraper state: {path}")
    state = dict(payload)
    state["target_bytes"] = int(target_bytes)
    return state


def _save_state(path: Path, state: MutableMapping[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_write_json(path, dict(state))


def _stat_bucket(state: MutableMapping[str, Any], key: str, group: str) -> MutableMapping[str, Any]:
    collection = state.setdefault(group, {})
    if not isinstance(collection, MutableMapping):
        collection = {}
        state[group] = collection
    bucket = collection.setdefault(
        key,
        {"attempts": 0, "accepted": 0, "accepted_bytes": 0, "duplicates": 0, "rejected": 0, "errors": 0},
    )
    return bucket


def _productivity(state: Mapping[str, Any], source: str, topic: str) -> float:
    source_stats = state.get("source_stats", {})
    topic_stats = state.get("topic_stats", {})
    source_bucket = source_stats.get(source, {}) if isinstance(source_stats, Mapping) else {}
    topic_bucket = topic_stats.get(topic, {}) if isinstance(topic_stats, Mapping) else {}
    attempts = float(source_bucket.get("attempts", 0)) + 1.0
    accepted = float(source_bucket.get("accepted", 0))
    bytes_per = float(source_bucket.get("accepted_bytes", 0)) / attempts
    errors = float(source_bucket.get("errors", 0))
    duplicates = float(source_bucket.get("duplicates", 0))
    topic_attempts = float(topic_bucket.get("attempts", 0)) + 1.0
    topic_acceptance = (float(topic_bucket.get("accepted", 0)) + 1.0) / topic_attempts
    exploitation = math.log1p(bytes_per) * (accepted + 1.0) / attempts * topic_acceptance
    penalty = 1.0 + (errors + duplicates) / attempts
    exploration = 1.0 / math.sqrt(attempts)
    return exploitation / penalty + exploration


def _choose_jobs(state: Mapping[str, Any], enabled_sources: Sequence[str], count: int = 6) -> list[SourceJob]:
    cursors = state.get("cursors", {})
    candidates: list[tuple[float, SourceJob]] = []
    for source in enabled_sources:
        for topic in TOPICS:
            key = f"{source}:{topic}"
            cursor = int(cursors.get(key, 0)) if isinstance(cursors, Mapping) else 0
            score = _productivity(state, source, topic)
            jitter = int.from_bytes(hashlib.sha256(f"{key}:{cursor}".encode()).digest()[:2], "big") / 65535.0
            candidates.append((score + 0.05 * jitter, SourceJob(source, topic, cursor)))
    candidates.sort(key=lambda item: (-item[0], item[1].source, item[1].topic))
    return [job for _, job in candidates[: max(1, int(count))]]


def _fetch_job(client: HttpClient, job: SourceJob) -> list[Candidate]:
    if job.source == "wikipedia":
        return _mediawiki_candidates(
            client, api=WIKIPEDIA_API, source="wikipedia", topic=job.topic, cursor=job.cursor
        )
    if job.source == "wikisource":
        return _mediawiki_candidates(
            client, api=WIKISOURCE_API, source="wikisource", topic=job.topic, cursor=job.cursor
        )
    if job.source == "gutenberg":
        return _gutenberg_candidates(client, topic=job.topic, cursor=job.cursor)
    raise ValueError(f"Unsupported scraper source: {job.source}")


def _append_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_training_document(path: Path, title: str, content: str) -> int:
    payload = f"{title.strip()}\n\n{content.strip()}\n".encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return len(payload)


def _index_existing_corpus(index: NearDuplicateIndex, config: Mapping[str, Any], library: Path) -> None:
    historical = discover_txt(_historical_roots(config))
    library_files = discover_txt([library])
    LOGGER.info("DEDUPE INDEX | historical=%d | library=%d", len(historical), len(library_files))
    for path in historical:
        try:
            index.add_path(path, source_kind="historical")
        except (OSError, UnicodeError, ValueError) as exc:
            LOGGER.warning("INDEX SKIP | %s | %s", path, exc)
    for path in library_files:
        try:
            index.add_path(path, source_kind="library")
        except (OSError, UnicodeError, ValueError) as exc:
            LOGGER.warning("INDEX SKIP | %s | %s", path, exc)


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    config = _pipeline_config(Path(args.config))
    scraper_cfg = config.get("scraper", {})
    if not isinstance(scraper_cfg, Mapping):
        raise ValueError("lantra_pipeline.scraper must be a mapping.")

    target_bytes = int(args.target_bytes or scraper_cfg.get("target_bytes", 500_000_000))
    minimum_words = int(config.get("minimum_document_words", 50))
    threshold = float(config.get("duplicate_similarity_threshold", 0.85))
    library = _resolve(str(scraper_cfg.get("library_path", "data/library")))
    state_path = _resolve(str(scraper_cfg.get("state_path", "data/processed/lantra/scraper/slai_scraper_state.json")))
    manifest_path = _resolve(str(scraper_cfg.get("manifest_path", "data/processed/lantra/scraper/accepted_manifest.jsonl")))
    index_path = _resolve(str(scraper_cfg.get("dedup_index_path", "data/processed/lantra/state/corpus_dedup.sqlite3")))
    quality_min = float(scraper_cfg.get("quality_minimum_score", 0.55))

    sources_cfg = scraper_cfg.get("sources", {})
    enabled_sources = [
        source for source in SOURCE_ORDER
        if not isinstance(sources_cfg, Mapping) or bool(sources_cfg.get(source, True))
    ]
    if not enabled_sources:
        raise ValueError("At least one scraper source must be enabled.")

    state = _load_state(state_path, target_bytes, reset=bool(args.reset_campaign))
    processed = {str(value) for value in state.get("processed_source_keys", ())}
    team = SLAITeam(enabled=args.agents == "team")
    client = HttpClient(
        timeout=float(scraper_cfg.get("request_timeout_seconds", 30)),
        retry_limit=int(scraper_cfg.get("retry_limit", 3)),
        backoff=float(scraper_cfg.get("retry_backoff_seconds", 2.0)),
        requests_per_minute=int(scraper_cfg.get("requests_per_minute", 40)),
        user_agent=str(scraper_cfg.get("user_agent", "SLAI-LANTRA-Corpus/2.3")),
    )
    consecutive_failures = 0
    max_consecutive_failures = int(scraper_cfg.get("max_consecutive_failures", 12))
    started = time.perf_counter()

    try:
        with NearDuplicateIndex(
            index_path,
            threshold=threshold,
            shingle_size=int(config.get("duplicate_shingle_size", 5)),
            permutations=int(config.get("duplicate_minhash_permutations", 64)),
            bands=int(config.get("duplicate_lsh_bands", 8)),
        ) as index:
            _index_existing_corpus(index, config, library)

            while int(state.get("accepted_bytes", 0)) < target_bytes:
                if _STOP:
                    raise KeyboardInterrupt

                jobs = team.order_jobs(_choose_jobs(state, enabled_sources))
                made_progress = False
                for job in jobs:
                    if int(state.get("accepted_bytes", 0)) >= target_bytes:
                        break
                    source_bucket = _stat_bucket(state, job.source, "source_stats")
                    topic_bucket = _stat_bucket(state, job.topic, "topic_stats")
                    source_bucket["attempts"] = int(source_bucket.get("attempts", 0)) + 1
                    topic_bucket["attempts"] = int(topic_bucket.get("attempts", 0)) + 1
                    cursor_key = f"{job.source}:{job.topic}"
                    cursors = state.setdefault("cursors", {})
                    if isinstance(cursors, MutableMapping):
                        cursors[cursor_key] = int(job.cursor) + 8

                    try:
                        candidates = _fetch_job(client, job)
                        consecutive_failures = 0
                    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                        consecutive_failures += 1
                        state["recoverable_failures"] = int(state.get("recoverable_failures", 0)) + 1
                        source_bucket["errors"] = int(source_bucket.get("errors", 0)) + 1
                        topic_bucket["errors"] = int(topic_bucket.get("errors", 0)) + 1
                        LOGGER.warning("FETCH | %s | %s | %s: %s", job.source, job.topic, type(exc).__name__, exc)
                        _save_state(state_path, state)
                        if consecutive_failures >= max_consecutive_failures:
                            raise RuntimeError(
                                f"Stopped after {consecutive_failures} consecutive recoverable source failures."
                            ) from exc
                        continue

                    for candidate in candidates:
                        if candidate.source_key in processed:
                            continue
                        processed.add(candidate.source_key)
                        state["processed_source_keys"] = sorted(processed)
                        words = meaningful_words(candidate.content)
                        reject_reason: str | None = None
                        if len(words) < minimum_words:
                            reject_reason = "minimum_words"
                        elif not looks_english(candidate.content):
                            reject_reason = "english_language_gate"
                        deterministic_quality = quality_score(candidate.content)
                        if reject_reason is None and deterministic_quality < quality_min:
                            reject_reason = "deterministic_quality"

                        if reject_reason is not None:
                            state["rejected_documents"] = int(state.get("rejected_documents", 0)) + 1
                            source_bucket["rejected"] = int(source_bucket.get("rejected", 0)) + 1
                            topic_bucket["rejected"] = int(topic_bucket.get("rejected", 0)) + 1
                            _save_state(state_path, state)
                            continue

                        duplicate = index.find_duplicate(
                            candidate.content,
                            source_kinds=("historical", "library", "session"),
                        )
                        if duplicate is not None:
                            state["duplicates"] = int(state.get("duplicates", 0)) + 1
                            source_bucket["duplicates"] = int(source_bucket.get("duplicates", 0)) + 1
                            topic_bucket["duplicates"] = int(topic_bucket.get("duplicates", 0)) + 1
                            _save_state(state_path, state)
                            continue

                        topic = team.classify_topic(candidate.title, candidate.content, job.topic)
                        accepted_by_agent, agent_report = team.quality_accepts(
                            candidate=candidate,
                            topic=topic,
                            deterministic_score=deterministic_quality,
                        )
                        if not accepted_by_agent:
                            state["rejected_documents"] = int(state.get("rejected_documents", 0)) + 1
                            source_bucket["rejected"] = int(source_bucket.get("rejected", 0)) + 1
                            topic_bucket["rejected"] = int(topic_bucket.get("rejected", 0)) + 1
                            _save_state(state_path, state)
                            continue

                        try:
                            title = title_from_text(candidate.title)
                            filename = sanitize_windows_filename(title) + ".txt"
                        except ValueError:
                            state["rejected_documents"] = int(state.get("rejected_documents", 0)) + 1
                            _save_state(state_path, state)
                            continue

                        destination = collision_safe_path(library, filename)
                        accepted_bytes = _atomic_write_training_document(destination, title, candidate.content)
                        index.add_text(
                            path=destination,
                            text=candidate.content,
                            source_kind="session",
                            size=accepted_bytes,
                            mtime_ns=destination.stat().st_mtime_ns,
                        )
                        team.index_accepted(candidate, topic)
                        state["accepted_documents"] = int(state.get("accepted_documents", 0)) + 1
                        state["accepted_bytes"] = int(state.get("accepted_bytes", 0)) + accepted_bytes
                        state["accepted_words"] = int(state.get("accepted_words", 0)) + len(words)
                        source_bucket["accepted"] = int(source_bucket.get("accepted", 0)) + 1
                        source_bucket["accepted_bytes"] = int(source_bucket.get("accepted_bytes", 0)) + accepted_bytes
                        topic_bucket["accepted"] = int(topic_bucket.get("accepted", 0)) + 1
                        topic_bucket["accepted_bytes"] = int(topic_bucket.get("accepted_bytes", 0)) + accepted_bytes
                        distribution = state.setdefault("distribution", {})
                        if isinstance(distribution, MutableMapping):
                            key = f"{candidate.source}/{topic}"
                            distribution[key] = int(distribution.get(key, 0)) + 1

                        try:
                            relative_path = str(destination.relative_to(ROOT))
                        except ValueError:
                            relative_path = str(destination)
                        _append_manifest(
                            manifest_path,
                            {
                                "accepted_at": utc_now(),
                                "path": relative_path,
                                "source": candidate.source,
                                "source_key": candidate.source_key,
                                "topic": topic,
                                "normalized_sha256": normalized_sha256(candidate.content),
                                "words": len(words),
                                "bytes": accepted_bytes,
                                "deterministic_quality_score": deterministic_quality,
                                "agent_verdict": str(agent_report.get("verdict", agent_report.get("decision", "accepted"))),
                            },
                        )
                        _save_state(state_path, state)
                        made_progress = True
                        LOGGER.info(
                            "ACCEPT | %s | %s | words=%d | bytes=%d | total=%d/%d",
                            candidate.source, title, len(words), accepted_bytes,
                            int(state["accepted_bytes"]), target_bytes,
                        )
                        if int(state.get("accepted_bytes", 0)) >= target_bytes:
                            break

                if not made_progress:
                    time.sleep(0.5)

    finally:
        state["elapsed_seconds_last_run"] = time.perf_counter() - started
        _save_state(state_path, state)
        team.close()

    return state


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="slai_scraper",
        description="Grow data/library with deduplicated, English-language LANTRA training documents.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--target-bytes", type=int, default=None)
    parser.add_argument("--agents", choices=("team", "off"), default="team")
    parser.add_argument(
        "--reset-campaign",
        action="store_true",
        help="Start target accounting from zero without deleting existing corpus files.",
    )
    return parser.parse_args(argv)


def _install_signal_handlers() -> None:
    def stop_handler(signum: int, frame: Any) -> None:
        del signum, frame
        global _STOP
        _STOP = True
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, stop_handler)
        except (ValueError, OSError):
            continue


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args(argv)
    _install_signal_handlers()
    try:
        state = run(args)
    except KeyboardInterrupt:
        LOGGER.warning("Scraper interrupted; committed documents and state are preserved.")
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Scraper stopped: %s", exc)
        return 2

    distribution = state.get("distribution", {})
    top_distribution = []
    if isinstance(distribution, Mapping):
        top_distribution = sorted(distribution.items(), key=lambda item: (-int(item[1]), str(item[0])))[:12]
    print(
        json.dumps(
            {
                "accepted_documents": int(state.get("accepted_documents", 0)),
                "rejected_documents": int(state.get("rejected_documents", 0)),
                "duplicates": int(state.get("duplicates", 0)),
                "total_accepted_words": int(state.get("accepted_words", 0)),
                "total_accepted_bytes": int(state.get("accepted_bytes", 0)),
                "target_bytes": int(state.get("target_bytes", 0)),
                "major_source_topic_distribution": dict(top_distribution),
                "elapsed_seconds": float(state.get("elapsed_seconds_last_run", 0.0)),
                "recoverable_failures": int(state.get("recoverable_failures", 0)),
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
