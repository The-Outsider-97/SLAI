#!/usr/bin/env python3
"""Continuously collect American-history text for later LANTRA training.

Python 3.10+.

Design goals
------------
* Resume exactly from SQLite checkpoints after Ctrl+C, crashes, or restarts.
* Never export the same normalized document twice.
* Keep LANTRA-facing TXT files clean: title + source content only.
* Keep URLs, rights notices, external IDs, checksums, and QA bookkeeping OUT of
  training text. Minimal internal keys/hashes remain in SQLite only because they
  are required for restart-safe deduplication.
* Continue collecting until the user stops the process. With the default
  unlimited bounds, category discovery keeps expanding; if the discovered graph
  is temporarily exhausted, the collector idles and periodically refreshes
  category checkpoints for newly-added Wikipedia material.
* Integrate SLAI Knowledge, Quality, Reasoning, Planning, and Evaluation agents
  opportunistically. Quality and Knowledge have concrete low-overhead uses.
  Planning/Reasoning/Evaluation are used only when a compatible lightweight
  interface is exposed; otherwise they are logged and skipped rather than
  slowing or destabilizing collection.

Default output directory (always beside this script):
    ./american_history/

Important outputs:
    american_history.txt                     global deduplicated reading copy
    north_america.txt                        unique docs first assigned here
    middle_america.txt
    south_america.txt
    carribean.txt                            spelling retained for compatibility
    training_text/american_history/*.txt    one unique document per LANTRA file
    history.sqlite3                         resume/dedup/checkpoint state
    collection_summary.json                 counts only; no training content
    collector.log

Typical run:
    python scrape_american_history.py

Stop safely with Ctrl+C. Re-run the same command to continue.
Use --help for bounds, agent modes, and maintenance options.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import sys
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

LOG = logging.getLogger("american_history_collector")
VERSION = "2.0.2"
SCHEMA = "american-history-collector-v2"
WIKI = "https://en.wikipedia.org/w/api.php"
REGION_ORDER = ("north_america", "middle_america", "south_america", "carribean")

REGIONS = {
    "north_america": [
        "Canada", "the United States", "Greenland", "Bermuda",
    ],
    "middle_america": [
        "Mexico", "Central America", "Belize", "Costa Rica", "El Salvador",
        "Guatemala", "Honduras", "Nicaragua", "Panama",
    ],
    "south_america": [
        "South America", "Argentina", "Bolivia", "Brazil", "Chile", "Colombia",
        "Ecuador", "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay",
        "Venezuela", "French Guiana", "the Falkland Islands",
    ],
    "carribean": [
        "the Caribbean", "Antigua and Barbuda", "Aruba", "the Bahamas", "Barbados",
        "Bonaire", "Cuba", "Curacao", "Curaçao", "Dominica", "the Dominican Republic",
        "Grenada", "Haiti", "Jamaica", "Saint Kitts and Nevis", "Saint Lucia",
        "Saint Vincent and the Grenadines", "Trinidad and Tobago", "Puerto Rico",
        "Anguilla", "the British Virgin Islands", "the United States Virgin Islands",
        "the Cayman Islands", "Guadeloupe", "Martinique", "Montserrat", "Saint Martin",
        "Sint Maarten", "Saba", "Sint Eustatius", "Saint Barthélemy",
        "the Turks and Caicos Islands", "the Netherlands Antilles",
    ],
}

EXTRA = {
    "north_america": [
        "Pre-Columbian North America", "Indigenous peoples of North America",
        "Colonial history of the United States", "History of Canada",
        "History of the United States", "History of Greenland",
    ],
    "middle_america": [
        "Mesoamerica", "Mesoamerican chronology", "Maya civilization", "Aztecs",
        "Olmecs", "New Spain", "History of Central America", "History of Mexico",
    ],
    "south_america": [
        "Pre-Columbian South America", "Inca Empire", "Andean civilizations",
        "History of South America", "Spanish colonization of the Americas",
        "Portuguese colonization of the Americas",
    ],
    "carribean": [
        "Indigenous peoples of the Caribbean", "Taíno", "Kalinago",
        "Slavery in the British and French Caribbean", "Haitian Revolution",
        "History of the Caribbean", "Colonization of the Caribbean",
    ],
}

# Broad enough for historical material but rejects obvious maintenance/biography drift.
SKIP_CATEGORY = re.compile(
    r"\b(wikipedia|articles|pages|templates|stubs|births|deaths|living people|alumni|"
    r"award winners|sportspeople|filmographies|discographies|lists of people)\b",
    re.I,
)
HISTORY_SIGNAL = re.compile(
    r"\b(history|historical|pre[- ]?columbian|indigenous|native|colonial|colonization|"
    r"empire|kingdom|republic|revolution|war|conflict|treaty|independence|slavery|"
    r"abolition|civilization|culture|archaeolog|migration|settlement|exploration|"
    r"conquest|occupation|constitution|political|economic|social|military|maritime|"
    r"territor|province|statehood|dynasty|era|period|century|ancient|modern)\b",
    re.I,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # Preserve paragraph boundaries while normalizing noisy intra-line whitespace.
    return "\n\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def normalized_digest(text: str) -> str:
    # Internal-only dedup key. It is never written into LANTRA-facing text exports.
    normalized = " ".join(clean(text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class Skip(Exception):
    """Permanent record/task failure."""


class Deferred(Exception):
    """Transient source-wide failure; committed work is retained."""


class RestrictedRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Wikimedia may canonicalize protocol details, but never leave the API host.
        from urllib.parse import urlsplit
        old, new = urlsplit(req.full_url), urlsplit(newurl)
        if new.scheme != "https" or new.hostname != "en.wikipedia.org" or new.path != "/w/api.php":
            raise Skip("Redirect outside the selected Wikipedia API")
        if old.hostname != new.hostname:
            raise Skip("Cross-host redirect refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class APIClient:
    def __init__(self, delay=1.0, contact="", retries=5, timeout=45):
        self.delay = delay
        self.retries = retries
        self.timeout = timeout
        self.last = 0.0
        self.opener = build_opener(RestrictedRedirect())
        self.ua = f"AmericasHistoryCollector/{VERSION}" + (f" ({contact})" if contact else "")

    def query(self, **params):
        params = dict(action="query", format="json", formatversion=2, maxlag=5, **params)
        url = WIKI + "?" + urlencode(params)
        failure = None
        for attempt in range(self.retries):
            time.sleep(max(0.0, self.delay - (time.monotonic() - self.last)))
            pause = min(60.0, 2 ** (attempt + 1) + random.random())
            try:
                self.last = time.monotonic()
                req = Request(url, headers={"User-Agent": self.ua, "Accept": "application/json", "Accept-Encoding": "identity"})
                with self.opener.open(req, timeout=self.timeout) as response:
                    mime = response.headers.get_content_type()
                    if mime not in ("application/json", "text/json"):
                        raise Deferred(f"Unexpected API response type: {mime}")
                    raw = response.read(20 * 1024 * 1024 + 1)
                    if len(raw) > 20 * 1024 * 1024:
                        raise Skip("API response exceeds 20 MiB")
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise Deferred("Unexpected JSON structure")
                if result.get("error"):
                    error = result["error"]
                    code = error.get("code") if isinstance(error, dict) else ""
                    if code in {"missingtitle", "invalidtitle", "nosuchpageid"}:
                        raise Skip(str(error))
                    raise Deferred(str(error))
                return result
            except HTTPError as exc:
                if exc.code in (401, 403):
                    raise Deferred(f"HTTP {exc.code}: source denied access; no bypass attempted") from exc
                if exc.code not in (429, 500, 502, 503, 504):
                    raise Skip(f"HTTP {exc.code}") from exc
                failure = exc
                retry_after = exc.headers.get("Retry-After", "")
                try:
                    pause = max(pause, float(retry_after))
                except ValueError:
                    try:
                        pause = max(pause, parsedate_to_datetime(retry_after).timestamp() - time.time())
                    except (TypeError, ValueError, OverflowError):
                        pass
                if not math.isfinite(pause) or pause > 120:
                    raise Deferred("Server requested a long pause; retry on a later cycle") from exc
            except (URLError, OSError, ValueError, Deferred) as exc:
                failure = exc
            if attempt + 1 < self.retries:
                LOG.warning("HTTP | failed=%s | retry=%d/%d | in=%.1fs", failure, attempt + 2, self.retries, pause)
                time.sleep(max(0.0, pause))
        raise Deferred(f"Request failed after {self.retries} attempts: {failure}")


# ----------------------------- database ------------------------------------

def create_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS collector_meta(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tasks(
            id INTEGER PRIMARY KEY,
            region TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'wikipedia',
            kind TEXT NOT NULL,
            target TEXT NOT NULL,
            depth INTEGER NOT NULL DEFAULT 0,
            payload TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
            error TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT '',
            UNIQUE(region, source, kind, target)
        );
        CREATE TABLE IF NOT EXISTS documents(
            id TEXT PRIMARY KEY,
            primary_region TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'pending',
            quality TEXT NOT NULL DEFAULT '{}',
            collected_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memberships(
            region TEXT NOT NULL,
            document_id TEXT NOT NULL,
            origin TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(region, document_id)
        );
        CREATE TABLE IF NOT EXISTS origins(
            source_key TEXT PRIMARY KEY,
            document_id TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS doc_aliases(
            title TEXT PRIMARY KEY,
            document_id TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS task_pending ON tasks(status, region, kind, depth, id);
        CREATE INDEX IF NOT EXISTS document_status ON documents(status, primary_region);
        CREATE INDEX IF NOT EXISTS membership_doc ON memberships(document_id, region);
        """
    )


def migrate_legacy(db: sqlite3.Connection) -> None:
    """Import the v1 American-history DB in-place without refetching old content."""
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"articles", "membership", "queue", "categories"} <= tables:
        return
    if db.execute("SELECT 1 FROM collector_meta WHERE key='legacy_v1_migrated'").fetchone():
        return

    LOG.info("MIGRATE | detected legacy American-history database; importing checkpoints/content")
    with db:
        # Documents + memberships. Same page shared by multiple regions is stored once.
        rows = db.execute(
            """SELECT a.pageid,a.title,a.text,m.region,m.origin
               FROM membership m JOIN articles a ON a.pageid=m.pageid
               ORDER BY a.pageid,m.rowid"""
        ).fetchall()
        seen_docs = set()
        for row in rows:
            body = clean(row["text"])
            if not body:
                continue
            content_hash = normalized_digest(body)
            existing = db.execute("SELECT id FROM documents WHERE content_hash=?", (content_hash,)).fetchone()
            doc_id = existing[0] if existing else f"wikipedia:{row['pageid']}"
            if not existing and doc_id not in seen_docs:
                db.execute(
                    """INSERT OR IGNORE INTO documents
                       (id,primary_region,title,body,content_hash,status,quality,collected_at)
                       VALUES(?,?,?,?,?,'standalone','{}',?)""",
                    (doc_id, row["region"], clean(row["title"]), body, content_hash, now()),
                )
            seen_docs.add(doc_id)
            db.execute("INSERT OR IGNORE INTO memberships VALUES(?,?,?)", (row["region"], doc_id, row["origin"] or "legacy"))
            db.execute("INSERT OR IGNORE INTO origins VALUES(?,?)", (f"wikipedia:{row['pageid']}", doc_id))
            db.execute("INSERT OR IGNORE INTO doc_aliases VALUES(?,?)", (row["title"], doc_id))

        # Legacy aliases can prevent a repeated request after restart.
        if "aliases" in tables:
            for alias in db.execute("SELECT title,pageid FROM aliases"):
                source = db.execute("SELECT document_id FROM origins WHERE source_key=?", (f"wikipedia:{alias['pageid']}",)).fetchone()
                if source:
                    db.execute("INSERT OR IGNORE INTO doc_aliases VALUES(?,?)", (alias["title"], source[0]))

        # Queue/article checkpoints.
        for row in db.execute("SELECT region,title,origin,done FROM queue"):
            enqueue(db, row["region"], "article", row["title"], 0, {"origin": row["origin"] or "legacy"},
                    status="done" if row["done"] else "pending")

        # Category continuation checkpoints. Old continuation is directly compatible.
        for row in db.execute("SELECT region,title,depth,continuation,done FROM categories"):
            payload = {}
            if row["continuation"]:
                try:
                    payload["continue"] = json.loads(row["continuation"])
                except json.JSONDecodeError:
                    pass
            enqueue(db, row["region"], "category", row["title"], row["depth"], payload,
                    status="done" if row["done"] else "pending")

        db.execute("INSERT OR REPLACE INTO collector_meta VALUES('legacy_v1_migrated',?)", (now(),))
    LOG.info("MIGRATE | imported %d legacy unique documents", db.execute("SELECT count(*) FROM documents").fetchone()[0])


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    create_schema(db)
    row = db.execute("SELECT value FROM collector_meta WHERE key='schema'").fetchone()
    if row and row[0] != SCHEMA:
        db.close()
        raise RuntimeError(f"Incompatible collector schema: {row[0]!r}")
    migrate_legacy(db)
    with db:
        db.execute("INSERT OR IGNORE INTO collector_meta VALUES('schema',?)", (SCHEMA,))
    return db


def enqueue(db, region, kind, target, depth=0, payload=None, *, status="pending") -> None:
    if region not in REGIONS:
        return
    db.execute(
        """INSERT OR IGNORE INTO tasks(region,source,kind,target,depth,payload,status,updated_at)
           VALUES(?,'wikipedia',?,?,?,?,?,?)""",
        (region, kind, target, int(depth), json.dumps(payload or {}, ensure_ascii=False), status, now()),
    )


def seed(db, args) -> None:
    with db:
        # Transient errors are retried automatically on every process start.
        db.execute("UPDATE tasks SET status='pending',error='',updated_at=? WHERE status='error'", (now(),))
        if args.retry_skipped:
            db.execute("UPDATE tasks SET status='pending',error='',updated_at=? WHERE status='skipped'", (now(),))
        if args.recheck_quality:
            db.execute("UPDATE documents SET status='pending',quality='{}' WHERE status IN ('pass','warn','block','standalone')")
        for region, places in REGIONS.items():
            for place in places:
                title = "History of " + place
                enqueue(db, region, "article", title, payload={"origin": "country/region overview"})
                enqueue(db, region, "category", "Category:" + title, 0, {"root": True})
            for title in EXTRA[region]:
                enqueue(db, region, "article", title, payload={"origin": "historical topic seed"})
                enqueue(db, region, "category", "Category:" + title, 0, {"root": True})


def task_done(db, task) -> None:
    db.execute("UPDATE tasks SET status='done',error='',updated_at=? WHERE id=?", (now(), task["id"]))


# ----------------------------- collection ----------------------------------

def ingest(db, region: str, source_key: str, title: str, body: str, origin: str, args, pipeline=None) -> tuple[str | None, bool]:
    title, body = clean(title), clean(body)
    if not title or not body:
        return None, False
    words = re.findall(r"\b[\w'-]+\b", body)
    if len(words) < args.min_words:
        return None, False
    if len(body) and sum(ch.isalpha() for ch in body) / len(body) < 0.35:
        return None, False
    if re.search(r"\b(may refer to:|this disambiguation page)\b", body[:1200], re.I):
        return None, False

    # Strong local relevance check; seeded overview pages bypass because their titles
    # are already intentionally selected and some extracts may omit the word history.
    if not HISTORY_SIGNAL.search(title + " " + body[:12000]):
        region_terms = "|".join(re.escape(x.replace("the ", "")) for x in REGIONS[region])
        if not re.search(region_terms, title + " " + body[:12000], re.I):
            return None, False
        # This is the only point where Reasoning may be useful: locally ambiguous
        # material that names the region but lacks an explicit historical signal.
        if pipeline is not None and not pipeline.maybe_reason_relevance(title, body, region):
            return None, False

    existing_origin = db.execute("SELECT document_id FROM origins WHERE source_key=?", (source_key,)).fetchone()
    if existing_origin:
        doc_id = existing_origin[0]
        added = db.execute("INSERT OR IGNORE INTO memberships VALUES(?,?,?)", (region, doc_id, origin)).rowcount > 0
        return doc_id, added

    content_hash = normalized_digest(body)
    existing = db.execute("SELECT id FROM documents WHERE content_hash=?", (content_hash,)).fetchone()
    created = existing is None
    if existing:
        doc_id = existing[0]
    else:
        # The source key is intentionally internal and never appears in training files.
        doc_id = source_key
        db.execute(
            """INSERT INTO documents(id,primary_region,title,body,content_hash,status,quality,collected_at)
               VALUES(?,?,?,?,?,'pending','{}',?)""",
            (doc_id, region, title, body, content_hash, now()),
        )
    db.execute("INSERT OR IGNORE INTO origins VALUES(?,?)", (source_key, doc_id))
    db.execute("INSERT OR IGNORE INTO memberships VALUES(?,?,?)", (region, doc_id, origin))
    db.execute("INSERT OR IGNORE INTO doc_aliases VALUES(?,?)", (title, doc_id))
    return doc_id, created


def category_task(db, api: APIClient, task, args) -> int:
    payload = json.loads(task["payload"] or "{}")
    params = dict(list="categorymembers", cmtitle=task["target"], cmtype="page|subcat", cmnamespace="0|14", cmlimit=500)
    params.update(payload.get("continue", {}))
    data = api.query(**params)
    members = data.get("query", {}).get("categorymembers")
    if not isinstance(members, list):
        raise Deferred("Category response lacks members")

    discovered = 0
    with db:
        for member in members:
            title = clean(member.get("title", ""))
            if not title:
                continue
            if member.get("ns") == 0:
                before = db.total_changes
                enqueue(db, task["region"], "article", title, task["depth"], {"origin": task["target"]})
                discovered += int(db.total_changes > before)
            elif member.get("ns") == 14 and not SKIP_CATEGORY.search(title):
                child_depth = task["depth"] + 1
                if args.depth and child_depth > args.depth:
                    continue
                # At shallow depth trust root category membership. At deeper levels,
                # require a history signal to reduce category-graph drift.
                if child_depth > 2 and not HISTORY_SIGNAL.search(title):
                    continue
                before = db.total_changes
                enqueue(db, task["region"], "category", title, child_depth, {"origin": task["target"]})
                discovered += int(db.total_changes > before)

        continuation = data.get("continue")
        if continuation:
            new_payload = dict(payload)
            new_payload["continue"] = continuation
            if new_payload == payload:
                raise Deferred("Repeated category continuation cursor")
            db.execute("UPDATE tasks SET payload=?,updated_at=? WHERE id=?", (json.dumps(new_payload), now(), task["id"]))
        else:
            task_done(db, task)
            db.execute(
                """INSERT INTO collector_meta(key,value) VALUES('categories_completed','1')
                   ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER)+1"""
            )
    return discovered


def article_task(db, api: APIClient, task, args, pipeline=None) -> int:
    # Reuse a known alias without touching the network.
    known = db.execute("SELECT document_id FROM doc_aliases WHERE title=?", (task["target"],)).fetchone()
    if known:
        payload = json.loads(task["payload"] or "{}")
        with db:
            added = db.execute(
                "INSERT OR IGNORE INTO memberships VALUES(?,?,?)",
                (task["region"], known[0], payload.get("origin", "alias")),
            ).rowcount
            task_done(db, task)
        return int(bool(added))

    data = api.query(
        titles=task["target"], redirects=1,
        prop="extracts|pageprops", explaintext=1, exsectionformat="plain", exlimit=1,
    )
    pages = data.get("query", {}).get("pages")
    if not isinstance(pages, list) or not pages:
        raise Deferred("Missing Wikipedia pages")
    page = pages[0]
    if page.get("missing") or page.get("invalid") or "disambiguation" in page.get("pageprops", {}):
        raise Skip("Missing/invalid/disambiguation page")
    body = page.get("extract", "")
    if not str(body).strip():
        raise Skip("No article text")
    pageid = page.get("pageid")
    if not isinstance(pageid, int):
        raise Deferred("Wikipedia page lacks stable page identifier")

    payload = json.loads(task["payload"] or "{}")
    with db:
        doc_id, created = ingest(
            db, task["region"], f"wikipedia:{pageid}", page.get("title", task["target"]), body,
            payload.get("origin", "category discovery"), args, pipeline,
        )
        if doc_id:
            db.execute("INSERT OR IGNORE INTO doc_aliases VALUES(?,?)", (task["target"], doc_id))
        task_done(db, task)
    return int(created)


def select_task(db, region: str, args):
    depth_clause = "" if args.depth == 0 else " AND (kind='article' OR depth<=?)"
    params = (region,) if args.depth == 0 else (region, args.depth)
    return db.execute(
        "SELECT * FROM tasks WHERE region=? AND status='pending'" + depth_clause +
        " ORDER BY CASE kind WHEN 'article' THEN 0 ELSE 1 END, depth, id LIMIT 1",
        params,
    ).fetchone()


def refresh_categories_for_continuous_mode(db) -> int:
    """Refresh known categories after the graph is exhausted, without duplicating text."""
    with db:
        count = db.execute("UPDATE tasks SET status='pending',payload='{}',error='',updated_at=? WHERE kind='category' AND status='done'", (now(),)).rowcount
        db.execute("INSERT OR REPLACE INTO collector_meta VALUES('last_category_refresh',?)", (now(),))
    return count


# ----------------------------- SLAI agents ---------------------------------

def _compatible_method(agent, names, *, min_positional=0):
    for name in names:
        method = getattr(agent, name, None)
        if not callable(method):
            continue
        try:
            sig = inspect.signature(method)
            positional = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            required = [p for p in positional if p.default is inspect._empty]
            if len(required) <= min_positional or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                return method
        except (TypeError, ValueError):
            continue
    return None


AGENT_IMPORT_SPECS = {
    "quality": ("src.agents.quality_agent", "QualityAgent"),
    "knowledge": ("src.agents.knowledge_agent", "KnowledgeAgent"),
    "planning": ("src.agents.planning_agent", "PlanningAgent"),
    "reasoning": ("src.agents.reasoning_agent", "ReasoningAgent"),
    "evaluation": ("src.agents.evaluation_agent", "EvaluationAgent"),
}


def _exception_chain(exc):
    """Yield an exception and its explicit/implicit causes without looping."""
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _root_import_problem(exc):
    """Return the most useful nested import failure hidden by AgentFactory."""
    for item in _exception_chain(exc):
        if isinstance(item, ModuleNotFoundError):
            return {
                "type": type(item).__name__,
                "missing": getattr(item, "name", None) or "unknown",
                "message": str(item),
            }
        if isinstance(item, ImportError):
            return {"type": type(item).__name__, "missing": None, "message": str(item)}
    tail = list(_exception_chain(exc))[-1]
    return {"type": type(tail).__name__, "missing": None, "message": str(tail)}


def preflight_agent_imports():
    """Import each requested SLAI agent once and preserve the real failure reason.

    AgentFactory v2.3 intentionally wraps ModuleNotFoundError as FAC-1401, which
    hides whether the agent module itself is absent or one of its transitive
    dependencies failed. Preflight prevents five opaque Factory errors and lets
    auto mode skip only the unavailable agent. Successful imports are cached in
    sys.modules, so AgentFactory does not pay the import cost again.
    """
    status = {}
    for kind, (module_path, class_name) in AGENT_IMPORT_SPECS.items():
        try:
            module = importlib.import_module(module_path)
            agent_cls = getattr(module, class_name, None)
            if not inspect.isclass(agent_cls):
                raise ImportError(f"{module_path} does not expose {class_name}")
            status[kind] = {"ok": True, "module": module_path, "class": class_name}
        except Exception as exc:
            problem = _root_import_problem(exc)
            status[kind] = {
                "ok": False, "module": module_path, "class": class_name,
                "error_type": problem["type"], "missing": problem["missing"],
                "error": problem["message"],
            }
    return status


class AgentPipeline:
    """Use agents only where their interface improves collection without rewriting source text."""

    def __init__(self, factory=None, memory=None, mode="off", owns=False, import_status=None):
        self.factory, self.memory, self.mode, self.owns = factory, memory, mode, owns
        self.import_status = import_status or {}
        self.closed = False
        self.indexed = set()
        self.agents = {}
        self.quality = None
        self.knowledge = None
        self.planner = None
        self.reasoner = None
        self.evaluator = None
        self.plan_method = None
        self.reason_method = None
        self.eval_method = None

        if mode == "off" or factory is None:
            return

        # Per-agent creation is isolated: auto mode must never lose collection because
        # one optional agent is unavailable or incompatible.
        self.quality = self._try_create("quality", useful=True, config={"enabled": True, "auto_route_via_workflow": False})
        if self.quality and not callable(getattr(self.quality, "evaluate_batch", None)):
            LOG.warning("AGENT | quality available but evaluate_batch is missing; skipped")
            self.quality = None

        self.knowledge = self._try_create("knowledge", useful=True, config={
            "source": "american_history_collector", "directory_path": "", "retrieval_mode": "tfidf",
            "bias_detection_enabled": False, "use_ontology_expansion": False,
        })
        if self.knowledge and (not callable(getattr(self.knowledge, "add_document", None)) or not isinstance(getattr(self.knowledge, "doc_index", None), dict)):
            LOG.warning("AGENT | knowledge lacks add_document/doc_index; skipped")
            self.knowledge = None

        self.planner = self._try_create("planning", useful=False)
        if self.planner:
            self.plan_method = _compatible_method(self.planner, ("prioritize_tasks", "prioritize", "rank_tasks"), min_positional=1)
            if not self.plan_method:
                LOG.info("AGENT | planning loaded but no low-overhead task-priority interface; not forced")

        self.reasoner = self._try_create("reasoning", useful=False)
        if self.reasoner:
            self.reason_method = _compatible_method(self.reasoner, ("assess_relevance", "classify_relevance", "score_relevance"), min_positional=1)
            if not self.reason_method:
                LOG.info("AGENT | reasoning loaded but no lightweight relevance interface; not forced")

        self.evaluator = self._try_create("evaluation", useful=False)
        if self.evaluator:
            self.eval_method = _compatible_method(self.evaluator, ("evaluate_collection", "evaluate_metrics", "evaluate"), min_positional=1)
            if not self.eval_method:
                LOG.info("AGENT | evaluation loaded but no lightweight metrics interface; not forced")

        if mode == "required":
            missing = [name for name, value in (("quality", self.quality), ("knowledge", self.knowledge)) if value is None]
            if missing:
                raise RuntimeError("Required core agents unavailable/incompatible: " + ", ".join(missing))

    def _try_create(self, kind, useful=False, config=None):
        preflight = self.import_status.get(kind)
        if preflight and not preflight.get("ok"):
            missing = preflight.get("missing")
            if missing:
                detail = (f"missing import/dependency '{missing}' while loading "
                          f"{preflight.get('module')}: {preflight.get('error')}")
            else:
                detail = (f"{preflight.get('error_type')} while loading "
                          f"{preflight.get('module')}: {preflight.get('error')}")
            if self.mode == "required" and useful:
                raise RuntimeError(f"Required {kind} agent unavailable: {detail}")
            LOG.warning("AGENT | %s unavailable; skipped | %s", kind, detail)
            return None
        try:
            agent = self.factory.create(kind, shared_memory=self.memory, config=config or {})
            self.agents[kind] = agent
            LOG.info("AGENT | loaded=%s%s", kind, " | active" if useful else " | optional")
            return agent
        except Exception as exc:
            problem = _root_import_problem(exc)
            missing = problem.get("missing")
            detail = (f"missing import/dependency '{missing}': {problem['message']}"
                      if missing else f"{problem['type']}: {problem['message']}")
            if self.mode == "required" and useful:
                raise RuntimeError(f"Required {kind} agent failed to initialize: {detail}") from exc
            LOG.warning("AGENT | %s unavailable; skipped | %s", kind, detail)
            return None

    def assess(self, db, batch_size=12):
        eligible = ("pending", "standalone") if self.quality else ("pending",)
        placeholders = ",".join("?" for _ in eligible)
        while True:
            rows = db.execute(
                f"SELECT * FROM documents WHERE status IN ({placeholders}) ORDER BY rowid LIMIT ?",
                (*eligible, batch_size),
            ).fetchall()
            if not rows:
                break
            if self.quality:
                records = [{
                    "id": row["id"], "title": row["title"], "text": row["body"],
                    "source_id": "wikipedia", "source_type": "historical_encyclopedia",
                    "collected_at": row["collected_at"], "word_count": len(row["body"].split()),
                } for row in rows]
                schema = {
                    "schema_version": SCHEMA,
                    "required_fields": ["id", "title", "text", "source_id", "source_type", "collected_at", "word_count"],
                    "fields": {
                        "id": {"type": "str", "required": True}, "title": {"type": "str", "required": True},
                        "text": {"type": "str", "required": True}, "source_id": {"type": "str", "required": True},
                        "source_type": {"type": "str", "required": True}, "collected_at": {"type": "str", "required": True},
                        "word_count": {"type": "int", "required": True},
                    },
                }
                checksum = hashlib.sha256("|".join(row["content_hash"] for row in rows).encode("ascii")).hexdigest()
                result = self.quality.evaluate_batch(
                    records,
                    dataset_id="american_history_collector",
                    source_id="wikipedia",
                    batch_id="american-history-" + checksum[:24],
                    schema=schema,
                    use_case="knowledge_ingestion",
                    feature_fields=["word_count"],
                    provenance={
                        "source_id": "wikipedia", "source_type": "historical_encyclopedia",
                        "collected_at": rows[0]["collected_at"], "collector": "american_history_collector",
                        "checksum": checksum, "uri": "", "schema_version": SCHEMA,
                    },
                    source_metadata={"source_id": "wikipedia", "source_type": "historical_encyclopedia"},
                    context={
                        "route": "american_history_collector->quality->knowledge",
                        "external_content": True, "content_type": "historical_encyclopedia",
                        "fact_verification_performed": False,
                    },
                )
                if not isinstance(result, dict) or result.get("verdict") not in {"pass", "warn", "block"}:
                    raise RuntimeError("QualityAgent returned an invalid verdict; records remain pending")
                status = result["verdict"]
                if result.get("quarantine_count", 0) or result.get("quarantine_entries"):
                    status = "block"
                LOG.info("QUALITY | docs=%d | verdict=%s | score=%s", len(rows), status, result.get("batch_score"))
            else:
                status = "standalone"
                result = {"verdict": "not_assessed", "reason": "Quality Agent unavailable/off"}
            with db:
                db.executemany(
                    "UPDATE documents SET status=?,quality=? WHERE id=?",
                    [(status, json.dumps(result, ensure_ascii=False, default=str), row["id"]) for row in rows],
                )

    def sync(self, db, strict=False):
        if not self.knowledge:
            return
        statuses = ("pass",) if strict else ("pass", "warn", "standalone")
        placeholders = ",".join("?" for _ in statuses)
        added = 0
        for row in db.execute(f"SELECT * FROM documents WHERE status IN ({placeholders}) ORDER BY rowid", statuses):
            if row["id"] in self.indexed:
                continue
            words = row["body"].split()
            for start in range(0, len(words), 1000):
                part = start // 1000
                text = f"American history: {row['title']}\n\n" + " ".join(words[start:start + 1000])
                doc_id = row["id"] + f":chunk:{part}"
                existing = self.knowledge.doc_index.get(doc_id)
                if existing is None:
                    self.knowledge.add_document(
                        text=text, doc_id=doc_id,
                        metadata={"region": row["primary_region"], "parent_id": row["id"], "chunk": part,
                                  "quality_verdict": row["status"], "trust": "unverified_external_content"},
                    )
                    existing = self.knowledge.doc_index.get(doc_id)
                if not isinstance(existing, dict):
                    raise RuntimeError("KnowledgeAgent did not acknowledge collected text")
            self.indexed.add(row["id"])
            added += 1
        if added:
            LOG.info("KNOWLEDGE | indexed=%d new documents this process", added)

    def maybe_prioritize_regions(self, db, regions):
        """Planning is advisory and only used if it returns a safe permutation."""
        if not self.plan_method or len(regions) < 2:
            return regions
        metrics = []
        for region in regions:
            pending = db.execute("SELECT count(*) FROM tasks WHERE region=? AND status='pending'", (region,)).fetchone()[0]
            docs = db.execute("SELECT count(*) FROM documents WHERE primary_region=?", (region,)).fetchone()[0]
            metrics.append({"region": region, "pending": pending, "documents": docs})
        try:
            result = self.plan_method(metrics)
            if isinstance(result, (list, tuple)):
                names = [x.get("region") if isinstance(x, dict) else x for x in result]
                if set(names) == set(regions) and len(names) == len(regions):
                    return names
        except Exception as exc:
            LOG.warning("PLANNING | advisory call failed; using fair round-robin (%s)", exc)
            self.plan_method = None
        return regions

    def maybe_reason_relevance(self, title, body, region):
        """Reasoning can reject borderline text, but never rewrites source material."""
        if not self.reason_method:
            return True
        try:
            result = self.reason_method({"title": title, "text": body[:6000], "region": region, "task": "historical_relevance"})
            if isinstance(result, bool):
                return result
            if isinstance(result, dict):
                if "relevant" in result:
                    return bool(result["relevant"])
                score = result.get("score")
                if isinstance(score, (int, float)):
                    return float(score) >= 0.5
        except Exception as exc:
            LOG.warning("REASONING | advisory call failed; disabling advisory relevance (%s)", exc)
            self.reason_method = None
        return True

    def maybe_evaluate(self, db):
        if not self.eval_method:
            return
        metrics = {
            "documents": db.execute("SELECT count(*) FROM documents").fetchone()[0],
            "pending_tasks": db.execute("SELECT count(*) FROM tasks WHERE status='pending'").fetchone()[0],
            "errors": db.execute("SELECT count(*) FROM tasks WHERE status='error'").fetchone()[0],
            "blocked": db.execute("SELECT count(*) FROM documents WHERE status='block'").fetchone()[0],
        }
        try:
            # Operational evaluation only; its output never changes historical text.
            result = self.eval_method(metrics)
            LOG.info("EVALUATION | %s", str(result)[:500])
        except Exception as exc:
            LOG.warning("EVALUATION | advisory call failed; disabling periodic evaluation (%s)", exc)
            self.eval_method = None

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.owns:
            try:
                self.factory.shutdown()
            finally:
                self.memory.close()


def find_slai_root(explicit=None) -> Path:
    candidates = [explicit.expanduser().resolve()] if explicit else []
    if not explicit:
        for base in (Path.cwd(), Path(__file__).resolve().parent):
            candidates.extend((base, *base.parents))
    for root in candidates:
        if (root / "src/agents/agent_factory.py").is_file():
            return root
    raise RuntimeError("SLAI root not found")


def _slai_venv_paths(root: Path) -> tuple[Path, Path]:
    """Return the authoritative SLAI venv directory and interpreter path."""
    venv_dir = root / "venv"
    python_path = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return venv_dir, python_path


def ensure_slai_venv(root: Path) -> None:
    """Force agent-enabled runs onto SLAI's own virtual environment.

    Windows' ``py`` launcher can ignore an already activated virtual environment.
    Agent imports must therefore not trust the shell prompt; ``sys.prefix`` is the
    authoritative runtime check. If this process is not running from SLAI/venv,
    replace it with that interpreter while preserving every CLI argument.
    """
    venv_dir, venv_python = _slai_venv_paths(root)
    if not venv_python.is_file():
        raise RuntimeError(f"SLAI virtual-environment interpreter not found: {venv_python}")

    try:
        current_prefix = Path(sys.prefix).resolve()
        expected_prefix = venv_dir.resolve()
    except OSError:
        current_prefix = Path(sys.prefix)
        expected_prefix = venv_dir

    if os.path.normcase(str(current_prefix)) == os.path.normcase(str(expected_prefix)):
        return

    marker = "SLAI_COLLECTOR_VENV_REEXEC"
    if os.environ.get(marker) == "1":
        raise RuntimeError(
            "Failed to enter the SLAI virtual environment after re-exec. "
            f"current={sys.executable!r} expected={str(venv_python)!r}"
        )

    LOG.warning(
        "RUNTIME | current interpreter is outside SLAI venv; restarting with %s",
        venv_python,
    )
    env = os.environ.copy()
    env[marker] = "1"
    argv = [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]]

    # os.execve replaces this process, so no duplicate collector/agent process is
    # left behind and lock/checkpoint semantics remain unchanged.
    os.execve(str(venv_python), argv, env)


def verify_agent_runtime_dependencies() -> None:
    """Fail early with the real dependency error before importing SLAI agents."""
    failures = []
    for module_name in ("yaml", "numpy"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError(
            "SLAI venv dependency check failed. The collector is using "
            f"{sys.executable}. " + " | ".join(failures)
        )


def create_agents(root: Path, mode: str) -> AgentPipeline:
    global LOG

    # This must happen BEFORE importing AgentFactory, SharedMemory, NumPy-backed
    # subsystems, or adding project-local import paths. It makes agent availability
    # independent of whether the user launched with `python` or Windows `py`.
    ensure_slai_venv(root)
    verify_agent_runtime_dependencies()

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.agents.agent_factory import AgentFactory
    from src.agents.collaborative.shared_memory import SharedMemory
    # Keep this collector on its existing standard logger. Reconfiguring logging
    # through SLAI here duplicates every collector message because main() has
    # already attached console/file handlers. AgentFactory keeps its own logger.

    # Resolve real import failures before asking AgentFactory to create agents.
    # This avoids FAC-1401 masking a missing transitive dependency.
    import_status = preflight_agent_imports()
    memory = SharedMemory()
    factory = None
    try:
        factory = AgentFactory()
        return AgentPipeline(factory, memory, mode=mode, owns=True, import_status=import_status)
    except BaseException:
        try:
            if factory is not None:
                factory.shutdown()
        finally:
            memory.close()
        raise


# ----------------------------- export --------------------------------------
@contextmanager
def atomic_writer(path: Path):
    temp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def export(db, folder: Path, strict=False):
    # Normal mode keeps pass/warn/standalone/pending content but excludes explicit
    # Quality blocks. Strict mode is pass-only.
    selection = "status='pass'" if strict else "length(trim(body))>0 AND status!='block'"
    rows = db.execute(f"SELECT rowid AS sequence,* FROM documents WHERE {selection} ORDER BY rowid").fetchall()

    # Unique per-document files: this is the preferred LANTRA training input.
    manifest_path = folder / "training_files.json"
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    except (OSError, json.JSONDecodeError):
        previous = []
    managed = []
    for row in rows:
        relative = f"training_text/american_history/{row['sequence']:08d}.txt"
        path = folder / relative
        text = row["title"] + "\n\n" + row["body"] + "\n"
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            with atomic_writer(path) as stream:
                stream.write(text)
        managed.append(relative)
    current = set(managed)
    for relative in previous:
        if isinstance(relative, str) and re.fullmatch(r"training_text/american_history/[0-9]{8,}\.txt", relative) and relative not in current:
            (folder / relative).unlink(missing_ok=True)
    with atomic_writer(manifest_path) as stream:
        json.dump(managed, stream, indent=2)

    # Global aggregate: every document exactly once.
    with atomic_writer(folder / "american_history.txt") as stream:
        for row in rows:
            stream.write(row["title"] + "\n\n" + row["body"] + "\n\n\n")

    # Region aggregates use primary_region only, so concatenating all four still
    # cannot duplicate a document that happened to belong to multiple regions.
    region_counts = {}
    for region in REGION_ORDER:
        count = words = 0
        with atomic_writer(folder / f"{region}.txt") as stream:
            for row in db.execute(
                f"SELECT title,body FROM documents WHERE primary_region=? AND {selection} ORDER BY rowid",
                (region,),
            ):
                stream.write(row["title"] + "\n\n" + row["body"] + "\n\n\n")
                count += 1
                words += len(row["body"].split())
        region_counts[region] = {"documents": count, "words": words}

    summary = {
        "exported_at": now(), "version": VERSION, "strict_quality": strict,
        "unique_documents": len(rows), "regions": region_counts,
        "document_statuses": dict(db.execute("SELECT status,count(*) FROM documents GROUP BY status")),
        "task_statuses": dict(db.execute("SELECT status,count(*) FROM tasks GROUP BY status")),
    }
    with atomic_writer(folder / "collection_summary.json") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    LOG.info("EXPORT | unique_docs=%d | %s", len(rows), " | ".join(f"{r}={v['documents']}" for r, v in region_counts.items()))
    return summary


@contextmanager
def collection_lock(folder: Path):
    lock_path = folder / "collector.lock"
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another American-history collector is already using this folder") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_UN)


def collect(db, api, args, pipeline: AgentPipeline):
    steps = 0
    new_docs = 0
    started = time.monotonic()
    next_refresh = time.monotonic() + args.refresh_hours * 3600
    disabled_until = 0.0

    while True:
        if args.max_steps and steps >= args.max_steps:
            LOG.info("STOP | --max-steps reached (%d)", args.max_steps)
            return

        progress = False
        regions = pipeline.maybe_prioritize_regions(db, list(REGION_ORDER))
        for region in regions:
            if args.max_steps and steps >= args.max_steps:
                return
            if args.max_articles:
                current = db.execute("SELECT count(*) FROM documents WHERE primary_region=?", (region,)).fetchone()[0]
                if current >= args.max_articles:
                    continue
            if args.max_categories:
                row = db.execute("SELECT value FROM collector_meta WHERE key='categories_completed'").fetchone()
                if row and int(row[0]) >= args.max_categories:
                    # Existing article queue remains eligible; only suppress category work.
                    task = db.execute(
                        "SELECT * FROM tasks WHERE region=? AND status='pending' AND kind='article' ORDER BY id LIMIT 1",
                        (region,),
                    ).fetchone()
                else:
                    task = select_task(db, region, args)
            else:
                task = select_task(db, region, args)
            if task is None:
                continue

            progress = True
            steps += 1
            LOG.info("FETCH | step=%d | region=%s | kind=%s | target=%s", steps, region, task["kind"], task["target"])
            try:
                if task["kind"] == "category":
                    discovered = category_task(db, api, task, args)
                    LOG.info("DISCOVER | region=%s | new_tasks=%d", region, discovered)
                elif task["kind"] == "article":
                    created = article_task(db, api, task, args, pipeline)
                    new_docs += created
                    if created:
                        LOG.info("SAVE | region=%s | unique_total=%d", region, db.execute("SELECT count(*) FROM documents").fetchone()[0])
                else:
                    raise Skip("Unknown task kind")
            except Skip as exc:
                with db:
                    db.execute("UPDATE tasks SET status='skipped',error=?,updated_at=? WHERE id=?", (str(exc), now(), task["id"]))
                LOG.warning("SKIP | %s | %s", task["target"], exc)
            except (Deferred, OSError, ValueError) as exc:
                with db:
                    db.execute("UPDATE tasks SET status='error',error=?,updated_at=? WHERE id=?", (str(exc), now(), task["id"]))
                disabled_until = time.monotonic() + args.error_cooldown
                LOG.warning("DEFER | %s | source cooldown %.0fs", exc, args.error_cooldown)
                break

            if steps % args.checkpoint_every == 0:
                pipeline.assess(db, args.batch_size)
                export(db, args.output_dir, args.strict_quality)
                pipeline.sync(db, args.strict_quality)
                if steps % args.evaluation_every == 0:
                    pipeline.maybe_evaluate(db)
                LOG.info(
                    "PROGRESS | steps=%d | new_docs=%d | total_docs=%d | pending=%d | minutes=%.1f",
                    steps, new_docs,
                    db.execute("SELECT count(*) FROM documents").fetchone()[0],
                    db.execute("SELECT count(*) FROM tasks WHERE status='pending'").fetchone()[0],
                    (time.monotonic() - started) / 60,
                )

        if disabled_until > time.monotonic():
            sleep_for = min(args.idle_seconds, max(1.0, disabled_until - time.monotonic()))
            LOG.info("IDLE | transient source failure | sleeping %.1fs", sleep_for)
            time.sleep(sleep_for)
            if time.monotonic() >= disabled_until:
                with db:
                    db.execute("UPDATE tasks SET status='pending',error='',updated_at=? WHERE status='error'", (now(),))
            continue

        if progress:
            continue

        if not args.continuous:
            LOG.info("DONE | no pending tasks remain within configured bounds")
            return

        # Continuous mode deliberately stays alive until Ctrl+C. Refreshing completed
        # category cursors periodically can discover newly added Wikimedia content while
        # source-key/content-hash dedupe prevents corpus duplication.
        if time.monotonic() >= next_refresh:
            refreshed = refresh_categories_for_continuous_mode(db)
            next_refresh = time.monotonic() + args.refresh_hours * 3600
            LOG.info("REFRESH | reopened=%d completed category tasks", refreshed)
            if refreshed:
                continue
        LOG.info("IDLE | no new pending work; sleeping %.1fs (Ctrl+C to stop)", args.idle_seconds)
        time.sleep(args.idle_seconds)


# ----------------------------- CLI / tests ---------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, help="Compatibility option; output is always ./american_history beside this script")
    p.add_argument("--slai-root", type=Path)
    p.add_argument("--agents", choices=("auto", "required", "off"), default="auto",
                   help="auto: use compatible agents when available; required: require Quality+Knowledge; off: standalone")
    p.add_argument("--max-articles", type=int, default=0, help="Per primary region; 0 = unlimited")
    p.add_argument("--max-categories", type=int, default=0, help="Total completed categories; 0 = unlimited")
    p.add_argument("--depth", type=int, default=0, help="Category depth; 0 = unlimited")
    p.add_argument("--max-steps", type=int, default=0, help="Debug/testing bound; 0 = unlimited")
    p.add_argument("--min-words", type=int, default=80)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--evaluation-every", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=45)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--contact", default="")
    p.add_argument("--idle-seconds", type=float, default=30.0)
    p.add_argument("--error-cooldown", type=float, default=60.0)
    p.add_argument("--refresh-hours", type=float, default=6.0)
    p.add_argument("--strict-quality", action="store_true", help="Export/index only Quality pass documents")
    p.add_argument("--retry-skipped", action="store_true")
    p.add_argument("--recheck-quality", action="store_true")
    p.add_argument("--process-only", action="store_true", help="Run agent processing/export on saved corpus; no network")
    p.add_argument("--export-only", action="store_true", help="Rebuild content-only exports; no agents/network")
    p.add_argument("--check-agents", action="store_true")
    p.add_argument("--no-continuous", dest="continuous", action="store_false", help="Exit when current task graph is exhausted")
    p.set_defaults(continuous=True)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    for field in ("max_articles", "max_categories", "depth", "max_steps"):
        if getattr(args, field) < 0:
            p.error(field + " must be nonnegative")
    for field in ("min_words", "checkpoint_every", "evaluation_every", "batch_size", "retries"):
        if getattr(args, field) < 1:
            p.error(field + " must be positive")
    for field in ("delay", "timeout", "idle_seconds", "error_cooldown", "refresh_hours"):
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0:
            p.error(field + " must be finite and positive")
    if args.delay < 1:
        p.error("--delay must be at least 1 second")
    if any(ord(c) < 32 or ord(c) > 126 for c in args.contact):
        p.error("--contact must contain printable ASCII only")
    if args.strict_quality and args.agents == "off" and not args.export_only:
        p.error("--strict-quality requires QualityAgent")
    if args.check_agents and (args.agents == "off" or args.export_only):
        p.error("--check-agents requires agents and cannot accompany --export-only")
    return args


def self_test():
    import tempfile

    class FakeAPI:
        def query(self, **params):
            if params.get("list") == "categorymembers":
                return {
                    "query": {"categorymembers": [
                        {"ns": 0, "title": "Example history"},
                        {"ns": 14, "title": "Category:Colonial history"},
                    ]}
                }
            return {"query": {"pages": [{
                "pageid": 101,
                "title": "Example history",
                "extract": ("This historical account describes colonial settlement, independence, "
                            "migration, government, war, society and economic change across the Americas. ") * 8,
            }]}}

    class Memory:
        def close(self):
            pass

    class Knowledge:
        def __init__(self):
            self.doc_index = {}
        def add_document(self, text, doc_id=None, metadata=None):
            self.doc_index[doc_id] = {"text": text.strip(), "metadata": metadata}

    class Quality:
        def evaluate_batch(self, records, **kwargs):
            assert records
            return {"verdict": "pass", "batch_score": 0.95, "flags": []}

    class Factory:
        def __init__(self):
            self.knowledge = Knowledge()
        def create(self, kind, shared_memory=None, config=None):
            if kind == "quality":
                return Quality()
            if kind == "knowledge":
                return self.knowledge
            raise RuntimeError("optional agent intentionally unavailable")
        def shutdown(self):
            pass

    args = parse_args(["--agents", "off", "--min-words", "5", "--no-continuous"])
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        db = connect(folder / "history.sqlite3")
        with db:
            enqueue(db, "north_america", "category", "Category:History of Canada", 0, {"root": True})
        cat = db.execute("SELECT * FROM tasks WHERE kind='category'").fetchone()
        assert category_task(db, FakeAPI(), cat, args) >= 1
        article = db.execute("SELECT * FROM tasks WHERE kind='article'").fetchone()
        assert article_task(db, FakeAPI(), article, args) == 1
        # Requeued alias must not refetch/duplicate the document.
        with db:
            db.execute("UPDATE tasks SET status='pending' WHERE id=?", (article["id"],))
        assert article_task(db, FakeAPI(), db.execute("SELECT * FROM tasks WHERE id=?", (article["id"],)).fetchone(), args) in (0, 1)
        assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 1

        pipeline = AgentPipeline(Factory(), Memory(), mode="auto", owns=False)
        pipeline.assess(db)
        pipeline.sync(db)
        assert len(pipeline.knowledge.doc_index) >= 1
        summary = export(db, folder)
        assert summary["unique_documents"] == 1
        text = (folder / "american_history.txt").read_text(encoding="utf-8")
        lowered = text.casefold()
        assert "http://" not in lowered and "https://" not in lowered
        assert "sha-256" not in lowered and "rights:" not in lowered and "pageid" not in lowered
        files = list((folder / "training_text" / "american_history").glob("*.txt"))
        assert len(files) == 1
        db.close()
    print("PASS: resume-safe dedupe, clean LANTRA exports, category discovery, and optional agent pipeline.")


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    requested = args.output_dir
    args.output_dir = Path(__file__).resolve().parent / "american_history"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "collector.log", encoding="utf-8")],
    )
    if requested is not None and requested.expanduser().resolve() != args.output_dir:
        LOG.warning("Ignoring --output-dir %s; output is fixed beside the script at %s", requested, args.output_dir)
    LOG.info("OUTPUT | %s", args.output_dir)

    pipeline = None
    db = None
    previous_cwd = Path.cwd()
    code = 0
    try:
        # --check-agents is a read-only SLAI diagnostic. It must not acquire the
        # corpus writer lock or open SQLite, because a live collector may already
        # own american_history/collector.lock. Running this diagnostic alongside
        # collection is safe: it creates its own temporary agent runtime only.
        if args.check_agents:
            try:
                root = find_slai_root(args.slai_root)
                os.chdir(root)
                pipeline = create_agents(root, args.agents)
                LOG.info(
                    "AGENTS | quality=%s knowledge=%s planning=%s reasoning=%s evaluation=%s",
                    bool(pipeline.quality), bool(pipeline.knowledge), bool(pipeline.plan_method),
                    bool(pipeline.reason_method), bool(pipeline.eval_method),
                )
                return 0
            except Exception as exc:
                if args.agents == "required":
                    raise
                LOG.error("AGENTS | diagnostic failed before collection state was touched (%s)", exc)
                return 1

        with collection_lock(args.output_dir):
            try:
                if args.agents != "off" and not args.export_only:
                    try:
                        root = find_slai_root(args.slai_root)
                        os.chdir(root)
                        pipeline = create_agents(root, args.agents)
                    except Exception as exc:
                        if args.agents == "required":
                            raise
                        LOG.warning("AGENTS | SLAI unavailable/incompatible; continuing standalone (%s)", exc)
                        pipeline = AgentPipeline(mode="off")
                else:
                    pipeline = AgentPipeline(mode="off")

                db = connect(args.output_dir / "history.sqlite3")
                if not args.export_only:
                    seed(db, args)
                    pipeline.assess(db, args.batch_size)
                    pipeline.sync(db, args.strict_quality)
                    if not args.process_only:
                        collect(db, APIClient(args.delay, args.contact, args.retries, args.timeout), args, pipeline)
                    pipeline.assess(db, args.batch_size)
                    pipeline.sync(db, args.strict_quality)
                export(db, args.output_dir, args.strict_quality)
            finally:
                if db is not None:
                    try:
                        export(db, args.output_dir, args.strict_quality)
                    finally:
                        db.close()
    except KeyboardInterrupt:
        LOG.warning("INTERRUPT | committed data preserved; run again to resume")
        code = 130
    except Exception:
        LOG.exception("Collector stopped; committed data/checkpoints remain available")
        code = 1
    finally:
        try:
            if pipeline is not None:
                pipeline.close()
        except Exception:
            LOG.exception("Agent shutdown reported an error")
            code = 1
        os.chdir(previous_cwd)
    return code


if __name__ == "__main__":
    sys.exit(main())
