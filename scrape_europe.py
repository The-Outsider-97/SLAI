#!/usr/bin/env python3
"""Continuously collect Europe-focused text for later LANTRA training.

Python 3.12 for SLAI agent mode. Run from the SLAI repository root or beside it.

Purpose
-------
Build a large, restart-safe European corpus covering history, politics and
public institutions, society and demography, economy, culture and religion,
law and institutions, conflict and diplomacy, and science and education.

The LANTRA-facing corpus contains ONLY:

    title

    source text

No URL, external ID, checksum, rights field, collection timestamp, agent score,
or other metadata is written into training TXT files. Minimal internal keys and
a normalized content digest are kept only in SQLite because exact resumption
and duplicate prevention are impossible to do reliably without persistent
state.

SLAI collaboration (default)
----------------------------
* PlanningAgent  - schedules a fair batch of pending discovery/fetch work.
* ReasoningAgent - resolves the primary topical domain from explicit evidence.
* QualityAgent   - assesses newly collected source text without rewriting it.
* KnowledgeAgent - indexes quality-accepted chunks for the collector's shared
                   knowledge context.

All four agents are created through AgentFactory and share one SharedMemory.
They are used only for distinct responsibilities; source text is never generated,
rewritten, summarized, or expanded by an agent.

Output directory (always beside this script)
--------------------------------------------
    ./europe/
        europe.txt
        history.txt
        politics_governance.txt
        society_demography.txt
        economy.txt
        culture_religion.txt
        law_institutions.txt
        conflict_diplomacy.txt
        science_education.txt
        training_text/<domain>/<sequence>.txt
        _state/collection.sqlite3
        _state/collector.log
        _state/summary.json

Typical run
-----------
    py -m scrape_europe

Stop with Ctrl+C. Re-run the same command to resume from committed SQLite state.
Use --help for testing and maintenance options.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import importlib
import importlib.util
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import random
import re
import signal
import sqlite3
import sys
import threading
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


LOG = logging.getLogger("europe_collector")
VERSION = "1.0.2"
SCHEMA = "europe-collector-v1"
WIKI = "https://en.wikipedia.org/w/api.php"
STOP = threading.Event()

DOMAIN_ORDER = (
    "history",
    "politics_governance",
    "society_demography",
    "economy",
    "culture_religion",
    "law_institutions",
    "conflict_diplomacy",
    "science_education",
)

# Deliberately descriptive rather than ideological. These signals classify topic,
# not viewpoint, party, policy quality, or political preference.
DOMAIN_PATTERNS = {
    "history": re.compile(
        r"\b(history|historical|prehistory|ancient|medieval|renaissance|early modern|"
        r"modern history|archaeolog|empire|kingdom|dynasty|revolution|industrialization|"
        r"industrialisation|chronology|heritage|historical period|centur(?:y|ies))\b", re.I
    ),
    "politics_governance": re.compile(
        r"\b(politics|political|government|governance|parliament|cabinet|president|"
        r"prime minister|election|electoral|political party|public administration|"
        r"local government|constitution|legislature|executive branch|state system)\b", re.I
    ),
    "society_demography": re.compile(
        r"\b(society|social|demograph|population|migration|immigration|emigration|"
        r"ethnic|minority|language|urbanization|urbanisation|rural|class|gender|"
        r"family|welfare|public health|labour|labor|diaspora|standard of living)\b", re.I
    ),
    "economy": re.compile(
        r"\b(economy|economic|industry|industrial|trade|commerce|finance|banking|"
        r"agriculture|energy|transport|infrastructure|tourism|currency|taxation|"
        r"employment|unemployment|market|gdp|business|manufacturing)\b", re.I
    ),
    "culture_religion": re.compile(
        r"\b(culture|cultural|religion|religious|christian|islam|jewish|judaism|"
        r"orthodox|catholic|protestant|art|architecture|literature|music|film|"
        r"folklore|tradition|festival|cuisine|heritage|museum|philosophy)\b", re.I
    ),
    "law_institutions": re.compile(
        r"\b(law|legal|judiciary|court|supreme court|constitutional court|justice|"
        r"civil law|criminal law|human rights|civil rights|treaty law|institution|"
        r"regulation|legislation|legal history|rule of law)\b", re.I
    ),
    "conflict_diplomacy": re.compile(
        r"\b(war|battle|conflict|military|army|navy|air force|occupation|invasion|"
        r"resistance|diplomacy|diplomatic|foreign relations|international relations|"
        r"alliance|treaty|peace process|security policy|cold war|world war)\b", re.I
    ),
    "science_education": re.compile(
        r"\b(science|scientific|technology|technological|research|university|"
        r"education|school|academy|innovation|engineering|medicine|scientist|"
        r"mathematics|physics|chemistry|biology|space programme|space program)\b", re.I
    ),
}

DOMAIN_SEEDS = {
    "history": [
        "History of Europe", "European history", "Prehistoric Europe", "Classical antiquity",
        "Middle Ages in Europe", "Early modern Europe", "Modern history of Europe",
        "European colonialism", "Industrial Revolution in Europe", "History of the European Union",
    ],
    "politics_governance": [
        "Politics of Europe", "Government in Europe", "European Union politics",
        "European integration", "Political systems of Europe", "Elections in Europe",
        "Public administration in Europe", "Local government in Europe",
    ],
    "society_demography": [
        "Demographics of Europe", "Society of Europe", "Migration in Europe",
        "Languages of Europe", "Ethnic groups in Europe", "Urbanization in Europe",
        "European social model", "Health in Europe", "Poverty in Europe",
    ],
    "economy": [
        "Economy of Europe", "Economic history of Europe", "European single market",
        "Trade in Europe", "Energy in Europe", "Transport in Europe", "Agriculture in Europe",
        "Industry in Europe", "Tourism in Europe", "Eurozone",
    ],
    "culture_religion": [
        "Culture of Europe", "Religion in Europe", "Art of Europe", "Architecture of Europe",
        "European literature", "Music of Europe", "Cinema of Europe", "European cuisine",
        "Cultural heritage of Europe", "Philosophy in Europe",
    ],
    "law_institutions": [
        "Law in Europe", "European Union law", "Council of Europe", "European Court of Human Rights",
        "Court of Justice of the European Union", "Constitutional law in Europe",
        "Human rights in Europe", "Legal history of Europe",
    ],
    "conflict_diplomacy": [
        "Military history of Europe", "Wars in Europe", "Diplomatic history of Europe",
        "International relations in Europe", "World War I in Europe", "World War II in Europe",
        "Cold War in Europe", "NATO relations with Europe", "European security",
    ],
    "science_education": [
        "Science and technology in Europe", "Education in Europe", "Universities in Europe",
        "European Research Area", "History of science in Europe", "European Space Agency",
        "Research in Europe", "Technology in Europe",
    ],
}

# Broad country/territory coverage. Transcontinental states are included because
# substantial parts of their history, institutions, and society belong to European
# history. The list is used for discovery/relevance, not for political recognition.
EUROPE_PLACES = (
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium",
    "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark",
    "Estonia", "Finland", "France", "Georgia", "Germany", "Greece", "Hungary", "Iceland",
    "Ireland", "Italy", "Kazakhstan", "Kosovo", "Latvia", "Liechtenstein", "Lithuania",
    "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands",
    "North Macedonia", "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino",
    "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine",
    "United Kingdom", "Vatican City", "Faroe Islands", "Gibraltar", "Guernsey", "Isle of Man",
    "Jersey", "Svalbard",
)

# Country-specific page/category templates broaden the corpus without encoding
# fragile page IDs. Missing titles are simply marked skipped and do not harm resume.
COUNTRY_TEMPLATES = {
    "history": (
        "History of {place}", "Military history of {place}", "Economic history of {place}",
        "Social history of {place}",
    ),
    "politics_governance": (
        "Politics of {place}", "Government of {place}", "Elections in {place}",
        "Political parties in {place}",
    ),
    "society_demography": (
        "Demographics of {place}", "Society of {place}", "Immigration to {place}",
        "Languages of {place}",
    ),
    "economy": (
        "Economy of {place}", "Industry in {place}", "Transport in {place}", "Energy in {place}",
    ),
    "culture_religion": (
        "Culture of {place}", "Religion in {place}", "Architecture of {place}", "Music of {place}",
    ),
    "law_institutions": (
        "Law of {place}", "Judiciary of {place}", "Constitution of {place}", "Human rights in {place}",
    ),
    "conflict_diplomacy": (
        "Foreign relations of {place}", "Armed Forces of {place}", "Wars involving {place}",
    ),
    "science_education": (
        "Education in {place}", "Science and technology in {place}", "Universities in {place}",
    ),
}

SEARCH_TERMS = {
    "history": ("history", "historical", "medieval", "modern history"),
    "politics_governance": ("politics", "government", "parliament", "elections"),
    "society_demography": ("society", "demographics", "migration", "population"),
    "economy": ("economy", "economic history", "industry", "trade"),
    "culture_religion": ("culture", "religion", "architecture", "literature"),
    "law_institutions": ("law", "constitution", "judiciary", "human rights"),
    "conflict_diplomacy": ("military history", "war", "foreign relations", "diplomacy"),
    "science_education": ("education", "science", "technology", "university"),
}

SKIP_CATEGORY = re.compile(
    r"\b(wikipedia|articles|pages|templates|stubs|maintenance|births|deaths|living people|"
    r"alumni|award winners|sportspeople|footballers|filmographies|discographies|"
    r"lists of people|people by occupation|surname|given name)\b", re.I
)

_place_terms = ["Europe", "European", "European Union", "Council of Europe", "Eurozone", "Schengen"]
_place_terms.extend(EUROPE_PLACES)
EUROPE_SIGNAL = re.compile(r"\b(?:" + "|".join(re.escape(x) for x in sorted(_place_terms, key=len, reverse=True)) + r")\b", re.I)


class Skip(Exception):
    """Permanent record/task failure."""


class Deferred(Exception):
    """Transient source failure. Committed work remains valid."""

    def __init__(self, message: str, seconds: float = 60.0):
        super().__init__(message)
        self.seconds = max(5.0, float(seconds))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    paragraphs = []
    for raw in re.split(r"\n\s*\n", text):
        line = " ".join(raw.split())
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs).strip()


def normalized_digest(text: str) -> str:
    """Internal-only duplicate key. Never exported into training text."""
    normalized = " ".join(clean(text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def words_in(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text, flags=re.UNICODE))


def domain_candidates(title: str, body: str, fallback: str) -> list[str]:
    sample = clean(title + "\n" + body[:16000])
    scored = []
    for domain, pattern in DOMAIN_PATTERNS.items():
        count = len(pattern.findall(sample))
        if count:
            scored.append((count, domain == fallback, domain))
    scored.sort(reverse=True)
    names = [d for _, _, d in scored[:4]]
    return names or [fallback]


def relevant_to_europe(title: str, body: str, *, seeded: bool = False) -> bool:
    if seeded:
        return True
    sample = title + "\n" + body[:16000]
    return bool(EUROPE_SIGNAL.search(sample))


def category_relevant(title: str, domain: str, depth: int) -> bool:
    if SKIP_CATEGORY.search(title):
        return False
    if depth <= 2:
        return True
    if EUROPE_SIGNAL.search(title):
        return True
    pattern = DOMAIN_PATTERNS.get(domain)
    return bool(pattern and pattern.search(title))


def pause(seconds: float) -> None:
    if STOP.wait(max(0.0, seconds)):
        raise KeyboardInterrupt


class RestrictedRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old, new = urlsplit(req.full_url), urlsplit(newurl)
        if new.scheme != "https" or new.hostname != "en.wikipedia.org" or new.path != "/w/api.php":
            raise Skip("Redirect outside the selected Wikipedia API")
        if old.hostname != new.hostname:
            raise Skip("Cross-host redirect refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class APIClient:
    def __init__(self, delay=1.0, contact="", retries=5, timeout=45.0):
        self.delay = float(delay)
        self.retries = int(retries)
        self.timeout = float(timeout)
        self.last = 0.0
        self.opener = build_opener(RestrictedRedirect())
        self.ua = f"LANTRAEuropeCollector/{VERSION}" + (f" ({contact})" if contact else "")

    def query(self, **params):
        params = dict(action="query", format="json", formatversion=2, maxlag=5, **params)
        url = WIKI + "?" + urlencode(params)
        failure = None
        for attempt in range(self.retries):
            pause(max(0.0, self.delay - (time.monotonic() - self.last)))
            backoff = min(90.0, 2 ** (attempt + 1) + random.random())
            try:
                self.last = time.monotonic()
                request = Request(
                    url,
                    headers={
                        "User-Agent": self.ua,
                        "Accept": "application/json",
                        "Accept-Encoding": "identity",
                    },
                )
                with self.opener.open(request, timeout=self.timeout) as response:
                    mime = response.headers.get_content_type()
                    if mime not in ("application/json", "text/json"):
                        raise Deferred(f"Unexpected API response type: {mime}")
                    raw = response.read(24 * 1024 * 1024 + 1)
                    if len(raw) > 24 * 1024 * 1024:
                        raise Skip("API response exceeds 24 MiB")
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise Deferred("Unexpected JSON response structure")
                if result.get("error"):
                    error = result["error"]
                    code = error.get("code") if isinstance(error, dict) else ""
                    if code in {"missingtitle", "invalidtitle", "nosuchpageid"}:
                        raise Skip(str(error))
                    raise Deferred(str(error))
                return result
            except HTTPError as exc:
                if exc.code in (401, 403):
                    raise Deferred(f"HTTP {exc.code}: source denied access", 3600) from exc
                if exc.code not in (429, 500, 502, 503, 504):
                    raise Skip(f"HTTP {exc.code}") from exc
                failure = exc
                retry = exc.headers.get("Retry-After", "")
                try:
                    backoff = max(backoff, float(retry))
                except ValueError:
                    try:
                        backoff = max(backoff, parsedate_to_datetime(retry).timestamp() - time.time())
                    except (TypeError, ValueError, OverflowError):
                        pass
                if not math.isfinite(backoff) or backoff > 900:
                    raise Deferred(f"HTTP {exc.code}", min(max(backoff, 60.0), 3600.0)) from exc
            except (URLError, TimeoutError, OSError, ValueError, Deferred) as exc:
                failure = exc
            if attempt + 1 < self.retries:
                LOG.warning("HTTP | retry=%d/%d | in=%.1fs | %s", attempt + 2, self.retries, backoff, failure)
                pause(backoff)
        raise Deferred(f"Request failed after {self.retries} attempts: {failure}", 300)


# ---------------------------------------------------------------------------
# Persistent collection state
# ---------------------------------------------------------------------------

def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tasks(
            id INTEGER PRIMARY KEY,
            domain TEXT NOT NULL,
            kind TEXT NOT NULL,
            target TEXT NOT NULL,
            depth INTEGER NOT NULL DEFAULT 0,
            payload TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
            due REAL NOT NULL DEFAULT 0,
            attempts INTEGER NOT NULL DEFAULT 0,
            error TEXT NOT NULL DEFAULT '',
            UNIQUE(domain, kind, target)
        );
        CREATE INDEX IF NOT EXISTS task_ready ON tasks(status, due, domain, kind, depth, id);
        CREATE TABLE IF NOT EXISTS documents(
            id INTEGER PRIMARY KEY,
            primary_domain TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            assessment TEXT NOT NULL DEFAULT '{}',
            quality_status TEXT NOT NULL DEFAULT 'pending',
            assessed INTEGER NOT NULL DEFAULT 0,
            collected_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS document_domain ON documents(primary_domain, id);
        CREATE INDEX IF NOT EXISTS document_assessed ON documents(assessed, id);
        CREATE TABLE IF NOT EXISTS origins(
            source_key TEXT PRIMARY KEY,
            document_id INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS aliases(
            title TEXT PRIMARY KEY,
            document_id INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS memberships(
            domain TEXT NOT NULL,
            document_id INTEGER NOT NULL,
            origin TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(domain, document_id)
        );
        CREATE TABLE IF NOT EXISTS exports(
            domain TEXT PRIMARY KEY,
            last_id INTEGER NOT NULL DEFAULT 0,
            bytes INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS cooldowns(
            source TEXT PRIMARY KEY,
            until REAL NOT NULL DEFAULT 0
        );
        """
    )
    columns = {row[1] for row in db.execute('PRAGMA table_info(documents)')}
    if 'quality_status' not in columns:
        db.execute("ALTER TABLE documents ADD COLUMN quality_status TEXT NOT NULL DEFAULT 'pending'")
    row = db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
    if row and row[0] != SCHEMA:
        db.close()
        raise RuntimeError(f"Incompatible collector schema: {row[0]!r}")
    with db:
        db.execute("INSERT OR IGNORE INTO meta VALUES('schema', ?)", (SCHEMA,))
    return db


def enqueue(db, domain: str, kind: str, target: str, depth=0, payload=None) -> bool:
    if domain not in DOMAIN_ORDER or kind not in {"article", "category", "search"}:
        return False
    before = db.total_changes
    db.execute(
        """INSERT OR IGNORE INTO tasks(domain,kind,target,depth,payload,status,due,attempts,error)
           VALUES(?,?,?,?,?,'pending',0,0,'')""",
        (domain, kind, clean(target), int(depth), json.dumps(payload or {}, ensure_ascii=False)),
    )
    return db.total_changes > before


def seed(db, args) -> None:
    """Seed broad European discovery without touching already completed work."""
    with db:
        # Transient failures are eligible on restart. Permanent skips stay skipped
        # unless explicitly requested by the user.
        db.execute("UPDATE tasks SET status='pending',due=0,error='' WHERE status='error'")
        if args.retry_skipped:
            db.execute("UPDATE tasks SET status='pending',due=0,error='' WHERE status='skipped'")
        if args.recheck_quality:
            db.execute("UPDATE documents SET assessed=0,assessment='{}',quality_status='pending'")
            db.execute("INSERT INTO meta VALUES('exports_dirty','1') ON CONFLICT(key) DO UPDATE SET value='1'")

        for domain in DOMAIN_ORDER:
            for title in DOMAIN_SEEDS[domain]:
                enqueue(db, domain, "article", title, payload={"origin": "europe_topic_seed", "seeded": True})
                enqueue(db, domain, "category", "Category:" + title, 0, {"origin": "europe_topic_seed"})

            for place in EUROPE_PLACES:
                for template in COUNTRY_TEMPLATES[domain]:
                    title = template.format(place=place)
                    enqueue(db, domain, "article", title, payload={"origin": "place_topic_seed", "seeded": True})
                    # Category names often mirror overview titles. Missing categories are harmless.
                    enqueue(db, domain, "category", "Category:" + title, 0, {"origin": "place_topic_seed"})

                # Search supplies breadth where category naming is irregular.
                for term in SEARCH_TERMS[domain]:
                    query = f'"{place}" {term}'
                    enqueue(db, domain, "search", query, 0, {"origin": "place_search"})

            # Broad Europe searches catch cross-border institutions/events.
            for term in SEARCH_TERMS[domain]:
                enqueue(db, domain, "search", f'Europe {term}', 0, {"origin": "europe_search"})


def task_done(db, task) -> None:
    db.execute("UPDATE tasks SET status='done',attempts=0,error='' WHERE id=?", (task["id"],))


def defer_task(db, task, exc: Deferred) -> None:
    wait = max(exc.seconds, min(1800.0, 30.0 * 2 ** min(int(task["attempts"]), 6)))
    db.execute(
        "UPDATE tasks SET status='pending',due=?,attempts=attempts+1,error=? WHERE id=?",
        (time.time() + wait, str(exc), task["id"]),
    )
    db.execute(
        "INSERT INTO cooldowns VALUES('wikipedia', ?) ON CONFLICT(source) DO UPDATE SET until=excluded.until",
        (time.time() + wait,),
    )


def ingest(
    db,
    domain: str,
    source_key: str,
    title: str,
    body: str,
    origin: str,
    args,
) -> tuple[int | None, bool]:
    title, body = clean(title), clean(body)
    if not title or not body:
        return None, False
    if words_in(body) < args.min_words:
        return None, False
    if len(body) and sum(ch.isalpha() for ch in body) / len(body) < 0.35:
        return None, False
    if re.search(r"\b(may refer to:|this disambiguation page)\b", body[:1200], re.I):
        return None, False

    prior = db.execute("SELECT document_id FROM origins WHERE source_key=?", (source_key,)).fetchone()
    if prior:
        doc_id = int(prior[0])
        db.execute("INSERT OR IGNORE INTO memberships VALUES(?,?,?)", (domain, doc_id, origin))
        return doc_id, False

    content_hash = normalized_digest(body)
    duplicate = db.execute("SELECT id FROM documents WHERE content_hash=?", (content_hash,)).fetchone()
    if duplicate:
        doc_id = int(duplicate[0])
        created = False
    else:
        doc_id = db.execute(
            """INSERT INTO documents(primary_domain,title,body,content_hash,assessment,quality_status,assessed,collected_at)
               VALUES(?,?,?,?, '{}', 'pending', 0, ?)""",
            (domain, title, body, content_hash, now()),
        ).lastrowid
        created = True
        LOG.info("SAVED | %s | %s | %s words", domain, title[:100], f"{words_in(body):,}")

    db.execute("INSERT OR IGNORE INTO origins VALUES(?,?)", (source_key, doc_id))
    db.execute("INSERT OR IGNORE INTO aliases VALUES(?,?)", (title, doc_id))
    db.execute("INSERT OR IGNORE INTO memberships VALUES(?,?,?)", (domain, doc_id, origin))
    return int(doc_id), created


# ---------------------------------------------------------------------------
# Wikipedia discovery/fetch work
# ---------------------------------------------------------------------------

def category_task(db, api: APIClient, task, args) -> int:
    payload = json.loads(task["payload"] or "{}")
    params = dict(
        list="categorymembers",
        cmtitle=task["target"],
        cmtype="page|subcat",
        cmnamespace="0|14",
        cmlimit=500,
    )
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
            ns = member.get("ns")
            if ns == 0:
                discovered += int(enqueue(
                    db, task["domain"], "article", title, task["depth"],
                    {"origin": task["target"], "seeded": False},
                ))
            elif ns == 14:
                child_depth = int(task["depth"]) + 1
                if args.depth and child_depth > args.depth:
                    continue
                if not category_relevant(title, task["domain"], child_depth):
                    continue
                discovered += int(enqueue(
                    db, task["domain"], "category", title, child_depth,
                    {"origin": task["target"]},
                ))

        continuation = data.get("continue")
        if continuation:
            new_payload = dict(payload)
            new_payload["continue"] = continuation
            if new_payload == payload:
                raise Deferred("Repeated category continuation cursor")
            db.execute("UPDATE tasks SET payload=? WHERE id=?", (json.dumps(new_payload), task["id"]))
        else:
            task_done(db, task)
    return discovered


def search_task(db, api: APIClient, task, args) -> int:
    payload = json.loads(task["payload"] or "{}")
    params = dict(
        list="search",
        srsearch=task["target"],
        srnamespace=0,
        srlimit=50,
        srwhat="text",
    )
    if "sroffset" in payload:
        params["sroffset"] = int(payload["sroffset"])
    data = api.query(**params)
    results = data.get("query", {}).get("search")
    if not isinstance(results, list):
        raise Deferred("Search response lacks results")

    discovered = 0
    with db:
        for result in results:
            title = clean(result.get("title", ""))
            if not title:
                continue
            discovered += int(enqueue(
                db, task["domain"], "article", title, 0,
                {"origin": "search:" + task["target"], "seeded": False},
            ))

        continuation = data.get("continue", {})
        next_offset = continuation.get("sroffset") if isinstance(continuation, dict) else None
        if isinstance(next_offset, int) and next_offset > int(payload.get("sroffset", -1)):
            # Bound a single query to avoid one broad search starving all domains.
            if next_offset < args.search_result_limit:
                payload["sroffset"] = next_offset
                db.execute("UPDATE tasks SET payload=? WHERE id=?", (json.dumps(payload), task["id"]))
            else:
                task_done(db, task)
        else:
            task_done(db, task)
    return discovered


def article_task(db, api: APIClient, task, args, team) -> int:
    known = db.execute("SELECT document_id FROM aliases WHERE title=?", (task["target"],)).fetchone()
    if known:
        payload = json.loads(task["payload"] or "{}")
        with db:
            db.execute(
                "INSERT OR IGNORE INTO memberships VALUES(?,?,?)",
                (task["domain"], int(known[0]), payload.get("origin", "alias")),
            )
            task_done(db, task)
        return 0

    data = api.query(
        titles=task["target"],
        redirects=1,
        prop="extracts|pageprops",
        explaintext=1,
        exsectionformat="plain",
        exlimit=1,
    )
    pages = data.get("query", {}).get("pages")
    if not isinstance(pages, list) or not pages:
        raise Deferred("Missing Wikipedia page response")
    page = pages[0]
    if page.get("missing") or page.get("invalid") or "disambiguation" in page.get("pageprops", {}):
        raise Skip("Missing/invalid/disambiguation page")
    pageid = page.get("pageid")
    if not isinstance(pageid, int):
        raise Deferred("Wikipedia page lacks a stable page identifier")
    title = page.get("title", task["target"])
    body = page.get("extract", "")
    if not str(body).strip():
        raise Skip("No article text")

    payload = json.loads(task["payload"] or "{}")
    if not relevant_to_europe(title, body, seeded=bool(payload.get("seeded"))):
        raise Skip("Article lacks a European relevance signal")

    candidates = domain_candidates(title, body, task["domain"])
    chosen = team.choose_domain(candidates, task["domain"])
    with db:
        doc_id, created = ingest(
            db,
            chosen,
            f"wikipedia:{pageid}",
            title,
            body,
            payload.get("origin", "discovery"),
            args,
        )
        if doc_id:
            db.execute("INSERT OR IGNORE INTO aliases VALUES(?,?)", (task["target"], doc_id))
            # Preserve discovery-domain membership even if Reasoning selects a better
            # primary domain. This is internal only and never duplicates training text.
            db.execute(
                "INSERT OR IGNORE INTO memberships VALUES(?,?,?)",
                (task["domain"], doc_id, payload.get("origin", "discovery")),
            )
        task_done(db, task)
    return int(created)


def ready_candidates(db, args) -> list[sqlite3.Row]:
    timestamp = time.time()
    cooldown = db.execute("SELECT until FROM cooldowns WHERE source='wikipedia'").fetchone()
    if cooldown and float(cooldown[0]) > timestamp:
        return []

    rows = []
    for domain in DOMAIN_ORDER:
        depth_sql = "" if args.depth == 0 else " AND (kind!='category' OR depth<=?)"
        params = [domain, timestamp]
        if args.depth:
            params.append(args.depth)
        row = db.execute(
            "SELECT * FROM tasks WHERE domain=? AND status='pending' AND due<=?" + depth_sql +
            " ORDER BY CASE kind WHEN 'article' THEN 0 WHEN 'category' THEN 1 ELSE 2 END, depth, id LIMIT 1",
            tuple(params),
        ).fetchone()
        if row:
            rows.append(row)

    counts = dict(db.execute("SELECT primary_domain,count(*) FROM documents GROUP BY primary_domain"))
    rows.sort(key=lambda r: (counts.get(r["domain"], 0), r["id"]))
    return rows


def reopen_discovery(db) -> int:
    with db:
        count = db.execute(
            "UPDATE tasks SET status='pending',payload='{}',due=0,attempts=0,error='' "
            "WHERE kind IN ('category','search') AND status='done'"
        ).rowcount
        db.execute(
            "INSERT INTO meta VALUES('last_refresh',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (now(),),
        )
    return count


# ---------------------------------------------------------------------------
# SLAI agent collaboration
# ---------------------------------------------------------------------------

def europe_domain_rules(facts):
    inferred = {}
    for fact, confidence in list(facts.items()):
        if not isinstance(fact, tuple) or len(fact) != 3 or fact[0] != "europe_candidate":
            continue
        _, predicate, domain = fact
        if domain not in DOMAIN_ORDER:
            continue
        weights = {
            "strong_text_signal": 0.98,
            "text_signal": 0.90,
            "weak_text_signal": 0.82,
            "candidate_signal": 0.74,
            "discovery_domain": 0.78,
            "fallback_domain": 0.65,
        }
        weight = weights.get(predicate)
        if weight is None:
            continue
        key = ("europe_candidate", "selected_domain", domain)
        inferred[key] = max(inferred.get(key, 0.0), float(confidence) * weight)
    return inferred


def find_slai_root(explicit=None) -> Path:
    candidates = [explicit.expanduser().resolve()] if explicit else []
    if not explicit:
        for base in (Path.cwd(), Path(__file__).resolve().parent):
            candidates.extend((base, *base.parents))
    seen = set()
    for root in candidates:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "src/agents/agent_factory.py").is_file():
            return resolved
    raise RuntimeError("SLAI root not found. Run inside SLAI or pass --slai-root PATH")



def verify_python_runtime() -> None:
    """Require Python 3.12 without changing or re-executing environments.

    The collector respects the interpreter that launched it. This allows a separate
    Python 3.12 environment to run the collector while another SLAI process, such as
    LANTRA training, continues using a different repository-local environment.
    """
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "Europe collector agent mode requires Python 3.12 for this SLAI build. "
            f"Current interpreter: Python {sys.version_info.major}.{sys.version_info.minor} "
            f"at {sys.executable}. Launch explicitly with a Python 3.12 environment, "
            "for example: .\\venv312\\Scripts\\python.exe .\\scrape_europe.py"
        )
    LOG.info("RUNTIME | Python 3.12 confirmed | interpreter=%s", sys.executable)


def _numpy_binary_tags() -> list[str]:
    """Inspect NumPy extension filenames without importing NumPy."""
    try:
        spec = importlib.util.find_spec("numpy")
    except Exception:
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    package = Path(next(iter(spec.submodule_search_locations)))
    core = package / "_core"
    if not core.is_dir():
        return []
    tags = []
    for pattern in ("*.pyd", "*.so"):
        for path in core.glob(pattern):
            match = re.search(r"\.(cp\d{2,3})[-.]", path.name, re.I)
            if match:
                tags.append(match.group(1).lower())
    return sorted(set(tags))


def verify_agent_runtime_dependencies() -> None:
    """Fail before AgentFactory when SLAI's runtime/dependencies are ABI-incompatible."""
    expected = f"cp{sys.version_info.major}{sys.version_info.minor}"
    discovered = _numpy_binary_tags()

    incompatible = [tag for tag in discovered if tag != expected]
    if discovered and expected not in discovered and incompatible:
        raise RuntimeError(
            "SLAI venv ABI mismatch: the active interpreter is "
            f"Python {sys.version_info.major}.{sys.version_info.minor} ({expected}), "
            f"but NumPy contains compiled extension(s) for {', '.join(incompatible)}. "
            "The collector will not modify your environment. Repair/reinstall the venv "
            "dependencies for the active Python before using SLAI agents. "
            f"Interpreter: {sys.executable}"
        )

    failures = []
    for module_name in ("yaml", "numpy"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError(
            "SLAI agent dependency preflight failed before AgentFactory initialization. "
            f"Interpreter: {sys.executable}. " + " | ".join(failures)
        )

class Team:
    """Four distinct SLAI agents sharing one runtime and one SharedMemory."""

    def __init__(self, root: Path | None, enabled=True):
        global LOG
        self.enabled = bool(enabled)
        self.root = root
        self.factory = None
        self.memory = None
        self.planning = None
        self.reasoning = None
        self.quality = None
        self.knowledge = None
        self.Task = None
        self.TaskType = None
        self.Resources = None
        self.rounds = 0
        self.indexed_documents = set()

        if not self.enabled:
            return
        if root is None:
            raise RuntimeError("Agent mode requires an SLAI root")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from src.agents.agent_factory import AgentFactory
        from src.agents.collaborative.shared_memory import SharedMemory
        from src.agents.planning.planning_types import Task, TaskType, ResourceProfile
        from logs.logger import get_logger

        LOG = get_logger("europe_collector")
        self.Task, self.TaskType, self.Resources = Task, TaskType, ResourceProfile
        self.memory = SharedMemory()
        try:
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
                    "source": "europe_collector",
                    "directory_path": "",
                    "retrieval_mode": "tfidf",
                    "bias_detection_enabled": False,
                    "use_ontology_expansion": False,
                },
            )

            required = [
                (self.planning, ("generate_plan",)),
                (self.reasoning, ("add_fact", "add_rule", "forward_chaining", "forget_by_subject")),
                (self.quality, ("evaluate_batch",)),
                (self.knowledge, ("add_document",)),
            ]
            for agent, methods in required:
                missing = [name for name in methods if not callable(getattr(agent, name, None))]
                if missing:
                    raise RuntimeError(f"{type(agent).__name__} lacks required methods: {', '.join(missing)}")
            if not isinstance(getattr(self.knowledge, "doc_index", None), dict):
                raise RuntimeError("KnowledgeAgent lacks the expected doc_index acknowledgement")
            self.reasoning.add_rule(europe_domain_rules, rule_name="europe_domain_evidence", weight=1.0)
        except BaseException:
            self.close()
            raise

        LOG.info("AGENTS | Planning + Reasoning + Quality + Knowledge active through AgentFactory")

    def choose_domain(self, candidates: list[str], fallback: str) -> str:
        candidates = [d for d in dict.fromkeys(candidates) if d in DOMAIN_ORDER]
        if fallback not in DOMAIN_ORDER:
            fallback = "history"
        if not self.enabled:
            return candidates[0] if candidates else fallback

        subject = "europe_candidate"
        self.reasoning.forget_by_subject(subject)
        try:
            evidence_names = ("strong_text_signal", "text_signal", "weak_text_signal", "candidate_signal")
            for index, domain in enumerate(candidates):
                predicate = evidence_names[min(index, len(evidence_names) - 1)]
                if not self.reasoning.add_fact((subject, predicate, domain), publish=False):
                    raise RuntimeError("ReasoningAgent rejected text-domain evidence")
            self.reasoning.add_fact((subject, "discovery_domain", fallback), publish=False)
            self.reasoning.add_fact((subject, "fallback_domain", fallback), publish=False)
            self.reasoning.forward_chaining(max_iterations=3)
            scores = {
                domain: self.reasoning.knowledge_base.get((subject, "selected_domain", domain), 0)
                for domain in DOMAIN_ORDER
            }
            chosen = max(DOMAIN_ORDER, key=lambda d: (scores[d], d == fallback))
            if not scores[chosen]:
                raise RuntimeError("ReasoningAgent produced no topical conclusion")
            LOG.info(
                "REASONING | domain=%s | fallback=%s | candidates=%s | evidence=%.2f",
                chosen, fallback, candidates, scores[chosen],
            )
            return chosen
        finally:
            self.reasoning.forget_by_subject(subject)

    def order(self, rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
        if not self.enabled or len(rows) < 2:
            return rows
        self.rounds += 1
        deadline = time.time() + 86400
        tasks = []
        for row in rows:
            duration = 15 if row["kind"] == "article" else 25
            tasks.append(self.Task(
                name=f"collect_{row['id']}",
                id=f"collect_{row['id']}",
                task_type=self.TaskType.PRIMITIVE,
                preconditions=[],
                effects=[],
                resource_requirements=self.Resources(gpu=0, ram=0.05),
                duration=duration,
                deadline=deadline,
                dependencies=[],
            ))
        goal = self.Task(
            name=f"europe_collection_round_{self.rounds}",
            task_type=self.TaskType.ABSTRACT,
            methods=[tasks],
            resource_requirements=self.Resources(gpu=0, ram=0),
            duration=25 * len(tasks),
            deadline=deadline,
        )
        plan = self.planning.generate_plan(goal)
        if not plan:
            raise RuntimeError("PlanningAgent could not schedule collection work")
        mapping = {f"collect_{row['id']}": row for row in rows}
        ids = [task.id for task in plan]
        if len(ids) != len(mapping) or set(ids) != set(mapping):
            raise RuntimeError("PlanningAgent returned an incomplete or unrecognized plan")
        LOG.info("PLANNING | scheduled=%d | domains=%s", len(ids), [mapping[x]["domain"] for x in ids])
        return [mapping[x] for x in ids]

    def assess(self, db, batch_size=8) -> int:
        """Assess up to ``batch_size`` documents individually for precise QA state."""
        if not self.enabled:
            return 0
        rows = db.execute(
            "SELECT * FROM documents WHERE assessed=0 ORDER BY id LIMIT ?",
            (int(batch_size),),
        ).fetchall()
        if not rows:
            return 0

        schema = {
            "schema_version": SCHEMA,
            "required_fields": ["id", "title", "text", "source_id", "source_type", "collected_at", "word_count"],
            "fields": {
                "id": {"type": "str", "required": True},
                "title": {"type": "str", "required": True},
                "text": {"type": "str", "required": True},
                "source_id": {"type": "str", "required": True},
                "source_type": {"type": "str", "required": True},
                "collected_at": {"type": "str", "required": True},
                "word_count": {"type": "int", "required": True},
            },
        }
        processed = 0
        blocked = False
        for row in rows:
            if STOP.is_set():
                raise KeyboardInterrupt
            record = {
                "id": str(row["id"]),
                "title": row["title"],
                "text": row["body"][:16000],
                "source_id": "wikipedia",
                "source_type": "europe_reference_text",
                "collected_at": row["collected_at"],
                "word_count": words_in(row["body"]),
            }
            checksum = row["content_hash"]
            result = self.quality.evaluate_batch(
                [record],
                dataset_id="europe_collector",
                source_id="wikipedia",
                batch_id="europe-" + checksum[:24],
                schema=schema,
                use_case="knowledge_ingestion",
                feature_fields=["word_count"],
                provenance={
                    "source_id": "wikipedia",
                    "source_type": "europe_reference_text",
                    "collected_at": row["collected_at"],
                    "collector": "europe_collector",
                    "checksum": checksum,
                    "uri": "",
                    "schema_version": SCHEMA,
                },
                source_metadata={"source_id": "wikipedia", "source_type": "europe_reference_text"},
                context={
                    "route": "planning->reasoning->quality->knowledge",
                    "external_content": True,
                    "content_type": "europe_reference_text",
                    "fact_verification_performed": False,
                },
            )
            if not isinstance(result, dict) or result.get("verdict") not in {"pass", "warn", "block"}:
                raise RuntimeError("QualityAgent returned no valid verdict")
            verdict = result["verdict"]
            if result.get("quarantine_count") or result.get("quarantine_entries"):
                verdict = "block"
            stored = dict(result)
            stored["verdict"] = verdict
            stored["assessment_scope"] = "first_16000_characters"
            with db:
                db.execute(
                    "UPDATE documents SET assessment=?,quality_status=?,assessed=1 WHERE id=?",
                    (json.dumps(stored, ensure_ascii=False, default=str), verdict, row["id"]),
                )
            blocked = blocked or verdict == "block"
            processed += 1
            LOG.info(
                "QUALITY | id=%s | %s | verdict=%s | score=%s",
                row["id"], row["title"][:70], verdict, result.get("batch_score"),
            )

        if blocked:
            with db:
                db.execute(
                    "INSERT INTO meta VALUES('exports_dirty','1') "
                    "ON CONFLICT(key) DO UPDATE SET value='1'"
                )
        return processed

    def sync_knowledge(self, db, max_documents=16) -> int:
        if not self.enabled:
            return 0
        added = 0
        rows = db.execute(
            "SELECT * FROM documents WHERE assessed=1 ORDER BY id"
        ).fetchall()
        for row in rows:
            if row["id"] in self.indexed_documents:
                continue
            if row["quality_status"] == "block":
                self.indexed_documents.add(row["id"])
                continue
            words = row["body"].split()
            for start in range(0, len(words), 1000):
                part = start // 1000
                text = row["title"] + "\n\n" + " ".join(words[start:start + 1000])
                doc_id = f"europe:{row['id']}:{part}"
                existing = self.knowledge.doc_index.get(doc_id)
                if existing is None:
                    self.knowledge.add_document(
                        text=text,
                        doc_id=doc_id,
                        metadata={
                            "domain": row["primary_domain"],
                            "chunk": part,
                            "trust": "external_reference_content",
                        },
                    )
                    existing = self.knowledge.doc_index.get(doc_id)
                if not isinstance(existing, dict):
                    raise RuntimeError("KnowledgeAgent did not acknowledge collected text")
            self.indexed_documents.add(row["id"])
            added += 1
            if added >= max_documents:
                break
        if added:
            LOG.info("KNOWLEDGE | indexed=%d new documents", added)
        return added

    def close(self):
        factory, memory = self.factory, self.memory
        self.factory = self.memory = None
        try:
            if factory is not None:
                factory.shutdown()
        finally:
            if memory is not None:
                memory.close()


# ---------------------------------------------------------------------------
# Content-only exports
# ---------------------------------------------------------------------------

@contextmanager
def atomic_writer(path: Path):
    temp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def collection_lock(folder: Path):
    lock_path = folder / "collector.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
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
            raise RuntimeError("Another Europe collector is already using this output directory") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _safe_training_filename(row_id: int) -> str:
    return f"{int(row_id):09d}.txt"


def export_content(db, folder: Path, rebuild=False) -> None:
    """Export each unique non-blocked document as title + body only."""
    dirty = db.execute("SELECT value FROM meta WHERE key='exports_dirty'").fetchone()
    rebuild = bool(rebuild or (dirty and dirty[0] == '1'))
    if rebuild:
        with db:
            db.execute("DELETE FROM exports")
        for path in (folder / "training_text").glob("*/*.txt") if (folder / "training_text").exists() else ():
            path.unlink(missing_ok=True)
        for name in ("europe", *DOMAIN_ORDER):
            (folder / f"{name}.txt").unlink(missing_ok=True)
        with db:
            db.execute("INSERT INTO meta VALUES('exports_dirty','0') ON CONFLICT(key) DO UPDATE SET value='0'")

    # Global aggregate.
    _append_export(db, folder / "europe.txt", "__all__", None, folder)
    # Primary-domain aggregates. A document belongs to only one primary domain, so
    # concatenating these domain files does not duplicate content.
    for domain in DOMAIN_ORDER:
        _append_export(db, folder / f"{domain}.txt", domain, domain, folder)


def _append_export(db, path: Path, export_key: str, domain: str | None, folder: Path) -> None:
    checkpoint = db.execute("SELECT last_id,bytes FROM exports WHERE domain=?", (export_key,)).fetchone()
    last_id, byte_count = tuple(checkpoint) if checkpoint else (0, 0)
    if not path.exists() or path.stat().st_size < byte_count:
        last_id, byte_count = 0, 0

    where = "id>? AND quality_status!='block'" if domain is None else "id>? AND primary_domain=? AND quality_status!='block'"
    params = (last_id,) if domain is None else (last_id, domain)
    with path.open("r+b" if path.exists() else "w+b") as aggregate:
        aggregate.truncate(byte_count)
        aggregate.seek(byte_count)
        for row in db.execute(f"SELECT * FROM documents WHERE {where} ORDER BY id", params):
            text = row["title"] + "\n\n" + row["body"] + "\n\n\n"
            aggregate.write(text.encode("utf-8"))
            if domain is not None:
                destination = folder / "training_text" / domain / _safe_training_filename(row["id"])
                if not destination.exists():
                    with atomic_writer(destination) as stream:
                        stream.write(row["title"] + "\n\n" + row["body"] + "\n")
            last_id = row["id"]
        aggregate.flush()
        os.fsync(aggregate.fileno())
        byte_count = aggregate.tell()

    with db:
        db.execute(
            "INSERT INTO exports(domain,last_id,bytes) VALUES(?,?,?) "
            "ON CONFLICT(domain) DO UPDATE SET last_id=excluded.last_id,bytes=excluded.bytes",
            (export_key, last_id, byte_count),
        )


def write_summary(db, folder: Path, steps: int) -> None:
    docs = dict(db.execute("SELECT primary_domain,count(*) FROM documents GROUP BY primary_domain"))
    statuses = Counter(row["quality_status"] for row in db.execute("SELECT quality_status FROM documents"))
    payload = {
        "updated_at": now(),
        "version": VERSION,
        "steps_this_run": steps,
        "unique_documents": db.execute("SELECT count(*) FROM documents").fetchone()[0],
        "documents_by_primary_domain": docs,
        "assessment_statuses": dict(statuses),
        "task_statuses": dict(db.execute("SELECT status,count(*) FROM tasks GROUP BY status")),
        "pending_tasks": db.execute("SELECT count(*) FROM tasks WHERE status='pending'").fetchone()[0],
    }
    with atomic_writer(folder / "_state" / "summary.json") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    LOG.info(
        "PROGRESS | steps=%d | documents=%d | pending=%d | domains=%s",
        steps,
        payload["unique_documents"],
        payload["pending_tasks"],
        docs,
    )


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def collect(db, api: APIClient, args, team: Team, folder: Path) -> None:
    steps = 0
    new_docs = 0
    last_status = 0.0
    next_refresh = time.monotonic() + args.refresh_hours * 3600
    agent_retry = 0.0

    while not STOP.is_set():
        if args.max_steps and steps >= args.max_steps:
            LOG.info("STOP | --max-steps reached (%d)", args.max_steps)
            return

        rows = ready_candidates(db, args)
        if rows:
            try:
                rows = team.order(rows)
            except Exception:
                LOG.exception("PLANNING | scheduling failed; preserving queue and retrying")
                pause(60)
                continue

            for task in rows:
                if STOP.is_set() or (args.max_steps and steps >= args.max_steps):
                    break
                steps += 1
                LOG.info(
                    "FETCH | step=%d | domain=%s | kind=%s | target=%s",
                    steps, task["domain"], task["kind"], task["target"][:120],
                )
                try:
                    if task["kind"] == "article":
                        new_docs += article_task(db, api, task, args, team)
                    elif task["kind"] == "category":
                        discovered = category_task(db, api, task, args)
                        LOG.info("DISCOVER | category | new_tasks=%d", discovered)
                    elif task["kind"] == "search":
                        discovered = search_task(db, api, task, args)
                        LOG.info("DISCOVER | search | new_tasks=%d", discovered)
                    else:
                        raise Skip("Unknown task kind")
                except Skip as exc:
                    with db:
                        db.execute("UPDATE tasks SET status='skipped',error=? WHERE id=?", (str(exc), task["id"]))
                    LOG.warning("SKIP | %s | %s", task["target"], exc)
                except Deferred as exc:
                    with db:
                        defer_task(db, task, exc)
                    LOG.warning("DEFER | wikipedia | %.0fs | %s", exc.seconds, exc)
                    break
                except Exception as exc:
                    with db:
                        db.execute(
                            "UPDATE tasks SET status='error',due=?,attempts=attempts+1,error=? WHERE id=?",
                            (time.time() + 300, str(exc), task["id"]),
                        )
                    LOG.exception("TASK | retry on restart/next cycle; committed corpus preserved")

                if steps % args.checkpoint_every == 0:
                    if time.time() >= agent_retry:
                        try:
                            team.assess(db, args.batch_size)
                            team.sync_knowledge(db, args.knowledge_sync_batch)
                        except Exception:
                            agent_retry = time.time() + 300
                            LOG.exception("AGENTS | quality/indexing delayed; source corpus preserved")
                    export_content(db, folder)
                    write_summary(db, folder, steps)
                    LOG.info("CHECKPOINT | steps=%d | new_docs=%d", steps, new_docs)
        else:
            # No source work is ready; use the idle period for agent processing.
            if time.time() >= agent_retry:
                try:
                    assessed = team.assess(db, args.batch_size)
                    indexed = team.sync_knowledge(db, args.knowledge_sync_batch)
                    if assessed or indexed:
                        export_content(db, folder)
                except Exception:
                    agent_retry = time.time() + 300
                    LOG.exception("AGENTS | idle processing delayed; corpus preserved")

            cooldown = db.execute("SELECT until FROM cooldowns WHERE source='wikipedia'").fetchone()
            if cooldown and float(cooldown[0]) <= time.time():
                with db:
                    db.execute("DELETE FROM cooldowns WHERE source='wikipedia'")

            pending_ready = db.execute(
                "SELECT count(*) FROM tasks WHERE status='pending' AND due<=?",
                (time.time(),),
            ).fetchone()[0]
            if pending_ready == 0 and args.continuous and time.monotonic() >= next_refresh:
                reopened = reopen_discovery(db)
                next_refresh = time.monotonic() + args.refresh_hours * 3600
                LOG.info("REFRESH | reopened=%d completed discovery tasks", reopened)
                if reopened:
                    continue
            if pending_ready == 0 and not args.continuous:
                LOG.info("DONE | no ready tasks remain within configured bounds")
                return
            pause(args.idle_seconds)

        if time.time() - last_status >= 60:
            write_summary(db, folder, steps)
            last_status = time.time()


# ---------------------------------------------------------------------------
# CLI and offline validation
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", choices=("team", "off"), default="team",
                        help="team: require all four SLAI agents (default); off: standalone maintenance/testing")
    parser.add_argument("--slai-root", type=Path)
    parser.add_argument("--depth", type=int, default=0, help="Category depth; 0 = unlimited with relevance guards")
    parser.add_argument("--search-result-limit", type=int, default=500,
                        help="Maximum Wikipedia search offset per discovery query")
    parser.add_argument("--min-words", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--knowledge-sync-batch", type=int, default=16)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--contact", default="")
    parser.add_argument("--idle-seconds", type=float, default=20.0)
    parser.add_argument("--refresh-hours", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=0, help="Testing bound; 0 = run until stopped")
    parser.add_argument("--retry-skipped", action="store_true")
    parser.add_argument("--recheck-quality", action="store_true")
    parser.add_argument("--process-only", action="store_true", help="Agent processing/export only; no network")
    parser.add_argument("--export-only", action="store_true", help="Rebuild content exports; no agents/network")
    parser.add_argument("--rebuild-exports", action="store_true")
    parser.add_argument("--check-agents", action="store_true")
    parser.add_argument("--no-continuous", dest="continuous", action="store_false",
                        help="Exit when the current discovery graph has no ready tasks")
    parser.set_defaults(continuous=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    for field in ("depth", "max_steps"):
        if getattr(args, field) < 0:
            parser.error(field + " must be nonnegative")
    for field in ("search_result_limit", "min_words", "checkpoint_every", "batch_size", "knowledge_sync_batch", "retries"):
        if getattr(args, field) < 1:
            parser.error(field + " must be positive")
    for field in ("delay", "timeout", "idle_seconds", "refresh_hours"):
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0:
            parser.error(field + " must be finite and positive")
    if args.delay < 1.0:
        parser.error("--delay must be at least 1 second")
    if any(ord(c) < 32 or ord(c) > 126 for c in args.contact):
        parser.error("--contact must contain printable ASCII only")
    if args.check_agents and (args.agents == "off" or args.export_only):
        parser.error("--check-agents requires --agents team and cannot accompany --export-only")
    if args.process_only and args.export_only:
        parser.error("--process-only and --export-only are mutually exclusive")
    return args


def self_test() -> None:
    import tempfile

    class FakeAPI:
        def query(self, **params):
            if params.get("list") == "categorymembers":
                return {
                    "query": {"categorymembers": [
                        {"ns": 0, "title": "European integration"},
                        {"ns": 14, "title": "Category:Political history of Europe"},
                    ]}
                }
            if params.get("list") == "search":
                return {"query": {"search": [{"title": "European integration"}]}}
            return {
                "query": {"pages": [{
                    "pageid": 101,
                    "title": "European integration",
                    "extract": (
                        "European integration describes historical and political cooperation in Europe, "
                        "including institutions, treaties, government, society, economic coordination, "
                        "law, diplomacy, education and cultural exchange. "
                    ) * 20,
                }]}
            }

    class FakeTeam:
        enabled = False
        def choose_domain(self, candidates, fallback):
            return fallback
        def order(self, rows):
            return rows

    args = parse_args(["--agents", "off", "--min-words", "5", "--no-continuous"])
    assert domain_candidates("Politics of Europe", "government parliament elections", "history")[0] == "politics_governance"
    assert relevant_to_europe("European integration", "Institutions in Europe")
    assert not relevant_to_europe("History of Mars", "A planetary chronology concerning settlements on Mars")

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder / "_state").mkdir()
        db = connect(folder / "_state" / "collection.sqlite3")
        try:
            with db:
                enqueue(db, "history", "category", "Category:History of Europe", 0, {"origin": "test"})
                enqueue(db, "politics_governance", "search", "Europe politics", 0, {"origin": "test"})
            cat = db.execute("SELECT * FROM tasks WHERE kind='category'").fetchone()
            assert category_task(db, FakeAPI(), cat, args) >= 1
            search = db.execute("SELECT * FROM tasks WHERE kind='search'").fetchone()
            assert search_task(db, FakeAPI(), search, args) >= 0
            article = db.execute("SELECT * FROM tasks WHERE kind='article' ORDER BY id LIMIT 1").fetchone()
            assert article_task(db, FakeAPI(), article, args, FakeTeam()) == 1

            # Duplicate content through a different source key must remain one document.
            with db:
                _, created = ingest(
                    db, "politics_governance", "synthetic:duplicate", "Duplicate title",
                    db.execute("SELECT body FROM documents LIMIT 1").fetchone()[0],
                    "test", args,
                )
            assert not created
            assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 1

            export_content(db, folder)
            europe_text = (folder / "europe.txt").read_text(encoding="utf-8")
            lowered = europe_text.casefold()
            assert "european integration" in lowered
            assert "http://" not in lowered and "https://" not in lowered
            assert "sha-256" not in lowered and "rights:" not in lowered and "pageid" not in lowered
            files = list((folder / "training_text").rglob("*.txt"))
            assert len(files) == 1

            before = (folder / "europe.txt").read_bytes()
            export_content(db, folder)
            assert before == (folder / "europe.txt").read_bytes()

            # A later Quality block must repair all LANTRA-facing exports and remove
            # the already-exported document rather than leaving stale training text.
            with db:
                db.execute("UPDATE documents SET quality_status='block',assessed=1 WHERE id=1")
                db.execute("INSERT INTO meta VALUES('exports_dirty','1') ON CONFLICT(key) DO UPDATE SET value='1'")
            export_content(db, folder)
            assert (folder / "europe.txt").read_text(encoding="utf-8") == ""
            assert not list((folder / "training_text").rglob("*.txt"))
            with db:
                db.execute("UPDATE documents SET quality_status='pass',assessed=1 WHERE id=1")
                db.execute("INSERT INTO meta VALUES('exports_dirty','1') ON CONFLICT(key) DO UPDATE SET value='1'")
            export_content(db, folder)
            before = (folder / "europe.txt").read_bytes()

            # Simulate interrupted append; export truncates to committed byte checkpoint.
            with (folder / "europe.txt").open("ab") as stream:
                stream.write(b"INTERRUPTED")
            export_content(db, folder)
            assert before == (folder / "europe.txt").read_bytes()

            db.close()
            db = connect(folder / "_state" / "collection.sqlite3")
            assert db.execute("SELECT count(*) FROM documents").fetchone()[0] == 1
        finally:
            db.close()
    print("PASS: Europe collector resume, dedupe, discovery, and content-only export checks passed.")


def check_agents(team: Team) -> None:
    """Exercise each real agent offline without fetching network content."""
    import tempfile

    chosen = team.choose_domain(["politics_governance", "history"], "history")
    assert chosen in {"politics_governance", "history"}
    rows = [
        {"id": 1, "kind": "article", "domain": "history"},
        {"id": 2, "kind": "article", "domain": "economy"},
    ]
    # sqlite.Row is not required by Team.order; mapping semantics are enough.
    ordered = team.order(rows)
    assert {r["id"] for r in ordered} == {1, 2}

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        db = connect(folder / "test.sqlite3")
        try:
            args = parse_args(["--agents", "off", "--min-words", "5"])
            with db:
                ingest(
                    db,
                    "history",
                    "synthetic:agent-check",
                    "European collector agent check",
                    (
                        "Europe has a long historical record involving government, society, economy, "
                        "law, diplomacy, science and education. "
                    ) * 30,
                    "agent_check",
                    args,
                )
            assert team.assess(db, 1) == 1
            team.sync_knowledge(db, 1)
            assert team.knowledge.doc_index
        finally:
            db.close()
    LOG.info("AGENTS | all four collector integrations passed offline checks")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    folder = Path(__file__).resolve().parent / "europe"
    state = folder / "_state"
    folder.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(state / "collector.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"),
        ],
    )
    LOG.info("OUTPUT | %s | runs until Ctrl+C", folder)
    LOG.info("PYTHON | %s | %s", sys.executable, sys.version.split()[0])

    original_cwd = Path.cwd()
    team = None
    db = None
    previous_signals = {}
    STOP.clear()

    def stop(signum, frame):
        STOP.set()
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_signals[signum] = signal.signal(signum, stop)

    code = 0
    try:
        with collection_lock(state):
            root = None
            if args.agents == "team" and not args.export_only:
                root = find_slai_root(args.slai_root)
                os.chdir(root)
                verify_python_runtime()
                verify_agent_runtime_dependencies()
                team = Team(root, True)
            else:
                team = Team(None, False)

            if args.check_agents:
                check_agents(team)
                return 0

            db = connect(state / "collection.sqlite3")
            export_content(db, folder, rebuild=args.rebuild_exports or args.export_only)
            if args.export_only:
                write_summary(db, folder, 0)
                return 0

            seed(db, args)
            if args.recheck_quality:
                export_content(db, folder, rebuild=True)

            if args.process_only:
                while team.assess(db, args.batch_size):
                    team.sync_knowledge(db, args.knowledge_sync_batch)
                team.sync_knowledge(db, args.knowledge_sync_batch)
            else:
                api = APIClient(args.delay, args.contact, args.retries, args.timeout)
                collect(db, api, args, team, folder)

            # Drain a bounded amount during normal shutdown; a huge backlog remains
            # checkpointed for the next run instead of making Ctrl+C appear hung.
            for _ in range(4):
                if not team.assess(db, args.batch_size):
                    break
                team.sync_knowledge(db, args.knowledge_sync_batch)
            export_content(db, folder)
            write_summary(db, folder, 0)
    except KeyboardInterrupt:
        LOG.warning("INTERRUPT | committed data preserved; re-run the same command to resume")
        code = 130
    except Exception:
        LOG.exception("Collector stopped; committed data/checkpoints remain available")
        code = 1
    finally:
        try:
            if db is not None:
                try:
                    export_content(db, folder)
                    write_summary(db, folder, 0)
                finally:
                    db.close()
        finally:
            try:
                if team is not None:
                    team.close()
            finally:
                os.chdir(original_cwd)
                for signum, previous in previous_signals.items():
                    signal.signal(signum, previous)
    return code


if __name__ == "__main__":
    sys.exit(main())
