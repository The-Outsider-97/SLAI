#!/usr/bin/env python3
"""Continuously collect romance, horror and drama text for LANTRA.

Python 3.10+. Run beside SLAI's src/ directory.
Run: py -m scrape_literature_01
Repair mixed/broken agent dependencies: py -m scrape_literature_01 --repair-environment
The repair creates a private runtime; your SLAI venv and corpus are preserved.
Ctrl+C saves and stops.
Default: Planning + Reasoning + Quality + Knowledge via AgentFactory.
Output: literature/{romance,horror,drama}.txt beside this module.
"""
from __future__ import annotations
import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from html import unescape
from html.parser import HTMLParser
import subprocess
import uuid
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import sys
import threading
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit, parse_qs
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.robotparser import RobotFileParser

LOG = logging.getLogger('literature_collector')
VERSION = '2.0.0'
GENRES = ('romance', 'horror', 'drama')
SOURCES = ('gutenberg', 'wikisource', 'wikipedia')
WIKI = 'https://en.wikipedia.org/w/api.php'
WORKS = 'https://en.wikisource.org/w/api.php'
CATALOG = 'https://gutendex.com/books/'
MIRROR = 'https://gutenberg.pglaf.org'
TOPICS = {
    'romance': ['love stories', 'romance', 'courtship', 'romantic fiction'],
    'horror': ['horror', 'ghost stories', 'gothic fiction', 'supernatural', 'vampires'],
    'drama': ['drama', 'tragedies', 'comedies', 'one-act plays'],
}
CATEGORIES = {
    'wikipedia': {
        'romance': ['Romance novels', 'Romantic fiction', 'Romantic poetry'],
        'horror': ['Horror fiction', 'Horror short stories', 'Gothic fiction'],
        'drama': ['Drama', 'Plays', 'Dramatic genres'],
    },
    'wikisource': {
        'romance': ['Romance novels', 'Love stories', 'Romantic poetry'],
        'horror': ['Horror', 'Horror fiction', 'Ghost stories', 'Gothic novels'],
        'drama': ['Plays', 'Tragedies', 'Comedies', 'Drama'],
    },
}
ARTICLES = {
    'romance': ['Romance novel', 'Romantic fiction', 'Love story', 'Romantic poetry', 'Courtly love'],
    'horror': ['Horror fiction', 'Gothic fiction', 'Ghost story', 'Weird fiction', 'Supernatural fiction'],
    'drama': ['Drama', 'Tragedy', 'Comedy (drama)', 'Melodrama', 'Tragicomedy', 'Play (theatre)'],
}
# Stable catalogue IDs give immediate full-book work while discovery expands.
SEED_BOOKS = {'romance': [1342, 161, 158, 141, 105],
              'horror': [345, 84, 43, 42, 10007],
              'drama': [1514, 1513, 1533, 1532, 2542]}
LABELS = {
    'romance': re.compile(r'\b(romance|romantic fiction|love stor\w*|courtship)\b', re.I),
    'horror': re.compile(r'\b(horror|ghost stor\w*|gothic|supernatural|vampir\w*|weird fiction)\b', re.I),
    'drama': re.compile(r'\b(drama\w*|plays?|traged\w*|comed\w*|theat\w*)\b', re.I),
}
SKIP_CATEGORY = re.compile(r'\b(wikipedia|templates|stubs|maintenance|births|deaths|people|authors|writers)\b', re.I)
STOP = threading.Event()


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def clean(value):
    value = unicodedata.normalize('NFC', unescape(value)).replace('\r\n', '\n').replace('\r', '\n')
    value = re.sub(r'[\u200b\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]', '', value)
    # Keep verse, dialogue, paragraph breaks and stage directions intact.
    return re.sub(r'\n{4,}', '\n\n\n', '\n'.join(line.rstrip() for line in value.splitlines())).strip()


class Skip(Exception):
    pass


class Deferred(Exception):
    def __init__(self, message, seconds=300):
        super().__init__(message)
        self.seconds = max(5, float(seconds))


def pause(seconds):
    if STOP.wait(max(0, seconds)):
        raise KeyboardInterrupt


def allowed_url(url):
    try:
        p = urlsplit(url)
        if p.scheme != 'https' or p.username or p.password or p.port not in (None, 443) or p.fragment:
            return False
    except ValueError:
        return False
    if (p.hostname, p.path) in {('en.wikipedia.org', '/w/api.php'), ('en.wikisource.org', '/w/api.php')}:
        return True
    if p.hostname == 'gutendex.com':
        return bool(re.fullmatch(r'/books(?:/\d+)?/?', p.path))
    if p.hostname == 'gutenberg.pglaf.org':
        return p.path == '/robots.txt' or bool(re.fullmatch(r'/cache/epub/\d+/pg\d+\.txt', p.path))
    return False


class RestrictedRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed_url(newurl) or urlsplit(req.full_url).hostname != urlsplit(newurl).hostname:
            raise Skip('Redirect outside approved source')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Client:
    def __init__(self, args):
        self.args, self.last, self.robots = args, {}, None
        self.opener = build_opener(RestrictedRedirect())
        self.ua = f'LANTRALiteratureCollector/{VERSION}' + (f' ({args.contact})' if args.contact else '')
    def request(self, url, *, text=False, **params):
        if params:
            url += ('&' if '?' in url else '?') + urlencode(params)
        if not allowed_url(url):
            raise Skip('Unsupported source URL')
        host = urlsplit(url).hostname
        if host == 'gutenberg.pglaf.org' and urlsplit(url).path != '/robots.txt':
            if not self.robots or time.time()-self.robots[0] > 86400:
                try:
                    rules = self.request(MIRROR+'/robots.txt', text=True)
                except Skip as exc:
                    if str(exc) != 'HTTP 404':
                        raise
                    rules = ''
                parser = RobotFileParser()
                parser.parse(rules.splitlines())
                self.robots = (time.time(), parser)
            if not self.robots[1].can_fetch(self.ua, url):
                raise Deferred('Mirror robots.txt disallows this download', 86400)
        failure = None
        for attempt in range(self.args.retries):
            gap = self.args.delay
            if self.robots and host == 'gutenberg.pglaf.org':
                gap = max(gap, self.robots[1].crawl_delay(self.ua) or 0)
            pause(gap-(time.monotonic()-self.last.get(host, 0)))
            try:
                self.last[host] = time.monotonic()
                req = Request(url, headers={'User-Agent': self.ua, 'Accept-Encoding': 'identity',
                              'Accept': 'text/plain' if text else 'application/json'})
                with self.opener.open(req, timeout=self.args.timeout) as response:
                    mime = response.headers.get_content_type()
                    allowed = {'text/plain', 'application/octet-stream'} if text else {'application/json', 'text/json'}
                    if mime not in allowed:
                        raise Deferred(f'Unexpected response type {mime}')
                    raw = response.read(24*1024*1024+1)
                    if len(raw) > 24*1024*1024:
                        raise Skip('Response exceeds 24 MiB limit')
                    encoding = response.headers.get_content_charset() or 'utf-8-sig'
                if text:
                    try:
                        return raw.decode(encoding)
                    except (UnicodeDecodeError, LookupError) as exc:
                        raise Skip('Unrecognized text encoding') from exc
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise Deferred('Invalid API response')
                if data.get('error'):
                    error = data['error']
                    if isinstance(error, dict) and error.get('code') in {'missingtitle', 'invalidtitle', 'nosuchpageid'}:
                        raise Skip(str(error))
                    raise Deferred(str(error))
                return data
            except HTTPError as exc:
                if exc.code in (401, 403):
                    raise Deferred(f'HTTP {exc.code}: source denied access', 3600) from exc
                if exc.code not in (429, 500, 502, 503, 504):
                    raise Skip(f'HTTP {exc.code}') from exc
                delay = min(900, 15*2**attempt)
                retry = exc.headers.get('Retry-After')
                if retry:
                    try:
                        delay = max(delay, float(retry))
                    except ValueError:
                        try:
                            delay = max(delay, parsedate_to_datetime(retry).timestamp()-time.time())
                        except (ValueError, TypeError, OverflowError):
                            pass
                if not math.isfinite(delay):
                    delay = 3600
                # Persisted source cooldown instead of holding an in-flight job.
                raise Deferred(f'HTTP {exc.code}', delay) from exc
            except (URLError, TimeoutError, OSError, ValueError) as exc:
                failure = exc
                if attempt+1 < self.args.retries:
                    LOG.warning('NETWORK | retry %d/%d | %s', attempt+2, self.args.retries, exc)
                    pause(min(30, 2**(attempt+1)))
        raise Deferred(f'Request failed: {failure}', 300)
    def wiki(self, source, **params):
        return self.request(WIKI if source == 'wikipedia' else WORKS,
                            format='json', formatversion=2, maxlag=5, **params)


class TextParser(HTMLParser):
    BLOCK = {'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'li', 'br', 'blockquote', 'section'}
    VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'wbr'}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.stack = [], []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        excluded = tag in {'script', 'style', 'table', 'nav', 'header', 'footer', 'noscript'}
        excluded |= bool(set((attrs.get('class') or '').split()) &
                         {'ws-noexport', 'noprint', 'navbox', 'mw-editsection', 'reference', 'licenseContainer'})
        excluded |= attrs.get('id') in {'header', 'license', 'toc', 'ws-header'}
        blocked = excluded or bool(self.stack and self.stack[-1][1])
        if not blocked and tag in self.BLOCK:
            self.parts.append('\n')
        if tag not in self.VOID:
            self.stack.append((tag, blocked))
    def handle_endtag(self, tag):
        for i in range(len(self.stack)-1, -1, -1):
            if self.stack[i][0] == tag:
                blocked = self.stack[i][1]
                del self.stack[i:]
                if not blocked and tag in self.BLOCK:
                    self.parts.append('\n')
                break
    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)
    def handle_data(self, data):
        if not self.stack or not self.stack[-1][1]:
            self.parts.append(data)


def html_text(value):
    parser = TextParser()
    parser.feed(value or '')
    parser.close()
    return clean(''.join(parser.parts))


@contextmanager
def atomic_writer(path):
    temp = path.with_name(path.name+'.tmp')
    try:
        with temp.open('w', encoding='utf-8', newline='\n') as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def collection_lock(folder):
    with (folder/'collector.lock').open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another collector is using this output directory') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def find_slai_root(explicit=None):
    candidates = [explicit.expanduser().resolve()] if explicit else []
    if not explicit:
        for base in (Path.cwd(), Path(__file__).resolve().parent):
            candidates.extend((base, *base.parents))
    for root in candidates:
        if (root/'src/agents/agent_factory.py').is_file():
            return root
    raise RuntimeError('SLAI root not found. Use --slai-root PATH or explicitly choose --agents off')



def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS tasks(
      id INTEGER PRIMARY KEY, source TEXT, kind TEXT, target TEXT, genre TEXT,
      depth INTEGER DEFAULT 0, payload TEXT DEFAULT '{}', status TEXT DEFAULT 'pending',
      due REAL DEFAULT 0, attempts INTEGER DEFAULT 0, error TEXT DEFAULT '',
      UNIQUE(source,kind,target,genre));
    CREATE INDEX IF NOT EXISTS tasks_due ON tasks(status,due,genre);
    CREATE TABLE IF NOT EXISTS documents(
      id INTEGER PRIMARY KEY, genre TEXT, title TEXT, body TEXT, content_hash TEXT UNIQUE,
      metadata TEXT, assessment TEXT DEFAULT '{}', assessed INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS origins(source_key TEXT PRIMARY KEY, document_id INTEGER);
    CREATE TABLE IF NOT EXISTS exports(genre TEXT PRIMARY KEY,last_id INTEGER,bytes INTEGER);
    CREATE TABLE IF NOT EXISTS cooldowns(source TEXT PRIMARY KEY,until REAL);
    ''')
    row = db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
    if row and row[0] != 'literature-v1':
        raise RuntimeError('Incompatible collector database')
    with db:
        db.execute("INSERT OR IGNORE INTO meta VALUES('schema','literature-v1')")
    return db


def enqueue(db, source, kind, target, genre, depth=0, payload=None):
    db.execute('INSERT OR IGNORE INTO tasks(source,kind,target,genre,depth,payload) VALUES(?,?,?,?,?,?)',
               (source, kind, target, genre, depth, json.dumps(payload or {}, ensure_ascii=False)))


def seed(db, args):
    with db:
        # Verified public-domain seeds can start even during a catalogue outage.
        for book_id, title, genre, author in [
            (1342, 'Pride and Prejudice', 'romance', 'Jane Austen'),
            (345, 'Dracula', 'horror', 'Bram Stoker'),
            (1514, "A Midsummer Night's Dream", 'drama', 'William Shakespeare'),
        ]:
            enqueue(db, 'gutenberg', 'book', str(book_id), genre, payload={
                'id': book_id, 'title': title, 'copyright': False, 'languages': ['en'],
                'authors': [{'name': author}], 'subjects': [genre], 'bookshelves': [],
            })
        for genre in GENRES:
            for book_id in SEED_BOOKS[genre]:
                enqueue(db, 'gutenberg', 'lookup', str(book_id), genre)
            for topic in TOPICS[genre]:
                enqueue(db, 'gutenberg', 'catalog', topic, genre)
            for source, categories in CATEGORIES.items():
                for category in categories[genre]:
                    enqueue(db, source, 'category', 'Category:'+category, genre)
            for title in ARTICLES[genre]:
                enqueue(db, 'wikipedia', 'article', title, genre)
        if args.retry_skipped:
            db.execute("UPDATE tasks SET status='pending',due=0 WHERE status='skipped'")


def finish(db, task, args, payload=None):
    if payload is not None:
        db.execute("UPDATE tasks SET payload=?,attempts=0,error='' WHERE id=?",
                   (json.dumps(payload, ensure_ascii=False), task['id']))
        # Move a continued discovery behind existing jobs to avoid starvation.
        db.execute('UPDATE tasks SET id=(SELECT max(id)+1 FROM tasks) WHERE id=?', (task['id'],))
    elif task['kind'] in {'category', 'catalog'}:
        db.execute("UPDATE tasks SET payload='{}',due=?,attempts=0,error='' WHERE id=?",
                   (time.time()+args.refresh_hours*3600, task['id']))
    else:
        db.execute("UPDATE tasks SET status='done',error='' WHERE id=?", (task['id'],))


def ingest(db, key, genre, title, body, metadata):
    body, title = clean(body), clean(title)
    if not body:
        raise Skip('No readable text')
    old = db.execute('SELECT document_id FROM origins WHERE source_key=?', (key,)).fetchone()
    if old:
        return old[0]
    hashed = digest(' '.join(body.casefold().split()))
    old = db.execute('SELECT id FROM documents WHERE content_hash=?', (hashed,)).fetchone()
    if old:
        doc_id = old[0]
    else:
        doc_id = db.execute('INSERT INTO documents(genre,title,body,content_hash,metadata) VALUES(?,?,?,?,?)',
            (genre, title, body, hashed, json.dumps(dict(metadata, collected_at=now()), ensure_ascii=False))).lastrowid
        LOG.info('SAVED | %s | %s | %s words', genre, title[:100], f'{len(body.split()):,}')
    db.execute('INSERT INTO origins VALUES(?,?)', (key, doc_id))
    return doc_id


def classify_labels(record, fallback):
    labels = '\n'.join(record.get('subjects', []) + record.get('bookshelves', []))
    genres = [genre for genre, pattern in LABELS.items() if pattern.search(labels)]
    return genres or [fallback]


def strip_book(text):
    start = re.search(r'^\s*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*{3}\s*$', text, re.I|re.M)
    end = re.search(r'^\s*\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*{3}\s*$', text, re.I|re.M)
    if not start or not end or end.start() <= start.end():
        raise Skip('Complete Gutenberg text boundary markers unavailable')
    return clean(text[start.end():end.start()])


def catalog_record(db, record, genre):
    if record.get('copyright') is not False or 'en' not in record.get('languages', []):
        return
    if record.get('media_type') != 'Text' or not any(k.startswith('text/plain') for k in record.get('formats', {})):
        return
    book_id = record.get('id')
    if isinstance(book_id, int) and book_id > 0:
        enqueue(db, 'gutenberg', 'book', str(book_id), genre, payload=record)


def gutenberg_task(db, client, task, args, team):
    payload = json.loads(task['payload'])
    if task['kind'] == 'lookup':
        record = client.request(CATALOG+task['target']+'/')
        with db:
            catalog_record(db, record, task['genre'])
            finish(db, task, args)
        return
    if task['kind'] == 'catalog':
        data = client.request(CATALOG, topic=task['target'], languages='en', copyright='false',
                              mime_type='text/plain', sort='ascending', page=payload.get('page', 1))
        records = data.get('results')
        if not isinstance(records, list):
            raise Deferred('Missing catalogue results')
        next_page = None
        if data.get('next'):
            nxt = urlsplit(data['next'])
            if nxt.hostname != 'gutendex.com' or nxt.path not in {'/books', '/books/'}:
                raise Deferred('Unexpected catalogue continuation')
            try:
                next_page = int(parse_qs(nxt.query)['page'][0])
            except (KeyError, ValueError, IndexError) as exc:
                raise Deferred('Missing catalogue page number') from exc
            if next_page <= payload.get('page', 1):
                raise Deferred('Repeated catalogue page')
        with db:
            for record in records:
                catalog_record(db, record, task['genre'])
            finish(db, task, args, {'page': next_page} if next_page else None)
        return
    # Generated UTF-8 texts from the approved mirror, never main-site scraping.
    book_id = int(task['target'])
    url = f'{MIRROR}/cache/epub/{book_id}/pg{book_id}.txt'
    body = strip_book(client.request(url, text=True))
    labels = classify_labels(payload, task['genre'])
    try:
        genre = team.choose_genre(labels, task['genre'])
    except Exception:
        LOG.exception('REASONING | temporary failure; keeping catalogue genre and saved text')
        genre = task['genre'] if task['genre'] in labels else labels[0]
    with db:
        ingest(db, 'gutenberg:'+str(book_id), genre, payload.get('title', 'Untitled work'), body,
               {'source': 'gutenberg', 'kind': 'literary_full_text', 'url': url,
                'authors': payload.get('authors', []), 'genres': labels,
                'subjects': payload.get('subjects', []), 'bookshelves': payload.get('bookshelves', []),
                'rights': 'Catalogue marks public domain in the USA; local jurisdiction may differ'})
        finish(db, task, args)


def wiki_task(db, client, task, args):
    payload = json.loads(task['payload'])
    source, genre = task['source'], task['genre']
    if task['kind'] == 'category':
        params = dict(action='query', list='categorymembers', cmtitle=task['target'],
                      cmlimit=50, cmtype='page|subcat')
        params.update(payload.get('continue', {}))
        data = client.wiki(source, **params)
        members = data.get('query', {}).get('categorymembers')
        if not isinstance(members, list):
            raise Deferred('Missing category members')
        continuation = data.get('continue')
        if continuation and continuation == payload.get('continue'):
            raise Deferred('Repeated category cursor')
        with db:
            for member in members:
                title = member.get('title', '')
                if member.get('ns') == 0:
                    enqueue(db, source, 'article' if source == 'wikipedia' else 'work', title, genre,
                            0, {'root': title})
                elif member.get('ns') == 14 and task['depth'] < args.depth and not SKIP_CATEGORY.search(title):
                    enqueue(db, source, 'category', title, genre, task['depth']+1)
            finish(db, task, args, {'continue': continuation} if continuation else None)
        return
    if source == 'wikipedia':
        data = client.wiki(source, action='query', titles=task['target'], redirects=1,
                           prop='extracts|info|pageprops', explaintext=1, inprop='url')
        pages = data.get('query', {}).get('pages')
        if not pages:
            raise Deferred('Missing Wikipedia pages')
        page = pages[0]
        if page.get('missing') or 'disambiguation' in page.get('pageprops', {}):
            raise Skip('Missing page or disambiguation')
        body = page.get('extract', '')
        meta = {'source': source, 'kind': 'encyclopedia_article', 'url': page.get('fullurl', ''),
                'genres': [genre], 'rights': 'Wikipedia attribution/share-alike terms apply'}
    else:
        data = client.wiki(source, action='parse', page=task['target'], prop='text|links|revid', redirects=1)
        page = data.get('parse')
        if not isinstance(page, dict):
            raise Deferred('Missing Wikisource text')
        root = payload.get('root', task['target'])
        title = page.get('title', task['target'])
        if title != root and not title.startswith(root+'/'):
            # A root redirect identifies the canonical title; child redirects must stay scoped.
            if task['target'] == root and '/' not in task['target']:
                root = title
            else:
                raise Skip('Chapter redirect leaves selected work')
        raw = page.get('text', '')
        body = html_text(raw.get('*', '') if isinstance(raw, dict) else raw)
        children = [link['title'] for link in page.get('links', [])
                    if link.get('ns') == 0 and link.get('title', '').startswith(root+'/')]
        with db:
            if task['depth'] < args.book_depth:
                for child in children:
                    enqueue(db, source, 'work', child, genre, task['depth']+1, {'root': root})
        if children and len(body.split()) < 200 and re.search(r'\bcontents\b', body, re.I):
            with db:
                finish(db, task, args)
            return
        meta = {'source': source, 'kind': 'literary_page',
                'url': 'https://en.wikisource.org/wiki/'+quote(title.replace(' ', '_')),
                'genres': [genre], 'rights': 'Wikisource work and transcription terms apply'}
    with db:
        ingest(db, source+':'+str(page.get('pageid', page.get('title', task['target']))), genre,
               page.get('title', task['target']), body, meta)
        finish(db, task, args)


def export_new(db, folder, rebuild=False):
    """Append committed records; truncate an interrupted append before replaying it."""
    if rebuild:
        with db:
            db.execute('DELETE FROM exports')
    for genre in GENRES:
        checkpoint = db.execute('SELECT last_id,bytes FROM exports WHERE genre=?', (genre,)).fetchone()
        last_id, byte_count = tuple(checkpoint) if checkpoint else (0, 0)
        path = folder/(genre+'.txt')
        if not path.exists() or path.stat().st_size < byte_count:
            last_id, byte_count = 0, 0
        with path.open('r+b' if path.exists() else 'w+b') as aggregate:
            aggregate.truncate(byte_count)
            aggregate.seek(byte_count)
            for row in db.execute('SELECT * FROM documents WHERE genre=? AND id>? ORDER BY id', (genre,last_id)):
                text = row['title']+'\n\n'+row['body']+'\n\n\n'
                destination = folder/'training_text'/genre/f"{row['id']:09d}.txt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    with atomic_writer(destination) as stream:
                        stream.write(text)
                aggregate.write(text.encode('utf-8'))
                last_id = row['id']
            aggregate.flush()
            os.fsync(aggregate.fileno())
            byte_count = aggregate.tell()
        with db:
            db.execute('INSERT INTO exports VALUES(?,?,?) ON CONFLICT(genre) DO UPDATE SET last_id=excluded.last_id,bytes=excluded.bytes',
                       (genre, last_id, byte_count))
    # Only newly discovered/assessed metadata is rewritten, never the corpus.
    previous = db.execute("SELECT value FROM meta WHERE key='metadata_last'").fetchone()
    last_meta = int(previous[0]) if previous and not rebuild else 0
    for row in db.execute('SELECT * FROM documents WHERE id>? ORDER BY id', (last_meta,)):
        write_metadata(folder, row)
        last_meta = row['id']
    with db:
        db.execute("INSERT INTO meta VALUES('metadata_last',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(last_meta),))


def write_metadata(folder, row):
    path = folder/'_state'/'metadata'/f"{row['id']:09d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_writer(path) as stream:
        json.dump({'title': row['title'], 'genre': row['genre'], 'metadata': json.loads(row['metadata']),
                   'assessment': json.loads(row['assessment'])}, stream, ensure_ascii=False, indent=2)

def genre_rules(facts):
    inferred = {}
    for fact, confidence in list(facts.items()):
        if not isinstance(fact, tuple) or len(fact) != 3 or fact[0] != 'literature_candidate':
            continue
        _, predicate, genre = fact
        if genre in GENRES and predicate in {'catalog_genre', 'discovery_genre'}:
            weight = .95 if predicate == 'catalog_genre' else .65
            key = ('literature_candidate', 'selected_genre', genre)
            inferred[key] = max(inferred.get(key, 0), confidence*weight)
    return inferred


class Team:
    def __init__(self, root=None, enabled=True):
        global LOG
        self.enabled, self.root = enabled, root
        self.factory = self.memory = None
        self.planning = self.reasoning = self.quality = self.knowledge = None
        self.rounds, self.indexed_words = 0, 0
        if not enabled:
            return
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from src.agents.agent_factory import AgentFactory
        from src.agents.collaborative.shared_memory import SharedMemory
        from src.agents.planning.planning_types import Task, TaskType, ResourceProfile
        from logs.logger import get_logger
        LOG = get_logger('literature_collector')
        self.Task, self.TaskType, self.Resources = Task, TaskType, ResourceProfile
        self.memory = SharedMemory()
        try:
            self.factory = AgentFactory()
            self.planning = self.factory.create('planning', shared_memory=self.memory,
                config={'plan_history_window': 64, 'execution_history_window': 64})
            self.reasoning = self.factory.create('reasoning', shared_memory=self.memory)
            self.quality = self.factory.create('quality', shared_memory=self.memory,
                config={'enabled': True, 'auto_route_via_workflow': False})
            self.knowledge = self.factory.create('knowledge', shared_memory=self.memory,
                config={'source': 'literature_collector', 'directory_path': '', 'retrieval_mode': 'tfidf',
                        'bias_detection_enabled': False, 'use_ontology_expansion': False})
            for agent, names in [(self.planning, ['generate_plan']),
                (self.reasoning, ['add_fact', 'add_rule', 'forward_chaining', 'forget_by_subject']),
                (self.quality, ['evaluate_batch']), (self.knowledge, ['add_document'])]:
                if not all(callable(getattr(agent, name, None)) for name in names):
                    raise RuntimeError(f'{type(agent).__name__} lacks required collector interface')
            if not isinstance(getattr(self.knowledge, 'doc_index', None), dict):
                raise RuntimeError('KnowledgeAgent lacks doc_index acknowledgement')
            self.reasoning.add_rule(genre_rules, rule_name='literature_genre_evidence', weight=1.0)
        except BaseException:
            self.close()
            raise
        LOG.info('AGENTS | Planning + Reasoning + Quality + Knowledge active through AgentFactory')

    def choose_genre(self, labels, fallback):
        labels = [g for g in labels if g in GENRES]
        if not self.enabled:
            return fallback if fallback in labels or not labels else labels[0]
        subject = 'literature_candidate'
        self.reasoning.forget_by_subject(subject)
        try:
            for genre in labels:
                if not self.reasoning.add_fact((subject, 'catalog_genre', genre), publish=False):
                    raise RuntimeError('ReasoningAgent rejected genre evidence')
            self.reasoning.add_fact((subject, 'discovery_genre', fallback), publish=False)
            self.reasoning.forward_chaining(max_iterations=3)
            scores = {g: self.reasoning.knowledge_base.get((subject, 'selected_genre', g), 0) for g in GENRES}
            best = max(scores, key=lambda g: (scores[g], g == fallback))
            if not scores[best]:
                raise RuntimeError('ReasoningAgent produced no genre conclusion')
            LOG.info('REASONING | genre=%s | catalogue labels=%s | evidence strength=%.2f', best, labels, scores[best])
            return best
        finally:
            self.reasoning.forget_by_subject(subject)

    def order(self, rows):
        if not self.enabled:
            return rows
        self.rounds += 1
        deadline = time.time()+86400
        tasks = []
        for row in rows:
            task = self.Task(name='fetch_'+str(row['id']), id='fetch_'+str(row['id']),
                task_type=self.TaskType.PRIMITIVE, preconditions=[], effects=[],
                resource_requirements=self.Resources(gpu=0, ram=.05), duration=30,
                deadline=deadline, dependencies=[tasks[-1].id] if tasks else [])
            tasks.append(task)
        goal = self.Task(name=f'literature_round_{self.rounds}', task_type=self.TaskType.ABSTRACT,
            methods=[tasks], resource_requirements=self.Resources(gpu=0, ram=0),
            duration=30*len(tasks), deadline=deadline)
        plan = self.planning.generate_plan(goal)
        if not plan:
            raise RuntimeError('PlanningAgent could not schedule queued collection work')
        by_id = {'fetch_'+str(row['id']): row for row in rows}
        ids = [task.id for task in plan]
        if len(ids) != len(by_id) or set(ids) != set(by_id):
            raise RuntimeError('PlanningAgent returned an incomplete/unrecognized plan')
        LOG.info('PLANNING | scheduled %d queued tasks', len(plan))
        return [by_id[key] for key in ids]

    def assess_one(self, db, folder):
        if not self.enabled:
            return
        row = db.execute('SELECT * FROM documents WHERE assessed=0 ORDER BY id LIMIT 1').fetchone()
        if row is None:
            return
        meta = json.loads(row['metadata'])
        # Structural quality assessment samples long books; original text is untouched.
        record = {'id': str(row['id']), 'title': row['title'], 'text': row['body'][:16000],
                  'source_id': meta['source'], 'source_type': meta['kind'],
                  'collected_at': meta['collected_at'], 'word_count': len(row['body'].split())}
        types = {key: ('int' if key == 'word_count' else 'str') for key in record}
        schema = {'schema_version': 'literature-v1', 'required_fields': list(record),
                  'fields': {key: {'type': kind, 'required': True} for key, kind in types.items()}}
        result = self.quality.evaluate_batch([record], dataset_id='literature/'+row['genre'],
            source_id=meta['source'], batch_id='literature-'+str(row['id']), schema=schema,
            use_case='knowledge_ingestion', feature_fields=['word_count'],
            provenance={'source_id': meta['source'], 'source_type': meta['kind'],
                'collected_at': meta['collected_at'], 'collector': 'literature_collector',
                'checksum': row['content_hash'], 'uri': meta.get('url', ''), 'schema_version': 'literature-v1'},
            source_metadata={'source_id': meta['source'], 'source_type': meta['kind']},
            context={'external_content': True, 'literary_content': True, 'fact_verification_performed': False})
        if not isinstance(result, dict) or result.get('verdict') not in {'pass', 'warn', 'block'}:
            raise RuntimeError('QualityAgent returned no valid verdict')
        if any('disabled' in str(flag).lower() for flag in result.get('flags', [])):
            raise RuntimeError('QualityAgent reports disabled')
        if result.get('quarantine_count') or result.get('quarantine_entries'):
            result['verdict'] = 'block'
        result['assessment_scope'] = 'first_16000_characters'
        if result['verdict'] != 'block':
            words = row['body'].split()
            for start in range(0, len(words), 1000):
                if STOP.is_set():
                    raise KeyboardInterrupt
                text = row['title']+'\n\n'+' '.join(words[start:start+1000])
                identifier = f"literature:{row['id']}:{start//1000}"
                self.knowledge.add_document(text=text, doc_id=identifier,
                    metadata={**meta, 'genre': row['genre'], 'trust': 'external_literary_content'})
                existing = self.knowledge.doc_index.get(identifier)
                if (not isinstance(existing, dict) or existing.get('text') != text.strip()) and digest(text.strip()) not in getattr(self.knowledge, 'content_hashes', set()):
                    raise RuntimeError('KnowledgeAgent did not acknowledge text')
                self.indexed_words += len(words[start:start+1000])
        with db:
            db.execute('UPDATE documents SET assessment=?,assessed=1 WHERE id=?',
                (json.dumps(result, ensure_ascii=False, default=str), row['id']))
        write_metadata(folder, db.execute('SELECT * FROM documents WHERE id=?', (row['id'],)).fetchone())
        LOG.info('QUALITY | %s | %s | text already exported', row['title'][:70], result['verdict'])

    def close(self):
        factory, memory = self.factory, self.memory
        self.factory = self.memory = None
        try:
            if factory is not None:
                factory.shutdown()
        finally:
            if memory is not None:
                memory.close()


def candidates(db, args):
    timestamp = time.time()
    choices = []
    for genre in GENRES:
        for source in args.sources:
            row = db.execute('''SELECT * FROM tasks WHERE status='pending' AND genre=? AND source=? AND due<=?
                AND NOT EXISTS(SELECT 1 FROM cooldowns WHERE source=tasks.source AND until>?)
                ORDER BY CASE WHEN kind IN ('book','work','article') THEN 0 ELSE 1 END,id LIMIT 1''',
                (genre, source, timestamp, timestamp)).fetchone()
            if row:
                choices.append(row)
    counts = dict(db.execute('SELECT genre,count(*) FROM documents GROUP BY genre'))
    choices.sort(key=lambda row: (counts.get(row['genre'], 0), row['id']))
    return choices


def progress(db, folder, steps):
    counts = dict(db.execute('SELECT genre,count(*) FROM documents GROUP BY genre'))
    pending = db.execute("SELECT count(*) FROM tasks WHERE status='pending'").fetchone()[0]
    LOG.info('PROGRESS | tasks this run=%d | documents=%s | queued=%d | Ctrl+C to stop', steps, counts, pending)
    with atomic_writer(folder/'_state'/'summary.json') as stream:
        json.dump({'updated_at': now(), 'documents': counts, 'queued_tasks': pending,
                   'text_files': {g: str(folder/(g+'.txt')) for g in GENRES}}, stream, indent=2)


# Dependencies for the four-agent collector, not SLAI's unrelated UI/server stack.
# SLAI's own agent configuration loaders remain authoritative for agent settings.
AGENT_DEPENDENCIES = (
    'PyYAML', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'psutil',
    'requests', 'networkx', 'cryptography', 'pydantic', 'jsonschema',
    'cachetools', 'tenacity', 'beautifulsoup4', 'nltk', 'tqdm', 'joblib',
    'pgmpy', 'ruptures', 'rdflib', 'diskcache',
)
PROBE_MODULES = (
    'yaml', 'numpy', 'scipy.linalg', 'sklearn.tree', 'pandas', 'torch',
    'src.agents.planning_agent', 'src.agents.reasoning_agent',
    'src.agents.quality_agent', 'src.agents.knowledge_agent',
)


def runtime_python(directory):
    return directory/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def run_visible(command, **kwargs):
    """Inherit terminal output and Ctrl+C; never invoke a shell."""
    result = subprocess.run([str(part) for part in command], **kwargs)
    if result.returncode:
        raise RuntimeError(f'Runtime setup failed (exit {result.returncode}); see output above')


def probe_environment(executable, root, report):
    """Test actual imports in separate processes, including native extensions.

    A failed import cannot leave partially initialized SLAI modules in the collector.
    Preserve every diagnostic instead of reporting only the first broken wheel.
    """
    failures = []
    print(f'RUNTIME | {executable}', flush=True)
    with report.open('w', encoding='utf-8') as stream:
        for module in PROBE_MODULES:
            print(f'CHECK | {module}', flush=True)
            code = ('import sys, importlib; '
                    f'sys.path.insert(0, {str(root)!r}); '
                    f'importlib.import_module({module!r}); '
                    'print(sys.executable); print(sys.version)')
            result = subprocess.run([str(executable), '-c', code], cwd=root,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding='utf-8', errors='replace', timeout=180)
            stream.write(f'\n=== {module} | exit={result.returncode} ===\n{result.stdout}')
            stream.flush()
            if result.returncode:
                failures.append(module)
    return failures


def prepare_runtime(root, args, argv, folder):
    """Select a consistent interpreter or explicitly build a fresh private venv.

    Never install into, delete, or upgrade the user's existing SLAI environment.
    Only publish a replacement runtime after all four real agent checks pass.
    """
    home = Path(__file__).resolve().parent/'.literature_runtime'
    marker = home/'active.json'
    script = Path(__file__).resolve()
    command_name = f'py -m {script.stem}'
    report = folder/'_state'/'environment_check.log'
    executable = Path(sys.executable).absolute()
    selected = executable
    if marker.is_file() and not args.repair_environment:
        saved = json.loads(marker.read_text(encoding='utf-8'))
        candidate = home/saved['directory']
        if candidate.resolve().parent != home.resolve():
            raise RuntimeError('Invalid collector runtime location')
        selected = runtime_python(candidate).absolute()
        if not selected.is_file():
            raise RuntimeError(f'Collector runtime is missing. Run: {command_name} --repair-environment')
    elif sys.prefix == sys.base_prefix and not args.repair_environment:
        for name in ('venv', '.venv'):
            candidate = runtime_python(root/name)
            if candidate.is_file():
                selected = candidate.absolute()
                break
    forwarded = [arg for arg in argv if arg not in
                 {'--repair-environment', '--runtime-ready', '--check-environment'}]
    if args.repair_environment:
        home.mkdir(exist_ok=True)
        # Unique target also preserves any earlier working collector runtime.
        target = home/('py'+str(sys.version_info.major)+str(sys.version_info.minor)+'-'+uuid.uuid4().hex[:12])
        print(f'SETUP | Building isolated agent environment: {target}', flush=True)
        run_visible([executable, '-m', 'venv', target])
        selected = runtime_python(target).absolute()
        run_visible([selected, '-m', 'pip', 'install', '--upgrade', 'pip'])
        run_visible([selected, '-m', 'pip', 'install', '--only-binary=:all:',
                     '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch'])
        run_visible([selected, '-m', 'pip', 'install', '--only-binary=:all:', *AGENT_DEPENDENCIES])
        failures = probe_environment(selected, root, report)
        if failures:
            raise RuntimeError(f'New runtime failed imports: {", ".join(failures)}. Details: {report}')
        run_visible([selected, script, '--runtime-ready', '--slai-root', root, '--check-agents'], cwd=root)
        with atomic_writer(marker) as stream:
            json.dump({'directory': target.name, 'created_at': now()}, stream)
        print('SETUP | All four agents verified; starting collection.', flush=True)
    elif not args.runtime_ready:
        failures = probe_environment(selected, root, report)
        if failures:
            raise RuntimeError(
                f'Agent environment is unusable: {", ".join(failures)}. '
                f'Details: {report}\n'
                f'Create a clean collector environment: {command_name} --repair-environment')
    if args.check_environment:
        print(f'Environment checks passed: {selected}', flush=True)
        return True
    if selected != executable:
        os.execv(str(selected), [str(selected), str(script), *forwarded, '--runtime-ready'])
    return False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repair-environment', action='store_true',
        help='Build a separate compatible agent venv, verify all four agents, then collect')
    parser.add_argument('--check-environment', action='store_true', help='Check agent imports and exit')
    parser.add_argument('--runtime-ready', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--agents', choices=('team','off'), default='team')
    parser.add_argument('--slai-root', type=Path)
    parser.add_argument('--sources', default=','.join(SOURCES))
    parser.add_argument('--delay', type=float, default=3.0)
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--refresh-hours', type=float, default=24)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--book-depth', type=int, default=15)
    parser.add_argument('--contact', default='')
    parser.add_argument('--max-steps', type=int, default=0, help='Optional testing limit; 0 means run until stopped')
    parser.add_argument('--retry-skipped', action='store_true')
    parser.add_argument('--export-only', action='store_true', help='Repair/rebuild text files, then exit')
    parser.add_argument('--check-agents', action='store_true', help='Exercise all four agents offline, then exit')
    parser.add_argument('--self-test', action='store_true', help='Run offline regression checks, then exit')
    args = parser.parse_args(argv)
    args.sources = list(dict.fromkeys(args.sources.split(',')))
    if not args.sources or not set(args.sources) <= set(SOURCES):
        parser.error('--sources must use gutenberg,wikisource,wikipedia')
    if not math.isfinite(args.delay) or args.delay < 2:
        parser.error('--delay must be at least 2 seconds')
    if not math.isfinite(args.refresh_hours) or args.refresh_hours < 1:
        parser.error('--refresh-hours must be at least 1 hour')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.retries < 1 or min(args.depth, args.book_depth, args.max_steps) < 0:
        parser.error('Invalid retry/depth/step limit')
    if any(ord(c) < 32 or ord(c) > 126 for c in args.contact):
        parser.error('--contact must be printable ASCII')
    if args.check_agents and (args.agents == 'off' or args.export_only):
        parser.error('--check-agents requires --agents team and cannot use --export-only')
    if (args.repair_environment or args.check_environment) and (args.agents == 'off' or args.export_only or args.self_test):
        parser.error('Environment setup/check requires team mode')
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    folder = Path(__file__).resolve().parent/'literature'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'_state').mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(), RotatingFileHandler(folder/'_state'/'collector.log',
                    maxBytes=5_000_000, backupCount=3, encoding='utf-8')])
    LOG.info('OUTPUT | %s | runs until Ctrl+C', folder)
    try:
        root = find_slai_root(args.slai_root) if args.agents == 'team' and not args.export_only else None
        if root and prepare_runtime(root, args, argv, folder):
            return 0
    except KeyboardInterrupt:
        LOG.info('Runtime setup/check cancelled; existing text preserved')
        return 130
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        LOG.error('STARTUP | %s', exc)
        return 2
    original = Path.cwd()
    team = db = lock = None
    lock_acquired = False
    steps, last_status, agent_retry = 0, 0.0, 0.0
    STOP.clear()
    def stop(signum, frame):
        STOP.set()
        raise KeyboardInterrupt
    previous_signals = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_signals[signum] = signal.signal(signum, stop)
    try:
        lock = collection_lock(folder/'_state')
        lock.__enter__()
        lock_acquired = True
        db = connect(folder/'_state'/'collection.sqlite3')
        export_new(db, folder, rebuild=args.export_only)
        if args.export_only:
            return 0
        if root:
            os.chdir(root)
        if not args.check_agents:
            seed(db, args)
        team = Team(root, args.agents == 'team')
        if args.check_agents:
            agent_check(team)
            LOG.info('All four agent checks passed')
            return 0
        client = Client(args)
        while not STOP.is_set():
            if team.enabled and (team.rounds >= 100 or team.indexed_words >= 100_000):
                team.close()
                team = Team(root)
                LOG.info('AGENTS | renewed rolling in-memory index; all text remains on disk')
            rows = candidates(db, args)
            if rows:
                try:
                    rows = team.order(rows)
                except Exception:
                    LOG.exception('PLANNING | schedule unavailable; retrying in 60 seconds')
                    pause(60)
                    continue
                for task in rows:
                    if STOP.is_set() or (args.max_steps and steps >= args.max_steps):
                        break
                    # A preceding task in this plan may have cooled down this source.
                    cooldown = db.execute('SELECT until FROM cooldowns WHERE source=?', (task['source'],)).fetchone()
                    if cooldown and cooldown[0] > time.time():
                        continue
                    LOG.info('FETCH | %s | %s | %s', task['genre'], task['source'], task['target'][:110])
                    steps += 1
                    try:
                        if task['source'] == 'gutenberg':
                            gutenberg_task(db, client, task, args, team)
                        else:
                            wiki_task(db, client, task, args)
                    except Skip as exc:
                        with db:
                            db.execute("UPDATE tasks SET status='skipped',error=? WHERE id=?", (str(exc), task['id']))
                        LOG.warning('SKIP | %s | %s', task['target'], exc)
                    except Deferred as exc:
                        wait = max(exc.seconds, min(3600, 30*2**min(task['attempts'],7)))
                        with db:
                            db.execute('UPDATE tasks SET due=?,attempts=attempts+1,error=? WHERE id=?',
                                (time.time()+wait, str(exc), task['id']))
                            db.execute('INSERT INTO cooldowns VALUES(?,?) ON CONFLICT(source) DO UPDATE SET until=excluded.until',
                                (task['source'], time.time()+wait))
                        LOG.warning('DEFER | %s | %.0fs | %s', task['source'], wait, exc)
                    except Exception as exc:
                        with db:
                            db.execute('UPDATE tasks SET due=?,attempts=attempts+1,error=? WHERE id=?',
                                (time.time()+300, str(exc), task['id']))
                        LOG.exception('TASK | retry in 5 minutes; previously saved text retained')
                    export_new(db, folder)
                    if time.time() >= agent_retry:
                        try:
                            team.assess_one(db, folder)
                        except Exception:
                            agent_retry = time.time()+300
                            LOG.exception('AGENTS | assessment/indexing delayed; text has been saved')
            else:
                if time.time() >= agent_retry:
                    try:
                        team.assess_one(db, folder)
                    except Exception:
                        agent_retry = time.time()+300
                        LOG.exception('AGENTS | assessment retry delayed; corpus preserved')
                pause(10)
            if time.time()-last_status > 60:
                progress(db, folder, steps)
                last_status = time.time()
                if not rows:
                    LOG.info('WAIT | no ready tasks; watching cooldowns and daily discovery refresh')
            if args.max_steps and steps >= args.max_steps:
                break
        return 0
    except KeyboardInterrupt:
        LOG.info('STOP | saving committed text; rerun to resume')
        return 130 if args.check_agents else 0
    except Exception:
        LOG.exception('Collector stopped; committed text remains available')
        return 1
    finally:
        try:
            if db is not None:
                try:
                    export_new(db, folder)
                    progress(db, folder, steps)
                finally:
                    db.close()
        finally:
            try:
                if team is not None:
                    team.close()
            finally:
                os.chdir(original)
                for signum, previous in previous_signals.items():
                    signal.signal(signum, previous)
                if lock_acquired:
                    lock.__exit__(None, None, None)


def agent_check(team):
    """Exercise actual methods with small synthetic records, without network calls."""
    import tempfile
    assert team.choose_genre(['horror'], 'romance') == 'horror'
    rows = [{'id': 1}, {'id': 2}]
    assert {r['id'] for r in team.order(rows)} == {1, 2}
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        db = connect(folder/'test.sqlite3')
        try:
            with db:
                ingest(db, 'synthetic', 'drama', 'Collector interface test',
                    'A performer enters the empty stage. The lights rise, and the curtain opens. '*30,
                    {'source': 'wikisource', 'kind': 'literary_page'})
            export_new(db, folder)
            team.assess_one(db, folder)
            row = db.execute('SELECT * FROM documents').fetchone()
            assert row['assessed'] == 1
            # A valid block verdict does not exercise Knowledge; test its actual API too.
            team.knowledge.add_document(text='A synthetic stage dialogue for interface validation.',
                doc_id='literature:interface-check', metadata={'kind': 'test'})
            assert team.knowledge.doc_index.get('literature:interface-check')
        finally:
            db.close()


def self_test():
    import tempfile
    args = parse_args(['--agents','off'])
    assert args.max_steps == 0
    assert not allowed_url('https://127.0.0.1/anything')
    assert not allowed_url('https://gutenberg.pglaf.org/cache/epub/1/../../secret')
    assert not allowed_url('https://www.gutenberg.org/ebooks/1342.txt.utf-8')
    assert allowed_url(MIRROR+'/cache/epub/1342/pg1342.txt')
    fixture = 'Boilerplate\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nAct I\nAlice: Hello.\n\n[Exit.]\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nLicense'
    body = strip_book(fixture)
    assert 'Boilerplate' not in body and 'License' not in body and '[Exit.]' in body
    assert '\nAlice:' in body
    try:
        strip_book('truncated book')
        raise AssertionError('Truncated book accepted')
    except Skip:
        pass
    assert classify_labels({'subjects':['Ghost stories','Gothic fiction'],'bookshelves':[]}, 'romance') == ['horror']
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        db = connect(folder/'test.sqlite3')
        seed(db, args)
        with db:
            for genre in GENRES:
                ingest(db, genre, genre, genre.title(), 'A '+genre+' passage.\n\nACT I\nSpeaker: Hello.',
                       {'source':'wikisource','kind':'literary_page'})
            ingest(db, 'duplicate', 'horror', 'Duplicate', 'A romance passage.\n\nACT I\nSpeaker: Hello.',
                   {'source':'wikisource','kind':'literary_page'})
        assert db.execute('SELECT count(*) FROM documents').fetchone()[0] == 3
        export_new(db, folder)
        before = {g: (folder/(g+'.txt')).read_bytes() for g in GENRES}
        with (folder/'romance.txt').open('ab') as stream:
            stream.write(b'UNCOMMITTED INTERRUPTED APPEND')
        export_new(db, folder)
        assert before['romance'] == (folder/'romance.txt').read_bytes()
        (folder/'horror.txt').unlink()
        export_new(db, folder)
        assert before['horror'] == (folder/'horror.txt').read_bytes()
        export_new(db, folder, rebuild=True)
        assert before == {g: (folder/(g+'.txt')).read_bytes() for g in GENRES}
        assert len(list((folder/'training_text').rglob('*.txt'))) == 3
        cat = db.execute("SELECT * FROM tasks WHERE kind='catalog' LIMIT 1").fetchone()
        with db:
            finish(db, cat, args)
        resumed = db.execute('SELECT * FROM tasks WHERE id=?', (cat['id'],)).fetchone()
        assert resumed['status'] == 'pending' and resumed['due'] > time.time()
        with db:
            db.execute('INSERT INTO cooldowns VALUES(?,?)', ('gutenberg',time.time()+100))
        assert all(row['source'] != 'gutenberg' for row in candidates(db,args))
        db.close()
        db = connect(folder/'test.sqlite3')
        assert db.execute('SELECT count(*) FROM documents').fetchone()[0] == 3
        assert db.execute('SELECT due FROM tasks WHERE id=?',(cat['id'],)).fetchone()[0] == resumed['due']
        db.close()
    print('Self-test passed: content extraction, deduplication, crash recovery, resumption, and cooldown scheduling.')


if __name__ == '__main__':
    sys.exit(main())
