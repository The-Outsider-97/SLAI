#!/usr/bin/env python3
"""Collect readable astrophysics content for a later LANTRA corpus build.

Python 3.10+. Standard library for standalone collection. Default agent mode uses
SLAI Quality + Knowledge through AgentFactory; it never silently disables agents.
TXT exports contain titles and content only. Provenance and decisions live in
SQLite/JSONL sidecars. Run --help; see ASTROPHYSICS_SCRAPER_README.md.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from html import unescape
from html.parser import HTMLParser
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import xml.etree.ElementTree as ET

LOG = logging.getLogger('astrophysics_collector')
VERSION = '1.0.0'
SCHEMA = 'astrophysics-collector-v1'
WIKI = 'https://en.wikipedia.org/w/api.php'
BOOKS = 'https://en.wikisource.org/w/api.php'
ARXIV = 'https://export.arxiv.org/api/query'
BUCKETS = ('astrophysics_articles', 'astrophysics_papers', 'astrophysics_books')
ARTICLE_SEEDS = '''Astrophysics|Astronomy|Cosmology|Physical cosmology|Observational astronomy|
Theoretical astronomy|Stellar astronomy|Extragalactic astronomy|Galactic astronomy|
Planetary science|Astrometry|Astronomical spectroscopy|Photometry (astronomy)|
Radio astronomy|Infrared astronomy|X-ray astronomy|Gamma-ray astronomy|
Gravitational-wave astronomy|Neutrino astronomy|Multi-messenger astronomy|
Astronomical interferometry|Adaptive optics|Space telescope|Hubble Space Telescope|
James Webb Space Telescope|Chandra X-ray Observatory|Fermi Gamma-ray Space Telescope|
Kepler space telescope|Gaia (spacecraft)|Hertzsprung–Russell diagram|
Stellar classification|Star formation|Stellar evolution|Main sequence|Red giant|
White dwarf|Neutron star|Pulsar|Magnetar|Black hole|Supermassive black hole|
Accretion disk|Event horizon|Hawking radiation|Gravitational lens|
General relativity|Special relativity|Gravitational wave|LIGO|Virgo interferometer|
Milky Way|Galaxy|Galaxy formation and evolution|Active galactic nucleus|Quasar|
Interstellar medium|Intergalactic medium|Nebula|Planetary nebula|Supernova|
Supernova remnant|Cosmic ray|Dark matter|Dark energy|Lambda-CDM model|
Big Bang|Inflation (cosmology)|Cosmic microwave background|Cosmic web|
Large-scale structure of the cosmos|Observable universe|Hubble's law|
Redshift|Baryon acoustic oscillations|Reionization|Nucleosynthesis|
Exoplanet|Exoplanet detection methods|Planetary system|Protoplanetary disk|
Solar System|Sun|Solar physics|Helioseismology|Solar wind|Solar flare|
Magnetosphere|Aurora|Planet|Moon|Small Solar System body|Kuiper belt|
Oort cloud|Asteroid|Comet|Meteorite|Orbital mechanics|Celestial mechanics|
Kepler's laws of planetary motion|Newton's law of universal gravitation|
History of astronomy|Archaeoastronomy|Astronomical unit|Parsec|Light-year'''.replace('\n', '').split('|')
CATEGORY_SEEDS = ['Astrophysics', 'Astronomy', 'Cosmology', 'Stars', 'Galaxies',
                  'Black holes', 'Exoplanets', 'Planetary science', 'Solar physics',
                  'Stellar astronomy', 'Observational astronomy', 'Radio astronomy',
                  'X-ray astronomy', 'Gravitational waves', 'Astronomical spectroscopy',
                  'Astronomical instruments', 'Nebulae', 'Supernovae', 'Dark matter',
                  'Cosmic microwave background', 'History of astronomy']
RESEARCH_TERMS = ['astro-ph.CO', 'astro-ph.GA', 'astro-ph.HE', 'astro-ph.IM',
                  'astro-ph.SR', 'astro-ph.EP', 'gr-qc', 'physics.space-ph']
# Traverse only pages under these explicitly selected historical work roots.
WORKS = {
    'The Sidereal Messenger of Galileo Galilei':
        ('astrophysics_books', 'Galileo Galilei; translated by Edward Stafford Carlos'),
    'A Short History of Astronomy (1898)': ('astrophysics_books', 'Arthur Berry'),
    'The Music of the Spheres': ('astrophysics_books', 'Florence Armstrong Grondal'),
}
ASTRO_RE = re.compile(r'\b(astro\w*|cosmo\w*|galax\w*|stellar|stars?|solar|planet\w*|'
                      r'nebula\w*|quasar\w*|pulsar\w*|magnetar\w*|black hole\w*|'
                      r'supernova\w*|celestial|telescope\w*|universe|orbital|'
                      r'gravitational wave\w*|interstellar|exoplanet\w*|moon\w*|'
                      r'light.year\w*|dark matter|dark energy)\b', re.I)
SKIP_CATEGORY = re.compile(r'\b(wikipedia|articles|templates|stubs|births|deaths|people|fictional)\b', re.I)

def now():
    return datetime.now(timezone.utc).isoformat()


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def clean(text):
    text = unicodedata.normalize('NFC', unescape(text)).replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return '\n\n'.join(' '.join(line.split()) for line in text.splitlines() if line.strip())


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


class Skip(Exception):
    """Permanent record failure; can be retried explicitly."""


class Deferred(Exception):
    """Transient/source-wide failure; preserve its checkpoint."""


def allowed_url(url):
    p = urlsplit(url)
    if p.scheme != 'https' or p.username or p.password or p.port not in (None, 443):
        return False
    return (p.hostname, p.path) in {('en.wikipedia.org', '/w/api.php'),
                                    ('en.wikisource.org', '/w/api.php'),
                                    ('export.arxiv.org', '/api/query')}


class RestrictedRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed_url(newurl) or urlsplit(req.full_url).hostname != urlsplit(newurl).hostname:
            raise Skip('Redirect outside the selected API')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class APIClient:
    def __init__(self, delay=1.2, contact='', retries=4, timeout=35):
        self.delay, self.retries, self.timeout = delay, retries, timeout
        self.last = {}
        self.opener = build_opener(RestrictedRedirect())
        self.ua = f'AstrophysicsCollector/{VERSION}' + (f' ({contact})' if contact else '')
    def request(self, url, *, xml=False, **params):
        if params:
            url += '?' + urlencode(params)
        if not allowed_url(url):
            raise Skip('Unsupported API URL')
        failure = None
        for attempt in range(self.retries):
            host = urlsplit(url).hostname
            gap = max(self.delay, 3.0) if host == 'export.arxiv.org' else self.delay
            time.sleep(max(0, gap-(time.monotonic()-self.last.get(host, 0.0))))
            pause = min(30, 2**(attempt+1))
            try:
                self.last[host] = time.monotonic()
                request = Request(url, headers={'User-Agent': self.ua, 'Accept-Encoding': 'identity',
                                  'Accept': 'application/xml' if xml else 'application/json'})
                with self.opener.open(request, timeout=self.timeout) as response:
                    mime = response.headers.get_content_type()
                    if mime not in (('application/xml', 'text/xml', 'application/atom+xml') if xml else ('application/json', 'text/json')):
                        raise Deferred(f'Unexpected API response type: {mime}')
                    raw = response.read(20*1024*1024+1)
                    if len(raw) > 20*1024*1024:
                        raise Skip('API response exceeds 20 MiB')
                if xml:
                    return raw
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise Deferred('Unexpected JSON structure')
                if result.get('error'):
                    error = result['error']
                    code = error.get('code') if isinstance(error, dict) else None
                    if code in {'missingtitle', 'invalidtitle', 'nosuchpageid'}:
                        raise Skip(str(error))
                    raise Deferred(str(error))
                return result
            except HTTPError as exc:
                if exc.code in (401, 403):
                    raise Deferred(f'HTTP {exc.code}: source denied access; no bypass') from exc
                if exc.code not in (429, 500, 502, 503, 504):
                    raise Skip(f'HTTP {exc.code}') from exc
                failure = exc
                retry = exc.headers.get('Retry-After', '')
                try:
                    pause = max(pause, float(retry))
                except ValueError:
                    try:
                        pause = max(pause, parsedate_to_datetime(retry).timestamp()-time.time())
                    except (TypeError, ValueError, OverflowError):
                        pass
                if not math.isfinite(pause) or pause > 60:
                    raise Deferred('Server requested a long pause; retry in a later run') from exc
            except (URLError, OSError, ValueError, Deferred) as exc:
                failure = exc
            if attempt+1 < self.retries:
                LOG.warning('Request failed (%s); retry %d/%d in %.1fs', failure, attempt+2, self.retries, pause)
                time.sleep(pause)
        raise Deferred(f'Request failed after {self.retries} attempts: {failure}')
    def wiki(self, source='wikipedia', **params):
        return self.request(WIKI if source == 'wikipedia' else BOOKS,
                            format='json', formatversion=2, maxlag=5, **params)


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if tables:
        if 'collector_meta' not in tables:
            db.close()
            raise RuntimeError('Output database belongs to another collector; choose a new directory')
        row = db.execute("SELECT value FROM collector_meta WHERE key='schema'").fetchone()
        if not row or row[0] != SCHEMA:
            db.close()
            raise RuntimeError('Incompatible output database schema')
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS collector_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY,source TEXT,kind TEXT,target TEXT,
      depth INTEGER DEFAULT 0,payload TEXT DEFAULT '{}',status TEXT DEFAULT 'pending',error TEXT DEFAULT '',
      UNIQUE(source,kind,target));
    CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY,bucket TEXT,title TEXT,body TEXT,
      content_hash TEXT,metadata TEXT,status TEXT DEFAULT 'pending',quality TEXT DEFAULT '{}',
      UNIQUE(bucket,content_hash));
    CREATE TABLE IF NOT EXISTS origins(document_id TEXT,source_key TEXT PRIMARY KEY,metadata TEXT);
    CREATE INDEX IF NOT EXISTS task_pending ON tasks(source,status,id);
    CREATE INDEX IF NOT EXISTS doc_pending ON documents(status,bucket);
    ''')
    with db:
        db.execute('INSERT OR IGNORE INTO collector_meta VALUES(?,?)', ('schema', SCHEMA))
    return db


def enqueue(db, source, kind, target, depth=0, payload=None):
    db.execute('INSERT OR IGNORE INTO tasks(source,kind,target,depth,payload) VALUES(?,?,?,?,?)',
               (source, kind, target, depth, json.dumps(payload or {}, ensure_ascii=False)))


def seed(db, args):
    with db:
        db.execute("UPDATE tasks SET status='pending' WHERE status='error'")
        if args.retry_skipped:
            db.execute("UPDATE tasks SET status='pending' WHERE status='skipped'")
        if args.refresh_research:
            db.execute("UPDATE tasks SET status='pending',payload='{}' WHERE kind='search'")
        if args.recheck_quality:
            db.execute("UPDATE documents SET status='pending',quality='{}' WHERE status IN ('pass','warn','block','standalone')")
        for title in ARTICLE_SEEDS:
            enqueue(db, 'wikipedia', 'article', title)
        for title in CATEGORY_SEEDS:
            enqueue(db, 'wikipedia', 'category', 'Category:'+title)
        for term in RESEARCH_TERMS:
            enqueue(db, 'research', 'search', term)
        for root, (bucket, author) in WORKS.items():
            enqueue(db, 'wikisource', 'work', root, payload={'root': root, 'bucket': bucket, 'author': author})


def ingest(db, key, bucket, title, body, metadata, args):
    title, body = clean(title), clean(body)
    words = re.findall(r"\b[\w'-]+\b", body)
    minimum = min(args.min_words, 35) if bucket == 'astrophysics_books' else args.min_words
    reasons = []
    if len(words) < minimum:
        reasons.append('too_short')
    if len(body) and sum(c.isalpha() for c in body)/len(body) < 0.35:
        reasons.append('low_text_density')
    if bucket in ('astrophysics_articles', 'astrophysics_papers') and not ASTRO_RE.search(title+' '+body[:10000]):
        reasons.append('no_astronomy_signal')
    if re.search(r'\b(may refer to:|this disambiguation page)\b', body[:1000], re.I):
        reasons.append('disambiguation')
    content_hash = digest(' '.join(body.casefold().split()))
    existing = db.execute('SELECT id FROM documents WHERE bucket=? AND content_hash=?', (bucket, content_hash)).fetchone()
    doc_id = existing[0] if existing else 'astrophysics:'+bucket+':'+content_hash
    meta = dict(metadata, collected_at=now(), source_key=key, verified=False)
    # Repeated identifiers do not duplicate a document, even if its source text changes.
    old = db.execute('SELECT document_id FROM origins WHERE source_key=?', (key,)).fetchone()
    if old:
        return old[0]
    if not existing:
        db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?,?)',
                   (doc_id, bucket, title, body, content_hash, json.dumps(meta, ensure_ascii=False),
                    'filtered' if reasons else 'pending', json.dumps({'local_filters': reasons})))
    db.execute('INSERT INTO origins VALUES(?,?,?)', (doc_id, key, json.dumps(meta, ensure_ascii=False)))
    return doc_id


def done(db, task):
    db.execute("UPDATE tasks SET status='done',error='' WHERE id=?", (task['id'],))


def wiki_task(db, api, task, args):
    payload = json.loads(task['payload'])
    if task['kind'] == 'category':
        params = dict(action='query', list='categorymembers', cmtitle=task['target'], cmlimit=50, cmtype='page|subcat')
        params.update(payload.get('continue', {}))
        data = api.wiki(**params)
        members = data.get('query', {}).get('categorymembers')
        if not isinstance(members, list):
            raise Deferred('Category response lacks members')
        with db:
            db.execute("INSERT INTO collector_meta VALUES('category_pages','1') ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER)+1")
            for member in members:
                title = member.get('title', '')
                if member.get('ns') == 0:
                    enqueue(db, 'wikipedia', 'article', title, task['depth'])
                elif member.get('ns') == 14 and not SKIP_CATEGORY.search(title):
                    enqueue(db, 'wikipedia', 'category', title, task['depth']+1)
            if data.get('continue'):
                token = {'continue': data['continue']}
                if token == payload:
                    raise Deferred('Repeated category cursor')
                db.execute('UPDATE tasks SET payload=? WHERE id=?', (json.dumps(token), task['id']))
            else:
                done(db, task)
        return
    data = api.wiki(action='query', titles=task['target'], redirects=1,
                    prop='extracts|info|revisions|pageprops', explaintext=1,
                    inprop='url', rvprop='ids', rvlimit=1)
    pages = data.get('query', {}).get('pages')
    if not isinstance(pages, list) or not pages:
        raise Deferred('Missing Wikipedia pages')
    page = pages[0]
    if page.get('missing') or 'disambiguation' in page.get('pageprops', {}):
        raise Skip('Missing page or disambiguation')
    body = page.get('extract', '')
    if not body.strip():
        raise Skip('No article text')
    with db:
        ingest(db, f"wikipedia:{page['pageid']}", 'astrophysics_articles', page['title'], body,
               {'source': 'wikipedia', 'url': page.get('fullurl', ''), 'kind': 'encyclopedia',
                'rights': 'Wikipedia attribution/share-alike terms apply',
                'revision': page.get('revisions', [{}])[0].get('revid')}, args)
        done(db, task)


def work_task(db, api, task, args):
    payload = json.loads(task['payload'])
    root = payload['root']
    data = api.wiki('wikisource', action='parse', page=task['target'], prop='text|links|revid', redirects=1)
    page = data.get('parse')
    if not isinstance(page, dict):
        raise Deferred('Missing Wikisource parse result')
    title = page.get('title', task['target'])
    if title != root and not title.startswith(root+'/'):
        raise Skip('Work redirect leaves selected title tree')
    raw = page.get('text', '')
    if isinstance(raw, dict):
        raw = raw.get('*', '')
    body = html_text(raw)
    with db:
        scoped_links = [link for link in page.get('links', []) if link.get('ns') == 0 and link.get('title', '').startswith(root+'/')]
        navigation_only = bool(scoped_links) and len(body.split()) < 200 and bool(re.search(r'\bcontents\b', body, re.I))
        if body and not navigation_only:
            ingest(db, f"wikisource:{page.get('pageid', title)}", payload['bucket'], title, body,
                   {'source': 'wikisource', 'url': 'https://en.wikisource.org/wiki/'+quote(title.replace(' ', '_')),
                    'kind': 'historical_astronomy',
                    'author': payload['author'], 'root': root, 'revision': page.get('revid'),
                    'rights': 'Historical work; check translation and transcription terms before reuse'}, args)
        for link in page.get('links', []):
            child = link.get('title', '')
            if link.get('ns') == 0 and child.startswith(root+'/'):
                enqueue(db, 'wikisource', 'work', child, task['depth']+1, payload)
        done(db, task)


def research_search(db, api, task, args):
    """Fetch an Atom page and retain the paper's actual abstract text."""
    payload = json.loads(task['payload'])
    seen = payload.get('seen', 0)
    remaining = args.results_per_topic-seen if args.results_per_topic else 50
    capacity = args.max_studies - db.execute(
        "SELECT count(*) FROM documents WHERE bucket='astrophysics_papers'").fetchone()[0] if args.max_studies else 50
    size = min(50, remaining, capacity)
    if size <= 0:
        with db:
            done(db, task)
        return
    start = payload.get('start', 0)
    raw = api.request(ARXIV, xml=True, search_query='cat:'+task['target'], start=start,
                      max_results=size, sortBy='submittedDate', sortOrder='descending')
    if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise Skip('XML entity declarations are unsupported')
    try:
        feed = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise Deferred('Malformed arXiv Atom feed') from exc
    atom = '{http://www.w3.org/2005/Atom}'
    opensearch = '{http://a9.com/-/spec/opensearch/1.1/}'
    arxiv = '{http://arxiv.org/schemas/atom}'
    if feed.tag != atom+'feed':
        raise Deferred('Unexpected arXiv feed')
    entries = feed.findall(atom+'entry')
    try:
        total = int(feed.findtext(opensearch+'totalResults', default='-1'))
    except ValueError as exc:
        raise Deferred('Unexpected arXiv result count') from exc
    if total < 0 or (not entries and start < total):
        raise Deferred('Incomplete arXiv result page; checkpoint retained')
    with db:
        for entry in entries:
            identifier = (entry.findtext(atom+'id') or '').strip()
            parsed = urlsplit(identifier)
            if parsed.scheme not in {'http', 'https'} or parsed.hostname not in {'arxiv.org', 'export.arxiv.org'} or not re.fullmatch(r'/abs/[A-Za-z0-9.\-]+', parsed.path):
                continue
            title = clean(entry.findtext(atom+'title') or '')
            abstract = clean(entry.findtext(atom+'summary') or '')
            if not title or not abstract:
                continue
            authors = [clean(author.findtext(atom+'name') or '') for author in entry.findall(atom+'author')]
            category = entry.find(arxiv+'primary_category')
            license_url = entry.findtext(arxiv+'license') or ''
            ingest(db, 'arxiv:'+parsed.path.removeprefix('/abs/'), 'astrophysics_papers', title,
                   abstract, {'source': 'arxiv', 'url': 'https://arxiv.org'+parsed.path, 'kind': 'research_abstract',
                   'authors': authors, 'published': entry.findtext(atom+'published'),
                   'updated': entry.findtext(atom+'updated'),
                   'primary_category': category.get('term') if category is not None else '',
                   'doi': entry.findtext(arxiv+'doi') or '', 'rights': license_url or 'Unspecified on API entry',
                   'peer_review_verified': False}, args)
        next_start = start+len(entries)
        if entries and next_start < total and (not args.results_per_topic or seen+len(entries) < args.results_per_topic):
            db.execute('UPDATE tasks SET payload=? WHERE id=?',
                       (json.dumps({'start': next_start, 'seen': seen+len(entries)}), task['id']))
            db.execute('UPDATE tasks SET id=(SELECT max(id)+1 FROM tasks) WHERE id=?', (task['id'],))
        else:
            done(db, task)

class AgentPipeline:
    """Explicit Quality -> Knowledge data flow, one factory and shared memory.

    No generated rewriting: the agent verdict filters source text; Knowledge
    indexes it. A failed/unknown verdict never becomes an implicit pass.
    """
    def __init__(self, factory, memory, mode='team', owns=False):
        self.factory, self.memory, self.mode, self.owns = factory, memory, mode, owns
        self.closed, self.indexed = False, set()
        self.quality = None
        self.knowledge = None
        if mode == 'team':
            self.quality = factory.create('quality', shared_memory=memory,
                                          config={'enabled': True, 'auto_route_via_workflow': False})
            if not callable(getattr(self.quality, 'evaluate_batch', None)):
                raise RuntimeError('QualityAgent must provide evaluate_batch')
        if mode in ('team', 'knowledge'):
            self.knowledge = factory.create('knowledge', shared_memory=memory, config={
                'source': 'astrophysics_collector', 'directory_path': '', 'retrieval_mode': 'tfidf',
                'bias_detection_enabled': False, 'use_ontology_expansion': False})
            if not callable(getattr(self.knowledge, 'add_document', None)) or not isinstance(getattr(self.knowledge, 'doc_index', None), dict):
                raise RuntimeError('KnowledgeAgent must provide add_document and doc_index acknowledgement')

    def assess(self, db, batch_size=12):
        # Standalone records are reconsidered when agents are enabled later.
        eligible = ('pending', 'standalone') if self.quality else ('pending',)
        placeholders = ','.join('?' for _ in eligible)
        while True:
            first = db.execute(f'SELECT * FROM documents WHERE status IN ({placeholders}) ORDER BY rowid LIMIT 1', eligible).fetchone()
            if first is None:
                break
            source = json.loads(first['metadata'])['source']
            # Preserve homogeneous source/type batches without scanning the whole corpus.
            rows = db.execute(f'SELECT * FROM documents WHERE status IN ({placeholders}) AND bucket=? ORDER BY rowid LIMIT ?',
                              (*eligible, first['bucket'], batch_size)).fetchall()
            if self.quality:
                records = []
                for row in rows:
                    meta = json.loads(row['metadata'])
                    records.append({'id': row['id'], 'title': row['title'], 'text': row['body'],
                                    'source_id': source, 'source_type': meta['kind'],
                                    'collected_at': meta['collected_at'], 'word_count': len(row['body'].split())})
                fields = {'id': 'str', 'title': 'str', 'text': 'str', 'source_id': 'str',
                          'source_type': 'str', 'collected_at': 'str', 'word_count': 'int'}
                schema = {'schema_version': SCHEMA, 'required_fields': list(fields),
                          'fields': {key: {'type': value, 'required': True} for key, value in fields.items()}}
                batch_id = 'astrophysics-'+digest('|'.join(row['id'] for row in rows))[:24]
                metadata = json.loads(rows[0]['metadata'])
                result = self.quality.evaluate_batch(
                    records, dataset_id='astrophysics_collector/'+first['bucket'], source_id=source,
                    batch_id=batch_id, schema=schema, use_case='knowledge_ingestion',
                    feature_fields=['word_count'],
                    provenance={'source_id': source, 'source_type': metadata['kind'],
                                'collected_at': metadata['collected_at'], 'collector': 'astrophysics_collector',
                                'checksum': digest('|'.join(row['content_hash'] for row in rows)),
                                'uri': metadata.get('url', ''), 'schema_version': SCHEMA},
                    source_metadata={'source_id': source, 'source_type': metadata['kind']},
                    context={'route': 'astrophysics_collector->quality->knowledge', 'external_content': True,
                             'content_type': metadata['kind'], 'fact_verification_performed': False})
                if not isinstance(result, dict) or result.get('verdict') not in {'pass', 'warn', 'block'}:
                    raise RuntimeError('QualityAgent returned an invalid decision; records remain pending')
                status = result['verdict']
                if result.get('quarantine_count', 0) or result.get('quarantine_entries'):
                    status = 'block'  # Do not accidentally export quarantined batch members.
                if any('disabled' in str(flag).lower() for flag in result.get('flags', [])):
                    raise RuntimeError('QualityAgent is disabled; cannot mark collection as assessed')
                LOG.info('QUALITY | %s | %d documents | verdict=%s score=%s',
                         first['bucket'], len(rows), status, result.get('batch_score'))
            else:
                status = 'standalone'
                result = {'verdict': 'not_assessed', 'reason': 'Quality Agent not selected'}
            with db:
                db.executemany('UPDATE documents SET status=?,quality=? WHERE id=?',
                               [(status, json.dumps(result, ensure_ascii=False, default=str), r['id']) for r in rows])

    def sync(self, db, strict=False):
        if not self.knowledge:
            return
        statuses = ('pass',) if strict else ('pass', 'warn', 'standalone')
        placeholders = ','.join('?' for _ in statuses)
        for row in db.execute(f'SELECT * FROM documents WHERE status IN ({placeholders}) ORDER BY rowid', statuses):
            if row['id'] in self.indexed:
                continue
            meta = json.loads(row['metadata'])
            words = row['body'].split()
            for start in range(0, len(words), 1000):
                part = start//1000
                # Genre prefix affects only the retrieval index, never TXT content.
                text = f"{meta['kind']}: {row['title']}\n\n" + ' '.join(words[start:start+1000])
                doc_id = row['id']+f':chunk:{part}'
                existing = self.knowledge.doc_index.get(doc_id)
                if existing is None:
                    self.knowledge.add_document(text=text, doc_id=doc_id, metadata={
                        **meta, 'bucket': row['bucket'], 'parent_id': row['id'], 'chunk': part,
                        'quality_verdict': row['status'], 'trust': 'unverified_external_content'})
                    existing = self.knowledge.doc_index.get(doc_id)
                if not isinstance(existing, dict) or existing.get('text') != text.strip():
                    # The agent deduplicates identical chunk text across documents too.
                    indexed_hash = digest(text.strip())
                    if indexed_hash not in getattr(self.knowledge, 'content_hashes', set()):
                        raise RuntimeError('KnowledgeAgent did not acknowledge collected text')
            self.indexed.add(row['id'])
        LOG.info('KNOWLEDGE | indexed %d documents in this process', len(self.indexed))

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.owns:
            try:
                self.factory.shutdown()
            finally:
                self.memory.close()


def find_slai_root(explicit=None):
    candidates = [explicit.expanduser().resolve()] if explicit else []
    if not explicit:
        for base in (Path.cwd(), Path(__file__).resolve().parent):
            candidates.extend((base, *base.parents))
    for root in candidates:
        if (root/'src/agents/agent_factory.py').is_file():
            return root
    raise RuntimeError('SLAI root not found. Use --slai-root PATH or explicitly choose --agents off')


def create_agents(root, mode):
    global LOG
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if importlib.util.find_spec('yaml') is None:
        venv_python = root/'venv'/'Scripts'/'python.exe' if os.name == 'nt' else root/'venv'/'bin'/'python'
        if venv_python.is_file() and venv_python.absolute() != Path(sys.executable).absolute():
            LOG.warning('Python %s cannot import PyYAML; restarting with %s', sys.executable, venv_python)
            os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])
        raise RuntimeError(
            f'PyYAML is unavailable to the interpreter running this scraper: {sys.executable}. '
            f'Check its environment with: {sys.executable} -m pip show PyYAML. '
            f'Expected virtual environment Python: {venv_python}'
        )
    # Direct project imports, intentionally no optional-import exception fallback.
    from src.agents.agent_factory import AgentFactory
    from src.agents.collaborative.shared_memory import SharedMemory
    from logs.logger import configure_logging, get_logger
    configure_logging()
    LOG = get_logger('Astrophysics Collector')
    memory = SharedMemory()
    factory = None
    try:
        factory = AgentFactory()
        return AgentPipeline(factory, memory, mode, owns=True)
    except BaseException:
        try:
            if factory is not None:
                factory.shutdown()
        finally:
            memory.close()
        raise


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


def export(db, folder, strict=False, write_metadata=True):
    # Keep collection and assessment separate. In the normal mode, every
    # nonempty fetched body is exported, even if Quality has not assessed it or
    # has flagged it. The sidecar retains that decision for later selection.
    # --strict-quality remains an explicit pass-only selection.
    selection = "status='pass'" if strict else "length(trim(body))>0"
    counts = {}
    # One document per training file preserves document-level split boundaries in
    # LANTRA's current raw-text loader. Aggregate files are convenient reading copies.
    manifest_path = folder/'training_files.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
    managed = []
    for row in db.execute(f'SELECT rowid AS sequence,* FROM documents WHERE {selection} ORDER BY rowid'):
        relative = f"training_text/{row['bucket']}/{row['sequence']:08d}.txt"
        destination = folder/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = row['title']+'\n\n'+row['body']+'\n'
        if not destination.exists() or destination.read_text(encoding='utf-8') != text:
            with atomic_writer(destination) as stream:
                stream.write(text)
        managed.append(relative)
    current = set(managed)
    for relative in previous:
        if isinstance(relative, str) and re.fullmatch(r'training_text/astrophysics_(articles|papers|books)/[0-9]{8,}\.txt', relative) and relative not in current:
            (folder/relative).unlink(missing_ok=True)
    with atomic_writer(manifest_path) as stream:
        json.dump(managed, stream, indent=2)
    for bucket in BUCKETS:
        count, words = 0, 0
        with atomic_writer(folder/(bucket+'.txt')) as stream:
            for row in db.execute(f'SELECT title,body FROM documents WHERE bucket=? AND {selection} ORDER BY rowid',
                                  (bucket,)):
                # No URLs/IDs/checksums/rights blocks or QA reports in training text.
                stream.write(row['title']+'\n\n'+row['body']+'\n\n\n')
                count += 1
                words += len(row['body'].split())
        counts[bucket] = {'documents': count, 'words': words}
    if write_metadata:
        with atomic_writer(folder/'collection_metadata.jsonl') as stream:
            for row in db.execute('SELECT * FROM documents ORDER BY rowid'):
                record = {k: row[k] for k in ('id', 'bucket', 'title', 'content_hash', 'status')}
                record['metadata'] = json.loads(row['metadata'])
                record['quality'] = json.loads(row['quality'])
                record['origins'] = [json.loads(r[0]) for r in db.execute('SELECT metadata FROM origins WHERE document_id=?', (row['id'],))]
                stream.write(json.dumps(record, ensure_ascii=False, default=str)+'\n')
    with atomic_writer(folder/'collection_summary.json') as stream:
        json.dump({'exported_at': now(), 'version': VERSION, 'strict_quality': strict,
                   'files': counts,
                   'document_statuses': dict(db.execute('SELECT status,count(*) FROM documents GROUP BY status')),
                   'task_statuses': dict(db.execute('SELECT status,count(*) FROM tasks GROUP BY status'))},
                  stream, ensure_ascii=False, indent=2)
    LOG.info('EXPORT | %s', ' | '.join(f"{b}: {v['documents']} docs/{v['words']} words" for b, v in counts.items()))
    return counts


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


def select_task(db, source, args):
    if source == 'wikipedia':
        rows = db.execute("SELECT * FROM tasks WHERE source=? AND status='pending' AND (kind='article' OR depth<=?) ORDER BY id LIMIT 1",
                          (source, args.depth)).fetchall()
    elif source == 'wikisource':
        rows = db.execute("SELECT * FROM tasks WHERE source=? AND status='pending' AND depth<=? ORDER BY id LIMIT 1",
                          (source, args.book_depth)).fetchall()
    else:
        rows = db.execute("SELECT * FROM tasks WHERE source=? AND status='pending' ORDER BY id LIMIT 1", (source,)).fetchall()
    return rows[0] if rows else None


def collect(db, api, args, pipeline):
    disabled, steps = set(), 0
    start = time.monotonic()
    while True:
        progress = False
        for source in args.sources:
            if source in disabled:
                continue
            if args.max_steps and steps >= args.max_steps:
                return
            buckets = {'wikipedia': ('astrophysics_articles',), 'research': ('astrophysics_papers',),
                       'wikisource': ('astrophysics_books',)}[source]
            cap = {'wikipedia': args.max_articles, 'research': args.max_studies, 'wikisource': args.max_book_pages}[source]
            # Budgets count retained candidates, including QA-blocked texts, avoiding unbounded collection.
            count = db.execute('SELECT count(*) FROM documents WHERE bucket IN ('+','.join('?' for _ in buckets)+')', buckets).fetchone()[0]
            if cap and count >= cap:
                continue
            task = select_task(db, source, args)
            if task is None:
                continue
            if task['kind'] == 'category' and args.max_categories:
                category_row = db.execute("SELECT value FROM collector_meta WHERE key='category_pages'").fetchone()
                categories = int(category_row[0]) if category_row else 0
                if categories >= args.max_categories:
                    # Article tasks already queued remain eligible.
                    task = db.execute("SELECT * FROM tasks WHERE source='wikipedia' AND kind='article' AND status='pending' ORDER BY id LIMIT 1").fetchone()
                    if task is None:
                        continue
            progress = True
            steps += 1
            LOG.info('FETCH | step=%d source=%s kind=%s title=%s', steps, source, task['kind'], task['target'])
            try:
                if source == 'wikipedia':
                    wiki_task(db, api, task, args)
                elif source == 'wikisource':
                    work_task(db, api, task, args)
                elif task['kind'] == 'search':
                    research_search(db, api, task, args)
                else:
                    raise Skip('Unknown research task')
            except Skip as exc:
                with db:
                    db.execute("UPDATE tasks SET status='skipped',error=? WHERE id=?", (str(exc), task['id']))
                LOG.warning('SKIP | %s | %s', task['target'], exc)
            except (Deferred, OSError, ValueError) as exc:
                with db:
                    db.execute("UPDATE tasks SET status='error',error=? WHERE id=?", (str(exc), task['id']))
                disabled.add(source)
                LOG.warning('DEFER | %s | %s', source, exc)
            if steps % args.checkpoint_every == 0:
                pipeline.assess(db, args.batch_size)
                export(db, args.output_dir, args.strict_quality, write_metadata=False)
                pipeline.sync(db, args.strict_quality)
                LOG.info('PROGRESS | %d tasks | %.1f minutes | pending=%d', steps,
                         (time.monotonic()-start)/60,
                         db.execute("SELECT count(*) FROM tasks WHERE status='pending'").fetchone()[0])
        if not progress:
            break


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, help='Legacy option: ignored; output is always beside this script in astrophysics/')
    p.add_argument('--slai-root', type=Path)
    p.add_argument('--agents', choices=('team', 'knowledge', 'off'), default='team')
    p.add_argument('--max-articles', type=int, default=3000)
    p.add_argument('--max-studies', type=int, default=1500)
    p.add_argument('--max-book-pages', type=int, default=500)
    p.add_argument('--max-categories', type=int, default=1000)
    p.add_argument('--results-per-topic', type=int, default=500)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--book-depth', type=int, default=4)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--min-words', type=int, default=80)
    p.add_argument('--checkpoint-every', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=12)
    p.add_argument('--sources', default='wikipedia,research,wikisource')
    p.add_argument('--delay', type=float, default=1.2)
    p.add_argument('--timeout', type=float, default=35)
    p.add_argument('--retries', type=int, default=4)
    p.add_argument('--contact', default='')
    p.add_argument('--strict-quality', action='store_true', help='Export/index only Quality pass records')
    p.add_argument('--retry-skipped', action='store_true')
    p.add_argument('--refresh-research', action='store_true')
    p.add_argument('--recheck-quality', action='store_true')
    p.add_argument('--process-only', action='store_true', help='Assess and index saved content; no network')
    p.add_argument('--export-only', action='store_true', help='Rebuild exports; no agents or network')
    p.add_argument('--check-agents', action='store_true', help='Initialize and validate selected agents; no collection')
    p.add_argument('--knowledge-query')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args(argv)
    for field in ('max_articles', 'max_studies', 'max_book_pages', 'max_categories', 'results_per_topic', 'depth', 'book_depth', 'max_steps'):
        if getattr(args, field) < 0:
            p.error(field+' must be nonnegative')
    for field in ('min_words', 'checkpoint_every', 'batch_size', 'retries'):
        if getattr(args, field) < 1:
            p.error(field+' must be positive')
    if not math.isfinite(args.delay) or args.delay < 1:
        p.error('--delay must be finite and at least 1 second')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        p.error('--timeout must be finite and positive')
    if any(ord(c) < 32 or ord(c) > 126 for c in args.contact):
        p.error('--contact must be printable ASCII')
    args.sources = list(dict.fromkeys(args.sources.split(',')))
    if not set(args.sources) <= {'wikipedia', 'research', 'wikisource'} or not args.sources:
        p.error('--sources accepts wikipedia,research,wikisource')
    if args.strict_quality and args.agents != 'team' and not args.export_only:
        p.error('--strict-quality requires --agents team')
    if args.check_agents and (args.agents == 'off' or args.export_only):
        p.error('--check-agents requires agents and cannot accompany --export-only')
    if args.knowledge_query and (args.agents == 'off' or args.export_only):
        p.error('--knowledge-query requires agents and cannot accompany --export-only')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    requested_output_dir = args.output_dir
    args.output_dir = Path(__file__).resolve().parent/'astrophysics'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir/'collector.log', encoding='utf-8')])
    if requested_output_dir is not None and requested_output_dir.expanduser().resolve() != args.output_dir:
        LOG.warning('Ignoring --output-dir %s; content is saved beside this script.', requested_output_dir)
    LOG.info('OUTPUT | All collected documents will be saved in %s', args.output_dir)
    pipeline, db = None, None
    previous_cwd = Path.cwd()
    code = 0
    try:
        with collection_lock(args.output_dir):
            try:
                if args.agents != 'off' and not args.export_only:
                    root = find_slai_root(args.slai_root)
                    os.chdir(root)
                    pipeline = create_agents(root, args.agents)
                    LOG.info('AGENTS | active=%s | factory-owned lifecycle', args.agents)
                else:
                    pipeline = AgentPipeline(None, None, mode='off')
                if args.check_agents:
                    LOG.info('Agent initialization and interface checks passed')
                else:
                    db = connect(args.output_dir/'history.sqlite3')
                    if not args.export_only:
                        seed(db, args)
                        pipeline.assess(db, args.batch_size)
                        pipeline.sync(db, args.strict_quality)
                        if not args.process_only:
                            api = APIClient(args.delay, args.contact, args.retries, args.timeout)
                            collect(db, api, args, pipeline)
                        pipeline.assess(db, args.batch_size)
                        # Export before indexing so an indexing failure cannot hide accepted content.
                        export(db, args.output_dir, args.strict_quality, write_metadata=False)
                        pipeline.sync(db, args.strict_quality)
                        if args.knowledge_query:
                            assert pipeline.knowledge is not None
                            results = pipeline.knowledge.retrieve(args.knowledge_query, k=5)
                            for score, doc in results:
                                print(f"\nScore {float(score):.3f}\n{doc.get('text', '')}")
                        if db.execute("SELECT count(*) FROM tasks WHERE status='error'").fetchone()[0]:
                            code = 2
            finally:
                if db is not None:
                    try:
                        export(db, args.output_dir, args.strict_quality, write_metadata=not args.export_only)
                    finally:
                        db.close()
    except KeyboardInterrupt:
        LOG.warning('Interrupted; committed content retained. Run again to resume.')
        code = 130
    except Exception:
        LOG.exception('Collector stopped. Existing committed content remains available.')
        code = 1
    finally:
        try:
            if pipeline is not None:
                pipeline.close()
        except Exception:
            LOG.exception('Agent shutdown reported an error')
            code = 1
        os.chdir(previous_cwd)
    return code


def self_test():
    """Offline tests for content exports, resumability and agent wiring."""
    import tempfile
    args = parse_args(['--agents', 'off', '--min-words', '5', '--results-per-topic', '3'])
    class Memory:
        closed = False
        def close(self):
            self.closed = True
    class Knowledge:
        def __init__(self):
            self.doc_index, self.content_hashes = {}, set()
        def add_document(self, text, doc_id=None, metadata=None):
            self.doc_index[doc_id] = {'text': text.strip(), 'metadata': metadata}
            self.content_hashes.add(digest(text.strip()))
    class Quality:
        verdict = 'pass'
        def evaluate_batch(self, records, *, dataset_id, source_id, batch_id, schema,
                           use_case, feature_fields, provenance, source_metadata, context):
            assert records and provenance['checksum'] and source_id
            assert all(r['source_id'] == source_id for r in records)
            return {'verdict': self.verdict, 'batch_score': 0.95, 'flags': []}
    class Factory:
        def __init__(self):
            self.k, self.q, self.calls, self.closed = Knowledge(), Quality(), [], False
        def create(self, kind, shared_memory=None, **kwargs):
            self.calls.append((kind, shared_memory))
            return self.q if kind == 'quality' else self.k
        def shutdown(self):
            self.closed = True
    class API:
        def wiki(self, source='wikipedia', **params):
            if params.get('list') == 'categorymembers':
                return {'query': {'categorymembers': [{'ns': 0, 'title': 'Astrophysics'},
                     {'ns': 14, 'title': 'Category:Galaxies'}]}, 'continue': {'continue': '-||', 'cmcontinue': 'next'}}
            if source == 'wikisource':
                return {'parse': {'title': 'The Sidereal Messenger of Galileo Galilei', 'pageid': 2,
                    'text': '<p>The telescope revealed the moons of Jupiter in the night sky.</p><script>bad</script>',
                    'links': [{'ns': 0, 'title': 'The Sidereal Messenger of Galileo Galilei/Part I'}]}}
            return {'query': {'pages': [{'pageid': 1, 'title': 'Astrophysics',
                'extract': 'Astrophysics studies the stars, galaxies and universe using physical laws.',
                'fullurl': 'https://en.wikipedia.org/wiki/Astrophysics'}]}}
        def request(self, url, **params):
            assert url == ARXIV and params['search_query'].startswith('cat:')
            if params['start']:
                return b'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"><opensearch:totalResults>1</opensearch:totalResults></feed>'
            return b'''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom"><opensearch:totalResults>2</opensearch:totalResults>
            <entry><id>http://arxiv.org/abs/2609.12345v1</id><title>Galaxy dynamics</title>
            <summary>Galaxy rotation and dark matter can be measured by stellar motions in disk galaxies.</summary>
            <author><name>A. Scientist</name></author><arxiv:primary_category term="astro-ph.GA" /></entry></feed>'''
    assert html_text('<p>Galaxy <em>formation</em>.</p><script>bad</script>') == 'Galaxy formation.'
    assert not allowed_url('http://export.arxiv.org/api/query')
    assert not allowed_url('https://export.arxiv.org/private')
    assert not allowed_url('https://127.0.0.1/api/query')
    assert allowed_url(ARXIV)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        db = connect(folder/'history.sqlite3')
        seed(db, args)
        api = API()
        wiki_task(db, api, db.execute("SELECT * FROM tasks WHERE kind='article' LIMIT 1").fetchone(), args)
        cat = db.execute("SELECT * FROM tasks WHERE kind='category' LIMIT 1").fetchone()
        wiki_task(db, api, cat, args)
        assert json.loads(db.execute('SELECT payload FROM tasks WHERE id=?', (cat['id'],)).fetchone()[0])['continue']['cmcontinue'] == 'next'
        work_task(db, api, db.execute("SELECT * FROM tasks WHERE target='The Sidereal Messenger of Galileo Galilei'").fetchone(), args)
        assert db.execute("SELECT 1 FROM tasks WHERE target='The Sidereal Messenger of Galileo Galilei/Part I'").fetchone()
        search = db.execute("SELECT * FROM tasks WHERE kind='search' LIMIT 1").fetchone()
        research_search(db, api, search, args)
        assert db.execute("SELECT body FROM documents WHERE bucket='astrophysics_papers'").fetchone()[0].startswith('Galaxy rotation')
        updated = db.execute("SELECT * FROM tasks WHERE kind='search' AND target=?", (search['target'],)).fetchone()
        assert json.loads(updated['payload'])['start'] == 1
        research_search(db, api, updated, args)
        counts = export(db, folder)
        assert all(value['documents'] == 1 for value in counts.values()), counts
        assert len(list((folder/'training_text').rglob('*.txt'))) == 3
        factory, memory = Factory(), Memory()
        team = AgentPipeline(factory, memory, 'team', owns=True)
        team.assess(db)
        team.sync(db)
        assert [kind for kind, _ in factory.calls] == ['quality', 'knowledge']
        assert all(shared is memory for _, shared in factory.calls)
        assert len(factory.k.doc_index) == 3
        with db:
            ingest(db, 'short', 'astrophysics_articles', 'Stellar candidate', 'Star.',
                   {'source': 'wikipedia', 'kind': 'encyclopedia'}, args)
        assert db.execute("SELECT status FROM documents WHERE title='Stellar candidate'").fetchone()[0] == 'filtered'
        assert export(db, folder)['astrophysics_articles']['documents'] == 2
        with db:
            ingest(db, 'blocked', 'astrophysics_articles', 'Black hole candidate',
                   'The black hole absorbs matter from a nearby star.',
                   {'source': 'wikipedia', 'kind': 'encyclopedia'}, args)
        factory.q.verdict = 'block'
        team.assess(db)
        assert export(db, folder)['astrophysics_articles']['documents'] == 3
        assert export(db, folder, strict=True)['astrophysics_articles']['documents'] == 1
        assert export(db, folder)['astrophysics_articles']['documents'] == 3
        for path in folder.glob('*.txt'):
            contents = path.read_text(encoding='utf-8')
            assert 'https://' not in contents and 'sha256' not in contents and 'rights' not in contents
        team.close()
        assert factory.closed and memory.closed
        db.close()
    print('Self-test passed: content exports, research pagination, safe URLs, and Quality/Knowledge integration.')


if __name__ == '__main__':
    sys.exit(main())
