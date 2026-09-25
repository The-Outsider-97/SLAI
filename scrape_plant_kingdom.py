#!/usr/bin/env python3
"""Collect readable plant content for a later LANTRA corpus build.

Python 3.10+. Standard library for standalone collection. Default agent mode uses
SLAI Quality + Knowledge through AgentFactory; it never silently disables agents.
TXT exports contain titles and content only. Provenance and decisions live in
SQLite/JSONL sidecars. Run --help; see PLANT_COLLECTOR_README.md.
"""
from __future__ import annotations

import re
import sys
import time
import json
import math
import os
import unicodedata
import argparse
import logging
import hashlib
import sqlite3
import xml.etree.ElementTree as ET

from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

LOG = logging.getLogger('plant_collector')
VERSION = '1.0.0'
SCHEMA = 'plant-collector-v1'
WIKI = 'https://en.wikipedia.org/w/api.php'
BOOKS = 'https://en.wikisource.org/w/api.php'
EPMC = 'https://www.ebi.ac.uk/europepmc/webservices/rest/'
BUCKETS = ('plant_articles', 'plant_studies', 'plant_books', 'plant_stories')
ARTICLE_SEEDS = '''Plant|Botany|Plant taxonomy|Plant anatomy|Plant physiology|Plant ecology|
Plant evolution|Plant reproduction|Plant genetics|Plant cell|Photosynthesis|Chloroplast|
Chlorophyll|Calvin cycle|Photorespiration|C3 carbon fixation|C4 carbon fixation|
Crassulacean acid metabolism|Root|Stem (botany)|Leaf|Flower|Fruit|Seed|Pollen|
Pollination|Seed dispersal|Germination|Meristem|Xylem|Phloem|Stoma|Transpiration|
Plant hormone|Auxin|Gibberellin|Abscisic acid|Phototropism|Gravitropism|Photoperiodism|
Dormancy|Plant defense against herbivory|Plant disease|Plant pathology|
Plant secondary metabolism|Mycorrhiza|Rhizosphere|Nitrogen fixation|Legume|
Bryophyte|Moss|Liverwort|Hornwort|Fern|Lycophyte|Gymnosperm|Conifer|Angiosperm|
Monocotyledon|Eudicots|Algae|Green algae|Tree|Shrub|Herbaceous plant|Grass|
Poaceae|Orchidaceae|Asteraceae|Fabaceae|Rosaceae|Cactaceae|Arecaceae|Fagaceae|
Arabidopsis thaliana|Zea mays|Rice|Wheat|Potato|Tomato|Banana|Coffee|Cocoa bean|
Quercus|Pinus|Mangrove|Seagrass|Carnivorous plant|Parasitic plant|Epiphyte|
Succulent plant|Aquatic plant|Desert vegetation|Tropical rainforest|Temperate forest|
Boreal forest|Tundra|Grassland|Savanna|Wetland|Biodiversity|Plant conservation|
Seed bank|Botanical garden|Ecological restoration|Invasive species|Agroforestry|
Agroecology|Horticulture|Agronomy|Forestry|Ethnobotany|Domestication|Crop wild relative|
Plant breeding|Sustainable agriculture|Hydroponics|Plant tissue culture|Bonsai|
History of botany|Plant intelligence|Plant perception (physiology)'''.replace('\n', '').split('|')
CATEGORY_SEEDS = ['Botany', 'Plants', 'Plant physiology', 'Plant anatomy', 'Plant ecology',
                  'Plant reproduction', 'Plant genetics', 'Plant taxonomy', 'Plant evolution',
                  'Plant diseases', 'Plant conservation', 'Trees', 'Flowers', 'Crops',
                  'Horticulture', 'Forestry', 'Ethnobotany', 'Flora']
RESEARCH_TERMS = [
    'plant physiology', 'plant photosynthesis', 'plant genetics genomics',
    'plant roots rhizosphere', 'plant reproduction pollination', 'plant evolution taxonomy',
    'plant ecology conservation', 'plant drought salinity adaptation',
    'plant pathogens immunity', 'plant hormones development', 'forest tree ecology',
    'crop breeding agriculture', 'seed germination dormancy', 'plant mycorrhiza symbiosis',
    'bryophyte fern biology', 'aquatic plant seagrass', 'plant secondary metabolism',
    'plant tissue culture', 'plant herbivore interaction', 'plant climate change',
]
# Whole works are traversed only beneath these explicit title roots.
WORKS = {
    'Enquiry into Plants': ('plant_books', 'Theophrastus; translated by Arthur Hort'),
    'English Botany (1st edition)': ('plant_books', 'James Edward Smith and James Sowerby'),
    'History of botany (1530–1860)': ('plant_books', 'Julius von Sachs; historical translation'),
    'Life Movements in Plants': ('plant_books', 'Jagadish Chandra Bose'),
    'The Botanic Garden (Darwin, 1791)': ('plant_stories', 'Erasmus Darwin; poetry'),
    "Hans Andersen's Fairy Tales/The Fir-Tree": ('plant_stories', 'Hans Christian Andersen; historical translation'),
    'The Pink Fairy Book/The Fir-Tree': ('plant_stories', 'Andrew Lang; historical collection'),
}
PLANT_RE = re.compile(r'\b(plant\w*|botan\w*|photosynth\w*|chloroplast\w*|flora|floral|flower\w*|'
                      r'pollinat\w*|seed\w*|forest\w*|tree\w*|crop\w*|root\w*|leaf|leaves|'
                      r'angiosperm\w*|gymnosperm\w*|bryophy\w*|fern\w*|orchid\w*|'
                      r'rhiz\w*|mycorrhiz\w*|arabidopsis|maize|rice|wheat|alga\w*|seagrass\w*)\b', re.I)
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
    return ((p.hostname, p.path) in {('en.wikipedia.org', '/w/api.php'), ('en.wikisource.org', '/w/api.php')}
            or (p.hostname == 'www.ebi.ac.uk' and bool(re.fullmatch(
                r'/europepmc/webservices/rest/(search|PMC\d+/fullTextXML)', p.path))))


class RestrictedRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed_url(newurl) or urlsplit(req.full_url).hostname != urlsplit(newurl).hostname:
            raise Skip('Redirect outside the selected API')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class APIClient:
    def __init__(self, delay=1.2, contact='', retries=4, timeout=35):
        self.delay, self.retries, self.timeout = delay, retries, timeout
        self.last = 0.0
        self.opener = build_opener(RestrictedRedirect())
        self.ua = f'PlantKingdomCollector/{VERSION}' + (f' ({contact})' if contact else '')
    def request(self, url, *, xml=False, **params):
        if params:
            url += '?' + urlencode(params)
        if not allowed_url(url):
            raise Skip('Unsupported API URL')
        failure = None
        for attempt in range(self.retries):
            time.sleep(max(0, self.delay-(time.monotonic()-self.last)))
            pause = min(30, 2**(attempt+1))
            try:
                self.last = time.monotonic()
                request = Request(url, headers={'User-Agent': self.ua, 'Accept-Encoding': 'identity',
                                  'Accept': 'application/xml' if xml else 'application/json'})
                with self.opener.open(request, timeout=self.timeout) as response:
                    mime = response.headers.get_content_type()
                    if mime not in (('application/xml', 'text/xml') if xml else ('application/json', 'text/json')):
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
    minimum = min(args.min_words, 35) if bucket == 'plant_stories' else args.min_words
    reasons = []
    if len(words) < minimum:
        reasons.append('too_short')
    if len(body) and sum(c.isalpha() for c in body)/len(body) < 0.35:
        reasons.append('low_text_density')
    if bucket in ('plant_articles', 'plant_studies') and not PLANT_RE.search(title+' '+body[:10000]):
        reasons.append('no_botanical_signal')
    if re.search(r'\b(may refer to:|this disambiguation page)\b', body[:1000], re.I):
        reasons.append('disambiguation')
    content_hash = digest(' '.join(body.casefold().split()))
    existing = db.execute('SELECT id FROM documents WHERE bucket=? AND content_hash=?', (bucket, content_hash)).fetchone()
    doc_id = existing[0] if existing else 'plant:'+bucket+':'+content_hash
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
        ingest(db, f"wikipedia:{page['pageid']}", 'plant_articles', page['title'], body,
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
                    'kind': 'historical_botany' if payload['bucket']=='plant_books' else 'literature',
                    'author': payload['author'], 'root': root, 'revision': page.get('revid'),
                    'rights': 'Historic work; Wikisource transcription and jurisdiction-specific terms apply'}, args)
        for link in page.get('links', []):
            child = link.get('title', '')
            if link.get('ns') == 0 and child.startswith(root+'/'):
                enqueue(db, 'wikisource', 'work', child, task['depth']+1, payload)
        done(db, task)


def research_search(db, api, task, args):
    payload = json.loads(task['payload'])
    seen = payload.get('seen', 0)
    remaining = args.results_per_topic-seen if args.results_per_topic else 50
    if remaining <= 0:
        with db:
            done(db, task)
        return
    cursor = payload.get('cursor', '*')
    size = min(50, remaining)
    # All terms required; avoids relevance expansion into unrelated human studies.
    terms = ' AND '.join(task['target'].split())
    query = f'TITLE_ABS:({terms}) AND OPEN_ACCESS:Y AND IN_EPMC:Y'
    data = api.request(EPMC+'search', query=query, format='json', resultType='core',
                       pageSize=size, cursorMark=cursor)
    records = data.get('resultList', {}).get('result')
    if not isinstance(records, list):
        raise Deferred('Research response lacks a result list')
    next_cursor = data.get('nextCursorMark')
    if records and (not next_cursor or next_cursor == cursor):
        raise Deferred('Missing or repeated research cursor; checkpoint retained')
    with db:
        for item in records:
            pmcid = item.get('pmcid', '')
            if re.fullmatch(r'PMC\d+', pmcid) and item.get('isOpenAccess') == 'Y':
                enqueue(db, 'research', 'paper', pmcid, payload=item)
        if records and (not args.results_per_topic or seen+len(records) < args.results_per_topic):
            db.execute('UPDATE tasks SET payload=? WHERE id=?',
                       (json.dumps({'cursor': next_cursor, 'seen': seen+len(records)}), task['id']))
            # A fresh queue position lets other topics and papers progress fairly.
            db.execute('UPDATE tasks SET id=(SELECT max(id)+1 FROM tasks) WHERE id=?', (task['id'],))
        else:
            done(db, task)


def local_tag(tag):
    return tag.rsplit('}', 1)[-1]


def jats_text(node):
    # Recursive block extraction preserves section order and superscript text.
    blocked = {'ref-list', 'table-wrap', 'fig', 'supplementary-material', 'ack', 'fn-group',
               'author-notes', 'permissions', 'xref', 'ext-link'}
    blocks = {'p', 'title', 'sec', 'list-item', 'disp-quote', 'abstract'}
    def walk(el):
        if local_tag(el.tag) in blocked:
            return ''
        parts = [el.text or '']
        for child in el:
            parts.extend((walk(child), child.tail or ''))
        text = ''.join(parts)
        return '\n'+text+'\n' if local_tag(el.tag) in blocks else text
    return clean(walk(node))


def permitted_license(elements):
    evidence = []
    allowed = False
    for element in elements:
        text = ' '.join(element.itertext())
        attrs = ' '.join(str(v) for el in element.iter() for v in el.attrib.values())
        combined = (text+' '+attrs).lower()
        evidence.append(clean(text+' '+attrs))
        if re.search(r'creativecommons\.org/(?:licenses/(?:by|by-sa|by-nc|by-nc-sa)/[\d.]+|publicdomain/(?:zero|mark)/[\d.]+)', combined):
            allowed = True
        elif re.search(r'\bcc[ -]?by(?:[ -](?:nc|sa)){0,2}\b', combined) and not re.search(r'\bnd\b|no.?derivatives', combined):
            allowed = True
        elif 'creative commons attribution' in combined and not re.search(r'no.?derivatives', combined):
            allowed = True
        elif re.search(r'\bcc0\b', combined):
            allowed = True
    return allowed, evidence


def research_paper(db, api, task, args):
    raw = api.request(EPMC+task['target']+'/fullTextXML', xml=True)
    if b'<!ENTITY' in raw.upper():
        raise Skip('XML entity declarations are unsupported')
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise Skip('Malformed research XML') from exc
    for el in root.iter():
        el.tag = local_tag(el.tag)
    # Article-level permissions only: a figure's separate license is insufficient.
    licenses = root.findall('./front/article-meta/permissions/license')
    allowed, rights = permitted_license(licenses)
    if not allowed:
        raise Skip('No recognized article-level reuse license; no training text collected')
    record = json.loads(task['payload'])
    if root.attrib.get('article-type') in {'retraction', 'retraction-notice'} or str(record.get('isRetracted', '')).upper() == 'Y':
        raise Skip('Retraction notice or flagged retracted record')
    title = root.find('./front/article-meta/title-group/article-title')
    title_text = jats_text(title) if title is not None else record.get('title', task['target'])
    body_node = root.find('./body')
    if body_node is None:
        raise Skip('Full-text body unavailable')
    sections = [jats_text(e) for e in root.findall('./front/article-meta/abstract')]
    sections.append(jats_text(body_node))
    body = clean('\n\n'.join(sections))
    doi = record.get('doi', '').strip().lower()
    for identifier in root.findall('./front/article-meta/article-id'):
        if identifier.attrib.get('pub-id-type') == 'doi' and identifier.text:
            doi = identifier.text.strip().lower()
    key = 'doi:'+doi if doi else 'epmc:'+task['target']
    with db:
        ingest(db, key, 'plant_studies', title_text, body,
               {'source': 'europepmc', 'url': f"https://europepmc.org/articles/{task['target']}",
                'kind': 'research_full_text', 'rights': rights, 'doi': doi,
                'authors': record.get('authorString'), 'year': record.get('pubYear'),
                'journal': record.get('journalInfo', {}), 'pmcid': task['target'],
                'peer_review_verified': False}, args)
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
                'source': 'plant_collector', 'directory_path': '', 'retrieval_mode': 'tfidf',
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
                batch_id = 'plants-'+digest('|'.join(row['id'] for row in rows))[:24]
                metadata = json.loads(rows[0]['metadata'])
                result = self.quality.evaluate_batch(
                    records, dataset_id='plant_collector/'+first['bucket'], source_id=source,
                    batch_id=batch_id, schema=schema, use_case='knowledge_ingestion',
                    feature_fields=['word_count'],
                    provenance={'source_id': source, 'source_type': metadata['kind'],
                                'collected_at': metadata['collected_at'], 'collector': 'plant_collector',
                                'checksum': digest('|'.join(row['content_hash'] for row in rows)),
                                'uri': metadata.get('url', ''), 'schema_version': SCHEMA},
                    source_metadata={'source_id': source, 'source_type': metadata['kind']},
                    context={'route': 'plant_collector->quality->knowledge', 'external_content': True,
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
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Direct project imports, intentionally no optional-import exception fallback.
    from src.agents.agent_factory import AgentFactory
    from src.agents.collaborative.shared_memory import SharedMemory
    from logs.logger import configure_logging, get_logger
    configure_logging()
    global LOG
    LOG = get_logger('Plant Collector')
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


def export(db, folder, strict=False):
    statuses = ('pass',) if strict else ('pass', 'warn', 'standalone')
    placeholders = ','.join('?' for _ in statuses)
    counts = {}
    # One document per training file preserves document-level split boundaries in
    # LANTRA's current raw-text loader. Aggregate files are convenient reading copies.
    manifest_path = folder/'training_files.json'
    previous = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
    managed = []
    for row in db.execute(f'SELECT rowid AS sequence,* FROM documents WHERE status IN ({placeholders}) ORDER BY rowid', statuses):
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
        if isinstance(relative, str) and re.fullmatch(r'training_text/plant_(articles|studies|books|stories)/[0-9]{8,}\.txt', relative) and relative not in current:
            (folder/relative).unlink(missing_ok=True)
    with atomic_writer(manifest_path) as stream:
        json.dump(managed, stream, indent=2)
    for bucket in BUCKETS:
        count, words = 0, 0
        with atomic_writer(folder/(bucket+'.txt')) as stream:
            for row in db.execute(f'SELECT title,body FROM documents WHERE bucket=? AND status IN ({placeholders}) ORDER BY rowid',
                                  (bucket, *statuses)):
                # No URLs/IDs/checksums/rights blocks or QA reports in training text.
                stream.write(row['title']+'\n\n'+row['body']+'\n\n\n')
                count += 1
                words += len(row['body'].split())
        counts[bucket] = {'documents': count, 'words': words}
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
            buckets = {'wikipedia': ('plant_articles',), 'research': ('plant_studies',),
                       'wikisource': ('plant_books', 'plant_stories')}[source]
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
                    research_paper(db, api, task, args)
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
                export(db, args.output_dir, args.strict_quality)
                pipeline.sync(db, args.strict_quality)
                LOG.info('PROGRESS | %d tasks | %.1f minutes | pending=%d', steps,
                         (time.monotonic()-start)/60,
                         db.execute("SELECT count(*) FROM tasks WHERE status='pending'").fetchone()[0])
        if not progress:
            break


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-dir', type=Path, default=Path('plant_kingdom'))
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
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir/'collector.log', encoding='utf-8')])
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
                        export(db, args.output_dir, args.strict_quality)
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
                        export(db, args.output_dir, args.strict_quality)
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
    """Offline regression tests; no SLAI imports or remote requests."""
    import tempfile
    args = parse_args(['--agents', 'off', '--min-words', '5'])
    class Memory:
        closed = False
        def close(self):
            self.closed = True
    class Knowledge:
        def __init__(self):
            self.doc_index, self.content_hashes = {}, set()
        def add_document(self, text, doc_id=None, metadata=None):
            h = digest(text.strip())
            if h in self.content_hashes:
                return
            self.content_hashes.add(h)
            self.doc_index[doc_id] = {'text': text.strip(), 'metadata': metadata}
    class Quality:
        verdict = 'pass'
        def evaluate_batch(self, records, *, dataset_id, source_id, batch_id, schema,
                           use_case, feature_fields, provenance, source_metadata, context):
            assert use_case == 'knowledge_ingestion'
            assert all(r['source_id'] == source_id for r in records)
            assert provenance['checksum'] and 'text' in schema['fields']
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
        def __init__(self):
            self.calls = []
        def wiki(self, source='wikipedia', **params):
            self.calls.append(params)
            if params.get('list') == 'categorymembers':
                return {'query': {'categorymembers': [{'ns': 0, 'title': 'Plant'},
                    {'ns': 14, 'title': 'Category:Trees'}]}, 'continue': {'cmcontinue': 'next', 'continue': '-||'}}
            if source == 'wikisource':
                return {'parse': {'title': 'Enquiry into Plants', 'pageid': 2,
                    'text': '<p>Plants grow roots and leaves for water and light.</p><script>garbage</script>',
                    'links': [{'ns': 0, 'title': 'Enquiry into Plants/Book I'}, {'ns': 0, 'title': 'Unrelated'}]}}
            return {'query': {'pages': [{'pageid': 1, 'title': 'Plant', 'extract':
                'Plants grow roots and leaves for water and light.', 'fullurl': 'https://en.wikipedia.org/wiki/Plant'}]}}
        def request(self, url, **params):
            self.calls.append(params)
            if params.get('xml'):
                return b'''<article><front><article-meta><title-group><article-title>Plant growth research</article-title></title-group>
                <permissions><license xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="https://creativecommons.org/licenses/by/4.0/">CC BY</license></permissions>
                <abstract><p>Plant roots absorb water from soil.</p></abstract></article-meta></front>
                <body><sec><title>Results</title><p>Photosynthesis allows plants to capture sunlight for growth.</p>
                <table-wrap><p>TABLE GARBAGE</p></table-wrap></sec></body><back><ref-list><p>REFERENCES</p></ref-list></back></article>'''
            if params['cursorMark'] == '*':
                return {'resultList': {'result': [{'pmcid': 'PMC123', 'isOpenAccess': 'Y', 'doi': '10.1/plant'}]}, 'nextCursorMark': 'next'}
            return {'resultList': {'result': []}, 'nextCursorMark': 'next'}
    assert html_text('<p>Plant <em>growth</em>.</p><script>bad</script>') == 'Plant growth.'
    assert clean(' leaf\u200b\n\n root ') == 'leaf\n\nroot'
    assert not allowed_url('https://127.0.0.1/w/api.php')
    assert not allowed_url('https://www.ebi.ac.uk/europepmc/webservices/rest/../../private')
    assert allowed_url(EPMC+'PMC123/fullTextXML')
    assert not permitted_license([ET.fromstring('<license>All rights reserved</license>')])[0]
    assert not permitted_license([ET.fromstring('<license>CC BY-ND</license>')])[0]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        db = connect(folder/'history.sqlite3')
        seed(db, args)
        api = API()
        article = db.execute("SELECT * FROM tasks WHERE kind='article' LIMIT 1").fetchone()
        wiki_task(db, api, article, args)
        wiki_task(db, api, article, args)
        assert db.execute('SELECT count(*) FROM documents').fetchone()[0] == 1
        cat = db.execute("SELECT * FROM tasks WHERE kind='category' LIMIT 1").fetchone()
        wiki_task(db, api, cat, args)
        assert json.loads(db.execute('SELECT payload FROM tasks WHERE id=?', (cat['id'],)).fetchone()[0])['continue']['cmcontinue'] == 'next'
        work = db.execute("SELECT * FROM tasks WHERE target='Enquiry into Plants'").fetchone()
        work_task(db, api, work, args)
        assert db.execute("SELECT 1 FROM tasks WHERE target='Enquiry into Plants/Book I'").fetchone()
        assert not db.execute("SELECT 1 FROM tasks WHERE target='Unrelated'").fetchone()
        search = db.execute("SELECT * FROM tasks WHERE kind='search' LIMIT 1").fetchone()
        research_search(db, api, search, args)
        updated = db.execute('SELECT * FROM tasks WHERE target=? AND kind=?', (search['target'], search['kind'])).fetchone()
        assert json.loads(updated['payload'])['cursor'] == 'next'
        research_search(db, api, updated, args)
        paper = db.execute("SELECT * FROM tasks WHERE kind='paper'").fetchone()
        research_paper(db, api, paper, args)
        scientific = db.execute("SELECT body FROM documents WHERE bucket='plant_studies'").fetchone()[0]
        assert 'Results' in scientific and 'TABLE GARBAGE' not in scientific and 'REFERENCES' not in scientific
        with db:
            ingest(db, 'fiction', 'plant_stories', 'A talking tree',
                   'Plants grow roots and leaves for water and light.',
                   {'source': 'wikisource', 'kind': 'literature', 'url': 'https://example.org'}, args)
        factory, memory = Factory(), Memory()
        team = AgentPipeline(factory, memory, 'team', owns=True)
        team.assess(db)
        team.sync(db)
        assert [c[0] for c in factory.calls] == ['quality', 'knowledge']
        assert all(c[1] is memory for c in factory.calls)
        assert len(factory.k.doc_index) == 4
        team.sync(db)
        assert len(factory.k.doc_index) == 4
        counts = export(db, folder)
        assert all(v['documents'] == 1 for v in counts.values())
        assert len(list((folder/'training_text').rglob('*.txt'))) == 4
        for file in folder.glob('*.txt'):
            text = file.read_text()
            assert 'https://' not in text and 'sha256' not in text and 'rights' not in text
        with db:
            ingest(db, 'blocked', 'plant_articles', 'Plant candidate', 'Plant roots need water and minerals from soil.',
                   {'source': 'wikipedia', 'kind': 'encyclopedia'}, args)
        factory.q.verdict = 'block'
        team.assess(db)
        counts = export(db, folder)
        assert counts['plant_articles']['documents'] == 1
        # Strict export excludes warnings and removes their managed training files.
        with db:
            db.execute("UPDATE documents SET status='warn' WHERE bucket='plant_books'")
        strict_counts = export(db, folder, strict=True)
        assert strict_counts['plant_books']['documents'] == 0
        assert not list((folder/'training_text'/'plant_books').glob('*.txt'))
        export(db, folder)
        assert len(list((folder/'training_text').rglob('*.txt'))) == 4
        team.close()
        team.close()
        assert factory.closed and memory.closed
        db.close()
        db = connect(folder/'history.sqlite3')
        replay = AgentPipeline(Factory(), Memory(), 'team')
        replay.sync(db)
        assert replay.knowledge is not None
        assert len(replay.knowledge.doc_index) == 4
        # A broken Quality contract preserves pending documents for recovery.
        with db:
            ingest(db, 'pending', 'plant_articles', 'Plant flowers', 'Plant flowers attract many insects for pollination.',
                   {'source': 'wikipedia', 'kind': 'encyclopedia'}, args)
        assert replay.quality is not None
        replay.quality.verdict = 'unknown'
        try:
            replay.assess(db)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Invalid verdict accepted')
        assert db.execute("SELECT count(*) FROM documents WHERE status='pending'").fetchone()[0] == 1
        db.close()
    print('PASS: source parsing, XML full text, scope, pagination/resume, dedup, content-only exports, quality blocking, factory sharing, Knowledge acknowledgement/replay, and failure durability.')


if __name__ == '__main__':
    sys.exit(main())
