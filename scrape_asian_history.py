#!/usr/bin/env python3
"""Collect Asian history into six UTF-8 text files. Python 3.10+. SLAI dependencies required for Knowledge Agent (enabled by default).

QUICK START (Windows / PowerShell):
    py scrape_asian_history.py
LARGER COLLECTION (limits are totals PER REGION, not per execution):
    py scrape_asian_history.py --max-articles 25000 --max-categories 10000 --depth 4
NO ARTICLE/CATEGORY COUNT CAP (category depth still bounds traversal):
    py scrape_asian_history.py --max-articles 0 --max-categories 0 --depth 4
RESUME: run the same command again. Increasing depth/limits expands existing work.
EXPORT ONLY: py scrape_asian_history.py --export-only
SELF TEST: py scrape_asian_history.py --self-test

KNOWLEDGE AGENT (run from SLAI root in its installed environment):
    py scrape_asian_history.py
    py scrape_asian_history.py --knowledge-only --knowledge-query "Silk Road trade"
    py scrape_asian_history.py --knowledge-agent off  # standalone, standard library only
    py scrape_asian_history.py --slai-root G:\\GAIA\\AI\\SLAI
The factory creates the Knowledge Agent once. Cached articles rebuild its in-memory
retrieval index on every run; new articles are indexed after SQLite commits.
Indexing is NOT model training, fact verification or durable global knowledge-memory
synchronization. The persistent corpus remains history.sqlite3. A failed agent
operation stops the run without deleting collected text; restart replays it.

Output directory defaults to ./asian_history, with six UTF-8 corpus files:
    north_asia.txt          Russia, Siberia and the Russian Far East
    east_asia.txt           China, Mongolia, the Koreas, Japan and Taiwan
    central_asia.txt        Kazakhstan, Uzbekistan, Turkmenistan, Tajikistan, Kyrgyzstan
    western_asia.txt        West Asia, the Caucasus and Cyprus
    southern_asia.txt       South Asia, including Sri Lanka and the Maldives
    southeastern_asia.txt   Southeast Asia, including Thailand and East Timor
Use --output-dir for another directory. Do not run two instances in one directory.
Existing six exports are rebuilt from the database, not merged with manual edits.
Keep history.sqlite3 to resume. Use a new output directory for a fresh collection.

SCOPE AND LIMITS
This is an Internet collector using the official English Wikipedia Action API,
not a general search-engine crawler. It discovers history category members and
nested categories, and seeds country overview articles explicitly. TextExtracts
provides article prose, not just introductions; tables, images, some templates,
and other complex markup may be omitted. No machine-generated history is added.
It is a source corpus, not a verified, comprehensive or chronological history.
Category membership is imperfect: deeper traversal increases topic drift.
Pages shared by regions can occur once in each relevant regional file. Within a
region, canonical page IDs and identical normalized text prevent duplicates.
English coverage is uneven; Indigenous and non-English sources are underrepresented.
The collector does not follow external citations or download paywalled material.

All downloaded articles have title, canonical URL, attribution/history URL,
retrieval time, revision metadata, discovery category and content SHA-256. Revision
metadata is returned in the same API query, but the extract itself may be cached
and is not guaranteed to match that exact revision. The text is a plain-text
transformation of Wikipedia content; retain source attribution and applicable
license terms when reusing it. See each source's history for contributors.

DEFAULTS AND RECOVERY
5,000 stored articles and 2,000 expanded categories per region, depth 3. A large
run can take many hours and substantial disk space. Requests are serial with a
minimum 1-second interval, maxlag, timeout, and bounded exponential retries.
HTTP 403 is not bypassed. Set --contact to your email or public project URL so
Wikimedia can identify your client. Pending work remains pending on network errors.
Ctrl+C exports committed results; forcibly killed runs can re-export from SQLite.
Every 100 newly saved regional articles, all six files are atomically refreshed.
Previously saved articles are cached; resuming expands coverage, not freshness.

Implementation references:
https://www.mediawiki.org/wiki/API:Categorymembers
https://www.mediawiki.org/wiki/API:Etiquette
https://www.mediawiki.org/wiki/Extension:TextExtracts#API
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import sys
import time

from pathlib import Path
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from contextlib import contextmanager

API = 'https://en.wikipedia.org/w/api.php'
REGIONS = {
    'north_asia': ['Russia'],
    'east_asia': ['Mongolia', 'China', 'North Korea', 'South Korea', 'Japan', 'Taiwan'],
    'central_asia': ['Kazakhstan', 'Uzbekistan', 'Turkmenistan', 'Tajikistan', 'Kyrgyzstan'],
    'western_asia': ['Armenia', 'Georgia', 'Azerbaijan', 'Kurdistan', 'Iran', 'Iraq',
                    'Syria', 'Turkey', 'Lebanon', 'Jordan', 'Israel', 'Palestine',
                    'Gaza', 'Kuwait', 'Bahrain', 'the United Arab Emirates', 'Oman',
                    'Yemen', 'Saudi Arabia', 'Cyprus'],
    'southern_asia': ['Afghanistan', 'Pakistan', 'Nepal', 'Bhutan', 'Bangladesh',
                     'India', 'Sri Lanka', 'the Maldives'],
    'southeastern_asia': ['Myanmar', 'Laos', 'Vietnam', 'Cambodia', 'Thailand',
                         'Malaysia', 'the Philippines', 'Brunei', 'Singapore',
                         'Indonesia', 'East Timor'],
}
# Historical topics intentionally overlap modern regions. They are discovery seeds,
# not endorsements of borders or claims that an empire belongs to one modern state.
EXTRA = {
    'north_asia': ['History of Siberia', 'Russian Far East', 'Indigenous peoples of Siberia',
                   'Russian conquest of Siberia', 'Khanate of Sibir', 'Russian Empire',
                   'Soviet Union', 'History of the Soviet Union', 'Trans-Siberian Railway'],
    'east_asia': ['History of East Asia', 'Imperial China', 'Silk Road', 'Xiongnu',
                  'Han dynasty', 'Tang dynasty', 'Song dynasty', 'Yuan dynasty',
                  'Ming dynasty', 'Qing dynasty', 'Mongol Empire', 'History of Korea',
                  'Three Kingdoms of Korea', 'Goryeo', 'Joseon', 'Tokugawa shogunate',
                  'Meiji Restoration', 'Empire of Japan', 'History of Tibet',
                  'History of Manchuria', 'Ainu people', 'Ryukyu Kingdom'],
    'central_asia': ['History of Central Asia', 'Silk Road', 'Sogdia', 'Bactria',
                     'Scythians', 'Göktürks', 'Turkic Khaganate', 'Kara-Khanid Khanate',
                     'Khwarazmian Empire', 'Timurid Empire', 'Mongol Empire',
                     'Kazakh Khanate', 'Khanate of Bukhara', 'Khanate of Khiva',
                     'Khanate of Kokand', 'Russian conquest of Central Asia', 'Soviet Union'],
    'western_asia': ['History of West Asia', 'Ancient Near East', 'Mesopotamia',
                     'Sumer', 'Akkadian Empire', 'Babylonia', 'Assyria', 'Canaan',
                     'Phoenicia', 'Hittites', 'Urartu', 'Achaemenid Empire',
                     'Parthian Empire', 'Sasanian Empire', 'Byzantine Empire',
                     'Rashidun Caliphate', 'Umayyad Caliphate', 'Abbasid Caliphate',
                     'Seljuk Empire', 'Ottoman Empire', 'Safavid Iran',
                     'History of the Levant', 'History of Arabia', 'History of the Caucasus'],
    'southern_asia': ['History of South Asia', 'Indus Valley Civilisation',
                      'Vedic period', 'Mahajanapadas', 'Maurya Empire', 'Gupta Empire',
                      'Kushan Empire', 'Chola dynasty', 'Pala Empire', 'Delhi Sultanate',
                      'Vijayanagara Empire', 'Mughal Empire', 'Maratha Empire',
                      'Sikh Empire', 'Durrani Empire', 'British Raj', 'Partition of India',
                      'Indian independence movement', 'Bengal', 'Kingdom of Kandy'],
    'southeastern_asia': ['History of Southeast Asia', 'Funan', 'Chenla', 'Khmer Empire',
                          'Champa', 'Srivijaya', 'Majapahit', 'Dvaravati', 'Pagan Kingdom',
                          'Ayutthaya Kingdom', 'Sukhothai Kingdom', 'Lan Xang',
                          'Malacca Sultanate', 'Dutch East Indies', 'French Indochina',
                          'Spanish East Indies', 'Japanese occupation of Southeast Asia',
                          'Vietnam War', 'ASEAN'],
}
SKIP_CATEGORY = re.compile(r'\b(wikipedia|articles|pages|stubs|templates|births|deaths|'
                           r'living people|alumni)\b', re.I)
LOG = logging.getLogger('asian_history')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(text):
    return hashlib.sha256(' '.join(text.split()).encode('utf-8')).hexdigest()


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS categories (
          region TEXT, title TEXT, depth INTEGER, continuation TEXT,
          done INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(region,title));
        CREATE TABLE IF NOT EXISTS queue (
          region TEXT, title TEXT, origin TEXT, done INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(region,title));
        CREATE TABLE IF NOT EXISTS articles (
          pageid INTEGER PRIMARY KEY, title TEXT, url TEXT, revid INTEGER,
          revision_time TEXT, retrieved TEXT, text TEXT, hash TEXT);
        CREATE TABLE IF NOT EXISTS membership (
          region TEXT, pageid INTEGER, origin TEXT, hash TEXT,
          PRIMARY KEY(region,pageid), UNIQUE(region,hash));
        CREATE TABLE IF NOT EXISTS aliases (title TEXT PRIMARY KEY, pageid INTEGER);
        CREATE INDEX IF NOT EXISTS articles_hash ON articles(hash);
        CREATE INDEX IF NOT EXISTS membership_page ON membership(pageid);
        CREATE INDEX IF NOT EXISTS queue_pending ON queue(region,done);
        CREATE INDEX IF NOT EXISTS categories_pending ON categories(region,done,depth);
    ''')
    regions = {row[0] for row in db.execute(
        'SELECT region FROM membership UNION SELECT region FROM queue UNION SELECT region FROM categories')}
    if not regions.issubset(REGIONS):
        db.close()
        raise RuntimeError('This database contains non-Asian regions. Use a separate --output-dir.')
    return db


def seed(db):
    with db:
        for region, places in REGIONS.items():
            for place in places:
                title = 'History of ' + place
                db.execute('INSERT OR IGNORE INTO categories(region,title,depth) VALUES(?,?,0)',
                           (region, 'Category:' + title))
                db.execute('INSERT OR IGNORE INTO queue(region,title,origin) VALUES(?,?,?)',
                           (region, title, 'Country/region overview'))
            for title in EXTRA.get(region, ()):
                db.execute('INSERT OR IGNORE INTO queue(region,title,origin) VALUES(?,?,?)',
                           (region, title, 'Historical topic seed'))
                db.execute('INSERT OR IGNORE INTO categories(region,title,depth) VALUES(?,?,0)',
                           (region, 'Category:' + title))


class APIClient:
    def __init__(self, contact='', delay=1.0, retries=6):
        self.user_agent = 'AsianHistoryCollector/2.0' + (f' ({contact})' if contact else '')
        self.delay, self.retries, self.last = delay, retries, 0.0

    def query(self, **params):
        url = API + '?' + urlencode(dict(action='query', format='json',
                                         formatversion=2, maxlag=5, **params))
        last_error = None
        for attempt in range(self.retries):
            time.sleep(max(0, self.delay - (time.monotonic() - self.last)))
            retry_after = 0.0
            try:
                self.last = time.monotonic()
                req = Request(url, headers={'User-Agent': self.user_agent,
                                            'Accept': 'application/json'})
                with urlopen(req, timeout=60) as response:
                    data = json.load(response)
                if 'error' in data:
                    err = data['error']
                    if err.get('code') not in ('maxlag', 'ratelimited', 'readonly'):
                        raise RuntimeError(f'API error: {err}')
                    raise URLError(str(err))
                if 'warnings' in data:
                    LOG.warning('API warning: %s', data['warnings'])
                if 'query' not in data:
                    raise URLError('API response is missing query data')
                return data
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f'HTTP {exc.code}. Access refused; no bypass attempted. '
                                       'Check connectivity and --contact.') from exc
                raw = exc.headers.get('Retry-After', '0')
                try:
                    retry_after = float(raw)
                except ValueError:
                    try:
                        retry_after = max(0, parsedate_to_datetime(raw).timestamp() - time.time())
                    except (ValueError, TypeError, OverflowError):
                        pass
                last_error = exc
            except (URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = exc
            if attempt + 1 < self.retries:
                pause = max(retry_after, min(120, 2 ** (attempt + 1)) + random.random())
                LOG.warning('Request failed (%s); retry in %.1fs', last_error, pause)
                time.sleep(pause)
        raise RuntimeError(f'Request failed after {self.retries} attempts: {last_error}')


def expand_category(db, client, row):
    params = dict(list='categorymembers', cmtitle=row['title'],
                  cmtype='page|subcat', cmnamespace='0|14', cmlimit=500)
    if row['continuation']:
        params.update(json.loads(row['continuation']))
    data = client.query(**params)
    if 'categorymembers' not in data['query']:
        raise RuntimeError('Missing categorymembers in API response')
    continuation = data.get('continue')
    with db:
        for member in data['query']['categorymembers']:
            title = member['title']
            if member['ns'] == 0:
                db.execute('INSERT OR IGNORE INTO queue(region,title,origin) VALUES(?,?,?)',
                           (row['region'], title, row['title']))
            elif member['ns'] == 14 and not SKIP_CATEGORY.search(title):
                # Keep boundary children so a later --depth increase works without resetting.
                db.execute('''INSERT INTO categories(region,title,depth) VALUES(?,?,?)
                              ON CONFLICT(region,title) DO UPDATE SET depth=min(depth,excluded.depth)''',
                           (row['region'], title, row['depth'] + 1))
        db.execute('UPDATE categories SET continuation=?,done=? WHERE region=? AND title=?',
                   (json.dumps(continuation) if continuation else None,
                    int(not continuation), row['region'], row['title']))
    LOG.info('%s: explored %s%s', row['region'], row['title'],
             ' (more members pending)' if continuation else '')


def fetch_article(db, client, row, knowledge=None):
    article = db.execute('''SELECT a.* FROM aliases x JOIN articles a ON a.pageid=x.pageid
                            WHERE x.title=?''', (row['title'],)).fetchone()
    if article is None:
        data = client.query(titles=row['title'], redirects=1,
                            prop='extracts|info|revisions|pageprops', inprop='url',
                            explaintext=1, exsectionformat='plain', exlimit=1,
                            rvprop='ids|timestamp', rvlimit=1, ppprop='disambiguation')
        pages = data['query'].get('pages')
        if not pages:
            raise RuntimeError('Missing pages in article API response')
        page = pages[0]
        if page.get('missing') or page.get('invalid') or 'disambiguation' in page.get('pageprops', {}):
            LOG.info('Skipped missing/disambiguation page: %s', row['title'])
            with db:
                db.execute('UPDATE queue SET done=1 WHERE region=? AND title=?',
                           (row['region'], row['title']))
            return False
        content = page.get('extract', '').strip()
        if not content:
            # Do not mark done; transient empty extracts must not silently lose an article.
            raise RuntimeError(f'Empty extract for {row["title"]}; retry on the next run.')
        revision = page.get('revisions', [{}])[0]
        article = dict(pageid=page['pageid'], title=page['title'],
                       url=page.get('fullurl', f'https://en.wikipedia.org/?curid={page["pageid"]}'),
                       revid=revision.get('revid'), revision_time=revision.get('timestamp'),
                       retrieved=utcnow(), text=content, hash=digest(content))
    with db:
        db.execute('''INSERT OR IGNORE INTO articles
                      (pageid,title,url,revid,revision_time,retrieved,text,hash)
                      VALUES(:pageid,:title,:url,:revid,:revision_time,:retrieved,:text,:hash)''',
                   dict(article))
        # Reuse the original stored snapshot if another alias resolves to an existing page.
        article = db.execute('SELECT * FROM articles WHERE pageid=?', (article['pageid'],)).fetchone()
        db.execute('INSERT OR REPLACE INTO aliases VALUES(?,?)', (row['title'], article['pageid']))
        cursor = db.execute('INSERT OR IGNORE INTO membership VALUES(?,?,?,?)',
                            (row['region'], article['pageid'], row['origin'], article['hash']))
        saved = cursor.rowcount > 0
        db.execute('UPDATE queue SET done=1 WHERE region=? AND title=?',
                   (row['region'], row['title']))
    if knowledge is not None:
        # The corpus transaction is durable before the agent is invoked.
        knowledge.ingest(db, article)
    if saved:
        LOG.info('%s: saved %s', row['region'], article['title'])
    return saved


def export(db, directory):
    for region, places in REGIONS.items():
        path = directory / (region + '.txt')
        temporary = path.with_suffix('.txt.tmp')
        count = db.execute('SELECT count(*) FROM membership WHERE region=?', (region,)).fetchone()[0]
        with temporary.open('w', encoding='utf-8', newline='\n') as output:
            output.write(f'{region.upper()} — HISTORICAL SOURCE CORPUS\n'
                         f'Geographic seeds: {", ".join(places)}\n'
                         f'Exported (UTC): {utcnow()}\nArticles: {count}\n'
                         'Source: English Wikipedia and its contributors.\n'
                         'License: CC BY-SA 4.0; consult individual source notices.\n'
                         'https://creativecommons.org/licenses/by-sa/4.0/\n'
                         'Transformation: extracted plain text; complex markup may be omitted.\n'
                         'Category-based collection; not comprehensive or independently verified.\n\n')
            for a in db.execute('''SELECT a.*,m.origin FROM membership m JOIN articles a
                                   ON a.pageid=m.pageid WHERE m.region=?
                                   ORDER BY a.title COLLATE NOCASE,a.pageid''', (region,)):
                output.write('=' * 80 + '\n' + a['title'] + '\n' + '=' * 80 + '\n')
                output.write(f'Source: {a["url"]}\n'
                             f'Authors/history: https://en.wikipedia.org/w/index.php?curid={a["pageid"]}&action=history\n'
                             f'Revision metadata: {a["revid"]} ({a["revision_time"]})\n'
                             f'Retrieved (UTC): {a["retrieved"]}\n'
                             f'Discovered through: {a["origin"]}\n'
                             f'SHA-256 (whitespace-normalized text): {a["hash"]}\n\n'
                             f'{a["text"]}\n\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    LOG.info('Exported six text files to %s', directory)


def collect(db, client, args, knowledge=None):
    counts = {r: db.execute('SELECT count(*) FROM membership WHERE region=?', (r,)).fetchone()[0]
              for r in REGIONS}
    completed = {r: db.execute('SELECT count(*) FROM categories WHERE region=? AND done=1',
                              (r,)).fetchone()[0] for r in REGIONS}
    new_articles = 0
    while True:
        progress = False
        for region in REGIONS:  # Fair scheduling across regions, including after failures.
            if args.max_articles and counts[region] >= args.max_articles:
                continue
            row = db.execute('SELECT * FROM queue WHERE region=? AND done=0 ORDER BY rowid LIMIT 1',
                             (region,)).fetchone()
            if row:
                saved = fetch_article(db, client, row, knowledge=knowledge)
                counts[region] += int(saved)
                new_articles += int(saved)
                progress = True
                if saved and new_articles % 100 == 0:
                    export(db, args.output_dir)
            elif not args.max_categories or completed[region] < args.max_categories:
                row = db.execute('''SELECT * FROM categories WHERE region=? AND done=0 AND depth<=?
                                    ORDER BY depth,rowid LIMIT 1''', (region, args.depth)).fetchone()
                if row:
                    expand_category(db, client, row)
                    done = db.execute('SELECT done FROM categories WHERE region=? AND title=?',
                                      (region, row['title'])).fetchone()[0]
                    completed[region] += int(done)
                    progress = True
        if not progress:
            break
    for region in REGIONS:
        pending = db.execute('SELECT count(*) FROM queue WHERE region=? AND done=0', (region,)).fetchone()[0]
        LOG.info('%s: %d articles saved, %d categories completed, %d queued articles remain',
                 region, counts[region], completed[region], pending)
    LOG.info('Finished within configured bounds. Increase limits/depth to expand coverage.')


def self_test():
    import tempfile
    class FakeAPI:
        def __init__(self):
            self.calls = []
        def query(self, **params):
            self.calls.append(params)
            if params.get('list') == 'categorymembers':
                if 'cmcontinue' not in params:
                    return {'query': {'categorymembers': [
                        {'ns': 0, 'title': 'Example'}, {'ns': 14, 'title': 'Category:Child'}]},
                        'continue': {'cmcontinue': 'next', 'continue': '-||'}}
                return {'query': {'categorymembers': []}}
            return {'query': {'pages': [{'pageid': 1, 'title': 'Example',
                    'fullurl': 'https://en.wikipedia.org/wiki/Example',
                    'extract': 'Historical text with Unicode: Taíno, Curaçao.',
                    'revisions': [{'revid': 10, 'timestamp': '2020-01-01T00:00:00Z'}]}]}}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        db = connect(root / 'history.sqlite3')
        fake = FakeAPI()
        with db:
            db.execute("INSERT INTO categories(region,title,depth) VALUES('north_asia','Category:Root',0)")
        row = db.execute('SELECT * FROM categories').fetchone()
        expand_category(db, fake, row)
        assert db.execute('SELECT continuation FROM categories WHERE title=?', (row['title'],)).fetchone()[0]
        assert db.execute("SELECT depth FROM categories WHERE title='Category:Child'").fetchone()[0] == 1
        expand_category(db, fake, db.execute('SELECT * FROM categories WHERE title=?', (row['title'],)).fetchone())
        assert fake.calls[-1]['cmcontinue'] == 'next'
        row = db.execute('SELECT * FROM queue').fetchone()
        assert fetch_article(db, fake, row)
        assert not fetch_article(db, fake, row)
        export(db, root)
        original = (root / 'north_asia.txt').read_text(encoding='utf-8')
        assert original.count('Historical text with Unicode:') == 1 and 'Curaçao' in original
        assert len(list(root.glob('*.txt'))) == len(REGIONS)
        db.close()
        db = connect(root / 'history.sqlite3')
        assert db.execute('SELECT count(*) FROM membership').fetchone()[0] == 1
        assert db.execute('SELECT done FROM queue').fetchone()[0] == 1
        export(db, root)
        assert (root / 'north_asia.txt').read_text(encoding='utf-8').count('Historical text with Unicode:') == 1
        db.close()
    knowledge_self_test()
    print('PASS: pagination, depth frontier, duplicate filtering, Unicode, six exports, restart persistence.')


def find_slai_root(explicit=None):
    if explicit is not None:
        candidates = [Path(explicit).expanduser().resolve()]
    else:
        candidates = []
        for start in (Path(__file__).resolve().parent, Path.cwd()):
            candidates.extend([start, *start.parents])
    for root in dict.fromkeys(candidates):
        if (root / 'src/agents/agent_factory.py').is_file() and (root / 'src/agents/knowledge_agent.py').is_file():
            return root
    raise RuntimeError('SLAI root not found. Put this script in SLAI root, use --slai-root, '
                       'or explicitly select --knowledge-agent off for standalone collection')


@contextmanager
def corpus_lock(directory):
    """OS lock releases on crash; protects export temporary files and one corpus writer."""
    with (directory / 'collector.lock').open('a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if not handle.tell():
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another process is using this corpus directory') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class KnowledgeBridge:
    """Factory-owned retrieval integration, with SQLite as its replayable source.

    Uses the current SLAI-v.2.3 factory create()/shutdown(), KnowledgeAgent
    add_document()/retrieve()/doc_index and SharedMemory.close() contracts.
    No constructor fallback, generic execute() prompt, extracted triples, or
    synthesized facts. A caller can inject a factory and memory without granting
    this bridge ownership of the caller's runtime.
    """
    def __init__(self, factory, shared_memory, *, owns_runtime=False):
        self.factory = factory
        self.shared_memory = shared_memory
        self.owns_runtime = owns_runtime
        self.closed = False
        try:
            self.agent = factory.create('knowledge', shared_memory=shared_memory, config={
                'source': 'asian_history_collector',
                'retrieval_mode': 'tfidf',
                'directory_path': '',  # Articles are ingested separately, not six giant files.
                'bias_detection_enabled': False,
                'use_ontology_expansion': False,
            })
            if not callable(getattr(self.agent, 'add_document', None)) or not callable(getattr(self.agent, 'retrieve', None)):
                raise RuntimeError('Factory Knowledge Agent does not expose add_document()/retrieve()')
            if not isinstance(getattr(self.agent, 'doc_index', None), dict):
                raise RuntimeError('Knowledge Agent must expose its in-process doc_index for ingestion verification')
        except BaseException:
            if owns_runtime:
                try:
                    factory.shutdown()
                finally:
                    shared_memory.close()
            raise

    @classmethod
    def from_slai(cls, root):
        root = str(root)
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from src.agents.agent_factory import AgentFactory
            from src.agents.collaborative.shared_memory import SharedMemory
            from logs.logger import configure_logging, get_logger
            configure_logging()
            get_logger('asian_history').info('Creating Knowledge Agent through SLAI AgentFactory')
            shared = SharedMemory()
            try:
                factory = AgentFactory()
            except BaseException:
                shared.close()
                raise
            bridge = cls(factory, shared, owns_runtime=True)
            LOG.info('Knowledge Agent ready (factory-managed, TF-IDF)')
            return bridge
        except ImportError as exc:
            raise RuntimeError('SLAI dependency import failed. Run in your installed SLAI environment. '
                               'Use --knowledge-agent off only if you want collection without the agent. '
                               f'Original import error: {exc}') from exc

    def ingest(self, db, article):
        if self.closed:
            raise RuntimeError('Knowledge bridge is closed')
        doc_id = 'history:asia:sha256:' + article['hash']
        provenance = [dict(row) for row in db.execute('''
            SELECT DISTINCT a.pageid,a.title,a.url,a.revid,a.revision_time,a.retrieved,m.region,m.origin
            FROM articles a JOIN membership m ON m.pageid=a.pageid
            WHERE a.hash=? ORDER BY a.pageid,m.region
        ''', (article['hash'],))]
        metadata = {
            'source': 'english_wikipedia', 'collector': 'asian_history_collector',
            'type': 'historical_source', 'title': article['title'], 'url': article['url'],
            'page_id': article['pageid'], 'revision_id': article['revid'],
            'retrieved_utc': article['retrieved'], 'content_sha256': article['hash'],
            'regions': sorted({row['region'] for row in provenance}),
            'provenance': provenance,
            'license': 'CC BY-SA 4.0; consult individual page notices',
            'license_url': 'https://creativecommons.org/licenses/by-sa/4.0/',
            'contributors_url': f'https://en.wikipedia.org/w/index.php?curid={article["pageid"]}&action=history',
            'verified': False, 'trust': 'external_source_unverified',
        }
        existing = self.agent.doc_index.get(doc_id)
        if existing is None:
            self.agent.add_document(text=article['text'], doc_id=doc_id, metadata=metadata)
            # add_document returns None and can skip invalid input: do not mistake
            # a no-exception return for successful ingestion.
            existing = self.agent.doc_index.get(doc_id)
        if not isinstance(existing, dict) or digest(existing.get('text', '')) != article['hash']:
            raise RuntimeError(f'Knowledge Agent did not index expected document {doc_id}')
        # The same article may later enter another region. Refresh its provenance
        # without reindexing text or adding another retrieval document.
        existing.setdefault('metadata', {}).update(metadata)
        return doc_id

    def hydrate(self, db):
        count = 0
        for article in db.execute('SELECT a.* FROM articles a WHERE EXISTS '
                                  '(SELECT 1 FROM membership m WHERE m.pageid=a.pageid) ORDER BY a.pageid'):
            self.ingest(db, article)
            count += 1
            if count % 100 == 0:
                LOG.info('Knowledge index replay: %d saved articles processed', count)
        LOG.info('Knowledge index replay complete: %d saved articles processed', count)

    def query(self, text, top_k=5):
        return [{'score': float(score), 'doc_id': doc['doc_id'],
                 'title': doc.get('metadata', {}).get('title'),
                 'url': doc.get('metadata', {}).get('url'),
                 'regions': doc.get('metadata', {}).get('regions', []),
                 'excerpt': doc['text'][:1000]}
                for score, doc in self.agent.retrieve(query=text, k=top_k)]

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.owns_runtime:
            try:
                self.factory.shutdown()  # Factory releases the agent and its resources.
            finally:
                self.shared_memory.close()


def knowledge_self_test():
    """Contract doubles test lifecycle and restart behavior without importing SLAI."""
    import tempfile
    class FakeMemory:
        closed = False
        def close(self):
            self.closed = True
    class FakeAgent:
        def __init__(self):
            self.doc_index = {}
            self.calls = 0
        def add_document(self, text, doc_id=None, metadata=None):
            self.calls += 1
            self.doc_index.setdefault(doc_id, {'doc_id': doc_id, 'text': text,
                                               'metadata': dict(metadata or {})})
        def retrieve(self, _query, k=5):
            return [(1.0, d) for d in list(self.doc_index.values())[:k]]
    class FakeFactory:
        def __init__(self):
            self.agent = FakeAgent()
            self.closed = False
            self.created = 0
        def create(self, kind, shared_memory=None, **kwargs):
            assert kind == 'knowledge' and isinstance(shared_memory, FakeMemory)
            assert kwargs['config']['retrieval_mode'] == 'tfidf'
            self.created += 1
            return self.agent
        def shutdown(self):
            self.closed = True
    with tempfile.TemporaryDirectory() as directory:
        db = connect(Path(directory) / 'history.sqlite3')
        seed(db)
        assert set(EXTRA) == set(REGIONS) and all(EXTRA.values())
        assert db.execute('SELECT count(DISTINCT region) FROM queue').fetchone()[0] == 6
        factory, memory = FakeFactory(), FakeMemory()
        bridge = KnowledgeBridge(factory, memory, owns_runtime=True)
        class FixtureAPI:
            def query(self, **_params):
                return {'query': {'pages': [{'pageid': 42, 'title': 'Silk Road',
                         'extract': 'Historic trade routes across Asia.', 'fullurl': 'https://en.wikipedia.org/wiki/Silk_Road',
                         'revisions': [{'revid': 99, 'timestamp': '2026-01-01T00:00:00Z'}]}]}}
        row = {'region': 'central_asia', 'title': 'Silk Road', 'origin': 'test'}
        assert fetch_article(db, FixtureAPI(), row, knowledge=bridge)
        row2 = dict(row, region='east_asia')
        assert fetch_article(db, FixtureAPI(), row2, knowledge=bridge)
        assert factory.created == 1 and factory.agent.calls == 1
        doc = next(iter(factory.agent.doc_index.values()))
        assert doc['metadata']['regions'] == ['central_asia', 'east_asia']
        assert doc['metadata']['verified'] is False
        assert bridge.query('Silk Road')[0]['url'].endswith('Silk_Road')
        bridge.close(); bridge.close()
        assert factory.closed and memory.closed
        # A fresh runtime must rebuild the index even though every article was previously sent.
        restarted = FakeFactory()
        bridge2 = KnowledgeBridge(restarted, FakeMemory())
        bridge2.hydrate(db)
        assert restarted.agent.calls == 1
        bridge2.close()
        assert not restarted.closed  # Injected application runtime is not owned.
        class FailingBridge:
            def ingest(self, *_args):
                raise RuntimeError('simulated indexing failure')
        try:
            fetch_article(db, FixtureAPI(), dict(row, region='western_asia'), knowledge=FailingBridge())
        except RuntimeError:
            pass
        else:
            raise AssertionError('Agent failure was hidden')
        assert db.execute("SELECT count(*) FROM membership WHERE region='western_asia'").fetchone()[0] == 1
        fresh = FakeFactory()
        KnowledgeBridge(fresh, FakeMemory()).hydrate(db)
        assert 'western_asia' in next(iter(fresh.agent.doc_index.values()))['metadata']['regions']
        # A silent no-op add_document must not count as a successful ingestion.
        bad = FakeFactory()
        bad.agent.add_document = lambda text, doc_id=None, metadata=None: None
        try:
            KnowledgeBridge(bad, FakeMemory()).hydrate(db)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Unacknowledged ingestion was accepted')
        db.close()
    print('PASS: factory creation, provenance merge, restart replay, failure durability, query, ownership/cleanup.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, default=Path('asian_history'))
    parser.add_argument('--max-articles', type=int, default=5000, help='Per region; 0 means unlimited')
    parser.add_argument('--max-categories', type=int, default=2000, help='Per region; 0 means unlimited')
    parser.add_argument('--depth', type=int, default=3, help='Maximum category nesting below seeds')
    parser.add_argument('--delay', type=float, default=1.0, help='Request spacing in seconds, minimum 1')
    parser.add_argument('--contact', default='', help='Your email or public project URL for User-Agent')
    parser.add_argument('--export-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--slai-root', type=Path, help='SLAI checkout root; otherwise find it among script/cwd ancestors')
    parser.add_argument('--knowledge-agent', choices=('required', 'off'), default='required')
    parser.add_argument('--knowledge-only', action='store_true', help='Index the saved corpus without downloading')
    parser.add_argument('--knowledge-query', help='Retrieve matching documents after ingestion')
    parser.add_argument('--knowledge-top-k', type=int, default=5)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if min(args.max_articles, args.max_categories, args.depth) < 0 or not 1 <= args.delay < float('inf'):
        parser.error('Limits and depth must be nonnegative; delay must be finite and at least 1 second.')
    if any(ord(c) < 32 or ord(c) > 126 for c in args.contact):
        parser.error('--contact must contain printable ASCII only')
    if args.knowledge_top_k < 1:
        parser.error('--knowledge-top-k must be positive')
    if args.export_only and (args.knowledge_only or args.knowledge_query):
        parser.error('--export-only cannot be combined with Knowledge Agent operations')
    if args.knowledge_agent == 'off' and (args.knowledge_only or args.knowledge_query):
        parser.error('Knowledge operations require --knowledge-agent required')
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(args.output_dir / 'collector.log', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    console = logging.StreamHandler()
    for sink in (handler, console):
        sink.setFormatter(formatter)
        LOG.addHandler(sink)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    database = args.output_dir / 'history.sqlite3'
    if (args.export_only or args.knowledge_only) and not database.exists():
        parser.error('No history.sqlite3 found in this output directory; collect first.')
    status = 0
    knowledge = None
    original_cwd = Path.cwd()
    try:
        with corpus_lock(args.output_dir):
            db = connect(database)
            try:
                if not args.export_only:
                    seed(db)
                    export(db, args.output_dir)
                    if args.knowledge_agent == 'required':
                        root = find_slai_root(args.slai_root)
                        # SLAI configuration contains paths relative to its repository root.
                        os.chdir(root)
                        knowledge = KnowledgeBridge.from_slai(root)
                        knowledge.hydrate(db)
                    if not args.knowledge_only:
                        collect(db, APIClient(args.contact, args.delay), args, knowledge=knowledge)
                    if args.knowledge_query:
                        assert knowledge is not None
                        results = knowledge.query(args.knowledge_query, args.knowledge_top_k)
                        print(json.dumps(results, ensure_ascii=False, indent=2))
            except KeyboardInterrupt:
                LOG.info('Interrupted. Committed articles are safe; rerun to rebuild the agent index.')
                status = 130
            except Exception as exc:
                LOG.error('%s. Collected articles remain in SQLite; rerun to retry.', exc, exc_info=True)
                status = 1
            finally:
                try:
                    export(db, args.output_dir)
                except (OSError, sqlite3.Error) as exc:
                    LOG.error('Export failed: %s. Keep SQLite and retry with --export-only.', exc)
                    status = 1
                finally:
                    try:
                        if knowledge is not None:
                            knowledge.close()
                    except Exception as exc:
                        LOG.error('Agent lifecycle cleanup failed: %s', exc)
                        status = 1
                    db.close()
    except (RuntimeError, OSError, sqlite3.Error) as exc:
        LOG.error('%s', exc)
        status = 1
    finally:
        os.chdir(original_cwd)
        for sink in (handler, console):
            LOG.removeHandler(sink)
            sink.close()
    return status


if __name__ == '__main__':
    sys.exit(main())
