#!/usr/bin/env python3
"""Collect sociology, politics, religion and psychology into four UTF-8 TXT files.
Python 3.10+. SLAI Knowledge Agent integration is enabled by default.

Run from SLAI root in its installed environment:
    python scrape_society_topics.py
Standalone collection (standard library only):
    python scrape_society_topics.py --knowledge-agent off
Larger collection (count limits are totals PER SUBJECT, not per invocation):
    python scrape_society_topics.py --max-articles 25000 --max-categories 10000 --depth 4
No count cap (category depth still bounds discovery):
    python scrape_society_topics.py --max-articles 0 --max-categories 0 --depth 4
Query the saved corpus without downloading:
    python scrape_society_topics.py --knowledge-only --knowledge-query "Social identity and group behavior"
Export without SLAI or network access:
    python scrape_society_topics.py --export-only
Offline tests:
    python scrape_society_topics.py --self-test

Outputs in ./society_topics: sociology.txt, politics.txt, religion.txt, psychology.txt,
plus history.sqlite3, collector.log and collector.lock. The SQLite basename is kept
for consistency with the earlier collectors; this is a separate subject database.
Rerun the same command to resume. Raising limits/depth expands existing work.
Exports are rebuilt from SQLite, so manual TXT edits are not preserved.

SOURCE SCOPE
English Wikipedia, through the official Action API. Explicit overview articles,
subject topics and category roots cover theories, methods, history and traditions.
Nested categories broaden coverage but can include tangential material. Category
membership is not relevance scoring. Missing seeds are skipped. There is no claim
of exhaustive, balanced or independently verified coverage. A topic shared by two
subjects can appear once in each output; Knowledge Agent indexes its text once.

POLITICS / RELIGION / PSYCHOLOGY
Texts are attributed external source material: beliefs, ideologies, theories,
historical claims and clinical topics are not endorsements, diagnoses, treatment
instructions or verified facts. The collector does not synthesize answers, profile
people, execute page instructions or train a model. English-language and source
selection biases remain. The Knowledge Agent performs document retrieval only.

PERSISTENCE AND AGENT INTEGRATION
AgentFactory.create('knowledge', ...) creates the agent once with shared memory.
Saved articles rebuild the agent's in-memory TF-IDF retrieval index on startup.
New text is committed to SQLite before indexing, preserving it on agent failure.
This does not persist a global SLAI KnowledgeMemory index or train LANTRA.
Use --slai-root for a checkout outside the current/script directory. All integrated
mode dependencies must already be installed in your SLAI environment.

ACCESS AND ATTRIBUTION
Serial requests, a minimum one-second spacing, timeout, maxlag and bounded retries.
No authentication/paywall/HTTP 403 bypass. Use --contact with your email/project URL
for an identifiable User-Agent. Source URLs, contributor history, CC BY-SA reference,
retrieval date, revision metadata and content hashes are retained in every export.
Extracts may omit tables/media/complex markup and may be cached independently of
revision metadata. Retain attribution and applicable rights when reusing content.

Ctrl+C exports committed records. Keep SQLite to recover after a forced kill.
Default limits: 5,000 articles and 2,000 categories per subject, category depth 3.
Large runs and Knowledge Agent index rebuilding can take substantial time and RAM.
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
SUBJECTS = {
    'sociology': ['Sociology', 'History of sociology', 'Social theory', 'Social research'],
    'politics': ['Politics', 'Political science', 'Political philosophy', 'Comparative politics'],
    'religion': ['Religion', 'Religious studies', 'History of religion', 'Comparative religion'],
    'psychology': ['Psychology', 'History of psychology', 'Psychological research', 'Cognitive science'],
}
EXTRA = {
    'sociology': [
        'Sociological theory', 'Social structure', 'Social institution', 'Socialization',
        'Social stratification', 'Social class', 'Social inequality', 'Social mobility',
        'Social change', 'Social movement', 'Social network', 'Social identity theory',
        'Symbolic interactionism', 'Structural functionalism', 'Conflict theories',
        'Critical theory', 'Feminist sociology', 'Sociology of race and ethnic relations',
        'Sociology of gender', 'Sociology of the family', 'Sociology of education',
        'Sociology of religion', 'Political sociology', 'Economic sociology',
        'Medical sociology', 'Environmental sociology', 'Urban sociology',
        'Rural sociology', 'Sociology of culture', 'Sociology of knowledge',
        'Deviance (sociology)', 'Criminology', 'Demography', 'Migration studies',
        'Globalization', 'Digital sociology', 'Computational sociology',
        'Ethnography', 'Survey methodology', 'Social network analysis',
        'Qualitative research', 'Quantitative research', 'Research ethics',
        'Postcolonialism', 'Decolonization of knowledge',
    ],
    'politics': [
        'History of political thought', 'Political theory', 'Political ideology',
        'State (polity)', 'Nation', 'Nation state', 'Sovereignty', 'Government',
        'Democracy', 'Republic', 'Monarchy', 'Authoritarianism', 'Totalitarianism',
        'Constitutionalism', 'Rule of law', 'Separation of powers', 'Federalism',
        'Parliamentary system', 'Presidential system', 'Electoral system',
        'Political party', 'Political participation', 'Civil society', 'Citizenship',
        'Public policy', 'Public administration', 'Political economy',
        'International relations', 'Geopolitics', 'Diplomacy', 'Human rights',
        'Liberalism', 'Conservatism', 'Socialism', 'Communism', 'Anarchism',
        'Nationalism', 'Fascism', 'Populism', 'Feminism', 'Green politics',
        'Colonialism', 'Imperialism', 'Decolonization', 'Indigenous politics',
        'Political corruption', 'Political violence', 'Peace and conflict studies',
        'Political communication', 'Propaganda', 'Political psychology',
        'Politics of Africa', 'Politics of Asia', 'Politics of Europe',
        'Politics of Latin America', 'Politics of the Caribbean',
        'Politics of North America', 'Politics of Oceania',
    ],
    'religion': [
        'Philosophy of religion', 'Sociology of religion', 'Psychology of religion',
        'Anthropology of religion', 'Cognitive science of religion',
        'Theology', 'Mythology', 'Ritual', 'Religious text', 'Sacred',
        'Religious experience', 'Religious ethics', 'Mysticism', 'Pilgrimage',
        'Monotheism', 'Polytheism', 'Pantheism', 'Animism', 'Shamanism',
        'Christianity', 'Catholic Church', 'Eastern Orthodox Church', 'Protestantism',
        'Islam', 'Sunni Islam', 'Shia Islam', 'Sufism', 'Judaism',
        'Hinduism', 'Buddhism', 'Jainism', 'Sikhism',
        'Taoism', 'Confucianism', 'Shinto', 'Chinese folk religion',
        'Zoroastrianism', 'Baháʼí Faith', 'Druze', 'Yazidism',
        'Traditional African religions', 'Yoruba religion', 'Vodun',
        'Haitian Vodou', 'Santería', 'Rastafari',
        'Native American religions', 'Australian Aboriginal religion and mythology',
        'Polynesian religion', 'Ancient Egyptian religion', 'Ancient Greek religion',
        'Norse religion', 'New religious movement', 'Religious syncretism',
        'Secularism', 'Secularization', 'Atheism', 'Agnosticism',
        'Freedom of religion', 'Religious pluralism', 'Interfaith dialogue',
    ],
    'psychology': [
        'Cognitive psychology', 'Developmental psychology', 'Social psychology',
        'Personality psychology', 'Biological psychology', 'Neuropsychology',
        'Experimental psychology', 'Comparative psychology', 'Cultural psychology',
        'Cross-cultural psychology', 'Educational psychology', 'School psychology',
        'Industrial and organizational psychology', 'Health psychology',
        'Community psychology', 'Environmental psychology', 'Forensic psychology',
        'Sport psychology', 'Clinical psychology', 'Counseling psychology',
        'Psychometrics', 'Psychophysics', 'Behaviorism', 'Cognitivism (psychology)',
        'Humanistic psychology', 'Psychoanalysis', 'Gestalt psychology',
        'Evolutionary psychology', 'Positive psychology',
        'Perception', 'Attention', 'Memory', 'Learning', 'Emotion', 'Motivation',
        'Intelligence', 'Language acquisition', 'Consciousness', 'Decision-making',
        'Cognitive bias', 'Attachment theory', 'Social cognition',
        'Psychopathology', 'Mental health', 'Psychotherapy',
        'Psychological testing', 'Psychological assessment',
        'Replication crisis', 'Open science', 'Preregistration (science)',
        'Meta-analysis', 'Research ethics', 'Indigenous psychology',
    ],
}
# Category roots are explicit; article titles are not assumed to have matching categories.
CATEGORY_ROOTS = {
    'sociology': ['Sociology', 'Branches of sociology', 'Sociological theories',
                  'Social research', 'Social institutions', 'Social inequality',
                  'Social movements', 'Sociologists'],
    'politics': ['Politics', 'Political science', 'Political philosophy',
                 'Political ideologies', 'Political systems', 'Public policy',
                 'International relations', 'Political history'],
    'religion': ['Religion', 'Religious studies', 'History of religion',
                 'Religious traditions', 'Philosophy of religion',
                 'Sociology of religion', 'Religious practices', 'Religious texts'],
    'psychology': ['Psychology', 'Branches of psychology', 'Psychological theories',
                   'Psychological concepts', 'Psychological research',
                   'History of psychology', 'Psychologists', 'Cognitive science'],
}
SKIP_CATEGORY = re.compile(r'\b(wikipedia|articles|pages|stubs|templates|births|deaths|'
                           r'living people|alumni)\b', re.I)
LOG = logging.getLogger('society_topics')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(text):
    return hashlib.sha256(' '.join(text.split()).encode('utf-8')).hexdigest()


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    existing_columns = {row[1] for row in db.execute('PRAGMA table_info(membership)')}
    if existing_columns and 'subject' not in existing_columns:
        db.close()
        raise RuntimeError('This is a geographic collector database. Choose a new --output-dir for subject collection.')
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS categories (
          subject TEXT, title TEXT, depth INTEGER, continuation TEXT,
          done INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(subject,title));
        CREATE TABLE IF NOT EXISTS queue (
          subject TEXT, title TEXT, origin TEXT, done INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(subject,title));
        CREATE TABLE IF NOT EXISTS articles (
          pageid INTEGER PRIMARY KEY, title TEXT, url TEXT, revid INTEGER,
          revision_time TEXT, retrieved TEXT, text TEXT, hash TEXT);
        CREATE TABLE IF NOT EXISTS membership (
          subject TEXT, pageid INTEGER, origin TEXT, hash TEXT,
          PRIMARY KEY(subject,pageid), UNIQUE(subject,hash));
        CREATE TABLE IF NOT EXISTS aliases (title TEXT PRIMARY KEY, pageid INTEGER);
        CREATE INDEX IF NOT EXISTS articles_hash ON articles(hash);
        CREATE INDEX IF NOT EXISTS membership_page ON membership(pageid);
        CREATE INDEX IF NOT EXISTS queue_pending ON queue(subject,done);
        CREATE INDEX IF NOT EXISTS categories_pending ON categories(subject,done,depth);
    ''')
    subjects = {row[0] for row in db.execute(
        'SELECT subject FROM membership UNION SELECT subject FROM queue UNION SELECT subject FROM categories')}
    if not subjects.issubset(SUBJECTS):
        db.close()
        raise RuntimeError('This database contains unrecognized subjects. Use a separate --output-dir.')
    return db


def seed(db):
    with db:
        for subject, overviews in SUBJECTS.items():
            for title in overviews:
                db.execute('INSERT OR IGNORE INTO queue(subject,title,origin) VALUES(?,?,?)',
                           (subject, title, 'Subject overview'))
            for title in EXTRA.get(subject, ()):
                db.execute('INSERT OR IGNORE INTO queue(subject,title,origin) VALUES(?,?,?)',
                           (subject, title, 'Subject topic seed'))
            for title in CATEGORY_ROOTS[subject]:
                db.execute('INSERT OR IGNORE INTO categories(subject,title,depth) VALUES(?,?,0)',
                           (subject, 'Category:' + title))


class APIClient:
    def __init__(self, contact='', delay=1.0, retries=6):
        self.user_agent = 'SocietyTopicsCollector/1.0' + (f' ({contact})' if contact else '')
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
                db.execute('INSERT OR IGNORE INTO queue(subject,title,origin) VALUES(?,?,?)',
                           (row['subject'], title, row['title']))
            elif member['ns'] == 14 and not SKIP_CATEGORY.search(title):
                # Keep boundary children so a later --depth increase works without resetting.
                db.execute('''INSERT INTO categories(subject,title,depth) VALUES(?,?,?)
                              ON CONFLICT(subject,title) DO UPDATE SET depth=min(depth,excluded.depth)''',
                           (row['subject'], title, row['depth'] + 1))
        db.execute('UPDATE categories SET continuation=?,done=? WHERE subject=? AND title=?',
                   (json.dumps(continuation) if continuation else None,
                    int(not continuation), row['subject'], row['title']))
    LOG.info('%s: explored %s%s', row['subject'], row['title'],
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
                db.execute('UPDATE queue SET done=1 WHERE subject=? AND title=?',
                           (row['subject'], row['title']))
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
                            (row['subject'], article['pageid'], row['origin'], article['hash']))
        saved = cursor.rowcount > 0
        db.execute('UPDATE queue SET done=1 WHERE subject=? AND title=?',
                   (row['subject'], row['title']))
    if knowledge is not None:
        # The corpus transaction is durable before the agent is invoked.
        knowledge.ingest(db, article)
    if saved:
        LOG.info('%s: saved %s', row['subject'], article['title'])
    return saved


def export(db, directory):
    for subject, places in SUBJECTS.items():
        path = directory / (subject + '.txt')
        temporary = path.with_suffix('.txt.tmp')
        count = db.execute('SELECT count(*) FROM membership WHERE subject=?', (subject,)).fetchone()[0]
        with temporary.open('w', encoding='utf-8', newline='\n') as output:
            output.write(f'{subject.upper()} — SUBJECT SOURCE CORPUS\n'
                         f'Subject seeds: {", ".join(places)}\n'
                         f'Exported (UTC): {utcnow()}\nArticles: {count}\n'
                         'Source: English Wikipedia and its contributors.\n'
                         'License: CC BY-SA 4.0; consult individual source notices.\n'
                         'https://creativecommons.org/licenses/by-sa/4.0/\n'
                         'Transformation: extracted plain text; complex markup may be omitted.\n'
                         'Category-based collection; not comprehensive or independently verified.\n\n')
            for a in db.execute('''SELECT a.*,m.origin FROM membership m JOIN articles a
                                   ON a.pageid=m.pageid WHERE m.subject=?
                                   ORDER BY a.title COLLATE NOCASE,a.pageid''', (subject,)):
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
    LOG.info('Exported four text files to %s', directory)


def collect(db, client, args, knowledge=None):
    counts = {r: db.execute('SELECT count(*) FROM membership WHERE subject=?', (r,)).fetchone()[0]
              for r in SUBJECTS}
    completed = {r: db.execute('SELECT count(*) FROM categories WHERE subject=? AND done=1',
                              (r,)).fetchone()[0] for r in SUBJECTS}
    new_articles = 0
    while True:
        progress = False
        for subject in SUBJECTS:  # Fair scheduling across subjects, including after failures.
            if args.max_articles and counts[subject] >= args.max_articles:
                continue
            row = db.execute('SELECT * FROM queue WHERE subject=? AND done=0 ORDER BY rowid LIMIT 1',
                             (subject,)).fetchone()
            if row:
                saved = fetch_article(db, client, row, knowledge=knowledge)
                counts[subject] += int(saved)
                new_articles += int(saved)
                progress = True
                if saved and new_articles % 100 == 0:
                    export(db, args.output_dir)
            elif not args.max_categories or completed[subject] < args.max_categories:
                row = db.execute('''SELECT * FROM categories WHERE subject=? AND done=0 AND depth<=?
                                    ORDER BY depth,rowid LIMIT 1''', (subject, args.depth)).fetchone()
                if row:
                    expand_category(db, client, row)
                    done = db.execute('SELECT done FROM categories WHERE subject=? AND title=?',
                                      (subject, row['title'])).fetchone()[0]
                    completed[subject] += int(done)
                    progress = True
        if not progress:
            break
    for subject in SUBJECTS:
        pending = db.execute('SELECT count(*) FROM queue WHERE subject=? AND done=0', (subject,)).fetchone()[0]
        LOG.info('%s: %d articles saved, %d categories completed, %d queued articles remain',
                 subject, counts[subject], completed[subject], pending)
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
            db.execute("INSERT INTO categories(subject,title,depth) VALUES('sociology','Category:Root',0)")
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
        original = (root / 'sociology.txt').read_text(encoding='utf-8')
        assert original.count('Historical text with Unicode:') == 1 and 'Curaçao' in original
        assert len(list(root.glob('*.txt'))) == len(SUBJECTS)
        db.close()
        db = connect(root / 'history.sqlite3')
        assert db.execute('SELECT count(*) FROM membership').fetchone()[0] == 1
        assert db.execute('SELECT done FROM queue').fetchone()[0] == 1
        export(db, root)
        assert (root / 'sociology.txt').read_text(encoding='utf-8').count('Historical text with Unicode:') == 1
        db.close()
    knowledge_self_test()
    print('PASS: pagination, depth frontier, duplicate filtering, Unicode, four exports, restart persistence.')


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
                'source': 'society_topics_collector',
                'retrieval_mode': 'tfidf',
                'directory_path': '',  # Articles are ingested separately, not four giant files.
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
            get_logger('society_topics').info('Creating Knowledge Agent through SLAI AgentFactory')
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
        doc_id = 'society:topics:sha256:' + article['hash']
        provenance = [dict(row) for row in db.execute('''
            SELECT DISTINCT a.pageid,a.title,a.url,a.revid,a.revision_time,a.retrieved,m.subject,m.origin
            FROM articles a JOIN membership m ON m.pageid=a.pageid
            WHERE a.hash=? ORDER BY a.pageid,m.subject
        ''', (article['hash'],))]
        metadata = {
            'source': 'english_wikipedia', 'collector': 'society_topics_collector',
            'type': 'subject_source', 'title': article['title'], 'url': article['url'],
            'page_id': article['pageid'], 'revision_id': article['revid'],
            'retrieved_utc': article['retrieved'], 'content_sha256': article['hash'],
            'subjects': sorted({row['subject'] for row in provenance}),
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
        # The same article may later enter another subject. Refresh its provenance
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
                 'subjects': doc.get('metadata', {}).get('subjects', []),
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
            self.doc_index.setdefault(doc_id, {'doc_id': doc_id, 'text': text, 'metadata': dict(metadata)})
        def retrieve(self, query, k=5):
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
        assert set(EXTRA) == set(SUBJECTS) == set(CATEGORY_ROOTS) and all(EXTRA.values())
        for subject, roots in SUBJECTS.items():
            for title in roots:
                assert db.execute('SELECT 1 FROM queue WHERE subject=? AND title=?', (subject, title)).fetchone()
        assert not db.execute("SELECT 1 FROM queue WHERE title='History of Sociology'").fetchone()
        assert db.execute('SELECT count(DISTINCT subject) FROM queue').fetchone()[0] == 4
        factory, memory = FakeFactory(), FakeMemory()
        bridge = KnowledgeBridge(factory, memory, owns_runtime=True)
        class FixtureAPI:
            def query(self, **params):
                return {'query': {'pages': [{'pageid': 42, 'title': 'Social identity theory',
                         'extract': 'Social identity and group behavior in social research.', 'fullurl': 'https://en.wikipedia.org/wiki/Social_identity_theory',
                         'revisions': [{'revid': 99, 'timestamp': '2026-01-01T00:00:00Z'}]}]}}
        row = {'subject': 'politics', 'title': 'Social identity theory', 'origin': 'test'}
        assert fetch_article(db, FixtureAPI(), row, knowledge=bridge)
        row2 = dict(row, subject='religion')
        assert fetch_article(db, FixtureAPI(), row2, knowledge=bridge)
        assert factory.created == 1 and factory.agent.calls == 1
        doc = next(iter(factory.agent.doc_index.values()))
        assert doc['metadata']['subjects'] == ['politics', 'religion']
        assert doc['metadata']['verified'] is False
        assert bridge.query('Social identity theory')[0]['url'].endswith('Social_identity_theory')
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
            def ingest(self, *args):
                raise RuntimeError('simulated indexing failure')
        try:
            fetch_article(db, FixtureAPI(), dict(row, subject='psychology'), knowledge=FailingBridge())
        except RuntimeError:
            pass
        else:
            raise AssertionError('Agent failure was hidden')
        assert db.execute("SELECT count(*) FROM membership WHERE subject='psychology'").fetchone()[0] == 1
        fresh = FakeFactory()
        KnowledgeBridge(fresh, FakeMemory()).hydrate(db)
        assert 'psychology' in next(iter(fresh.agent.doc_index.values()))['metadata']['subjects']
        # A silent no-op add_document must not count as a successful ingestion.
        bad = FakeFactory()
        bad.agent.add_document = lambda **kwargs: None
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
    parser.add_argument('--output-dir', type=Path, default=Path('society_topics'))
    parser.add_argument('--max-articles', type=int, default=5000, help='Per subject; 0 means unlimited')
    parser.add_argument('--max-categories', type=int, default=2000, help='Per subject; 0 means unlimited')
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
