#!/usr/bin/env python3
"""Animal kingdom collector: Wikipedia articles, Europe PMC research, Wikisource stories.
Python 3.10+. Standard library in standalone mode; installed SLAI in integrated mode.

Run from SLAI root: python scrape_animal_kingdom.py
Standalone:        python scrape_animal_kingdom.py --knowledge-agent off
Large run:         python scrape_animal_kingdom.py --max-articles 25000 --max-studies 10000 --depth 4
Replay/query:      python scrape_animal_kingdom.py --knowledge-only --knowledge-query "animal communication"
Offline export:    python scrape_animal_kingdom.py --export-only
Tests:             python scrape_animal_kingdom.py --self-test

Outputs: animal_kingdom/animal_articles.txt, animal_studies.txt, animal_stories.txt.
SQLite is authoritative. Rerun to resume; Ctrl+C exports committed results.
Knowledge Agent is required by default and created through AgentFactory. It indexes
text, not verified facts, and does not train LANTRA. Its in-memory index is rebuilt
from SQLite on every run. Fiction/literature is explicitly separated from research.
See ANIMAL_COLLECTOR_README.md for source scope, license handling, limits and recovery.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import sys
import time

from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote, urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener

LOG = logging.getLogger('animal_collector')
ENDPOINTS = {
    'wikipedia': 'https://en.wikipedia.org/w/api.php',
    'research': 'https://www.ebi.ac.uk/europepmc/webservices/rest/search',
    'stories': 'https://en.wikisource.org/w/api.php',
}
BUCKETS = ('animal_articles', 'animal_studies', 'animal_stories')
ARTICLE_SEEDS = [
    'Animal', 'Zoology', 'Biodiversity', 'Taxonomy (biology)', 'Animal anatomy',
    'Animal physiology', 'Evolution', 'Phylogenetics', 'Ecology', 'Ethology',
    'Vertebrate', 'Invertebrate', 'Mammal', 'Bird', 'Reptile', 'Amphibian', 'Fish',
    'Arthropod', 'Insect', 'Arachnid', 'Crustacean', 'Mollusca', 'Cephalopod',
    'Annelid', 'Echinoderm', 'Cnidaria', 'Sponge', 'Nematode', 'Tardigrade',
    'Primates', 'Cetacea', 'Carnivora', 'Rodent', 'Marsupial', 'Monotreme',
    'Bat', 'Elephant', 'Octopus', 'Bee', 'Ant', 'Butterfly', 'Shark', 'Coral',
    'Animal behavior', 'Animal cognition', 'Animal communication', 'Animal migration',
    'Social animal', 'Animal culture', 'Animal consciousness', 'Animal emotion',
    'Tool use by non-humans', 'Play (activity)', 'Animal navigation', 'Animal echolocation',
    'Predation', 'Herbivory', 'Foraging', 'Symbiosis', 'Parasitism', 'Mutualism (biology)',
    'Reproductive behaviour', 'Courtship display', 'Parental investment', 'Hibernation',
    'Camouflage', 'Mimicry', 'Bioluminescence', 'Eusociality', 'Swarm behaviour',
    'Conservation biology', 'Wildlife conservation', 'Endangered species',
    'Habitat fragmentation', 'Rewilding', 'Invasive species', 'Human–wildlife conflict',
    'Domestication', 'Animal welfare', 'Animal ethics', 'Companion animal',
    'Human–animal bond', 'Veterinary medicine', 'Animal testing', 'Animal model',
    'Comparative psychology', 'Behavioral ecology', 'Marine biology',
    'Ornithology', 'Mammalogy', 'Herpetology', 'Ichthyology', 'Entomology',
    'Animal tracking', 'Camera trap', 'Bioacoustics', 'Environmental DNA',
    'Animals in literature', 'Animals in folklore', 'Fable', 'Animal tale',
]
CATEGORY_SEEDS = [
    'Animals', 'Zoology', 'Animal anatomy', 'Animal physiology', 'Animal behavior',
    'Animal cognition', 'Animal communication', 'Mammals', 'Birds', 'Reptiles',
    'Amphibians', 'Fish', 'Invertebrates', 'Insects', 'Marine animals',
    'Wildlife conservation', 'Animal welfare', 'Human–animal interaction',
]
STORY_ROOTS = {
    'The Tale of Peter Rabbit (1901)': {'author': 'Beatrix Potter', 'year': 1901},
    'Black Beauty': {'author': 'Anna Sewell', 'year': 1877},
    'The Jungle Book (Century edition)': {'author': 'Rudyard Kipling', 'year': 1894},
    'The Second Jungle Book': {'author': 'Rudyard Kipling', 'year': 1895},
}
RESEARCH_TERMS = [
    '"animal behaviour" OR "animal behavior"',
    '"animal cognition" OR "animal communication"',
    '"wildlife conservation" OR "conservation biology"',
    '"animal migration" OR "animal navigation"',
    '"social behaviour" AND (birds OR primates OR mammals)',
    '"marine mammals" OR "cetacean behaviour"',
    '"insect behaviour" OR "insect behavior" OR "pollinator ecology"',
    '"cephalopod cognition" OR "octopus behaviour"',
    '"amphibian conservation" OR "reptile ecology"',
    '"fish behaviour" OR "fish behavior"',
    '"animal welfare" OR "human wildlife conflict"',
    '"comparative physiology" AND animals',
]
SKIP_CAT = re.compile(r'\b(wikipedia|articles|templates|stubs|births|deaths|living people|fictional)\b', re.I)


class Skip(Exception):
    pass


class Deferred(Exception):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def checksum(text):
    return hashlib.sha256(' '.join(text.split()).encode('utf-8')).hexdigest()


def wiki_url(host, title):
    return f'https://{host}/wiki/' + quote(title.replace(' ', '_'), safe='/')


class TextParser(HTMLParser):
    """Extract API-returned prose without executing markup or following HTML links."""
    VOID = {'br', 'hr', 'img', 'meta', 'link', 'input', 'wbr', 'source'}
    BLOCK = {'p', 'div', 'section', 'br', 'li', 'h1', 'h2', 'h3', 'h4', 'blockquote'}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.stack = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        excluded = tag in {'script', 'style', 'table', 'nav', 'header', 'footer', 'noscript', 'sup'}
        classes = attrs.get('class', '').split()
        excluded |= any(c in {'ws-noexport', 'noprint', 'navbox', 'mw-editsection',
                              'sister-wikipedia', 'licenseContainer'} for c in classes)
        excluded |= attrs.get('id', '') in {'header', 'license', 'ws-header', 'toc'}
        blocked = excluded or bool(self.stack and self.stack[-1][1])
        if not blocked and tag in self.BLOCK:
            self.parts.append('\n')
        if tag not in self.VOID:
            self.stack.append((tag, blocked))
    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)
    def handle_endtag(self, tag):
        for index in range(len(self.stack)-1, -1, -1):
            if self.stack[index][0] == tag:
                blocked = self.stack[index][1]
                del self.stack[index:]
                if not blocked and tag in self.BLOCK:
                    self.parts.append('\n')
                break
    def handle_data(self, data):
        if not self.stack or not self.stack[-1][1]:
            self.parts.append(data)
    def text(self):
        return '\n\n'.join(' '.join(line.split()) for line in ''.join(self.parts).splitlines() if line.strip())


def plain(text):
    p = TextParser(); p.feed(text or ''); p.close()
    return p.text()


class RestrictedRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(newurl)
        allowed = {(urlsplit(u).hostname, urlsplit(u).path) for u in ENDPOINTS.values()}
        if target.scheme != 'https' or target.username or target.password or target.port not in (None, 443) or (target.hostname, target.path) not in allowed:
            raise Skip('Redirect outside the supported HTTPS APIs')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class APIClient:
    def __init__(self, delay=1, contact='', retries=4, timeout=40):
        self.delay, self.retries, self.timeout = delay, retries, timeout
        self.last = 0
        self.ua = 'AnimalKingdomCollector/1.0' + (f' ({contact})' if contact else '')
        self.opener = build_opener(RestrictedRedirect())
    def request(self, source, **params):
        url = ENDPOINTS[source] + '?' + urlencode(params)
        failure = None
        for attempt in range(self.retries):
            time.sleep(max(0, self.delay - (time.monotonic()-self.last)))
            pause = min(60, 2 ** (attempt + 1))
            try:
                self.last = time.monotonic()
                req = Request(url, headers={'User-Agent': self.ua, 'Accept': 'application/json', 'Accept-Encoding': 'identity'})
                with self.opener.open(req, timeout=self.timeout) as response:
                    if response.headers.get_content_type() not in ('application/json', 'text/json'):
                        raise Deferred('API returned non-JSON content')
                    raw = response.read(12*1024*1024 + 1)
                    if len(raw) > 12*1024*1024:
                        raise Skip('API response exceeds 12 MiB limit')
                    data = json.loads(raw)
                if not isinstance(data, dict):
                    raise Deferred('Unexpected JSON structure')
                if 'error' in data:
                    error = data['error']
                    if isinstance(error, dict) and error.get('code') in ('missingtitle','invalidtitle','nosuchpageid'):
                        raise Skip(str(error))
                    raise Deferred(str(error))
                return data
            except HTTPError as exc:
                if exc.code in (401, 403):
                    raise Deferred(f'HTTP {exc.code}; source access denied, no bypass attempted') from exc
                if exc.code not in (429, 500, 502, 503, 504):
                    raise Skip(f'HTTP {exc.code}; access is not bypassed') from exc
                failure = exc
                header = exc.headers.get('Retry-After', '')
                try:
                    pause = max(pause, float(header))
                except ValueError:
                    try:
                        pause = max(pause, parsedate_to_datetime(header).timestamp()-time.time())
                    except (ValueError, TypeError, OverflowError):
                        pass
                if not math.isfinite(pause) or pause > 120:
                    raise Deferred('Long Retry-After; defer this source until a later run') from exc
            except (URLError, OSError, ValueError, Deferred) as exc:
                failure = exc
            if attempt+1 < self.retries:
                LOG.warning('%s request failed: %s; retry in %.1fs', source, failure, pause)
                time.sleep(pause)
        raise Deferred(f'{source} failed after {self.retries} attempts: {failure}')
    def wiki(self, source='wikipedia', **params):
        return self.request(source, format='json', formatversion=2, maxlag=5, **params)


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if tables and 'collector_meta' not in tables:
        db.close(); raise RuntimeError('Existing database is not an animal collector database; choose a new output directory')
    if 'collector_meta' in tables:
        ident = db.execute("SELECT value FROM collector_meta WHERE key='schema'").fetchone()
        if ident is None or ident[0] != 'animal-collector-v1':
            db.close(); raise RuntimeError('Incompatible collector schema')
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS collector_meta(key TEXT PRIMARY KEY,value TEXT);
    INSERT OR IGNORE INTO collector_meta VALUES('schema','animal-collector-v1');
    CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY,source TEXT,kind TEXT,target TEXT,
       depth INTEGER,origin TEXT,payload TEXT DEFAULT '{}',status TEXT DEFAULT 'pending',error TEXT DEFAULT '',
       UNIQUE(source,kind,target));
    CREATE INDEX IF NOT EXISTS tasks_work ON tasks(source,status,kind,depth);
    CREATE TABLE IF NOT EXISTS dirty(key TEXT PRIMARY KEY);
    CREATE TABLE IF NOT EXISTS records(key TEXT PRIMARY KEY,bucket TEXT,title TEXT,body TEXT,hash TEXT,
       metadata TEXT,retrieved TEXT);
    CREATE INDEX IF NOT EXISTS record_hash ON records(bucket,hash);
    ''')
    return db


def enqueue(db, source, kind, target, depth=0, origin='seed'):
    db.execute('''INSERT INTO tasks(source,kind,target,depth,origin) VALUES(?,?,?,?,?)
                  ON CONFLICT(source,kind,target) DO UPDATE SET depth=min(depth,excluded.depth)''',
               (source,kind,target,depth,origin))


def seed(db, args):
    with db:
        for title in ARTICLE_SEEDS:
            enqueue(db,'wikipedia','article',title)
        for title in CATEGORY_SEEDS:
            enqueue(db,'wikipedia','category','Category:'+title)
        for title in STORY_ROOTS:
            enqueue(db,'stories','story',title,origin=title)
        for term in RESEARCH_TERMS:
            query = 'TITLE_ABS:(' + term + ') AND HAS_ABSTRACT:Y AND OPEN_ACCESS:Y'
            enqueue(db,'research','research',query)
        db.execute("UPDATE tasks SET status='pending' WHERE status='error'")
        if args.retry_skipped:
            db.execute("UPDATE tasks SET status='pending' WHERE status='skipped'")
        if args.refresh_research:
            db.execute("UPDATE tasks SET status='pending',payload='{}' WHERE source='research'")


def save(db, key, bucket, title, body, meta):
    """Deduplicate canonical identifiers and exact normalized text within a bucket."""
    body = body.strip()
    if not body:
        return
    old = db.execute('SELECT * FROM records WHERE key=? OR (bucket=? AND hash=?) ORDER BY key LIMIT 1',
                     (key,bucket,checksum(body))).fetchone()
    if old:
        existing = json.loads(old['metadata'])
        existing['origins'] = sorted(set(existing.get('origins', []) + meta.get('origins', [])))
        alternatives = existing.setdefault('alternate_source_urls', [])
        if meta.get('url') != existing.get('url') and meta.get('url') not in alternatives:
            alternatives.append(meta['url'])
        db.execute('UPDATE records SET metadata=? WHERE key=?',(json.dumps(existing,ensure_ascii=False),old['key']))
        db.execute('INSERT OR IGNORE INTO dirty VALUES(?)',(old['key'],))
        return
    meta.setdefault('verified', False)
    meta.setdefault('trust', 'external_source_unverified')
    db.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?)',
               (key,bucket,title,body,checksum(body),json.dumps(meta,ensure_ascii=False),now()))
    db.execute('INSERT OR IGNORE INTO dirty VALUES(?)',(key,))
    LOG.info('%s | saved %s [%s]', bucket,title,meta['content_type'])


def done(db,row,payload=None):
    db.execute('UPDATE tasks SET status=?,payload=?,error=? WHERE id=?',
               ('pending' if payload else 'done',json.dumps(payload or {}),'',row['id']))


def wikipedia_task(db,client,row,args):
    if row['kind']=='category':
        params=dict(action='query',list='categorymembers',cmtitle=row['target'],
                    cmtype='page|subcat',cmnamespace='0|14',cmlimit=500)
        params.update(json.loads(row['payload']))
        data=client.wiki(**params)
        members=data.get('query',{}).get('categorymembers')
        if not isinstance(members,list):
            raise Deferred('Missing categorymembers')
        with db:
            for member in members:
                if member['ns']==0:
                    enqueue(db,'wikipedia','article',member['title'],row['depth'],row['target'])
                elif member['ns']==14 and not SKIP_CAT.search(member['title']):
                    enqueue(db,'wikipedia','category',member['title'],row['depth']+1,row['target'])
            done(db,row,data.get('continue'))
        return
    data=client.wiki(action='query',titles=row['target'],redirects=1,
                    prop='extracts|info|revisions|pageprops',inprop='url',explaintext=1,
                    exsectionformat='plain',exlimit=1,rvprop='ids|timestamp',rvlimit=1,ppprop='disambiguation')
    pages=data.get('query',{}).get('pages')
    if not pages:
        raise Deferred('Missing article response')
    p=pages[0]
    if p.get('missing') or p.get('invalid') or 'disambiguation' in p.get('pageprops',{}):
        raise Skip('Missing or disambiguation article')
    body=p.get('extract','').strip()
    if not body:
        raise Deferred('Empty article extract')
    with db:
        save(db,'wikipedia:'+str(p['pageid']),'animal_articles',p['title'],body,{
            'source':'English Wikipedia','url':p.get('fullurl',wiki_url('en.wikipedia.org',p['title'])),
            'contributors_url':f'https://en.wikipedia.org/w/index.php?curid={p["pageid"]}&action=history',
            'rights':'CC BY-SA 4.0; source-specific notices apply',
            'rights_url':'https://creativecommons.org/licenses/by-sa/4.0/',
            'content_type':'encyclopedia_article','revision_metadata':p.get('revisions',[]),
            'origins':[row['origin']], 'transform':'plain-text extract; tables/media may be omitted'})
        done(db,row)


def story_task(db,client,row,args):
    root=row['origin']
    if root not in STORY_ROOTS or not (row['target']==root or row['target'].startswith(root+'/')):
        raise Skip('Story outside the curated work/chapter scope')
    data=client.wiki('stories',action='parse',page=row['target'],prop='text|links|revid',redirects=1)
    p=data.get('parse')
    if not isinstance(p,dict):
        raise Deferred('Missing Wikisource parse response')
    title=p['title']
    if not (title==root or title.startswith(root+'/')):
        raise Skip('Story redirect outside curated edition')
    markup=p.get('text','')
    if isinstance(markup,dict):
        markup=markup.get('*','')
    body=plain(markup)
    if not body:
        raise Deferred('Empty story text')
    with db:
        for link in p.get('links',[]):
            target=link.get('title',link.get('*',''))
            if link.get('ns')==0 and target.startswith(root+'/'):
                enqueue(db,'stories','story',target,row['depth']+1,root)
        save(db,'wikisource:'+str(p.get('pageid',title)),'animal_stories',title,body,{
            'source':'English Wikisource','url':wiki_url('en.wikisource.org',title),
            'contributors_url':'https://en.wikisource.org/w/index.php?'+urlencode({'title':title,'action':'history'}),
            'author':STORY_ROOTS[root]['author'],'original_publication_year':STORY_ROOTS[root]['year'],
            'work':root,'content_type':'literary_text','fiction':True,
            'rights':'Curated historic public-domain work; Wikisource transcription terms and local jurisdiction apply',
            'rights_url':'https://en.wikisource.org/wiki/Wikisource:Copyright_policy',
            'revision_id':p.get('revid'),'origins':[root],
            'transform':'plain-text transcription; may include contents/preface and omit tables/illustrations'})
        done(db,row)


def abstract_mode(record):
    license_text=str(record.get('license','')).strip()
    normalized=re.sub(r'[\s_]+','-',license_text.lower()).rstrip('/')
    supported=bool(re.fullmatch(r'cc-(?:by(?:-nc)?(?:-sa)?|0)(?:-[\d.]+)?',normalized))
    if supported:
        return 'research_abstract',license_text
    return 'research_abstract_excerpt','Full reuse rights not established; consult article and publisher'


def research_task(db,client,row,args):
    count=db.execute("SELECT count(*) FROM records WHERE bucket='animal_studies'").fetchone()[0]
    page_size=min(50,args.max_studies-count) if args.max_studies else 50
    if page_size<=0:
        return
    cursor=json.loads(row['payload']).get('cursorMark','*')
    data=client.request('research',query=row['target'],format='json',resultType='core',
                        pageSize=page_size,cursorMark=cursor)
    items=data.get('resultList',{}).get('result')
    if not isinstance(items,list):
        raise Deferred('Missing research resultList')
    next_cursor=data.get('nextCursorMark')
    with db:
        for record in items:
            title=plain(record.get('title',''))
            abstract=plain(record.get('abstractText',''))
            if not title or not abstract or not record.get('id') or not record.get('source'):
                continue
            doi=str(record.get('doi','')).strip().lower()
            key='doi:'+doi if doi else 'epmc:'+record['source']+':'+str(record['id'])
            content_type,rights=abstract_mode(record)
            body=abstract if content_type=='research_abstract' else ' '.join(abstract.split()[:80])
            url='https://europepmc.org/article/'+quote(record['source'],safe='')+'/'+quote(str(record['id']),safe='')
            save(db,key,'animal_studies',title,body,{
                'source':'Europe PMC','url':url,'doi':doi,'authors':record.get('authorString',''),
                'publication_year':record.get('pubYear'),
                'journal':record.get('journalInfo',{}).get('journal',{}).get('title'),
                'publication_types':record.get('pubTypeList',{}).get('pubType',[]),
                'record_source':record['source'],'record_id':record['id'],
                'open_access_flag':record.get('isOpenAccess'),
                'retracted_flag':record.get('isRetracted','not supplied'),
                'content_type':content_type,'rights':rights,'reported_license':record.get('license'),
                'rights_url':'https://europepmc.org/Copyright','origins':[row['target']],
                'full_paper_downloaded':False,'peer_review_verified':False})
        if items and next_cursor and next_cursor!=cursor:
            done(db,row,{'cursorMark':next_cursor})
        elif items and ((len(items)>=page_size and not next_cursor) or next_cursor==cursor):
            raise Deferred('Full research page has no nextCursorMark; refusing to mark query complete')
        else:
            done(db,row)
    LOG.info('Research query: %s | matches=%s | page records=%d',row['target'],data.get('hitCount','unknown'),len(items))


def export(db,directory):
    for bucket in BUCKETS:
        path=directory/(bucket+'.txt'); temporary=path.with_suffix('.txt.tmp')
        count=db.execute('SELECT count(*) FROM records WHERE bucket=?',(bucket,)).fetchone()[0]
        with temporary.open('w',encoding='utf-8',newline='\n') as f:
            f.write(f'{bucket.upper()}\nExported UTC: {now()}\nRecords: {count}\n'
                    'Attributed source collection; not independently verified.\n'
                    'Literary text, encyclopedia articles and research abstracts have different evidentiary roles.\n'
                    'Retain each record\'s source and rights information.\n\n')
            for row in db.execute('SELECT * FROM records WHERE bucket=? ORDER BY title COLLATE NOCASE,key',(bucket,)):
                f.write('='*80+'\n'+row['title']+'\n'+'='*80+'\n')
                f.write(f'ID: {row["key"]}\nRetrieved UTC: {row["retrieved"]}\nSHA-256: {row["hash"]}\n')
                for key,value in json.loads(row['metadata']).items():
                    f.write(key+': '+(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False))+'\n')
                f.write('\n'+row['body']+'\n\n')
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary,path)
    LOG.info('Exported three TXT files to %s',directory)


class KnowledgeBridge:
    """Use the same factory/add_document/retrieve contract as earlier collectors."""
    def __init__(self,factory,memory,owns=False):
        self.factory,self.memory,self.owns=factory,memory,owns
        self.closed=False
        try:
            self.agent=factory.create('knowledge',shared_memory=memory,config={
                'source':'animal_kingdom_collector','retrieval_mode':'tfidf','directory_path':'',
                'bias_detection_enabled':False,'use_ontology_expansion':False})
            if not isinstance(getattr(self.agent,'doc_index',None),dict) or not callable(getattr(self.agent,'add_document',None)) or not callable(getattr(self.agent,'retrieve',None)):
                raise RuntimeError('Knowledge Agent lacks the required in-process ingestion/retrieval contract')
        except BaseException:
            if owns:
                try: factory.shutdown()
                finally: memory.close()
            raise
    def sync(self,db,full=False):
        count=0
        sql='SELECT * FROM records ORDER BY key' if full else 'SELECT r.* FROM records r JOIN dirty d ON d.key=r.key ORDER BY r.key'
        for row in db.execute(sql):
            # Include the content class in the ID: fiction must never merge with research.
            doc_id='animal:'+row['bucket']+':'+row['hash']
            meta=json.loads(row['metadata'])
            meta.update(title=row['title'],bucket=row['bucket'],record_id=row['key'],retrieved_utc=row['retrieved'])
            indexed_text='[Content type: '+meta['content_type']+'; source collection: '+row['bucket']+']\n\n'+row['body']
            existing=self.agent.doc_index.get(doc_id)
            if existing is None:
                self.agent.add_document(text=indexed_text,doc_id=doc_id,metadata=meta)
                existing=self.agent.doc_index.get(doc_id)
            if not isinstance(existing,dict) or existing.get('text','')!=indexed_text:
                raise RuntimeError('Knowledge Agent did not acknowledge expected document '+doc_id)
            existing.setdefault('metadata',{}).update(meta)
            with db: db.execute('DELETE FROM dirty WHERE key=?',(row['key'],))
            count+=1
        LOG.info('Knowledge index synchronized: %d records',count)
    def query(self,text,k):
        return [{'score':float(score),'title':doc['metadata'].get('title'),
                 'url':doc['metadata'].get('url'),'content_type':doc['metadata'].get('content_type'),
                 'bucket':doc['metadata'].get('bucket'),'excerpt':doc['text'][:1000]}
                for score,doc in self.agent.retrieve(query=text,k=k)]
    def close(self):
        if self.closed: return
        self.closed=True
        if self.owns:
            try: self.factory.shutdown()
            finally: self.memory.close()


def slai_root(explicit):
    candidates=[explicit.expanduser().resolve()] if explicit else []
    if not explicit:
        for base in (Path.cwd(),Path(__file__).resolve().parent):
            candidates.extend([base,*base.parents])
    for root in candidates:
        if (root/'src/agents/agent_factory.py').is_file(): return root
    raise RuntimeError('SLAI root not found. Set --slai-root or use --knowledge-agent off')


def create_bridge(root):
    if str(root) not in sys.path: sys.path.insert(0,str(root))
    from src.agents.agent_factory import AgentFactory
    from src.agents.collaborative.shared_memory import SharedMemory
    from logs.logger import configure_logging
    configure_logging()
    memory=SharedMemory()
    try: factory=AgentFactory()
    except BaseException:
        memory.close(); raise
    return KnowledgeBridge(factory,memory,owns=True)


@contextmanager
def lock(directory):
    with (directory/'collector.lock').open('a+b') as f:
        f.seek(0,os.SEEK_END)
        if not f.tell(): f.write(b'0'); f.flush()
        f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc: raise RuntimeError('Another collector owns this output directory') from exc
        try: yield
        finally:
            f.seek(0)
            if os.name=='nt': msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(f,fcntl.LOCK_UN)


def run(db,client,args,bridge=None):
    disabled=set(); steps=0
    while True:
        progressed=False
        for source in args.sources:
            if source in disabled: continue
            if args.max_steps and steps>=args.max_steps: return
            bucket={'wikipedia':'animal_articles','research':'animal_studies','stories':'animal_stories'}[source]
            cap={'wikipedia':args.max_articles,'research':args.max_studies,'stories':args.max_story_pages}[source]
            count=db.execute('SELECT count(*) FROM records WHERE bucket=?',(bucket,)).fetchone()[0]
            if cap and count>=cap: continue
            if source=='wikipedia':
                row=db.execute("SELECT * FROM tasks WHERE source=? AND kind='article' AND status='pending' ORDER BY id LIMIT 1",(source,)).fetchone()
                if row is None:
                    cats=db.execute("SELECT count(*) FROM tasks WHERE kind='category' AND status='done'").fetchone()[0]
                    if args.max_categories and cats>=args.max_categories: continue
                    row=db.execute("SELECT * FROM tasks WHERE kind='category' AND status='pending' AND depth<=? ORDER BY depth,id LIMIT 1",(args.depth,)).fetchone()
            else:
                row=db.execute("SELECT * FROM tasks WHERE source=? AND status='pending' AND depth<=? ORDER BY depth,id LIMIT 1",
                               (source,args.story_depth if source=='stories' else 0)).fetchone()
            if row is None: continue
            progressed=True; steps+=1
            try:
                {'wikipedia':wikipedia_task,'research':research_task,'stories':story_task}[source](db,client,row,args)
            except Skip as exc:
                with db: db.execute("UPDATE tasks SET status='skipped',error=? WHERE id=?",(str(exc),row['id']))
                LOG.warning('%s | skipped %s: %s',source,row['target'],exc)
            except (Deferred,OSError,ValueError) as exc:
                with db: db.execute("UPDATE tasks SET status='error',error=? WHERE id=?",(str(exc),row['id']))
                LOG.warning('%s deferred for this run: %s',source,exc); disabled.add(source)
            if steps%25==0:
                export(db,args.output_dir)
                if bridge: bridge.sync(db)  # After source transactions, never inside one.
        if not progressed: break
    for bucket in BUCKETS:
        LOG.info('%s total=%d',bucket,db.execute('SELECT count(*) FROM records WHERE bucket=?',(bucket,)).fetchone()[0])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir',type=Path,default=Path('animal_kingdom'))
    parser.add_argument('--max-articles',type=int,default=5000)
    parser.add_argument('--max-categories',type=int,default=2000)
    parser.add_argument('--max-studies',type=int,default=2000)
    parser.add_argument('--max-story-pages',type=int,default=300)
    parser.add_argument('--depth',type=int,default=3)
    parser.add_argument('--story-depth',type=int,default=3)
    parser.add_argument('--max-steps',type=int,default=0,help='Task cap this run; 0 means unlimited')
    parser.add_argument('--sources',default='wikipedia,research,stories')
    parser.add_argument('--delay',type=float,default=1)
    parser.add_argument('--contact',default='')
    parser.add_argument('--slai-root',type=Path)
    parser.add_argument('--knowledge-agent',choices=('required','off'),default='required')
    parser.add_argument('--knowledge-only',action='store_true')
    parser.add_argument('--knowledge-query')
    parser.add_argument('--knowledge-top-k',type=int,default=5)
    parser.add_argument('--export-only',action='store_true')
    parser.add_argument('--refresh-research',action='store_true',help='Restart research cursors; existing records stay deduplicated')
    parser.add_argument('--retry-skipped',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args(argv)
    if args.self_test: self_test(); return 0
    for key in ('max_articles','max_categories','max_studies','max_story_pages','depth','story_depth','max_steps'):
        if getattr(args,key)<0: parser.error(key+' must be nonnegative')
    if not math.isfinite(args.delay) or args.delay<1: parser.error('--delay must be finite and >=1')
    if args.knowledge_top_k<1: parser.error('--knowledge-top-k must be positive')
    if any(ord(c)<32 or ord(c)>126 for c in args.contact): parser.error('--contact must be printable ASCII')
    args.sources=list(dict.fromkeys(args.sources.split(',')))
    if not args.sources or not set(args.sources)<=set(ENDPOINTS): parser.error('Sources: wikipedia,research,stories')
    if args.export_only and (args.knowledge_only or args.knowledge_query): parser.error('--export-only cannot query/index')
    if args.knowledge_agent=='off' and (args.knowledge_only or args.knowledge_query): parser.error('Query/index needs Knowledge Agent')
    args.output_dir=args.output_dir.resolve(); args.output_dir.mkdir(parents=True,exist_ok=True)
    database=args.output_dir/'history.sqlite3'
    if (args.export_only or args.knowledge_only) and not database.exists(): parser.error('No existing corpus database')
    handlers=[logging.StreamHandler(),logging.FileHandler(args.output_dir/'collector.log',encoding='utf-8')]
    for h in handlers:
        h.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')); LOG.addHandler(h)
    LOG.setLevel(logging.INFO); LOG.propagate=False
    original=Path.cwd(); status=0; bridge=None
    try:
        with lock(args.output_dir):
            db=connect(database)
            try:
                if not args.export_only:
                    seed(db,args)
                    if args.knowledge_agent=='required':
                        root=slai_root(args.slai_root); os.chdir(root)
                        bridge=create_bridge(root); bridge.sync(db,full=True)
                    if not args.knowledge_only: run(db,APIClient(args.delay,args.contact),args,bridge)
                    if bridge: bridge.sync(db)
                    if args.knowledge_query: print(json.dumps(bridge.query(args.knowledge_query,args.knowledge_top_k),ensure_ascii=False,indent=2))
                errors=db.execute("SELECT count(*) FROM tasks WHERE status='error'").fetchone()[0]
                if errors: LOG.warning('%d deferred tasks; rerun to retry',errors); status=2
            except KeyboardInterrupt:
                LOG.info('Interrupted; exporting committed records. Rerun to resume.'); status=130
            except Exception:
                LOG.exception('Collector failed; committed source records are preserved'); status=1
            finally:
                try: export(db,args.output_dir)
                finally:
                    try:
                        if bridge: bridge.close()
                    finally: db.close()
    except (OSError,RuntimeError,sqlite3.Error):
        LOG.exception('Local initialization/export/cleanup error'); status=1
    finally:
        os.chdir(original)
        for h in handlers: LOG.removeHandler(h); h.close()
    return status


def self_test():
    import tempfile
    from types import SimpleNamespace
    class Memory:
        closed=False
        def close(self): self.closed=True
    class Agent:
        def __init__(self): self.doc_index={}; self.hashes=set(); self.calls=0
        def add_document(self,text,doc_id,metadata):
            self.calls+=1
            h=checksum(text)
            if h in self.hashes: return
            self.hashes.add(h)
            self.doc_index[doc_id]={'doc_id':doc_id,'text':text,'metadata':dict(metadata)}
        def retrieve(self,query,k): return [(1.0,doc) for doc in list(self.doc_index.values())[:k]]
    class Factory:
        def __init__(self): self.agent=Agent(); self.closed=False; self.calls=0
        def create(self,kind,shared_memory,config):
            assert kind=='knowledge' and config['retrieval_mode']=='tfidf'
            self.calls+=1; return self.agent
        def shutdown(self): self.closed=True
    class Fixture:
        def __init__(self): self.calls=[]
        def wiki(self,source='wikipedia',**params):
            self.calls.append((source,params))
            if params.get('list')=='categorymembers':
                if 'cmcontinue' not in params:
                    return {'query':{'categorymembers':[{'ns':0,'title':'Animal'},{'ns':14,'title':'Category:Mammals'}]},
                            'continue':{'continue':'-||','cmcontinue':'next'}}
                return {'query':{'categorymembers':[]}}
            if source=='stories':
                return {'parse':{'title':'Black Beauty','pageid':51,'revid':4,
                    'text':'<div><table><tr><td>Navigation</td></tr></table><p>A horse lived in a meadow.</p><script>bad()</script></div>',
                    'links':[{'ns':0,'title':'Black Beauty/1'},{'ns':0,'title':'Other book'},{'ns':2,'title':'User:Example'}]}}
            return {'query':{'pages':[{'title':'Animal','pageid':2,'extract':'Animals form a diverse biological kingdom.',
                                      'fullurl':'https://en.wikipedia.org/wiki/Animal','revisions':[{'revid':8}]}]}}
        def request(self,source,**params):
            self.calls.append((source,params))
            cursor=params.get('cursorMark')
            if cursor=='next': return {'resultList':{'result':[]},'nextCursorMark':'next','hitCount':1}
            return {'resultList':{'result':[{'id':'123','source':'MED','title':'Animal cognition',
                'doi':'10.1234/ABC','abstractText':'<p>'+('Animal learning is studied carefully. '*30)+'</p>',
                'authorString':'Example Author','isOpenAccess':'Y','license':'CC BY','pubYear':'2020'}]},
                'nextCursorMark':'next','hitCount':1}
    assert plain('<p>A <i>small</i> animal.</p><script>bad()</script>')=='A small animal.'
    assert abstract_mode({'license':'CC BY-NC-SA'})[0]=='research_abstract'
    assert abstract_mode({'isOpenAccess':'Y'})[0]=='research_abstract_excerpt'
    assert abstract_mode({'license':'CC BY-ND'})[0]=='research_abstract_excerpt'
    args=SimpleNamespace(retry_skipped=False,refresh_research=False,max_studies=1)
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp); db=connect(folder/'history.sqlite3'); seed(db,args)
        fixture=Fixture()
        row=db.execute("SELECT * FROM tasks WHERE kind='category' ORDER BY id LIMIT 1").fetchone()
        wikipedia_task(db,fixture,row,args)
        row=db.execute('SELECT * FROM tasks WHERE id=?',(row['id'],)).fetchone()
        assert row['status']=='pending' and 'next' in row['payload']
        wikipedia_task(db,fixture,row,args)
        assert fixture.calls[-1][1]['cmcontinue']=='next'
        row=db.execute("SELECT * FROM tasks WHERE kind='article' ORDER BY id LIMIT 1").fetchone()
        wikipedia_task(db,fixture,row,args); wikipedia_task(db,fixture,row,args)
        assert db.execute("SELECT count(*) FROM records WHERE bucket='animal_articles'").fetchone()[0]==1
        row=db.execute("SELECT * FROM tasks WHERE target='Black Beauty'").fetchone()
        story_task(db,fixture,row,args)
        assert db.execute("SELECT 1 FROM tasks WHERE target='Black Beauty/1'").fetchone()
        assert not db.execute("SELECT 1 FROM tasks WHERE target='Other book'").fetchone()
        stored=db.execute("SELECT body FROM records WHERE bucket='animal_stories'").fetchone()[0]
        assert 'bad()' not in stored and 'Navigation' not in stored
        row=db.execute("SELECT * FROM tasks WHERE source='research' ORDER BY id LIMIT 1").fetchone()
        research_task(db,fixture,row,args)
        assert fixture.calls[-1][1]['pageSize']==1  # Do not skip the rest of a page at the cap.
        updated=db.execute('SELECT * FROM tasks WHERE id=?',(row['id'],)).fetchone()
        assert json.loads(updated['payload'])['cursorMark']=='next'
        args.max_studies=2
        research_task(db,fixture,updated,args)
        assert db.execute('SELECT status FROM tasks WHERE id=?',(row['id'],)).fetchone()[0]=='done'
        # DOI dedup across queries and changed identifier casing.
        other=db.execute("SELECT * FROM tasks WHERE source='research' AND id!=? LIMIT 1",(row['id'],)).fetchone()
        research_task(db,fixture,other,args)
        assert db.execute("SELECT count(*) FROM records WHERE bucket='animal_studies'").fetchone()[0]==1
        fact=Factory(); mem=Memory(); bridge=KnowledgeBridge(fact,mem,True)
        bridge.sync(db,full=True); bridge.sync(db)
        assert fact.calls==1 and fact.agent.calls==3
        assert bridge.query('animal',2)[0]['content_type']
        # Identical words in a story and an article stay distinct in the agent index.
        with db:
            save(db,'fiction-identical','animal_stories','Fiction example','Animals form a diverse biological kingdom.',
                 {'source':'fixture','url':'https://example.org','content_type':'literary_text','origins':[]})
        bridge.sync(db)
        assert len(fact.agent.doc_index)==4
        bridge.close(); bridge.close(); assert fact.closed and mem.closed
        export(db,folder)
        assert sorted(p.name for p in folder.glob('*.txt'))==sorted(b+'.txt' for b in BUCKETS)
        db.close(); db=connect(folder/'history.sqlite3')
        fresh=Factory(); KnowledgeBridge(fresh,Memory()).sync(db,full=True)
        assert fresh.agent.calls==4  # Replay even when dirty queue was already drained.
        class BrokenAgent(Agent):
            def add_document(self,**kwargs): raise RuntimeError('Simulated indexing failure')
        broken=Factory(); broken.agent=BrokenAgent()
        try: KnowledgeBridge(broken,Memory()).sync(db,full=True)
        except RuntimeError: pass
        else: raise AssertionError('Indexing failure hidden')
        assert db.execute('SELECT count(*) FROM records').fetchone()[0]==4
        db.close()
        # Standalone main export uses no factory imports or network.
        assert main(['--output-dir',str(folder),'--export-only'])==0
    print('PASS: category cursors, canonical dedup, research cap/cursor resume, story scope and extraction, '
          'rights classification, three exports, factory lifecycle, dirty queue, fiction separation, '
          'restart replay, failure durability and offline CLI export.')



if __name__=='__main__':
    sys.exit(main())
