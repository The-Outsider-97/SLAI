from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import requests

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from statistics import mean, pstdev
from typing import Any, Callable, Deque, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple
from urllib.parse import quote, urlencode
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
from requests import Response, Session
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry
from dotenv import load_dotenv

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover - optional dependency
    SentimentIntensityAnalyzer = None

try:
    from textblob import TextBlob
except Exception:  # pragma: no cover - optional dependency
    TextBlob = None

from .config_loader import get_config_section, load_global_config
from .resource_loader import ResourceLoader
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Public Sentiment Scraper")
printer = PrettyPrinter


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}

DEFAULT_SOURCE_WEIGHTS = {
    "yahoo_finance": 1.40,
    "bloomberg": 1.50,
    "investing": 1.25,
    "finviz": 1.20,
    "news_api": 1.35,
    "google_news": 1.10,
    "reddit": 0.85,
    "twitter": 0.65,
}

NEGATION_SCOPE = 4
MAX_PHRASE_LENGTH = 3
DEFAULT_CACHE_TTL_SECONDS = 1800
DEFAULT_HISTORY_RETENTION = 1000


@dataclass(frozen=True)
class SentimentSnippet:
    title: str = ""
    text: str = ""
    source: str = ""
    url: str = ""
    published_at: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        # Backward-compatible aliases expected by legacy code.
        if self.text:
            payload.setdefault("snippet", self.text)
        if self.published_at:
            payload.setdefault("date", self.published_at)
        return payload


class TTLMemoryCache:
    def __init__(self, ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS, maxsize: int = 512) -> None:
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.maxsize = max(1, int(maxsize))
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any:
        now = time.time()
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at < now:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        now = time.time()
        with self._lock:
            if len(self._store) >= self.maxsize:
                oldest_key = min(self._store, key=lambda existing_key: self._store[existing_key][0])
                self._store.pop(oldest_key, None)
            self._store[key] = (now + self.ttl_seconds, value)


class RateLimiter:
    def __init__(self, default_interval_seconds: float = 1.0) -> None:
        self.default_interval_seconds = max(0.0, float(default_interval_seconds))
        self._last_called: Dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, key: str, interval_seconds: Optional[float] = None) -> None:
        interval = self.default_interval_seconds if interval_seconds is None else max(0.0, float(interval_seconds))
        with self._lock:
            now = time.monotonic()
            previous = self._last_called.get(key, 0.0)
            sleep_for = interval - (now - previous)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_called[key] = time.monotonic()


class PublicSentimentScraper:
    """
    Production-oriented financial sentiment scraper with:
    - resilient HTTP session and retries
    - thread-safe TTL cache
    - concurrent source fan-out
    - lazy optional sentiment analyzers
    - deterministic, non-synthetic history aggregation

    The public methods preserve the original integration points:
      - get_sentiment_snippets
      - compute_average_sentiment
      - get_sentiment_history
      - generate_sentiment_report
    """

    def __init__(self, *,
        session: Optional[Session] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        if load_dotenv:
            load_dotenv()
        self.config = load_global_config()
        self.pss_config = get_config_section('public_sentiment_scraper')

        self.headers = {**DEFAULT_HEADERS, **self.pss_config.get("headers", {})}
        self.rate_limit_seconds = float(self.pss_config.get("rate_limit", 1.0))
        self.max_workers = max_workers or int(self.pss_config.get("max_workers", 4))
        self.history_retention = int(self.pss_config.get("history_retention", DEFAULT_HISTORY_RETENTION))
        self.history: Dict[str, Any] = {
            "sources": defaultdict(lambda: deque(maxlen=self.history_retention)),
            "sentiment_observations": deque(maxlen=self.history_retention),
        }
        self.cache = TTLMemoryCache(
            ttl_seconds=int(self.pss_config.get("cache_duration", DEFAULT_CACHE_TTL_SECONDS)),
            maxsize=int(self.pss_config.get("cache_maxsize", 512)),
        )
        self.rate_limiter = RateLimiter(default_interval_seconds=self.rate_limit_seconds)
        self.session = session or self._build_session()
        self._owns_session = session is None

        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.reddit_api_key = os.getenv("REDDIT_API_KEY")
        self.sources = self._normalize_sources(self.pss_config.get("sources", {}))

        self.lexicon = self._load_lexicon()
        self._compiled_lexicon = self._compile_lexicon(self.lexicon)
        self._transformer_pipeline = None
        self._transformer_lock = threading.Lock()
        self.analyzers = self._initialize_analyzers()

        self._source_fetchers: Dict[str, Callable[[str, int], List[Dict[str, Any]]]] = {
            "finviz": self.scrape_finviz,
            "investing": self.scrape_investing,
            "bloomberg": self.scrape_bloomberg,
            "news_api": self.scrape_news_api,
            "google_news": self.scrape_google_news,
            "reddit": self.scrape_reddit,
            "twitter": self.scrape_twitter,
            "yahoo_finance": self.scrape_yahoo_finance,
        }

        self._validate_api_keys()

    # ------------------------------------------------------------------
    # Lifecycle / setup
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def __enter__(self) -> "PublicSentimentScraper":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _resolve_config(self, provided: Optional[MutableMapping[str, Any]]) -> Dict[str, Any]:
        if provided is not None:
            return dict(provided)
        if load_global_config is None:
            return {}
        try:
            loaded = load_global_config()
            return loaded if isinstance(loaded, dict) else {}
        except Exception as exc:  # pragma: no cover - depends on external project layout
            logger.warning("Falling back to defaults because config loading failed: %s", exc)
            return {}

    def _normalize_sources(self, sources: Any) -> Dict[str, bool]:
        if isinstance(sources, dict):
            return {str(name): bool(enabled) for name, enabled in sources.items()}
        if isinstance(sources, (list, tuple, set)):
            return {str(name): True for name in sources}
        return {
            "finviz": True,
            "google_news": True,
            "news_api": True,
            "yahoo_finance": True,
            "reddit": False,
            "twitter": False,
            "investing": False,
            "bloomberg": False,
        }

    def _build_session(self) -> Session:
        session = requests.Session()
        retry = Retry(
            total=4,
            connect=4,
            read=4,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "HEAD"]),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(self.headers)
        return session

    def _load_lexicon(self) -> Dict[str, Any]:
        if ResourceLoader is None:
            return {"positive": {}, "negative": {}, "intensifiers": {}, "negators": []}
        try:
            return ResourceLoader.get_loughran_mcdonald_lexicon()
        except Exception as exc:
            logger.warning("Failed to load financial lexicon; falling back to empty lexicon: %s", exc)
            return {"positive": {}, "negative": {}, "intensifiers": {}, "negators": []}

    def _compile_lexicon(self, lexicon: Dict[str, Any]) -> Dict[str, Any]:
        positive = {str(k).lower(): float(v) for k, v in lexicon.get("positive", {}).items()}
        negative = {str(k).lower(): -abs(float(v)) for k, v in lexicon.get("negative", {}).items()}
        combined = {**positive, **negative}
        phrases = {
            term: len(term.split())
            for term in combined
            if 1 < len(term.split()) <= MAX_PHRASE_LENGTH
        }
        intensifiers = {str(k).lower(): float(v) for k, v in lexicon.get("intensifiers", {}).items()}
        negators = {str(term).lower() for term in lexicon.get("negators", [])}
        return {
            "combined": combined,
            "phrases": phrases,
            "intensifiers": intensifiers,
            "negators": negators,
        }

    def _initialize_analyzers(self) -> Dict[str, Any]:
        analyzers: Dict[str, Any] = {"lexicon": self._lexicon_sentiment}
        if SentimentIntensityAnalyzer is not None:
            try:
                analyzers["vader"] = SentimentIntensityAnalyzer()
            except Exception as exc:
                sf.logger.warning("Failed to initialize VADER: %s", exc)
        if TextBlob is not None:
            analyzers["textblob"] = lambda text: TextBlob(text).sentiment.polarity
        if self.pss_config.get("use_transformers", False):
            analyzers["transformer"] = self._transformer_sentiment
        return analyzers

    def _validate_api_keys(self) -> None:
        if self.sources.get("news_api") and not self.news_api_key:
            logger.warning("NEWS_API_KEY is missing; disabling news_api source.")
            self.sources["news_api"] = False
        if self.sources.get("reddit") and not self.reddit_api_key:
            # The public JSON endpoints work without a token. Keep source enabled.
            logger.info("REDDIT_API_KEY is not configured; using public Reddit JSON endpoints.")

    # ------------------------------------------------------------------
    # Core request helpers
    # ------------------------------------------------------------------
    def _request(self, source: str, url: str, *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 15,
        accept_json: bool = False,
    ) -> Optional[Response]:
        self.rate_limiter.wait(source)
        request_headers = dict(self.headers)
        if headers:
            request_headers.update(headers)

        for attempt in range(4):
            try:
                response = self.session.get(url, params=params, headers=request_headers, timeout=timeout)
                if response.status_code == 429 and attempt < 3:
                    sleep_for = (2 ** attempt) + random.uniform(0, 0.25)
                    logger.warning("Rate-limited by %s. Retrying in %.2fs", source, sleep_for)
                    time.sleep(sleep_for)
                    continue
                response.raise_for_status()
                if accept_json and "json" not in response.headers.get("Content-Type", "").lower():
                    logger.warning("%s returned non-JSON content for JSON request: %s", source, url)
                return response
            except RequestException as exc:
                if attempt >= 3:
                    logger.warning("Request failed for source=%s url=%s: %s", source, url, exc)
                    return None
                time.sleep((2 ** attempt) * 0.25)
        return None

    def _record_results(self, source: str, results: Sequence[Dict[str, Any]]) -> None:
        source_history: Deque[Dict[str, Any]] = self.history["sources"][source]
        source_history.append(
            {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "count": len(results),
                "results": list(results),
            }
        )

    def _parse_iso_datetime(self, value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
        try:
            dt = parsedate_to_datetime(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def _canonical_timestamp(self, value: Optional[str]) -> str:
        if not value:
            return datetime.now(timezone.utc).isoformat()
        parsed = self._parse_iso_datetime(value)
        return parsed.isoformat() if parsed else datetime.now(timezone.utc).isoformat()

    def _snippet_from_parts(
        self,
        *,
        title: str = "",
        text: str = "",
        source: str,
        url: str = "",
        published_at: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        snippet = SentimentSnippet(
            title=(title or "").strip(),
            text=(text or "").strip(),
            source=source,
            url=(url or "").strip(),
            published_at=(published_at or "").strip(),
            timestamp=self._canonical_timestamp(published_at),
            metadata=metadata or {},
        )
        return snippet.to_dict()

    def _is_source_enabled(self, source: str) -> bool:
        return bool(self.sources.get(source, False))

    # ------------------------------------------------------------------
    # Source scrapers
    # ------------------------------------------------------------------
    def scrape_finviz(self, symbol: str, max_results: int = 10) -> List[Dict[str, Any]]:
        if not symbol:
            return []
        url = f"https://finviz.com/quote.ashx?t={quote(symbol)}"
        response = self._request("finviz", url)
        if response is None:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", class_="fullview-news-outer")
        if table is None:
            return []

        results: List[Dict[str, Any]] = []
        last_seen_date: Optional[datetime] = None

        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            time_cell, headline_cell = cells[0], cells[1]
            anchor = headline_cell.find("a")
            title = headline_cell.get_text(" ", strip=True)
            link = anchor.get("href", "").strip() if anchor else ""
            raw_time = time_cell.get_text(" ", strip=True)

            published_at = datetime.now(timezone.utc)
            try:
                if " " in raw_time:
                    date_str, time_str = raw_time.split(maxsplit=1)
                    parsed = datetime.strptime(f"{date_str} {time_str}", "%b-%d-%y %I:%M%p")
                    last_seen_date = parsed.replace(tzinfo=timezone.utc)
                    published_at = last_seen_date
                elif last_seen_date is not None:
                    time_only = datetime.strptime(raw_time, "%I:%M%p").time()
                    published_at = datetime.combine(last_seen_date.date(), time_only, tzinfo=timezone.utc)
            except ValueError:
                published_at = datetime.now(timezone.utc)

            results.append(
                self._snippet_from_parts(
                    title=title,
                    source="Finviz",
                    url=link,
                    published_at=published_at.isoformat(),
                )
            )
            if len(results) >= max_results:
                break

        self._record_results("finviz", results)
        return results

    def scrape_investing(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        if not query:
            return []
        url = f"https://www.investing.com/search/"
        response = self._request(
            "investing",
            url,
            params={"q": query},
            headers={"User-Agent": DEFAULT_HEADERS["User-Agent"]},
        )
        if response is None:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        selectors = [
            "section.js-article-item",
            "article[data-test='article-item']",
            "article",
        ]

        results: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()

        for selector in selectors:
            articles = soup.select(selector)
            if not articles:
                continue
            for article in articles:
                title_element = (
                    article.select_one("a.title")
                    or article.select_one("a[data-test='article-title-link']")
                    or article.find("a")
                )
                if title_element is None:
                    continue
                title = title_element.get_text(" ", strip=True)
                href = title_element.get("href", "")
                if href and href.startswith("/"):
                    href = f"https://www.investing.com{href}"
                date_element = article.select_one("span.date") or article.find("time")
                text_element = article.select_one("p")
                result = self._snippet_from_parts(
                    title=title,
                    text=text_element.get_text(" ", strip=True) if text_element else "",
                    source="Investing.com",
                    url=href,
                    published_at=(date_element.get_text(" ", strip=True) if date_element else ""),
                )
                signature = (result.get("title", ""), result.get("url", ""))
                if signature in seen:
                    continue
                seen.add(signature)
                results.append(result)
                if len(results) >= max_results:
                    self._record_results("investing", results)
                    return results
            break

        self._record_results("investing", results)
        return results

    def scrape_bloomberg(self, ticker: str, max_results: int = 10) -> List[Dict[str, Any]]:
        if not ticker:
            return []
        try:
            import blpapi  # type: ignore
            from blpapi import Session, SessionOptions  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.info("Bloomberg dependency is unavailable; skipping Bloomberg source: %s", exc)
            return []

        session = None
        results: List[Dict[str, Any]] = []
        try:
            options = SessionOptions()
            options.setServerHost("localhost")
            options.setServerPort(8194)
            session = Session(options)
            if not session.start() or not session.openService("//blp/news"):
                logger.warning("Bloomberg session/service could not be opened.")
                return []
            service = session.getService("//blp/news")
            request = service.createRequest("newsRequest")
            request.getElement("query").setValue(ticker)
            request.getElement("maxStories").setValue(max_results)
            session.sendRequest(request)

            while True:
                event = session.nextEvent(500)
                for msg in event:
                    if msg.hasElement("storyTitle"):
                        title = msg.getElementAsString("storyTitle")
                        results.append(
                            self._snippet_from_parts(
                                title=title,
                                source="Bloomberg",
                                published_at=datetime.now(timezone.utc).isoformat(),
                            )
                        )
                        if len(results) >= max_results:
                            self._record_results("bloomberg", results)
                            return results
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
        except Exception as exc:
            logger.warning("Bloomberg scrape failed for %s: %s", ticker, exc)
        finally:
            if session is not None:
                try:
                    session.stop()
                except Exception:
                    pass

        self._record_results("bloomberg", results)
        return results

    def scrape_google_news(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        if not query:
            return []
        rss_query = urlencode({"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
        url = f"https://news.google.com/rss/search?{rss_query}"
        response = self._request("google_news", url, headers={"Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8"})
        if response is None:
            return []

        results: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(response.content)
            channel = root.find("channel")
            if channel is None:
                return []
            for item in channel.findall("item")[:max_results]:
                title = item.findtext("title", default="")
                link = item.findtext("link", default="")
                pub_date = item.findtext("pubDate", default="")
                description = item.findtext("description", default="")
                results.append(
                    self._snippet_from_parts(
                        title=title,
                        text=BeautifulSoup(description, "html.parser").get_text(" ", strip=True),
                        source="Google News",
                        url=link,
                        published_at=pub_date,
                    )
                )
        except ET.ParseError as exc:
            logger.warning("Google News RSS parsing failed: %s", exc)

        self._record_results("google_news", results)
        return results

    def scrape_news_api(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        if not query or not self.news_api_key:
            return []
        response = self._request(
            "news_api",
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "pageSize": min(max_results, 100),
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": self.news_api_key,
            },
            accept_json=True,
        )
        if response is None:
            return []

        try:
            payload = response.json()
        except ValueError:
            logger.warning("News API returned invalid JSON.")
            return []

        results = [
            self._snippet_from_parts(
                title=article.get("title", ""),
                text=(article.get("description") or article.get("content") or ""),
                source=(article.get("source") or {}).get("name", "News API"),
                url=article.get("url", ""),
                published_at=article.get("publishedAt", ""),
                metadata={"author": article.get("author", "")},
            )
            for article in payload.get("articles", [])[:max_results]
        ]
        self._record_results("news_api", results)
        return results

    def scrape_yahoo_finance(self, symbol: str, max_results: int = 10) -> List[Dict[str, Any]]:
        if not symbol:
            return []
        url = f"https://finance.yahoo.com/quote/{quote(symbol)}/news"
        response = self._request("yahoo_finance", url)
        if response is None:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        results: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()

        containers = soup.select("li.js-stream-content") or soup.select("section section article") or soup.select("article")
        for item in containers:
            title_element = item.select_one("h3") or item.find("h3") or item.find("a")
            if title_element is None:
                continue
            title = title_element.get_text(" ", strip=True)
            link_element = title_element.find_parent("a") if title_element.name != "a" else title_element
            if link_element is None:
                link_element = item.find("a")
            href = link_element.get("href", "").strip() if link_element else ""
            if href.startswith("/"):
                href = f"https://finance.yahoo.com{href}"
            summary_element = item.find("p")
            source_element = item.select_one("div.C\\(#959595\\)") or item.find("span")
            signature = (title, href)
            if not title or signature in seen:
                continue
            seen.add(signature)
            results.append(
                self._snippet_from_parts(
                    title=title,
                    text=summary_element.get_text(" ", strip=True) if summary_element else "",
                    source=(source_element.get_text(" ", strip=True) if source_element else "Yahoo Finance"),
                    url=href,
                    published_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            if len(results) >= max_results:
                break

        self._record_results("yahoo_finance", results)
        return results

    def scrape_reddit(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        if not query:
            return []
        response = self._request(
            "reddit",
            "https://www.reddit.com/search.json",
            params={"q": query, "limit": max_results, "sort": "relevance", "raw_json": 1},
            headers={"User-Agent": "financial-agent/1.0"},
            accept_json=True,
        )
        if response is None:
            return []

        try:
            payload = response.json()
        except ValueError:
            logger.warning("Reddit returned invalid JSON.")
            return []

        children = payload.get("data", {}).get("children", [])
        results: List[Dict[str, Any]] = []
        for child in children[:max_results]:
            post = child.get("data", {})
            created_utc = post.get("created_utc")
            published_at = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat() if created_utc else ""
            results.append(
                self._snippet_from_parts(
                    title=post.get("title", ""),
                    text=post.get("selftext", "")[:1000],
                    source=f"Reddit/r/{post.get('subreddit', '')}" if post.get("subreddit") else "Reddit",
                    url=(post.get("url_overridden_by_dest") or f"https://www.reddit.com{post.get('permalink', '')}"),
                    published_at=published_at,
                    metadata={
                        "author": post.get("author", ""),
                        "score": post.get("score", 0),
                        "comments": post.get("num_comments", 0),
                    },
                )
            )

        self._record_results("reddit", results)
        return results

    def scrape_twitter(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        if not query:
            return []
        response = self._request(
            "twitter",
            "https://nitter.net/search",
            params={"f": "tweets", "q": query},
        )
        if response is None:
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        results: List[Dict[str, Any]] = []
        for item in soup.select("div.timeline-item"):
            text_element = item.select_one("div.tweet-content")
            link_element = item.select_one("a.tweet-link") or item.find("a")
            if text_element is None:
                continue
            href = link_element.get("href", "") if link_element else ""
            if href.startswith("/"):
                href = f"https://nitter.net{href}"
            results.append(
                self._snippet_from_parts(
                    text=text_element.get_text(" ", strip=True),
                    source="Twitter/Nitter",
                    url=href,
                    published_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            if len(results) >= max_results:
                break

        self._record_results("twitter", results)
        return results

    # ------------------------------------------------------------------
    # Sentiment scoring
    # ------------------------------------------------------------------
    def _normalize_text(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s']", " ", (text or "").lower())
        return [token for token in cleaned.split() if token]

    def _lexicon_sentiment(self, text: str) -> float:
        tokens = self._normalize_text(text)
        if not tokens:
            return 0.0

        lexicon = self._compiled_lexicon["combined"]
        phrase_dict = self._compiled_lexicon["phrases"]
        intensifiers = self._compiled_lexicon["intensifiers"]
        negators = self._compiled_lexicon["negators"]

        score = 0.0
        weight_sum = 0.0
        negation_active = False
        negation_distance = 0
        intensity_factor = 1.0
        index = 0

        while index < len(tokens):
            token = tokens[index]
            matched_phrase = None

            for length in range(min(MAX_PHRASE_LENGTH, len(tokens) - index), 0, -1):
                phrase = " ".join(tokens[index : index + length])
                if phrase in phrase_dict:
                    matched_phrase = phrase
                    break

            if matched_phrase is None:
                if token in negators:
                    negation_active = True
                    negation_distance = NEGATION_SCOPE
                    index += 1
                    continue
                if token in intensifiers:
                    intensity_factor *= intensifiers[token]
                    index += 1
                    continue

            if matched_phrase or token in lexicon:
                key = matched_phrase or token
                raw_sentiment = lexicon[key]
                token_count = len(key.split()) if matched_phrase else 1
                if negation_active and negation_distance > 0:
                    raw_sentiment *= -1.0
                    negation_distance -= token_count
                adjusted = raw_sentiment * intensity_factor
                score += adjusted
                weight_sum += abs(adjusted)
                intensity_factor = 1.0
                index += token_count
                continue

            index += 1
            if negation_active:
                negation_distance -= 1
                if negation_distance <= 0:
                    negation_active = False

        return score / weight_sum if weight_sum > 0 else 0.0

    def _ensure_transformer(self) -> Optional[Callable[..., Any]]:
        if self._transformer_pipeline is not None:
            return self._transformer_pipeline
        with self._transformer_lock:
            if self._transformer_pipeline is not None:
                return self._transformer_pipeline
            try:
                import torch  # type: ignore
                from transformers import AutoTokenizer, pipeline  # type: ignore

                model_name = self.pss_config.get(
                    "transformer_model",
                    "distilbert-base-uncased-finetuned-sst-2-english",
                )
                self._transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=AutoTokenizer.from_pretrained(model_name),
                    device=0 if torch.cuda.is_available() else -1,
                )
                logger.info("Loaded transformer model: %s", model_name)
            except Exception as exc:
                logger.warning("Transformer model could not be initialized: %s", exc)
                self._transformer_pipeline = None
        return self._transformer_pipeline

    def _transformer_sentiment(self, text: str) -> float:
        model = self._ensure_transformer()
        if model is None:
            return 0.0
        try:
            cleaned = text.strip()
            if not cleaned:
                return 0.0
            max_chars = 400
            chunks = [cleaned[i : i + max_chars] for i in range(0, len(cleaned), max_chars)]
            chunk_scores: List[float] = []
            for chunk in chunks[:8]:
                result = model(chunk, truncation=True, max_length=512)
                label = result[0].get("label", "").upper()
                score = float(result[0].get("score", 0.0))
                chunk_scores.append(score if "POSITIVE" in label else -score)
            return sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
        except Exception as exc:
            logger.warning("Transformer sentiment failed: %s", exc)
            return 0.0

    def compute_sentiment(self, text: str, method: str = "ensemble") -> float:
        candidate = (text or "").strip()
        if not candidate:
            return 0.0

        if method != "ensemble":
            return self._compute_sentiment_single(candidate, method)

        methods = [name for name in ("lexicon", "vader", "textblob", "transformer") if name in self.analyzers]
        text_length = len(candidate.split())
        contains_financial_terms = any(
            term in candidate.lower()
            for term in ("stock", "earnings", "guidance", "dividend", "market", "revenue", "margin", "eps")
        )

        weights = {
            "lexicon": 0.42 if text_length >= 6 else 0.30,
            "vader": 0.20,
            "textblob": 0.12 if text_length >= 12 else 0.22,
            "transformer": 0.35 if text_length >= 16 else 0.28,
        }
        if contains_financial_terms:
            weights["lexicon"] = weights.get("lexicon", 0.0) + 0.08
            if "transformer" in weights:
                weights["transformer"] = weights.get("transformer", 0.0) + 0.03

        weighted_sum = 0.0
        total_weight = 0.0
        for analyzer_name in methods:
            score = self._compute_sentiment_single(candidate, analyzer_name)
            weight = weights.get(analyzer_name, 0.2)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0
        return max(-1.0, min(1.0, weighted_sum / total_weight))

    def _compute_sentiment_single(self, text: str, method: str) -> float:
        try:
            if method == "vader" and "vader" in self.analyzers:
                return float(self.analyzers["vader"].polarity_scores(text)["compound"])
            if method in self.analyzers:
                return float(self.analyzers[method](text))
        except Exception as exc:
            logger.warning("Sentiment analyzer '%s' failed: %s", method, exc)
        return 0.0

    # ------------------------------------------------------------------
    # Aggregation API
    # ------------------------------------------------------------------
    def get_sentiment_snippets(self, query: str, max_results: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        cache_key = f"snippets::{query.lower().strip()}::{int(max_results)}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        enabled_sources = [source for source, enabled in self.sources.items() if enabled and source in self._source_fetchers]
        if not enabled_sources:
            return {}

        all_results: Dict[str, List[Dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(enabled_sources))) as executor:
            future_to_source = {
                executor.submit(self._source_fetchers[source], query, max_results): source
                for source in enabled_sources
            }
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    raw_results = future.result() or []
                    all_results[source] = self._deduplicate_snippets(raw_results)[:max_results]
                except Exception as exc:
                    logger.warning("Source '%s' failed during scrape: %s", source, exc)
                    all_results[source] = []

        self.cache.set(cache_key, all_results)
        return all_results

    def _deduplicate_snippets(self, snippets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()
        for snippet in snippets:
            title = (snippet.get("title") or "").strip().lower()
            url = (snippet.get("url") or "").strip().lower()
            text = (snippet.get("text") or snippet.get("snippet") or "").strip().lower()[:180]
            signature = (title, url, text)
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(snippet)
        return deduped

    def _extract_text_payload(self, snippet: Dict[str, Any]) -> str:
        return " ".join(
            part.strip()
            for part in (
                snippet.get("title", ""),
                snippet.get("text", ""),
                snippet.get("snippet", ""),
                snippet.get("description", ""),
                snippet.get("content", ""),
            )
            if isinstance(part, str) and part.strip()
        )

    def _recency_weight(self, timestamp: str) -> float:
        parsed = self._parse_iso_datetime(timestamp)
        if parsed is None:
            return 1.0
        age_hours = max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds() / 3600.0)
        if age_hours <= 6:
            return 1.20
        if age_hours <= 24:
            return 1.10
        if age_hours <= 72:
            return 1.00
        return 0.90

    def compute_average_sentiment(self, query: str, max_results: int = 10, method: str = "ensemble") -> float:
        snippets_by_source = self.get_sentiment_snippets(query, max_results=max_results)
        weighted_sum = 0.0
        total_weight = 0.0

        for source_key, snippets in snippets_by_source.items():
            base_weight = DEFAULT_SOURCE_WEIGHTS.get(source_key, 1.0)
            for snippet in snippets:
                text = self._extract_text_payload(snippet)
                if not text:
                    continue
                sentiment = self.compute_sentiment(text, method=method)
                recency_weight = self._recency_weight(snippet.get("timestamp") or snippet.get("published_at") or "")
                final_weight = base_weight * recency_weight
                weighted_sum += sentiment * final_weight
                total_weight += final_weight
                self.history["sentiment_observations"].append(
                    {
                        "query": query,
                        "source": source_key,
                        "timestamp": snippet.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                        "sentiment": sentiment,
                        "title": snippet.get("title", ""),
                    }
                )

        if total_weight == 0:
            return 0.0
        return max(-1.0, min(1.0, weighted_sum / total_weight))

    def get_sentiment_history(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        if days <= 0:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        grouped: Dict[str, List[float]] = defaultdict(list)

        for observation in list(self.history["sentiment_observations"]):
            if observation.get("query") != symbol:
                continue
            parsed = self._parse_iso_datetime(observation.get("timestamp", ""))
            if parsed is None or parsed < cutoff:
                continue
            grouped[parsed.date().isoformat()].append(float(observation.get("sentiment", 0.0)))

        history = [
            {"date": date_key, "sentiment": sum(values) / len(values), "symbol": symbol, "samples": len(values)}
            for date_key, values in sorted(grouped.items())
        ]
        return history

    def generate_sentiment_report(self, symbol: str, max_results: int = 10, days: int = 7) -> Dict[str, Any]:
        current_sentiment = self.compute_average_sentiment(symbol, max_results=max_results)
        historical = self.get_sentiment_history(symbol, days=days)
        historical_values = [item["sentiment"] for item in historical]

        mean_sentiment = mean(historical_values) if historical_values else current_sentiment
        volatility = pstdev(historical_values) if len(historical_values) > 1 else 0.0

        trend = "Neutral"
        if current_sentiment > mean_sentiment + volatility:
            trend = "Bullish"
        elif current_sentiment < mean_sentiment - volatility:
            trend = "Bearish"

        return {
            "symbol": symbol,
            "current_sentiment": current_sentiment,
            "mean_sentiment": mean_sentiment,
            "sentiment_volatility": volatility,
            "sentiment_trend": trend,
            "historical": historical,
            "sources_enabled": [source for source, enabled in self.sources.items() if enabled],
            "report_time": datetime.now(timezone.utc).isoformat(),
        }


if __name__ == "__main__":
    scraper = PublicSentimentScraper()
    try:
        query = "AAPL"
        snippets = scraper.get_sentiment_snippets(query, max_results=5)
        print(json.dumps(snippets, indent=2)[:4000])
        print("Average sentiment:", scraper.compute_average_sentiment(query, max_results=5))
        print(json.dumps(scraper.generate_sentiment_report(query, max_results=5), indent=2))
    finally:
        scraper.close()
