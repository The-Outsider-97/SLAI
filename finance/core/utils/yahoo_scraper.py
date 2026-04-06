from __future__ import annotations

import json
import re
import threading
import time
import requests

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from bs4 import BeautifulSoup
from requests import Session
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry

from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Yahoo Market Scraper")
printer = PrettyPrinter


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
ROOT_APP_MAIN_PATTERN = re.compile(r"root\.App\.main\s*=\s*(\{.*?\})\s*;\n", re.DOTALL)


@dataclass(frozen=True)
class YahooTickerRecord:
    symbol: str
    short_name: str = ""
    regular_market_price: Optional[float] = None
    regular_market_change_percent: Optional[float] = None
    regular_market_volume: Optional[int] = None
    market_cap: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _RateLimiter:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = max(0.0, float(interval_seconds))
        self._last_called = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_for = self.interval_seconds - (now - self._last_called)
            if wait_for > 0:
                time.sleep(wait_for)
            self._last_called = time.monotonic()


class YahooMarketScraper:
    """Resilient Yahoo Finance market scraper with retrying sessions and selector fallbacks."""

    BASE_URL = "https://finance.yahoo.com/markets/stocks/most-active"
    PAGE_SIZE = 100

    def __init__(
        self,
        *,
        session: Optional[Session] = None,
        headers: Optional[Dict[str, str]] = None,
        proxy: Optional[Dict[str, str]] = None,
        timeout: int = 10,
        rate_limit_seconds: float = 0.5,
    ) -> None:
        self.timeout = timeout
        self.proxy = proxy
        self.rate_limiter = _RateLimiter(rate_limit_seconds)
        self.session = session or self._build_session(headers=headers)
        self._owns_session = session is None

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def __enter__(self) -> "YahooMarketScraper":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _build_session(self, headers: Optional[Dict[str, str]] = None) -> Session:
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
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(DEFAULT_HEADERS)
        if headers:
            session.headers.update(headers)
        return session

    def _request_page(self, start: int) -> Optional[str]:
        self.rate_limiter.wait()
        try:
            response = self.session.get(
                self.BASE_URL,
                params={"start": start, "count": self.PAGE_SIZE},
                proxies=self.proxy,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.text
        except RequestException as exc:
            logger.warning("Failed to fetch Yahoo Finance page at start=%s: %s", start, exc)
            return None

    def fetch_most_active_stocks(
        self,
        count: int = 300,
        *,
        include_metadata: bool = False,
    ) -> Union[List[str], List[Dict[str, Any]]]:
        if count <= 0:
            return []

        normalized_count = min(max(int(count), 1), 1000)
        results: List[YahooTickerRecord] = []
        seen_symbols: set[str] = set()

        for start in range(0, normalized_count, self.PAGE_SIZE):
            html = self._request_page(start)
            if not html:
                continue

            page_records = self._extract_records_from_html(html)
            if not page_records:
                logger.warning("Could not parse most-active page at start=%s", start)
                continue

            for record in page_records:
                if record.symbol in seen_symbols:
                    continue
                seen_symbols.add(record.symbol)
                results.append(record)
                if len(results) >= normalized_count:
                    break
            if len(results) >= normalized_count:
                break

        if include_metadata:
            return [record.to_dict() for record in results]
        return [record.symbol for record in results]

    def _extract_records_from_html(self, html: str) -> List[YahooTickerRecord]:
        records = self._extract_records_from_embedded_json(html)
        if records:
            return records
        return self._extract_records_from_dom(html)

    def _extract_records_from_embedded_json(self, html: str) -> List[YahooTickerRecord]:
        match = ROOT_APP_MAIN_PATTERN.search(html)
        if not match:
            return []
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return []

        records: List[YahooTickerRecord] = []
        seen: set[str] = set()

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                symbol = node.get("symbol")
                if isinstance(symbol, str) and TICKER_PATTERN.match(symbol):
                    price = self._coerce_float(node.get("regularMarketPrice", node.get("price")))
                    volume = self._coerce_int(node.get("regularMarketVolume", node.get("volume")))
                    change_pct = self._coerce_float(
                        node.get("regularMarketChangePercent", node.get("changePercent"))
                    )
                    short_name = str(node.get("shortName") or node.get("longName") or "")
                    market_cap = self._coerce_int(node.get("marketCap"))
                    signature = (symbol, short_name, price, volume)
                    if symbol not in seen and any(value is not None for value in (price, volume, change_pct)):
                        seen.add(symbol)
                        records.append(
                            YahooTickerRecord(
                                symbol=symbol,
                                short_name=short_name,
                                regular_market_price=price,
                                regular_market_change_percent=change_pct,
                                regular_market_volume=volume,
                                market_cap=market_cap,
                            )
                        )
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(payload)
        return records

    def _extract_records_from_dom(self, html: str) -> List[YahooTickerRecord]:
        soup = BeautifulSoup(html, "html.parser")
        rows = soup.select("table tbody tr")
        if not rows:
            rows = soup.select("div.tableContainer a[href*='/quote/']")
            return self._extract_symbol_only_records(rows)

        records: List[YahooTickerRecord] = []
        seen: set[str] = set()

        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue
            link = row.find("a", href=re.compile(r"/quote/[^/?]+"))
            symbol = ""
            if link is not None:
                symbol = link.get_text(strip=True) or self._extract_symbol_from_href(link.get("href", ""))
            if not symbol or symbol in seen or not TICKER_PATTERN.match(symbol):
                continue
            seen.add(symbol)
            short_name = cells[1].get_text(" ", strip=True) if len(cells) > 1 else ""
            price = self._coerce_float(cells[2].get_text(strip=True) if len(cells) > 2 else None)
            change_pct = self._coerce_percent(cells[4].get_text(strip=True) if len(cells) > 4 else None)
            volume = self._coerce_int(cells[6].get_text(strip=True) if len(cells) > 6 else None)
            market_cap = self._coerce_market_cap(cells[9].get_text(strip=True) if len(cells) > 9 else None)
            records.append(
                YahooTickerRecord(
                    symbol=symbol,
                    short_name=short_name,
                    regular_market_price=price,
                    regular_market_change_percent=change_pct,
                    regular_market_volume=volume,
                    market_cap=market_cap,
                )
            )
        return records

    def _extract_symbol_only_records(self, links: Sequence[Any]) -> List[YahooTickerRecord]:
        records: List[YahooTickerRecord] = []
        seen: set[str] = set()
        for link in links:
            href = link.get("href", "") if hasattr(link, "get") else ""
            symbol = link.get_text(strip=True) if hasattr(link, "get_text") else self._extract_symbol_from_href(href)
            if not symbol:
                symbol = self._extract_symbol_from_href(href)
            if symbol and symbol not in seen and TICKER_PATTERN.match(symbol):
                seen.add(symbol)
                records.append(YahooTickerRecord(symbol=symbol))
        return records

    def _extract_symbol_from_href(self, href: str) -> str:
        match = re.search(r"/quote/([^/?]+)", href or "")
        if not match:
            return ""
        return match.group(1)

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None
        candidate = value.replace(",", "").replace("%", "").strip()
        try:
            return float(candidate)
        except ValueError:
            return None

    @staticmethod
    def _coerce_percent(value: Any) -> Optional[float]:
        return YahooMarketScraper._coerce_float(value)

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if not isinstance(value, str):
            return None
        candidate = value.replace(",", "").strip()
        if not candidate:
            return None
        multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
        suffix = candidate[-1].upper()
        if suffix in multipliers:
            try:
                return int(float(candidate[:-1]) * multipliers[suffix])
            except ValueError:
                return None
        try:
            return int(float(candidate))
        except ValueError:
            return None

    @staticmethod
    def _coerce_market_cap(value: Any) -> Optional[int]:
        return YahooMarketScraper._coerce_int(value)


def scrape_yahoo_most_active_stocks(
    count: int = 300,
    proxy: Optional[Dict[str, str]] = None,
    *,
    include_metadata: bool = False,
    timeout: int = 10,
) -> Union[List[str], List[Dict[str, Any]]]:
    """Backward-compatible wrapper around the production scraper."""
    with YahooMarketScraper(proxy=proxy, timeout=timeout) as scraper:
        return scraper.fetch_most_active_stocks(count=count, include_metadata=include_metadata)


if __name__ == "__main__":
    output = scrape_yahoo_most_active_stocks(count=50, include_metadata=True)
    print(json.dumps(output[:10], indent=2))
