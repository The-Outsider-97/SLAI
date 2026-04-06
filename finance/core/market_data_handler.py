from __future__ import annotations

import datetime as dt
import math
import os
import re
import time
import yfinance as yf
import requests

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from requests import Session
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry

from finance.core.utils.financial_errors import (DataUnavailableError, log_error,
                                                 ErrorContext, FinancialAgentError,
                                                 InvalidConfigurationError, ProviderError,
                                                 ProviderParsingError, ValidationError,
                                                 classify_external_exception)
from finance.core.finance_memory import FinanceMemory
from finance.core.utils.yahoo_scraper import YahooMarketScraper, scrape_yahoo_most_active_stocks
from finance.core.utils.public_sentiment_scraper import PublicSentimentScraper
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Market Data Handler")
printer = PrettyPrinter

DEFAULT_TIMEOUT_SECONDS = 12
DEFAULT_CACHE_TTL_SECONDS = 300
DEFAULT_MAX_WORKERS = 8
DEFAULT_PROVIDER_ORDER = ("polygon", "finnhub", "alpha_vantage", "yahoo")
VALID_FETCH_MODES = {"fallback", "fuse"}
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-/_]{0,19}$", re.IGNORECASE)


@dataclass(frozen=True)
class PriceBar:
    date: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    source: str
    adjusted_close: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["date"] = self.date
        return payload


@dataclass(frozen=True)
class ProviderAttempt:
    provider: str
    success: bool
    records: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class APIClientBase:
    provider_name = "base"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        session: Optional[Session] = None,
    ) -> None:
        self.api_key = api_key
        self.timeout = int(timeout)
        self.session = session or self._build_session()
        self._owns_session = session is None

    def close(self) -> None:
        if self._owns_session:
            self.session.close()

    def _build_session(self) -> Session:
        session = requests.Session()
        retry = Retry(
            total=4,
            connect=4,
            read=4,
            backoff_factor=0.4,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "HEAD"]),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json,text/plain,*/*",
            }
        )
        return session

    def fetch(self, symbol: str, lookback: int = 30) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _validate_response_payload(self, payload: Any, *, message: str) -> None:
        if payload is None:
            raise ProviderParsingError(message)

    def _request_json(self, url: str, *, params: Optional[Mapping[str, Any]] = None) -> Any:
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # pragma: no cover - depends on network
            raise classify_external_exception(
                exc,
                context=ErrorContext(component="market_data_handler", operation="provider_request", provider=self.provider_name, endpoint=url),
                message=f"{self.provider_name} request failed.",
            ) from exc


class AlphaVantageAPI(APIClientBase):
    provider_name = "alpha_vantage"
    BASE_URL = "https://www.alphavantage.co/query"

    def fetch(self, symbol: str, lookback: int = 30) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []

        if "-" in symbol and symbol.count("-") == 1:
            base, market = symbol.split("-", 1)
            params = {
                "function": "DIGITAL_CURRENCY_DAILY",
                "symbol": base,
                "market": market,
                "apikey": self.api_key,
            }
            payload = self._request_json(self.BASE_URL, params=params)
            series = payload.get("Time Series (Digital Currency Daily)", {})
        elif "/" in symbol and symbol.count("/") == 1:
            base, quote = symbol.split("/", 1)
            params = {
                "function": "FX_DAILY",
                "from_symbol": base,
                "to_symbol": quote,
                "apikey": self.api_key,
            }
            payload = self._request_json(self.BASE_URL, params=params)
            series = payload.get("Time Series FX (Daily)", {})
        else:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "compact" if lookback <= 100 else "full",
                "apikey": self.api_key,
            }
            payload = self._request_json(self.BASE_URL, params=params)
            series = payload.get("Time Series (Daily)", {})

        if not isinstance(series, Mapping) or not series:
            return []

        rows: List[PriceBar] = []
        for date_str in sorted(series.keys())[-lookback:]:
            row = series[date_str]
            try:
                rows.append(
                    PriceBar(
                        date=dt.datetime.strptime(date_str, "%Y-%m-%d"),
                        open=float(row.get("1a. open (USD)", row.get("1. open", 0.0))),
                        high=float(row.get("2a. high (USD)", row.get("2. high", 0.0))),
                        low=float(row.get("3a. low (USD)", row.get("3. low", 0.0))),
                        close=float(row.get("4a. close (USD)", row.get("4. close", 0.0))),
                        volume=float(row.get("5. volume", row.get("5a. volume (USD)", 0.0))),
                        symbol=symbol.upper(),
                        source=self.provider_name,
                    )
                )
            except Exception:
                continue
        return [bar.to_dict() for bar in rows]


class PolygonAPI(APIClientBase):
    provider_name = "polygon"

    def fetch(self, symbol: str, lookback: int = 30) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        end_date = dt.datetime.utcnow().date()
        start_date = end_date - dt.timedelta(days=max(int(lookback * 2.5), lookback + 10))
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/1/day/{start_date}/{end_date}"
        payload = self._request_json(
            url,
            params={
                "adjusted": "true",
                "sort": "asc",
                "limit": 5000,
                "apiKey": self.api_key,
            },
        )
        results = payload.get("results", []) if isinstance(payload, Mapping) else []
        if not isinstance(results, list) or not results:
            return []
        bars = [
            PriceBar(
                date=dt.datetime.fromtimestamp(item["t"] / 1000.0, tz=dt.timezone.utc).replace(tzinfo=None),
                open=float(item.get("o", 0.0)),
                high=float(item.get("h", 0.0)),
                low=float(item.get("l", 0.0)),
                close=float(item.get("c", 0.0)),
                volume=float(item.get("v", 0.0)),
                symbol=symbol.upper(),
                source=self.provider_name,
            )
            for item in results[-lookback:]
            if isinstance(item, Mapping) and "t" in item
        ]
        return [bar.to_dict() for bar in bars]


class FinnhubAPI(APIClientBase):
    provider_name = "finnhub"

    def fetch(self, symbol: str, lookback: int = 30) -> List[Dict[str, Any]]:
        if not self.api_key:
            return []
        end_ts = int(time.time())
        start_ts = end_ts - int(max(lookback * 2.5, lookback + 10) * 86400)
        payload = self._request_json(
            "https://finnhub.io/api/v1/stock/candle",
            params={
                "symbol": symbol.upper(),
                "resolution": "D",
                "from": start_ts,
                "to": end_ts,
                "token": self.api_key,
            },
        )
        if not isinstance(payload, Mapping) or payload.get("s") != "ok":
            return []
        timestamps = payload.get("t") or []
        opens = payload.get("o") or []
        highs = payload.get("h") or []
        lows = payload.get("l") or []
        closes = payload.get("c") or []
        volumes = payload.get("v") or []
        bars: List[PriceBar] = []
        for idx, ts in enumerate(timestamps[-lookback:]):
            try:
                data_idx = len(timestamps[-lookback:])
                # Map back to absolute position.
                pos = len(timestamps) - len(timestamps[-lookback:]) + idx
                bars.append(
                    PriceBar(
                        date=dt.datetime.fromtimestamp(ts),
                        open=float(opens[pos]),
                        high=float(highs[pos]),
                        low=float(lows[pos]),
                        close=float(closes[pos]),
                        volume=float(volumes[pos]),
                        symbol=symbol.upper(),
                        source=self.provider_name,
                    )
                )
            except Exception:
                continue
        return [bar.to_dict() for bar in bars]


class YahooFinanceAPI(APIClientBase):
    provider_name = "yahoo"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if yf is None:
            logger.warning("yfinance not available; YahooFinanceAPI will be inactive.")

    def fetch(self, symbol: str, lookback: int = 30) -> List[Dict[str, Any]]:
        if yf is None:
            return []
        try:
            history = yf.Ticker(symbol.upper()).history(period=f"{max(lookback + 5, lookback)}d", auto_adjust=False)
        except Exception as exc:  # pragma: no cover - depends on network
            raise classify_external_exception(
                exc,
                context=ErrorContext(component="market_data_handler", operation="fetch_provider", provider=self.provider_name, symbol=symbol.upper()),
                message=f"{self.provider_name} fetch failed.",
            ) from exc

        if history is None or history.empty:
            return []
        bars: List[PriceBar] = []
        tail = history.tail(lookback)
        for idx, row in tail.iterrows():
            try:
                when = idx.to_pydatetime().replace(tzinfo=None)
            except Exception:
                when = dt.datetime.utcnow()
            bars.append(
                PriceBar(
                    date=when,
                    open=float(row.get("Open", 0.0)),
                    high=float(row.get("High", 0.0)),
                    low=float(row.get("Low", 0.0)),
                    close=float(row.get("Close", 0.0)),
                    volume=float(row.get("Volume", 0.0)),
                    adjusted_close=float(row.get("Adj Close", row.get("Close", 0.0))),
                    symbol=symbol.upper(),
                    source=self.provider_name,
                )
            )
        return [bar.to_dict() for bar in bars]


class MarketDataHandler:
    """Production-ready market data orchestration layer.

    Key behavior:
    - canonicalizes OHLCV data from heterogeneous providers
    - supports fallback and multi-provider fusion modes
    - integrates with FinanceMemory for caching
    - exposes batch fetching, returns matrix building, and market snapshots
    - records provider diagnostics for health-aware orchestration
    """

    def __init__(
        self,
        api_config: Optional[Mapping[str, Any]] = None,
        *,
        memory: Optional[FinanceMemory] = None,
        sentiment_scraper: Optional[Any] = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        max_workers: int = DEFAULT_MAX_WORKERS,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self.config = dict(api_config or {})
        self.timeout = int(timeout)
        self.max_workers = max(1, int(max_workers))
        self.cache_ttl_seconds = max(1, int(cache_ttl_seconds))
        self.lock = RLock()
        self.provider_stats: MutableMapping[str, Dict[str, Any]] = {}
        self.last_batch_report: Dict[str, Any] = {}
        self.memory = memory if memory is not None else self._safe_build_memory()
        self.sentiment_scraper = sentiment_scraper
        if self.sentiment_scraper is None and self.config.get("enable_sentiment") and PublicSentimentScraper is not None:
            try:
                self.sentiment_scraper = PublicSentimentScraper()
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.warning("Failed to initialize sentiment scraper: %s", exc)

        self.provider_order = self._resolve_provider_order()
        self.api_clients = self._build_api_clients()

        if printer is not None:  # pragma: no cover - presentation only
            printer.status("INIT", f"MarketDataHandler ready with providers={list(self.api_clients)}", "success")

    def _safe_build_memory(self) -> Optional[FinanceMemory]:
        try:
            return FinanceMemory()
        except Exception as exc:  # pragma: no cover - depends on runtime wiring
            logger.warning("FinanceMemory unavailable for MarketDataHandler: %s", exc)
            return None

    def _resolve_provider_order(self) -> Tuple[str, ...]:
        configured = self.config.get("provider_order") or self.config.get("providers")
        if isinstance(configured, (list, tuple)) and configured:
            order = tuple(str(item).strip().lower() for item in configured if str(item).strip())
        else:
            order = DEFAULT_PROVIDER_ORDER
        return order

    def _build_api_clients(self) -> Dict[str, APIClientBase]:
        keys = self.config.get("api_keys", {}) if isinstance(self.config.get("api_keys"), Mapping) else {}
        polygon_key = self.config.get("polygon") or keys.get("polygon") or os.getenv("POLYGON_API_KEY")
        alpha_key = self.config.get("alpha_vantage") or keys.get("alpha_vantage") or os.getenv("ALPHA_VANTAGE_KEY")
        finnhub_key = self.config.get("finnhub") or keys.get("finnhub") or os.getenv("FINNHUB_KEY")

        clients: Dict[str, APIClientBase] = {}
        if polygon_key:
            clients["polygon"] = PolygonAPI(api_key=polygon_key, timeout=self.timeout)
        if finnhub_key:
            clients["finnhub"] = FinnhubAPI(api_key=finnhub_key, timeout=self.timeout)
        if alpha_key:
            clients["alpha_vantage"] = AlphaVantageAPI(api_key=alpha_key, timeout=self.timeout)
        if self.config.get("enable_yahoo", True):
            clients["yahoo"] = YahooFinanceAPI(timeout=self.timeout)

        ordered_clients: Dict[str, APIClientBase] = {}
        for name in self.provider_order:
            if name in clients:
                ordered_clients[name] = clients[name]
        for name, client in clients.items():
            ordered_clients.setdefault(name, client)
        return ordered_clients

    def close(self) -> None:
        for client in self.api_clients.values():
            try:
                client.close()
            except Exception:
                continue

    def _context(self, operation: str, *, symbol: Optional[str] = None, provider: Optional[str] = None, extra: Optional[Mapping[str, Any]] = None) -> ErrorContext:
        return ErrorContext(
            component="market_data_handler",
            operation=operation,
            symbol=symbol,
            provider=provider,
            metadata=dict(extra or {}),
        )

    def _validate_symbol(self, symbol: str) -> str:
        cleaned = (symbol or "").strip().upper()
        if not cleaned or not SYMBOL_PATTERN.match(cleaned):
            raise ValidationError(
                f"Invalid symbol '{symbol}'.",
                context=self._context("validate_symbol", symbol=str(symbol)),
                details={"symbol": symbol},
            )
        return cleaned

    def _canonicalize_series(self, symbol: str, provider: str, raw_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        canonical: List[Dict[str, Any]] = []
        seen_dates = set()
        for row in raw_rows:
            try:
                date_value = row.get("date")
                if hasattr(date_value, "to_pydatetime"):
                    date_value = date_value.to_pydatetime()
                if isinstance(date_value, dt.date) and not isinstance(date_value, dt.datetime):
                    date_value = dt.datetime.combine(date_value, dt.time.min)
                if isinstance(date_value, str):
                    date_value = dt.datetime.fromisoformat(date_value.replace("Z", "+00:00")).replace(tzinfo=None)
                if not isinstance(date_value, dt.datetime):
                    continue
                key = date_value.date().isoformat()
                if key in seen_dates:
                    continue
                open_ = float(row.get("open", 0.0))
                high = float(row.get("high", 0.0))
                low = float(row.get("low", 0.0))
                close = float(row.get("close", 0.0))
                volume = float(row.get("volume", 0.0))
                if any(math.isnan(value) or math.isinf(value) for value in (open_, high, low, close, volume)):
                    continue
                if close <= 0 or high < low:
                    continue
                canonical.append(
                    PriceBar(
                        date=date_value.replace(tzinfo=None),
                        open=open_,
                        high=high,
                        low=low,
                        close=close,
                        volume=max(volume, 0.0),
                        symbol=symbol,
                        source=provider,
                        adjusted_close=(float(row["adjusted_close"]) if row.get("adjusted_close") is not None else None),
                        metadata={k: v for k, v in row.items() if k not in {"date", "open", "high", "low", "close", "volume", "symbol", "source", "adjusted_close"}},
                    ).to_dict()
                )
                seen_dates.add(key)
            except Exception:
                continue
        canonical.sort(key=lambda item: item["date"])
        return canonical

    def _cache_get(self, namespace: str, key: str) -> Any:
        if self.memory is None:
            return None
        try:
            return self.memory.get_cache(key, namespace=namespace)
        except Exception:
            return None

    def _cache_set(self, namespace: str, key: str, value: Any, *, tags: Optional[List[str]] = None, priority: str = "medium") -> None:
        if self.memory is None:
            return
        try:
            self.memory.set_cache(
                key,
                value,
                namespace=namespace,
                ttl_seconds=self.cache_ttl_seconds,
                tags=tags or [],
                priority=priority,
            )
        except Exception as exc:
            logger.debug("Failed to write MarketDataHandler cache: %s", exc)

    def _record_attempt(self, attempt: ProviderAttempt) -> None:
        with self.lock:
            state = self.provider_stats.setdefault(
                attempt.provider,
                {
                    "successes": 0,
                    "failures": 0,
                    "last_error": None,
                    "last_success_at": None,
                    "last_duration_ms": 0.0,
                    "last_records": 0,
                },
            )
            if attempt.success:
                state["successes"] += 1
                state["last_success_at"] = time.time()
                state["last_records"] = attempt.records
            else:
                state["failures"] += 1
                state["last_error"] = attempt.error
            state["last_duration_ms"] = attempt.duration_ms

    def _fetch_from_provider(self, provider_name: str, client: APIClientBase, symbol: str, lookback: int) -> Tuple[str, List[Dict[str, Any]], ProviderAttempt]:
        started = time.perf_counter()
        try:
            data = client.fetch(symbol, lookback=lookback)
            canonical = self._canonicalize_series(symbol, provider_name, data)
            attempt = ProviderAttempt(
                provider=provider_name,
                success=bool(canonical),
                records=len(canonical),
                duration_ms=(time.perf_counter() - started) * 1000.0,
                error=None if canonical else "empty_result",
            )
            self._record_attempt(attempt)
            return provider_name, canonical, attempt
        except BaseException as exc:
            handled = exc if isinstance(exc, FinancialAgentError) else classify_external_exception(
                exc,
                context=self._context("fetch_provider", symbol=symbol, provider=provider_name),
                message=f"Provider fetch failed for {symbol}.",
            )
            log_error(handled, logger_=logger, include_traceback=False)
            attempt = ProviderAttempt(
                provider=provider_name,
                success=False,
                records=0,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                error=str(handled),
            )
            self._record_attempt(attempt)
            return provider_name, [], attempt

    def fetch_data(
        self,
        symbol: str,
        lookback: int = 30,
        mode: str = "fuse",
        *,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        symbol = self._validate_symbol(symbol)
        lookback = max(1, int(lookback))
        mode = (mode or "fuse").strip().lower()
        if mode not in VALID_FETCH_MODES:
            raise ValidationError(
                f"Invalid fetch mode '{mode}'.",
                context=self._context("fetch_data", symbol=symbol, extra={"mode": mode}),
                details={"allowed_modes": sorted(VALID_FETCH_MODES)},
            )

        cache_key = f"ohlcv:{symbol}:{lookback}:{mode}"
        if use_cache:
            cached = self._cache_get("market_data", cache_key)
            if cached:
                return cached

        if not self.api_clients:
            raise InvalidConfigurationError(
                "No market data providers are configured.",
                context=self._context("fetch_data", symbol=symbol),
            )

        attempts: List[ProviderAttempt] = []
        provider_results: Dict[str, List[Dict[str, Any]]] = {}

        if mode == "fallback":
            for provider_name, client in self.api_clients.items():
                provider_name, series, attempt = self._fetch_from_provider(provider_name, client, symbol, lookback)
                attempts.append(attempt)
                if series:
                    provider_results[provider_name] = series
                    break
        else:
            with ThreadPoolExecutor(max_workers=min(len(self.api_clients), self.max_workers)) as executor:
                futures = {
                    executor.submit(self._fetch_from_provider, provider_name, client, symbol, lookback): provider_name
                    for provider_name, client in self.api_clients.items()
                }
                for future in as_completed(futures):
                    provider_name, series, attempt = future.result()
                    attempts.append(attempt)
                    if series:
                        provider_results[provider_name] = series

        if not provider_results:
            raise DataUnavailableError(
                f"No market data retrieved for {symbol}.",
                context=self._context("fetch_data", symbol=symbol, extra={"lookback": lookback}),
                details={"attempts": [attempt.to_dict() for attempt in attempts]},
            )

        if len(provider_results) == 1:
            result = next(iter(provider_results.values()))
        else:
            result = self._blend(list(provider_results.values()))

        if use_cache:
            self._cache_set("market_data", cache_key, result, tags=[symbol.lower(), "market_data"], priority="high")

        self.last_batch_report = {
            "symbol": symbol,
            "lookback": lookback,
            "mode": mode,
            "providers_attempted": [attempt.to_dict() for attempt in attempts],
            "providers_used": list(provider_results),
            "records": len(result),
        }
        return result

    def fetch_data_batch(
        self,
        symbols: Sequence[str],
        lookback: int = 30,
        mode: str = "fuse",
        *,
        raise_on_error: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        normalized_symbols = [self._validate_symbol(symbol) for symbol in symbols]
        results: Dict[str, List[Dict[str, Any]]] = {}
        failures: Dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=min(len(normalized_symbols) or 1, self.max_workers)) as executor:
            futures = {
                executor.submit(self.fetch_data, symbol, lookback, mode): symbol for symbol in normalized_symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    data = future.result()
                    if data:
                        results[symbol] = data
                except BaseException as exc:
                    handled = exc if isinstance(exc, FinancialAgentError) else classify_external_exception(
                        exc,
                        context=self._context("fetch_data_batch", symbol=symbol),
                        message=f"Batch market data fetch failed for {symbol}.",
                    )
                    failures[symbol] = str(handled)
                    log_error(handled, logger_=logger, include_traceback=False)
                    if raise_on_error:
                        raise handled from exc

        self.last_batch_report = {
            "requested": list(normalized_symbols),
            "succeeded": list(results),
            "failed": failures,
            "lookback": lookback,
            "mode": mode,
        }
        return results

    def _blend(self, datasets: Sequence[Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
        by_date: Dict[dt.date, List[Mapping[str, Any]]] = {}
        for dataset in datasets:
            for row in dataset:
                date_value = row["date"]
                if hasattr(date_value, "date"):
                    key = date_value.date()
                else:
                    continue
                by_date.setdefault(key, []).append(row)

        blended: List[Dict[str, Any]] = []
        for day in sorted(by_date):
            rows = by_date[day]
            if not rows:
                continue
            template = rows[0]
            blended.append(
                {
                    "date": template["date"],
                    "open": float(sum(float(row["open"]) for row in rows) / len(rows)),
                    "high": float(sum(float(row["high"]) for row in rows) / len(rows)),
                    "low": float(sum(float(row["low"]) for row in rows) / len(rows)),
                    "close": float(sum(float(row["close"]) for row in rows) / len(rows)),
                    "volume": float(sum(float(row["volume"]) for row in rows) / len(rows)),
                    "symbol": template.get("symbol"),
                    "source": "fused",
                    "providers": sorted({str(row.get("source", "unknown")) for row in rows}),
                }
            )
        return blended

    def get_last_price(self, symbol: str) -> Optional[float]:
        series = self.fetch_data(symbol, lookback=2, mode="fallback")
        if not series:
            return None
        return float(series[-1]["close"])

    def get_returns_matrix(self, symbols: Sequence[str], lookback: int = 252) -> Tuple[List[str], List[List[float]]]:
        batch = self.fetch_data_batch(symbols, lookback=lookback, mode="fallback")
        if not batch:
            return [], []
        date_maps: Dict[str, Dict[dt.date, float]] = {}
        for symbol, series in batch.items():
            closes = {row["date"].date(): float(row["close"]) for row in series if row.get("close")}
            if len(closes) >= 2:
                date_maps[symbol] = closes
        if not date_maps:
            return [], []
        common_dates = None
        for closes in date_maps.values():
            date_set = set(closes)
            common_dates = date_set if common_dates is None else common_dates & date_set
        common_dates = sorted(common_dates or [])
        if len(common_dates) < 3:
            return [], []

        matrix: List[List[float]] = []
        ordered_symbols = sorted(date_maps)
        previous: Optional[Dict[str, float]] = None
        for day in common_dates:
            today_prices = {symbol: date_maps[symbol][day] for symbol in ordered_symbols}
            if previous is not None:
                row: List[float] = []
                for symbol in ordered_symbols:
                    prev_close = previous[symbol]
                    close = today_prices[symbol]
                    if prev_close <= 0:
                        row.append(0.0)
                    else:
                        row.append((close / prev_close) - 1.0)
                matrix.append(row)
            previous = today_prices
        return ordered_symbols, matrix

    def fetch_market_snapshot(self, symbol: str, lookback: int = 60, *, include_sentiment: bool = False) -> Dict[str, Any]:
        series = self.fetch_data(symbol, lookback=lookback, mode="fuse")
        snapshot: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "lookback": lookback,
            "bars": series,
            "last_price": float(series[-1]["close"]) if series else None,
            "provider_report": dict(self.last_batch_report),
            "generated_at": dt.datetime.utcnow().isoformat(),
        }
        if include_sentiment and self.sentiment_scraper is not None:
            try:
                snapshot["sentiment"] = {
                    "score": float(self.sentiment_scraper.compute_average_sentiment(symbol, max_results=8)),
                    "sources": self.sentiment_scraper.get_sentiment_snippets(symbol, max_results=4),
                }
            except Exception as exc:
                logger.warning("Failed to attach sentiment snapshot for %s: %s", symbol, exc)
        return snapshot

    def get_most_active_symbols(self, count: int = 100, *, include_metadata: bool = False) -> List[Any]:
        count = max(1, min(int(count), 1000))
        cache_key = f"most_active:{count}:{include_metadata}"
        cached = self._cache_get("market_lists", cache_key)
        if cached:
            return cached

        symbols: List[Any] = []
        if YahooMarketScraper is not None:
            try:
                with YahooMarketScraper(timeout=self.timeout) as scraper:
                    symbols = scraper.fetch_most_active_stocks(count=count, include_metadata=include_metadata)
            except Exception as exc:
                logger.warning("YahooMarketScraper failed: %s", exc)
        elif scrape_yahoo_most_active_stocks is not None:
            try:
                symbols = scrape_yahoo_most_active_stocks(count=count)
            except Exception as exc:
                logger.warning("Legacy Yahoo scrape failed: %s", exc)

        self._cache_set("market_lists", cache_key, symbols, tags=["most_active", "market_lists"], priority="medium")
        return symbols

    def get_provider_health_report(self) -> Dict[str, Any]:
        report: Dict[str, Any] = {}
        for provider, state in self.provider_stats.items():
            successes = int(state.get("successes", 0))
            failures = int(state.get("failures", 0))
            total = successes + failures
            reliability = successes / total if total else 1.0
            report[provider] = {
                **state,
                "reliability": reliability,
                "total_attempts": total,
            }
        return report


__all__ = [
    "AlphaVantageAPI",
    "APIClientBase",
    "FinnhubAPI",
    "MarketDataHandler",
    "PolygonAPI",
    "PriceBar",
    "ProviderAttempt",
    "YahooFinanceAPI",
]
