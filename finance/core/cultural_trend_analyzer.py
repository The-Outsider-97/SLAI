from __future__ import annotations

import math
import re
import threading
import time
import numpy as np

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Any, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - optional dependency
    cosine_similarity = None

try:
    from scipy.stats import pearsonr
except Exception:  # pragma: no cover - optional dependency
    pearsonr = None

from finance.core.utils.config_loader import load_global_config, get_config_section
from finance.core.utils.financial_errors import (ErrorContext, classify_external_exception,
                                                FinancialAgentError, TrendMonitoringError, log_error,
                                                SentimentError, ValidationError, PartialDataError,
                                                DataUnavailableError, PersistenceError)
from finance.core.utils.public_sentiment_scraper import PublicSentimentScraper
from finance.core.finance_memory import FinanceMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Cultural Trend Analyzer")
printer = PrettyPrinter


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TREND_DECAY = 0.95
DEFAULT_BETA = 0.30
DEFAULT_GAMMA = 0.10
DEFAULT_SENTIMENT_WINDOW_DAYS = 30
DEFAULT_TOP_N = 10
DEFAULT_PATTERN_LOOKBACK_DAYS = 90
DEFAULT_MIN_SIMILARITY = 0.70
DEFAULT_MAX_FEATURES = 500
DEFAULT_CORPUS_WINDOW = 24
DEFAULT_MIN_TERM_SCORE = 0.075
DEFAULT_SNAPSHOT_LIMIT = 250
DEFAULT_MONITOR_INTERVAL_HOURS = 6
DEFAULT_REPORT_TOP_TRENDS = 15
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")
STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "will", "into", "their", "they",
    "about", "after", "before", "would", "could", "should", "there", "which", "been", "were", "has",
    "had", "its", "than", "them", "over", "more", "also", "said", "says", "amid", "under", "into",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TermState:
    score: float
    last_updated: float
    observations: int = 1
    last_exposure: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PropagationState:
    S: float = 0.99
    I: float = 0.01
    R: float = 0.0
    last_update: float = field(default_factory=time.time)
    velocity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "S": float(self.S),
            "I": float(self.I),
            "R": float(self.R),
            "last_update": float(self.last_update),
            "velocity": float(self.velocity),
        }


@dataclass(slots=True)
class TrendSnapshot:
    symbol: str
    timestamp: str
    trends: Dict[str, float]
    epidemic_states: Dict[str, Dict[str, float]]
    sentiment: Dict[str, Any]
    corpus_stats: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class CulturalTrendAnalyzer:
    """
    Production-ready cultural trend analyzer for a financial agent.

    Responsibilities:
    - extract and score thematic trends from public sentiment text
    - model propagation / decay of terms over time
    - persist snapshots and reports into FinanceMemory
    - correlate sentiment with stored market prices when available
    - expose consistent reporting and signal-generation APIs
    """

    def __init__(
        self,
        *,
        sentiment_scraper: Optional[PublicSentimentScraper] = None,
        finance_memory: Optional[FinanceMemory] = None,
        max_features: Optional[int] = None,
        corpus_window: Optional[int] = None,
    ) -> None:
        self.config = load_global_config() or {}
        self.cta_config = get_config_section("cultural_trend_analyzer") or {}

        self.trend_decay_factor = float(self.cta_config.get("trend_decay_factor", DEFAULT_TREND_DECAY))
        self.beta = float(self.cta_config.get("epidemic_beta", DEFAULT_BETA))
        self.gamma = float(self.cta_config.get("epidemic_gamma", DEFAULT_GAMMA))
        self.sentiment_window = int(self.cta_config.get("sentiment_window_days", DEFAULT_SENTIMENT_WINDOW_DAYS))
        self.max_features = int(max_features or self.cta_config.get("max_features", DEFAULT_MAX_FEATURES))
        self.corpus_window = int(corpus_window or self.cta_config.get("corpus_window", DEFAULT_CORPUS_WINDOW))
        self.min_term_score = float(self.cta_config.get("min_term_score", DEFAULT_MIN_TERM_SCORE))
        self.snapshot_limit = int(self.cta_config.get("snapshot_limit", DEFAULT_SNAPSHOT_LIMIT))

        if not (0.0 < self.trend_decay_factor <= 1.0):
            raise ValidationError("trend_decay_factor must be in the interval (0, 1].")
        if self.beta < 0.0 or self.gamma < 0.0:
            raise ValidationError("epidemic_beta and epidemic_gamma must be non-negative.")

        self.sentiment_scraper = sentiment_scraper or PublicSentimentScraper()
        self.finance_memory = finance_memory or FinanceMemory()
        self._owns_scraper = sentiment_scraper is None
        self._owns_memory = finance_memory is None

        self.lock = threading.RLock()
        self.trend_vector: Dict[str, TermState] = {}
        self.sir_states: Dict[str, PropagationState] = {}
        self.last_sentiment: Dict[str, float] = {}
        self.correlation_history: Deque[Dict[str, Any]] = deque(maxlen=500)
        self.trend_history: Deque[Dict[str, Any]] = deque(maxlen=500)
        self.pattern_cache: Dict[Tuple[str, int, float], Tuple[float, List[Dict[str, Any]]]] = {}
        self._local_cache_ttl_seconds = int(self.cta_config.get("cache_ttl_seconds", 900))
        self._corpus_by_symbol: MutableMapping[str, Deque[str]] = defaultdict(
            lambda: deque(maxlen=self.corpus_window)
        )
        self._vectorizers: Dict[str, Any] = {}

        if printer is not None:  # pragma: no cover
            printer.status("INIT", "Cultural Trend Analyzer initialized", "success")
        logger.info(
            "Cultural Trend Analyzer initialized | decay=%.3f beta=%.3f gamma=%.3f sentiment_window=%s",
            self.trend_decay_factor,
            self.beta,
            self.gamma,
            self.sentiment_window,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._owns_scraper and hasattr(self.sentiment_scraper, "close"):
            try:
                self.sentiment_scraper.close()
            except Exception:
                pass

    def __enter__(self) -> "CulturalTrendAnalyzer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _context(self, operation: str, symbol: Optional[str] = None, **metadata: Any) -> ErrorContext:
        return ErrorContext(
            component="cultural_trend_analyzer",
            operation=operation,
            symbol=symbol,
            metadata=metadata or {},
        )

    def _now(self) -> float:
        return time.time()

    def _utcnow(self) -> datetime:
        return datetime.now(timezone.utc)

    def _normalize_symbol(self, symbol: Optional[str]) -> Optional[str]:
        if symbol is None:
            return None
        cleaned = str(symbol).strip().upper()
        return cleaned or None

    # ------------------------------------------------------------------
    # Memory compatibility helpers
    # ------------------------------------------------------------------

    def _memory_get_cache(self, key: str, *, namespace: str = "cultural_trends") -> Any:
        try:
            getter = getattr(self.finance_memory, "get_cache", None)
            if callable(getter):
                return getter(key, namespace=namespace)
        except Exception:
            return None
        return None

    def _memory_set_cache(
        self,
        key: str,
        value: Any,
        *,
        namespace: str = "cultural_trends",
        ttl_seconds: Optional[int] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> None:
        setter = getattr(self.finance_memory, "set_cache", None)
        if callable(setter):
            try:
                setter(
                    key,
                    value,
                    namespace=namespace,
                    ttl_seconds=ttl_seconds or self._local_cache_ttl_seconds,
                    tags=list(tags or []),
                    priority="medium",
                )
            except Exception:
                return

    def _memory_add_entry(
        self,
        *,
        data: Mapping[str, Any],
        data_type: str,
        tags: Sequence[str],
        priority: str = "medium",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        try:
            return self.finance_memory.add_financial_data(
                data=dict(data),
                data_type=data_type,
                tags=list(tags),
                priority=priority,
                metadata=dict(metadata or {}),
            )
        except Exception as exc:
            handled = PersistenceError(
                "Failed to persist trend artifact to finance memory.",
                context=self._context("memory_add", metadata={"data_type": data_type}),
                details={"data_type": data_type, "tags": list(tags)},
                cause=exc,
            )
            log_error(handled, logger_=logger)
            return None

    def _query_memory(self, *, data_type: Optional[str] = None, tags: Optional[Sequence[str]] = None, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            return list(
                self.finance_memory.query(
                    data_type=data_type,
                    tags=list(tags or []),
                    limit=int(limit),
                )
            )
        except TypeError:
            try:
                records = self.finance_memory.get(tag=tags[0]) if tags else self.finance_memory.get()
                if isinstance(records, list):
                    return records[:limit]
            except Exception:
                return []
        except Exception:
            return []
        return []

    # ------------------------------------------------------------------
    # Text processing and vectorization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        return [
            token.lower()
            for token in TOKEN_PATTERN.findall(text or "")
            if token.lower() not in STOPWORDS
        ]

    def _score_terms_fallback(self, texts: Sequence[str]) -> Tuple[Dict[str, float], List[str]]:
        if not texts:
            return {}, []
        current_tokens = self._tokenize(texts[-1])
        if not current_tokens:
            return {}, []

        doc_freq: Dict[str, int] = defaultdict(int)
        tokenized_docs: List[List[str]] = []
        for text in texts:
            tokens = self._tokenize(text)
            tokenized_docs.append(tokens)
            for token in set(tokens):
                doc_freq[token] += 1

        term_counts: Dict[str, int] = defaultdict(int)
        for token in current_tokens:
            term_counts[token] += 1

        n_docs = max(1, len(tokenized_docs))
        current_total = max(1, len(current_tokens))
        scores: Dict[str, float] = {}
        for token, count in term_counts.items():
            tf = count / current_total
            idf = math.log((1 + n_docs) / (1 + doc_freq.get(token, 1))) + 1.0
            scores[token] = tf * idf

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: self.max_features]
        return dict(ranked), [item[0] for item in ranked]

    def _score_terms_tfidf(self, symbol: str, texts: Sequence[str]) -> Tuple[Dict[str, float], List[str]]:
        if not texts:
            return {}, []

        if TfidfVectorizer is None:
            return self._score_terms_fallback(texts)

        try:
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words="english",
                ngram_range=(1, 2),
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b",
            )
            matrix = vectorizer.fit_transform(list(texts))
            if matrix.shape[0] == 0 or matrix.shape[1] == 0:
                return {}, []
            feature_names = list(vectorizer.get_feature_names_out())
            current_row = matrix[-1].toarray().flatten()
            scores = {
                feature_names[idx]: float(score)
                for idx, score in enumerate(current_row)
                if float(score) >= self.min_term_score
            }
            self._vectorizers[symbol] = vectorizer
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            return dict(ranked), [name for name, _ in ranked]
        except Exception as exc:
            logger.warning("TF-IDF scoring failed for symbol=%s: %s. Falling back to token scoring.", symbol, exc)
            return self._score_terms_fallback(texts)

    # ------------------------------------------------------------------
    # Trend state updates
    # ------------------------------------------------------------------

    def update_trends(
        self,
        text_corpus: str,
        query_symbol: Optional[str] = None,
        *,
        source_metadata: Optional[Mapping[str, Any]] = None,
        store_snapshot: bool = True,
        refresh_sentiment: bool = True,
    ) -> Dict[str, Any]:
        symbol = self._normalize_symbol(query_symbol)
        if not isinstance(text_corpus, str):
            raise ValidationError(
                "text_corpus must be a string.",
                context=self._context("update_trends", symbol=symbol),
            )

        cleaned = " ".join(text_corpus.split())
        if not cleaned:
            logger.warning("update_trends received empty text for symbol=%s", symbol)
            snapshot = self.store_trend_snapshot(symbol or "GLOBAL") if symbol and store_snapshot else {}
            return {"symbol": symbol, "updated_terms": 0, "snapshot": snapshot, "warning": "empty_corpus"}

        with self.lock:
            corpus_key = symbol or "GLOBAL"
            self._corpus_by_symbol[corpus_key].append(cleaned)
            corpus_window = list(self._corpus_by_symbol[corpus_key])

            try:
                term_scores, feature_names = self._score_terms_tfidf(corpus_key, corpus_window)
                now = self._now()
                updated_terms = 0
                for term, score in term_scores.items():
                    if score < self.min_term_score:
                        continue
                    prior = self.trend_vector.get(term)
                    if prior is not None:
                        elapsed_days = max(0.0, (now - prior.last_updated) / 86400.0)
                        decayed_prior = prior.score * (self.trend_decay_factor ** elapsed_days)
                        new_score = decayed_prior + float(score)
                        observations = prior.observations + 1
                    else:
                        new_score = float(score)
                        observations = 1

                    self.trend_vector[term] = TermState(
                        score=float(new_score),
                        last_updated=now,
                        observations=observations,
                        last_exposure=float(score),
                    )
                    self.update_epidemic_model(term, float(score))
                    updated_terms += 1

                self.trend_history.append(
                    {
                        "symbol": symbol,
                        "timestamp": self._utcnow().isoformat(),
                        "updated_terms": updated_terms,
                        "feature_count": len(feature_names),
                        "corpus_length": len(cleaned.split()),
                    }
                )
            except FinancialAgentError:
                raise
            except Exception as exc:
                handled = TrendMonitoringError(
                    "Trend update failed.",
                    context=self._context("update_trends", symbol=symbol),
                    details={"source_metadata": dict(source_metadata or {})},
                    cause=exc,
                )
                log_error(handled, logger_=logger)
                raise handled from exc

        snapshot: Dict[str, Any] = {}
        if symbol and refresh_sentiment:
            self.refresh_sentiment(symbol)
        if symbol and store_snapshot:
            snapshot = self.store_trend_snapshot(symbol, source_metadata=source_metadata)

        return {
            "symbol": symbol,
            "updated_terms": updated_terms,
            "feature_count": len(feature_names),
            "snapshot": snapshot,
        }

    def update_epidemic_model(self, term: str, exposure: float) -> Dict[str, float]:
        with self.lock:
            state = self.sir_states.get(term)
            if state is None:
                state = PropagationState()
                self.sir_states[term] = state

            now = self._now()
            delta_days = max(1.0 / 1440.0, (now - state.last_update) / 86400.0)
            state.last_update = now

            S, I, R = state.S, state.I, state.R
            effective_beta = max(0.0, self.beta * (1.0 + max(0.0, exposure)))

            dS = -effective_beta * S * I * delta_days
            dI = (effective_beta * S * I - self.gamma * I) * delta_days
            dR = (self.gamma * I) * delta_days

            next_S = max(0.0, min(1.0, S + dS))
            next_I = max(0.0, min(1.0, I + dI))
            next_R = max(0.0, min(1.0, R + dR))
            total = next_S + next_I + next_R
            if total <= 1e-12:
                next_S, next_I, next_R = 0.99, 0.01, 0.0
                total = 1.0

            state.S = next_S / total
            state.I = next_I / total
            state.R = next_R / total
            state.velocity = float((state.I - I) / max(delta_days, 1e-9))
            return state.to_dict()

    # ------------------------------------------------------------------
    # Snapshot and reporting persistence
    # ------------------------------------------------------------------

    def store_trend_snapshot(
        self,
        symbol: str,
        *,
        source_metadata: Optional[Mapping[str, Any]] = None,
        top_n: int = 20,
    ) -> Dict[str, Any]:
        normalized_symbol = self._normalize_symbol(symbol)
        if not normalized_symbol:
            raise ValidationError("symbol is required to store a trend snapshot.")

        current_trends = self.get_current_trends(top_n=top_n)
        epidemic_states_serializable = {
            term: self.sir_states.get(term, PropagationState()).to_dict()
            for term in current_trends.keys()
        }
        payload = TrendSnapshot(
            symbol=normalized_symbol,
            timestamp=self._utcnow().isoformat(),
            trends=current_trends,
            epidemic_states=epidemic_states_serializable,
            sentiment={
                "score": float(self.last_sentiment.get(normalized_symbol, 0.0)),
                "sources": [
                    source for source, enabled in getattr(self.sentiment_scraper, "sources", {}).items() if enabled
                ],
            },
            corpus_stats={
                "trend_count": len(current_trends),
                "total_terms_tracked": len(self.trend_vector),
            },
            sources=list((source_metadata or {}).get("sources", [])),
        ).to_dict()

        self._memory_add_entry(
            data=payload,
            data_type="trend",
            tags=[f"trend_{normalized_symbol}", "cultural_analysis", normalized_symbol.lower()],
            priority="medium",
            metadata={"symbol": normalized_symbol},
        )
        self._memory_set_cache(
            f"snapshot::{normalized_symbol}",
            payload,
            namespace="cultural_trends",
            ttl_seconds=self._local_cache_ttl_seconds,
            tags=[f"trend_{normalized_symbol}"],
        )
        return payload

    # ------------------------------------------------------------------
    # Sentiment and price integration
    # ------------------------------------------------------------------

    def refresh_sentiment(self, symbol: str) -> float:
        normalized_symbol = self._normalize_symbol(symbol)
        if not normalized_symbol:
            raise ValidationError("symbol is required to refresh sentiment.")

        try:
            sentiment_score = float(self.sentiment_scraper.compute_average_sentiment(normalized_symbol))
            self.last_sentiment[normalized_symbol] = sentiment_score
        except Exception as exc:
            handled = SentimentError(
                "Failed to refresh sentiment for symbol.",
                context=self._context("refresh_sentiment", symbol=normalized_symbol),
                cause=exc,
            )
            log_error(handled, logger_=logger)
            raise handled from exc

        correlation = self._compute_and_store_correlation(normalized_symbol)
        self._memory_set_cache(
            f"sentiment::{normalized_symbol}",
            {"score": sentiment_score, "correlation": correlation, "timestamp": self._utcnow().isoformat()},
            namespace="cultural_trends",
            ttl_seconds=self._local_cache_ttl_seconds,
            tags=[f"trend_{normalized_symbol}", "sentiment"],
        )
        return sentiment_score

    def get_recent_price_data(self, symbol: str, days: int = 30) -> Dict[datetime, float]:
        normalized_symbol = self._normalize_symbol(symbol)
        if not normalized_symbol:
            raise ValidationError("symbol is required to retrieve recent price data.")
        if days <= 0:
            return {}

        cutoff = self._utcnow() - timedelta(days=int(days))
        candidate_types = ("market_ohlcv_daily", "market", "market_data", "ohlcv")
        price_points: Dict[datetime, float] = {}

        for data_type in candidate_types:
            records = self._query_memory(
                data_type=data_type,
                tags=[normalized_symbol.lower()],
                limit=days * 8,
            )
            if not records:
                records = self._query_memory(data_type=data_type, limit=days * 8)

            for record in records:
                data = record.get("data", {}) if isinstance(record, Mapping) else {}
                record_symbol = self._normalize_symbol(
                    data.get("symbol")
                    or record.get("metadata", {}).get("extra", {}).get("symbol")
                    or record.get("metadata", {}).get("symbol")
                )
                if record_symbol != normalized_symbol:
                    continue

                date_value = data.get("date") or data.get("timestamp") or data.get("datetime")
                close_value = data.get("close") or data.get("adj_close") or data.get("price")
                parsed_dt = self._parse_datetime(date_value)
                if parsed_dt is None or parsed_dt < cutoff:
                    continue
                try:
                    close_price = float(close_value)
                except (TypeError, ValueError):
                    continue
                price_points[parsed_dt] = close_price

        return dict(sorted(price_points.items(), key=lambda item: item[0]))

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def _compute_and_store_correlation(self, symbol: str) -> float:
        sentiment_history = []
        try:
            history = self.sentiment_scraper.get_sentiment_history(symbol, days=self.sentiment_window)
            sentiment_history = list(history or [])
        except Exception as exc:
            logger.warning("Sentiment history lookup failed for %s: %s", symbol, exc)

        price_data = self.get_recent_price_data(symbol, days=self.sentiment_window)
        if not sentiment_history or len(price_data) < 3:
            return self._fallback_recent_correlation(symbol)

        sentiment_by_date: Dict[str, float] = {}
        for item in sentiment_history:
            try:
                sentiment_by_date[str(item["date"])] = float(item.get("sentiment", 0.0))
            except Exception:
                continue

        aligned_dates = sorted([
            dt for dt in price_data.keys() if dt.date().isoformat() in sentiment_by_date
        ])
        if len(aligned_dates) < 3:
            return self._fallback_recent_correlation(symbol)

        aligned_sentiments = [sentiment_by_date[dt.date().isoformat()] for dt in aligned_dates]
        aligned_prices = [price_data[dt] for dt in aligned_dates]
        if len(aligned_sentiments) < 3 or len(aligned_prices) < 3:
            return self._fallback_recent_correlation(symbol)

        price_changes = np.diff(np.array(aligned_prices, dtype=float))
        sentiment_changes = np.diff(np.array(aligned_sentiments, dtype=float))
        if len(price_changes) < 2 or len(sentiment_changes) < 2:
            return self._fallback_recent_correlation(symbol)

        try:
            if pearsonr is not None:
                correlation, _ = pearsonr(sentiment_changes, price_changes)
                correlation_value = float(0.0 if np.isnan(correlation) else correlation)
            else:
                if np.std(sentiment_changes) < 1e-12 or np.std(price_changes) < 1e-12:
                    correlation_value = 0.0
                else:
                    correlation_value = float(np.corrcoef(sentiment_changes, price_changes)[0, 1])
        except Exception:
            correlation_value = 0.0

        record = {
            "symbol": symbol,
            "timestamp": self._utcnow().isoformat(),
            "correlation": round(correlation_value, 6),
            "samples": len(aligned_dates),
            "sentiment_mean": round(float(mean(aligned_sentiments)), 6),
            "price_change": round(float(aligned_prices[-1] - aligned_prices[0]), 6),
            "sources": [
                source for source, enabled in getattr(self.sentiment_scraper, "sources", {}).items() if enabled
            ],
        }
        self.correlation_history.append(record)
        self._memory_add_entry(
            data=record,
            data_type="trend_correlation",
            tags=[f"trend_{symbol}", "cultural_analysis", "correlation"],
            priority="medium",
            metadata={"symbol": symbol},
        )
        return float(record["correlation"])

    def _fallback_recent_correlation(self, symbol: str) -> float:
        cutoff = self._utcnow() - timedelta(days=self.sentiment_window)
        recent = [
            item["correlation"]
            for item in self.correlation_history
            if item.get("symbol") == symbol and self._parse_datetime(item.get("timestamp")) and self._parse_datetime(item.get("timestamp")) >= cutoff
        ]
        return float(round(mean(recent), 6)) if recent else 0.0

    # ------------------------------------------------------------------
    # Public query APIs
    # ------------------------------------------------------------------

    def get_current_trends(self, top_n: int = DEFAULT_TOP_N) -> Dict[str, float]:
        with self.lock:
            if not self.trend_vector:
                return {}
            now = self._now()
            decayed_scores: Dict[str, float] = {}
            for term, state in self.trend_vector.items():
                elapsed_days = max(0.0, (now - state.last_updated) / 86400.0)
                decayed_score = float(state.score * (self.trend_decay_factor ** elapsed_days))
                if decayed_score >= 0.01:
                    decayed_scores[term] = decayed_score

        ranked = sorted(decayed_scores.items(), key=lambda item: item[1], reverse=True)
        return {term: float(score) for term, score in ranked[: max(1, int(top_n))]}

    def find_similar_patterns(
        self,
        symbol: str,
        lookback_days: int = DEFAULT_PATTERN_LOOKBACK_DAYS,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> List[Dict[str, Any]]:
        normalized_symbol = self._normalize_symbol(symbol)
        if not normalized_symbol:
            raise ValidationError("symbol is required to find similar patterns.")

        cache_key = (normalized_symbol, int(lookback_days), int(min_similarity * 1000))
        cached = self.pattern_cache.get(cache_key)
        if cached and (self._now() - cached[0]) < self._local_cache_ttl_seconds:
            return list(cached[1])

        current_trends = self.get_current_trends(top_n=50)
        if not current_trends:
            return []

        records = self._query_memory(
            data_type="trend",
            tags=[f"trend_{normalized_symbol}"],
            limit=self.snapshot_limit,
        )
        cutoff = self._utcnow() - timedelta(days=int(max(1, lookback_days)))
        historical_snapshots: List[Mapping[str, Any]] = []
        for entry in records:
            data = entry.get("data", {})
            timestamp = self._parse_datetime(data.get("timestamp"))
            if timestamp is None or timestamp < cutoff:
                continue
            if data.get("symbol") != normalized_symbol:
                continue
            historical_snapshots.append(data)

        if not historical_snapshots:
            self.pattern_cache[cache_key] = (self._now(), [])
            return []

        all_terms = sorted(set(current_trends.keys()).union(*[set((snap.get("trends") or {}).keys()) for snap in historical_snapshots]))
        if not all_terms:
            return []

        current_vector = np.array([float(current_trends.get(term, 0.0)) for term in all_terms], dtype=float).reshape(1, -1)
        current_norm = float(np.linalg.norm(current_vector))
        if current_norm <= 1e-12:
            return []

        similar_patterns: List[Dict[str, Any]] = []
        for snapshot in historical_snapshots:
            hist_trends = snapshot.get("trends") or {}
            hist_vector = np.array([float(hist_trends.get(term, 0.0)) for term in all_terms], dtype=float).reshape(1, -1)
            hist_norm = float(np.linalg.norm(hist_vector))
            if hist_norm <= 1e-12:
                continue
            if cosine_similarity is not None:
                similarity = float(cosine_similarity(current_vector, hist_vector)[0][0])
            else:
                similarity = float(np.dot(current_vector[0], hist_vector[0]) / (current_norm * hist_norm))
            if similarity < float(min_similarity):
                continue
            ts = self._parse_datetime(snapshot.get("timestamp"))
            similar_patterns.append(
                {
                    "timestamp": snapshot.get("timestamp"),
                    "similarity": round(similarity, 6),
                    "trends": hist_trends,
                    "sentiment": snapshot.get("sentiment", {}),
                    "days_ago": (self._utcnow() - ts).days if ts else -1,
                }
            )

        similar_patterns.sort(key=lambda item: item["similarity"], reverse=True)
        self.pattern_cache[cache_key] = (self._now(), list(similar_patterns))
        self._memory_set_cache(
            f"patterns::{normalized_symbol}::{lookback_days}::{min_similarity}",
            similar_patterns,
            namespace="cultural_trends",
            ttl_seconds=self._local_cache_ttl_seconds,
            tags=[f"trend_{normalized_symbol}", "pattern_match"],
        )
        return similar_patterns

    def predict_trend_impact(self, term: str, simulation_days: int = 30) -> Dict[str, Any]:
        prediction = {
            "term": term,
            "current_impact": 0.0,
            "predicted_peak": 0.0,
            "peak_time_days": None,
            "recovery_time_days": None,
            "confidence": 0.0,
            "simulation": [],
            "error": None,
        }

        with self.lock:
            state = self.sir_states.get(term)
            term_state = self.trend_vector.get(term)
        if state is None or term_state is None:
            prediction["error"] = "Term not found in trend state"
            return prediction

        try:
            S, I, R = float(state.S), float(state.I), float(state.R)
            current_impact = float(I)
            peak_impact = current_impact
            peak_day = 0
            recovery_day = None
            simulation: List[Dict[str, Any]] = []

            variability_proxy = 1.0 / max(1.0, float(term_state.observations))
            trend_strength = min(1.0, float(term_state.score) / max(1e-9, self.get_current_trends(top_n=1).get(term, term_state.score)))

            for day in range(max(1, int(simulation_days))):
                dS = -self.beta * S * I
                dI = self.beta * S * I - self.gamma * I
                dR = self.gamma * I

                S = max(0.0, min(1.0, S + dS))
                I = max(0.0, min(1.0, I + dI))
                R = max(0.0, min(1.0, R + dR))

                if I > peak_impact:
                    peak_impact = I
                    peak_day = day
                if recovery_day is None and day > peak_day and I < 0.05:
                    recovery_day = day

                simulation.append(
                    {
                        "day": day,
                        "susceptible": round(float(S), 6),
                        "infected": round(float(I), 6),
                        "recovered": round(float(R), 6),
                    }
                )
                if I < 0.001 and R > 0.95:
                    break

            confidence = max(0.1, min(0.95, 1.0 - variability_proxy * 0.6 + trend_strength * 0.2))
            prediction.update(
                {
                    "current_impact": round(current_impact, 6),
                    "predicted_peak": round(float(peak_impact), 6),
                    "peak_time_days": int(peak_day),
                    "recovery_time_days": int(recovery_day if recovery_day is not None else simulation_days),
                    "confidence": round(float(confidence), 6),
                    "simulation": simulation,
                }
            )
            return prediction
        except Exception as exc:
            logger.warning("Trend prediction failed for term=%s: %s", term, exc)
            prediction["error"] = str(exc)
            return prediction

    def get_trend_propagation(self, term: str) -> Dict[str, float]:
        with self.lock:
            state = self.sir_states.get(term)
        if state is None:
            return {
                "term": term,
                "susceptible": 0.99,
                "infected": 0.01,
                "recovered": 0.0,
                "virality": 0.0,
                "velocity": 0.0,
            }
        return {
            "term": term,
            "susceptible": round(float(state.S), 6),
            "infected": round(float(state.I), 6),
            "recovered": round(float(state.R), 6),
            "virality": round(float(self.beta / self.gamma), 6) if self.gamma > 1e-9 else 0.0,
            "velocity": round(float(state.velocity), 6),
        }

    def get_latest_sentiment(self, symbol: str) -> float:
        normalized_symbol = self._normalize_symbol(symbol)
        if not normalized_symbol:
            return 0.0
        if normalized_symbol in self.last_sentiment:
            return float(self.last_sentiment[normalized_symbol])
        cached = self._memory_get_cache(f"sentiment::{normalized_symbol}", namespace="cultural_trends")
        if isinstance(cached, Mapping):
            try:
                return float(cached.get("score", 0.0))
            except Exception:
                return 0.0
        return 0.0

    def get_recent_correlation(self, symbol: str, window_days: Optional[int] = None) -> float:
        normalized_symbol = self._normalize_symbol(symbol)
        if not normalized_symbol:
            return 0.0
        if window_days is None or window_days <= 0:
            window_days = self.sentiment_window

        cutoff = self._utcnow() - timedelta(days=int(window_days))
        recent_correlations = []
        for item in self.correlation_history:
            timestamp = self._parse_datetime(item.get("timestamp"))
            if item.get("symbol") == normalized_symbol and timestamp and timestamp >= cutoff:
                recent_correlations.append(float(item.get("correlation", 0.0)))
        if recent_correlations:
            return float(round(mean(recent_correlations), 6))

        records = self._query_memory(
            data_type="trend_correlation",
            tags=[f"trend_{normalized_symbol}"],
            limit=window_days * 4,
        )
        values = []
        for record in records:
            data = record.get("data", {})
            timestamp = self._parse_datetime(data.get("timestamp"))
            if timestamp and timestamp >= cutoff:
                try:
                    values.append(float(data.get("correlation", 0.0)))
                except Exception:
                    continue
        return float(round(mean(values), 6)) if values else 0.0

    def get_insider_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Kept for backward compatibility. These are synthesized public-trend signals,
        not actual insider-trading data.
        """
        try:
            report = self.generate_trend_report(symbol)
            signals: List[Dict[str, Any]] = []
            for term, score in list((report.get("top_trends") or {}).items())[:3]:
                prediction = self.predict_trend_impact(term)
                signals.append(
                    {
                        "source_type": "Cultural Trend",
                        "summary": (
                            f"'{term}' is trending (score={score:.3f}) with predicted peak impact "
                            f"{prediction.get('predicted_peak', 0.0):.3f}."
                        ),
                        "potential_impact_string": (
                            "High" if prediction.get("predicted_peak", 0.0) > 0.60
                            else "Moderate" if prediction.get("predicted_peak", 0.0) > 0.30
                            else "Low"
                        ),
                        "confidence_score": round(float(prediction.get("confidence", 0.0)), 4),
                    }
                )
            return signals
        except Exception as exc:
            logger.warning("Failed to generate public trend signals for %s: %s", symbol, exc)
            return []

    def generate_trend_report(self, symbol: str) -> Dict[str, Any]:
        normalized_symbol = self._normalize_symbol(symbol)
        if not normalized_symbol:
            raise ValidationError("symbol is required to generate a trend report.")

        current_trends = self.get_current_trends(top_n=DEFAULT_REPORT_TOP_TRENDS)
        similar_patterns = self.find_similar_patterns(normalized_symbol) if current_trends else []
        sentiment = self.get_latest_sentiment(normalized_symbol)
        correlation = self.get_recent_correlation(normalized_symbol)

        trend_predictions: Dict[str, Dict[str, Any]] = {}
        for idx, term in enumerate(current_trends.keys()):
            if idx >= 5:
                break
            trend_predictions[term] = self.predict_trend_impact(term)

        scores = list(current_trends.values())
        trend_strength_index = float(sum(scores)) if scores else 0.0
        trend_concentration = float(max(scores) / sum(scores)) if len(scores) > 1 and sum(scores) > 1e-12 else (1.0 if scores else 0.0)
        volatility = float(pstdev(scores)) if len(scores) > 1 else 0.0

        report: Dict[str, Any] = {
            "symbol": normalized_symbol,
            "timestamp": self._utcnow().isoformat(),
            "top_trends": current_trends,
            "sentiment": round(float(sentiment), 6),
            "sentiment_correlation": round(float(correlation), 6),
            "similar_patterns": similar_patterns[:3],
            "trend_predictions": trend_predictions,
            "trend_count": len(current_trends),
            "pattern_match_count": len(similar_patterns),
            "trend_strength_index": round(trend_strength_index, 6),
            "trend_concentration": round(trend_concentration, 6),
            "trend_volatility": round(volatility, 6),
            "sentiment_sources": [
                source for source, enabled in getattr(self.sentiment_scraper, "sources", {}).items() if enabled
            ],
        }
        if not current_trends:
            report["warning"] = "No active trends detected for symbol"
        return report

    def monitor_trends(
        self,
        symbol: str,
        interval_hours: int = DEFAULT_MONITOR_INTERVAL_HOURS,
        *,
        max_results: int = 10,
        store_report: bool = True,
    ) -> Dict[str, Any]:
        normalized_symbol = self._normalize_symbol(symbol)
        if not normalized_symbol:
            raise ValidationError("symbol is required to monitor trends.")
        if interval_hours <= 0:
            raise ValidationError("interval_hours must be positive.")

        try:
            snippets = self.sentiment_scraper.get_sentiment_snippets(normalized_symbol, max_results=max_results)
            aggregated_text = self._aggregate_text(snippets)
            source_metadata = {
                "sources": list(snippets.keys()),
                "max_results": max_results,
                "interval_hours": interval_hours,
            }

            if not aggregated_text:
                logger.warning("No text content found for %s. Proceeding with sentiment refresh/report only.", normalized_symbol)
                try:
                    self.refresh_sentiment(normalized_symbol)
                except Exception:
                    pass
            else:
                self.update_trends(
                    aggregated_text,
                    normalized_symbol,
                    source_metadata=source_metadata,
                    store_snapshot=True,
                    refresh_sentiment=True,
                )

            report = self.generate_trend_report(normalized_symbol)
            report["monitor_interval_hours"] = int(interval_hours)
            report["source_count"] = len(snippets)

            if store_report:
                self._memory_add_entry(
                    data=report,
                    data_type="trend_report",
                    tags=[f"trend_report_{normalized_symbol}", "cultural_analysis", normalized_symbol.lower()],
                    priority="high",
                    metadata={"symbol": normalized_symbol},
                )
            return report
        except FinancialAgentError:
            raise
        except Exception as exc:
            handled = TrendMonitoringError(
                "Trend monitoring failed.",
                context=self._context("monitor_trends", symbol=normalized_symbol),
                cause=exc,
            )
            log_error(handled, logger_=logger)
            minimal_report = {
                "symbol": normalized_symbol,
                "timestamp": self._utcnow().isoformat(),
                "error": str(handled),
            }
            if store_report:
                self._memory_add_entry(
                    data=minimal_report,
                    data_type="trend_report",
                    tags=[f"trend_report_{normalized_symbol}", "cultural_analysis", normalized_symbol.lower()],
                    priority="high",
                    metadata={"symbol": normalized_symbol},
                )
            return minimal_report

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate_text(self, sentiment_data: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
        if not isinstance(sentiment_data, Mapping):
            return ""

        text_parts: List[str] = []
        seen: set[str] = set()
        for items in sentiment_data.values():
            if not isinstance(items, Sequence):
                continue
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                candidate_parts = []
                for field in ("title", "text", "content", "snippet", "description", "summary", "selftext"):
                    value = item.get(field)
                    if isinstance(value, str) and value.strip():
                        candidate_parts.append(value.strip())
                if not candidate_parts:
                    continue
                merged = " ".join(candidate_parts)
                if merged not in seen:
                    seen.add(merged)
                    text_parts.append(merged)
        return " ".join(text_parts).strip()


if __name__ == "__main__":  # pragma: no cover
    analyzer = CulturalTrendAnalyzer()
    try:
        symbol = "AAPL"
        report = analyzer.monitor_trends(symbol)
        print(report)
    finally:
        analyzer.close()