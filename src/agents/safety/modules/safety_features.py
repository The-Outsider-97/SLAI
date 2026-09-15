"""Safety feature extraction primitives for the Safety Agent subsystem.

The module converts email and URL evidence into deterministic, bounded security
features.  It intentionally remains a *feature* layer: policy enforcement,
incident response, model training, secure-memory persistence, and orchestration
belong to the surrounding Safety Agent components.

Compatibility contract
----------------------
AdaptiveSecurity currently consumes the historical 11-value email vector and
8-value URL vector.  Those legacy vectors are therefore preserved exactly in
shape and semantics.  New normalized/versioned features are exposed through the
named feature maps and configured vectors so models can migrate explicitly
instead of silently changing their input distribution.
"""

from __future__ import annotations

import ipaddress
import math
import re
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union
from urllib.parse import parse_qsl, urlparse

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.security_error import *
from ..utils.safety_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Safety Features")
printer = PrettyPrinter()

MODULE_VERSION = "2.3.0"
FEATURE_SCHEMA_VERSION = "safety_features.v3"
LEGACY_EMAIL_SCHEMA_VERSION = "safety_features.email.legacy.v1"
LEGACY_URL_SCHEMA_VERSION = "safety_features.url.legacy.v1"


@dataclass(frozen=True)
class FeatureExtractionResult:
    """Structured, audit-safe result of a feature extraction operation."""

    source_type: str
    feature_map: Dict[str, float]
    feature_vector: List[float]
    risk_score: float
    risk_level: str
    decision: str
    indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    schema_version: str = FEATURE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp or utc_iso()
        data["metadata"] = sanitize_for_logging(data.get("metadata", {}))
        return data


@dataclass(frozen=True)
class DomainReputationRecord:
    """Cached domain reputation record.  Score 1.0 means most trusted."""

    domain_fingerprint: str
    score: float
    source: str
    timestamp: float
    indicators: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_fingerprint": self.domain_fingerprint,
            "score": clamp_score(self.score, default=0.5),
            "source": self.source,
            "timestamp": float(self.timestamp),
            "indicators": list(self.indicators),
        }


class SafetyFeatures:
    """Config-driven phishing/security feature extraction for email and URLs.

    Class-level domain resources are intentionally shared because every Safety
    Agent instance uses the same immutable configuration.  All shared mutation
    is protected by an RLock to make concurrent runtime use deterministic.
    """

    _state_lock = threading.RLock()
    _cached_config: Optional[Dict[str, Any]] = None
    _cached_config_fingerprint: Optional[str] = None
    _config_checked_at: float = 0.0
    _trusted_domains: Optional[Set[str]] = None
    _malicious_domains: Optional[Set[str]] = None
    _disposable_domains: Optional[Set[str]] = None
    _domain_reputation_cache: Dict[str, DomainReputationRecord] = {}
    _domain_lists_loaded = False

    _EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    _URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>'\"()]+")
    _IP_LITERAL_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\[[0-9a-fA-F:]+\]")
    _WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)

    def __init__(self) -> None:
        self.config = load_global_config()
        self.sf_config = dict(get_config_section("safety_features") or {})
        with self._state_lock:
            self.__class__._cached_config = dict(self.sf_config)
            self.__class__._cached_config_fingerprint = fingerprint(self.sf_config)
            self.__class__._config_checked_at = time.monotonic()
        if self.sf_config.get("strict_config_validation", True):
            self.validate_configuration()
        self._ensure_domain_lists_loaded()
        logger.info(
            "Safety Features initialized: %s",
            safe_log_payload(
                "safety_features_initialized",
                {
                    "module_version": MODULE_VERSION,
                    "schema_version": self.sf_config.get("schema_version", FEATURE_SCHEMA_VERSION),
                    "configured": bool(self.sf_config),
                    "trusted_domain_count": len(self._trusted_domains or set()),
                    "malicious_domain_count": len(self._malicious_domains or set()),
                },
            ),
        )

    # ------------------------------------------------------------------
    # Configuration / shared-state lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def validate_configuration(cls) -> None:
        cfg = dict(get_config_section("safety_features") or {})
        required_keys = [
            "basic_keywords",
            "urgent_keywords",
            "trusted_domains",
            "malicious_domains",
            "sender_reputation_weights",
            "email_risk_weights",
            "url_risk_weights",
            "feature_order",
        ]
        require_keys(cfg, required_keys, context="safety_features")
        for key in ("sender_reputation_weights", "email_risk_weights", "url_risk_weights"):
            if not isinstance(cfg.get(key), Mapping):
                raise ConfigurationTamperingError(
                    config_file_path=str(load_global_config().get("__config_path__", "secure_config.yaml")),
                    suspicious_change=f"safety_features.{key} must be a mapping",
                    component="safety_features",
                )
        feature_order = cfg.get("feature_order", {})
        if not isinstance(feature_order, Mapping):
            raise ConfigurationTamperingError(
                config_file_path=str(load_global_config().get("__config_path__", "secure_config.yaml")),
                suspicious_change="safety_features.feature_order must be a mapping",
                component="safety_features",
            )

    @classmethod
    def _cfg(cls) -> Dict[str, Any]:
        """Return a bounded-lifetime configuration snapshot.

        The central config loader already performs thread-safe mtime/TTL caching.
        This second, very small cache avoids repeated section copies in the hot
        feature path without disabling runtime configuration refresh forever.
        """

        with cls._state_lock:
            now = time.monotonic()
            refresh_seconds = 60.0
            if cls._cached_config is not None:
                refresh_seconds = coerce_float(
                    cls._cached_config.get("runtime_config_refresh_seconds"),
                    60.0,
                    minimum=1.0,
                )
            due = cls._cached_config is None or (now - cls._config_checked_at) >= refresh_seconds
            if due:
                fresh = dict(get_config_section("safety_features") or {})
                fresh_fingerprint = fingerprint(fresh)
                changed = (
                    cls._cached_config_fingerprint is not None
                    and fresh_fingerprint != cls._cached_config_fingerprint
                )
                cls._cached_config = fresh
                cls._cached_config_fingerprint = fresh_fingerprint
                cls._config_checked_at = now
                if changed:
                    cls._domain_lists_loaded = False
                    cls._trusted_domains = None
                    cls._malicious_domains = None
                    cls._disposable_domains = None
                    cls._domain_reputation_cache.clear()
            if cls._cached_config is None:
                cls._cached_config = dict(get_config_section("safety_features") or {})
                cls._cached_config_fingerprint = fingerprint(cls._cached_config)
                cls._config_checked_at = time.monotonic()
            return dict(cls._cached_config)

    @classmethod
    def refresh_configuration(cls) -> Dict[str, Any]:
        """Refresh feature configuration and dependent caches atomically."""

        with cls._state_lock:
            cls._cached_config = dict(get_config_section("safety_features") or {})
            cls._cached_config_fingerprint = fingerprint(cls._cached_config)
            cls._config_checked_at = time.monotonic()
            cls._domain_lists_loaded = False
            cls._trusted_domains = None
            cls._malicious_domains = None
            cls._disposable_domains = None
            cls._domain_reputation_cache.clear()
        cls.validate_configuration()
        cls._ensure_domain_lists_loaded()
        return dict(cls._cfg())

    @staticmethod
    def _list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Iterable):
            return [str(item) for item in value if str(item).strip()]
        return [str(value)]

    @staticmethod
    def _mapping(value: Any) -> Dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    @classmethod
    def _set(cls, value: Any) -> Set[str]:
        result: Set[str] = set()
        for item in cls._list(value):
            domain = cls._normalize_domain(item)
            if domain:
                result.add(domain)
        return result

    # ------------------------------------------------------------------
    # Common normalization helpers
    # ------------------------------------------------------------------

    @classmethod
    def _tokenize(cls, text: Any) -> List[str]:
        normalized = normalize_text(
            text,
            max_length=coerce_int(cls._cfg().get("max_text_length"), 4096),
            lowercase=True,
        )
        return cls._WORD_RE.findall(normalized)

    @classmethod
    def _phrase_score(cls, text: Any, phrases: Sequence[str], *, cap: Optional[float] = None) -> float:
        normalized = normalize_text(
            text,
            max_length=coerce_int(cls._cfg().get("max_text_length"), 4096),
            lowercase=True,
        )
        if not normalized or not phrases:
            return 0.0
        hits = sum(
            1.0
            for phrase in phrases
            if (candidate := normalize_text(phrase, lowercase=True)) and candidate in normalized
        )
        divisor = cap if cap is not None and cap > 0 else max(float(len(phrases)), 1.0)
        return clamp_score(hits / divisor)

    @classmethod
    def _keyword_score(cls, text: Any, keywords: Sequence[str], *, cap: Optional[float] = None) -> float:
        tokens = set(cls._tokenize(text))
        normalized_keywords = {
            normalize_text(keyword, lowercase=True)
            for keyword in keywords
            if normalize_text(keyword, lowercase=True)
        }
        if not tokens or not normalized_keywords:
            return 0.0
        hits = len(tokens & normalized_keywords)
        divisor = cap if cap is not None and cap > 0 else max(float(len(normalized_keywords)), 1.0)
        return clamp_score(hits / divisor)

    @staticmethod
    def _normalize_domain(domain: Any) -> str:
        text = normalize_text(domain, max_length=253, lowercase=True).strip(".[] ")
        if not text:
            return ""
        return text[4:] if text.startswith("www.") else text

    @classmethod
    def _sender_address(cls, sender: Any) -> str:
        _, address = parseaddr("" if sender is None else str(sender))
        if address:
            return normalize_text(address, max_length=320, lowercase=True)
        text = normalize_text(sender, max_length=320, lowercase=True)
        match = cls._EMAIL_RE.search(text)
        return match.group(0).lower() if match else text

    @classmethod
    def _sender_domain(cls, sender: Any) -> str:
        address = cls._sender_address(sender)
        if "@" not in address:
            return ""
        return cls._normalize_domain(address.rsplit("@", 1)[-1])

    @classmethod
    def _safe_url_bundle(cls, raw_url: Any) -> Optional[SanitizedURL]:
        text = normalize_text(
            raw_url,
            max_length=coerce_int(cls._cfg().get("max_url_length"), 2048),
            preserve_newlines=False,
        )
        if not text:
            return None
        try:
            return sanitize_url(text)
        except (SecurityError, ValueError, TypeError):
            # Invalid/disallowed input is represented by explicit risk features;
            # parsing helpers must not accidentally reinterpret it as safe.
            return None
        except Exception as exc:
            raise wrap_security_exception(
                exc,
                operation="sanitize_url_for_features",
                component="safety_features",
                context={"url_fingerprint": fingerprint(text)},
                error_type=SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                severity=SecuritySeverity.MEDIUM,
            ) from exc

    @classmethod
    def _safe_domain_from_url(cls, raw_url: Any) -> str:
        bundle = cls._safe_url_bundle(raw_url)
        if bundle:
            return cls._normalize_domain(bundle.hostname)
        text = normalize_text(
            raw_url,
            max_length=coerce_int(cls._cfg().get("max_url_length"), 2048),
            lowercase=True,
        )
        if not text:
            return ""
        host = re.sub(r"^[a-z][a-z0-9+.-]*://", "", text).split("/", 1)[0].split("?", 1)[0]
        host = host.rsplit("@", 1)[-1].split(":", 1)[0]
        return cls._normalize_domain(host)

    @classmethod
    def _extract_links_from_text(cls, text: Any) -> List[str]:
        cfg = cls._cfg()
        max_links = coerce_int(cfg.get("max_links_per_email"), 100, minimum=1)
        normalized = normalize_text(text, max_length=coerce_int(cfg.get("max_text_length"), 4096))
        return cls._URL_RE.findall(normalized)[:max_links]

    @classmethod
    def _coerce_links(cls, links: Any) -> List[str]:
        cfg = cls._cfg()
        max_links = coerce_int(cfg.get("max_links_per_email"), 100, minimum=1)
        if links is None:
            return []
        if isinstance(links, str):
            candidates = cls._extract_links_from_text(links)
        elif isinstance(links, Iterable):
            candidates = [str(link) for link in links]
        else:
            candidates = [str(links)]
        max_url_length = coerce_int(cfg.get("max_url_length"), 2048)
        return [
            normalize_text(link, max_length=max_url_length)
            for link in candidates
            if str(link).strip()
        ][:max_links]

    @classmethod
    def _coerce_attachments(cls, email: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        raw = email.get("attachments", [])
        maximum = coerce_int(cls._cfg().get("max_attachments_per_email"), 50, minimum=1)
        if not raw:
            return []
        if isinstance(raw, Mapping):
            return [raw]
        if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
            return [
                item if isinstance(item, Mapping) else {"filename": str(item)}
                for item in list(raw)[:maximum]
            ]
        return [{"filename": str(raw)}]

    # ------------------------------------------------------------------
    # Domain-list / reputation resources
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_domain_lists_loaded(cls) -> None:
        with cls._state_lock:
            if cls._domain_lists_loaded:
                return
            cfg = cls._cfg()
            trusted = cls._set(cfg.get("trusted_domains"))
            malicious = cls._set(cfg.get("malicious_domains"))
            disposable = cls._set(cfg.get("disposable_email_domains"))

            for path in cls._list(cfg.get("trusted_domains_paths")):
                trusted.update(cls._load_domain_file(path, list_name="trusted_domains"))
            for path in cls._list(cfg.get("malicious_domains_paths")):
                malicious.update(cls._load_domain_file(path, list_name="malicious_domains"))
            for path in cls._list(cfg.get("disposable_domains_paths")):
                disposable.update(cls._load_domain_file(path, list_name="disposable_email_domains"))

            trusted -= malicious
            if coerce_bool(cfg.get("disposable_domains_override_trust"), True):
                trusted -= disposable

            cls._trusted_domains = trusted
            cls._malicious_domains = malicious
            cls._disposable_domains = disposable
            cls._domain_lists_loaded = True

    @classmethod
    def _load_domain_lists(cls) -> None:
        """Backward-compatible alias; now idempotent rather than reloading each call."""
        cls._ensure_domain_lists_loaded()

    @classmethod
    def _load_domain_file(cls, path: str, *, list_name: str) -> Set[str]:
        if not path:
            return set()
        text = load_text_file(
            path,
            max_bytes=coerce_int(cls._cfg().get("domain_list_max_bytes"), 1_048_576, minimum=1024),
        )
        domains: Set[str] = set()
        for line in text.splitlines():
            clean = line.strip().split("#", 1)[0].strip()
            domain = cls._normalize_domain(clean)
            if domain:
                domains.add(domain)
        logger.info(
            "Loaded domain list: %s",
            safe_log_payload(
                list_name,
                {"count": len(domains), "path_fingerprint": fingerprint(str(Path(path)))},
            ),
        )
        return domains

    @classmethod
    def _get_domain_reputation(cls, domain: str) -> float:
        cls._ensure_domain_lists_loaded()
        cfg = cls._cfg()
        ttl = coerce_float(cfg.get("domain_cache_ttl_seconds"), 86400.0, minimum=1.0)
        domain = cls._normalize_domain(domain)
        if not domain:
            return clamp_score(cfg.get("unknown_domain_reputation"), default=0.5)

        now = datetime.now(timezone.utc).timestamp()
        with cls._state_lock:
            cached = cls._domain_reputation_cache.get(domain)
            if cached and now - cached.timestamp < ttl:
                return clamp_score(cached.score, default=0.5)

        overrides = {
            cls._normalize_domain(k): clamp_score(v, default=0.5)
            for k, v in cls._mapping(cfg.get("domain_reputation_overrides")).items()
        }
        if domain in overrides:
            score, source = overrides[domain], "configured_override"
        elif cls._malicious_domains and domain in cls._malicious_domains:
            score, source = 0.0, "malicious_domain_list"
        elif cls._trusted_domains and domain in cls._trusted_domains:
            score, source = 1.0, "trusted_domain_list"
        elif cls._disposable_domains and domain in cls._disposable_domains:
            score, source = clamp_score(cfg.get("disposable_domain_reputation"), default=0.25), "disposable_domain_list"
        else:
            score, source = clamp_score(cfg.get("unknown_domain_reputation"), default=0.5), "unknown_default"

        record = DomainReputationRecord(
            domain_fingerprint=fingerprint(domain),
            score=score,
            source=source,
            timestamp=now,
        )
        with cls._state_lock:
            cls._domain_reputation_cache[domain] = record
        return score

    # ------------------------------------------------------------------
    # Email features (legacy methods retained for AdaptiveSecurity)
    # ------------------------------------------------------------------

    @classmethod
    def _contains_suspicious_keywords(cls, text: str) -> float:
        keywords = cls._list(cls._cfg().get("basic_keywords"))
        cap = coerce_float(cls._cfg().get("keyword_hit_cap"), 5.0, minimum=1.0)
        return cls._keyword_score(text, keywords, cap=cap)

    @classmethod
    def _contains_urgent_language(cls, email_body: str) -> float:
        cfg = cls._cfg()
        cap = coerce_float(cfg.get("urgent_hit_cap"), 4.0, minimum=1.0)
        return combine_risk_scores(
            cls._keyword_score(email_body, cls._list(cfg.get("urgent_keywords")), cap=cap),
            cls._phrase_score(email_body, cls._list(cfg.get("urgent_phrases")), cap=cap),
            method="weighted_high",
        )

    @classmethod
    def _contains_attachment(cls, email: Dict) -> float:
        return 1.0 if isinstance(email, Mapping) and cls._coerce_attachments(email) else 0.0

    @classmethod
    def _attachment_risk_score(cls, email: Mapping[str, Any]) -> float:
        attachments = cls._coerce_attachments(email)
        if not attachments:
            return 0.0
        cfg = cls._cfg()
        dangerous = {ext.lower().lstrip(".") for ext in cls._list(cfg.get("dangerous_attachment_extensions"))}
        suspicious = {ext.lower().lstrip(".") for ext in cls._list(cfg.get("suspicious_attachment_extensions"))}
        large_bytes = coerce_int(cfg.get("large_attachment_bytes"), 10_485_760, minimum=1)
        scores: List[float] = []
        for attachment in attachments:
            filename = normalize_text(attachment.get("filename", ""), max_length=256, lowercase=True)
            ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
            size = coerce_float(attachment.get("size_bytes", attachment.get("size", 0.0)), 0.0, minimum=0.0)
            macro = coerce_bool(attachment.get("has_macro", attachment.get("macro", False)))
            ext_score = 1.0 if ext in dangerous else 0.65 if ext in suspicious else 0.15
            scores.append(
                combine_risk_scores(
                    ext_score,
                    clamp_score(size / large_bytes) if size else 0.0,
                    1.0 if macro else 0.0,
                    method="weighted_high",
                )
            )
        return max(scores, default=0.0)

    @classmethod
    def _domain_mismatch_score(cls, email: Dict) -> float:
        if not isinstance(email, Mapping):
            return 1.0
        from_header = normalize_text(email.get("from", ""), max_length=512)
        actual = cls._sender_domain(from_header)
        if not actual:
            return 1.0
        display_name, _ = parseaddr(from_header)
        display_domains = {
            cls._normalize_domain(item)
            for item in re.findall(r"\b[\w.-]+\.[a-zA-Z]{2,}\b", display_name or "")
        }
        display_domains.discard("")
        if not display_domains:
            return 0.0
        if actual in display_domains or any(actual.endswith(f".{domain}") for domain in display_domains):
            return 0.0
        return 1.0

    @classmethod
    def _reply_to_mismatch_score(cls, email: Mapping[str, Any]) -> float:
        sender = cls._sender_domain(email.get("from", ""))
        reply = cls._sender_domain(email.get("reply_to", email.get("reply-to", "")))
        if not sender or not reply:
            return 0.0
        return 0.0 if sender == reply or sender.endswith(f".{reply}") else 1.0

    @classmethod
    def _avg_url_length(cls, links: List[str]) -> float:
        safe_links = cls._coerce_links(links)
        return sum(float(len(link)) for link in safe_links) / len(safe_links) if safe_links else 0.0

    @classmethod
    def _normalized_url_length_score(cls, url_or_links: Union[str, Sequence[str]]) -> float:
        maximum = coerce_float(cls._cfg().get("max_url_length"), 2048.0, minimum=1.0)
        if isinstance(url_or_links, str):
            length = len(normalize_text(url_or_links, max_length=int(maximum)))
        else:
            length = cls._avg_url_length(list(url_or_links))
        return clamp_score(length / maximum)

    @classmethod
    def _ssl_cert_score(cls, links: List[str]) -> float:
        safe_links = cls._coerce_links(links)
        if not safe_links:
            return 1.0
        secure = sum(1 for link in safe_links if (bundle := cls._safe_url_bundle(link)) and bundle.scheme == "https")
        return clamp_score(secure / len(safe_links))

    @classmethod
    def _unusual_sender_score(cls, sender: str) -> float:
        cls._ensure_domain_lists_loaded()  # idempotent; no repeated disk I/O
        cfg = cls._cfg()
        domain = cls._sender_domain(sender)
        if not domain:
            logger.warning(
                "Invalid sender format: %s",
                safe_log_payload("invalid_sender", {"sender_fingerprint": fingerprint(sender)}),
            )
            return 1.0

        domain_fp = fingerprint(domain)
        if cls._malicious_domains and domain in cls._malicious_domains:
            logger.warning(
                "Malicious sender domain detected: %s",
                safe_log_payload("malicious_sender_domain", {"domain_fingerprint": domain_fp}),
            )
            return 1.0
        if cls._trusted_domains and domain in cls._trusted_domains:
            return 0.0

        suspicious_tlds = {item.lower().lstrip(".") for item in cls._list(cfg.get("suspicious_tlds"))}
        free_webmail = cls._set(cfg.get("free_webmail_domains"))
        disposable = cls._disposable_domains or set()
        tld = domain.rsplit(".", 1)[-1] if "." in domain else domain
        age_score = clamp_score(cfg.get("new_tld_score"), default=0.70) if tld in suspicious_tlds else clamp_score(cfg.get("normal_tld_score"), default=0.30)
        reputation = cls._get_domain_reputation(domain)
        entropy = clamp_score(cls._url_entropy(domain) * coerce_float(cfg.get("domain_entropy_scale"), 2.0, minimum=0.0))
        subdomain = clamp_score(cls._num_subdomains(domain) / coerce_float(cfg.get("subdomain_suspicious_count"), 5.0, minimum=1.0))
        score = weighted_average(
            {
                "age": age_score,
                "reputation": 1.0 - reputation,
                "entropy": entropy,
                "subdomains": subdomain,
                "disposable": 1.0 if domain in disposable else 0.0,
                "webmail": clamp_score(cfg.get("free_webmail_sender_score"), default=0.15) if domain in free_webmail else 0.0,
            },
            cls._mapping(cfg.get("sender_reputation_weights")),
        )
        logger.debug(
            "Sender reputation analysis: %s",
            safe_log_payload(
                "sender_reputation_analysis",
                {"domain_fingerprint": domain_fp, "score": score, "risk_level": categorize_risk(score)},
            ),
        )
        return score

    @classmethod
    def _unusual_time_score(cls, timestamp: float) -> float:
        if not timestamp:
            return 0.0
        cfg = cls._cfg()
        start = coerce_int(cfg.get("off_hours_start"), 21, minimum=0, maximum=23)
        end = coerce_int(cfg.get("off_hours_end"), 6, minimum=0, maximum=23)
        dt = datetime.fromtimestamp(float(timestamp))
        if start > end:
            return 1.0 if dt.hour >= start or dt.hour < end else 0.0
        return 1.0 if start <= dt.hour < end else 0.0

    # ------------------------------------------------------------------
    # URL features
    # ------------------------------------------------------------------

    @classmethod
    def _url_entropy(cls, url: str) -> float:
        text = normalize_text(url, max_length=coerce_int(cls._cfg().get("max_url_length"), 2048), lowercase=True)
        if not text:
            return 0.0
        freq: Dict[str, int] = defaultdict(int)
        for char in text:
            freq[char] += 1
        total = len(text)
        entropy = -sum((count / total) * math.log2(count / total) for count in freq.values())
        return clamp_score(entropy / 8.0)

    @classmethod
    def _num_subdomains(cls, url: str) -> float:
        domain = cls._safe_domain_from_url(url)
        if not domain or cls._is_ip_literal(domain):
            return 0.0
        parts = [part for part in domain.split(".") if part and part != "www"]
        return float(max(len(parts) - 2, 0))

    @classmethod
    def _https_used(cls, url: str) -> float:
        bundle = cls._safe_url_bundle(url)
        return 1.0 if bundle and bundle.scheme == "https" else 0.0

    @classmethod
    def _invalid_url_score(cls, url: str) -> float:
        text = normalize_text(url, max_length=coerce_int(cls._cfg().get("max_url_length"), 2048))
        if not text:
            return 1.0
        return 0.0 if cls._safe_url_bundle(text) is not None else 1.0

    @classmethod
    def _url_redirect_count(cls, url: str) -> float:
        cfg = cls._cfg()
        params = {normalize_text(item, lowercase=True) for item in cls._list(cfg.get("redirect_query_keys"))}
        path_terms = {normalize_text(item, lowercase=True) for item in cls._list(cfg.get("redirect_path_terms"))}
        normalized = normalize_text(url, max_length=coerce_int(cfg.get("max_url_length"), 2048), lowercase=True)
        parsed = urlparse(normalized if re.match(r"^[a-z][a-z0-9+.-]*://", normalized) else f"https://{normalized}")
        hits = 0.0
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            key_l, value_l = key.lower(), value.lower()
            if key_l in params or any(term and term in key_l for term in params):
                hits += 1.0
            if any(value_l.startswith(prefix) for prefix in ("http://", "https://", "//")):
                hits += 1.0
        if any(term and term in parsed.path.lower() for term in path_terms):
            hits += 1.0
        return clamp_score(hits / coerce_float(cfg.get("redirect_hit_cap"), 5.0, minimum=1.0))

    @classmethod
    def _special_char_count(cls, url: str) -> float:
        text = normalize_text(url, max_length=coerce_int(cls._cfg().get("max_url_length"), 2048))
        if not text:
            return 0.0
        allowed_pattern = r"[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]"
        suspicious = sum(1 for char in text if not re.match(allowed_pattern, char))
        return clamp_score(suspicious / len(text))

    @classmethod
    def _contains_ip_address(cls, url: str) -> float:
        domain = cls._safe_domain_from_url(url)
        return 1.0 if cls._is_ip_literal(domain) or cls._IP_LITERAL_RE.search(str(url or "")) else 0.0

    @staticmethod
    def _is_ip_literal(value: str) -> bool:
        text = str(value or "").strip("[]")
        if not text:
            return False
        try:
            ipaddress.ip_address(text)
            return True
        except ValueError:
            return False

    @classmethod
    def _private_host_score(cls, url: str) -> float:
        bundle = cls._safe_url_bundle(url)
        return 1.0 if bundle and bundle.is_private_host else 0.0

    @classmethod
    def _credential_in_url_score(cls, url: str) -> float:
        normalized = normalize_text(url, max_length=coerce_int(cls._cfg().get("max_url_length"), 2048))
        parsed = urlparse(normalized if re.match(r"^[a-z][a-z0-9+.-]*://", normalized) else f"https://{normalized}")
        return 1.0 if parsed.username is not None or parsed.password is not None else 0.0

    @classmethod
    def _url_shortener_score(cls, url: str) -> float:
        return 1.0 if cls._safe_domain_from_url(url) in cls._set(cls._cfg().get("url_shortener_domains")) else 0.0

    @classmethod
    def _suspicious_tld_score(cls, url: str) -> float:
        domain = cls._safe_domain_from_url(url)
        if not domain or "." not in domain:
            return 0.0
        tld = domain.rsplit(".", 1)[-1]
        suspicious = {item.lower().lstrip(".") for item in cls._list(cls._cfg().get("suspicious_tlds"))}
        return 1.0 if tld in suspicious else 0.0

    @classmethod
    def _path_depth_score(cls, url: str) -> float:
        cfg = cls._cfg()
        normalized = normalize_text(url, max_length=coerce_int(cfg.get("max_url_length"), 2048))
        parsed = urlparse(normalized if re.match(r"^[a-z][a-z0-9+.-]*://", normalized) else f"https://{normalized}")
        depth = len([segment for segment in parsed.path.split("/") if segment])
        return clamp_score(depth / coerce_float(cfg.get("path_depth_suspicious_count"), 6.0, minimum=1.0))

    @classmethod
    def _query_risk_score(cls, url: str) -> float:
        cfg = cls._cfg()
        normalized = normalize_text(url, max_length=coerce_int(cfg.get("max_url_length"), 2048))
        parsed = urlparse(normalized if re.match(r"^[a-z][a-z0-9+.-]*://", normalized) else f"https://{normalized}")
        sensitive = {normalize_text(item, lowercase=True) for item in cls._list(cfg.get("sensitive_query_keys"))}
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        if not pairs:
            return 0.0
        hits = 0.0
        for key, value in pairs:
            key_l = key.lower()
            if key_l in sensitive or get_sensitive_key_regex().search(key_l):
                hits += 1.0
            if len(value) >= coerce_int(cfg.get("long_query_value_length"), 80, minimum=8):
                hits += 0.5
        return clamp_score(hits / coerce_float(cfg.get("query_hit_cap"), 5.0, minimum=1.0))

    @classmethod
    def _brand_impersonation_score(cls, text_or_url: str) -> float:
        """Score possible protected-brand impersonation.

        The compact brand is deliberately recomputed *inside* the domain loop;
        the previous implementation reused the last value from a different loop.
        """

        cfg = cls._cfg()
        brands = [normalize_text(item, lowercase=True) for item in cls._list(cfg.get("protected_brand_terms"))]
        if not brands:
            return 0.0
        text = normalize_text(text_or_url, max_length=coerce_int(cfg.get("max_text_length"), 4096), lowercase=True)
        text_compact = re.sub(r"[^a-z0-9]", "", text)
        score = 0.0

        for brand in brands:
            compact = re.sub(r"[^a-z0-9]", "", brand)
            if brand and brand in text:
                score = max(score, 0.65)
            if compact and compact in text_compact:
                score = max(score, 0.75)

        candidates = cls._extract_links_from_text(text)
        if not candidates and "." in text and not any(char.isspace() for char in text):
            candidates = [text]
        for url in candidates:
            domain = cls._safe_domain_from_url(url)
            if not domain:
                continue
            compact_domain = re.sub(r"[^a-z0-9]", "", domain)
            for brand in brands:
                compact = re.sub(r"[^a-z0-9]", "", brand)  # fixed scope
                if brand and brand in domain:
                    score = max(score, 0.85)
                elif compact and compact in compact_domain:
                    score = max(score, 0.85)
        return clamp_score(score)

    @classmethod
    def _punycode_or_homograph_score(cls, url: str) -> float:
        text = normalize_text(url, max_length=coerce_int(cls._cfg().get("max_url_length"), 2048), lowercase=True)
        domain = cls._safe_domain_from_url(text)
        if "xn--" in domain:
            return 1.0
        return 0.75 if text and any(ord(char) >= 128 for char in text) else 0.0

    # ------------------------------------------------------------------
    # Public feature maps / vectors
    # ------------------------------------------------------------------

    def extract_email_feature_map(self, email: Mapping[str, Any]) -> Dict[str, float]:
        if not isinstance(email, Mapping):
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                "Email feature extraction expected a mapping payload.",
                severity=SecuritySeverity.MEDIUM,
                context={"input_type": type(email).__name__},
                component="safety_features",
            )
        try:
            links = self._coerce_links(email.get("links") or self._extract_links_from_text(email.get("body", "")))
            from_header = email.get("from", "")
            subject = email.get("subject", "")
            body = email.get("body", "")
            timestamp = email.get("timestamp", 0.0)
            from_length = float(len(normalize_text(from_header, max_length=512)))
            subject_length = float(len(normalize_text(subject, max_length=512)))
            link_count = float(len(links))
            avg_url_length = self._avg_url_length(links)
            return {
                # Historical raw features retained for existing model artifacts.
                "from_length": from_length,
                "subject_length": subject_length,
                "link_count": link_count,
                "avg_url_length": avg_url_length,
                # Normalized migration-safe counterparts for new models.
                "normalized_from_length_score": clamp_score(from_length / 512.0),
                "normalized_subject_length_score": clamp_score(subject_length / 512.0),
                "normalized_link_count_score": clamp_score(link_count / coerce_float(self.sf_config.get("max_links_per_email"), 100.0, minimum=1.0)),
                "normalized_url_length_score": self._normalized_url_length_score(links),
                "suspicious_keyword_score": self._contains_suspicious_keywords(str(subject)),
                "urgent_language_score": self._contains_urgent_language(str(body)),
                "attachment_present": self._contains_attachment(dict(email)),
                "attachment_risk_score": self._attachment_risk_score(email),
                "domain_mismatch_score": self._domain_mismatch_score(dict(email)),
                "reply_to_mismatch_score": self._reply_to_mismatch_score(email),
                "ssl_cert_score": self._ssl_cert_score(links),
                "unusual_sender_score": self._unusual_sender_score(str(from_header)),
                "unusual_time_score": self._unusual_time_score(coerce_float(timestamp, 0.0, minimum=0.0)),
                "brand_impersonation_score": self._brand_impersonation_score(f"{subject} {body}"),
            }
        except SecurityError:
            raise
        except Exception as exc:
            raise wrap_security_exception(
                exc,
                operation="extract_email_feature_map",
                component="safety_features",
                context={"email_fingerprint": fingerprint(sanitize_for_logging(dict(email)))},
                error_type=SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                severity=SecuritySeverity.HIGH,
            ) from exc

    def extract_email_feature_vector(self, email: Mapping[str, Any], *, legacy: bool = True) -> List[float]:
        features = self.extract_email_feature_map(email)
        if legacy:
            return [
                features["from_length"],
                features["subject_length"],
                features["suspicious_keyword_score"],
                features["link_count"],
                features["urgent_language_score"],
                features["attachment_present"],
                features["domain_mismatch_score"],
                features["avg_url_length"],
                features["ssl_cert_score"],
                features["unusual_sender_score"],
                features["unusual_time_score"],
            ]
        order = self._list(self.sf_config.get("feature_order", {}).get("email"))
        return [float(features[name]) for name in order if name in features]

    def assess_email_risk(self, email: Mapping[str, Any]) -> FeatureExtractionResult:
        features = self.extract_email_feature_map(email)
        risk_features = {
            "suspicious_keywords": features["suspicious_keyword_score"],
            "urgent_language": features["urgent_language_score"],
            "attachments": combine_risk_scores(features["attachment_present"] * 0.25, features["attachment_risk_score"], method="weighted_high"),
            "domain_mismatch": features["domain_mismatch_score"],
            "reply_to_mismatch": features["reply_to_mismatch_score"],
            "insecure_links": 1.0 - features["ssl_cert_score"],
            "url_length": features["normalized_url_length_score"],
            "sender": features["unusual_sender_score"],
            "off_hours": features["unusual_time_score"],
            "brand_impersonation": features["brand_impersonation_score"],
        }
        risk = weighted_average(risk_features, self._mapping(self.sf_config.get("email_risk_weights")))
        threshold = coerce_float(self.sf_config.get("indicator_threshold"), 0.5)
        indicators = [name for name, value in risk_features.items() if clamp_score(value) >= threshold]
        return FeatureExtractionResult(
            source_type="email",
            feature_map=features,
            feature_vector=self.extract_email_feature_vector(email, legacy=True),
            risk_score=risk,
            risk_level=categorize_risk(risk),
            decision=threshold_decision(
                risk,
                block_threshold=self.sf_config.get("block_threshold"),
                review_threshold=self.sf_config.get("review_threshold"),
            ),
            indicators=indicators,
            metadata={
                "source_fingerprint": fingerprint(sanitize_for_logging(dict(email))),
                "legacy_vector_schema": LEGACY_EMAIL_SCHEMA_VERSION,
            },
            timestamp=utc_iso(),
        )

    def extract_url_feature_map(self, url: str) -> Dict[str, float]:
        text = normalize_text(url, max_length=coerce_int(self.sf_config.get("max_url_length"), 2048))
        if not text:
            raise SecurityError(
                SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                "URL feature extraction received an empty URL.",
                severity=SecuritySeverity.MEDIUM,
                context={"url_fingerprint": fingerprint(url)},
                component="safety_features",
            )
        try:
            domain = self._safe_domain_from_url(text)
            reputation = self._get_domain_reputation(domain)
            subdomains = self._num_subdomains(text)
            return {
                "url_length": float(len(text)),
                "normalized_url_length_score": self._normalized_url_length_score(text),
                "url_entropy": self._url_entropy(text),
                "num_subdomains": float(subdomains),
                "subdomain_score": clamp_score(subdomains / coerce_float(self.sf_config.get("subdomain_suspicious_count"), 5.0, minimum=1.0)),
                "contains_ip": self._contains_ip_address(text),
                "https_used": self._https_used(text),
                "invalid_or_disallowed_url_score": self._invalid_url_score(text),
                "redirect_score": self._url_redirect_count(text),
                "special_char_score": self._special_char_count(text),
                "private_host_score": self._private_host_score(text),
                "credential_in_url_score": self._credential_in_url_score(text),
                "url_shortener_score": self._url_shortener_score(text),
                "suspicious_tld_score": self._suspicious_tld_score(text),
                "path_depth_score": self._path_depth_score(text),
                "query_risk_score": self._query_risk_score(text),
                "brand_impersonation_score": self._brand_impersonation_score(text),
                "punycode_or_homograph_score": self._punycode_or_homograph_score(text),
                "domain_reputation_risk": 1.0 - reputation,
            }
        except SecurityError:
            raise
        except Exception as exc:
            raise wrap_security_exception(
                exc,
                operation="extract_url_feature_map",
                component="safety_features",
                context={"url_fingerprint": fingerprint(text)},
                error_type=SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                severity=SecuritySeverity.HIGH,
            ) from exc

    def extract_url_feature_vector(self, url: str, *, legacy: bool = True, domain_age_score: float = 0.8) -> List[float]:
        features = self.extract_url_feature_map(url)
        if legacy:
            return [
                features["url_length"],
                features["url_entropy"],
                features["num_subdomains"],
                features["contains_ip"],
                features["https_used"],
                features["redirect_score"],
                clamp_score(domain_age_score, default=0.8),
                features["special_char_score"],
            ]
        order = self._list(self.sf_config.get("feature_order", {}).get("url"))
        return [float(features[name]) for name in order if name in features]

    def assess_url_risk(self, url: str) -> FeatureExtractionResult:
        features = self.extract_url_feature_map(url)
        risk_features = {
            "url_length": features["normalized_url_length_score"],
            "entropy": features["url_entropy"],
            "subdomains": features["subdomain_score"],
            "ip_literal": features["contains_ip"],
            "insecure_scheme": 1.0 - features["https_used"],
            "invalid_url": features["invalid_or_disallowed_url_score"],
            "redirect": features["redirect_score"],
            "special_chars": features["special_char_score"],
            "private_host": features["private_host_score"],
            "credentials": features["credential_in_url_score"],
            "shortener": features["url_shortener_score"],
            "suspicious_tld": features["suspicious_tld_score"],
            "path_depth": features["path_depth_score"],
            "query": features["query_risk_score"],
            "brand_impersonation": features["brand_impersonation_score"],
            "homograph": features["punycode_or_homograph_score"],
            "domain_reputation": features["domain_reputation_risk"],
        }
        weights = self._mapping(self.sf_config.get("url_risk_weights"))
        # New indicator has no configured weight in older YAML.  Give it the
        # average configured weight only when present as a true risk, avoiding a
        # config migration requirement while still failing conservatively.
        if risk_features["invalid_url"] > 0.0 and "invalid_url" not in weights:
            positive = [float(v) for v in weights.values() if float(v) > 0.0]
            if positive:
                weights["invalid_url"] = sum(positive) / len(positive)
        risk = weighted_average(risk_features, weights)
        threshold = coerce_float(self.sf_config.get("indicator_threshold"), 0.5)
        indicators = [name for name, value in risk_features.items() if clamp_score(value) >= threshold]
        return FeatureExtractionResult(
            source_type="url",
            feature_map=features,
            feature_vector=self.extract_url_feature_vector(url, legacy=True),
            risk_score=risk,
            risk_level=categorize_risk(risk),
            decision=threshold_decision(
                risk,
                block_threshold=self.sf_config.get("block_threshold"),
                review_threshold=self.sf_config.get("review_threshold"),
            ),
            indicators=indicators,
            metadata={
                "url_fingerprint": fingerprint(url),
                "domain_fingerprint": fingerprint(self._safe_domain_from_url(url)),
                "legacy_vector_schema": LEGACY_URL_SCHEMA_VERSION,
            },
            timestamp=utc_iso(),
        )

    def get_feature_schema(self) -> Dict[str, Any]:
        return {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "module_version": MODULE_VERSION,
            "email_legacy_version": LEGACY_EMAIL_SCHEMA_VERSION,
            "url_legacy_version": LEGACY_URL_SCHEMA_VERSION,
            "email_legacy": [
                "from_length", "subject_length", "suspicious_keyword_score",
                "link_count", "urgent_language_score", "attachment_present",
                "domain_mismatch_score", "avg_url_length", "ssl_cert_score",
                "unusual_sender_score", "unusual_time_score",
            ],
            "url_legacy": [
                "url_length", "url_entropy", "num_subdomains", "contains_ip",
                "https_used", "redirect_score", "domain_age_score_external",
                "special_char_score",
            ],
            "email_configured": self._list(self.sf_config.get("feature_order", {}).get("email")),
            "url_configured": self._list(self.sf_config.get("feature_order", {}).get("url")),
        }

    @classmethod
    def clear_caches(cls) -> None:
        """Clear runtime caches while preserving the loaded config snapshot."""
        with cls._state_lock:
            cls._domain_reputation_cache.clear()
            cls._trusted_domains = None
            cls._malicious_domains = None
            cls._disposable_domains = None
            cls._domain_lists_loaded = False
        cls._ensure_domain_lists_loaded()

__all__ = [
    # Module metadata
    "MODULE_VERSION",
    # Dataclasses
    "FeatureExtractionResult",
    "DomainReputationRecord",
    # Core class
    "SafetyFeatures",
]

if __name__ == "__main__":
    print("\n=== Running Safety Features ===\n")
    printer.status("TEST", "Safety Features initialized", "info")
    features = SafetyFeatures()
    sample = {
        "from": "Security Team <security@paypa1-login.example>",
        "reply_to": "support@shady-site.cc",
        "subject": "Urgent account suspended - verify now",
        "body": "Verify at https://paypa1-login.example/reset?password=secret immediately.",
        "links": ["https://paypa1-login.example/reset?password=secret"],
        "timestamp": datetime.now().timestamp(),
    }
    result = features.assess_email_risk(sample)
    assert len(result.feature_vector) == 11
    assert len(features.extract_url_feature_vector(sample["links"][0])) == 8
    assert 0.0 <= result.risk_score <= 1.0
    assert features._brand_impersonation_score("https://paypa1-login.example") >= 0.0
    printer.status("TEST", f"Email risk={result.risk_score:.3f}", "info")
    print("\n=== Test ran successfully ===\n")