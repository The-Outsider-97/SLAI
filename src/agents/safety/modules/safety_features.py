"""
Safety feature extraction primitives for the Safety Agent subsystem.

This module converts emails, URLs, and compact security-relevant payloads into
bounded numeric features for phishing, abuse, and cyber-safety models. It is a
feature layer only: it does not own model training, intervention policy,
allow/block decisions, incident response, persistence, or human-oversight state.
Those concerns remain with the neural network, adaptive security, safety guard,
secure memory, and orchestration layers.

Design goals:
- keep feature extraction deterministic, auditable, and configuration-backed;
- avoid leaking PII, secrets, tokens, raw URLs, or raw email content into logs;
- fail closed for malformed or suspicious security inputs where appropriate;
- preserve the legacy private feature methods used by AdaptiveSecurity;
- use shared safety helpers and structured security errors rather than
  re-implementing redaction, hashing, URL normalization, score clamping, or
  exception wrapping locally.
"""

from __future__ import annotations

import ipaddress
import math
import re

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parseaddr
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union
from urllib.parse import parse_qsl, urlparse

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.security_error import *
from ..utils.safety_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Safety Features")
printer = PrettyPrinter()


@dataclass(frozen=True)
class FeatureExtractionResult:
    """Structured, audit-safe feature extraction result."""

    source_type: str
    feature_map: Dict[str, float]
    feature_vector: List[float]
    risk_score: float
    risk_level: str
    decision: str
    indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp or utc_iso()
        data["metadata"] = sanitize_for_logging(data.get("metadata", {}))
        return data


@dataclass(frozen=True)
class DomainReputationRecord:
    """Cached domain reputation record with safe metadata."""

    domain: str
    score: float
    source: str
    timestamp: float
    indicators: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "score": clamp_score(self.score, default=0.5),
            "source": self.source,
            "timestamp": self.timestamp,
            "indicators": list(self.indicators),
        }


class SafetyFeatures:
    """
    Feature extraction for email, URL, and lightweight cyber-safety signals.

    Public aggregate methods return structured results. Legacy underscore
    methods are retained because AdaptiveSecurity calls them directly.
    """

    _trusted_domains: Optional[Set[str]] = None
    _malicious_domains: Optional[Set[str]] = None
    _disposable_domains: Optional[Set[str]] = None
    _domain_reputation_cache: Dict[str, DomainReputationRecord] = {}

    _EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    _URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>'\"()]+")
    _IP_LITERAL_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\[[0-9a-fA-F:]+\]")
    _HOST_ALLOWED_RE = re.compile(r"^[a-z0-9.-]+$", re.IGNORECASE)
    _WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)

    def __init__(self) -> None:
        self.config = load_global_config()
        self.sf_config = get_config_section("safety_features")
        if self.sf_config.get("strict_config_validation", True):
            self.validate_configuration()
        SafetyFeatures._load_domain_lists()
        logger.info("Safety Features initialized: %s", safe_log_payload(
            "safety_features_initialized",
            {
                "configured": bool(self.sf_config),
                "trusted_domain_count": len(SafetyFeatures._trusted_domains or set()),
                "malicious_domain_count": len(SafetyFeatures._malicious_domains or set()),
                "schema_version": self.sf_config.get("schema_version"),
            },
        ))

    # ==============================
    # Configuration and normalization
    # ==============================

    @staticmethod
    def validate_configuration() -> None:
        """Validate the YAML-backed safety_features section without owning defaults."""

        cfg = get_config_section("safety_features")
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
        for weight_group in ("sender_reputation_weights", "email_risk_weights", "url_risk_weights"):
            weights = cfg.get(weight_group, {})
            if not isinstance(weights, Mapping):
                raise ConfigurationTamperingError(
                    config_file_path=str(load_global_config().get("__config_path__", "secure_config.yaml")),
                    suspicious_change=f"safety_features.{weight_group} must be a mapping",
                    component="safety_features",
                )

    @staticmethod
    def _cfg() -> Dict[str, Any]:
        # Thin access to the existing loader only. No separate config ownership or defaults live here.
        return get_config_section("safety_features")

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
    def _set(value: Any) -> Set[str]:
        return {SafetyFeatures._normalize_domain(item) for item in SafetyFeatures._list(value) if SafetyFeatures._normalize_domain(item)}

    @staticmethod
    def _mapping(value: Any) -> Dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _tokenize(text: Any) -> List[str]:
        normalized = normalize_text(text, max_length=coerce_int(SafetyFeatures._cfg().get("max_text_length"), 4096), lowercase=True)
        return SafetyFeatures._WORD_RE.findall(normalized)

    @staticmethod
    def _phrase_score(text: Any, phrases: Sequence[str], *, cap: Optional[float] = None) -> float:
        normalized = normalize_text(text, max_length=coerce_int(SafetyFeatures._cfg().get("max_text_length"), 4096), lowercase=True)
        if not normalized or not phrases:
            return 0.0
        hits = 0.0
        for phrase in phrases:
            phrase_text = normalize_text(phrase, lowercase=True)
            if phrase_text and phrase_text in normalized:
                hits += 1.0
        divisor = cap if cap and cap > 0 else max(float(len(phrases)), 1.0)
        return clamp_score(hits / divisor)

    @staticmethod
    def _keyword_score(text: Any, keywords: Sequence[str], *, cap: Optional[float] = None) -> float:
        tokens = set(SafetyFeatures._tokenize(text))
        normalized_keywords = {normalize_text(keyword, lowercase=True) for keyword in keywords if normalize_text(keyword)}
        if not tokens or not normalized_keywords:
            return 0.0
        hits = len(tokens & normalized_keywords)
        divisor = cap if cap and cap > 0 else max(float(len(normalized_keywords)), 1.0)
        return clamp_score(hits / divisor)

    @staticmethod
    def _normalize_domain(domain: Any) -> str:
        text = normalize_text(domain, max_length=253, lowercase=True)
        text = text.strip(".[] ")
        if not text:
            return ""
        return text[4:] if text.startswith("www.") else text

    @staticmethod
    def _sender_address(sender: Any) -> str:
        _, address = parseaddr("" if sender is None else str(sender))
        if address:
            return normalize_text(address, max_length=320, lowercase=True)
        text = normalize_text(sender, max_length=320, lowercase=True)
        match = SafetyFeatures._EMAIL_RE.search(text)
        return match.group(0).lower() if match else text

    @staticmethod
    def _sender_domain(sender: Any) -> str:
        address = SafetyFeatures._sender_address(sender)
        if "@" not in address:
            return ""
        return SafetyFeatures._normalize_domain(address.rsplit("@", 1)[-1])

    @staticmethod
    def _safe_url_bundle(raw_url: Any) -> Optional[SanitizedURL]:
        text = normalize_text(raw_url, max_length=coerce_int(SafetyFeatures._cfg().get("max_url_length"), 2048), preserve_newlines=False)
        if not text:
            return None
        try:
            return sanitize_url(text)
        except (SecurityError, ValueError, TypeError):
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

    @staticmethod
    def _safe_domain_from_url(raw_url: Any) -> str:
        bundle = SafetyFeatures._safe_url_bundle(raw_url)
        if bundle:
            return SafetyFeatures._normalize_domain(bundle.hostname)
        text = normalize_text(raw_url, max_length=coerce_int(SafetyFeatures._cfg().get("max_url_length"), 2048), lowercase=True)
        if not text:
            return ""
        host = re.sub(r"^[a-z][a-z0-9+.-]*://", "", text).split("/", 1)[0].split("?", 1)[0]
        return SafetyFeatures._normalize_domain(host)

    @staticmethod
    def _extract_links_from_text(text: Any) -> List[str]:
        cfg = SafetyFeatures._cfg()
        max_links = coerce_int(cfg.get("max_links_per_email"), 100, minimum=1)
        normalized = normalize_text(text, max_length=coerce_int(cfg.get("max_text_length"), 4096))
        return SafetyFeatures._URL_RE.findall(normalized)[:max_links]

    @staticmethod
    def _coerce_links(links: Any) -> List[str]:
        cfg = SafetyFeatures._cfg()
        max_links = coerce_int(cfg.get("max_links_per_email"), 100, minimum=1)
        if links is None:
            return []
        if isinstance(links, str):
            candidates = SafetyFeatures._extract_links_from_text(links)
        elif isinstance(links, Iterable):
            candidates = [str(link) for link in links]
        else:
            candidates = [str(links)]
        return [normalize_text(link, max_length=coerce_int(cfg.get("max_url_length"), 2048)) for link in candidates if str(link).strip()][:max_links]

    @staticmethod
    def _coerce_attachments(email: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        raw_attachments = email.get("attachments", [])
        max_attachments = coerce_int(SafetyFeatures._cfg().get("max_attachments_per_email"), 50, minimum=1)
        if not raw_attachments:
            return []
        if isinstance(raw_attachments, Mapping):
            return [raw_attachments]
        if isinstance(raw_attachments, Iterable) and not isinstance(raw_attachments, (str, bytes)):
            result: List[Mapping[str, Any]] = []
            for item in list(raw_attachments)[:max_attachments]:
                result.append(item if isinstance(item, Mapping) else {"filename": str(item)})
            return result
        return [{"filename": str(raw_attachments)}]

    # ==============================
    # 1. Extracting email features
    # ==============================

    @staticmethod
    def _contains_suspicious_keywords(text: str) -> float:
        """Check for phishing-related keywords in subject/header text."""

        keywords = SafetyFeatures._list(SafetyFeatures._cfg().get("basic_keywords"))
        cap = coerce_float(SafetyFeatures._cfg().get("keyword_hit_cap"), 5.0, minimum=1.0)
        return SafetyFeatures._keyword_score(text, keywords, cap=cap)

    @staticmethod
    def _contains_urgent_language(email_body: str) -> float:
        """Calculate urgency/social-engineering score of email content."""

        cfg = SafetyFeatures._cfg()
        urgent_keywords = SafetyFeatures._list(cfg.get("urgent_keywords"))
        urgent_phrases = SafetyFeatures._list(cfg.get("urgent_phrases"))
        cap = coerce_float(cfg.get("urgent_hit_cap"), 4.0, minimum=1.0)
        keyword_score = SafetyFeatures._keyword_score(email_body, urgent_keywords, cap=cap)
        phrase_score = SafetyFeatures._phrase_score(email_body, urgent_phrases, cap=cap)
        return combine_risk_scores(keyword_score, phrase_score, method="weighted_high")

    @staticmethod
    def _contains_attachment(email: Dict) -> float:
        """Check for email attachments while retaining legacy binary behavior."""

        if not isinstance(email, Mapping):
            return 0.0
        return 1.0 if SafetyFeatures._coerce_attachments(email) else 0.0

    @staticmethod
    def _attachment_risk_score(email: Mapping[str, Any]) -> float:
        """Score attachment risk from extension, size, and macro/script indicators."""

        attachments = SafetyFeatures._coerce_attachments(email)
        if not attachments:
            return 0.0
        cfg = SafetyFeatures._cfg()
        dangerous_ext = {ext.lower().lstrip(".") for ext in SafetyFeatures._list(cfg.get("dangerous_attachment_extensions"))}
        suspicious_ext = {ext.lower().lstrip(".") for ext in SafetyFeatures._list(cfg.get("suspicious_attachment_extensions"))}
        max_size = coerce_int(cfg.get("large_attachment_bytes"), 10_485_760, minimum=1)
        scores: List[float] = []
        for attachment in attachments:
            filename = normalize_text(attachment.get("filename", ""), max_length=256, lowercase=True)
            extension = filename.rsplit(".", 1)[-1] if "." in filename else ""
            size = coerce_float(attachment.get("size_bytes", attachment.get("size", 0.0)), 0.0, minimum=0.0)
            has_macro = coerce_bool(attachment.get("has_macro", attachment.get("macro", False)))
            ext_score = 1.0 if extension in dangerous_ext else 0.65 if extension in suspicious_ext else 0.15
            size_score = clamp_score(size / max_size) if size else 0.0
            macro_score = 1.0 if has_macro else 0.0
            scores.append(combine_risk_scores(ext_score, size_score, macro_score, method="weighted_high"))
        return max(scores) if scores else 0.0

    @staticmethod
    def _domain_mismatch_score(email: Dict) -> float:
        """Compare display-name domain hints with the actual sender domain."""

        if not isinstance(email, Mapping):
            return 1.0
        from_header = normalize_text(email.get("from", ""), max_length=512)
        actual_domain = SafetyFeatures._sender_domain(from_header)
        if not actual_domain:
            return 1.0
        display_name, _ = parseaddr(from_header)
        display_domains = {SafetyFeatures._normalize_domain(item) for item in re.findall(r"\b[\w.-]+\.[a-zA-Z]{2,}\b", display_name or "")}
        if not display_domains:
            return 0.0
        if actual_domain in display_domains or any(actual_domain.endswith(f".{domain}") for domain in display_domains):
            return 0.0
        return 1.0

    @staticmethod
    def _reply_to_mismatch_score(email: Mapping[str, Any]) -> float:
        sender_domain = SafetyFeatures._sender_domain(email.get("from", ""))
        reply_domain = SafetyFeatures._sender_domain(email.get("reply_to", email.get("reply-to", "")))
        if not sender_domain or not reply_domain:
            return 0.0
        return 0.0 if sender_domain == reply_domain or sender_domain.endswith(f".{reply_domain}") else 1.0

    @staticmethod
    def _avg_url_length(links: List[str]) -> float:
        """Calculate average length of URLs in email; retained as a raw legacy feature."""

        safe_links = SafetyFeatures._coerce_links(links)
        if not safe_links:
            return 0.0
        return sum(float(len(link)) for link in safe_links) / float(len(safe_links))

    @staticmethod
    def _normalized_url_length_score(url_or_links: Union[str, Sequence[str]]) -> float:
        cfg = SafetyFeatures._cfg()
        max_url_length = coerce_float(cfg.get("max_url_length"), 2048.0, minimum=1.0)
        if isinstance(url_or_links, str):
            length = len(normalize_text(url_or_links, max_length=int(max_url_length)))
        else:
            length = SafetyFeatures._avg_url_length(list(url_or_links))
        return clamp_score(length / max_url_length)

    @staticmethod
    def _ssl_cert_score(links: List[str]) -> float:
        """Calculate HTTPS adoption rate among links. A higher score means safer transport."""

        safe_links = SafetyFeatures._coerce_links(links)
        if not safe_links:
            return 1.0
        secure = 0
        for link in safe_links:
            bundle = SafetyFeatures._safe_url_bundle(link)
            if bundle and bundle.scheme == "https":
                secure += 1
        return clamp_score(secure / len(safe_links))

    @classmethod
    def _load_domain_lists(cls) -> None:
        """Load trusted, malicious, and disposable domain sets from secure_config.yaml."""

        cfg = get_config_section("safety_features")
        trusted = cls._set(cfg.get("trusted_domains"))
        malicious = cls._set(cfg.get("malicious_domains"))
        disposable = cls._set(cfg.get("disposable_email_domains"))

        for path in SafetyFeatures._list(cfg.get("trusted_domains_paths")):
            trusted.update(cls._load_domain_file(path, list_name="trusted_domains"))
        for path in SafetyFeatures._list(cfg.get("malicious_domains_paths")):
            malicious.update(cls._load_domain_file(path, list_name="malicious_domains"))
        for path in SafetyFeatures._list(cfg.get("disposable_domains_paths")):
            disposable.update(cls._load_domain_file(path, list_name="disposable_email_domains"))

        trusted -= malicious
        if coerce_bool(cfg.get("disposable_domains_override_trust"), True):
            trusted -= disposable

        cls._trusted_domains = trusted
        cls._malicious_domains = malicious
        cls._disposable_domains = disposable

    @staticmethod
    def _load_domain_file(path: str, *, list_name: str) -> Set[str]:
        if not path:
            return set()
        text = load_text_file(path, max_bytes=coerce_int(SafetyFeatures._cfg().get("domain_list_max_bytes"), 1_048_576, minimum=1024))
        domains = set()
        for line in text.splitlines():
            clean = line.strip().split("#", 1)[0].strip()
            if clean:
                domains.add(SafetyFeatures._normalize_domain(clean))
        logger.info("Loaded domain list: %s", safe_log_payload(list_name, {"count": len(domains), "path": path}))
        return domains

    @staticmethod
    def _get_domain_reputation(domain: str) -> float:
        """Return reputation score where 1.0 is trusted and 0.0 is malicious."""

        cfg = SafetyFeatures._cfg()
        ttl = coerce_float(cfg.get("domain_cache_ttl_seconds"), 86400.0, minimum=1.0)
        domain = SafetyFeatures._normalize_domain(domain)
        if not domain:
            return clamp_score(cfg.get("unknown_domain_reputation"), default=0.5)

        cached = SafetyFeatures._domain_reputation_cache.get(domain)
        now = datetime.now(timezone.utc).timestamp()
        if cached and now - cached.timestamp < ttl:
            return clamp_score(cached.score, default=0.5)

        overrides = {SafetyFeatures._normalize_domain(k): clamp_score(v, default=0.5) for k, v in SafetyFeatures._mapping(cfg.get("domain_reputation_overrides")).items()}
        if domain in overrides:
            score = overrides[domain]
            source = "configured_override"
        elif SafetyFeatures._malicious_domains and domain in SafetyFeatures._malicious_domains:
            score = 0.0
            source = "malicious_domain_list"
        elif SafetyFeatures._trusted_domains and domain in SafetyFeatures._trusted_domains:
            score = 1.0
            source = "trusted_domain_list"
        elif SafetyFeatures._disposable_domains and domain in SafetyFeatures._disposable_domains:
            score = clamp_score(cfg.get("disposable_domain_reputation"), default=0.25)
            source = "disposable_domain_list"
        else:
            score = clamp_score(cfg.get("unknown_domain_reputation"), default=0.5)
            source = "unknown_default"

        SafetyFeatures._domain_reputation_cache[domain] = DomainReputationRecord(domain, score, source, now)
        return score

    @staticmethod
    def _unusual_sender_score(sender: str) -> float:
        """Enhanced sender reputation scoring with configuration-backed factors."""

        SafetyFeatures._load_domain_lists()
        cfg = SafetyFeatures._cfg()
        domain = SafetyFeatures._sender_domain(sender)
        if not domain:
            logger.warning("Invalid sender format: %s", safe_log_payload("invalid_sender", {"sender_fingerprint": fingerprint(sender)}))
            return 1.0

        if SafetyFeatures._malicious_domains and domain in SafetyFeatures._malicious_domains:
            logger.warning("Malicious sender domain detected: %s", safe_log_payload("malicious_sender_domain", {"domain": domain}))
            return 1.0
        if SafetyFeatures._trusted_domains and domain in SafetyFeatures._trusted_domains:
            return 0.0

        suspicious_tlds = {item.lower().lstrip(".") for item in SafetyFeatures._list(cfg.get("suspicious_tlds"))}
        free_webmail = SafetyFeatures._set(cfg.get("free_webmail_domains"))
        disposable = SafetyFeatures._disposable_domains or set()
        weights = SafetyFeatures._mapping(cfg.get("sender_reputation_weights"))

        tld = domain.rsplit(".", 1)[-1] if "." in domain else domain
        age_score = clamp_score(cfg.get("new_tld_score"), default=0.70) if tld in suspicious_tlds else clamp_score(cfg.get("normal_tld_score"), default=0.30)
        reputation_score = SafetyFeatures._get_domain_reputation(domain)
        entropy_score = clamp_score(SafetyFeatures._url_entropy(domain) * coerce_float(cfg.get("domain_entropy_scale"), 2.0, minimum=0.0))
        subdomain_score = clamp_score(SafetyFeatures._num_subdomains(domain) / coerce_float(cfg.get("subdomain_suspicious_count"), 5.0, minimum=1.0))
        disposable_score = 1.0 if domain in disposable else 0.0
        webmail_score = clamp_score(cfg.get("free_webmail_sender_score"), default=0.15) if domain in free_webmail else 0.0

        combined_score = weighted_average(
            {
                "age": age_score,
                "reputation": 1.0 - reputation_score,
                "entropy": entropy_score,
                "subdomains": subdomain_score,
                "disposable": disposable_score,
                "webmail": webmail_score,
            },
            weights,
        )

        logger.info("Sender reputation analysis: %s", safe_log_payload(
            "sender_reputation_analysis",
            {
                "domain": domain,
                "score": combined_score,
                "risk_level": categorize_risk(combined_score),
                "domain_fingerprint": fingerprint(domain),
            },
        ))
        return combined_score

    @staticmethod
    def _unusual_time_score(timestamp: float) -> float:
        """Detect off-hour emails using configured local-hour policy."""

        if not timestamp:
            return 0.0
        cfg = SafetyFeatures._cfg()
        start = coerce_int(cfg.get("off_hours_start"), 21, minimum=0, maximum=23)
        end = coerce_int(cfg.get("off_hours_end"), 6, minimum=0, maximum=23)
        dt = datetime.fromtimestamp(float(timestamp))
        if start > end:
            return 1.0 if dt.hour >= start or dt.hour < end else 0.0
        return 1.0 if start <= dt.hour < end else 0.0

    # ==============================
    # 2. Extracting URL features
    # ==============================

    @staticmethod
    def _url_entropy(url: str) -> float:
        """Calculate normalized Shannon entropy of URL characters."""

        text = normalize_text(url, max_length=coerce_int(SafetyFeatures._cfg().get("max_url_length"), 2048), lowercase=True)
        if not text:
            return 0.0
        freq: Dict[str, int] = defaultdict(int)
        for char in text:
            freq[char] += 1
        entropy = 0.0
        total = len(text)
        for count in freq.values():
            p = count / total
            entropy -= p * math.log2(p)
        return clamp_score(entropy / 8.0)

    @staticmethod
    def _num_subdomains(url: str) -> float:
        """Count the number of subdomains in a URL or hostname."""

        domain = SafetyFeatures._safe_domain_from_url(url)
        if not domain:
            return 0.0
        if SafetyFeatures._is_ip_literal(domain):
            return 0.0
        parts = [part for part in domain.split(".") if part and part != "www"]
        return float(max(len(parts) - 2, 0))

    @staticmethod
    def _https_used(url: str) -> float:
        """Check if URL uses HTTPS protocol."""

        bundle = SafetyFeatures._safe_url_bundle(url)
        return 1.0 if bundle and bundle.scheme == "https" else 0.0

    @staticmethod
    def _url_redirect_count(url: str) -> float:
        """Estimate redirect likelihood through URL parameters and path hints."""

        cfg = SafetyFeatures._cfg()
        redirect_params = {normalize_text(item, lowercase=True) for item in SafetyFeatures._list(cfg.get("redirect_query_keys"))}
        redirect_path_terms = {normalize_text(item, lowercase=True) for item in SafetyFeatures._list(cfg.get("redirect_path_terms"))}
        normalized = normalize_text(url, max_length=coerce_int(cfg.get("max_url_length"), 2048), lowercase=True)
        parsed = urlparse(normalized if re.match(r"^[a-z][a-z0-9+.-]*://", normalized) else f"https://{normalized}")
        hits = 0.0
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            key_l = key.lower()
            value_l = value.lower()
            if key_l in redirect_params or any(term and term in key_l for term in redirect_params):
                hits += 1.0
            if any(value_l.startswith(prefix) for prefix in ("http://", "https://", "//")):
                hits += 1.0
        path_l = parsed.path.lower()
        if any(term and term in path_l for term in redirect_path_terms):
            hits += 1.0
        return clamp_score(hits / coerce_float(cfg.get("redirect_hit_cap"), 5.0, minimum=1.0))

    @staticmethod
    def _special_char_count(url: str) -> float:
        """Count percentage of suspicious special characters in URL."""

        text = normalize_text(url, max_length=coerce_int(SafetyFeatures._cfg().get("max_url_length"), 2048))
        if not text:
            return 0.0
        allowed_pattern = r"[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]"
        suspicious = [char for char in text if not re.match(allowed_pattern, char)]
        return clamp_score(len(suspicious) / len(text))

    @staticmethod
    def _contains_ip_address(url: str) -> float:
        domain = SafetyFeatures._safe_domain_from_url(url)
        return 1.0 if SafetyFeatures._is_ip_literal(domain) or SafetyFeatures._IP_LITERAL_RE.search(str(url or "")) else 0.0

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

    @staticmethod
    def _private_host_score(url: str) -> float:
        bundle = SafetyFeatures._safe_url_bundle(url)
        return 1.0 if bundle and bundle.is_private_host else 0.0

    @staticmethod
    def _credential_in_url_score(url: str) -> float:
        normalized = normalize_text(url, max_length=coerce_int(SafetyFeatures._cfg().get("max_url_length"), 2048))
        parsed = urlparse(normalized if re.match(r"^[a-z][a-z0-9+.-]*://", normalized) else f"https://{normalized}")
        return 1.0 if parsed.username or parsed.password or "@" in parsed.netloc.rsplit("@", 1)[0] else 0.0

    @staticmethod
    def _url_shortener_score(url: str) -> float:
        domain = SafetyFeatures._safe_domain_from_url(url)
        shorteners = SafetyFeatures._set(SafetyFeatures._cfg().get("url_shortener_domains"))
        return 1.0 if domain in shorteners else 0.0

    @staticmethod
    def _suspicious_tld_score(url: str) -> float:
        domain = SafetyFeatures._safe_domain_from_url(url)
        if not domain or "." not in domain:
            return 0.0
        tld = domain.rsplit(".", 1)[-1]
        return 1.0 if tld in {item.lower().lstrip(".") for item in SafetyFeatures._list(SafetyFeatures._cfg().get("suspicious_tlds"))} else 0.0

    @staticmethod
    def _path_depth_score(url: str) -> float:
        cfg = SafetyFeatures._cfg()
        normalized = normalize_text(url, max_length=coerce_int(cfg.get("max_url_length"), 2048))
        parsed = urlparse(normalized if re.match(r"^[a-z][a-z0-9+.-]*://", normalized) else f"https://{normalized}")
        depth = len([segment for segment in parsed.path.split("/") if segment])
        return clamp_score(depth / coerce_float(cfg.get("path_depth_suspicious_count"), 6.0, minimum=1.0))

    @staticmethod
    def _query_risk_score(url: str) -> float:
        cfg = SafetyFeatures._cfg()
        normalized = normalize_text(url, max_length=coerce_int(cfg.get("max_url_length"), 2048))
        parsed = urlparse(normalized if re.match(r"^[a-z][a-z0-9+.-]*://", normalized) else f"https://{normalized}")
        sensitive_names = {normalize_text(item, lowercase=True) for item in SafetyFeatures._list(cfg.get("sensitive_query_keys"))}
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        if not pairs:
            return 0.0
        hits = 0.0
        for key, value in pairs:
            key_l = key.lower()
            if key_l in sensitive_names or get_sensitive_key_regex().search(key_l):
                hits += 1.0
            if len(value) >= coerce_int(cfg.get("long_query_value_length"), 80, minimum=8):
                hits += 0.5
        return clamp_score(hits / coerce_float(cfg.get("query_hit_cap"), 5.0, minimum=1.0))

    @staticmethod
    def _brand_impersonation_score(text_or_url: str) -> float:
        cfg = SafetyFeatures._cfg()
        brands = [normalize_text(item, lowercase=True) for item in SafetyFeatures._list(cfg.get("protected_brand_terms"))]
        if not brands:
            return 0.0
        text = normalize_text(text_or_url, max_length=coerce_int(cfg.get("max_text_length"), 4096), lowercase=True)
        domain = SafetyFeatures._safe_domain_from_url(text) if "." in text else ""
        score = 0.0
        
        # Check for brand terms in the raw text
        for brand in brands:
            brand_compact = re.sub(r"[^a-z0-9]", "", brand)
            text_compact = re.sub(r"[^a-z0-9]", "", text)
            if brand in text:
                score = max(score, 0.65)
            if brand_compact and brand_compact in text_compact:
                score = max(score, 0.75)
        
        # Check domains of any URLs present in the text
        urls = SafetyFeatures._extract_links_from_text(text)
        for url in urls:
            domain = SafetyFeatures._safe_domain_from_url(url)
            if not domain:
                continue
            for brand in brands:
                if brand in domain:
                    score = max(score, 0.85)   # Higher confidence when brand appears inside a domain
                elif brand_compact in re.sub(r"[^a-z0-9]", "", domain): # pyright: ignore[reportPossiblyUnboundVariable]
                    score = max(score, 0.85)
        
        return clamp_score(score)

    @staticmethod
    def _punycode_or_homograph_score(url: str) -> float:
        text = normalize_text(url, max_length=coerce_int(SafetyFeatures._cfg().get("max_url_length"), 2048), lowercase=True)
        domain = SafetyFeatures._safe_domain_from_url(text)
        if "xn--" in domain:
            return 1.0
        ascii_chars = sum(1 for char in text if ord(char) < 128)
        return 0.75 if text and ascii_chars != len(text) else 0.0

    # ==============================
    # 3. Public feature maps/vectors
    # ==============================

    def extract_email_feature_map(self, email: Mapping[str, Any]) -> Dict[str, float]:
        """Return named email features with audit-safe validation."""

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
            return {
                "from_length": float(len(normalize_text(from_header, max_length=512))),
                "subject_length": float(len(normalize_text(subject, max_length=512))),
                "suspicious_keyword_score": self._contains_suspicious_keywords(subject),
                "link_count": float(len(links)),
                "urgent_language_score": self._contains_urgent_language(body),
                "attachment_present": self._contains_attachment(dict(email)),
                "attachment_risk_score": self._attachment_risk_score(email),
                "domain_mismatch_score": self._domain_mismatch_score(dict(email)),
                "reply_to_mismatch_score": self._reply_to_mismatch_score(email),
                "avg_url_length": self._avg_url_length(links),
                "normalized_url_length_score": self._normalized_url_length_score(links),
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
                context={"email": sanitize_for_logging(dict(email))},
                error_type=SecurityErrorType.UNSAFE_EXECUTION_ATTEMPT,
                severity=SecuritySeverity.HIGH,
            ) from exc

    def extract_email_feature_vector(self, email: Mapping[str, Any], *, legacy: bool = True) -> List[float]:
        """Return email feature vector. Legacy order is retained for AdaptiveSecurity."""

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
        risk_score = weighted_average(risk_features, self._mapping(self.sf_config.get("email_risk_weights")))
        indicators = [name for name, score in risk_features.items() if clamp_score(score) >= coerce_float(self.sf_config.get("indicator_threshold"), 0.5)]
        return FeatureExtractionResult(
            source_type="email",
            feature_map=features,
            feature_vector=self.extract_email_feature_vector(email, legacy=True),
            risk_score=risk_score,
            risk_level=categorize_risk(risk_score),
            decision=threshold_decision(risk_score, block_threshold=self.sf_config.get("block_threshold"), review_threshold=self.sf_config.get("review_threshold")),
            indicators=indicators,
            metadata={"source_fingerprint": fingerprint(sanitize_for_logging(dict(email)))},
            timestamp=utc_iso(),
        )

    def extract_url_feature_map(self, url: str) -> Dict[str, float]:
        """Return named URL features for phishing and cyber-safety analysis."""

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
            num_subdomains = self._num_subdomains(text)
            return {
                "url_length": float(len(text)),
                "normalized_url_length_score": self._normalized_url_length_score(text),
                "url_entropy": self._url_entropy(text),
                "num_subdomains": float(num_subdomains),
                "subdomain_score": clamp_score(num_subdomains / coerce_float(self.sf_config.get("subdomain_suspicious_count"), 5.0, minimum=1.0)),
                "contains_ip": self._contains_ip_address(text),
                "https_used": self._https_used(text),
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
        risk_score = weighted_average(risk_features, self._mapping(self.sf_config.get("url_risk_weights")))
        indicators = [name for name, score in risk_features.items() if clamp_score(score) >= coerce_float(self.sf_config.get("indicator_threshold"), 0.5)]
        return FeatureExtractionResult(
            source_type="url",
            feature_map=features,
            feature_vector=self.extract_url_feature_vector(url, legacy=True),
            risk_score=risk_score,
            risk_level=categorize_risk(risk_score),
            decision=threshold_decision(risk_score, block_threshold=self.sf_config.get("block_threshold"), review_threshold=self.sf_config.get("review_threshold")),
            indicators=indicators,
            metadata={"url_fingerprint": fingerprint(url), "domain": self._safe_domain_from_url(url)},
            timestamp=utc_iso(),
        )

    def get_feature_schema(self) -> Dict[str, List[str]]:
        return {
            "email_legacy": [
                "from_length",
                "subject_length",
                "suspicious_keyword_score",
                "link_count",
                "urgent_language_score",
                "attachment_present",
                "domain_mismatch_score",
                "avg_url_length",
                "ssl_cert_score",
                "unusual_sender_score",
                "unusual_time_score",
            ],
            "url_legacy": [
                "url_length",
                "url_entropy",
                "num_subdomains",
                "contains_ip",
                "https_used",
                "redirect_score",
                "domain_age_score_external",
                "special_char_score",
            ],
            "email_configured": self._list(self.sf_config.get("feature_order", {}).get("email")),
            "url_configured": self._list(self.sf_config.get("feature_order", {}).get("url")),
        }

    def clear_caches(self) -> None:
        self._domain_reputation_cache.clear()
        self.__class__._trusted_domains = None
        self.__class__._malicious_domains = None
        self.__class__._disposable_domains = None
        self.__class__._load_domain_lists()


if __name__ == "__main__":
    print("\n=== Running Safety Features ===\n")
    printer.status("TEST", "Safety Features initialized", "info")

    features = SafetyFeatures()

    test_email = {
        "from": "Security Team <security@paypa1-login.example>",
        "reply_to": "support@shady-site.cc",
        "subject": "Urgent account suspended - verify now",
        "body": "Your account is suspended. Visit https://bit.ly/reset?token=secret123 immediately.",
        "links": ["http://bit.ly/reset?token=secret123", "https://trusted.org/help"],
        "attachments": [{"filename": "invoice.xlsm", "size_bytes": 204800, "has_macro": True}],
        "timestamp": datetime(2026, 4, 30, 23, 30).timestamp(),
    }

    test_url = "http://xn--paypa1-login.example/secure/login?redirect=https://evil.test&token=secret123"

    email_result = features.assess_email_risk(test_email)
    url_result = features.assess_url_risk(test_url)
    schema = features.get_feature_schema()

    assert len(features.extract_email_feature_vector(test_email, legacy=True)) == 11
    assert len(features.extract_url_feature_vector(test_url, legacy=True)) == 8
    assert 0.0 <= email_result.risk_score <= 1.0
    assert 0.0 <= url_result.risk_score <= 1.0
    assert "token=secret123" not in stable_json(email_result.to_dict())
    assert "token=secret123" not in stable_json(url_result.to_dict())
    assert schema["email_legacy"][0] == "from_length"
    assert features._contains_suspicious_keywords("verify your password urgently") > 0.0
    assert features._https_used("https://example.com") == 1.0
    assert features._contains_ip_address("http://192.168.1.1/login") == 1.0

    printer.status("TEST", f"Email risk: {email_result.risk_level} ({email_result.risk_score:.3f})", "info")
    printer.status("TEST", f"URL risk: {url_result.risk_level} ({url_result.risk_score:.3f})", "info")

    print("\n=== Test ran successfully ===\n")
