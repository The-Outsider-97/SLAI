from __future__ import annotations

"""
Production-grade security and safety guardrails for the browser subsystem.

This module is a protective layer for the browser agent and the user. It does
not attempt to evade site protections, bypass CAPTCHA challenges, defeat bot
controls, or force access to restricted content. Instead, it detects risky or
blocked browsing states, normalizes them into the shared browser error/result
contract, records safe telemetry, and gives the BrowserAgent a clear decision:
continue, warn, retry later, or stop.

Scope
-----
SecurityFeatures protects browser automation flows by providing:

- CAPTCHA, bot-detection, rate-limit, interstitial, and permission-wall checks.
- URL policy checks for schemes, domains, private/internal hosts, and downloads.
- Page/content checks for credential prompts, suspicious redirects, unsafe
  downloads, mixed trust signals, and sensitive data exposure in telemetry.
- Action checks for potentially sensitive browser actions such as paste, file
  navigation, cross-origin actions, or JavaScript URLs.
- Redacted, structured security reports that can be logged, stored in browser
  memory, attached to action results, and consumed by workflow orchestration.
- Backwards compatibility with the existing BrowserAgent call pattern:
  ``SecurityFeatures.detect_captcha(driver)``.

Design principles
-----------------
1. Protective, not evasive: detected CAPTCHA/bot/rate-limit states produce a
   block/warn decision rather than bypass instructions or evasion behavior.
2. Integration-first: shared browser errors and helpers are reused for result
   construction, serialization, redaction, timing, URL normalization, page
   snapshots, and retry/backoff timing.
3. Config-driven: policy thresholds, allow/block lists, indicators, limits, and
   diagnostics belong in ``browser_config.yaml`` under ``security_features``.
4. Deterministic contracts: scans return stable dataclasses/dictionaries for
   tests, logging, metrics, workflow decisions, and future learning modules.
5. Expandable: new detectors can be added as small methods without duplicating
   URL parsing, result payload construction, redaction, or memory recording.

Local imports are intentionally direct. They are not wrapped in try/except so
packaging or path problems fail clearly during development and deployment.
"""

import ipaddress
import re
import time as time_module

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

from .utils.config_loader import load_global_config, get_config_section
from .utils.browser_errors import *
from .utils.Browser_helpers import *
from .browser_memory import BrowserMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Security Features")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Constants and defaults
# ---------------------------------------------------------------------------
SECURITY_SCHEMA_VERSION = "1.0"
DEFAULT_SECURITY_NAMESPACE = "security"

SECURITY_ACTION_SCAN = "security_scan"
SECURITY_ACTION_ASSESS_URL = "assess_url"
SECURITY_ACTION_ASSESS_ACTION = "assess_action"
SECURITY_ACTION_ENFORCE = "security_enforce"

DEFAULT_ALLOWED_SCHEMES: Tuple[str, ...] = ("http", "https")
DEFAULT_BLOCKED_SCHEMES: Tuple[str, ...] = (
    "javascript",
    "vbscript",
    "data",
    "file",
    "ftp",
    "chrome",
    "chrome-extension",
    "edge",
    "about",
)

DEFAULT_CAPTCHA_INDICATORS: Tuple[str, ...] = (
    "captcha",
    "recaptcha",
    "hcaptcha",
    "verify you are human",
    "prove you are human",
    "robot check",
    "are you a robot",
    "human verification",
    "security check",
    "checking your browser",
    "unusual traffic",
    "automated queries",
    "cf-challenge",
    "turnstile",
)

DEFAULT_BOT_BLOCK_INDICATORS: Tuple[str, ...] = (
    "access denied",
    "request blocked",
    "temporarily blocked",
    "blocked due to suspicious activity",
    "automated access",
    "bot detection",
    "bot protection",
    "anti-bot",
    "traffic from your network",
    "your request has been blocked",
    "forbidden",
    "403 forbidden",
)

DEFAULT_RATE_LIMIT_INDICATORS: Tuple[str, ...] = (
    "rate limit",
    "too many requests",
    "429",
    "slow down",
    "try again later",
    "temporarily unavailable",
    "temporarily rate limited",
    "quota exceeded",
)

DEFAULT_LOGIN_WALL_INDICATORS: Tuple[str, ...] = (
    "sign in to continue",
    "log in to continue",
    "login required",
    "please sign in",
    "authentication required",
    "session expired",
)

DEFAULT_SENSITIVE_VALUE_PATTERNS: Tuple[str, ...] = (
    r"(?i)api[_-]?key\s*[:=]\s*[A-Za-z0-9_\-\.]{16,}",
    r"(?i)secret\s*[:=]\s*[A-Za-z0-9_\-\.]{16,}",
    r"(?i)password\s*[:=]\s*\S{6,}",
    r"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9_\-\.]{16,}",
    r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+",
)

DEFAULT_DOWNLOAD_EXTENSIONS: Tuple[str, ...] = (
    ".exe",
    ".msi",
    ".dmg",
    ".pkg",
    ".deb",
    ".rpm",
    ".apk",
    ".bat",
    ".cmd",
    ".com",
    ".scr",
    ".ps1",
    ".vbs",
    ".js",
    ".jar",
    ".sh",
    ".appimage",
)

DEFAULT_HIGH_RISK_TLDS: Tuple[str, ...] = ()
DEFAULT_INTERNAL_HOSTNAMES: Tuple[str, ...] = (
    "localhost",
    "ip6-localhost",
    "ip6-loopback",
    "metadata.google.internal",
)

DEFAULT_SECURITY_FEATURES_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "schema_version": SECURITY_SCHEMA_VERSION,
    "default_decision_on_error": "warn",
    "block_on_captcha": True,
    "block_on_bot_detection": True,
    "block_on_rate_limit": False,
    "block_on_unsafe_url": True,
    "block_private_network_urls": True,
    "block_local_file_urls": True,
    "warn_on_login_wall": True,
    "warn_on_sensitive_content": True,
    "risk_threshold_warn": 0.35,
    "risk_threshold_block": 0.75,
    "max_scan_text_chars": 250000,
    "allowed_schemes": list(DEFAULT_ALLOWED_SCHEMES),
    "blocked_schemes": list(DEFAULT_BLOCKED_SCHEMES),
    "allowed_domains": [],
    "blocked_domains": [],
    "blocked_domain_suffixes": [],
    "high_risk_tlds": list(DEFAULT_HIGH_RISK_TLDS),
    "internal_hostnames": list(DEFAULT_INTERNAL_HOSTNAMES),
    "blocked_download_extensions": list(DEFAULT_DOWNLOAD_EXTENSIONS),
    "indicators": {
        "captcha": list(DEFAULT_CAPTCHA_INDICATORS),
        "bot_block": list(DEFAULT_BOT_BLOCK_INDICATORS),
        "rate_limit": list(DEFAULT_RATE_LIMIT_INDICATORS),
        "login_wall": list(DEFAULT_LOGIN_WALL_INDICATORS),
        "sensitive_patterns": list(DEFAULT_SENSITIVE_VALUE_PATTERNS),
    },
    "backoff": {
        "base_delay": 1.0,
        "max_delay": 60.0,
        "multiplier": 2.0,
        "jitter": 0.0,
    },
    "diagnostics": {
        "include_page_snapshot": False,
        "include_screenshot_on_block": False,
        "include_text_evidence": True,
        "max_evidence_chars": 500,
        "redact_urls": True,
        "redact_text": True,
        "include_traceback_on_error": False,
    },
    "memory": {
        "enabled": True,
        "namespace": DEFAULT_SECURITY_NAMESPACE,
        "record_allowed": False,
        "record_warnings": True,
        "record_blocks": True,
    },
}


class SecurityDecisionStatus(str, Enum):
    """Decision status emitted by the security layer."""

    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


class SecurityFindingCategory(str, Enum):
    """Canonical categories for browser security findings."""

    CAPTCHA = "captcha"
    BOT_DETECTION = "bot_detection"
    RATE_LIMIT = "rate_limit"
    LOGIN_WALL = "login_wall"
    URL_POLICY = "url_policy"
    DOMAIN_POLICY = "domain_policy"
    PRIVATE_NETWORK = "private_network"
    DOWNLOAD = "download"
    SENSITIVE_CONTENT = "sensitive_content"
    INTERSTITIAL = "interstitial"
    PERMISSION = "permission"
    MIXED_TRUST = "mixed_trust"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SecurityOptions:
    """Resolved runtime policy for security checks."""

    enabled: bool = True
    default_decision_on_error: str = "warn"
    block_on_captcha: bool = True
    block_on_bot_detection: bool = True
    block_on_rate_limit: bool = False
    block_on_unsafe_url: bool = True
    block_private_network_urls: bool = True
    block_local_file_urls: bool = True
    warn_on_login_wall: bool = True
    warn_on_sensitive_content: bool = True
    risk_threshold_warn: float = 0.35
    risk_threshold_block: float = 0.75
    max_scan_text_chars: int = 250000
    allowed_schemes: Tuple[str, ...] = DEFAULT_ALLOWED_SCHEMES
    blocked_schemes: Tuple[str, ...] = DEFAULT_BLOCKED_SCHEMES
    allowed_domains: Tuple[str, ...] = ()
    blocked_domains: Tuple[str, ...] = ()
    blocked_domain_suffixes: Tuple[str, ...] = ()
    high_risk_tlds: Tuple[str, ...] = DEFAULT_HIGH_RISK_TLDS
    internal_hostnames: Tuple[str, ...] = DEFAULT_INTERNAL_HOSTNAMES
    blocked_download_extensions: Tuple[str, ...] = DEFAULT_DOWNLOAD_EXTENSIONS
    captcha_indicators: Tuple[str, ...] = DEFAULT_CAPTCHA_INDICATORS
    bot_block_indicators: Tuple[str, ...] = DEFAULT_BOT_BLOCK_INDICATORS
    rate_limit_indicators: Tuple[str, ...] = DEFAULT_RATE_LIMIT_INDICATORS
    login_wall_indicators: Tuple[str, ...] = DEFAULT_LOGIN_WALL_INDICATORS
    sensitive_patterns: Tuple[str, ...] = DEFAULT_SENSITIVE_VALUE_PATTERNS
    backoff_base_delay: float = 1.0
    backoff_max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    backoff_jitter: float = 0.0
    include_page_snapshot: bool = False
    include_screenshot_on_block: bool = False
    include_text_evidence: bool = True
    max_evidence_chars: int = 500
    redact_urls: bool = True
    redact_text: bool = True
    include_traceback_on_error: bool = False
    memory_enabled: bool = True
    memory_namespace: str = DEFAULT_SECURITY_NAMESPACE
    memory_record_allowed: bool = False
    memory_record_warnings: bool = True
    memory_record_blocks: bool = True

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "SecurityOptions":
        cfg = merge_dicts(DEFAULT_SECURITY_FEATURES_CONFIG, dict(config or {}), deep=True)
        indicators = dict(cfg.get("indicators") or {})
        backoff = dict(cfg.get("backoff") or {})
        diagnostics = dict(cfg.get("diagnostics") or {})
        memory_cfg = dict(cfg.get("memory") or {})

        return cls(
            enabled=coerce_bool(cfg.get("enabled"), default=True),
            default_decision_on_error=str(cfg.get("default_decision_on_error") or "warn").lower().strip(),
            block_on_captcha=coerce_bool(cfg.get("block_on_captcha"), default=True),
            block_on_bot_detection=coerce_bool(cfg.get("block_on_bot_detection"), default=True),
            block_on_rate_limit=coerce_bool(cfg.get("block_on_rate_limit"), default=False),
            block_on_unsafe_url=coerce_bool(cfg.get("block_on_unsafe_url"), default=True),
            block_private_network_urls=coerce_bool(cfg.get("block_private_network_urls"), default=True),
            block_local_file_urls=coerce_bool(cfg.get("block_local_file_urls"), default=True),
            warn_on_login_wall=coerce_bool(cfg.get("warn_on_login_wall"), default=True),
            warn_on_sensitive_content=coerce_bool(cfg.get("warn_on_sensitive_content"), default=True),
            risk_threshold_warn=coerce_float(cfg.get("risk_threshold_warn"), default=0.35, minimum=0.0, maximum=1.0),
            risk_threshold_block=coerce_float(cfg.get("risk_threshold_block"), default=0.75, minimum=0.0, maximum=1.0),
            max_scan_text_chars=coerce_int(cfg.get("max_scan_text_chars"), default=250000, minimum=1000),
            allowed_schemes=tuple(_normalize_string_list(cfg.get("allowed_schemes"))) or DEFAULT_ALLOWED_SCHEMES,
            blocked_schemes=tuple(_normalize_string_list(cfg.get("blocked_schemes"))) or DEFAULT_BLOCKED_SCHEMES,
            allowed_domains=tuple(_normalize_domain_list(cfg.get("allowed_domains"))),
            blocked_domains=tuple(_normalize_domain_list(cfg.get("blocked_domains"))),
            blocked_domain_suffixes=tuple(_normalize_domain_list(cfg.get("blocked_domain_suffixes"))),
            high_risk_tlds=tuple(_normalize_string_list(cfg.get("high_risk_tlds"))),
            internal_hostnames=tuple(_normalize_domain_list(cfg.get("internal_hostnames"))) or DEFAULT_INTERNAL_HOSTNAMES,
            blocked_download_extensions=tuple(_normalize_extensions(cfg.get("blocked_download_extensions"))) or DEFAULT_DOWNLOAD_EXTENSIONS,
            captcha_indicators=tuple(_normalize_indicators(indicators.get("captcha"))) or DEFAULT_CAPTCHA_INDICATORS,
            bot_block_indicators=tuple(_normalize_indicators(indicators.get("bot_block"))) or DEFAULT_BOT_BLOCK_INDICATORS,
            rate_limit_indicators=tuple(_normalize_indicators(indicators.get("rate_limit"))) or DEFAULT_RATE_LIMIT_INDICATORS,
            login_wall_indicators=tuple(_normalize_indicators(indicators.get("login_wall"))) or DEFAULT_LOGIN_WALL_INDICATORS,
            sensitive_patterns=tuple(_normalize_indicators(indicators.get("sensitive_patterns"))) or DEFAULT_SENSITIVE_VALUE_PATTERNS,
            backoff_base_delay=coerce_float(backoff.get("base_delay"), default=1.0, minimum=0.0, maximum=3600.0),
            backoff_max_delay=coerce_float(backoff.get("max_delay"), default=60.0, minimum=0.0, maximum=3600.0),
            backoff_multiplier=coerce_float(backoff.get("multiplier"), default=2.0, minimum=1.0, maximum=10.0),
            backoff_jitter=coerce_float(backoff.get("jitter"), default=0.0, minimum=0.0, maximum=1.0),
            include_page_snapshot=coerce_bool(diagnostics.get("include_page_snapshot"), default=False),
            include_screenshot_on_block=coerce_bool(diagnostics.get("include_screenshot_on_block"), default=False),
            include_text_evidence=coerce_bool(diagnostics.get("include_text_evidence"), default=True),
            max_evidence_chars=coerce_int(diagnostics.get("max_evidence_chars"), default=500, minimum=0, maximum=10000),
            redact_urls=coerce_bool(diagnostics.get("redact_urls"), default=True),
            redact_text=coerce_bool(diagnostics.get("redact_text"), default=True),
            include_traceback_on_error=coerce_bool(diagnostics.get("include_traceback_on_error"), default=False),
            memory_enabled=coerce_bool(memory_cfg.get("enabled"), default=True),
            memory_namespace=str(memory_cfg.get("namespace") or DEFAULT_SECURITY_NAMESPACE).strip() or DEFAULT_SECURITY_NAMESPACE,
            memory_record_allowed=coerce_bool(memory_cfg.get("record_allowed"), default=False),
            memory_record_warnings=coerce_bool(memory_cfg.get("record_warnings"), default=True),
            memory_record_blocks=coerce_bool(memory_cfg.get("record_blocks"), default=True),
        )


@dataclass(frozen=True)
class SecurityFinding:
    """One detected security signal."""

    category: str
    severity: str
    message: str
    score: float = 0.0
    signal: str = ""
    source: str = ""
    evidence: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return redact_mapping(prune_none(asdict(self)))


@dataclass(frozen=True)
class SecurityDecision:
    """Final decision returned by a security assessment."""

    status: str
    allowed: bool
    reason: str
    risk_score: float = 0.0
    severity: str = "low"
    categories: Tuple[str, ...] = ()
    recommendations: Tuple[str, ...] = ()
    code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return redact_mapping(prune_none(asdict(self)))


@dataclass(frozen=True)
class SecurityReport:
    """Structured security report for page/action/url assessment."""

    decision: SecurityDecision
    findings: Tuple[SecurityFinding, ...] = ()
    action: str = SECURITY_ACTION_SCAN
    url: str = ""
    domain: str = ""
    page_title: str = ""
    ready_state: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    page: Optional[Dict[str, Any]] = None
    correlation_id: str = field(default_factory=lambda: new_correlation_id("sec"))
    scanned_at: str = field(default_factory=utc_now_iso)
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return redact_mapping(prune_none(asdict(self)))

    def to_result(self) -> Dict[str, Any]:
        message = self.decision.reason or "Security assessment completed"
        data = self.to_dict()
        if self.decision.allowed:
            return success_result(
                action=self.action,
                message=message,
                data=data,
                duration_ms=self.duration_ms,
                correlation_id=self.correlation_id,
            )
        error = _error_for_decision(self.decision, context={"report": data})
        return error_result(
            action=self.action,
            message=message,
            error=error,
            metadata={"security_report": data},
            duration_ms=self.duration_ms,
            correlation_id=self.correlation_id,
        )


# ---------------------------------------------------------------------------
# Backoff compatibility
# ---------------------------------------------------------------------------
def exponential_backoff(retries: int) -> float:
    """Backward-compatible exponential backoff helper.

    BrowserAgent currently imports this free function. The default remains the
    original ``2 ** retries`` behavior while gaining input normalization.
    """

    attempt = max(0, int(retries or 0))
    delay = float(2 ** attempt)
    time_module.sleep(delay)
    return delay


def configured_backoff(retries: int, *, options: Optional[SecurityOptions] = None) -> float:
    """Config-backed backoff helper for security retry/wait decisions."""

    opts = options or SecurityOptions.from_config(None)
    delay = calculate_backoff_delay(
        attempt_index=max(0, int(retries or 0)),
        base_delay=opts.backoff_base_delay,
        max_delay=opts.backoff_max_delay,
        multiplier=opts.backoff_multiplier,
        jitter=opts.backoff_jitter,
    )
    time_module.sleep(delay)
    return delay


# ---------------------------------------------------------------------------
# Main security layer
# ---------------------------------------------------------------------------
class SecurityFeatures:
    """Protective security layer for BrowserAgent and browser functions."""

    def __init__(self, driver: Any = None, memory: Optional[BrowserMemory] = None):
        self.config = load_global_config()
        self.security_config = get_config_section("security_features") or {}
        self.driver = driver
        self.memory = memory if memory is not None else BrowserMemory()
        self.options = SecurityOptions.from_config(self.security_config)
        self.last_report: Optional[SecurityReport] = None
        logger.info("Security Features initialized.")

    # ------------------------------------------------------------------
    # Backwards-compatible static checks
    # ------------------------------------------------------------------
    @staticmethod
    def detect_captcha(driver: Any) -> bool:
        """Check for CAPTCHA challenges in page content.

        This static method intentionally remains available because BrowserAgent
        calls ``SecurityFeatures.detect_captcha(driver)`` without instantiating
        the security layer.
        """

        try:
            if driver is None:
                return False
            return bool(detect_captcha(driver))
        except Exception:
            try:
                page_source = str(getattr(driver, "page_source", "") or "").lower()
                page_text = str(get_body_text(driver, max_length=50000) or "").lower()
                url = str(get_current_url(driver) or "").lower()
                return _contains_any("\n".join([page_source, page_text, url]), DEFAULT_CAPTCHA_INDICATORS)
            except Exception:
                return False

    @staticmethod
    def detect_captcha_text(text: Any) -> bool:
        """Check text/html for CAPTCHA wording."""

        return _contains_any(str(text or ""), DEFAULT_CAPTCHA_INDICATORS)

    # ------------------------------------------------------------------
    # Driver and config handling
    # ------------------------------------------------------------------
    def attach_driver(self, driver: Any) -> "SecurityFeatures":
        self.driver = driver
        return self

    def set_driver(self, driver: Any) -> "SecurityFeatures":
        return self.attach_driver(driver)

    def detach_driver(self) -> None:
        self.driver = None

    def reload_config(self) -> "SecurityFeatures":
        self.config = load_global_config()
        self.security_config = get_config_section("security_features") or {}
        self.options = SecurityOptions.from_config(self.security_config)
        return self

    # ------------------------------------------------------------------
    # Public assessment APIs
    # ------------------------------------------------------------------
    def assess_navigation(self, url: str, *, action: str = "navigate", metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Assess whether navigation to a URL should be allowed."""

        report = self.scan_url(url, action=action, metadata=metadata)
        self._record_report(report)
        return report.to_result()

    def scan_url(self, url: str, *, action: str = SECURITY_ACTION_ASSESS_URL, metadata: Optional[Mapping[str, Any]] = None) -> SecurityReport:
        """Run URL policy checks and return a structured report."""

        start = monotonic_ms()
        correlation_id = new_correlation_id("sec-url")
        if not self.options.enabled:
            return self._build_report(
                action=action,
                findings=(),
                url=str(url or ""),
                metadata=metadata,
                correlation_id=correlation_id,
                duration_ms=elapsed_ms(start),
            )

        findings = self._url_findings(url)
        report = self._build_report(
            action=action,
            findings=findings,
            url=str(url or ""),
            metadata=metadata,
            correlation_id=correlation_id,
            duration_ms=elapsed_ms(start),
        )
        self.last_report = report
        return report

    def scan_current_page(self, *, action: str = SECURITY_ACTION_SCAN, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Assess the active browser page and return a BrowserAgent-style result."""

        report = self.build_current_page_report(action=action, metadata=metadata)
        self._record_report(report)
        return report.to_result()

    def build_current_page_report(self, *, action: str = SECURITY_ACTION_SCAN,
                                  metadata: Optional[Mapping[str, Any]] = None) -> SecurityReport:
        """Assess the current browser page and return a structured report."""

        start = monotonic_ms()
        correlation_id = new_correlation_id("sec-page")
        if not self.options.enabled:
            return self._build_report(
                action=action,
                findings=(),
                url=get_current_url(self.driver) if self.driver else "",
                metadata=metadata,
                correlation_id=correlation_id,
                duration_ms=elapsed_ms(start),
            )

        if self.driver is None:
            err = MissingDriverError("Security scan requires an attached driver", context={"action": action})
            finding = SecurityFinding(
                category=SecurityFindingCategory.UNKNOWN.value,
                severity="high",
                message=err.message,
                score=0.75,
                signal="missing_driver",
                source="security_features",
            )
            return self._build_report(
                action=action,
                findings=(finding,),
                metadata=metadata,
                correlation_id=correlation_id,
                duration_ms=elapsed_ms(start),
            )

        url = get_current_url(self.driver)
        title = get_page_title(self.driver)
        ready_state = get_document_ready_state(self.driver)
        text = self._scan_text_from_driver(self.driver)

        findings: List[SecurityFinding] = []
        findings.extend(self._url_findings(url))
        findings.extend(self._text_findings(text, url=url, title=title))

        page_data: Optional[Dict[str, Any]] = None
        include_screenshot = self.options.include_screenshot_on_block and any(f.score >= self.options.risk_threshold_block for f in findings)
        if self.options.include_page_snapshot or include_screenshot:
            page_data = page_snapshot_dict(
                self.driver,
                include_html=False,
                include_screenshot=include_screenshot,
                max_text=min(self.options.max_scan_text_chars, 5000),
            )

        report = self._build_report(
            action=action,
            findings=findings,
            url=url,
            page_title=title,
            ready_state=ready_state,
            page=page_data,
            metadata=metadata,
            correlation_id=correlation_id,
            duration_ms=elapsed_ms(start),
        )
        self.last_report = report
        return report

    def assess_action(self, action: str, payload: Optional[Mapping[str, Any]] = None,
                      *, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Assess a browser action payload before execution."""

        report = self.build_action_report(action=action, payload=payload, metadata=metadata)
        self._record_report(report)
        return report.to_result()

    def build_action_report(self, action: str, payload: Optional[Mapping[str, Any]] = None,
                            *, metadata: Optional[Mapping[str, Any]] = None) -> SecurityReport:
        """Build a structured report for an intended browser action."""

        start = monotonic_ms()
        normalized_action = normalize_action_name(action)
        payload_dict = dict(payload or {})
        findings: List[SecurityFinding] = []

        candidate_url = payload_dict.get("url") or payload_dict.get("href") or payload_dict.get("link")
        if candidate_url:
            findings.extend(self._url_findings(str(candidate_url)))

        if normalized_action in {"paste", "type", "enter_text", "input_text"}:
            text_value = payload_dict.get("text") or payload_dict.get("raw_input") or payload_dict.get("value")
            if text_value is not None:
                findings.extend(self._sensitive_value_findings(str(text_value), source=f"action.{normalized_action}"))

        if normalized_action in {"execute_script", "script", "javascript"}:
            findings.append(
                SecurityFinding(
                    category=SecurityFindingCategory.PERMISSION.value,
                    severity="high",
                    message="Direct script execution requested; require explicit trusted caller policy before running.",
                    score=0.70,
                    signal="script_execution_request",
                    source="action_policy",
                    metadata={"action": normalized_action},
                )
            )

        merged_metadata = merge_dicts(
            {"payload_keys": sorted(str(key) for key in payload_dict.keys())},
            metadata,
            deep=True,
        )
        report = self._build_report(
            action=normalized_action or SECURITY_ACTION_ASSESS_ACTION,
            findings=findings,
            url=str(candidate_url or get_current_url(self.driver) if self.driver else ""),
            metadata=merged_metadata,
            correlation_id=new_correlation_id("sec-act"),
            duration_ms=elapsed_ms(start),
        )
        self.last_report = report
        return report

    def guard_navigation(self, url: str, *, raise_on_block: bool = True) -> Dict[str, Any]:
        """Assess navigation and optionally raise when blocked."""

        report = self.scan_url(url, action="guard_navigation")
        self._record_report(report)
        if raise_on_block and not report.decision.allowed:
            raise _error_for_decision(report.decision, context={"report": report.to_dict()})
        return report.to_result()

    def guard_current_page(self, *, raise_on_block: bool = True) -> Dict[str, Any]:
        """Assess current page and optionally raise when blocked."""

        report = self.build_current_page_report(action="guard_current_page")
        self._record_report(report)
        if raise_on_block and not report.decision.allowed:
            raise _error_for_decision(report.decision, context={"report": report.to_dict()})
        return report.to_result()

    def should_continue(self, report_or_result: Union[SecurityReport, Mapping[str, Any], None] = None) -> bool:
        """Return whether the latest or provided security decision allows continuation."""

        if isinstance(report_or_result, SecurityReport):
            return bool(report_or_result.decision.allowed)
        if isinstance(report_or_result, Mapping):
            decision = _find_nested_decision(report_or_result)
            if isinstance(decision, Mapping):
                return bool(decision.get("allowed", True))
            return str(report_or_result.get("status") or "success") != "error"
        return self.last_report.decision.allowed if self.last_report else True

    def latest_report(self) -> Optional[Dict[str, Any]]:
        return self.last_report.to_dict() if self.last_report else None

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------
    def detect_bot_block(self, driver: Any = None) -> bool:
        text = self._scan_text_from_driver(driver or self.driver)
        return _contains_any(text, self.options.bot_block_indicators)

    def detect_rate_limit(self, driver: Any = None) -> bool:
        text = self._scan_text_from_driver(driver or self.driver)
        return _contains_any(text, self.options.rate_limit_indicators)

    def detect_login_wall(self, driver: Any = None) -> bool:
        text = self._scan_text_from_driver(driver or self.driver)
        return _contains_any(text, self.options.login_wall_indicators)

    def detect_sensitive_content(self, text: Any) -> bool:
        return bool(self._sensitive_value_findings(str(text or ""), source="text"))

    def url_is_allowed(self, url: str) -> bool:
        return self.scan_url(url).decision.allowed

    # ------------------------------------------------------------------
    # Result helpers
    # ------------------------------------------------------------------
    def safe_scan_current_page(self, *, action: str = SECURITY_ACTION_SCAN,
                               metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Scan current page and convert unexpected failures into safe results."""

        try:
            return self.scan_current_page(action=action, metadata=metadata)
        except Exception as exc:
            wrapped = wrap_browser_exception(exc, action=action, default_error_cls=BrowserSecurityError)
            if self.options.default_decision_on_error == "block":
                return wrapped.to_result(action=action)
            return error_result(action=action, message="Security scan failed", error=wrapped)

    def safe_assess_navigation(self, url: str, *, action: str = "navigate",
                               metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        try:
            return self.assess_navigation(url, action=action, metadata=metadata)
        except Exception as exc:
            wrapped = wrap_browser_exception(exc, action=action, context={"url": url}, default_error_cls=BrowserSecurityError)
            return error_result(action=action, message="Navigation security assessment failed", error=wrapped)

    # ------------------------------------------------------------------
    # Internal detectors and builders
    # ------------------------------------------------------------------
    def _url_findings(self, url: str) -> List[SecurityFinding]:
        findings: List[SecurityFinding] = []
        text_url = str(url or "").strip()
        if not text_url:
            return findings

        parsed = urlparse(text_url)
        scheme = (parsed.scheme or "").lower()
        host = (parsed.hostname or "").lower().strip(".")
        path = parsed.path or ""

        if not scheme and "://" not in text_url:
            normalized = normalize_url(text_url)
            parsed = urlparse(normalized)
            scheme = (parsed.scheme or "").lower()
            host = (parsed.hostname or "").lower().strip(".")
            path = parsed.path or path
            text_url = normalized

        if scheme in self.options.blocked_schemes:
            findings.append(
                SecurityFinding(
                    category=SecurityFindingCategory.URL_POLICY.value,
                    severity="high",
                    message=f"Blocked URL scheme: {scheme}",
                    score=0.90,
                    signal="blocked_scheme",
                    source="url_policy",
                    evidence=scheme,
                    metadata={"scheme": scheme},
                )
            )
        elif self.options.allowed_schemes and scheme and scheme not in self.options.allowed_schemes:
            findings.append(
                SecurityFinding(
                    category=SecurityFindingCategory.URL_POLICY.value,
                    severity="high",
                    message=f"URL scheme is not allowed: {scheme}",
                    score=0.85,
                    signal="scheme_not_allowed",
                    source="url_policy",
                    evidence=scheme,
                    metadata={"scheme": scheme, "allowed_schemes": list(self.options.allowed_schemes)},
                )
            )

        if scheme == "file" and self.options.block_local_file_urls:
            findings.append(
                SecurityFinding(
                    category=SecurityFindingCategory.URL_POLICY.value,
                    severity="critical",
                    message="Local file URL navigation is blocked by policy.",
                    score=0.95,
                    signal="local_file_url",
                    source="url_policy",
                    metadata={"scheme": scheme},
                )
            )

        if host:
            if self._domain_blocked(host):
                findings.append(
                    SecurityFinding(
                        category=SecurityFindingCategory.DOMAIN_POLICY.value,
                        severity="critical",
                        message=f"Domain is blocked by browser security policy: {host}",
                        score=1.0,
                        signal="blocked_domain",
                        source="domain_policy",
                        evidence=host,
                        metadata={"domain": host},
                    )
                )
            if self.options.allowed_domains and not self._domain_allowed(host):
                findings.append(
                    SecurityFinding(
                        category=SecurityFindingCategory.DOMAIN_POLICY.value,
                        severity="high",
                        message=f"Domain is not in the configured allowlist: {host}",
                        score=0.85,
                        signal="domain_not_allowlisted",
                        source="domain_policy",
                        evidence=host,
                        metadata={"domain": host, "allowed_domains": list(self.options.allowed_domains)},
                    )
                )
            if self.options.block_private_network_urls and self._is_private_or_internal_host(host):
                findings.append(
                    SecurityFinding(
                        category=SecurityFindingCategory.PRIVATE_NETWORK.value,
                        severity="high",
                        message=f"Private/internal network host is blocked by policy: {host}",
                        score=0.90,
                        signal="private_or_internal_host",
                        source="network_policy",
                        evidence=host,
                        metadata={"host": host},
                    )
                )
            if self._has_high_risk_tld(host):
                findings.append(
                    SecurityFinding(
                        category=SecurityFindingCategory.MIXED_TRUST.value,
                        severity="medium",
                        message=f"Host uses a configured high-risk TLD: {host}",
                        score=0.35,
                        signal="high_risk_tld",
                        source="domain_policy",
                        evidence=host,
                    )
                )

        download_ext = self._blocked_download_extension(path)
        if download_ext:
            findings.append(
                SecurityFinding(
                    category=SecurityFindingCategory.DOWNLOAD.value,
                    severity="high",
                    message=f"URL appears to target a blocked executable/script download: {download_ext}",
                    score=0.80,
                    signal="blocked_download_extension",
                    source="download_policy",
                    evidence=download_ext,
                    metadata={"extension": download_ext},
                )
            )

        return findings

    def _text_findings(self, text: str, *, url: str = "", title: str = "") -> List[SecurityFinding]:
        scan_text = "\n".join([url or "", title or "", text or ""])
        findings: List[SecurityFinding] = []
        findings.extend(self._indicator_findings(
            scan_text,
            indicators=self.options.captcha_indicators,
            category=SecurityFindingCategory.CAPTCHA.value,
            severity="high",
            score=0.90,
            signal="captcha_detected",
            message="CAPTCHA or human-verification challenge detected.",
            source="page_text",
        ))
        findings.extend(self._indicator_findings(
            scan_text,
            indicators=self.options.bot_block_indicators,
            category=SecurityFindingCategory.BOT_DETECTION.value,
            severity="high",
            score=0.85,
            signal="bot_or_access_block_detected",
            message="Automated browsing block or access-denied page detected.",
            source="page_text",
        ))
        findings.extend(self._indicator_findings(
            scan_text,
            indicators=self.options.rate_limit_indicators,
            category=SecurityFindingCategory.RATE_LIMIT.value,
            severity="medium",
            score=0.60,
            signal="rate_limit_detected",
            message="Rate-limit or temporary throttling message detected.",
            source="page_text",
        ))
        findings.extend(self._indicator_findings(
            scan_text,
            indicators=self.options.login_wall_indicators,
            category=SecurityFindingCategory.LOGIN_WALL.value,
            severity="medium",
            score=0.40,
            signal="login_wall_detected",
            message="Login or authentication wall detected.",
            source="page_text",
        ))
        findings.extend(self._sensitive_value_findings(scan_text, source="page_text"))
        return findings

    def _indicator_findings(self, text: str, *, indicators: Sequence[str], category: str, severity: str,
                            score: float, signal: str, message: str, source: str) -> List[SecurityFinding]:
        lowered = str(text or "").lower()
        matches: List[SecurityFinding] = []
        for indicator in indicators:
            needle = str(indicator or "").lower().strip()
            if not needle or needle not in lowered:
                continue
            matches.append(
                SecurityFinding(
                    category=category,
                    severity=severity,
                    message=message,
                    score=score,
                    signal=signal,
                    source=source,
                    evidence=self._evidence(needle),
                    metadata={"indicator": needle},
                )
            )
            break
        return matches

    def _sensitive_value_findings(self, text: str, *, source: str) -> List[SecurityFinding]:
        if not text:
            return []
        matches: List[SecurityFinding] = []
        for pattern in self.options.sensitive_patterns:
            try:
                match = re.search(pattern, text)
            except re.error:
                logger.warning("Invalid sensitive content regex configured: %s", pattern)
                continue
            if not match:
                continue
            matches.append(
                SecurityFinding(
                    category=SecurityFindingCategory.SENSITIVE_CONTENT.value,
                    severity="medium",
                    message="Potential sensitive value detected in browser security scan input/output.",
                    score=0.50,
                    signal="sensitive_value_pattern",
                    source=source,
                    evidence=self._evidence(match.group(0)),
                    metadata={"pattern_hash": stable_hash(pattern, length=12)},
                )
            )
            break
        return matches

    def _build_report(self, *, action: str, findings: Sequence[SecurityFinding], url: str = "",
                      page_title: str = "", ready_state: Optional[str] = None,
                      page: Optional[Dict[str, Any]] = None, metadata: Optional[Mapping[str, Any]] = None,
                      correlation_id: Optional[str] = None, duration_ms: Optional[float] = None) -> SecurityReport:
        clean_url = self._display_url(url)
        domain = self._domain_from_url(url)
        decision = self._decision_from_findings(findings)
        report = SecurityReport(
            decision=decision,
            findings=tuple(findings),
            action=normalize_action_name(action) or SECURITY_ACTION_SCAN,
            url=clean_url,
            domain=domain,
            page_title=truncate_text(page_title, 500),
            ready_state=ready_state,
            metadata=safe_serialize(redact_mapping(dict(metadata or {}))),
            page=page,
            correlation_id=correlation_id or new_correlation_id("sec"),
            duration_ms=duration_ms,
        )
        return report

    def _decision_from_findings(self, findings: Sequence[SecurityFinding]) -> SecurityDecision:
        if not findings:
            return SecurityDecision(
                status=SecurityDecisionStatus.ALLOW.value,
                allowed=True,
                reason="No browser security risks detected.",
                risk_score=0.0,
                severity="low",
                recommendations=("Continue browser operation.",),
            )

        categories = tuple(sorted({finding.category for finding in findings}))
        max_score = max((float(finding.score or 0.0) for finding in findings), default=0.0)
        max_severity = _max_severity(finding.severity for finding in findings)
        must_block = self._must_block(findings, max_score=max_score)
        should_warn = max_score >= self.options.risk_threshold_warn

        if must_block:
            return SecurityDecision(
                status=SecurityDecisionStatus.BLOCK.value,
                allowed=False,
                reason=self._decision_reason(findings, blocked=True),
                risk_score=round(min(1.0, max_score), 4),
                severity=max_severity,
                categories=categories,
                recommendations=tuple(self._recommendations(findings, blocked=True)),
                code=self._decision_code(findings),
            )
        if should_warn:
            return SecurityDecision(
                status=SecurityDecisionStatus.WARN.value,
                allowed=True,
                reason=self._decision_reason(findings, blocked=False),
                risk_score=round(min(1.0, max_score), 4),
                severity=max_severity,
                categories=categories,
                recommendations=tuple(self._recommendations(findings, blocked=False)),
                code=self._decision_code(findings),
            )
        return SecurityDecision(
            status=SecurityDecisionStatus.ALLOW.value,
            allowed=True,
            reason="Only low-risk browser security signals detected.",
            risk_score=round(min(1.0, max_score), 4),
            severity=max_severity,
            categories=categories,
            recommendations=("Continue browser operation with normal telemetry redaction.",),
            code=self._decision_code(findings),
        )

    def _must_block(self, findings: Sequence[SecurityFinding], *, max_score: float) -> bool:
        categories = {finding.category for finding in findings}
        signals = {finding.signal for finding in findings}
        if max_score >= self.options.risk_threshold_block:
            return True
        if self.options.block_on_captcha and SecurityFindingCategory.CAPTCHA.value in categories:
            return True
        if self.options.block_on_bot_detection and SecurityFindingCategory.BOT_DETECTION.value in categories:
            return True
        if self.options.block_on_rate_limit and SecurityFindingCategory.RATE_LIMIT.value in categories:
            return True
        if self.options.block_on_unsafe_url and (
            SecurityFindingCategory.URL_POLICY.value in categories
            or SecurityFindingCategory.DOMAIN_POLICY.value in categories
            or SecurityFindingCategory.DOWNLOAD.value in categories
        ):
            return True
        if self.options.block_private_network_urls and SecurityFindingCategory.PRIVATE_NETWORK.value in categories:
            return True
        if "local_file_url" in signals:
            return True
        return False

    def _decision_reason(self, findings: Sequence[SecurityFinding], *, blocked: bool) -> str:
        primary = max(findings, key=lambda item: item.score or 0.0)
        prefix = "Blocked browser operation" if blocked else "Browser security warning"
        return f"{prefix}: {primary.message}"

    def _decision_code(self, findings: Sequence[SecurityFinding]) -> str:
        categories = {finding.category for finding in findings}
        if SecurityFindingCategory.CAPTCHA.value in categories:
            return CaptchaDetectedError.default_code
        if SecurityFindingCategory.BOT_DETECTION.value in categories:
            return BotDetectionError.default_code
        if SecurityFindingCategory.RATE_LIMIT.value in categories:
            return RateLimitError.default_code
        if SecurityFindingCategory.URL_POLICY.value in categories or SecurityFindingCategory.DOMAIN_POLICY.value in categories:
            return BrowserSecurityError.default_code
        return BrowserSecurityError.default_code

    def _recommendations(self, findings: Sequence[SecurityFinding], *, blocked: bool) -> List[str]:
        categories = {finding.category for finding in findings}
        recs: List[str] = []
        if SecurityFindingCategory.CAPTCHA.value in categories:
            recs.append("Stop automation on this page and require user-supervised resolution or an alternate allowed source.")
        if SecurityFindingCategory.BOT_DETECTION.value in categories:
            recs.append("Do not attempt to bypass site protections; retry later only if permitted by policy.")
        if SecurityFindingCategory.RATE_LIMIT.value in categories:
            recs.append("Back off before retrying and reduce request frequency.")
        if SecurityFindingCategory.URL_POLICY.value in categories or SecurityFindingCategory.DOMAIN_POLICY.value in categories:
            recs.append("Verify the URL/domain against the configured browser security policy before continuing.")
        if SecurityFindingCategory.PRIVATE_NETWORK.value in categories:
            recs.append("Avoid navigating to private/internal network hosts unless explicitly allowed for this deployment.")
        if SecurityFindingCategory.DOWNLOAD.value in categories:
            recs.append("Do not download or execute files from browser automation without explicit trusted policy.")
        if SecurityFindingCategory.SENSITIVE_CONTENT.value in categories:
            recs.append("Redact sensitive values from logs, memory, screenshots, and downstream prompts.")
        if not recs:
            recs.append("Review the security report before continuing." if blocked else "Continue with caution.")
        return recs

    def _scan_text_from_driver(self, driver: Any) -> str:
        if driver is None:
            return ""
        source = safe_call(lambda: get_page_html(driver, max_length=self.options.max_scan_text_chars), default="") or ""
        body = safe_call(lambda: get_body_text(driver, max_length=min(self.options.max_scan_text_chars, 50000)), default="") or ""
        url = safe_call(lambda: get_current_url(driver), default="") or ""
        title = safe_call(lambda: get_page_title(driver), default="") or ""
        return truncate_text("\n".join([url, title, body, source]), self.options.max_scan_text_chars)

    def _domain_from_url(self, url: str) -> str:
        try:
            return parse_browser_url(url).host.lower().strip(".")
        except Exception:
            parsed = urlparse(str(url or ""))
            return (parsed.hostname or "").lower().strip(".")

    def _display_url(self, url: str) -> str:
        if not url:
            return ""
        return redact_url(url) if self.options.redact_urls else str(url)

    def _evidence(self, value: str) -> Optional[str]:
        if not self.options.include_text_evidence:
            return None
        text = truncate_text(normalize_whitespace(value), self.options.max_evidence_chars)
        return "[REDACTED]" if self.options.redact_text and text else text

    def _domain_allowed(self, host: str) -> bool:
        if not self.options.allowed_domains:
            return True
        host = host.lower().strip(".")
        return any(host == domain or host.endswith(f".{domain}") for domain in self.options.allowed_domains)

    def _domain_blocked(self, host: str) -> bool:
        host = host.lower().strip(".")
        if any(host == domain or host.endswith(f".{domain}") for domain in self.options.blocked_domains):
            return True
        return any(host.endswith(suffix) for suffix in self.options.blocked_domain_suffixes)

    def _has_high_risk_tld(self, host: str) -> bool:
        if not self.options.high_risk_tlds:
            return False
        return any(host.endswith(tld if tld.startswith(".") else f".{tld}") for tld in self.options.high_risk_tlds)

    def _is_private_or_internal_host(self, host: str) -> bool:
        host = host.lower().strip(".")
        if not host:
            return False
        if host in self.options.internal_hostnames or any(host.endswith(f".{name}") for name in self.options.internal_hostnames):
            return True
        try:
            ip = ipaddress.ip_address(host)
            return bool(ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast)
        except ValueError:
            return host.endswith(".local") or host.endswith(".internal") or host.endswith(".lan")

    def _blocked_download_extension(self, path: str) -> Optional[str]:
        lowered = str(path or "").lower()
        for extension in self.options.blocked_download_extensions:
            if lowered.endswith(extension):
                return extension
        return None

    def _record_report(self, report: SecurityReport) -> None:
        decision = report.decision.status
        if not self.options.memory_enabled or self.memory is None:
            return
        if decision == SecurityDecisionStatus.ALLOW.value and not self.options.memory_record_allowed:
            return
        if decision == SecurityDecisionStatus.WARN.value and not self.options.memory_record_warnings:
            return
        if decision == SecurityDecisionStatus.BLOCK.value and not self.options.memory_record_blocks:
            return

        payload = report.to_dict()
        key = f"{report.action}:{report.correlation_id}"
        # BrowserMemory is evolving. Use richer methods when present and fall
        # back to generic put/safe_put style methods without failing security.
        for method_name in ("remember_security_event", "safe_put", "put"):
            method = getattr(self.memory, method_name, None)
            if not callable(method):
                continue
            try:
                if method_name == "remember_security_event":
                    method(payload, key=key, url=report.url, tags=report.decision.categories)
                elif method_name == "safe_put":
                    method(
                        key=key,
                        value=payload,
                        namespace=self.options.memory_namespace,
                        kind="security_event",
                        url=report.url,
                        tags=report.decision.categories,
                    )
                else:
                    method(
                        key=key,
                        value=payload,
                        namespace=self.options.memory_namespace,
                        kind="security_event",
                        url=report.url,
                        tags=report.decision.categories,
                    )
                return
            except TypeError:
                continue
            except Exception as exc:
                logger.debug("Unable to record security report in memory: %s", exc)
                return


# ---------------------------------------------------------------------------
# Helper functions local to security policy composition
# ---------------------------------------------------------------------------
def _normalize_string_list(value: Any) -> List[str]:
    return [str(item).strip().lower() for item in ensure_list(value) if str(item or "").strip()]


def _normalize_domain_list(value: Any) -> List[str]:
    output: List[str] = []
    for item in ensure_list(value):
        text = str(item or "").lower().strip().strip(".")
        if text:
            output.append(text)
    return output


def _normalize_indicators(value: Any) -> List[str]:
    return [str(item).strip() for item in ensure_list(value) if str(item or "").strip()]


def _normalize_extensions(value: Any) -> List[str]:
    output: List[str] = []
    for item in ensure_list(value):
        text = str(item or "").lower().strip()
        if not text:
            continue
        output.append(text if text.startswith(".") else f".{text}")
    return output


def _contains_any(text: str, indicators: Iterable[str]) -> bool:
    lowered = str(text or "").lower()
    return any(str(indicator or "").lower().strip() in lowered for indicator in indicators if str(indicator or "").strip())


def _max_severity(values: Iterable[str]) -> str:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    best = "low"
    best_score = -1
    for value in values:
        normalized = str(value or "low").lower()
        score = order.get(normalized, 0)
        if score > best_score:
            best = normalized
            best_score = score
    return best


def _find_nested_decision(value: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    for key in ("decision",):
        item = value.get(key)
        if isinstance(item, Mapping):
            return item
    data = value.get("data")
    if isinstance(data, Mapping):
        decision = data.get("decision")
        if isinstance(decision, Mapping):
            return decision
    metadata = value.get("metadata")
    if isinstance(metadata, Mapping):
        report = metadata.get("security_report")
        if isinstance(report, Mapping) and isinstance(report.get("decision"), Mapping):
            return report["decision"]
    return None


def _error_for_decision(decision: SecurityDecision, *, context: Optional[Mapping[str, Any]] = None) -> BrowserError:
    categories = set(decision.categories)
    message = decision.reason or "Browser security policy blocked the operation"
    if SecurityFindingCategory.CAPTCHA.value in categories:
        return CaptchaDetectedError(message, context=context)
    if SecurityFindingCategory.BOT_DETECTION.value in categories:
        return BotDetectionError(message, context=context)
    if SecurityFindingCategory.RATE_LIMIT.value in categories:
        return RateLimitError(message, context=context)
    if SecurityFindingCategory.URL_POLICY.value in categories or SecurityFindingCategory.DOMAIN_POLICY.value in categories:
        return BrowserSecurityError(message, context=context)
    if SecurityFindingCategory.PRIVATE_NETWORK.value in categories:
        return PermissionDeniedError(message, context=context)
    return BrowserSecurityError(message, context=context)


# ---------------------------------------------------------------------------
# Self-contained smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Security Features ===\n")
    printer.status("TEST", "Security Features initialized", "info")

    class _FakeElement:
        def __init__(self, text: str = ""):
            self.text = text
            self.tag_name = "body"

        def get_attribute(self, name: str):
            if name in {"innerText", "textContent"}:
                return self.text
            return None

    class _FakeDriver:
        def __init__(self, url: str, title: str, body: str, html: str = ""):
            self.current_url = url
            self.title = title
            self.page_source = html or f"<html><title>{title}</title><body>{body}</body></html>"
            self._body = body

        def find_element(self, by=None, value=None):
            return _FakeElement(self._body)

        def execute_script(self, script: str, *args):
            if "document.readyState" in script:
                return "complete"
            if "window.innerWidth" in script:
                return {
                    "width": 1200,
                    "height": 800,
                    "scrollX": 0,
                    "scrollY": 0,
                    "pageWidth": 1200,
                    "pageHeight": 1600,
                    "devicePixelRatio": 1,
                }
            if "performance.getEntriesByType" in script:
                return {"duration": 12.3, "type": "navigate"}
            return None

        def get_screenshot_as_png(self):
            return b"fake-png"

    safe_driver = _FakeDriver("https://example.com", "Example", "Welcome to a normal page.")
    captcha_driver = _FakeDriver("https://example.com/check", "Security Check", "Please verify you are human. CAPTCHA required.")

    security = SecurityFeatures(driver=safe_driver, memory=None)

    url_result = security.assess_navigation("https://example.com/docs")
    assert url_result["status"] == "success", url_result

    blocked_url_result = security.assess_navigation("javascript:alert(1)")
    assert blocked_url_result["status"] == "error", blocked_url_result

    safe_page_result = security.scan_current_page()
    assert safe_page_result["status"] == "success", safe_page_result

    security.attach_driver(captcha_driver)
    assert SecurityFeatures.detect_captcha(captcha_driver) is True
    captcha_result = security.scan_current_page()
    assert captcha_result["status"] == "error", captcha_result

    action_result = security.assess_action("paste", {"selector": "#token", "text": "api_key = abcdefghijklmnopqrstuvwxyz123456"})
    assert action_result["status"] in {"success", "error"}, action_result

    delay = calculate_backoff_delay(0, base_delay=0.01, max_delay=0.01, jitter=0.0)
    assert delay == 0.01, delay

    print("\n=== Test ran successfully ===\n")
