from __future__ import annotations

"""
Production-grade content handling for the browser subsystem.

This module owns browser-adjacent content extraction and post-processing. It is
intentionally focused on converting URLs, browser pages, search-result records,
PDFs, arXiv pages, HTML documents, and plain text into safe, structured,
bounded content payloads for BrowserAgent, BrowserFunctions, BrowserMemory, and
future reasoning/learning layers.

Scope
-----
ContentHandling does not own browser lifecycle, navigation, clicking, typing,
scrolling, workflow execution, or task routing. Those concerns belong to the
browser agent and concrete browser function modules. This module focuses on one
stable contract: inspect content context, choose the appropriate extractor, and
return a safe, observable, configurable extraction result.

Design goals
------------
- Preserve existing public entry points used by the current BrowserAgent/search
  flow: ``handle_pdf()``, ``handle_arxiv()``, and ``postprocess_if_special()``.
- Use shared browser errors and helpers instead of redefining generic
  validation, URL normalization, serialization, redaction, result construction,
  hashing, text normalization, snapshots, or memory-safe payload behavior.
- Keep runtime behavior configurable from ``browser_config.yaml`` through the
  ``content_handling`` section.
- Provide structured extraction metadata while retaining backwards-compatible
  string-returning methods for older call sites.
- Extract only bounded content by default so downstream agents, logs, and memory
  remain safe and predictable.
- Remain extensible for future handlers such as Markdown, CSV, JSON, XML,
  notebooks, screenshots/OCR, reader-mode extraction, paywall-safe summaries,
  domain-specific parsers, and CDP/network-response capture.

Local imports are intentionally direct and are not wrapped in try/except. Import
or packaging failures should surface clearly during development and deployment.
"""

import asyncio
import mimetypes
import re
import time as time_module
import requests

from io import BytesIO
from pypdf import PdfReader
from urllib.parse import urlparse
from selenium.webdriver.common.by import By
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, cast
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException

from .utils.config_loader import load_global_config, get_config_section
from .utils.browser_errors import *
from .utils.Browser_helpers import *
from .browser_memory import BrowserMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Content Handling")
printer = PrettyPrinter()


CONTENT_ACTION = "extract_content"
SUPPORTED_CONTENT_KINDS = {
    "auto",
    "pdf",
    "arxiv",
    "html",
    "text",
    "json",
    "xml",
    "markdown",
    "page",
    "unknown",
}
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; BrowserAgentContentHandling/1.0)"
ARXIV_ABSTRACT_SELECTORS: Tuple[str, ...] = (
    "blockquote.abstract",
    ".abstract",
    "meta[name='citation_abstract']",
)
ARXIV_TITLE_SELECTORS: Tuple[str, ...] = (
    "h1.title",
    "meta[name='citation_title']",
    "title",
)
DEFAULT_TEXT_SEPARATORS: Tuple[str, ...] = ("\n\n", "\n", ". ", " ")
PDF_MIME_TYPES = {"application/pdf", "application/x-pdf"}
TEXTUAL_MIME_PREFIXES = ("text/",)
TEXTUAL_MIME_TYPES = {
    "application/json",
    "application/xml",
    "application/xhtml+xml",
    "application/ld+json",
    "application/rss+xml",
    "application/atom+xml",
}
HTML_MIME_TYPES = {"text/html", "application/xhtml+xml"}


@dataclass(frozen=True)
class ContentHandlingOptions:
    """Config-backed policy for content extraction."""

    enabled: bool = True
    request_timeout: float = 15.0
    connect_timeout: float = 5.0
    read_timeout: float = 15.0
    max_download_bytes: int = 12_000_000
    stream_downloads: bool = True
    user_agent: str = DEFAULT_USER_AGENT
    follow_redirects: bool = True
    verify_ssl: bool = True
    max_redirects: int = 8
    max_text_chars: int = 20_000
    preview_chars: int = 2_000
    pdf_max_pages: Optional[int] = 25
    pdf_password: Optional[str] = None
    normalize_text: bool = True
    collapse_whitespace: bool = False
    include_metadata: bool = True
    include_raw_html: bool = False
    include_page_snapshot: bool = False
    include_headers: bool = False
    redact_headers: bool = True
    remember_results: bool = True
    memory_namespace: str = "content"
    memory_ttl_seconds: Optional[int] = 86_400
    raise_on_failure: bool = False
    unsupported_content_policy: str = "text_fallback"
    arxiv_body_fallback_chars: int = 4_000
    arxiv_strip_abstract_label: bool = True
    min_extracted_chars: int = 0
    classify_from_headers: bool = True
    classify_from_url: bool = True
    safe_result_text_on_error: bool = True
    retries: int = 1
    retry_backoff_base: float = 0.25
    retry_backoff_multiplier: float = 1.8
    retry_backoff_max: float = 3.0
    retry_jitter: float = 0.05

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]], **overrides: Any) -> "ContentHandlingOptions":
        cfg = dict(config or {})
        request_cfg = dict(cfg.get("requests") or {})
        limits = dict(cfg.get("limits") or {})
        pdf_cfg = dict(cfg.get("pdf") or {})
        arxiv_cfg = dict(cfg.get("arxiv") or {})
        diagnostics = dict(cfg.get("diagnostics") or {})
        memory_cfg = dict(cfg.get("memory") or {})
        retry_cfg = dict(cfg.get("retry") or {})
        classification_cfg = dict(cfg.get("classification") or {})
        policy_cfg = dict(cfg.get("policy") or {})

        merged = {
            "enabled": cfg.get("enabled", True),
            "request_timeout": request_cfg.get("timeout", cfg.get("request_timeout", 15.0)),
            "connect_timeout": request_cfg.get("connect_timeout", cfg.get("connect_timeout", 5.0)),
            "read_timeout": request_cfg.get("read_timeout", cfg.get("read_timeout", 15.0)),
            "max_download_bytes": limits.get("max_download_bytes", cfg.get("max_download_bytes", 12_000_000)),
            "stream_downloads": request_cfg.get("stream_downloads", cfg.get("stream_downloads", True)),
            "user_agent": request_cfg.get("user_agent", cfg.get("user_agent", DEFAULT_USER_AGENT)),
            "follow_redirects": request_cfg.get("follow_redirects", cfg.get("follow_redirects", True)),
            "verify_ssl": request_cfg.get("verify_ssl", cfg.get("verify_ssl", True)),
            "max_redirects": request_cfg.get("max_redirects", cfg.get("max_redirects", 8)),
            "max_text_chars": limits.get("max_text_chars", cfg.get("max_text_chars", 20_000)),
            "preview_chars": limits.get("preview_chars", cfg.get("preview_chars", 2_000)),
            "pdf_max_pages": pdf_cfg.get("max_pages", cfg.get("pdf_max_pages", 25)),
            "pdf_password": pdf_cfg.get("password", cfg.get("pdf_password")),
            "normalize_text": cfg.get("normalize_text", True),
            "collapse_whitespace": cfg.get("collapse_whitespace", False),
            "include_metadata": diagnostics.get("include_metadata", cfg.get("include_metadata", True)),
            "include_raw_html": diagnostics.get("include_raw_html", cfg.get("include_raw_html", False)),
            "include_page_snapshot": diagnostics.get("include_page_snapshot", cfg.get("include_page_snapshot", False)),
            "include_headers": diagnostics.get("include_headers", cfg.get("include_headers", False)),
            "redact_headers": diagnostics.get("redact_headers", cfg.get("redact_headers", True)),
            "remember_results": memory_cfg.get("enabled", cfg.get("remember_results", True)),
            "memory_namespace": memory_cfg.get("namespace", cfg.get("memory_namespace", "content")),
            "memory_ttl_seconds": memory_cfg.get("ttl_seconds", cfg.get("memory_ttl_seconds", 86_400)),
            "raise_on_failure": policy_cfg.get("raise_on_failure", cfg.get("raise_on_failure", False)),
            "unsupported_content_policy": policy_cfg.get("unsupported_content_policy", cfg.get("unsupported_content_policy", "text_fallback")),
            "arxiv_body_fallback_chars": arxiv_cfg.get("body_fallback_chars", cfg.get("arxiv_body_fallback_chars", 4_000)),
            "arxiv_strip_abstract_label": arxiv_cfg.get("strip_abstract_label", cfg.get("arxiv_strip_abstract_label", True)),
            "min_extracted_chars": limits.get("min_extracted_chars", cfg.get("min_extracted_chars", 0)),
            "classify_from_headers": classification_cfg.get("from_headers", cfg.get("classify_from_headers", True)),
            "classify_from_url": classification_cfg.get("from_url", cfg.get("classify_from_url", True)),
            "safe_result_text_on_error": policy_cfg.get("safe_result_text_on_error", cfg.get("safe_result_text_on_error", True)),
            "retries": retry_cfg.get("max_attempts", cfg.get("retries", 1)),
            "retry_backoff_base": retry_cfg.get("base_delay", cfg.get("retry_backoff_base", 0.25)),
            "retry_backoff_multiplier": retry_cfg.get("multiplier", cfg.get("retry_backoff_multiplier", 1.8)),
            "retry_backoff_max": retry_cfg.get("max_delay", cfg.get("retry_backoff_max", 3.0)),
            "retry_jitter": retry_cfg.get("jitter", cfg.get("retry_jitter", 0.05)),
        }
        merged.update({key: value for key, value in overrides.items() if value is not None})

        pdf_max_pages_value = merged.get("pdf_max_pages")
        pdf_max_pages = None if pdf_max_pages_value in (None, "", "none", "None") else coerce_int(pdf_max_pages_value, default=25, minimum=1)
        memory_ttl_value = merged.get("memory_ttl_seconds")
        memory_ttl = None if memory_ttl_value in (None, "", "none", "None") else coerce_int(memory_ttl_value, default=86_400, minimum=0)

        unsupported_policy = str(merged.get("unsupported_content_policy") or "text_fallback").lower().strip()
        if unsupported_policy not in {"text_fallback", "skip", "error"}:
            unsupported_policy = "text_fallback"

        return cls(
            enabled=coerce_bool(merged.get("enabled"), default=True),
            request_timeout=coerce_float(merged.get("request_timeout"), default=15.0, minimum=0.1, maximum=300.0),
            connect_timeout=coerce_float(merged.get("connect_timeout"), default=5.0, minimum=0.1, maximum=300.0),
            read_timeout=coerce_float(merged.get("read_timeout"), default=15.0, minimum=0.1, maximum=300.0),
            max_download_bytes=coerce_int(merged.get("max_download_bytes"), default=12_000_000, minimum=1),
            stream_downloads=coerce_bool(merged.get("stream_downloads"), default=True),
            user_agent=str(merged.get("user_agent") or DEFAULT_USER_AGENT),
            follow_redirects=coerce_bool(merged.get("follow_redirects"), default=True),
            verify_ssl=coerce_bool(merged.get("verify_ssl"), default=True),
            max_redirects=coerce_int(merged.get("max_redirects"), default=8, minimum=0, maximum=50),
            max_text_chars=coerce_int(merged.get("max_text_chars"), default=20_000, minimum=1),
            preview_chars=coerce_int(merged.get("preview_chars"), default=2_000, minimum=1),
            pdf_max_pages=pdf_max_pages,
            pdf_password=merged.get("pdf_password"),
            normalize_text=coerce_bool(merged.get("normalize_text"), default=True),
            collapse_whitespace=coerce_bool(merged.get("collapse_whitespace"), default=False),
            include_metadata=coerce_bool(merged.get("include_metadata"), default=True),
            include_raw_html=coerce_bool(merged.get("include_raw_html"), default=False),
            include_page_snapshot=coerce_bool(merged.get("include_page_snapshot"), default=False),
            include_headers=coerce_bool(merged.get("include_headers"), default=False),
            redact_headers=coerce_bool(merged.get("redact_headers"), default=True),
            remember_results=coerce_bool(merged.get("remember_results"), default=True),
            memory_namespace=str(merged.get("memory_namespace") or "content"),
            memory_ttl_seconds=memory_ttl,
            raise_on_failure=coerce_bool(merged.get("raise_on_failure"), default=False),
            unsupported_content_policy=unsupported_policy,
            arxiv_body_fallback_chars=coerce_int(merged.get("arxiv_body_fallback_chars"), default=4_000, minimum=100),
            arxiv_strip_abstract_label=coerce_bool(merged.get("arxiv_strip_abstract_label"), default=True),
            min_extracted_chars=coerce_int(merged.get("min_extracted_chars"), default=0, minimum=0),
            classify_from_headers=coerce_bool(merged.get("classify_from_headers"), default=True),
            classify_from_url=coerce_bool(merged.get("classify_from_url"), default=True),
            safe_result_text_on_error=coerce_bool(merged.get("safe_result_text_on_error"), default=True),
            retries=coerce_int(merged.get("retries"), default=1, minimum=0, maximum=20),
            retry_backoff_base=coerce_float(merged.get("retry_backoff_base"), default=0.25, minimum=0.0, maximum=60.0),
            retry_backoff_multiplier=coerce_float(merged.get("retry_backoff_multiplier"), default=1.8, minimum=1.0, maximum=10.0),
            retry_backoff_max=coerce_float(merged.get("retry_backoff_max"), default=3.0, minimum=0.0, maximum=300.0),
            retry_jitter=coerce_float(merged.get("retry_jitter"), default=0.05, minimum=0.0, maximum=10.0),
        )


@dataclass(frozen=True)
class ContentRequest:
    """Normalized request for content extraction."""

    url: str = ""
    kind: str = "auto"
    source: str = "unknown"
    result: Optional[Mapping[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: new_correlation_id("content"))

    @classmethod
    def build(
        cls,
        *,
        url: Optional[str] = None,
        kind: str = "auto",
        source: str = "unknown",
        result: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> "ContentRequest":
        extracted_url = url or ""
        if not extracted_url and result:
            extracted_url = str(result.get("link") or result.get("url") or result.get("href") or "")
        normalized_kind = normalize_content_kind(kind)
        return cls(
            url=str(extracted_url or ""),
            kind=normalized_kind,
            source=normalize_whitespace(source or "unknown") or "unknown",
            result=result,
            metadata=dict(metadata or {}),
            correlation_id=correlation_id or new_correlation_id("content"),
        )


@dataclass(frozen=True)
class ContentMetadata:
    """Structured metadata describing extracted browser content."""

    url: str = ""
    final_url: str = ""
    domain: str = ""
    kind: str = "unknown"
    content_type: str = ""
    status_code: Optional[int] = None
    title: str = ""
    source: str = "unknown"
    text_length: int = 0
    truncated: bool = False
    fingerprint: str = ""
    extracted_at: str = field(default_factory=utc_now_iso)
    duration_ms: Optional[float] = None
    page_count: Optional[int] = None
    page_limit_applied: bool = False
    headers: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return prune_none(asdict(self))


@dataclass(frozen=True)
class ContentExtractionResult:
    """Stable structured extraction result."""

    status: str
    text: str = ""
    metadata: ContentMetadata = field(default_factory=ContentMetadata)
    error: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None
    correlation_id: str = field(default_factory=lambda: new_correlation_id("content"))

    def to_dict(self, *, include_raw: bool = True) -> Dict[str, Any]:
        payload = {
            "status": self.status,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
            "error": self.error,
            "correlation_id": self.correlation_id,
        }
        if include_raw and self.raw is not None:
            payload["raw"] = self.raw
        return redact_mapping(prune_none(payload))


class _LimitedBytesBuffer:
    """Small helper for enforcing download size limits while streaming."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.buffer = BytesIO()
        self.total = 0

    def write(self, chunk: bytes) -> None:
        self.total += len(chunk)
        if self.total > self.max_bytes:
            raise ContentExtractionError(
                "Downloaded content exceeded configured byte limit",
                context={"max_bytes": self.max_bytes, "downloaded_bytes": self.total},
            )
        self.buffer.write(chunk)

    def getvalue(self) -> bytes:
        return self.buffer.getvalue()


class ContentHandling:
    """Content extractor and post-processor for browser results."""

    def __init__(self, memory: Optional[BrowserMemory] = None, session: Optional[requests.Session] = None) -> None:
        self.config = load_global_config()
        self.content_config = get_config_section("content_handling") or {}
        self.options = ContentHandlingOptions.from_config(self.content_config)
        self.memory = memory if memory is not None else BrowserMemory()
        self.session = session or requests.Session()
        self._configure_session()
        logger.info("Content Handling initialized.")

    def _configure_session(self) -> None:
        self.session.headers.update({"User-Agent": self.options.user_agent})
        self.session.max_redirects = self.options.max_redirects

    # ------------------------------------------------------------------
    # Backwards-compatible public static entry points
    # ------------------------------------------------------------------
    @staticmethod
    def handle_pdf(url: str) -> str:
        """Extract text from a PDF URL and return text for legacy callers."""

        handler = ContentHandling()
        result = handler.extract_pdf(url, return_result=True)
        if result.status == "success":
            return result.text[: handler.options.preview_chars]
        message = result.error.get("message") if isinstance(result.error, Mapping) else "unknown error"
        return f"Failed to extract PDF: {message}"

    @staticmethod
    def handle_arxiv(driver) -> str:
        """Extract the abstract from an arXiv page and return text for legacy callers."""

        handler = ContentHandling()
        result = handler.extract_arxiv(driver=driver, return_result=True)
        if result.status == "success":
            return result.text[: handler.options.preview_chars]
        message = result.error.get("message") if isinstance(result.error, Mapping) else "unknown error"
        return f"Failed to extract arXiv content: {message}"

    @staticmethod
    def postprocess_if_special(result: dict, driver) -> dict:
        """Preserve legacy search-result post-processing contract.

        Mutates and returns ``result`` when a URL has a specialized handler.
        For richer diagnostics, use ``ContentHandling().postprocess_result``.
        """

        handler = ContentHandling()
        processed = handler.postprocess_result(result, driver=driver, return_result=False)
        return processed if isinstance(processed, dict) else result

    # ------------------------------------------------------------------
    # Async wrappers
    # ------------------------------------------------------------------
    async def do_extract_url(self, url: str, *, kind: str = "auto", **overrides: Any) -> Dict[str, Any]:
        return cast(
            Dict[str, Any],
            await asyncio.to_thread(self.extract_url, url, kind=kind, return_result=False, **overrides),
        )

    async def do_extract_page(self, driver, **overrides: Any) -> Dict[str, Any]:
        return cast(
            Dict[str, Any],
            await asyncio.to_thread(self.extract_page, driver, return_result=False, **overrides),
        )

    async def do_postprocess_result(self, result: Mapping[str, Any], driver=None, **overrides: Any) -> Dict[str, Any]:
        return cast(
            Dict[str, Any],
            await asyncio.to_thread(self.postprocess_result, result, driver=driver, return_result=False, **overrides),
        )

    # ------------------------------------------------------------------
    # High-level extraction methods
    # ------------------------------------------------------------------
    def postprocess_result(
        self, result: Mapping[str, Any], *, driver=None, return_result: Literal[True], **overrides: Any
    ) -> Dict[str, Any] | ContentExtractionResult:
        """Post-process a search/browser result if a specialized handler applies."""

        request = ContentRequest.build(result=result, source="postprocess_result")
        options = ContentHandlingOptions.from_config(self.content_config, **overrides)
        kind = self.classify(url=request.url, result=result, driver=driver, options=options)

        if kind not in {"pdf", "arxiv", "html", "text", "json", "xml", "markdown", "page"}:
            if return_result:
                return self._build_success_result(
                    text=str(result.get("text") or ""),
                    request=request,
                    options=options,
                    kind="unknown",
                    source="postprocess_result",
                    metadata_extra={"postprocessed": False, "reason": "no_special_handler"},
                )
            return dict(result)

        extraction = self.extract(request.url, kind=kind, driver=driver, request=request, options=options, return_result=True)
        if return_result:
            return extraction

        updated = dict(result)
        if extraction.status == "success":
            updated["text"] = extraction.text[: options.preview_chars]
            updated["content"] = extraction.to_dict(include_raw=False)
            updated["content_kind"] = extraction.metadata.kind
        else:
            updated["content_error"] = extraction.error
            if options.safe_result_text_on_error:
                updated.setdefault("text", "")
        return redact_mapping(updated)

    def extract_url(self, url: str, *, kind: str = "auto", return_result: bool = False, **overrides: Any) -> Dict[str, Any] | ContentExtractionResult:
        """Extract content from a URL and return a BrowserAgent-compatible result."""

        options = ContentHandlingOptions.from_config(self.content_config, **overrides)
        request = ContentRequest.build(url=url, kind=kind, source="url")
        extraction = cast(ContentExtractionResult, self.extract(request.url, kind=request.kind, request=request, options=options, return_result=True))
        if return_result:
            return extraction
        return self._to_action_result(extraction, action="extract_url")

    def extract(
        self,
        url: str = "",
        *,
        kind: str = "auto",
        driver=None,
        request: Optional[ContentRequest] = None,
        options: Optional[ContentHandlingOptions] = None,
        return_result: bool = True,
        **overrides: Any,
    ) -> Dict[str, Any] | ContentExtractionResult:
        """Extract content from URL or browser driver using the configured policy."""

        options = options or ContentHandlingOptions.from_config(self.content_config, **overrides)
        request = request or ContentRequest.build(url=url, kind=kind, source="extract")
        start_ms = monotonic_ms()

        try:
            if not options.enabled:
                raise ContentExtractionError("Content handling is disabled", context={"url": request.url})
            resolved_kind = self.classify(url=request.url, result=request.result, driver=driver, requested_kind=request.kind, options=options)
            if resolved_kind == "pdf":
                result = self.extract_pdf(request.url, options=options, request=request, return_result=True)
            elif resolved_kind == "arxiv":
                result = self.extract_arxiv(driver=driver, url=request.url, options=options, request=request, return_result=True)
            elif resolved_kind == "page":
                result = self.extract_page(driver, options=options, request=request, return_result=True)
            elif resolved_kind in {"html", "text", "json", "xml", "markdown"}:
                result = self.extract_remote_text(request.url, kind=resolved_kind, options=options, request=request, return_result=True)
            else:
                result = self._handle_unsupported(request=request, options=options, kind=resolved_kind)
            if result.metadata.duration_ms is None:
                result = self._replace_duration(result, elapsed_ms(start_ms))
            if options.remember_results:
                self._remember_result(result, request=request, options=options)
            return result if return_result else self._to_action_result(result, action=CONTENT_ACTION)
        except Exception as exc:
            browser_error = wrap_browser_exception(
                exc,
                action=CONTENT_ACTION,
                message=f"Content extraction failed for {request.url or '<driver>'}",
                context={"url": request.url, "kind": request.kind, "source": request.source},
                default_error_cls=ContentExtractionError,
            )
            logger.error("Content extraction failed: %s", browser_error.to_json(redact=True))
            if options.raise_on_failure:
                raise browser_error from exc
            result = self._build_error_result(browser_error, request=request, options=options, duration_ms=elapsed_ms(start_ms))
            return result if return_result else self._to_action_result(result, action=CONTENT_ACTION)

    # ------------------------------------------------------------------
    # Concrete extractors
    # ------------------------------------------------------------------
    def extract_pdf(
        self,
        url: str,
        *,
        options: Optional[ContentHandlingOptions] = None,
        request: Optional[ContentRequest] = None,
        return_result: bool = True,
    ) -> ContentExtractionResult | Dict[str, Any]:
        """Extract bounded text from a remote PDF."""

        options = options or self.options
        request = request or ContentRequest.build(url=url, kind="pdf", source="pdf")
        start_ms = monotonic_ms()
        try:
            validated_url = self._validate_remote_url(url)
            pdf_bytes, response_metadata = self._fetch_bytes(validated_url, options=options, expected_kind="pdf")
            text, page_count, page_limit_applied = self._extract_pdf_text(pdf_bytes, options=options)
            text, truncated = self._prepare_text(text, options=options)
            metadata = self._metadata(
                request=request,
                options=options,
                kind="pdf",
                text=text,
                final_url=response_metadata.get("final_url", validated_url),
                content_type=response_metadata.get("content_type", "application/pdf"),
                status_code=response_metadata.get("status_code"),
                duration_ms=elapsed_ms(start_ms),
                truncated=truncated,
                page_count=page_count,
                page_limit_applied=page_limit_applied,
                headers=response_metadata.get("headers", {}),
            )
            result = ContentExtractionResult(status="success", text=text, metadata=metadata, correlation_id=request.correlation_id)
            return result if return_result else self._to_action_result(result, action="extract_pdf")
        except Exception as exc:
            error = wrap_browser_exception(exc, action="extract_pdf", context={"url": url}, default_error_cls=PDFExtractionError)
            if options.raise_on_failure:
                raise error from exc
            result = self._build_error_result(error, request=request, options=options, duration_ms=elapsed_ms(start_ms), kind="pdf")
            return result if return_result else self._to_action_result(result, action="extract_pdf")

    def extract_arxiv(
        self,
        *,
        driver=None,
        url: str = "",
        options: Optional[ContentHandlingOptions] = None,
        request: Optional[ContentRequest] = None,
        return_result: bool = True,
    ) -> ContentExtractionResult | Dict[str, Any]:
        """Extract abstract/title content from an arXiv page."""

        options = options or self.options
        request = request or ContentRequest.build(url=url or safe_driver_url(driver), kind="arxiv", source="arxiv")
        start_ms = monotonic_ms()
        try:
            if driver is None and request.url:
                remote_result = self.extract_remote_text(request.url, kind="html", options=options, request=request, return_result=True)
                if remote_result.status != "success":
                    return remote_result
                abstract = self._extract_arxiv_abstract_from_html(remote_result.raw.get("html", "") if remote_result.raw else remote_result.text, options=options)
                title = self._extract_title_from_html(remote_result.raw.get("html", "") if remote_result.raw else "")
                text = abstract or remote_result.text[: options.arxiv_body_fallback_chars]
                text, truncated = self._prepare_text(text, options=options)
                metadata = self._metadata(
                    request=request,
                    options=options,
                    kind="arxiv",
                    text=text,
                    title=title,
                    duration_ms=elapsed_ms(start_ms),
                    truncated=truncated,
                    extra={"source_mode": "remote_html"},
                )
                result = ContentExtractionResult(status="success", text=text, metadata=metadata, correlation_id=request.correlation_id)
                return result if return_result else self._to_action_result(result, action="extract_arxiv")

            abstract = self._extract_arxiv_abstract_from_driver(driver, options=options)
            if not abstract:
                body = self._body_text_from_driver(driver)
                abstract = body[: options.arxiv_body_fallback_chars]
            title = safe_driver_title(driver)
            text, truncated = self._prepare_text(abstract, options=options)
            metadata = self._metadata(
                request=request,
                options=options,
                kind="arxiv",
                text=text,
                title=title,
                final_url=safe_driver_url(driver) or request.url,
                duration_ms=elapsed_ms(start_ms),
                truncated=truncated,
                extra={"source_mode": "driver"},
            )
            result = ContentExtractionResult(status="success", text=text, metadata=metadata, correlation_id=request.correlation_id)
            return result if return_result else self._to_action_result(result, action="extract_arxiv")
        except Exception as exc:
            error = wrap_browser_exception(exc, action="extract_arxiv", context={"url": request.url}, default_error_cls=ArxivExtractionError)
            if options.raise_on_failure:
                raise error from exc
            result = self._build_error_result(error, request=request, options=options, duration_ms=elapsed_ms(start_ms), kind="arxiv")
            return result if return_result else self._to_action_result(result, action="extract_arxiv")

    def extract_page(
        self,
        driver,
        *,
        options: Optional[ContentHandlingOptions] = None,
        request: Optional[ContentRequest] = None,
        return_result: bool = False,
    ) -> Dict[str, Any] | ContentExtractionResult:
        """Extract visible page text from the current browser driver."""

        options = options or self.options
        request = request or ContentRequest.build(url=safe_driver_url(driver), kind="page", source="driver")
        start_ms = monotonic_ms()
        try:
            if driver is None:
                raise MissingDriverError("Cannot extract page content without a driver")
            body_text = self._body_text_from_driver(driver)
            title = safe_driver_title(driver)
            text, truncated = self._prepare_text(body_text, options=options)
            raw: Dict[str, Any] = {}
            if options.include_page_snapshot:
                raw["page_snapshot"] = self._safe_page_snapshot(driver)
            if options.include_raw_html:
                raw["html"] = safe_driver_html(driver)
            metadata = self._metadata(
                request=request,
                options=options,
                kind="page",
                text=text,
                title=title,
                final_url=safe_driver_url(driver),
                duration_ms=elapsed_ms(start_ms),
                truncated=truncated,
            )
            result = ContentExtractionResult(status="success", text=text, metadata=metadata, raw=raw or None, correlation_id=request.correlation_id)
            if options.remember_results:
                self._remember_result(result, request=request, options=options)
            return result if return_result else self._to_action_result(result, action="extract_page")
        except Exception as exc:
            error = wrap_browser_exception(exc, action="extract_page", context={"url": request.url}, default_error_cls=PageSnapshotError)
            if options.raise_on_failure:
                raise error from exc
            result = self._build_error_result(error, request=request, options=options, duration_ms=elapsed_ms(start_ms), kind="page")
            return result if return_result else self._to_action_result(result, action="extract_page")

    def extract_remote_text(
        self,
        url: str,
        *,
        kind: str = "auto",
        options: Optional[ContentHandlingOptions] = None,
        request: Optional[ContentRequest] = None,
        return_result: bool = True,
    ) -> ContentExtractionResult | Dict[str, Any]:
        """Fetch and extract text-like remote content."""

        options = options or self.options
        request = request or ContentRequest.build(url=url, kind=kind, source="remote_text")
        start_ms = monotonic_ms()
        try:
            validated_url = self._validate_remote_url(url)
            body, response_metadata = self._fetch_bytes(validated_url, options=options, expected_kind=kind)
            content_type = response_metadata.get("content_type") or guess_content_type(validated_url)
            decoded = self._decode_bytes(body, content_type=content_type, headers=response_metadata.get("headers", {}))
            resolved_kind = self.classify(url=validated_url, headers=response_metadata.get("headers"), content_type=content_type, requested_kind=kind, options=options)
            if resolved_kind == "html":
                text = html_to_readable_text(decoded)
            elif resolved_kind in {"json", "xml", "markdown", "text"}:
                text = decoded
            elif options.unsupported_content_policy == "text_fallback":
                text = decoded
            elif options.unsupported_content_policy == "skip":
                text = ""
            else:
                raise UnsupportedContentTypeError(
                    "Unsupported remote content type",
                    context={"url": validated_url, "content_type": content_type, "kind": resolved_kind},
                )
            text, truncated = self._prepare_text(text, options=options)
            title = self._extract_title_from_html(decoded) if resolved_kind == "html" else ""
            raw = {"html": decoded} if options.include_raw_html and resolved_kind == "html" else None
            metadata = self._metadata(
                request=request,
                options=options,
                kind=resolved_kind,
                text=text,
                title=title,
                final_url=response_metadata.get("final_url", validated_url),
                content_type=content_type,
                status_code=response_metadata.get("status_code"),
                duration_ms=elapsed_ms(start_ms),
                truncated=truncated,
                headers=response_metadata.get("headers", {}),
            )
            result = ContentExtractionResult(status="success", text=text, metadata=metadata, raw=raw, correlation_id=request.correlation_id)
            return result if return_result else self._to_action_result(result, action="extract_remote_text")
        except Exception as exc:
            error = wrap_browser_exception(exc, action="extract_remote_text", context={"url": url, "kind": kind}, default_error_cls=ContentExtractionError)
            if options.raise_on_failure:
                raise error from exc
            result = self._build_error_result(error, request=request, options=options, duration_ms=elapsed_ms(start_ms), kind=kind)
            return result if return_result else self._to_action_result(result, action="extract_remote_text")

    # ------------------------------------------------------------------
    # Classification and fetch helpers
    # ------------------------------------------------------------------
    def classify(
        self,
        *,
        url: str = "",
        result: Optional[Mapping[str, Any]] = None,
        driver=None,
        headers: Optional[Mapping[str, Any]] = None,
        content_type: Optional[str] = None,
        requested_kind: str = "auto",
        options: Optional[ContentHandlingOptions] = None,
    ) -> str:
        """Classify content kind from explicit request, URL, headers, result, and driver."""

        options = options or self.options
        explicit = normalize_content_kind(requested_kind or "auto")
        if explicit != "auto":
            return explicit

        candidate_url = url or ""
        if not candidate_url and result:
            candidate_url = str(result.get("link") or result.get("url") or result.get("href") or "")
        if not candidate_url and driver is not None:
            candidate_url = safe_driver_url(driver)

        if options.classify_from_headers:
            header_type = content_type or get_header_value(headers, "content-type")
            header_kind = classify_content_type(header_type)
            if header_kind != "unknown":
                return header_kind

        lower_url = str(candidate_url or "").lower()
        if options.classify_from_url and lower_url:
            if lower_url.endswith(".pdf") or "/pdf/" in lower_url or lower_url.startswith("data:application/pdf"):
                return "pdf"
            if "arxiv.org" in lower_url:
                return "arxiv"
            guessed = guess_kind_from_url(lower_url)
            if guessed != "unknown":
                return guessed

        if driver is not None:
            page_url = safe_driver_url(driver).lower()
            if "arxiv.org" in page_url:
                return "arxiv"
            return "page"

        return "unknown"

    def _fetch_bytes(self, url: str, *, options: ContentHandlingOptions, expected_kind: str = "auto") -> Tuple[bytes, Dict[str, Any]]:
        last_error: Optional[BaseException] = None
        for attempt in range(options.retries + 1):
            try:
                timeout = (options.connect_timeout, options.read_timeout)
                response = self.session.get(
                    url,
                    timeout=timeout,
                    allow_redirects=options.follow_redirects,
                    verify=options.verify_ssl,
                    stream=options.stream_downloads,
                )
                response.raise_for_status()
                body = self._read_response_bytes(response, options=options)
                content_type = response.headers.get("content-type", "")
                metadata = {
                    "final_url": response.url,
                    "status_code": response.status_code,
                    "content_type": content_type,
                    "headers": dict(response.headers),
                    "downloaded_bytes": len(body),
                    "expected_kind": expected_kind,
                }
                return body, metadata
            except Exception as exc:
                last_error = exc
                if attempt >= options.retries:
                    break
                delay = calculate_backoff_delay(
                    attempt_index=attempt,
                    base_delay=options.retry_backoff_base,
                    max_delay=options.retry_backoff_max,
                    multiplier=options.retry_backoff_multiplier,
                    jitter=options.retry_jitter,
                )
                time_module.sleep(delay)
        raise wrap_browser_exception(
            last_error or RuntimeError("Unknown fetch failure"),
            action="fetch_content",
            message=f"Failed to fetch content from {url}",
            context={"url": url, "expected_kind": expected_kind},
            default_error_cls=HTTPRequestError,
        )

    def _read_response_bytes(self, response: requests.Response, *, options: ContentHandlingOptions) -> bytes:
        if not options.stream_downloads:
            content = response.content
            if len(content) > options.max_download_bytes:
                raise ContentExtractionError(
                    "Downloaded content exceeded configured byte limit",
                    context={"max_bytes": options.max_download_bytes, "downloaded_bytes": len(content)},
                )
            return content

        buffer = _LimitedBytesBuffer(options.max_download_bytes)
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if chunk:
                buffer.write(chunk)
        return buffer.getvalue()

    def _validate_remote_url(self, url: str) -> str:
        return validate_url(normalize_url(url), field_name="url", allowed_schemes=("http", "https"))

    # ------------------------------------------------------------------
    # Low-level extraction helpers
    # ------------------------------------------------------------------
    def _extract_pdf_text(self, pdf_bytes: bytes, *, options: ContentHandlingOptions) -> Tuple[str, int, bool]:
        try:
            with BytesIO(pdf_bytes) as pdf_file:
                reader = PdfReader(pdf_file)
                if getattr(reader, "is_encrypted", False) and options.pdf_password:
                    reader.decrypt(options.pdf_password)
                page_count = len(reader.pages)
                max_pages = options.pdf_max_pages or page_count
                selected_pages = reader.pages[:max_pages]
                text_parts = []
                for index, page in enumerate(selected_pages):
                    try:
                        text_parts.append(page.extract_text() or "")
                    except Exception as page_exc:
                        logger.warning("Failed to extract text from PDF page %s: %s", index, page_exc)
                        text_parts.append("")
                return "\n".join(text_parts), page_count, page_count > max_pages
        except Exception as exc:
            raise PDFExtractionError("Failed to extract PDF text", cause=exc) from exc

    def _extract_arxiv_abstract_from_driver(self, driver, *, options: ContentHandlingOptions) -> str:
        for selector in ARXIV_ABSTRACT_SELECTORS:
            try:
                element = driver.find_element(By.CSS_SELECTOR, selector)
                text = element.get_attribute("content") if selector.startswith("meta") else element.text
                text = text or ""
                if text:
                    return clean_arxiv_abstract(text, strip_label=options.arxiv_strip_abstract_label)
            except Exception:
                continue
        return ""

    def _extract_arxiv_abstract_from_html(self, html: str, *, options: ContentHandlingOptions) -> str:
        if not html:
            return ""
        meta_match = re.search(r'<meta\s+[^>]*name=["\']citation_abstract["\'][^>]*content=["\']([^"\']+)["\'][^>]*>', html, re.I | re.S)
        if meta_match:
            return clean_arxiv_abstract(unescape_html(meta_match.group(1)), strip_label=options.arxiv_strip_abstract_label)
        block_match = re.search(r'<blockquote[^>]*class=["\'][^"\']*abstract[^"\']*["\'][^>]*>(.*?)</blockquote>', html, re.I | re.S)
        if block_match:
            return clean_arxiv_abstract(html_to_readable_text(block_match.group(1)), strip_label=options.arxiv_strip_abstract_label)
        return ""

    def _body_text_from_driver(self, driver) -> str:
        try:
            return driver.find_element(By.TAG_NAME, "body").text or ""
        except Exception as exc:
            raise PageSnapshotError("Failed to read body text from driver", cause=exc) from exc

    def _decode_bytes(self, body: bytes, *, content_type: str = "", headers: Optional[Mapping[str, Any]] = None) -> str:
        charset = extract_charset(content_type) or extract_charset(get_header_value(headers, "content-type")) or "utf-8"
        try:
            return body.decode(charset, errors="replace")
        except LookupError:
            return body.decode("utf-8", errors="replace")

    def _extract_title_from_html(self, html: str) -> str:
        if not html:
            return ""
        meta_match = re.search(r'<meta\s+[^>]*name=["\']citation_title["\'][^>]*content=["\']([^"\']+)["\'][^>]*>', html, re.I | re.S)
        if meta_match:
            return normalize_whitespace(unescape_html(meta_match.group(1)))
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
        if title_match:
            return normalize_whitespace(unescape_html(strip_html_tags(title_match.group(1))))
        return ""

    def _prepare_text(self, text: str, *, options: ContentHandlingOptions) -> Tuple[str, bool]:
        prepared = text or ""
        if options.normalize_text:
            prepared = normalize_newlines(prepared)
        if options.collapse_whitespace:
            prepared = normalize_whitespace(prepared)
        truncated = len(prepared) > options.max_text_chars
        prepared = truncate_text(prepared, options.max_text_chars)
        if options.min_extracted_chars and len(prepared) < options.min_extracted_chars:
            logger.debug("Extracted text shorter than configured minimum: %s", len(prepared))
        return prepared, truncated

    def _handle_unsupported(self, *, request: ContentRequest, options: ContentHandlingOptions, kind: str) -> ContentExtractionResult:
        if options.unsupported_content_policy == "error":
            raise UnsupportedContentTypeError("Unsupported content kind", context={"kind": kind, "url": request.url})
        text = "" if options.unsupported_content_policy == "skip" else str((request.result or {}).get("text") or "")
        metadata = self._metadata(request=request, options=options, kind=kind, text=text, extra={"unsupported_policy": options.unsupported_content_policy})
        return ContentExtractionResult(status="success", text=text[: options.max_text_chars], metadata=metadata, correlation_id=request.correlation_id)

    # ------------------------------------------------------------------
    # Result/memory helpers
    # ------------------------------------------------------------------
    def _build_success_result(
        self,
        *,
        text: str,
        request: ContentRequest,
        options: ContentHandlingOptions,
        kind: str,
        source: str,
        metadata_extra: Optional[Mapping[str, Any]] = None,
    ) -> ContentExtractionResult:
        prepared, truncated = self._prepare_text(text, options=options)
        metadata = self._metadata(request=request, options=options, kind=kind, text=prepared, truncated=truncated, extra={"source": source, **dict(metadata_extra or {})})
        return ContentExtractionResult(status="success", text=prepared, metadata=metadata, correlation_id=request.correlation_id)

    def _build_error_result(
        self,
        error: BrowserError,
        *,
        request: ContentRequest,
        options: ContentHandlingOptions,
        duration_ms: Optional[float] = None,
        kind: Optional[str] = None,
    ) -> ContentExtractionResult:
        text = "" if options.safe_result_text_on_error else str(error)
        metadata = self._metadata(request=request, options=options, kind=kind or request.kind, text=text, duration_ms=duration_ms, extra={"failed": True})
        return ContentExtractionResult(
            status="error",
            text=text,
            metadata=metadata,
            error=error.to_result(action=CONTENT_ACTION, redact=True).get("error") or error.to_dict(redact=True),
            correlation_id=request.correlation_id,
        )

    def _replace_duration(self, result: ContentExtractionResult, duration_ms: float) -> ContentExtractionResult:
        metadata_dict = result.metadata.to_dict()
        metadata_dict["duration_ms"] = duration_ms
        metadata = ContentMetadata(**metadata_dict)
        return ContentExtractionResult(status=result.status, text=result.text, metadata=metadata, error=result.error, raw=result.raw, correlation_id=result.correlation_id)

    def _metadata(
        self,
        *,
        request: ContentRequest,
        options: ContentHandlingOptions,
        kind: str,
        text: str,
        final_url: str = "",
        content_type: str = "",
        status_code: Optional[int] = None,
        title: str = "",
        duration_ms: Optional[float] = None,
        truncated: bool = False,
        page_count: Optional[int] = None,
        page_limit_applied: bool = False,
        headers: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> ContentMetadata:
        url = request.url or final_url or ""
        normalized_final_url = normalize_url(final_url) if final_url else normalize_url(url) if url else ""
        domain = safe_domain_from_url(normalized_final_url or url)
        header_payload: Dict[str, Any] = {}
        if options.include_headers and headers:
            header_payload = redact_mapping(dict(headers)) if options.redact_headers else safe_serialize(dict(headers))
        return ContentMetadata(
            url=normalize_url(url) if url else "",
            final_url=normalized_final_url,
            domain=domain,
            kind=normalize_content_kind(kind),
            content_type=content_type,
            status_code=status_code,
            title=normalize_whitespace(title),
            source=request.source,
            text_length=len(text or ""),
            truncated=truncated,
            fingerprint=fingerprint_text(text or ""),
            duration_ms=duration_ms,
            page_count=page_count,
            page_limit_applied=page_limit_applied,
            headers=header_payload,
            extra=dict(extra or {}),
        )

    def _to_action_result(self, extraction: ContentExtractionResult, *, action: str) -> Dict[str, Any]:
        if extraction.status == "success":
            return success_result(
                action=action,
                message="Content extracted successfully",
                data={"text": extraction.text, "metadata": extraction.metadata.to_dict()},
                metadata={"correlation_id": extraction.correlation_id},
                duration_ms=extraction.metadata.duration_ms,
                correlation_id=extraction.correlation_id,
            )
        return error_result(
            action=action,
            message="Content extraction failed",
            error=extraction.error or {},
            metadata={"content_metadata": extraction.metadata.to_dict(), "correlation_id": extraction.correlation_id},
            duration_ms=extraction.metadata.duration_ms,
            correlation_id=extraction.correlation_id,
        )

    def _remember_result(self, result: ContentExtractionResult, *, request: ContentRequest, options: ContentHandlingOptions) -> None:
        try:
            if not options.remember_results or not getattr(self, "memory", None):
                return
            key_source = result.metadata.final_url or request.url or result.correlation_id
            key = f"content:{stable_hash(key_source, length=16)}"
            value = result.to_dict(include_raw=False)
            tags = ["content", result.metadata.kind]
            if result.metadata.domain:
                tags.append(result.metadata.domain)
            if hasattr(self.memory, "put"):
                self.memory.put(
                    key=key,
                    value=value,
                    namespace=options.memory_namespace,
                    kind="page_content" if result.metadata.kind in {"page", "html"} else result.metadata.kind,
                    tags=tags,
                    url=result.metadata.final_url or result.metadata.url,
                    source="content_handling",
                    ttl_seconds=options.memory_ttl_seconds,
                    metadata={"correlation_id": result.correlation_id, "content_kind": result.metadata.kind},
                )
            elif hasattr(self.memory, "remember_action"):
                self.memory.remember_action(
                    action="extract_content",
                    result=value,
                    request={"url": result.metadata.url, "kind": result.metadata.kind},
                )
        except Exception as exc:
            logger.debug("Failed to remember content extraction result: %s", exc)

    def _safe_page_snapshot(self, driver) -> Dict[str, Any]:
        try:
            snapshot = page_snapshot(driver)
            return snapshot.to_dict() if hasattr(snapshot, "to_dict") else safe_serialize(snapshot)
        except Exception:
            return {
                "url": safe_driver_url(driver),
                "title": safe_driver_title(driver),
                "text": truncate_text(self._body_text_from_driver(driver), 2_000),
            }


# ---------------------------------------------------------------------------
# Module-level content-specific helpers
# ---------------------------------------------------------------------------
def normalize_content_kind(kind: Any) -> str:
    text = str(kind or "auto").lower().strip().replace("-", "_")
    aliases = {
        "application_pdf": "pdf",
        "pdf_document": "pdf",
        "webpage": "html",
        "web_page": "html",
        "page_snapshot": "page",
        "plain_text": "text",
        "txt": "text",
        "md": "markdown",
        "xml_document": "xml",
        "json_document": "json",
    }
    text = aliases.get(text, text)
    return text if text in SUPPORTED_CONTENT_KINDS else "unknown"


def classify_content_type(content_type: Any) -> str:
    value = str(content_type or "").split(";", 1)[0].lower().strip()
    if not value:
        return "unknown"
    if value in PDF_MIME_TYPES:
        return "pdf"
    if value in HTML_MIME_TYPES:
        return "html"
    if value == "application/json" or value.endswith("+json"):
        return "json"
    if value in {"application/xml", "text/xml"} or value.endswith("+xml"):
        return "xml"
    if value in {"text/markdown", "text/x-markdown"}:
        return "markdown"
    if value.startswith(TEXTUAL_MIME_PREFIXES) or value in TEXTUAL_MIME_TYPES:
        return "text"
    return "unknown"


def guess_content_type(url: str) -> str:
    guessed, _ = mimetypes.guess_type(url or "")
    return guessed or ""


def guess_kind_from_url(url: str) -> str:
    content_type = guess_content_type(url)
    kind = classify_content_type(content_type)
    if kind != "unknown":
        return kind
    path = urlparse(url or "").path.lower()
    if path.endswith(".md"):
        return "markdown"
    if path.endswith(".txt"):
        return "text"
    return "unknown"


def get_header_value(headers: Optional[Mapping[str, Any]], name: str) -> str:
    if not headers:
        return ""
    wanted = name.lower()
    for key, value in headers.items():
        if str(key).lower() == wanted:
            return str(value or "")
    return ""


def extract_charset(content_type: Any) -> str:
    match = re.search(r"charset=([^;]+)", str(content_type or ""), re.I)
    return match.group(1).strip().strip('"\'') if match else ""


def strip_html_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html or "")


def unescape_html(text: str) -> str:
    # Avoid importing html as a module name into this namespace because helpers
    # often use an ``html`` variable for document text.
    import html as html_module

    return html_module.unescape(text or "")


def html_to_readable_text(html: str) -> str:
    if not html:
        return ""
    cleaned = re.sub(r"(?is)<(script|style|noscript|template).*?>.*?</\1>", " ", html)
    cleaned = re.sub(r"(?i)<br\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"(?i)</(p|div|section|article|header|footer|li|tr|h[1-6])>", "\n", cleaned)
    cleaned = strip_html_tags(cleaned)
    cleaned = unescape_html(cleaned)
    return normalize_newlines(cleaned)


def clean_arxiv_abstract(text: str, *, strip_label: bool = True) -> str:
    value = normalize_newlines(unescape_html(text or ""))
    if strip_label:
        value = re.sub(r"^\s*Abstract\s*:\s*", "", value, flags=re.I)
    return value.strip()


def safe_domain_from_url(url: str) -> str:
    try:
        parsed = urlparse(url or "")
        return (parsed.hostname or "").lower()
    except Exception:
        return ""


def safe_driver_url(driver) -> str:
    try:
        return str(getattr(driver, "current_url", "") or "")
    except Exception:
        return ""


def safe_driver_title(driver) -> str:
    try:
        return str(getattr(driver, "title", "") or "")
    except Exception:
        return ""


def safe_driver_html(driver) -> str:
    try:
        return str(driver.page_source or "")
    except Exception:
        return ""


if __name__ == "__main__":
    print("\n=== Running Content Handling ===\n")
    printer.status("TEST", "Content Handling initialized", "info")

    class FakeElement:
        def __init__(self, text: str = "", attrs: Optional[Mapping[str, str]] = None) -> None:
            self.text = text
            self.attrs = dict(attrs or {})

        def get_attribute(self, name: str) -> str:
            return self.attrs.get(name, "")

    class FakeDriver:
        current_url = "https://arxiv.org/abs/1234.56789"
        title = "A Test Paper"
        page_source = "<html><head><title>A Test Paper</title></head><body><blockquote class='abstract'>Abstract: This is a browser content extraction test.</blockquote></body></html>"

        def find_element(self, by, selector):
            if selector == "blockquote.abstract":
                return FakeElement("Abstract: This is a browser content extraction test.")
            if selector == "body":
                return FakeElement("Abstract: This is a browser content extraction test. Body fallback text.")
            if selector == "title":
                return FakeElement("A Test Paper")
            raise NoSuchElementException(selector)

    handler = ContentHandling(memory=None)

    # Classification tests
    assert normalize_content_kind("PDF") == "pdf"
    assert classify_content_type("application/pdf") == "pdf"
    assert classify_content_type("text/html; charset=utf-8") == "html"
    assert guess_kind_from_url("https://example.com/file.md") == "markdown"

    # HTML text extraction tests
    html_text = html_to_readable_text("<html><body><h1>Hello</h1><script>bad()</script><p>World</p></body></html>")
    assert "Hello" in html_text and "World" in html_text and "bad" not in html_text

    # arXiv driver extraction tests
    arxiv_result = handler.extract_arxiv(driver=FakeDriver(), return_result=True)
    assert arxiv_result.status == "success"
    assert "browser content extraction test" in arxiv_result.text

    # Legacy compatibility tests
    legacy_arxiv_text = ContentHandling.handle_arxiv(FakeDriver())
    assert "browser content extraction test" in legacy_arxiv_text

    result = {"link": "https://arxiv.org/abs/1234.56789", "text": ""}
    processed = ContentHandling.postprocess_if_special(result, FakeDriver())
    assert processed.get("text")
    assert processed.get("content_kind") == "arxiv"

    # Page extraction tests
    page_result = handler.extract_page(FakeDriver(), return_result=True)
    assert page_result.status == "success"
    assert "Body fallback" in page_result.text

    # Unsupported fallback test
    generic = handler._handle_unsupported(
        request=ContentRequest.build(url="https://example.com/blob.bin", kind="unknown", result={"text": "fallback"}),
        options=handler.options,
        kind="unknown",
    )
    assert generic.status == "success"

    print("\n=== Test ran successfully ===\n")
