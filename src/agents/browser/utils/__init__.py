from .browser_driver import *
from .browser_errors import *
from .Browser_helpers import *
from .config_loader import load_global_config, get_config_section

__all__ = [
    # Config loader
    "load_global_config",
    "get_config_section",

    # Browser Driver
    "BrowserDriver",
    "BrowserDriverManager",
    "BrowserDriverOptions",
    "BrowserDriverState",
    "BrowserDriverEvent",

    # Browser Errors – types, severity, payload
    "BrowserErrorType",
    "BrowserErrorSeverity",
    "BrowserErrorPayload",

    # Browser Errors – base and core exceptions
    "BrowserError",
    "UnknownBrowserError",
    "BrowserConfigurationError",
    "MissingConfigurationError",
    "InvalidConfigurationError",
    "BrowserInitializationError",
    "BrowserDriverStartupError",
    "BrowserDependencyError",
    "BrowserValidationError",
    "InvalidURLError",
    "InvalidSelectorError",
    "InvalidTaskPayloadError",
    "MissingRequiredFieldError",
    "InvalidTimeoutError",
    "BrowserDriverError",
    "BrowserTimeoutError",
    "BrowserSessionError",
    "BrowserWindowError",

    # Navigation errors
    "NavigationError",
    "PageLoadTimeoutError",
    "NavigationHistoryError",
    "RedirectError",

    # Search errors
    "SearchError",
    "SearchBoxNotFoundError",
    "SearchResultsNotFoundError",
    "CookieConsentError",

    # Element errors
    "ElementError",
    "ElementNotFoundError",
    "ElementNotVisibleError",
    "ElementNotInteractableError",
    "StaleElementError",
    "ShadowDomError",

    # Click errors
    "ClickError",
    "ClickInterceptedError",
    "JavaScriptClickError",
    "SpecialElementHandlingError",

    # Typing errors
    "BrowserTypingError",
    "InputClearError",
    "InputSendKeysError",

    # Scroll errors
    "ScrollError",
    "InvalidScrollTargetError",

    # Clipboard errors
    "ClipboardError",
    "CopyError",
    "CutError",
    "PasteError",
    "ClipboardValidationError",
    "ClipboardReadError",
    "ClipboardWriteError",
    "ClipboardStrategyError",
    "ClipboardVerificationError",

    # Content extraction errors
    "ContentExtractionError",
    "PDFExtractionError",
    "ArxivExtractionError",
    "PageSnapshotError",
    "UnsupportedContentTypeError",

    # Security errors
    "BrowserSecurityError",
    "CaptchaDetectedError",
    "BotDetectionError",
    "RateLimitError",
    "PermissionDeniedError",

    # Workflow errors
    "WorkflowError",
    "WorkflowValidationError",
    "UnsupportedWorkflowActionError",
    "WorkflowStepFailedError",

    # Retry and backoff errors
    "RetryError",
    "RetryExhaustedError",
    "BackoffError",

    # Network errors
    "NetworkError",
    "NetworkTimeoutError",
    "HTTPRequestError",

    # Task and state errors
    "BrowserTaskError",
    "UnsupportedBrowserTaskError",
    "BrowserStateError",
    "ClosedDriverError",
    "MissingDriverError",
    "JavaScriptExecutionError",

    # Drag‑and‑drop errors
    "DragAndDropError",
    "DragAndDropValidationError",
    "DragSourceNotFoundError",
    "DragTargetNotFoundError",
    "DragSourceNotReadyError",
    "DragTargetNotReadyError",
    "DragStrategyError",
    "Html5DragAndDropError",
    "DragVerificationError",

    # Error helpers and result constructors
    "sanitize_context",
    "wrap_browser_exception",
    "error_result",
    "raise_for_error_result",
    "is_retryable",
    "require",
    "require_mapping",
    "require_sequence",
    "require_non_empty_str",
    "validate_url",
    "validate_css_selector",
    "validate_timeout",
    "validate_choice",
    "validate_config_section",
    "validate_workflow_step",
    "validate_workflow_script",
    "validate_browser_task_payload",
    "element_not_found",
    "captcha_detected",
    "retry_exhausted",

    # Browser helpers – dataclasses
    "RetryPolicy",
    "BrowserURL",
    "ElementSnapshot",
    "PageSnapshot",
    "SearchResultSnapshot",
    "ActionOutcome",

    # Browser helpers – constants
    "BROWSER_ACTIONS",
    "DEFAULT_SEARCH_BOX_SELECTORS",
    "SEARCH_RESULT_LINK_SELECTORS",
    "CONSENT_BUTTON_SELECTORS",
    "CAPTCHA_INDICATORS",

    # Time, IDs, hashing, text normalisation
    "utc_now",
    "utc_now_iso",
    "monotonic_ms",
    "elapsed_ms",
    "new_correlation_id",
    "stable_hash",
    "fingerprint_text",
    "truncate_text",
    "normalize_whitespace",
    "normalize_newlines",
    "compact_text",

    # Safe serialisation and redaction
    "is_sensitive_key",
    "looks_like_secret",
    "redact_scalar",
    "redact_data",
    "redact_mapping",
    "safe_serialize",
    "safe_json_dumps",
    "safe_json_loads",
    "prune_none",
    "merge_dicts",

    # Type coercion and validation
    "ensure_list",
    "coerce_bool",
    "coerce_int",
    "coerce_float",
    "clamp",
    "require_non_empty_string",

    # Result construction
    "make_browser_exception",
    "exception_to_error_payload",
    "success_result",
    "error_result",
    "normalize_result",

    # URL helpers
    "ensure_url_scheme",
    "normalize_url",
    "parse_browser_url",
    "is_valid_url",
    "same_origin",
    "domain_matches",
    "strip_url_fragment",
    "redact_url",
    "join_browser_url",
    "classify_resource_url",
    "is_probably_pdf_url",
    "is_arxiv_url",

    # CSS selector helpers
    "normalize_selector",
    "validate_selector",
    "css_escape_identifier",
    "css_string",
    "attr_selector",
    "id_selector",
    "class_selector",
    "selector_candidates_from_metadata",
    "combine_selectors",

    # Driver and element safety
    "is_web_element",
    "safe_call",
    "safe_get_attribute",
    "element_text",
    "element_classes",
    "element_snapshot",
    "element_metadata",
    "is_interactive_element",
    "is_text_input_element",
    "find_first_element",
    "find_elements",
    "wait_for_page_load",
    "get_document_ready_state",
    "get_current_url",
    "get_page_title",
    "get_body_text",
    "get_page_html",
    "get_viewport",
    "get_performance_timing",
    "detect_captcha_text",
    "detect_captcha",
    "capture_screenshot_b64",
    "page_snapshot",
    "page_snapshot_dict",

    # Search and relevance
    "tokenize_query",
    "score_text_relevance",
    "search_result_from_element",
    "extract_link_snapshots",
    "select_best_link",
    "search_result_dicts",

    # Content helpers
    "html_to_text",
    "limit_content",
    "content_fingerprint",
    "extract_title_from_html",
    "classify_content_type",

    # Retry, backoff, workflow, config
    "calculate_backoff_delay",
    "sleep_backoff",
    "retry_operation",
    "normalize_action_name",
    "normalize_workflow_step",
    "normalize_workflow",
    "normalize_task_payload",
    "get_config_value",
    "get_browser_helper_config",
    "config_bool",
    "config_int",
    "config_float",
    "first_truthy",
    "dedupe_preserve_order",
    "chunk_sequence",
    "safe_filename",
    "infer_file_extension_from_url",
    "build_artifact_name",
    "env_flag",
    "log_result",
]