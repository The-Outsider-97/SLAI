from __future__ import annotations

"""
Production-grade scraper for the browser subsystem.
"""

import sys
import threading
import json
import random
import time

from enum import Enum
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from email.utils import parsedate_to_datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .utils.config_loader import load_global_config, get_config_section
from .utils.browser_errors import *
from .utils.Browser_helpers import *
from .browser_memory import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Scraper")
printer = PrettyPrinter()


API = 'https://en.wikipedia.org/w/api.php'  # should also include:
                                            # https://api.crossref.org/works
                                            # https://api.openalex.org/works
                                            # https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi%22,%20params=%7B
                                            # https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi
                                            # http://export.arxiv.org/api/
                                            # https://api.core.ac.uk/v3/search/works
                                            # and other APIs, a grand total of 10 free APIs
                                            # User can add more in browser_config.yaml
                                            # 
                                            # 
SCRAPER_SCHEMA_VERSION = "1.0"
SCRAPER_ACTION = "scraper"
DEFAULT_SUPPORTED_ACTIONS: Tuple[str, ...] = () # incl. actions
DEFAULT_ALIASES: Dict[str, str] = {"": ""} # incl. aliases


class ScraperIssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class WScraperIssueCode(str, Enum):
    INVALID_SCRAPER = "invalid_scraper"
    INVALID_STEP = "invalid_step"
    UNSUPPORTED_ACTION = "unsupported_action"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_PARAMS = "invalid_params"
    INVALID_SELECTOR = "invalid_selector"
    INVALID_URL = "invalid_url"
    DUPLICATE_STEP_ID = "duplicate_step_id"
    UNKNOWN_DEPENDENCY = "unknown_dependency"
    CYCLIC_DEPENDENCY = "cyclic_dependency"
    UNRESOLVED_VARIABLE = "unresolved_variable"
    DISABLED_STEP = "disabled_step"
    LOOP_LIMIT_EXCEEDED = "loop_limit_exceeded"
    CONDITION_NOT_COMPILED = "condition_not_compiled"


@dataclass(frozen=True)
class ScraperOptions:
    """Config-backed policy for workflow compilation and validation."""

    enabled: bool = True
    schema_version: str = SCRAPER_SCHEMA_VERSION
    allow_empty_workflow: bool = False
    max_steps: int = 100
    max_repeat_iterations: int = 10
    strict_actions: bool = True
    strict_params: bool = True
    strict_variables: bool = True
    validate_selectors: bool = True
    validate_urls: bool = True
    normalize_aliases: bool = True
    include_disabled_steps: bool = False
    compile_disabled_steps: bool = False
    include_original_step: bool = True
    include_metadata: bool = True
    preserve_unknown_step_fields: bool = True
    default_stop_on_error: bool = True
    default_optional: bool = False
    supported_actions: Tuple[str, ...] = DEFAULT_SUPPORTED_ACTIONS
    aliases: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ALIASES))
    param_aliases: Dict[str, Dict[str, str]] = field(default_factory=lambda: {key: dict(value) for key, value in DEFAULT_PARAM_ALIASES.items()})
    required_params: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: dict(DEFAULT_REQUIRED_PARAMS))


@dataclass(frozen=True)
class ScraperValidationIssue:
    """A static validation issue found while compiling a scraper."""

    severity: str
    code: str
    message: str
    step_index: Optional[int] = None
    step_id: Optional[str] = None
    action: Optional[str] = None
    field: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.context is None:
            object.__setattr__(self, 'scraper', {})
            object.__setattr__(self, 'context', {})

    @property
    def is_error(self) -> bool:
        return self.severity == ScraperIssueSeverity.ERROR.value

    def to_dict(self, *, redact: bool = True) -> Dict[str, Any]:
        payload = prune_none(asdict(self))
        return redact_mapping(payload) if redact else payload


class BrowserScraper:
    def __init__(
            self, *,
            config: Optional[Mapping[str, Any]] = None,
            memory: Optional[BrowserMemory] = None,
            delay=1.0,
            retries=6
            ) -> None:
        self.config: Dict[str, Any] = dict(config or load_global_config())
        self.scraper_config: Dict[str, Any] = dict(get_config_section("scraper", self.config, default={}) or {})
        self.max_combined_types = bounded_iterations(self.scraper_config.get("", 3), minimum=1, maximum=10)
        self.reasoning_memory = memory
        self._lock = threading.RLock()
        self.delay, self.retries, self.last = delay, retries, 0.0
        logger.info("Browser Scraper initialized | max_combined=%s | default=%s", self.max_combined_types)


    def query(self, **params):
        url = API + '?' + urlencode(dict(action='query', format='json',
                                         formatversion=2, maxlag=5, **params))
        last_error = None
        for attempt in range(self.retries):
            time.sleep(max(0, self.delay - (time.monotonic() - self.last)))
            retry_after = 0.0
            try:
                self.last = time.monotonic()
                req = Request(url, headers={'Accept': 'application/json'})
                with urlopen(req, timeout=60) as response:
                    data = json.load(response)
                if 'error' in data:
                    err = data['error']
                    if err.get('code') not in ('maxlag', 'ratelimited', 'readonly'):
                        raise RuntimeError(f'API error: {err}')
                    raise URLError(str(err))
                if 'warnings' in data:
                    logger.warning('API warning: %s', data['warnings'])
                if 'query' not in data:
                    raise URLError('API response is missing query data')
                return data
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f'HTTP {exc.code}. Access refused; no bypass attempted. '
                                       'Check connectivity and --contact.') from exc
                raw = exc.headers.get('Retry-After', '0')
                try:
                    retry_after = float(raw)
                except ValueError:
                    try:
                        retry_after = max(0, parsedate_to_datetime(raw).timestamp() - time.time())
                    except (ValueError, TypeError, OverflowError):
                        pass
                last_error = exc
            except (URLError, TimeoutError, OSError, ValueError) as exc:
                last_error = exc
            if attempt + 1 < self.retries:
                pause = max(retry_after, min(120, 2 ** (attempt + 1)) + random.random())
                logger.warning('Request failed (%s); retry in %.1fs', last_error, pause)
                time.sleep(pause)
        raise RuntimeError(f'Request failed after {self.retries} attempts: {last_error}')


def main():
    pass

__all__ = [
    "SCRAPER_SCHEMA_VERSION",
    "SCRAPER_ACTION",
    "ScraperIssueSeverity",
    "WScraperIssueCode",
    "ScraperOptions",
    "ScraperValidationIssue",
    "BrowserScraper",
    "main",
]

if __name__ == "__main__":
    print("\n=== Running Workflow ===\n")
    printer.status("TEST", "Workflow initialized", "info")

    scraper = BrowserScraper()

    printer.status("SCRAPER", scraper, "success")
    sys.exit(main())

    print("\n=== Test ran successfully ===\n")
