"""Provider-isolated external LLM support for LANTRA curriculum generation.

No provider SDK is required. Requests use the Python standard library so the
curriculum pipeline does not gain a heavyweight dependency. Secrets are loaded
from environment variables, with the repository-root .env used only to fill
missing variables. Secret values are never included in logs, cache keys, or
generated examples.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import threading

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

from .utils.training_errors import *
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]


LOGGER = get_logger("lantra_llm_providers")
printer = PrettyPrinter()

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LLMResult:
    provider: str
    model: str
    content: str
    cached: bool = False


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float
    base_url: Optional[str] = None
    max_output_tokens: int = 2048
    temperature: float | None = 0.2
    json_mode: bool = True
    token_limit_field: str = "max_tokens"
    tasks: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.model or self.timeout_seconds <= 0 or self.max_output_tokens <= 0:
            raise ValueError("Provider requires model and positive timeout/output token limit.")
        if self.token_limit_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Unsupported provider token_limit_field.")


Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]


def _default_transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    request = Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return int(getattr(response, "status", 200)), response.read()
    except HTTPError as exc:
        return int(exc.code), exc.read()


def load_root_env(root: Path = ROOT) -> dict[str, str]:
    """Load root .env values without overwriting process environment."""
    path = root / ".env"
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        values[key] = value
    return values


def resolve_secret(name: str, alternate: str | None = None) -> str:
    root_values = load_root_env()
    for key in (name, alternate):
        if not key:
            continue
        value = os.getenv(key)
        if value is None:
            value = root_values.get(key)
        if value and value.strip():
            return value.strip()
    return ""


class LLMCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        with self.lock, self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS responses(
                    cache_key TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    @staticmethod
    def key(provider: str, model: str, system: str, prompt: str) -> str:
        payload = json.dumps(
            {
                "provider": provider,
                "model": model,
                "system": system,
                "prompt": prompt,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, cache_key: str) -> str | None:
        with self.lock:
            row = self.connection.execute("SELECT content FROM responses WHERE cache_key=?", (cache_key,)).fetchone()
        return str(row[0]) if row else None

    def put(self, cache_key: str, *, provider: str, model: str, content: str) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO responses(cache_key,provider,model,content,created_at) "
                "VALUES(?,?,?,?,?)",
                (cache_key, provider, model, content, time.time()),
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "LLMCache":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class LLMProvider:
    name = "base"

    def __init__(self, spec: ProviderSpec, *, transport: Transport | None = None) -> None:
        self.spec = spec
        self.transport = transport or _default_transport

    @property
    def model(self) -> str:
        return self.spec.model

    def complete(self, *, system: str, prompt: str) -> str:
        raise NotImplementedError

    @staticmethod
    def _decode_json(status: int, payload: bytes) -> Mapping[str, Any]:
        if status == 429:
            # Inspect error codes, never echo provider error bodies or secrets.
            if any(code in payload.lower() for code in (b"insufficient_quota", b"billing_hard_limit")):
                raise LLMProviderQuotaError("Provider quota exhausted.")
            raise LLMProviderRateLimited("Provider returned HTTP 429.")
        if status in {401, 403, 404}:
            raise LLMProviderConfigurationError(f"Provider authentication/model HTTP {status}.")
        if status == 402:
            raise LLMProviderQuotaError("Provider billing quota exhausted.")
        if status in {400, 413, 422}:
            raise LLMProviderRequestError(f"Provider rejected request HTTP {status}.")
        if status in {408, 425, 500, 502, 503, 504}:
            raise LLMProviderUnavailable(f"Provider returned retryable HTTP {status}.")
        if not 200 <= status < 300:
            raise LLMProviderUnavailable(f"Provider returned HTTP {status}.")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMProviderResponseError("Provider returned malformed JSON.") from exc
        if not isinstance(value, Mapping):
            raise LLMProviderResponseError("Provider response must be a JSON object.")
        return value


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        spec: ProviderSpec,
        *,
        endpoint: str,
        transport: Transport | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(spec, transport=transport)
        self.endpoint = endpoint
        self.extra_headers = dict(extra_headers or {})

    def complete(self, *, system: str, prompt: str) -> str:
        request_body = {
            "model": self.spec.model,
            self.spec.token_limit_field: self.spec.max_output_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        if self.spec.temperature is not None:
            request_body["temperature"] = self.spec.temperature
        if self.spec.json_mode:
            request_body["response_format"] = {"type": "json_object"}
        body = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.spec.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.extra_headers,
        }
        try:
            status, payload = self.transport(
                self.endpoint,
                headers,
                body,
                self.spec.timeout_seconds,
            )
        except (URLError, TimeoutError, OSError) as exc:
            raise LLMProviderUnavailable(
                f"{self.name} transport failed: {type(exc).__name__}"
            ) from exc
        value = self._decode_json(status, payload)
        choices = value.get("choices")
        if not isinstance(choices, Sequence) or not choices:
            raise LLMProviderResponseError(f"{self.name} response has no choices.")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise LLMProviderResponseError(f"{self.name} choice is malformed.")
        message = first.get("message", {})
        if not isinstance(message, Mapping):
            raise LLMProviderResponseError(f"{self.name} message is malformed.")
        if message.get("refusal") or first.get("finish_reason") == "content_filter":
            raise LLMProviderRequestError("Provider refused this request.")
        if first.get("finish_reason") == "length":
            raise LLMProviderResponseError("Provider output was truncated.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderResponseError(f"{self.name} returned empty content.")
        return content.strip()


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"

    def __init__(self, spec: ProviderSpec, *, transport: Transport | None = None) -> None:
        endpoint = normalize_endpoint("openai", spec.base_url)
        super().__init__(spec, endpoint=endpoint, transport=transport)


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"

    def __init__(self, spec: ProviderSpec, *, transport: Transport | None = None) -> None:
        endpoint = normalize_endpoint("openrouter", spec.base_url)
        super().__init__(
            spec,
            endpoint=endpoint,
            transport=transport,
            extra_headers={
                "HTTP-Referer": "https://github.com/The-Outsider-97/SLAI",
                "X-Title": "SLAI LANTRA",
            },
        )


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"

    def __init__(self, spec: ProviderSpec, *, transport: Transport | None = None) -> None:
        endpoint = normalize_endpoint("groq", spec.base_url)
        super().__init__(spec, endpoint=endpoint, transport=transport)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def complete(self, *, system: str, prompt: str) -> str:
        model = self.spec.model
        base = normalize_endpoint("gemini", self.spec.base_url)
        endpoint = f"{base}/{model.removeprefix('models/')}:generateContent"
        body = json.dumps(
            {
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": self.spec.max_output_tokens,
                    **({"temperature": self.spec.temperature} if self.spec.temperature is not None else {}),
                    **({"responseMimeType": "application/json"} if self.spec.json_mode else {}),
                },
            },
            ensure_ascii=False,
        ).encode("utf-8")
        try:
            status, payload = self.transport(
                endpoint,
                {"Content-Type": "application/json", "Accept": "application/json", "x-goog-api-key": self.spec.api_key},
                body,
                self.spec.timeout_seconds,
            )
        except (URLError, TimeoutError, OSError) as exc:
            raise LLMProviderUnavailable(
                f"gemini transport failed: {type(exc).__name__}"
            ) from exc
        value = self._decode_json(status, payload)
        if value.get("promptFeedback", {}).get("blockReason"):
            raise LLMProviderRequestError("Gemini blocked this request.")
        candidates = value.get("candidates")
        if not isinstance(candidates, Sequence) or not candidates:
            raise LLMProviderResponseError("gemini response has no candidates.")
        first = candidates[0]
        if not isinstance(first, Mapping):
            raise LLMProviderResponseError("gemini candidate is malformed.")
        if first.get("finishReason") not in {None, "STOP"}:
            raise LLMProviderResponseError("Gemini did not finish a usable response.")
        content = first.get("content", {})
        parts = content.get("parts", []) if isinstance(content, Mapping) else []
        if not isinstance(parts, Sequence):
            raise LLMProviderResponseError("gemini response parts are malformed.")
        texts = [
            str(part.get("text"))
            for part in parts
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        ]
        result = "\n".join(texts).strip()
        if not result:
            raise LLMProviderResponseError("gemini returned empty content.")
        return result


def normalize_endpoint(name: str, value: str | None) -> str:
    defaults = {
        "openai": "https://api.openai.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    base = (value or defaults[name]).strip().rstrip("/")
    parsed = urlsplit(base)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise LLMProviderConfigurationError(f"Invalid HTTPS endpoint for {name}; check its URL variable.")
    if name == "gemini":
        if not parsed.path:
            base += "/v1beta/models"
        elif parsed.path.rstrip("/") in {"/v1", "/v1beta"}:
            base += "/models"
        elif not parsed.path.endswith("/models"):
            raise LLMProviderConfigurationError("Gemini URL must end at /models or the API root.")
        return base
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


_PROVIDER_TYPES: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
    "groq": GroqProvider,
}


def providers_from_config(config: Mapping[str, Any], *, transport: Transport | None = None) -> list[LLMProvider]:
    timeout = float(config.get("provider_timeout_seconds", 45))
    rows = config.get("providers", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return []
    providers: list[LLMProvider] = []
    for raw in rows:
        if not isinstance(raw, Mapping) or not bool(raw.get("enabled", True)):
            continue
        name = str(raw.get("name", "")).strip().casefold()
        provider_type = _PROVIDER_TYPES.get(name)
        if provider_type is None:
            LOGGER.warning("Ignoring unknown LANTRA LLM provider %s.", name or "<blank>")
            continue
        key_env = str(raw.get("api_key_env", "")).strip()
        alternate_env = str(raw.get("alternate_api_key_env", "")).strip() or None
        api_key = resolve_secret(key_env, alternate_env)
        if not api_key:
            LOGGER.info("LLM provider %s unavailable: configured API key is absent.", name)
            continue
        model_env = str(raw.get("model_env", "")).strip()
        model = (resolve_secret(model_env) if model_env else "").strip()
        if not model:
            model = str(raw.get("default_model", "")).strip()
        if not model:
            LOGGER.warning("LLM provider %s skipped: no model configured.", name)
            continue
        endpoint = resolve_secret(str(raw.get("url_env", name.upper() + "_URL"))) or raw.get("base_url")
        try:
            endpoint = normalize_endpoint(name, endpoint)
        except LLMProviderConfigurationError:
            LOGGER.warning("LLM provider %s skipped: invalid URL configuration.", name)
            continue
        spec = ProviderSpec(
            name=name,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout,
            base_url=endpoint,
            max_output_tokens=int(raw.get("max_output_tokens", config.get("max_output_tokens", 2048))),
            temperature=raw.get("temperature", 0.2),
            json_mode=bool(raw.get("json_mode", True)),
            token_limit_field=str(raw.get("token_limit_field", "max_tokens")),
            tasks=tuple(raw.get("tasks", ())),
        )
        providers.append(provider_type(spec, transport=transport))
    order: list[str] = list(config.get("provider_order", ("openai", "groq", "gemini", "openrouter")))
    if len(set(order)) != len(order) or any(name not in _PROVIDER_TYPES for name in order):
        raise ValueError("provider_order must contain unique supported provider names.")
    return sorted((p for p in providers if p.name in order), key=lambda p: order.index(p.name))


class ProviderPool:
    """Durable ordered allocation, shared by all teacher callers.

    Allocation order is transactional; response arrival order is unconstrained.
    Cache hits do not move the cursor. Attempts reserve budget before dispatch;
    uncertain remote outcomes never refund reservations. Reusing the same state
    DB across processes preserves ordering and quotas. SQLite must be on local disk.
    """

    def __init__(self, providers: Sequence[LLMProvider], *, cache: LLMCache | None = None,
                 retry_limit: int = 2, backoff_seconds: float = 2.0,
                 sleep: Callable[[float], None] = time.sleep,
                 state_path: Path | None = None, limits: Mapping[str, Any] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.providers = list(providers)
        if len({p.name for p in providers}) != len(providers):
            raise ValueError("Provider names must be unique.")
        self.cache = cache
        self.retry_limit = max(0, int(retry_limit))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self.sleep, self.clock = sleep, clock
        self.limits = dict(limits or {})
        self.lock = threading.RLock()
        if state_path is None and cache is not None:
            state_path = Path(str(cache.path) + ".scheduler")
        if state_path is not None:
            Path(state_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(state_path) if state_path else ":memory:", timeout=30,
                                  isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS rotation(id INTEGER PRIMARY KEY CHECK(id=1), next_name TEXT);
            INSERT OR IGNORE INTO rotation VALUES(1, '');
            CREATE TABLE IF NOT EXISTS health(name TEXT PRIMARY KEY, identity TEXT, until REAL, disabled INTEGER);
            CREATE TABLE IF NOT EXISTS attempts(
                id INTEGER PRIMARY KEY, request_key TEXT, provider TEXT, started REAL,
                expires REAL, tokens_reserved INTEGER, cost_reserved REAL, status TEXT);
            CREATE INDEX IF NOT EXISTS attempt_time ON attempts(provider, started);
            CREATE TABLE IF NOT EXISTS jobs(
                request_key TEXT PRIMARY KEY, provider TEXT, model TEXT, content TEXT, status TEXT);
        ''')

    @contextmanager
    def _transaction(self):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield self.db
            except BaseException:
                self.db.rollback()
                raise
            else:
                self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def _identity(provider: LLMProvider) -> str:
        spec = provider.spec
        # Only a digest enters health state; credentials never enter cache keys.
        return hashlib.sha256(json.dumps([spec.api_key, spec.model, spec.base_url]).encode()).hexdigest()

    def _request_key(self, system, prompt, excluded, task):
        specs = [{"name": p.name, "model": p.model, "url": p.spec.base_url,
                  "max_output_tokens": p.spec.max_output_tokens, "temperature": p.spec.temperature,
                  "json_mode": p.spec.json_mode, "token_limit_field": p.spec.token_limit_field}
                 for p in self.providers if p.name not in excluded]
        return hashlib.sha256(json.dumps([system, prompt, specs, sorted(excluded), task],
                                        sort_keys=True).encode()).hexdigest()

    def _limit(self, name, key, default: float | int = 0):
        return self.limits.get("per_provider", {}).get(name, {}).get(key, self.limits.get(key, default))

    def _reserve(self, key, system, prompt, excluded, task):
        now = self.clock()
        with self._transaction() as db:
            row = db.execute("SELECT provider,model,content,status FROM jobs WHERE request_key=?", (key,)).fetchone()
            if row and row[3] == "completed":
                return LLMResult(row[0], row[1], row[2], cached=True)
            if row and row[3] == "rejected":
                return None
            db.execute("UPDATE attempts SET status='uncertain' WHERE status='inflight' AND expires<=?", (now,))
            if db.execute("SELECT 1 FROM attempts WHERE request_key=? AND status='inflight'", (key,)).fetchone():
                return None
            names = [p.name for p in self.providers]
            if not names:
                return None
            next_name = db.execute("SELECT next_name FROM rotation WHERE id=1").fetchone()[0]
            start = names.index(next_name) if next_name in names else 0
            # Retries for one logical request stay bounded across restarts.
            count = db.execute("SELECT COUNT(*) FROM attempts WHERE request_key=?", (key,)).fetchone()[0]
            if count >= len(names) * (1 + self.retry_limit):
                return None
            for offset in range(len(names)):
                index = (start + offset) % len(names)
                p = self.providers[index]
                if p.name in excluded or (task and p.spec.tasks and task not in p.spec.tasks):
                    continue
                identity = self._identity(p)
                health = db.execute("SELECT identity,until,disabled FROM health WHERE name=?", (p.name,)).fetchone()
                if health and health[0] == identity and (health[2] or health[1] > now):
                    continue
                # Conservative UTF-8 byte proxy, not billed-token telemetry.
                tokens = len((system + prompt).encode()) + 256 + p.spec.max_output_tokens
                cost = float(self._limit(p.name, "max_request_cost", 0))
                budget = float(self._limit(p.name, "total_cost_budget", 0))
                if budget and cost <= 0:
                    raise ValueError("A monetary budget requires a conservative max_request_cost.")
                checks = [(0, "total_requests", "COUNT(*)", 1),
                          (now - now % 86400, "daily_requests", "COUNT(*)", 1),
                          (now - 60, "requests_per_minute", "COUNT(*)", 1),
                          (0, "total_token_budget", "COALESCE(SUM(tokens_reserved),0)", tokens),
                          (now - 60, "tokens_per_minute", "COALESCE(SUM(tokens_reserved),0)", tokens),
                          (0, "total_cost_budget", "COALESCE(SUM(cost_reserved),0)", cost)]
                permitted = True
                for since, limit_name, expression, reserve in checks:
                    limit = float(self._limit(p.name, limit_name, 0))
                    if limit:
                        used = db.execute(f"SELECT {expression} FROM attempts WHERE provider=? AND started>=?", (p.name, since)).fetchone()[0]
                        if used + reserve > limit:
                            permitted = False
                            break
                active = db.execute("SELECT COUNT(*) FROM attempts WHERE provider=? AND status='inflight'", (p.name,)).fetchone()[0]
                if not permitted or active >= int(self._limit(p.name, "max_concurrency", 1)):
                    continue
                db.execute("UPDATE rotation SET next_name=? WHERE id=1", (names[(index+1) % len(names)],))
                lease = max(60, p.spec.timeout_seconds * 3)
                cursor = db.execute("INSERT INTO attempts(request_key,provider,started,expires,tokens_reserved,cost_reserved,status) VALUES(?,?,?,?,?,?,?)",
                                    (key,p.name,now,now+lease,tokens,cost,"inflight"))
                db.execute("INSERT OR REPLACE INTO jobs VALUES(?,?,?,?,?)", (key,p.name,p.model,None,"pending"))
                return p, cursor.lastrowid
        return None

    def complete(self, *, system: str, prompt: str, exclude: Iterable[str] = (), task: str = "") -> LLMResult | None:
        excluded = {str(value).casefold() for value in exclude}
        key = self._request_key(system, prompt, excluded, task)
        for _ in range(max(1, len(self.providers)) * (self.retry_limit + 1)):
            reservation = self._reserve(key, system, prompt, excluded, task)
            if reservation is None or isinstance(reservation, LLMResult):
                return reservation
            provider, attempt_id = reservation
            LOGGER.info("Teacher dispatch id=%d provider=%s model=%s", attempt_id, provider.name, provider.model)
            try:
                content = provider.complete(system=system, prompt=prompt)
                parse_json_object(content)  # Do not durably cache malformed training payloads.
            except LLMProviderError as exc:
                disabled = isinstance(exc, LLMProviderConfigurationError)
                terminal = isinstance(exc, LLMProviderRequestError)
                cooldown = (float(self._limit(provider.name, "quota_cooldown_seconds", 3600))
                            if isinstance(exc, LLMProviderQuotaError) else
                            float(self._limit(provider.name, "cooldown_seconds", max(30, self.backoff_seconds))))
                with self._transaction() as db:
                    if db.execute("SELECT status FROM attempts WHERE id=?", (attempt_id,)).fetchone()[0] != "inflight":
                        return None  # Fenced stale worker; a newer attempt owns the job.
                    db.execute("UPDATE attempts SET status=? WHERE id=?", (type(exc).__name__, attempt_id))
                    db.execute("INSERT OR REPLACE INTO health VALUES(?,?,?,?)",
                               (provider.name, self._identity(provider), self.clock()+cooldown, int(disabled)))
                    db.execute("UPDATE jobs SET status=? WHERE request_key=?", ("rejected" if terminal else "pending", key))
                LOGGER.warning("Teacher attempt=%d provider=%s failure=%s", attempt_id, provider.name, type(exc).__name__)
                if terminal:
                    return None
                continue
            with self._transaction() as db:
                if db.execute("SELECT status FROM attempts WHERE id=?", (attempt_id,)).fetchone()[0] != "inflight":
                    return None
                db.execute("UPDATE attempts SET status='completed' WHERE id=?", (attempt_id,))
                db.execute("UPDATE jobs SET content=?,status='completed' WHERE request_key=?", (content,key))
                db.execute("DELETE FROM health WHERE name=?", (provider.name,))
            return LLMResult(provider.name, provider.model, content)
        return None

 
    def metrics(self) -> dict[str, Any]:
        with self.lock:
            rows = self.db.execute("SELECT provider,status,COUNT(*),SUM(tokens_reserved),SUM(cost_reserved) FROM attempts GROUP BY provider,status").fetchall()
            pointer = self.db.execute("SELECT next_name FROM rotation WHERE id=1").fetchone()[0]
        return {"next_provider": pointer, "accounting": "conservative reservations, not billed usage",
                "attempts": [{"provider": p,"status": s,"requests": n,"reserved_tokens": t,"reserved_cost": c} for p,s,n,t,c in rows]}


def parse_json_object(content: str) -> Mapping[str, Any]:
    text = str(content or "").strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = re.sub(r"^" + re.escape(fence) + r"(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*" + re.escape(fence) + r"$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMProviderResponseError("LLM content is not valid JSON.") from exc
    if not isinstance(value, Mapping):
        raise LLMProviderResponseError("LLM content must decode to a JSON object.")
    return value


__all__ = [
    "GeminiProvider",
    "GroqProvider",
    "LLMCache",
    "LLMProvider",
    "LLMProviderError",
    "LLMProviderRateLimited",
    "LLMProviderResponseError",
    "LLMProviderUnavailable",
    "LLMResult",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderPool",
    "ProviderSpec",
    "load_root_env",
    "parse_json_object",
    "providers_from_config",
    "resolve_secret",
]
