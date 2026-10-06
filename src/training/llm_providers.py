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
import logging
import os
import re
import sqlite3
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LOGGER = logging.getLogger("lantra_llm_providers")
ROOT = Path(__file__).resolve().parents[2]


class LLMProviderError(RuntimeError):
    pass


class LLMProviderUnavailable(LLMProviderError):
    pass


class LLMProviderRateLimited(LLMProviderError):
    pass


class LLMProviderResponseError(LLMProviderError):
    pass


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
    api_key: str
    timeout_seconds: float
    base_url: Optional[str] = None


Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]


def _default_transport(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> tuple[int, bytes]:
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
        self.connection = sqlite3.connect(self.path)
        with self.connection:
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
        row = self.connection.execute(
            "SELECT content FROM responses WHERE cache_key=?",
            (cache_key,),
        ).fetchone()
        return str(row[0]) if row else None

    def put(
        self,
        cache_key: str,
        *,
        provider: str,
        model: str,
        content: str,
    ) -> None:
        with self.connection:
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

    def __init__(
        self,
        spec: ProviderSpec,
        *,
        transport: Transport | None = None,
    ) -> None:
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
            raise LLMProviderRateLimited("Provider returned HTTP 429.")
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
        body = json.dumps(
            {
                "model": self.spec.model,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
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
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderResponseError(f"{self.name} returned empty content.")
        return content.strip()


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"

    def __init__(self, spec: ProviderSpec, *, transport: Transport | None = None) -> None:
        endpoint = spec.base_url or "https://api.openai.com/v1/chat/completions"
        super().__init__(spec, endpoint=endpoint, transport=transport)


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"

    def __init__(self, spec: ProviderSpec, *, transport: Transport | None = None) -> None:
        endpoint = spec.base_url or "https://openrouter.ai/api/v1/chat/completions"
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
        endpoint = spec.base_url or "https://api.groq.com/openai/v1/chat/completions"
        super().__init__(spec, endpoint=endpoint, transport=transport)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def complete(self, *, system: str, prompt: str) -> str:
        model = self.spec.model
        base = self.spec.base_url or "https://generativelanguage.googleapis.com/v1beta/models"
        endpoint = f"{base.rstrip('/')}/{model}:generateContent?key={self.spec.api_key}"
        body = json.dumps(
            {
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "responseMimeType": "application/json",
                },
            },
            ensure_ascii=False,
        ).encode("utf-8")
        try:
            status, payload = self.transport(
                endpoint,
                {"Content-Type": "application/json", "Accept": "application/json"},
                body,
                self.spec.timeout_seconds,
            )
        except (URLError, TimeoutError, OSError) as exc:
            raise LLMProviderUnavailable(
                f"gemini transport failed: {type(exc).__name__}"
            ) from exc
        value = self._decode_json(status, payload)
        candidates = value.get("candidates")
        if not isinstance(candidates, Sequence) or not candidates:
            raise LLMProviderResponseError("gemini response has no candidates.")
        first = candidates[0]
        if not isinstance(first, Mapping):
            raise LLMProviderResponseError("gemini candidate is malformed.")
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


_PROVIDER_TYPES: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
    "groq": GroqProvider,
}


def providers_from_config(
    config: Mapping[str, Any],
    *,
    transport: Transport | None = None,
) -> list[LLMProvider]:
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
        model = (os.getenv(model_env, "") if model_env else "").strip()
        if not model:
            model = str(raw.get("default_model", "")).strip()
        if not model:
            LOGGER.warning("LLM provider %s skipped: no model configured.", name)
            continue
        spec = ProviderSpec(
            name=name,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout,
            base_url=(str(raw.get("base_url")).strip() if raw.get("base_url") else None),
        )
        providers.append(provider_type(spec, transport=transport))
    return providers


class ProviderPool:
    """Sequential provider fallback with bounded retry and content cache."""

    def __init__(
        self,
        providers: Sequence[LLMProvider],
        *,
        cache: LLMCache | None = None,
        retry_limit: int = 2,
        backoff_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.providers = list(providers)
        self.cache = cache
        self.retry_limit = max(0, int(retry_limit))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self.sleep = sleep

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        exclude: Iterable[str] = (),
    ) -> LLMResult | None:
        excluded = {str(value).casefold() for value in exclude}
        for provider in self.providers:
            if provider.name.casefold() in excluded:
                continue
            cache_key = LLMCache.key(provider.name, provider.model, system, prompt)
            if self.cache is not None:
                cached = self.cache.get(cache_key)
                if cached is not None:
                    return LLMResult(provider.name, provider.model, cached, cached=True)

            for attempt in range(self.retry_limit + 1):
                try:
                    content = provider.complete(system=system, prompt=prompt)
                    if self.cache is not None:
                        self.cache.put(
                            cache_key,
                            provider=provider.name,
                            model=provider.model,
                            content=content,
                        )
                    return LLMResult(provider.name, provider.model, content, cached=False)
                except (LLMProviderRateLimited, LLMProviderUnavailable) as exc:
                    LOGGER.warning(
                        "LLM provider %s attempt %d/%d unavailable: %s",
                        provider.name,
                        attempt + 1,
                        self.retry_limit + 1,
                        exc,
                    )
                    if attempt < self.retry_limit and self.backoff_seconds > 0:
                        self.sleep(self.backoff_seconds * (2 ** attempt))
                except LLMProviderResponseError as exc:
                    LOGGER.warning(
                        "LLM provider %s returned unusable content: %s",
                        provider.name,
                        exc,
                    )
                    break
                except Exception as exc:
                    LOGGER.exception(
                        "LLM provider %s failed unexpectedly: %s",
                        provider.name,
                        exc,
                    )
                    break
        return None


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
