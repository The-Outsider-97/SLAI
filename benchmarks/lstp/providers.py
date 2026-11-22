"""Minimal provider clients; no SDK dependency, execution tools, or credential logging."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ModelPrice:
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    source: str


# Standard text API rates researched on 2026-11-17. Must be rechecked before live use.
# No unknown model/pricing fallback is permitted.
MODEL_PRICES: dict[str, dict[str, ModelPrice]] = {
    "openai": {
        "gpt-6-luna": ModelPrice(
            Decimal("0.10"),
            Decimal("0.50"),
            "https://developers.openai.com/api/docs/pricing",
        ),
        "gpt-5.4-mini": ModelPrice(
            Decimal("0.75"),
            Decimal("4.50"),
            "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
        ),
    },
    "gemini": {
        "gemini-2.5-flash-lite": ModelPrice(
            Decimal("0.10"),
            Decimal("0.40"),
            "https://ai.google.dev/gemini-api/docs/pricing",
        ),
        "gemini-3.1-flash-lite": ModelPrice(
            Decimal("0.25"),
            Decimal("1.50"),
            "https://ai.google.dev/gemini-api/docs/pricing",
        ),
    },
    "groq": {
        "openai/gpt-oss-20b": ModelPrice(
            Decimal(0),
            Decimal(0),
            "https://console.groq.com/docs/rate-limits",
        ),
        "openai/gpt-oss-120b": ModelPrice(
            Decimal(0),
            Decimal(0),
            "https://console.groq.com/docs/rate-limits",
        ),
    },
}

KEY_NAMES = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class ProviderError(RuntimeError):
    """A request failed; error strings must never contain credential material."""


def _json_request(
    url: str,
    *,
    key: str,
    headers: dict[str, str],
    body: dict[str, object] | None = None,
    timeout: int = 90,
) -> dict:
    method = "POST" if body is not None else "GET"
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    request = Request(
        url,
        data=payload,
        method=method,
        headers={
            "Accept": "application/json",
            **headers,
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(4_000_001)
    except HTTPError as exc:
        # Never echo response bodies: providers may reflect credential material.
        endpoint = urlsplit(url)
        raise ProviderError(
            f"HTTP {exc.code} from {endpoint.hostname} {method} {endpoint.path}; "
            "check API-key scope, account access, billing, and model availability"
        ) from None
    except URLError as exc:
        raise ProviderError(f"Provider connection failed ({type(exc.reason).__name__})") from None
    if len(raw) > 4_000_000:
        raise ProviderError("Provider response exceeded size limit")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        raise ProviderError("Provider returned invalid JSON") from None
    if not isinstance(value, dict):
        raise ProviderError("Provider returned a non-object response")
    return value


def _credential(provider: str) -> str:
    name = KEY_NAMES[provider]
    value = os.environ.get(name, "").strip()
    if not value:
        raise ProviderError(f"{name} is not configured in the PowerShell session")
    return value


def choose_models(
    providers: tuple[str, ...] = ("openai", "groq", "gemini"),
) -> dict[str, str]:
    """Select known-priced models without requiring the optional list-models grant.

    Providers still enforce model and generation permissions on every request.
    An HTTP 403 there is reported and is never retried through another route.
    Explicit overrides must already have a verified price entry.
    """
    selected: dict[str, str] = {}
    for provider in providers:
        _credential(provider)
        override = os.environ.get(f"LSTP_{provider.upper()}_MODEL", "").strip()
        selected_id = override or next(iter(MODEL_PRICES[provider]))
        if selected_id not in MODEL_PRICES[provider]:
            raise ProviderError(
                f"Model {selected_id!r} for {provider} has no verified price entry. "
                "Add its official rate to the whitelist before running."
            )
        selected[provider] = selected_id
    return selected


def query(
    provider: str, model: str, system: str, user: str, *, max_output_tokens: int = 512
) -> dict[str, object]:
    """Generate a text response; report exact provider usage, if supplied."""
    if provider not in KEY_NAMES or not isinstance(model, str) or not model.strip():
        raise ProviderError("A supported provider and explicit model are required")
    key = _credential(provider)
    started = time.perf_counter()
    if provider in {"openai", "groq"}:
        url = (
            "https://api.openai.com/v1/chat/completions"
            if provider == "openai"
            else "https://api.groq.com/openai/v1/chat/completions"
        )
        response = _json_request(
            url,
            key=key,
            headers={"Authorization": f"Bearer {key}"},
            body={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_completion_tokens": max_output_tokens,
            },
        )
        try:
            raw_text = response["choices"][0]["message"].get("content") or ""
            usage = response["usage"]
            input_tokens = int(usage["prompt_tokens"])
            output_tokens = int(usage["completion_tokens"])
            cached_input_tokens = int(
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            )
            response_model = response.get("model", model)
        except (KeyError, IndexError, TypeError, ValueError):
            raise ProviderError("Provider omitted completion or token usage") from None
    else:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(model, safe='')}:generateContent"
        )
        response = _json_request(
            url,
            key=key,
            headers={"x-goog-api-key": key},
            body={
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": f"System instructions:\n{system}\n\nUser message:\n{user}"}
                        ],
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": max_output_tokens,
                    "temperature": 0,
                },
            },
        )
        try:
            raw_text = "".join(
                p.get("text", "") for p in response["candidates"][0]["content"]["parts"]
            )
            usage = response["usageMetadata"]
            input_tokens = int(usage["promptTokenCount"])
            output_tokens = max(
                int(usage.get("candidatesTokenCount", 0)) + int(usage.get("thoughtsTokenCount", 0)),
                int(usage.get("totalTokenCount", 0)) - input_tokens,
            )
            cached_input_tokens = int(usage.get("cachedContentTokenCount", 0))
            response_model = response.get("modelVersion", model)
        except (KeyError, IndexError, TypeError, ValueError):
            raise ProviderError("Gemini omitted content or token usage") from None
    if input_tokens < 0 or output_tokens < 0 or not 0 <= cached_input_tokens <= input_tokens:
        raise ProviderError("Provider returned invalid token usage")
    if not isinstance(raw_text, str):
        raise ProviderError("Provider returned a non-text response")
    return {
        "text": raw_text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "response_model": response_model,
        "latency_seconds": round(time.perf_counter() - started, 4),
    }


def estimated_usd(provider: str, model: str, input_tokens: int, output_tokens: int) -> Decimal:
    rates = MODEL_PRICES[provider][model]
    return (
        Decimal(input_tokens) * rates.input_usd_per_million
        + Decimal(output_tokens) * rates.output_usd_per_million
    ) / Decimal(1_000_000)
