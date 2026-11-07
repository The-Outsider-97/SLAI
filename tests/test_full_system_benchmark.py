"""Offline regression tests for the SLAI v2.4 benchmark harness.

HTTP tests mock the transport and never spend money or send credentials.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

from src.full_system_benchmark import main, parse_levels, parse_models
from src.full_system_benchmark_adapters import (
    HttpBackend, discover_models, load_environment, provider_config, safe_error,
)
from src.full_system_benchmark_cases import Case, SCHEMA, load_cases, score


def test_score_is_objective_and_manual_is_unscored():
    assert score(Case("n", "math", "?", "number", 493), "The answer is 493.")["score"] == 1
    assert score(Case("n", "math", "?", "number", 493), "492 or 493")["score"] == 0
    assert score(Case("e", "lang", "?", "exact", "appel"), "Appel.")["score"] == 1
    assert score(Case("j", "format", "?", "json", {"ok": True}), '```json\n{"ok":true}\n```')["score"] == 1
    assert score(Case("m", "writing", "?", "manual"), "excellent answer")["score"] is None


def test_cases_schema_validation(tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"schema": SCHEMA, "cases": [
        {"id": "repeat", "category": "c", "prompt": "hi", "kind": "exact", "expected": "hi"},
        {"id": "repeat", "category": "c", "prompt": "hi", "kind": "exact", "expected": "hi"},
    ]}))
    with pytest.raises(ValueError, match="duplicated"):
        load_cases(str(cases))


def test_level_and_model_validation():
    assert parse_levels("1,2,4") == [1, 2, 4]
    assert parse_models(["openai=model-x", "groq=some-model"]) == {"openai": "model-x", "groq": "some-model"}
    for levels in ("1,1", "2,1", "0,1", "1,500", "one,two"):
        with pytest.raises(ValueError):
            parse_levels(levels)
    with pytest.raises(ValueError):
        parse_models(["unsupported=m"])


def test_dotenv_override_and_private_doctor(tmp_path, monkeypatch, capsys):
    (tmp_path / ".env").write_text('OPENAI_API_KEY="secret-env-key"\nOPENAI_MODEL=alpha\n')
    monkeypatch.setenv("OPENAI_API_KEY", "secret-process-key")
    values = load_environment(tmp_path)
    assert values["OPENAI_API_KEY"] == "secret-process-key"
    assert provider_config(values)[0] == {"provider": "openai", "key_present": True, "model": "alpha", "ready": True}
    assert safe_error("failed with secret-process-key", [values["OPENAI_API_KEY"]]) == "failed with [REDACTED]"
    assert main(["doctor", "--env-file", str(tmp_path / ".env")]) == 0
    output = capsys.readouterr().out
    assert "secret-process-key" not in output
    assert "secret-env-key" not in output


class _FakeResponse:
    def __init__(self, contents):
        self.contents = json.dumps(contents).encode("utf-8")

    def __enter__(self):
        return io.BytesIO(self.contents)

    def __exit__(self, *ignored):
        return None


@pytest.mark.parametrize("provider,model", [
    ("openai", "model-a"), ("groq", "model-b"),
    ("openrouter", "org/model-c"), ("gemini", "gemini-sample"),
])
def test_api_payload_and_result_are_real_transports(monkeypatch, provider, model):
    observed = {}

    def fake_urlopen(request, timeout):
        observed["url"] = request.full_url
        observed["payload"] = json.loads(request.data)
        observed["headers"] = dict(request.header_items())
        response = (
            {"candidates": [{"content": {"parts": [{"text": "493"}]}}],
             "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}}
            if provider == "gemini" else
            {"choices": [{"message": {"content": "493"}}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
        )
        return _FakeResponse(response)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = HttpBackend(provider=provider, model=model, key="private-test-key", token_limit=47).call("17*29?")
    assert (result.text, result.usage_input, result.usage_output, result.error) == ("493", 5, 2, None)
    assert observed["url"].startswith("https://")
    assert "17*29?" in json.dumps(observed["payload"])
    assert "private-test-key" not in json.dumps(result.to_dict())
    if provider == "gemini":
        assert ":generateContent" in observed["url"]
        assert observed["payload"]["generationConfig"]["maxOutputTokens"] == 47
    else:
        assert observed["payload"]["model"] == model
        assert observed["payload"]["stream"] is False
        field = "max_completion_tokens" if provider == "openai" else "max_tokens"
        assert observed["payload"][field] == 47


def test_http_error_redacts_key(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 429, "rate limit", None,
                                     io.BytesIO(b'provider secret: private-test-key'))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    response = HttpBackend(provider="groq", model="a", key="private-test-key").call("hi")
    assert response.error.startswith("HTTP 429")
    assert "private-test-key" not in response.error


@pytest.mark.parametrize("provider,expected_path", [
    ("openai", "/v1/models"),
    ("groq", "/openai/v1/models"),
    ("openrouter", "/api/v1/models"),
    ("gemini", "/v1beta/models"),
])
def test_discovery_is_read_only(monkeypatch, provider, expected_path):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["method"] = request.get_method()
        seen["url"] = request.full_url
        if provider == "gemini":
            return _FakeResponse({"models": [{"name": "models/gemini-test", "supportedGenerationMethods": ["generateContent"]}]})
        return _FakeResponse({"data": [{"id": "model-test"}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    listing = discover_models(provider, "private-test-key")
    assert seen["method"] == "GET"
    assert seen["url"].endswith(expected_path)
    assert listing["models"]
    assert "private-test-key" not in json.dumps(listing)


def test_full_command_benchmark_emits_native_sldc(tmp_path):
    command = [sys.executable, "-c", "import json,sys; json.load(sys.stdin); print(json.dumps({'text':'493'}))"]
    output = tmp_path / "run"
    result = main([
        "run", "--slai-mode", "command", "--slai-command", json.dumps(command),
        "--case-limit", "1", "--local-levels", "1,2", "--requests-per-level", "2",
        "--no-system-checks", "--max-memory-mb", "0", "--output-dir", str(output),
    ])
    assert result == 0
    assert len((output / "responses.jsonl").read_text().splitlines()) == 5
    assert (output / "benchmark_report.sldc").is_file()
    assert (output / "summary.csv").is_file()
    data = json.loads((output / "results.json").read_text())
    assert data["manifest"]["api_authorized"] is False
    assert data["summaries"][0]["quality"]["objective_accuracy"] == 1.0
    from src.utils.sldc import read_package
    assert read_package(output / "benchmark_report.sldc").document


def test_paid_calls_require_explicit_authorization_and_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-not-used")
    output = tmp_path / "should-not-be-created"
    command = json.dumps([sys.executable, "-c", "print('{}')"])
    args = ["run", "--slai-mode", "command", "--slai-command", command,
            "--no-system-checks", "--output-dir", str(output), "--api-model", "groq=model-z"]
    # Asking for an API model alone is not permission to make paid requests.
    assert main(args + ["--max-api-requests-per-model", "1", "--use-apis"]) == 2
    assert not output.exists()
