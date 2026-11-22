"""Offline tests: canonical packets, paired corpus, scoring and budget ceilings."""

import json
from decimal import Decimal
from urllib.error import HTTPError

import pytest

from benchmarks.lstp import benchmark
from benchmarks.lstp.benchmark import BudgetLedger, parse_answer, score
from benchmarks.lstp.cases import corpus_sha256, generate_cases
from benchmarks.lstp.providers import MODEL_PRICES, ProviderError, choose_models, query


def test_1000_cases_are_unique_and_deterministic():
    first = generate_cases(1000, 20261117)
    second = generate_cases(1000, 20261117)
    assert len(first) == len({c.case_id for c in first}) == 1000
    assert corpus_sha256(first) == corpus_sha256(second)
    assert {c.gold["permission_mode"] for c in first} == {"RO", "SUGGEST", "PREVIEW"}
    assert len({c.gold["resource"] for c in first}) == 20
    assert len({tuple(sorted(c.gold.items())) for c in first}) == 1000


def test_case_bounds():
    with pytest.raises(ValueError):
        generate_cases(0)
    with pytest.raises(ValueError):
        generate_cases(5401)


def test_model_selection_needs_no_list_models_endpoint(monkeypatch):
    for key in ("OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(key, "synthetic-never-sent")

    def unreachable(*args, **kwargs):
        raise AssertionError("model discovery should never make an HTTP call")

    monkeypatch.setattr("benchmarks.lstp.providers._json_request", unreachable)
    selected = choose_models()
    assert selected == {
        "openai": "gpt-6-luna",
        "groq": "openai/gpt-oss-20b",
        "gemini": "gemini-2.5-flash-lite",
    }
    assert MODEL_PRICES["openai"]["gpt-6-luna"].input_usd_per_million == Decimal("0.10")
    assert MODEL_PRICES["openai"]["gpt-6-luna"].output_usd_per_million == Decimal("0.50")
    monkeypatch.setenv("LSTP_OPENAI_MODEL", "unpriced-model")
    with pytest.raises(ProviderError, match="no verified price"):
        choose_models()


def test_generation_http_403_identifies_endpoint_without_leaking_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-key")

    def denied(request, *, timeout):
        raise HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr("benchmarks.lstp.providers.urlopen", denied)
    with pytest.raises(ProviderError) as error:
        query("openai", "gpt-6-luna", "system", "user")
    assert "HTTP 403" in str(error.value)
    assert "api.openai.com POST /v1/chat/completions" in str(error.value)
    assert "private-test-key" not in str(error.value)


def test_randomized_plan_keeps_paired_triads():
    cases = generate_cases(4)
    selected = {"openai": "gpt-6-luna", "gemini": "gemini-2.5-flash-lite"}
    jobs = benchmark._plan(cases, selected, 20261117)
    assert len(jobs) == 4 * 2 * 3
    for i in range(0, len(jobs), 3):
        block = jobs[i : i + 3]
        assert len({case.case_id for case, _, _ in block}) == 1
        assert len({provider for _, provider, _ in block}) == 1
        assert {fmt for _, _, fmt in block} == set(benchmark.FORMATS)


def test_exact_and_partial_scoring():
    case = generate_cases(1)[0]
    perfect = json.dumps(case.gold)
    assert score(perfect, case)["exact_match"] is True
    wrong = dict(case.gold)
    wrong["permission_mode"] = "COMMIT"
    result = score(json.dumps(wrong), case)
    assert result["json_valid"] is True
    assert result["exact_match"] is False
    assert result["field_accuracy"] == pytest.approx(6 / 7)
    assert parse_answer("```json\n" + perfect + "\n```") is None


def test_separate_budgets_and_crash_reservations(tmp_path):
    guard = BudgetLedger(tmp_path / "ledger.json")
    assert guard.reserve("openai:1", "openai", Decimal("4.80"))
    assert not guard.reserve("openai:2", "openai", Decimal("0.30"))
    assert guard.reserve("gemini:1", "gemini", Decimal("4.90"))
    assert guard.remaining("openai") == Decimal("0.20")
    resumed = BudgetLedger(tmp_path / "ledger.json")
    assert resumed.remaining("openai") == Decimal("0.20")
    assert resumed.finalize("openai:1", Decimal(1)) == Decimal(3)
    assert resumed.remaining("openai") == Decimal(2)
    assert not resumed.reserve("groq:1", "groq", Decimal("0.000001"))
    assert resumed.reserve("groq:2", "groq", Decimal(0))


def test_unknown_model_cost_is_rejected(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.json")
    with pytest.raises(KeyError):
        ledger.upper_bound("openai", "nonexistent-model", "hi", max_output_tokens=512)


def test_full_report_and_resume_with_fake_providers(tmp_path, monkeypatch):
    cases = generate_cases(2)
    answers = {
        getattr(case, condition): json.dumps(case.gold)
        for case in cases
        for condition in benchmark.FORMATS
    }
    selected = {
        "openai": "gpt-6-luna",
        "groq": "openai/gpt-oss-20b",
        "gemini": "gemini-2.5-flash-lite",
    }
    monkeypatch.setattr(benchmark, "choose_models", lambda: selected)
    monkeypatch.setattr(benchmark.time, "sleep", lambda seconds: None)
    for name in ("OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(name, "dummy-never-sent")

    def fake_query(provider, model, system, user, **kwargs):
        return {
            "text": answers[user],
            "input_tokens": 100,
            "output_tokens": 40,
            "latency_seconds": 0.1,
        }

    monkeypatch.setattr(benchmark, "query", fake_query)
    output = tmp_path / "result"
    output.mkdir()
    (output / "preflight.json").write_text(
        json.dumps(
            {
                "schema": True,
                "pytest": True,
                "interop_js": True,
                "slai_adapter": True,
                "release_runner": False,
            }
        ),
        encoding="utf-8",
    )
    arguments = [
        "--lstp-root",
        str(tmp_path),
        "--slai-root",
        str(tmp_path),
        "--output",
        str(output),
        "--cases",
        "2",
        "--live",
        "--confirm-free-groq",
        "--confirm-pricing",
    ]
    assert benchmark.main(arguments) == 0
    report = json.loads((output / "final_report.json").read_text(encoding="utf-8"))
    assert report["report_kind"] == "MEASURED"
    assert report["completed_successful_requests"] == 18
    assert report["aggregates"]["gemini/canonical_lstp"]["exact_match_rate"] == 1.0
    assert benchmark.main(arguments) == 0
    assert len((output / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 18


def test_budget_limit_produces_honest_partial_report(tmp_path, monkeypatch):
    monkeypatch.setitem(benchmark.CAPS, "openai", Decimal("0.00001"))
    monkeypatch.setitem(benchmark.CAPS, "gemini", Decimal("0.00001"))
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    monkeypatch.setattr(benchmark.time, "sleep", lambda seconds: None)
    selected = {
        "openai": "gpt-6-luna",
        "groq": "openai/gpt-oss-20b",
        "gemini": "gemini-2.5-flash-lite",
    }
    monkeypatch.setattr(benchmark, "choose_models", lambda: selected)
    answers = {
        getattr(case, condition): json.dumps(case.gold)
        for case in generate_cases(2)
        for condition in benchmark.FORMATS
    }

    def fake_query(provider, model, system, user, **kwargs):
        return {
            "text": answers[user],
            "input_tokens": 1,
            "output_tokens": 1,
            "latency_seconds": 0.1,
        }

    monkeypatch.setattr(benchmark, "query", fake_query)
    output = tmp_path / "budget"
    output.mkdir()
    (output / "preflight.json").write_text(
        json.dumps(dict.fromkeys(("schema", "pytest", "interop_js", "slai_adapter"), True)),
        encoding="utf-8",
    )
    assert (
        benchmark.main(
            [
                "--lstp-root",
                str(tmp_path),
                "--slai-root",
                str(tmp_path),
                "--output",
                str(output),
                "--cases",
                "2",
                "--live",
                "--confirm-free-groq",
                "--confirm-pricing",
            ]
        )
        == 0
    )
    report = json.loads((output / "final_report.json").read_text(encoding="utf-8"))
    assert report["report_kind"] == "BUDGET_LIMITED"
    assert report["stop_reasons"] == {"openai": "budget", "gemini": "budget"}
    assert report["completed_successful_requests"] == 6
