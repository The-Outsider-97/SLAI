import json
from dataclasses import replace
from decimal import Decimal

import pytest
from lstp import CanonicalizationError, canonical_loads, compact_dumps

from benchmarks.lstp import optimization
from benchmarks.lstp.cases import generate_cases
from src.integrations.lstp_adapter import (
    decode_compact_packet,
    decode_incremental_packet,
    encode_compact_packet,
    encode_incremental_packet,
    extract_action_fields,
    prepare_action_extraction,
)


def prices():
    return {
        "openai": {
            "model": "test-model",
            "input_per_million": "1",
            "output_per_million": "2",
            "source": "https://example.org",
            "verified_at": "2026-11-18",
        }
    }


def test_all_original_1000_cases_roundtrip_and_have_identical_fair_controls():
    report = optimization.offline(generate_cases(1000))
    assert report["lossless_canonical_byte_roundtrips"] == 1000
    assert report["deterministic_projection_exact_matches"] == 1000
    assert report["identical_json_tuple_controls"] == 1000
    assert report["model_correctness"] is None
    assert report["provider_cost_savings"] is None
    assert report["local_tokenizer_counts"] is None


def test_live_resume_does_not_repeat_paid_requests_and_counts_invalid_outputs(
    tmp_path, monkeypatch
):
    cases = generate_cases(2)
    answers = {}
    for case in cases:
        for system, user, tuples in optimization.conditions(case).values():
            answers[(system, user)] = (
                json.dumps(list(case.gold.values())) if tuples else json.dumps(case.gold)
            )
    calls = []

    def fake_query(provider, model, system, user, **kwargs):
        calls.append((system, user))
        return {
            "text": answers[(system, user)],
            "input_tokens": 100,
            "output_tokens": 30,
            "latency_seconds": 0.1,
        }

    monkeypatch.setattr(optimization, "query", fake_query)
    report = optimization.run_live(cases, tmp_path, prices(), Decimal(1), 512, 7)
    assert len(calls) == 18
    assert not report["stops"]
    assert report["metrics"]["openai/projection_tuple"]["successful_tasks"] == 2
    optimization.run_live(cases, tmp_path, prices(), Decimal(1), 512, 7)
    assert len(calls) == 18
    metrics = report["metrics"]
    # Identical usage cannot pass a 25% target, despite perfect correctness.
    assert all(
        row["status"] == "OBSERVED_FAIL" for row in optimization.compare_targets(metrics).values()
    )


def test_provider_error_retains_unknown_usage_and_cannot_pass_targets(tmp_path, monkeypatch):
    def failure(*args, **kwargs):
        raise optimization.ProviderError("synthetic connection failure")

    monkeypatch.setattr(optimization, "query", failure)
    report = optimization.run_live(generate_cases(1), tmp_path, prices(), Decimal(1), 512, 7)
    metric = next(iter(report["metrics"].values()))
    assert metric["cost_complete"] is False
    assert metric["cost_per_success"] is None
    assert report["stops"] == {"openai": "provider_error"}


def test_broken_json_is_scored_and_recorded_instead_of_losing_billable_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(
        optimization,
        "query",
        lambda *args, **kwargs: {
            "text": "{bad",
            "input_tokens": 100,
            "output_tokens": 20,
            "latency_seconds": 0.1,
        },
    )
    report = optimization.run_live(generate_cases(1), tmp_path, prices(), Decimal(1), 512, 7)
    assert len(report["metrics"]) == 9
    assert all(metric["successful_tasks"] == 0 for metric in report["metrics"].values())
    assert all(metric["known_total_tokens"] == 120 for metric in report["metrics"].values())


def test_adapter_uses_real_core_and_keeps_full_source():
    case = generate_cases(1)[0]
    packet = canonical_loads(case.canonical_lstp)
    assert decode_compact_packet(encode_compact_packet(packet)) == packet
    assert encode_compact_packet(packet) == compact_dumps(packet)
    assert extract_action_fields(packet).as_mapping() == case.gold
    assert prepare_action_extraction(
        packet, lambda system, user: len(system) + len(user)
    ).source_sha256
    base = replace(packet, carrier={"large": "x" * 2000})
    encoded = encode_incremental_packet(replace(base, packet_id="p2"), base=base)
    assert decode_incremental_packet(encoded, base=base).packet_id == "p2"
    with pytest.raises(CanonicalizationError):
        decode_incremental_packet(encoded)


def test_unresolved_reservation_is_never_resent_or_counted_as_free(tmp_path, monkeypatch):
    ledger = optimization.BudgetLedger(
        tmp_path / "budget_ledger.json", caps={"openai": Decimal("1")}
    )
    assert ledger.reserve("openai:case-00001:projection_tuple", "openai", Decimal("0.1"))

    def forbidden(*args, **kwargs):
        raise AssertionError("An unresolved paid request must not be repeated")

    monkeypatch.setattr(optimization, "query", forbidden)
    report = optimization.run_live(generate_cases(1), tmp_path, prices(), Decimal("1"), 512, 7)
    assert report["stops"] == {"openai": "unresolved_crash_reservation"}
    assert report["metrics"]["openai/projection_tuple"]["cost_complete"] is False
