from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from dependencies.deepmerge import merge_mappings
from dependencies.deepmerge.mapping import RecursiveMappingError


def test_nested_mapping_merge_preserves_unmodified_defaults() -> None:
    base = {
        "safety": {"risk_threshold": 0.35, "max_throughput": 1000},
        "telemetry": {"metrics_collection": True},
    }
    override = {"safety": {"risk_threshold": 0.5}}

    merged = merge_mappings(base, override)

    assert merged == {
        "safety": {"risk_threshold": 0.5, "max_throughput": 1000},
        "telemetry": {"metrics_collection": True},
    }


def test_fallback_values_and_type_conflicts_use_override() -> None:
    base = {
        "items": [1, 2],
        "mode": "safe",
        "nested": {"enabled": True},
        "nullable": {"value": 1},
    }
    override = {
        "items": [3],
        "mode": {"name": "fast"},
        "nested": False,
        "nullable": None,
    }

    assert merge_mappings(base, override) == override


def test_inputs_and_nested_values_are_not_mutated_or_aliased() -> None:
    base = {"a": {"keep": [1, 2], "value": 1}}
    override = {"a": {"value": 2, "new": {"x": 1}}}

    merged = merge_mappings(base, override)
    merged["a"]["keep"].append(3)
    merged["a"]["new"]["x"] = 99

    assert base == {"a": {"keep": [1, 2], "value": 1}}
    assert override == {"a": {"value": 2, "new": {"x": 1}}}


def test_key_order_is_deterministic() -> None:
    merged = merge_mappings(
        {"first": 1, "second": {"a": 1, "b": 2}},
        {"second": {"b": 3, "c": 4}, "third": 5},
    )

    assert list(merged) == ["first", "second", "third"]
    assert list(merged["second"]) == ["a", "b", "c"]


@pytest.mark.parametrize("base,override", [([], {}), ({}, []), (None, {}), ({}, None)])
def test_rejects_non_mapping_roots(base, override) -> None:
    with pytest.raises(TypeError):
        merge_mappings(base, override)


def test_rejects_recursive_mapping_pair_deterministically() -> None:
    base: dict[str, object] = {}
    override: dict[str, object] = {}
    base["self"] = base
    override["self"] = override

    with pytest.raises(RecursiveMappingError):
        merge_mappings(base, override)


def test_large_mapping_merge() -> None:
    base = {"values": {f"k{i}": i for i in range(5000)}}
    override = {"values": {f"k{i}": -i for i in range(2500, 7500)}}

    merged = merge_mappings(base, override)

    assert len(merged["values"]) == 7500
    assert merged["values"]["k2499"] == 2499
    assert merged["values"]["k2500"] == -2500
    assert merged["values"]["k7499"] == -7499


def test_concurrent_calls_have_no_shared_state() -> None:
    def run(index: int) -> dict[str, object]:
        return merge_mappings(
            {"safety": {"risk_threshold": 0.35, "worker": None}},
            {"safety": {"worker": index}},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(64)))

    assert [result["safety"]["worker"] for result in results] == list(range(64))
    assert all(result["safety"]["risk_threshold"] == 0.35 for result in results)


def test_json_round_trip_for_serializable_configuration() -> None:
    merged = merge_mappings(
        {"telemetry": {"enabled": True, "labels": ["core"]}},
        {"telemetry": {"labels": ["core", "audit"]}},
    )

    assert json.loads(json.dumps(merged)) == merged
