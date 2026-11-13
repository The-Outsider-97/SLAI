"""Adversarial-input, state-isolation, concurrency and performance-guard tests for ReasoningTypes.

Run from the repository root:  python -m unittest tests.test_reasoning_types_robustness
The timing guards are deliberately generous (order-of-magnitude regressions only); they are not benchmarks.
"""
from __future__ import annotations

import copy
import json
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from src.agents.reasoning.reasoning_types import ReasoningTypes
from src.agents.reasoning.strategy_selector import STRATEGIES

SOCRATES = {"premises": ["all humans are mortal", "socrates is human"], "hypothesis": "socrates is mortal"}
EVENTS = [
    {"id": "rain", "description": "Heavy rainfall", "timestamp": 1, "attributes": {"intensity": "high"}},
    {"id": "river", "description": "River level rises", "timestamp": 2, "attributes": {"intensity": "high"}},
    {"id": "flood", "description": "Flood warning issued", "timestamp": 3, "attributes": {"severity": "high"}},
]
SYSTEM = {"name": "svc", "components": {"a": {"functions": ["alpha"], "dependencies": ["b"]}, "b": {"functions": ["beta"]}}}
NUMERIC = [{"content": {"v": 10 + 2 * i}} for i in range(8)]
ANALOGY = {
    "target": {"name": "Urban Traffic", "functions": ["route vehicles"], "goals": ["efficiency"]},
    "sources": [{"name": "Computer Network", "functions": ["route packets"], "goals": ["efficiency"]}],
}

# strategy -> (problem, context) that is valid for that strategy
VALID_REQUESTS = {
    "deduction": ("x", SOCRATES),
    "induction": (NUMERIC, {}),
    "abduction": ("Server latency spike", {}),
    "decompositional": (SYSTEM, {}),
    "cause_effect": (EVENTS, {"conditions": {"intensity": "high"}, "max_time_gap": 5}),
    "analogical": (ANALOGY["target"], {"source_domain": ANALOGY["sources"]}),
}


_TIMING_KEYS = {"duration_seconds", "elapsed_seconds", "cache_hit", "timestamp", "timestamp_ms", "created_at", "latency_ms", "elapsed_ms"}


def _strip_timing(value):
    """Drop wall-clock fields so two runs of the same request can be compared structurally."""
    if isinstance(value, dict):
        return {k: _strip_timing(v) for k, v in value.items() if k not in _TIMING_KEYS}
    if isinstance(value, list):
        return [_strip_timing(v) for v in value]
    return value


class MalformedInputTests(unittest.TestCase):
    """No strategy may leak an untyped crash (RecursionError, TypeError, ...) for hostile input."""

    HOSTILE = [None, "", "   ", 0, 3.14, float("nan"), [], {}, [None], [[]], {"": None}, "\x00\x01", "é" * 5000, object()]

    def test_every_strategy_survives_hostile_problems(self) -> None:
        types = ReasoningTypes()
        for strategy in STRATEGIES:
            for problem in self.HOSTILE:
                for context in ({}, {"junk": object()}, {"premises": None}, {"observations": 5}):
                    with self.subTest(strategy=strategy, problem=repr(problem)[:30], context=list(context)):
                        try:
                            types.reason(strategy, problem, context)
                        except RecursionError:
                            self.fail("RecursionError leaked")
                        except Exception as exc:  # typed errors are acceptable; bare builtins from deep inside are not
                            self.assertNotIsInstance(exc, (TypeError, AttributeError, KeyError, IndexError, ZeroDivisionError),
                                                     f"untyped {type(exc).__name__}: {exc}")

    def test_cyclic_and_deep_structures_do_not_crash_any_strategy(self) -> None:
        loop: dict = {"name": "loop"}
        loop["self"] = loop
        deep: list = []
        node = deep
        for _ in range(3000):
            nxt: list = []
            node.append(nxt)
            node = nxt
        types = ReasoningTypes()
        for strategy in STRATEGIES:
            for problem in (loop, deep):
                with self.subTest(strategy=strategy):
                    try:
                        types.reason(strategy, problem, {})
                    except RecursionError:
                        self.fail("RecursionError leaked")
                    except Exception as exc:
                        self.assertNotIsInstance(exc, (TypeError, AttributeError, KeyError, IndexError))

    def test_excessive_nesting_is_rejected_with_a_typed_error(self) -> None:
        from src.agents.reasoning.utils.reasoning_errors import ReasoningValidationError
        deep: list = []
        node = deep
        for _ in range(500):
            nxt: list = []
            node.append(nxt)
            node = nxt
        types = ReasoningTypes()
        for strategy in STRATEGIES:
            with self.subTest(strategy=strategy), self.assertRaises(ReasoningValidationError):
                types.reason(strategy, deep, {})
            with self.subTest(strategy=strategy, where="context"), self.assertRaises(ReasoningValidationError):
                types.reason(strategy, "x", {"payload": deep})

    def test_unknown_and_empty_strategy_names_are_typed_errors(self) -> None:
        from src.agents.reasoning.utils.reasoning_errors import ReasoningError
        types = ReasoningTypes()
        for name in ("", "   ", None, "nonexistent", "deduction+nonexistent", "+", "++"):
            with self.subTest(name=name), self.assertRaises(ReasoningError):
                types.reason(name, "x", {})  # type: ignore[arg-type]

    def test_inputs_are_not_mutated(self) -> None:
        types = ReasoningTypes()
        for strategy, (problem, context) in VALID_REQUESTS.items():
            problem_before, context_before = copy.deepcopy(problem), copy.deepcopy(context)
            types.reason(strategy, problem, context)
            self.assertEqual(problem, problem_before, strategy)
            self.assertEqual(context, context_before, strategy)

    def test_results_are_json_serialisable(self) -> None:
        types = ReasoningTypes()
        for strategy, (problem, context) in VALID_REQUESTS.items():
            json.dumps(types.reason(strategy, problem, context), default=str)


class HelperRobustnessTests(unittest.TestCase):
    def test_json_safe_state_handles_self_reference_and_keeps_plain_behaviour(self) -> None:
        from src.agents.reasoning.utils.reasoning_helpers import json_safe_reasoning_state
        loop: dict = {"a": 1}
        loop["self"] = loop
        loop["items"] = [loop, {"x": {1, 2}}, {3}]
        safe = json_safe_reasoning_state(loop)
        json.dumps(safe)
        self.assertEqual(safe["a"], 1)
        plain = {"s": {2, 1}, "n": {"t": (1, {"k": 1})}, "l": [{"z": 1}, {4}]}
        self.assertEqual(json_safe_reasoning_state(plain), {"s": [1, 2], "n": {"t": [1, {"k": 1}]}, "l": [{"z": 1}, "{4}"]})


class StateIsolationTests(unittest.TestCase):
    def test_repeated_calls_are_deterministic_and_independent(self) -> None:
        types = ReasoningTypes()
        for strategy, (problem, context) in VALID_REQUESTS.items():
            first = types.reason(strategy, problem, context)
            second = types.reason(strategy, problem, context)
            self.assertEqual(_strip_timing(first), _strip_timing(second), strategy)

    def test_interleaved_requests_do_not_leak_between_problems(self) -> None:
        types = ReasoningTypes()
        negative = {"premises": ["all humans are mortal", "socrates is human"], "hypothesis": "plato is wise"}
        for _ in range(3):
            self.assertEqual(types.reason("deduction", "x", SOCRATES)["outcome"], "supported")
            self.assertNotEqual(types.reason("deduction", "x", negative)["outcome"], "supported")

    def test_each_execute_uses_a_fresh_instance(self) -> None:
        types = ReasoningTypes()
        self.assertIsNot(types.create("deduction"), types.create("deduction"))

    def test_abduction_evidence_does_not_accumulate_across_requests(self) -> None:
        types = ReasoningTypes()
        context = {"evidence_sources": [{"content": "memory leak confirmed", "confidence": 0.9, "source": "profiler"}]}
        first = types.reason("abduction", "latency spike", context)
        second = types.reason("abduction", "latency spike", {})
        self.assertEqual(json.dumps(second, default=str).count("memory leak confirmed"), 0, first)


class ConcurrencyTests(unittest.TestCase):
    def test_shared_instance_under_threads_gives_correct_results(self) -> None:
        types = ReasoningTypes()
        negative = {"premises": ["all humans are mortal", "socrates is human"], "hypothesis": "plato is wise"}
        errors: list = []

        def work(i: int):
            try:
                if i % 2:
                    return types.reason("deduction", "x", SOCRATES)["outcome"] == "supported"
                return types.reason("deduction", "x", negative)["outcome"] != "supported"
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                return False

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(work, range(160)))
        self.assertEqual(errors, [])
        self.assertTrue(all(results))

    def test_all_strategies_concurrently(self) -> None:
        types = ReasoningTypes()
        names = list(VALID_REQUESTS)
        expected = {n: types.reason(n, *VALID_REQUESTS[n])["outcome"] for n in names}

        def work(i: int):
            name = names[i % len(names)]
            return name, types.reason(name, *VALID_REQUESTS[name])["outcome"]

        with ThreadPoolExecutor(max_workers=12) as pool:
            for name, outcome in pool.map(work, range(120)):
                self.assertEqual(outcome, expected[name], name)

    def test_concurrent_registration_is_isolated_per_instance(self) -> None:
        from src.agents.reasoning.types import BaseReasoning

        class Echo(BaseReasoning):
            def perform_reasoning(self, input_data=None, context=None, **kw):
                return {"metrics": {"success": True}}

        instances = [ReasoningTypes() for _ in range(8)]
        barrier = threading.Barrier(len(instances))

        def work(idx: int):
            barrier.wait()
            instances[idx].register(f"echo{idx}", Echo)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(len(instances))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for idx, inst in enumerate(instances):
            for other in range(len(instances)):
                self.assertEqual(f"echo{other}" in inst._TASK_TYPES, other == idx)

    def test_statistics_are_consistent_under_concurrency(self) -> None:
        types = ReasoningTypes()
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda _: types.execute("deduction", "x", SOCRATES), range(200)))
        self.assertEqual(types.get_stats()["total_executions"], 200)


class PerformanceGuardTests(unittest.TestCase):
    """Order-of-magnitude regression guards. Real measurements live in the benchmark report."""

    def timed(self, fn) -> float:
        start = time.monotonic()
        fn()
        return time.monotonic() - start

    def test_deduction_scales_with_premise_count(self) -> None:
        premises = [f"entity{i} is a thing{i}" for i in range(250)]  # config max_premises is 256
        types = ReasoningTypes()
        elapsed = self.timed(lambda: types.reason("deduction", "x", {"premises": premises, "hypothesis": "entity5 is a thing5"}))
        self.assertLess(elapsed, 10.0)

    def test_cause_effect_handles_a_few_hundred_events(self) -> None:
        events = [{"id": f"e{i}", "description": f"event number {i}", "timestamp": float(i)} for i in range(200)]
        types = ReasoningTypes()
        self.assertLess(self.timed(lambda: types.reason("cause_effect", events, {"max_time_gap": 3})), 60.0)

    def test_decompositional_handles_wide_systems(self) -> None:
        system = {"name": "wide", "components": {f"c{i}": {"functions": [f"f{i}"], "dependencies": [f"c{(i + 1) % 150}"]} for i in range(150)}}
        types = ReasoningTypes()
        self.assertLess(self.timed(lambda: types.reason("decompositional", system, {})), 30.0)

    def test_cache_hit_is_faster_than_cold_run_for_analogical(self) -> None:
        from src.agents.reasoning.types import ReasoningAnalogical
        engine = ReasoningAnalogical()
        target = {"name": "T", "functions": [f"route item {i}" for i in range(40)], "goals": ["efficiency"]}
        source = {"name": "S", "functions": [f"route packet {i}" for i in range(40)], "goals": ["efficiency"]}
        cold = self.timed(lambda: engine.perform_reasoning(target, [source], {}))
        warm = self.timed(lambda: engine.perform_reasoning(target, [source], {}))
        self.assertLess(warm, cold + 0.05)


if __name__ == "__main__":
    unittest.main()
