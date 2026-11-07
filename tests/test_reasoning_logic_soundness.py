"""Logical-correctness tests for the six reasoning strategies, built around counterexamples.

Every test targets a way the strategy used to return a *wrong* answer (an unsound deduction,
a spurious "pattern", a vacuous validation, a template reported as an explanation, ...).
Run from the repository root:  python -m unittest tests.test_reasoning_logic_soundness
"""
from __future__ import annotations

import random
import unittest

from src.agents.reasoning.types import (
    ReasoningAbduction, ReasoningAnalogical, ReasoningCauseAndEffect, ReasoningDecompositional,
    ReasoningDeductive, ReasoningInductive,
)
from src.agents.reasoning.types.reasoning_inductive import correlation_p_value
from src.agents.reasoning.utils.reasoning_errors import ReasoningValidationError


def prove(premises, hypothesis, context=None, **attrs):
    engine = ReasoningDeductive()
    for name, value in attrs.items():
        setattr(engine, name, value)
    return engine.perform_reasoning(list(premises), hypothesis, context or {})


class DeductionSoundnessTests(unittest.TestCase):
    def test_nary_disjunctive_syllogism_does_not_derive_single_options(self) -> None:
        """A or B or C, not A  does NOT entail B.  (It entails  B or C.)"""
        premises = ["x is red or x is green or x is blue", "not x is red"]
        self.assertFalse(prove(premises, "x is green")["proven"])
        self.assertFalse(prove(premises, "x is blue")["proven"])
        self.assertEqual(prove(premises, "x is green")["verdict"], "undetermined")
        self.assertTrue(prove(premises, "x is green or x is blue")["proven"])

    def test_disjunctive_syllogism_resolves_when_one_option_remains(self) -> None:
        premises = ["x is red or x is green or x is blue", "not x is red", "not x is green"]
        self.assertTrue(prove(premises, "x is blue")["proven"])

    def test_two_way_disjunctive_syllogism_still_works(self) -> None:
        premises = ["system is stable or system is degraded", "not system is stable"]
        self.assertTrue(prove(premises, "system is degraded")["proven"])

    def test_all_options_refuted_derives_nothing(self) -> None:
        """No ex falso: refuting every disjunct must not 'prove' an arbitrary disjunct."""
        premises = ["x is red or x is blue", "not x is red", "not x is blue"]
        self.assertFalse(prove(premises, "x is red")["proven"])
        self.assertFalse(prove(premises, "x is green")["proven"])

    def test_elliptical_coordination_is_not_a_clausal_disjunction(self) -> None:
        result = prove(["alice likes tea or coffee", "not alice likes tea"], "coffee")
        self.assertFalse(result["proven"])

    def test_possession_is_not_class_membership(self) -> None:
        """'Socrates has a dog' + 'all dogs are loyal' used to prove 'Socrates is loyal'."""
        self.assertFalse(prove(["socrates has a dog", "all dogs are loyal"], "socrates is loyal")["proven"])
        self.assertTrue(prove(["rex is a dog", "all dogs are loyal"], "rex is loyal")["proven"])

    def test_valid_long_chain_is_not_discounted_by_length(self) -> None:
        """Each rule application multiplied certainty by 0.95, so a valid 6-step proof scored 0.735 < 0.85."""
        premises = ["a0"] + [f"if a{i} then a{i + 1}" for i in range(6)]
        result = prove(premises, "a6")
        self.assertTrue(result["proven"])
        self.assertEqual(result["certainty"], 1.0)
        self.assertEqual(result["verdict"], "entailed")

    def test_decay_remains_available_as_explicit_opt_in(self) -> None:
        premises = ["a0"] + [f"if a{i} then a{i + 1}" for i in range(6)]
        result = prove(premises, "a6", apply_rule_decay=True)
        self.assertLess(result["certainty"], 1.0)

    def test_certainty_is_limited_by_weakest_premise(self) -> None:
        premises = ["all humans are mortal", "socrates is human"]
        context = {"evidence": [{"content": "socrates is human", "confidence": 0.6}]}
        result = prove(premises, "socrates is mortal", context)
        self.assertFalse(result["proven"])
        self.assertAlmostEqual(result["certainty"], 0.6, places=6)
        self.assertEqual(result["verdict"], "entailed_low_confidence")

    def test_cache_does_not_ignore_evidence(self) -> None:
        """The cache key omitted evidence, so the second call returned the first call's answer."""
        engine = ReasoningDeductive()
        premises = ["all humans are mortal", "socrates is human"]
        high = engine.perform_reasoning(premises, "socrates is mortal", {"evidence": [{"content": "socrates is human", "confidence": 0.95}]})
        low = engine.perform_reasoning(premises, "socrates is mortal", {"evidence": [{"content": "socrates is human", "confidence": 0.1}]})
        self.assertTrue(high["proven"])
        self.assertFalse(low["proven"])
        self.assertIsNot(high, low)

    def test_cached_result_is_a_private_copy(self) -> None:
        engine = ReasoningDeductive()
        premises = ["all humans are mortal", "socrates is human"]
        first = engine.perform_reasoning(premises, "socrates is mortal", {})
        first["proven"] = "tampered"
        second = engine.perform_reasoning(premises, "socrates is mortal", {})
        self.assertIs(second["proven"], True)

    def test_hypothesis_that_is_a_premise_is_entailed(self) -> None:
        result = prove(["it is raining"], "it is raining")
        self.assertTrue(result["proven"])

    def test_premise_contradicting_hypothesis_refutes_it_even_if_unused(self) -> None:
        result = prove(["it is raining", "the sky is blue"], "it is not raining")
        self.assertFalse(result["proven"])
        self.assertEqual(result["verdict"], "refuted")

    def test_underivable_hypothesis_has_zero_certainty(self) -> None:
        """An unsupported hypothesis used to be reported with certainty 0.25."""
        result = prove(["socrates is human"], "plato is wise")
        self.assertEqual(result["certainty"], 0.0)
        self.assertEqual(result["verdict"], "undetermined")

    def test_inconsistent_premises_are_reported_not_exploited(self) -> None:
        result = prove(["it rains", "not it rains", "if it rains then ground is wet"], "ground is wet")
        self.assertFalse(result["proven"])
        self.assertEqual(result["verdict"], "inconsistent")
        self.assertEqual(result["stop_reason"], "inconsistent_premises")

    def test_non_strict_mode_still_refuses_to_prove_from_contradiction(self) -> None:
        result = prove(["it rains", "not it rains"], "it rains", strict_inputs=False)
        self.assertFalse(result["proven"])
        self.assertEqual(result["verdict"], "inconsistent")

    def test_budget_exhaustion_is_reported(self) -> None:
        premises = ["a0"] + [f"if a{i} then a{i + 1}" for i in range(40)]
        result = prove(premises, "a40", max_steps=5)
        self.assertFalse(result["proven"])
        self.assertTrue(result["truncated"])
        self.assertEqual(result["stop_reason"], "step_budget_exhausted")

    def test_malformed_optional_context_never_raises_untyped_errors(self) -> None:
        premises = ["all humans are mortal", "socrates is human"]
        for context in ({"known_falsehoods": None}, {"known_falsehoods": 5}, {"evidence": "oops"}, {"evidence": 7}):
            prove(premises, "socrates is mortal", context)

    def test_known_falsehood_premise_is_rejected(self) -> None:
        result = prove(["all humans are mortal", "socrates is human"], "socrates is mortal",
                       {"known_falsehoods": ["socrates is human"]})
        self.assertFalse(result["proven"])

    def test_contradiction_index_matches_exhaustive_pairwise_definition(self) -> None:
        """The linear-time detector must find exactly the pairs the O(n^2) definition finds."""
        engine = ReasoningDeductive()
        rng = random.Random(17)
        subjects, predicates = ["x", "y", "z"], ["red", "blue", "big", "small"]
        for _ in range(60):
            statements = []
            for _ in range(rng.randint(2, 12)):
                s, p = rng.choice(subjects), rng.choice(predicates)
                form = rng.choice(["{s} is {p}", "{s} is not {p}", "not {s} is {p}", "not {s} {p}", "{s} {p}"])
                statements.append(form.format(s=s, p=p))
            canonical = [engine._canonical_text(t) for t in statements]
            expected = set()
            for i, left in enumerate(canonical):
                for right in canonical[i + 1:]:
                    kind = engine._contradiction_type(left, right)
                    if kind:
                        expected.add((frozenset((left, right)), kind))
            found = {(frozenset((a, b)), kind) for a, b, kind in engine.identify_contradictions(statements)}
            self.assertEqual(found, expected, statements)

    def test_large_premise_set_is_fast(self) -> None:
        import time
        premises = [f"entity{i} is a thing{i}" for i in range(250)]
        started = time.monotonic()
        result = prove(premises, "entity3 is a thing3")
        self.assertTrue(result["proven"])
        self.assertLess(time.monotonic() - started, 5.0)  # was ~1 s on a quiet machine; 5 s is a CI-noise guard


class InductionSoundnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ReasoningInductive()

    def pattern_types(self, result):
        return [p["type"] for p in result["supporting_data"]["patterns_identified"]]

    def test_index_derived_order_is_not_a_temporal_pattern(self) -> None:
        """Timestamps assigned from list position are always increasing and regular: 'temporal_order' for ANY list."""
        for observations in (["a", "b", "c", "d", "e", "f"], [{"content": {"v": i % 2}} for i in range(8)]):
            result = ReasoningInductive().perform_reasoning(observations, {})
            kinds = self.pattern_types(result)
            self.assertNotIn("temporal_order", kinds)
            self.assertNotIn("periodic", kinds)

    def test_explicit_regular_timestamps_are_a_temporal_pattern(self) -> None:
        observations = [{"content": {"v": 1}, "timestamp": 10 * i} for i in range(8)]
        kinds = self.pattern_types(ReasoningInductive().perform_reasoning(observations, {}))
        self.assertIn("temporal_order", kinds)
        self.assertIn("periodic", kinds)

    def test_random_noise_is_rarely_declared_valid(self) -> None:
        """~25% of 5-point noise series pass an R^2>=0.4 gate; the slope significance test caps it near alpha."""
        rng = random.Random(1234)
        valid = 0
        trials = 200
        for _ in range(trials):
            observations = [{"content": {"v": rng.random()}} for _ in range(8)]
            if ReasoningInductive().perform_reasoning(observations, {})["validation"]["is_valid"]:
                valid += 1
        self.assertLessEqual(valid / trials, 0.10)

    def test_true_linear_trend_is_still_found(self) -> None:
        observations = [{"content": {"temp": 20 + 2 * i, "kind": "sunny"}, "source": f"s{i % 3}"} for i in range(8)]
        result = ReasoningInductive().perform_reasoning(observations, {})
        self.assertIn("numeric_trend", self.pattern_types(result))
        self.assertTrue(result["predictions"])

    def test_dict_keys_are_not_content_tokens(self) -> None:
        observations = [{"content": {"v": i * 0.37}} for i in range(8)]
        result = ReasoningInductive().perform_reasoning(observations, {})
        self.assertNotIn("recurring_content_token", self.pattern_types(result))

    def test_stopwords_are_not_recurring_content(self) -> None:
        texts = ["the cat sat on the mat", "the dog ran to the park", "the bird flew over the tree",
                 "the fish swam in the pond", "the cow ate the grass"]
        result = ReasoningInductive().perform_reasoning(texts, {})
        tokens = [p.get("value") for p in result["supporting_data"]["patterns_identified"]]
        self.assertNotIn("the", tokens)

    def test_token_frequency_is_document_frequency(self) -> None:
        """One verbose observation repeating a word must not make it 'recurring across observations'."""
        observations = ["spam spam spam spam spam spam spam spam", "alpha", "beta", "gamma", "delta", "epsilon"]
        result = ReasoningInductive().perform_reasoning(observations, {})
        self.assertNotIn("spam", [p.get("value") for p in result["supporting_data"]["patterns_identified"]])

    def test_holdout_that_contradicts_the_trend_fails_validation(self) -> None:
        """The holdout test scored predictions generated from the holdout itself, so it could not fail."""
        train = [{"content": {"v": 10 + 2 * i}} for i in range(8)]
        bad_holdout = [{"content": {"v": 5}}, {"content": {"v": 4}}, {"content": {"v": 3}}]
        result = ReasoningInductive().perform_reasoning(train, {"holdout_data": bad_holdout})
        tests = {t["test"]: t for t in result["validation"]["tests"]}
        self.assertFalse(tests["predictive_accuracy"]["result"])
        self.assertFalse(result["validation"]["is_valid"])

    def test_holdout_that_continues_the_trend_passes(self) -> None:
        train = [{"content": {"v": 10 + 2 * i}} for i in range(8)]
        good_holdout = [{"content": {"v": 10 + 2 * i}} for i in range(8, 11)]
        result = ReasoningInductive().perform_reasoning(train, {"holdout_data": good_holdout})
        tests = {t["test"]: t for t in result["validation"]["tests"]}
        self.assertTrue(tests["predictive_accuracy"]["result"])
        self.assertEqual(result["validation"]["evidence_level"], "holdout")

    def test_untestable_holdout_fails_closed(self) -> None:
        train = [{"content": {"v": 10 + 2 * i}} for i in range(8)]
        unrelated = [{"content": {"other": 1}}, {"content": {"other": 2}}]
        result = ReasoningInductive().perform_reasoning(train, {"holdout_data": unrelated})
        tests = {t["test"]: t for t in result["validation"]["tests"]}
        self.assertFalse(tests["predictive_accuracy"]["result"])
        self.assertFalse(tests["predictive_accuracy"]["testable"])

    def test_in_sample_validation_is_labelled(self) -> None:
        train = [{"content": {"v": 10 + 2 * i}} for i in range(8)]
        result = ReasoningInductive().perform_reasoning(train, {})
        self.assertEqual(result["validation"]["evidence_level"], "in_sample")

    def test_extrapolations_are_not_marked_validated(self) -> None:
        train = [{"content": {"v": 10 + 2 * i}} for i in range(8)]
        for prediction in ReasoningInductive().perform_reasoning(train, {})["predictions"]:
            self.assertFalse(prediction["validated"])

    def test_cached_result_is_a_private_copy(self) -> None:
        engine = ReasoningInductive()
        train = [{"content": {"v": 10 + 2 * i}} for i in range(8)]
        first = engine.perform_reasoning(train, {})
        first["theory"]["theory"] = "tampered"
        second = engine.perform_reasoning(train, {})
        self.assertNotEqual(second["theory"]["theory"], "tampered")

    def test_non_mapping_context_is_a_typed_error(self) -> None:
        with self.assertRaises(ReasoningValidationError):
            ReasoningInductive().perform_reasoning([1, 2, 3, 4, 5, 6], ["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_correlation_p_value_matches_reference_values(self) -> None:
        # Two-sided critical values from standard t tables.
        self.assertAlmostEqual(correlation_p_value(0.8783, 5), 0.05, delta=0.002)
        self.assertAlmostEqual(correlation_p_value(0.5, 10), 0.1411, delta=0.002)
        self.assertAlmostEqual(correlation_p_value(0.9, 3), 0.2871, delta=0.002)
        self.assertAlmostEqual(correlation_p_value(0.0, 20), 1.0, delta=1e-9)
        self.assertEqual(correlation_p_value(1.0, 20), 0.0)
        self.assertEqual(correlation_p_value(0.99, 2), 1.0)  # two points always fit a line


class AbductionSoundnessTests(unittest.TestCase):
    def test_template_hypotheses_are_flagged_and_never_grounded(self) -> None:
        result = ReasoningAbduction().perform_reasoning("The grass is wet", {})
        self.assertFalse(result["metrics"]["grounded"])
        for hypothesis in [result["best_explanation"], *result["alternative_hypotheses"]]:
            if hypothesis:
                self.assertNotEqual(hypothesis["origin"], "seed")

    def test_seed_beats_template_echo(self) -> None:
        """Template echo covered 100% of the observation, so it outranked every real candidate."""
        context = {"seed_hypotheses": [{"hypothesis": "Memory leak in the service", "prior": 0.7, "explains": [0]}]}
        result = ReasoningAbduction().perform_reasoning("Server latency spike", context)
        self.assertEqual(result["best_explanation"]["hypothesis"], "Memory leak in the service")
        self.assertTrue(result["metrics"]["grounded"])

    def test_unrelated_seed_without_declared_link_is_not_selected(self) -> None:
        result = ReasoningAbduction().perform_reasoning(
            "Server latency spike", {"seed_hypotheses": ["Quantum tunnelling in the lobby"]}
        )
        best = result["best_explanation"]
        self.assertTrue(best is None or best["origin"] != "seed")

    def test_contradicted_hypothesis_is_never_selected(self) -> None:
        context = {
            "seed_hypotheses": [{"hypothesis": "power failure", "explains": [0]}],
            "evidence_sources": [{"content": "power failure ruled out", "type": "contradiction",
                                  "confidence": 0.95, "source": "audit"}],
        }
        result = ReasoningAbduction().perform_reasoning("lights out", context)
        best = result["best_explanation"]
        self.assertTrue(best is None or best["hypothesis"] != "power failure")
        self.assertTrue(result["contradictions"])
        self.assertEqual(result["contradictions"][0]["hypothesis"], "power failure")

    def test_stopwords_do_not_create_explanations_or_contradictions(self) -> None:
        engine = ReasoningAbduction()
        evidence = [{"content": "it is on a table", "confidence": 1.0, "source": "x", "type": "contradiction"}]
        self.assertEqual(engine._find_contradictions("what is the cause of a fault", evidence, {}), [])
        self.assertEqual(engine._explanatory_power("is a the of", evidence), 0.0)

    def test_substring_overlap_is_not_a_match(self) -> None:
        engine = ReasoningAbduction()
        evidence = [{"content": "wetland drainage", "confidence": 1.0, "source": "x"}]
        self.assertEqual(engine._explanatory_power("wet grass", evidence), 0.0)

    def test_index_pattern_is_not_a_hypothesis_source(self) -> None:
        engine = ReasoningAbduction()
        engine.max_hypotheses = 20
        texts = [h["hypothesis"] for h in _all(engine.perform_reasoning(["alpha", "beta", "gamma", "delta"], {}))]
        self.assertFalse(any("Escalating" in t for t in texts))

    def test_numeric_observations_can_yield_a_pattern_hypothesis(self) -> None:
        engine = ReasoningAbduction()
        engine.max_hypotheses = 30
        texts = [h["hypothesis"] for h in _all(engine.perform_reasoning([1, 2, 3, 5, 8], {}))]
        self.assertTrue(any("Escalating" in t for t in texts))

    def test_instance_state_does_not_accumulate_between_runs(self) -> None:
        engine = ReasoningAbduction()
        engine.perform_reasoning("x y", {})
        engine.perform_reasoning("z w", {})
        self.assertEqual(len(engine.evidence), 1)

    def test_malformed_optional_context_is_tolerated(self) -> None:
        for context in (None, "text", {"evidence_sources": None}, {"evidence_sources": {"a": 1}},
                        {"seed_hypotheses": None}, {"seed_hypotheses": 5}, {"known_facts": 3}, {"constraints": None}):
            ReasoningAbduction().perform_reasoning("x y", context)  # type: ignore[arg-type]

    def test_rejected_evidence_is_counted_not_silently_dropped(self) -> None:
        context = {"evidence_sources": [{"content": "weak hint", "confidence": 0.2, "source": "rumour"}]}
        result = ReasoningAbduction().perform_reasoning("x y", context)
        self.assertEqual(result["metrics"]["evidence_rejected"], 1)

    def test_empty_observations_are_a_typed_error(self) -> None:
        with self.assertRaises(ReasoningValidationError):
            ReasoningAbduction().perform_reasoning([], {})


def _all(result):
    return ([result["best_explanation"]] if result["best_explanation"] else []) + result["alternative_hypotheses"]


class AnalogicalSoundnessTests(unittest.TestCase):
    SOURCE = {"name": "Computer Network", "functions": ["route packets"], "goals": ["efficiency"], "latency": 5}
    TARGET = {"name": "Urban Traffic Network", "functions": ["route vehicles"], "goals": ["efficiency"], "latency": None}

    def test_select_best_transfer_prefers_a_valid_transfer(self) -> None:
        engine = ReasoningAnalogical()
        invalid_high = {"transfer_score": 0.99, "mapping_score": 0.9, "validation": {"valid": False}}
        valid_low = {"transfer_score": 0.60, "mapping_score": 0.6, "validation": {"valid": True}}
        self.assertIs(engine.select_best_transfer([invalid_high, valid_low]), valid_low)
        self.assertIs(engine.select_best_transfer([invalid_high]), invalid_high)
        self.assertIsNone(engine.select_best_transfer([]))

    def test_cache_hit_does_not_mutate_the_stored_result(self) -> None:
        engine = ReasoningAnalogical()
        first = engine.perform_reasoning(self.TARGET, [self.SOURCE], {})
        second = engine.perform_reasoning(self.TARGET, [self.SOURCE], {})
        third = engine.perform_reasoning(self.TARGET, [self.SOURCE], {})
        self.assertFalse(first["metrics"]["cache_hit"])
        self.assertTrue(second["metrics"]["cache_hit"])
        self.assertTrue(third["metrics"]["cache_hit"])
        self.assertIsNot(second, third)
        second["best_transfer"] = "tampered"
        self.assertNotEqual(engine.perform_reasoning(self.TARGET, [self.SOURCE], {})["best_transfer"], "tampered")

    def test_large_inputs_sharing_a_prefix_do_not_share_a_cache_entry(self) -> None:
        """The cache key hashed repr(...)[:1500], so inputs identical for 1500 chars collided."""
        engine = ReasoningAnalogical()
        long_text = "x" * 2000
        target_a = {"name": "T", "notes": long_text + "A", "functions": ["route vehicles"]}
        target_b = {"name": "T", "notes": long_text + "B", "functions": ["route vehicles"]}
        engine.perform_reasoning(target_a, [self.SOURCE], {})
        second = engine.perform_reasoning(target_b, [self.SOURCE], {})
        self.assertFalse(second["metrics"]["cache_hit"])

    def test_unrelated_source_yields_no_transfer(self) -> None:
        volcano = {"name": "Volcano", "functions": ["erupt lava"], "goals": ["pressure release"]}
        hospital = {"name": "Hospital", "functions": ["treat patients"], "goals": ["health"]}
        result = ReasoningAnalogical().perform_reasoning(hospital, [volcano], {})
        self.assertIsNone(result["best_transfer"])
        self.assertFalse(result["metrics"]["success"])

    def test_non_mapping_context_is_a_typed_error(self) -> None:
        with self.assertRaises(ReasoningValidationError):
            ReasoningAnalogical().perform_reasoning(self.TARGET, [self.SOURCE], "oops")  # type: ignore[arg-type]

    def test_property_similarity_is_memoized_and_stable(self) -> None:
        engine = ReasoningAnalogical()
        first = engine._property_similarity("route_packets", "route packets")
        self.assertEqual(first, engine._property_similarity("route_packets", "route packets"))
        self.assertGreaterEqual(engine._property_similarity_bound("route_packets", "route packets"), first)

    def test_similarity_upper_bound_never_underestimates(self) -> None:
        engine = ReasoningAnalogical()
        rng = random.Random(5)
        words = ["route", "packet", "load", "balance", "heat", "pump", "valve", "signal", "queue", "buffer"]
        for _ in range(300):
            a = "_".join(rng.sample(words, rng.randint(1, 3)))
            b = "_".join(rng.sample(words, rng.randint(1, 3)))
            self.assertGreaterEqual(engine._property_similarity_bound(a, b) + 1e-12, engine._property_similarity(a, b), (a, b))


class CauseEffectSoundnessTests(unittest.TestCase):
    EVENTS = [
        {"id": "rain", "description": "Heavy rainfall", "timestamp": 1, "attributes": {"intensity": "high", "causes": ["flood"]}},
        {"id": "river", "description": "River water level rises", "timestamp": 2, "attributes": {"intensity": "high", "effects": ["flood"]}},
        {"id": "warning", "description": "Flood warning issued", "timestamp": 3, "attributes": {"severity": "high"}},
        {"id": "evac", "description": "Residents evacuate", "timestamp": 4, "attributes": {"severity": "high"}},
    ]

    def test_acyclicity_check_survives_very_long_chains(self) -> None:
        """The recursive DFS raised RecursionError at ~1000 edges."""
        engine = ReasoningCauseAndEffect()
        chain = [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(6000)]
        self.assertTrue(engine._is_acyclic(chain))
        self.assertFalse(engine._is_acyclic(chain + [{"source": "n6000", "target": "n0"}]))
        self.assertFalse(engine._is_acyclic([{"source": "a", "target": "a"}]))

    def test_projection_confidence_reflects_edge_strength(self) -> None:
        """Every projection was a constant 0.75 regardless of how weak the supporting edges were."""
        engine = ReasoningCauseAndEffect()

        def project(confidence):
            edges = [{"source": "a", "target": "b", "confidence": confidence},
                     {"source": "b", "target": "c", "confidence": confidence}]
            network = {"nodes": {"a": {}, "b": {}, "c": {}}, "edges": edges}
            return {(p["cause"], p["effect"]): p for p in engine._predict_with_bayesian_network(network, {})}

        strong, weak = project(0.9), project(0.3)
        self.assertGreater(strong[("a", "c")]["confidence"], weak[("a", "c")]["confidence"])
        self.assertAlmostEqual(strong[("a", "c")]["confidence"], 0.81, places=6)
        self.assertEqual(strong[("a", "c")]["path"], ["a", "b", "c"])
        self.assertLess(strong[("a", "c")]["confidence"], strong[("a", "b")]["confidence"])

    def test_event_count_is_bounded(self) -> None:
        engine = ReasoningCauseAndEffect()
        engine.max_events = 10
        with self.assertRaises(ReasoningValidationError):
            engine.perform_reasoning([f"event {i}" for i in range(11)], {}, {})

    def test_counterfactual_provenance_is_reported(self) -> None:
        result = ReasoningCauseAndEffect().perform_reasoning(
            self.EVENTS, conditions={"intensity": "high"}, context={"max_time_gap": 5})
        bases = {rel["counterfactual_basis"] for rel in result["valid_relationships"]}
        self.assertTrue(bases <= {"builtin_rule", "default", "context", "simulator"})
        simulated = ReasoningCauseAndEffect().perform_reasoning(
            self.EVENTS, conditions={"intensity": "high"},
            context={"max_time_gap": 5, "counterfactual_simulator": lambda c, e, ctx: "changed"})
        self.assertIn("simulator", {r["counterfactual_basis"] for r in simulated["valid_relationships"]})

    def test_intervention_is_labelled_as_confidence_scaling(self) -> None:
        result = ReasoningCauseAndEffect().perform_reasoning(
            self.EVENTS, conditions={"intensity": "high"},
            context={"max_time_gap": 5, "interventions": {"rain": {"strength": 0.5}}})
        intervened = [p for p in result["predictions"] if "intervention" in p]
        self.assertTrue(intervened)
        self.assertTrue(all(p["intervention_model"] == "confidence_scaling" for p in intervened))

    def test_cache_returns_independent_copies(self) -> None:
        engine = ReasoningCauseAndEffect()
        first = engine.perform_reasoning(self.EVENTS, conditions={"intensity": "high"}, context={"max_time_gap": 5})
        second = engine.perform_reasoning(self.EVENTS, conditions={"intensity": "high"}, context={"max_time_gap": 5})
        self.assertFalse(first["metrics"]["cache_hit"])
        self.assertTrue(second["metrics"]["cache_hit"])
        second["metrics"]["success"] = "tampered"
        third = engine.perform_reasoning(self.EVENTS, conditions={"intensity": "high"}, context={"max_time_gap": 5})
        self.assertIs(third["metrics"]["success"], True)

    def test_non_mapping_conditions_are_a_typed_error(self) -> None:
        with self.assertRaises(ReasoningValidationError):
            ReasoningCauseAndEffect().perform_reasoning(self.EVENTS, ["x"], {})  # type: ignore[arg-type]

    def test_duplicate_event_content_does_not_distort_distances(self) -> None:
        events = [{"id": f"e{i}", "description": "same text", "timestamp": float(i)} for i in range(6)]
        engine = ReasoningCauseAndEffect()
        normalized = engine._normalize_events(events, {})
        sequential = engine._identify_sequential_relationships(normalized)
        distances = {(r.cause["id"], r.effect["id"]): r.metadata["event_distance"] for r in sequential}
        self.assertEqual(distances[("e0", "e1")], 1)
        self.assertEqual(distances[("e2", "e4")], 2)


class DecompositionSoundnessTests(unittest.TestCase):
    CYCLIC = {"name": "svc", "components": {
        "a": {"functions": ["alpha"], "dependencies": ["b"]},
        "b": {"functions": ["beta"], "dependencies": ["c"]},
        "c": {"functions": ["gamma"], "dependencies": ["a"]},
    }}
    ACYCLIC = {"name": "svc", "components": {
        "a": {"functions": ["alpha"], "dependencies": ["b"]},
        "b": {"functions": ["beta"], "dependencies": ["c"]},
        "c": {"functions": ["gamma"]},
    }}

    def test_circular_dependencies_are_detected(self) -> None:
        result = ReasoningDecompositional().perform_reasoning(self.CYCLIC, {})
        cycles = result["system_understanding"]["circular_dependencies"]
        self.assertEqual(len(cycles), 1)
        self.assertEqual(len(cycles[0]), 3)
        kinds = [v["type"] for v in result["system_understanding"]["vulnerabilities"]]
        self.assertIn("circular_dependency", kinds)

    def test_acyclic_dependencies_report_no_cycle(self) -> None:
        result = ReasoningDecompositional().perform_reasoning(self.ACYCLIC, {})
        self.assertEqual(result["system_understanding"]["circular_dependencies"], [])

    def test_cycle_finder_handles_self_loops_and_long_chains_iteratively(self) -> None:
        find = ReasoningDecompositional._find_dependency_cycles
        self.assertEqual(find([{"source": "a", "target": "a"}]), [])  # self-loops are not multi-node cycles
        chain = [{"source": f"n{i}", "target": f"n{i + 1}"} for i in range(5000)]
        self.assertEqual(find(chain), [])
        self.assertEqual(len(find(chain + [{"source": "n5000", "target": "n0"}])), 1)

    def test_excessive_nesting_is_a_typed_error_not_recursion_error(self) -> None:
        root: dict = {"name": "x"}
        current = root
        for i in range(3000):
            nested = {"name": f"l{i}"}
            current["components"] = {"c": nested}
            current = nested
        with self.assertRaises(ReasoningValidationError):
            ReasoningDecompositional().perform_reasoning(root, {})

    def test_self_referencing_structures_terminate(self) -> None:
        loop: dict = {"name": "loop"}
        loop["components"] = {"self": loop}
        result = ReasoningDecompositional().perform_reasoning(loop, {})
        self.assertTrue(result["metrics"]["success"])
        cyclic_list: list = ["x"]
        cyclic_list.append(cyclic_list)
        ReasoningDecompositional().perform_reasoning({"name": "c", "items": cyclic_list}, {})

    def test_string_blob_is_bounded_and_cycle_safe(self) -> None:
        loop: dict = {"k": "v"}
        loop["self"] = loop
        self.assertIn("v", ReasoningDecompositional._string_blob(loop))
        deep: list = []
        current = deep
        for _ in range(5000):
            nxt: list = []
            current.append(nxt)
            current = nxt
        ReasoningDecompositional._string_blob(deep)

    def test_free_text_is_flagged_as_not_decomposed(self) -> None:
        result = ReasoningDecompositional().perform_reasoning("plan the migration", {})
        self.assertFalse(result["metrics"]["decomposed"])

    def test_large_inputs_sharing_a_prefix_do_not_share_a_cache_entry(self) -> None:
        """The cache key hashed a truncated repr, so inputs equal for the first 1500+ chars collided."""
        engine = ReasoningDecompositional()
        padding = "p" * 2500

        def system(tag):
            return {"name": "s", "components": {
                "a": {"functions": ["f"], "note": padding + tag}, "b": {"functions": ["g"]}}}

        engine.perform_reasoning(system("1"), {})
        second = engine.perform_reasoning(system("2"), {})
        self.assertTrue(second["decomposition_tree"]["content"]["components"]["a"]["note"].endswith("2"))

    def test_cached_report_is_a_private_copy(self) -> None:
        engine = ReasoningDecompositional()
        first = engine.perform_reasoning(self.ACYCLIC, {})
        first["metrics"]["total_components"] = -1
        second = engine.perform_reasoning(self.ACYCLIC, {})
        self.assertGreater(second["metrics"]["total_components"], 0)


if __name__ == "__main__":
    unittest.main()
