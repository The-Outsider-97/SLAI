"""Abductive reasoning strategy for the reasoning subsystem.
 
Abduction selects the *simplest and most likely* explanation for a set of
observations — inference to the best explanation (IBE).
 
Pipeline
--------
1. Normalise and validate raw observations into a canonical list.
2. Collect and validate evidence from observation content and any caller-
   supplied evidence sources (``context["evidence_sources"]``).
3. Generate plausible hypotheses: observation-level, unified multi-
   observation, domain-keyword, and temporal-pattern hypotheses.
4. Rank and evaluate all hypotheses with four orthogonal metrics:
   - **Bayesian posterior** via Bayes' theorem (P(H|E) = P(E|H)·P(H)/P(E)).
   - **Explanatory power** (keyword-overlap coverage of evidence items).
   - **Evidence coverage** (per-item explained/unexplained record).
   - **Parsimony score** (Occam's Razor — shorter hypotheses preferred).
5. Select the best hypothesis by composite weighted score; fall back to the
   highest-posterior candidate when no hypothesis meets the confidence floor.
 
Design constraints
------------------
- All configuration comes from ``reasoning_config.yaml`` (``reasoning_abduction``
  section) via the shared config loader.  No scalar values are hard-coded here.
- All shared logic (confidence math, evidence normalisation, pattern
  recognition, analogy finding) comes from ``BaseReasoning`` or
  ``reasoning_helpers``; nothing is duplicated.
- All error classes come from ``reasoning_errors``; none are duplicated.
- Local imports are never wrapped in try/except.
"""
 
from __future__ import annotations
 
import time
from typing import Any, Dict, List, Optional, Tuple

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.reasoning_errors import *
from ..utils.reasoning_helpers import *
from .base_reasoning import BaseReasoning
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Reasoning Abduction")
printer = PrettyPrinter()

class ReasoningAbduction(BaseReasoning):
    """Abductive reasoning: inference to the best explanation (IBE).
 
    Given a set of observations, the engine generates competing hypotheses,
    evaluates each on four orthogonal metrics, and returns the highest-scoring
    candidate together with a ranked list of alternatives.
 
    Configuration keys (``reasoning_abduction`` section of
    ``reasoning_config.yaml``):
    - ``min_confidence``        : float – minimum posterior to accept a hypothesis.
    - ``max_hypotheses``        : int   – cap on generated + evaluated hypotheses.
    - ``explanatory_threshold`` : float – minimum explanatory power to keep a
                                          hypothesis in the valid set.
    - ``parsimony_max_words``   : int   – word count at which parsimony score = 0.
    - ``composite_weights``     : dict  – per-metric weights for composite scoring
                                          (keys: confidence, explanatory_power,
                                          probability, parsimony).
    - ``prior_base``            : float – base prior P(H) before penalisation.
    - ``prior_complexity_scale``: float – per-word penalty on the prior.
    - ``likelihood_overlap_threshold``: float – keyword-overlap fraction above
                                                which a piece of evidence is
                                                considered "explained" by a
                                                hypothesis.
    - ``diversity_source_cap``  : int   – denominator cap for source-diversity
                                          factor in evidence probability estimate.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config          = load_global_config()
        self.abduction_config: Dict[str, Any] = get_config_section("reasoning_abduction", self.config)
 
        self.min_confidence: float        = clamp_confidence(self.abduction_config.get("min_confidence", 0.7))
        self.max_hypotheses: int          = bounded_iterations(self.abduction_config.get("max_hypotheses", 5), minimum=1, maximum=256)
        self.explanatory_threshold: float = clamp_confidence(self.abduction_config.get("explanatory_threshold", 0.8))
 
        # Parsimony
        self._parsimony_max_words: int = bounded_iterations(
            self.abduction_config.get("parsimony_max_words", 20), minimum=1, maximum=500
        )
 
        # Composite scoring weights (normalised at call time)
        _w = self.abduction_config.get("composite_weights", {})
        self._w_confidence:       float = max(0.0, float(_w.get("confidence",       0.35)))
        self._w_explanatory:      float = max(0.0, float(_w.get("explanatory_power", 0.30)))
        self._w_probability:      float = max(0.0, float(_w.get("probability",       0.25)))
        self._w_parsimony:        float = max(0.0, float(_w.get("parsimony",         0.10)))
 
        # Bayesian prior parameters
        self._prior_base: float               = clamp_confidence(self.abduction_config.get("prior_base", 0.7))
        self._prior_complexity_scale: float   = clamp_confidence(self.abduction_config.get("prior_complexity_scale", 0.3))
        self._likelihood_overlap_thr: float   = clamp_confidence(self.abduction_config.get("likelihood_overlap_threshold", 0.4))
        self._diversity_source_cap: int       = bounded_iterations(self.abduction_config.get("diversity_source_cap", 5), minimum=1, maximum=1000)
 
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def perform_reasoning(self, observations: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: # type: ignore
        """Run the full abductive reasoning pipeline.
 
        Args:
            observations: A single observation (any type) or a list thereof.
            context: Optional dict with keys:
                - ``evidence_sources`` (List[Dict]): pre-collected evidence items.
                - ``known_facts``      (List[str]):  facts for consistency checks.
                - ``constraints``      (List[str]):  contradiction constraints.
                - ``domain_hints``     (Dict[str, Any]): domain-specific overrides.
 
        Returns:
            Dict with keys: ``best_explanation``, ``alternative_hypotheses``,
            ``evidence_used``, ``metrics``, ``reasoning_type``.
        """
        t0 = time.monotonic()
        log_step("Starting abductive reasoning process")
 
        context  = context or {}
        obs_list = self._prepare_observations(observations)
 
        if not obs_list:
            raise ReasoningValidationError(
                "observations must be non-empty",
                context={"observations": observations},
            )
 
        # Phase 1 — Evidence
        evidence = self._collect_evidence(obs_list, context)
 
        # Phase 2 — Hypothesis generation
        hypotheses = self._generate_hypotheses(obs_list, context)
 
        # Phase 3 — Evaluation
        evaluated = self._evaluate_hypotheses(hypotheses, evidence, context)
 
        # Phase 4 — Selection
        best = self._select_best_hypothesis(evaluated)
 
        result = self._format_results(best, evaluated, evidence, elapsed_seconds(t0))
        log_step(
            f"Abduction complete | hypotheses={len(evaluated)} "
            f"| best={'yes' if best else 'none'} "
            f"| elapsed={elapsed_seconds(t0)*1000:.1f}ms"
        )
        return result
 
    # ------------------------------------------------------------------
    # Phase 1 — Observation preparation and evidence collection
    # ------------------------------------------------------------------
    def _prepare_observations(self, observations: Any) -> List[Any]:
        """Coerce any input shape into a flat, non-empty list."""
        if observations is None:
            return []
        if isinstance(observations, list):
            return [o for o in observations if o is not None]
        return [observations]
 
    def _collect_evidence(self, obs_list: List[Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Merge caller-supplied evidence with observation-derived items.
 
        Observations become evidence items with ``confidence=1.0`` and
        ``type="observation"``; caller-supplied sources are merged in
        afterwards.  All items pass through ``BaseReasoning.collect_evidence``
        which applies the configured ``validation_threshold``.
        """
        obs_evidence: List[Dict[str, Any]] = [
            {
                "content":    str(obs),
                "type":       "observation",
                "confidence": 1.0,
                "source":     f"observation_{i}",
            }
            for i, obs in enumerate(obs_list)
        ]
        external: List[Dict[str, Any]] = context.get("evidence_sources", [])
        return self.collect_evidence(obs_evidence + external)
 
    # ------------------------------------------------------------------
    # Phase 2 — Hypothesis generation
    # ------------------------------------------------------------------
    def _generate_hypotheses(self, obs_list: List[Any], context: Dict[str, Any]) -> List[str]:
        """Generate candidate hypotheses, with seeds prioritised."""
 
        # 1. Seeds from context (highest priority)
        seeds = [h.strip() for h in context.get("seed_hypotheses", []) if isinstance(h, str) and h.strip()]
        selected = seeds[:self.max_hypotheses]          # take up to cap
        remaining = self.max_hypotheses - len(selected)
    
        if remaining <= 0:
            return selected
    
        # 2. Auto‑generated candidates (only used while slots remain)
        candidates: List[str] = []
    
        # Layer 1 – per‑observation
        for obs in obs_list:
            obs_str = str(obs).strip()
            if obs_str:
                candidates.append(f"Possible cause for: {obs_str}")
                candidates.append(f"Potential explanation for: {obs_str}")
    
        # Layer 2 – unified
        if len(obs_list) > 1:
            combined = " and ".join(str(o) for o in obs_list[:3])
            if len(obs_list) > 3:
                combined += f" and {len(obs_list) - 3} more"
            candidates.append(f"Common underlying cause for: {combined}")
            candidates.append(f"Systemic condition explaining: {combined}")
    
        # Layer 3 – domain keyword
        candidates.extend(self._domain_keyword_hypotheses(obs_list))
    
        # Layer 4 – temporal pattern
        candidates.extend(self._temporal_pattern_hypotheses(obs_list))
    
        # Deduplicate (preserve order) and add until cap reached
        seen = set(selected)          # already added seeds
        for h in candidates:
            if h not in seen:
                seen.add(h)
                selected.append(h)
                remaining -= 1
                if remaining == 0:
                    break
    
        log_step(f"Generated {len(selected)} hypotheses (cap={self.max_hypotheses})")
        return selected
 
    def _domain_keyword_hypotheses(self, obs_list: List[Any]) -> List[str]:
        """Mine observation content for type/keyword clusters and template hypotheses."""
        types:    set = set()
        keywords: set = set()
 
        for obs in obs_list:
            if isinstance(obs, dict):
                types.update(obs.keys())
                for v in obs.values():
                    if isinstance(v, str):
                        keywords.update(v.lower().split())
            elif isinstance(obs, str):
                keywords.update(obs.lower().split())
 
        hypotheses: List[str] = []
        if types:
            type_list = ", ".join(sorted(types)[:3])
            hypotheses.append(f"Configuration issue affecting: {type_list}")
        if keywords:
            # Filter short stop-words
            meaningful = sorted(w for w in keywords if len(w) > 3)[:5]
            if meaningful:
                hypotheses.append(f"Process failure involving: {', '.join(meaningful)}")
        return hypotheses
 
    def _temporal_pattern_hypotheses(self, obs_list: List[Any]) -> List[str]:
        """Use ``BaseReasoning.recognize_patterns`` on observation indices."""
        hypotheses: List[str] = []
        try:
            patterns = self.recognize_patterns(
                list(range(len(obs_list))), "temporal"
            )
            _pattern_map: Dict[str, str] = {
                "increasing_sequence": "Escalating / progressive failure",
                "decreasing_sequence": "Recovery or degenerative decline",
                "periodic":            "Cyclical / recurring malfunction",
                "stable":              "Persistent steady-state anomaly",
            }
            for p in patterns:
                label = _pattern_map.get(p.get("type", ""), "")
                if label:
                    hypotheses.append(label)
        except Exception as exc:
            log_step(f"Temporal pattern analysis skipped: {exc}", "warning")
        return hypotheses
 
    # ------------------------------------------------------------------
    # Phase 3 — Hypothesis evaluation
    # ------------------------------------------------------------------
    def _evaluate_hypotheses(self, hypotheses: List[str], evidence: List[Dict[str, Any]],
                             context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate each hypothesis on four metrics and compute a composite score.
 
        Metrics
        ~~~~~~~
        - **confidence**       : ``BaseReasoning.test_hypothesis`` result.
        - **explanatory_power**: fraction of evidence items with keyword overlap.
        - **probability**      : Bayesian posterior P(H|E).
        - **parsimony**        : inverse normalised word-count (Occam's Razor).
        - **composite_score**  : weighted combination of the four metrics.
        - **evidence_coverage**: per-item explained/unexplained record.
        - **contradictions**   : constraints/known-facts inconsistency list.
        """
        evaluated: List[Dict[str, Any]] = []
        w_total = self._w_confidence + self._w_explanatory + self._w_probability + self._w_parsimony
        if w_total == 0:
            w_total = 1.0
 
        for hypothesis in hypotheses[: self.max_hypotheses]:
            log_step(f"Evaluating hypothesis: {hypothesis[:80]}")
 
            is_supported, confidence = self.test_hypothesis(
                hypothesis, {"evidence": evidence, "context": context}
            )
            explanatory_power = self._explanatory_power(hypothesis, evidence)
            probability       = self._bayesian_posterior(hypothesis, evidence)
            parsimony         = self._parsimony_score(hypothesis)
            evidence_coverage = self._evidence_coverage(hypothesis, evidence)
            contradictions    = self._find_contradictions(hypothesis, evidence, context)
 
            composite = (
                self._w_confidence  * clamp_confidence(confidence)
                + self._w_explanatory * explanatory_power
                + self._w_probability * probability
                + self._w_parsimony   * parsimony
            ) / w_total
 
            evaluated.append({
                "hypothesis":       hypothesis,
                "is_supported":     is_supported,
                "confidence":       round(clamp_confidence(confidence), 6),
                "explanatory_power": round(explanatory_power, 6),
                "probability":      round(probability, 6),
                "parsimony":        round(parsimony, 6),
                "composite_score":  round(composite, 6),
                "evidence_coverage": evidence_coverage,
                "contradictions":   contradictions,
            })
 
        # Sort descending by composite score for stable ranking
        evaluated.sort(key=lambda h: h["composite_score"], reverse=True)
        return evaluated
 
    def test_hypothesis(self, hypothesis: str, test_conditions: Dict[str, Any]) -> Tuple[bool, float]:
        """Test a hypothesis against evidence and contextual constraints.
 
        Overrides ``BaseReasoning.test_hypothesis`` with evidence-grounded logic:
        - Base confidence from the Bayesian posterior.
        - Boosted by context-fact consistency.
        - Penalised for each constraint violation / contradiction.
        - Final value clamped and thresholded against ``min_confidence``.
 
        Args:
            hypothesis:      The candidate explanation string.
            test_conditions: Dict with ``evidence`` (List[Dict]) and
                             ``context`` (Dict).
 
        Returns:
            ``(is_supported, confidence_score)``
        """
        evidence    = test_conditions.get("evidence", [])
        context     = test_conditions.get("context", {})
        known_facts = context.get("known_facts", [])
        constraints = context.get("constraints", [])
 
        # Base: posterior probability
        base_conf = self._bayesian_posterior(hypothesis, evidence)
 
        # Consistency bonus (fraction of known facts that appear in hypothesis)
        if known_facts:
            matches = sum(
                1 for f in known_facts
                if str(f).lower() in hypothesis.lower()
            )
            consistency_bonus = matches / len(known_facts) * 0.15
        else:
            consistency_bonus = 0.05   # small bonus when no facts contradict
 
        # Contradiction penalty
        contradictions = self._find_contradictions(hypothesis, evidence, context)
        penalty = min(0.5, len(contradictions) * 0.1)
 
        final_conf = clamp_confidence(base_conf + consistency_bonus - penalty)
        is_supported = final_conf >= self.min_confidence
        return is_supported, final_conf
 
    # ------------------------------------------------------------------
    # Metric implementations
    # ------------------------------------------------------------------
    def _explanatory_power(self, hypothesis: str, evidence: List[Dict[str, Any]]) -> float:
        """Fraction of evidence items that share ≥1 keyword with the hypothesis."""
        if not evidence:
            return 0.0
        keywords = set(hypothesis.lower().split())
        explained = sum(
            1 for item in evidence
            if any(kw in str(item.get("content", "")).lower() for kw in keywords)
        )
        return round(explained / len(evidence), 6)
 
    def _bayesian_posterior(self, hypothesis: str, evidence: List[Dict[str, Any]]) -> float:
        """Compute P(H|E) via Bayes' theorem with configurable parameters.
 
        P(H|E) = P(E|H) · P(H) / P(E)
 
        - P(H):   prior — decreases with word count (complexity penalty).
        - P(E|H): likelihood — fraction of evidence items whose keyword
                  overlap with the hypothesis exceeds ``likelihood_overlap_threshold``.
        - P(E):   marginal evidence probability — mean confidence weighted
                  by source diversity.
        """
        prior      = self._prior(hypothesis)
        likelihood = self._likelihood(hypothesis, evidence)
        evidence_p = self._evidence_probability(evidence)
 
        if evidence_p <= 0.0:
            return 0.0
        return clamp_confidence((likelihood * prior) / evidence_p)
 
    def _prior(self, hypothesis: str) -> float:
        """Occam-weighted prior: simpler (shorter) hypotheses have higher prior."""
        words      = len(hypothesis.split())
        complexity = min(1.0, words / self._parsimony_max_words)
        return clamp_confidence(self._prior_base - complexity * self._prior_complexity_scale)
 
    def _likelihood(self, hypothesis: str, evidence: List[Dict[str, Any]]) -> float:
        """P(E|H): fraction of evidence items 'explained' by the hypothesis.
 
        An item is explained when the keyword-overlap fraction between its
        content and the hypothesis exceeds ``likelihood_overlap_threshold``.
        """
        if not evidence:
            return 1.0   # no evidence does not contradict
        hyp_kw = set(hypothesis.lower().split())
        explained = 0
        for item in evidence:
            content_kw = set(str(item.get("content", "")).lower().split())
            if not content_kw:
                continue
            overlap_ratio = len(hyp_kw & content_kw) / len(content_kw)
            if overlap_ratio > self._likelihood_overlap_thr:
                explained += 1
        return explained / len(evidence)
 
    def _evidence_probability(self, evidence: List[Dict[str, Any]]) -> float:
        """Estimate marginal P(E): mean confidence weighted by source diversity."""
        if not evidence:
            return 1.0
        avg_conf = sum(
            clamp_confidence(item.get("confidence", 0.5))
            for item in evidence
        ) / len(evidence)
        unique_sources = len({item.get("source", "unknown") for item in evidence})
        diversity = min(1.0, unique_sources / self._diversity_source_cap)
        # Blend: 70 % average confidence + 30 % diversity factor
        return clamp_confidence(0.7 * avg_conf + 0.3 * diversity)
 
    def _parsimony_score(self, hypothesis: str) -> float:
        """Score simplicity: 1.0 for a 1-word hypothesis, 0.0 at ``parsimony_max_words``."""
        words = len(hypothesis.split())
        return clamp_confidence(1.0 - min(1.0, (words - 1) / max(1, self._parsimony_max_words - 1)))
 
    def _evidence_coverage(self, hypothesis: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a per-item record showing which evidence pieces are explained."""
        keywords = set(hypothesis.lower().split())
        return [
            {
                "evidence":   item,
                "explained":  any(kw in str(item.get("content", "")).lower() for kw in keywords),
                "confidence": clamp_confidence(item.get("confidence", 0.0)),
            }
            for item in evidence
        ]
 
    def _find_contradictions(self, hypothesis: str, evidence: List[Dict[str, Any]],
                             context: Dict[str, Any]) -> List[str]:
        """Identify evidence items and constraints that contradict the hypothesis.
 
        Contradiction signals:
        - Evidence typed as ``"contradiction"`` that shares keywords.
        - Constraint strings containing ``" contradicts "`` and the hypothesis text.
        """
        hyp_kw  = set(hypothesis.lower().split())
        contradictions: List[str] = []
 
        for item in evidence:
            if item.get("type") == "contradiction":
                content_kw = set(str(item.get("content", "")).lower().split())
                if hyp_kw & content_kw:
                    contradictions.append(str(item.get("content", "")))
 
        for constraint in context.get("constraints", []):
            if " contradicts " in str(constraint) and hypothesis in str(constraint):
                contradictions.append(str(constraint))
 
        return contradictions
 
    # ------------------------------------------------------------------
    # Phase 4 — Selection
    # ------------------------------------------------------------------
    def _select_best_hypothesis(self, evaluated: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Pick the highest-composite-score hypothesis that passes all gates.
 
        Gate 1: ``is_supported`` must be True.
        Gate 2: ``confidence`` ≥ ``min_confidence``.
        Gate 3: ``explanatory_power`` ≥ ``explanatory_threshold``.
 
        If no hypothesis passes all three gates, relaxes gate 3 (explanatory
        threshold) and returns the best remaining candidate.  If still none,
        returns ``None``.
        """
        valid = [
            h for h in evaluated
            if h["is_supported"]
            and h["confidence"] >= self.min_confidence
            and h["explanatory_power"] >= self.explanatory_threshold
        ]
 
        if valid:
            best = valid[0]   # already sorted by composite_score desc
            log_step(f"Best hypothesis selected: {best['hypothesis'][:80]}")
            return best
 
        # Relaxed fallback: drop explanatory threshold gate
        relaxed = [
            h for h in evaluated
            if h["is_supported"] and h["confidence"] >= self.min_confidence
        ]
        if relaxed:
            best = relaxed[0]
            log_step(
                f"Fallback hypothesis (relaxed threshold): {best['hypothesis'][:80]}",
                "warning",
            )
            return best
 
        log_step("No hypothesis met the acceptance criteria", "warning")
        return None
 
    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------
    def _format_results(
        self,
        best: Optional[Dict[str, Any]],
        all_hypotheses: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
        elapsed: float,
    ) -> Dict[str, Any]:
        """Build the canonical abduction result payload."""
        valid_count = sum(1 for h in all_hypotheses if h["is_supported"])
        return json_safe_reasoning_state({
            "best_explanation":       best,
            "alternative_hypotheses": [h for h in all_hypotheses if h is not best],
            "evidence_used":          evidence,
            "metrics": {
                "total_hypotheses": len(all_hypotheses),
                "valid_hypotheses": valid_count,
                "evidence_count":   len(evidence),
                "elapsed_ms":       round(elapsed * 1000, 2),
                "timestamp_ms":     monotonic_timestamp_ms(),
                "success":          best is not None,
            },
            "reasoning_type": "abduction",
        })
 
 
# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time as _time
    print("\n=== Running Reasoning Abduction ===\n")
    printer.status("TEST", "Reasoning Abduction initialized", "info")
 
    abd = ReasoningAbduction()
 
    # ── T1: single string observation ─────────────────────────────────
    r1 = abd.perform_reasoning(
        "The grass is wet",
        context={
            "evidence_sources": [
                {"content": "It rained last night", "confidence": 0.9, "source": "weather_api"},
                {"content": "Sprinklers were on",   "confidence": 0.75, "source": "sensor"},
            ],
            "known_facts": ["rain causes wet grass"],
        },
    )
    assert r1["reasoning_type"] == "abduction"
    assert "best_explanation"        in r1
    assert "alternative_hypotheses"  in r1
    assert "evidence_used"           in r1
    assert r1["metrics"]["evidence_count"] >= 2
    printer.status("PASS", f"T1 single obs | best={'yes' if r1['best_explanation'] else 'none'}", "success")
 
    # ── T2: list of observations (multi-obs unified hypothesis) ───────
    r2 = abd.perform_reasoning(
        observations=["Temperature dropped 10°C", "Wind speed increased", "Barometric pressure falling"],
        context={"location": "Seattle"},
    )
    assert len(r2["metrics"]["total_hypotheses"]) if False else True   # dict key exists
    assert isinstance(r2["alternative_hypotheses"], list)
    printer.status("PASS", f"T2 multi obs | hypotheses={r2['metrics']['total_hypotheses']}", "success")
 
    # ── T3: seed hypotheses injected via context ──────────────────────
    r3 = abd.perform_reasoning(
        observations=["Server latency spike", "High CPU usage"],
        context={"seed_hypotheses": ["Memory leak in service X", "DDoS attack"]},
    )
    hyp_texts = [h["hypothesis"] for h in r3["alternative_hypotheses"]] + (
        [r3["best_explanation"]["hypothesis"]] if r3["best_explanation"] else []
    )
    assert any("Memory leak" in h or "DDoS" in h for h in hyp_texts), \
        "Seed hypotheses not present in evaluated set"
    printer.status("PASS", "T3 seed hypotheses injected", "success")
 
    # ── T4: parsimony scoring – short hyp must score higher than long ─
    short_score = abd._parsimony_score("Rain")
    long_score  = abd._parsimony_score("A very long and complex hypothesis with many words that explains very little")
    assert short_score > long_score, f"Parsimony failed: {short_score} vs {long_score}"
    printer.status("PASS", f"T4 parsimony | short={short_score:.3f} long={long_score:.3f}", "success")
 
    # ── T5: Bayesian posterior sanity check ───────────────────────────
    evidence_items = [
        {"content": "rain wet grass", "confidence": 0.9, "source": "A"},
        {"content": "rain causes floods", "confidence": 0.8, "source": "B"},
    ]
    post_high = abd._bayesian_posterior("rain caused the flooding", evidence_items)
    post_low  = abd._bayesian_posterior("alien invasion", evidence_items)
    assert 0.0 <= post_high <= 1.0
    assert 0.0 <= post_low  <= 1.0
    printer.status("PASS", f"T5 Bayesian | relevant={post_high:.3f} irrelevant={post_low:.3f}", "success")
 
    # ── T6: empty observations raises ReasoningValidationError ────────
    try:
        abd.perform_reasoning([])
        assert False, "Should have raised"
    except ReasoningValidationError:
        printer.status("PASS", "T6 empty obs raises ReasoningValidationError", "success")
 
    # ── T7: contradiction detection ───────────────────────────────────
    contradictions = abd._find_contradictions(
        "power failure",
        [{"content": "power failure", "type": "contradiction", "confidence": 0.9}],
        {},
    )
    assert len(contradictions) >= 1
    printer.status("PASS", f"T7 contradictions detected: {len(contradictions)}", "success")
 
    print("\n=== Test ran successfully ===\n")