import math

from abc import ABC
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from collections import defaultdict

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.reasoning_errors import *
from ..utils.reasoning_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Base Reasoning")
printer = PrettyPrinter()


class BaseReasoning(ABC):
    """
    Production‑ready base class for all reasoning types.

    Provides:
    - Configuration management (YAML, caching, thread‑safe reload)
    - Standardised logging and pretty printing
    - Common reasoning methods (evidence collection, hypothesis generation,
      pattern recognition, analogy finding, contradiction detection, etc.)
    - Integration with reasoning_helpers and reasoning_errors modules
    - Lifecycle methods that subclasses must implement or can override
    """

    def __init__(self) -> None:
        # Global and section‑specific config
        self.config = load_global_config()
        self.base_config = get_config_section("base_reasoning")

        # Core thresholds (safe defaults if missing)
        self.confidence: float = self.base_config.get("confidence", 0.75)
        self.is_supported: bool = self.base_config.get("is_supported", True)
        self.validation_threshold: float = self.base_config.get("validation_threshold", 0.8)
        self.similarity_threshold: float = self.base_config.get("similarity_threshold", 0.7)

        # Runtime state
        self.evidence: List[Dict[str, Any]] = []
        self.active_hypotheses: List[str] = []
        self.validated_results: List[Any] = []

    # ----------------------------------------------------------------------
    # Core reasoning interface (subclasses MUST implement)
    # ----------------------------------------------------------------------
    @abstractmethod
    def perform_reasoning(self, input_data: Any, context: Optional[Dict] = None) -> Any:
        """
        Main reasoning method to be implemented by subclasses.

        Args:
            input_data: Domain‑specific input (observations, premises, system, etc.)
            context: Additional contextual information (evidence, constraints, etc.)

        Returns:
            Reasoning result (structure depends on concrete reasoning type)
        """
        raise NotImplementedError("Subclasses must implement perform_reasoning")

    # ----------------------------------------------------------------------
    # Evidence handling
    # ----------------------------------------------------------------------
    def collect_evidence(self, sources: List[Any]) -> List[Dict]:
        """
        Collect and validate evidence from multiple sources.

        Each source is expected to be a dict with at least a 'confidence' key.
        Sources that pass validation are stored in `self.evidence`.

        Returns:
            List of validated evidence items.
        """
        validated = []
        for source in sources:
            if not isinstance(source, dict):
                log_step(f"Skipping non‑dict evidence source: {source}", "warning")
                continue
            if self.validate_evidence(source, self.validation_threshold):
                self.evidence.append(source)
                validated.append(source)
                log_step(f"Evidence collected: {source.get('id', 'unknown')}")
        return validated

    def validate_evidence(self, evidence: Dict, min_confidence: float) -> bool:
        """
        Validate a single evidence item based on confidence and basic structure.

        Args:
            evidence: Dict containing at least 'confidence' (float 0‑1)
            min_confidence: Minimum required confidence

        Returns:
            True if evidence is valid and confidence >= min_confidence.
        """
        try:
            conf = clamp_confidence(evidence.get("confidence", 0.0))
        except ConfidenceBoundsError as e:
            log_step(f"Invalid confidence in evidence: {e}", "warning")
            return False

        if conf < min_confidence:
            log_step(
                f"Low confidence evidence rejected: {evidence.get('source', 'unknown')} "
                f"(score: {conf})",
                "warning"
            )
            return False
        return True

    # ----------------------------------------------------------------------
    # Hypothesis generation and testing
    # ----------------------------------------------------------------------
    def generate_hypotheses(self, observations: List[str]) -> List[str]:
        """
        Generate simple explanatory hypotheses from observations.

        Subclasses may override this with domain‑specific heuristics.

        Args:
            observations: List of observed phenomena (strings).

        Returns:
            List of hypothesis strings.
        """
        hypotheses = []
        for obs in observations:
            # Basic pattern: "Potential explanation for: <observation>"
            hypothesis = f"Potential explanation for: {obs}"
            hypotheses.append(hypothesis)
            self.active_hypotheses.append(hypothesis)
            log_step(f"Hypothesis generated: {hypothesis}")
        return hypotheses

    def test_hypothesis(self, hypothesis: str, test_conditions: Dict) -> Tuple[bool, float]:
        """
        Basic hypothesis testing against given conditions.

        Subclasses should override with domain‑specific logic.
        Default implementation returns the configured `is_supported` and `confidence`.

        Args:
            hypothesis: The hypothesis to test.
            test_conditions: Dict containing evidence, context, etc.

        Returns:
            Tuple (is_supported, confidence_score).
        """
        log_step(f"Testing hypothesis: {hypothesis[:80]}...")
        return self.is_supported, self.confidence

    # ----------------------------------------------------------------------
    # Causal and relational analysis
    # ----------------------------------------------------------------------
    def identify_causal_relationships(self, events: List[str]) -> List[Tuple[str, str]]:
        """
        Identify simple cause‑effect relationships between sequential events.

        Args:
            events: List of event descriptions in temporal order.

        Returns:
            List of (cause, effect) pairs for consecutive events.
        """
        relationships = []
        for i in range(len(events) - 1):
            cause = events[i]
            effect = events[i + 1]
            relationships.append((cause, effect))
            log_step(f"Identified causal relationship: {cause} -> {effect}")
        return relationships

    # ----------------------------------------------------------------------
    # Decomposition
    # ----------------------------------------------------------------------
    def decompose_whole(self, whole: Any, decomposition_strategy: str = "functional") -> List[Any]:
        """
        Break down a whole into constituent parts.

        Subclasses should override with domain‑specific decomposition logic.

        Args:
            whole: The entity to decompose.
            decomposition_strategy: Strategy hint ("functional", "structural", etc.)

        Returns:
            List of components.
        """
        # Placeholder – in real systems this would be domain‑specific
        components = [f"Component_{i}" for i in range(3)]
        log_step(f"Decomposed {whole} into {len(components)} parts using {decomposition_strategy} strategy")
        return components

    # ----------------------------------------------------------------------
    # Pattern recognition
    # ----------------------------------------------------------------------
    def recognize_patterns(self, data_series: List[Any], pattern_type: str = "temporal") -> List[Dict]:
        """
        Recognise patterns in data series (temporal, spatial, behavioural).

        Args:
            data_series: Sequential data points (numeric or comparable).
            pattern_type: Type of pattern to detect.

        Returns:
            List of pattern dictionaries with 'type', 'confidence', and other metadata.
        """
        patterns = []

        if pattern_type == "temporal" and len(data_series) > 1:
            # Detect increasing sequence (strict monotonic increase)
            try:
                if all(data_series[i] < data_series[i + 1] for i in range(len(data_series) - 1)):
                    patterns.append({
                        "type": "increasing_sequence",
                        "length": len(data_series),
                        "confidence": 0.9
                    })
            except TypeError:
                log_step("Cannot compare elements for increasing sequence", "warning")

            # Detect periodic pattern (constant step)
            if len(data_series) > 2 and all(isinstance(x, (int, float)) for x in data_series):
                diffs = [data_series[i + 1] - data_series[i] for i in range(len(data_series) - 1)]
                if all(abs(diffs[i] - diffs[0]) < 0.1 * abs(diffs[0]) for i in range(1, len(diffs))):
                    patterns.append({
                        "type": "periodic",
                        "period": diffs[0],
                        "confidence": 0.85
                    })

        elif pattern_type == "spatial" and len(data_series) > 2 and all(isinstance(x, (int, float)) for x in data_series):
            # Simple cluster detection by gap analysis
            sorted_data = sorted(data_series)
            gaps = [sorted_data[i + 1] - sorted_data[i] for i in range(len(sorted_data) - 1)]
            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                clusters = []
                current_cluster = [sorted_data[0]]
                for i in range(1, len(sorted_data)):
                    if sorted_data[i] - sorted_data[i - 1] > 2 * avg_gap:
                        clusters.append(current_cluster)
                        current_cluster = [sorted_data[i]]
                    else:
                        current_cluster.append(sorted_data[i])
                clusters.append(current_cluster)
                if len(clusters) > 1:
                    patterns.append({
                        "type": "clusters",
                        "count": len(clusters),
                        "confidence": 0.8
                    })

        if not patterns:
            patterns.append({
                "type": "unknown",
                "message": f"No recognised patterns of type '{pattern_type}'",
                "confidence": 0.0
            })

        log_step(f"Recognised {len(patterns)} {pattern_type} pattern(s)")
        return patterns

    # ----------------------------------------------------------------------
    # Logical conclusion
    # ----------------------------------------------------------------------
    def draw_logical_conclusion(self, premises: List[str], logic_type: str = "deductive") -> str:
        """
        Draw a logical conclusion from a set of premises.

        Placeholder – subclasses should implement proper inference.

        Args:
            premises: List of premise strings.
            logic_type: "deductive", "inductive", "abductive".

        Returns:
            Conclusion string.
        """
        conclusion = f"Conclusion based on {len(premises)} premises using {logic_type} reasoning"
        log_step(f"Drew logical conclusion: {conclusion}")
        return conclusion

    # ----------------------------------------------------------------------
    # Analogical reasoning helpers
    # ----------------------------------------------------------------------
    def find_analogies(self, target: Any, domain: List[Any]) -> List[Dict]:
        """
        Find analogies between a target and items in a domain.

        Uses property extraction and Jaccard similarity.

        Args:
            target: The subject to find analogies for.
            domain: Collection of candidate analogs.

        Returns:
            List of dicts with keys: 'item', 'similarity', 'matching_properties'.
        """
        analogs = []
        target_props = self._extract_properties(target)

        for item in domain:
            item_props = self._extract_properties(item)
            similarity = self._calculate_similarity(target_props, item_props)
            if similarity >= self.similarity_threshold:
                analogs.append({
                    "item": item,
                    "similarity": similarity,
                    "matching_properties": list(target_props & item_props)
                })

        log_step(f"Found {len(analogs)} analogies for {target} (threshold: {self.similarity_threshold})")
        return analogs

    def _extract_properties(self, entity: Any) -> Set[str]:
        """
        Extract a set of structural/functional properties from an entity.

        Supports dict, object, and string representations.
        """
        if isinstance(entity, str):
            # Split by spaces and underscores to get meaningful tokens
            tokens = set()
            for part in entity.replace("_", " ").split():
                tokens.update(part.lower().split())
            return tokens
        elif isinstance(entity, dict):
            props = set()
            for k, v in entity.items():
                props.add(str(k).lower())
                if isinstance(v, list):
                    for elem in v:
                        props.update(self._extract_properties(elem))
                elif isinstance(v, dict):
                    props.update(self._extract_properties(v))
                elif isinstance(v, str):
                    props.update(self._extract_properties(v))
            return props
        elif hasattr(entity, "__dict__"):
            return set(str(k).lower() for k in entity.__dict__.keys())
        else:
            return {str(entity).lower()}

    def _calculate_similarity(self, props1: Set[str], props2: Set[str]) -> float:
        """Calculate Jaccard similarity between two property sets."""
        if not props1 and not props2:
            return 1.0
        intersection = len(props1 & props2)
        union = len(props1 | props2)
        return intersection / union if union > 0 else 0.0

    # ----------------------------------------------------------------------
    # Probabilistic evaluation
    # ----------------------------------------------------------------------
    def evaluate_probability(self, hypothesis: str, evidence: List[Dict]) -> float:
        """
        Evaluate the probability of a hypothesis given evidence using a simplified
        Bayesian approach.

        Subclasses may override with more sophisticated models.

        Returns:
            Probability in [0, 1].
        """
        # Prior: simple heuristic based on hypothesis length
        prior = 0.5 - 0.02 * len(hypothesis.split())  # longer → lower prior
        prior = clamp_confidence(prior)

        # Likelihood: fraction of evidence items that contain hypothesis keywords
        keywords = set(hypothesis.lower().split())
        if not evidence:
            return prior

        supportive = 0
        for item in evidence:
            content = str(item.get("content", "")).lower()
            if any(kw in content for kw in keywords):
                supportive += 1
        likelihood = supportive / len(evidence)

        # Evidence probability (marginal) – simplified as average confidence
        evidence_prob = weighted_confidence([item.get("confidence", 0.5) for item in evidence])

        if evidence_prob <= 0:
            return 0.0

        posterior = (likelihood * prior) / evidence_prob
        return clamp_confidence(posterior)

    # ----------------------------------------------------------------------
    # Contradiction detection (enhanced with helpers)
    # ----------------------------------------------------------------------
    def identify_contradictions(self, statements: List[str]) -> List[Tuple[str, str, str]]:
        """
        Identify logical contradictions among a list of statements.

        Uses:
        - Direct negation detection (A vs not A)
        - Inverse‑object contradiction (using `detect_inverse_contradiction` helper)
        - Basic predicate conflict (same subject, different predicates)

        Returns:
            List of tuples (statement1, statement2, contradiction_type).
        """
        contradictions = []
        # Normalize statements to canonical fact form where possible
        facts = []
        for stmt in statements:
            try:
                fact = normalize_fact(stmt)
                facts.append((stmt, fact))
            except FactNormalizationError:
                # If we can't normalise, treat as raw string for later checks
                facts.append((stmt, None))

        # 1. Direct negation (based on string patterns)
        for i, (stmt_i, _) in enumerate(facts):
            for j, (stmt_j, _) in enumerate(facts[i + 1:], i + 1):
                if self._are_direct_negations(stmt_i, stmt_j):
                    contradictions.append((stmt_i, stmt_j, "direct_negation"))

        # 2. Inverse‑object contradiction using helper (requires fact normalisation)
        fact_dict = {stmt: fact for stmt, fact in facts if fact is not None}
        for stmt, fact in fact_dict.items():
            # Convert to knowledge base dict with arbitrary confidence (1.0)
            temp_kb = {fact: 1.0}
            if detect_inverse_contradiction(fact, temp_kb, threshold=0.5):
                # Find the inverse statement in the original list
                s, p, o = fact
                inverse_fact = (s, p, f"not_{o}")
                for other_stmt, other_fact in fact_dict.items():
                    if other_fact == inverse_fact:
                        contradictions.append((stmt, other_stmt, "inverse_object"))
                        break

        # 3. Predicate conflict (same subject, different predicates)
        subject_map = defaultdict(list)
        for stmt, fact in fact_dict.items():
            s, p, _ = fact
            subject_map[s].append((stmt, p))

        for s, pairs in subject_map.items():
            if len(pairs) < 2:
                continue
            for i in range(len(pairs)):
                stmt_i, pred_i = pairs[i]
                for j in range(i + 1, len(pairs)):
                    stmt_j, pred_j = pairs[j]
                    if pred_i != pred_j:
                        contradictions.append((stmt_i, stmt_j, "predicate_conflict"))

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for c in contradictions:
            key = tuple(sorted(c[:2]) + [c[2]])
            if key not in seen:
                seen.add(key)
                unique.append(c)

        log_step(f"Identified {len(unique)} logical contradiction(s)")
        return unique

    def _are_direct_negations(self, stmt1: str, stmt2: str) -> bool:
        """Check if two statements are direct negations (e.g., 'A is B' vs 'A is not B')."""
        neg_prefixes = ("not ", "does not ", "is not ", "are not ", "cannot ", "no ")
        stmt1_low = stmt1.lower().strip()
        stmt2_low = stmt2.lower().strip()

        # Check if stmt2 is stmt1 with a negation prefix
        for prefix in neg_prefixes:
            if stmt2_low.startswith(prefix) and stmt2_low[len(prefix):] == stmt1_low:
                return True
            if stmt1_low.startswith(prefix) and stmt1_low[len(prefix):] == stmt2_low:
                return True

        # Simple word‑based negation (e.g., "rain" vs "not rain")
        words1 = set(stmt1_low.split())
        words2 = set(stmt2_low.split())
        if len(words1) == 1 and len(words2) == 1:
            return False  # Single‑word negation not handled here
        # If one statement contains "not" and the other does not, and their tokens otherwise match
        if ("not" in words1) ^ ("not" in words2):
            base1 = words1 - {"not"}
            base2 = words2 - {"not"}
            if base1 == base2:
                return True
        return False

    # ----------------------------------------------------------------------
    # Additional helpers that subclasses may find useful
    # ----------------------------------------------------------------------
    def normalize_evidence_dict(self, evidence: Optional[Dict]) -> Dict[str, Any]:
        """Wrapper around `normalize_evidence` from helpers."""
        return normalize_evidence(evidence)

    def merge_evidence_dicts(self, *evidence_blobs: Optional[Dict]) -> Dict[str, Any]:
        """Wrapper around `merge_evidence` from helpers."""
        return merge_evidence(*evidence_blobs)

    def clamp_confidence(self, value: float) -> float:
        """Clamp confidence to [0,1] using helper."""
        return clamp_confidence(value)

    def merge_confidences(self, conf_a: float, conf_b: float) -> float:
        """Probabilistic OR merge of confidences."""
        return merge_confidence(conf_a, conf_b)


# ----------------------------------------------------------------------
# Self‑test block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Base Reasoning ===\n")
    printer.status("TEST", "Base Reasoning initialised", "info")

    # Minimal concrete subclass for testing
    class ConcreteReasoning(BaseReasoning):
        def perform_reasoning(self, input_data: Any, context: Optional[Dict] = None) -> Any:
            return {"test": "result", "input": input_data, "context": context}

    reasoning = ConcreteReasoning()
    printer.pretty("START", reasoning, "success" if reasoning else "error")

    # 1. Evidence collection
    sources = [
        {"id": "E1", "confidence": 0.95, "source": "sensor_A", "content": "It rained"},
        {"id": "E2", "confidence": 0.4, "source": "sensor_B", "content": "Sprinkler on"},
        {"id": "E3", "confidence": 0.85, "source": "user", "content": "Grass wet"},
    ]
    validated = reasoning.collect_evidence(sources)
    printer.pretty("Evidence collected", validated, "success" if validated else "error")
    assert len(validated) == 2, "Low confidence evidence should be filtered"

    # 2. Hypothesis generation
    hypotheses = reasoning.generate_hypotheses(["Grass wet", "Temperature drop"])
    printer.pretty("Hypotheses", hypotheses, "success" if hypotheses else "error")

    # 3. Causal relationships
    events = ["Rain starts", "Ground gets wet", "Puddles form"]
    causal = reasoning.identify_causal_relationships(events)
    printer.pretty("Causal relationships", causal, "success" if causal else "error")
    assert len(causal) == 2, "Should have 2 cause‑effect pairs"

    # 4. Pattern recognition
    temp_series = [22.5, 23.1, 23.8, 24.6, 25.4]
    patterns = reasoning.recognize_patterns(temp_series, "temporal")
    printer.pretty("Temporal patterns", patterns, "success" if patterns else "error")
    assert any(p["type"] == "increasing_sequence" for p in patterns), "Increasing sequence not detected"

    # 5. Analogy finding
    target = {"name": "Smart Fridge", "features": ["cooling", "wifi", "energy_saving"]}
    domain = [
        {"name": "Smart Oven", "features": ["heating", "wifi", "energy_saving"]},
        {"name": "Dumb Fridge", "features": ["cooling"]},
        {"name": "Smart Thermostat", "features": ["cooling", "wifi", "energy_saving"]}
    ]
    analogies = reasoning.find_analogies(target, domain)
    printer.pretty("Analogies", analogies, "success" if analogies else "error")
    assert len(analogies) >= 1, "Should find at least one analogy"

    # 6. Contradiction detection – use direct negations to guarantee detection
    statements = [
        "sky is blue",
        "sky is not blue",
        "water is wet",
        "water is not wet"
    ]
    contradictions = reasoning.identify_contradictions(statements)
    printer.pretty("Contradictions", contradictions, "success" if contradictions else "error")
    # Expect at least two contradictions: (sky blue vs not blue) and (water wet vs not wet)
    assert len(contradictions) >= 2, "Should detect both direct negations"

    # 7. Probability evaluation
    hypothesis = "Grass is wet because of rain"
    evidence_list = [
        {"content": "It rained last night", "confidence": 0.9},
        {"content": "Sprinklers were off", "confidence": 0.8}
    ]
    prob = reasoning.evaluate_probability(hypothesis, evidence_list)
    printer.pretty("Probability", prob, "success" if prob > 0 else "error")
    assert 0.0 <= prob <= 1.0, "Probability out of bounds"

    # 8. Decomposition (placeholder)
    decomposed = reasoning.decompose_whole("E‑commerce System", "functional")
    printer.pretty("Decomposition", decomposed, "success" if decomposed else "error")

    # 9. Logical conclusion
    conclusion = reasoning.draw_logical_conclusion(["All men are mortal", "Socrates is a man"], "deductive")
    printer.pretty("Conclusion", conclusion, "success" if conclusion else "error")

    print("\n=== Test ran successfully ===\n")