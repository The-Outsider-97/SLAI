
from abc import ABC
from typing import List, Dict, Any, Tuple, Optional

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Base Reasoning")
printer = PrettyPrinter

class BaseReasoning(ABC):
    """
    Base class for all reasoning types providing common functionality for:
    - Configuration management
    - Logging and pretty printing
    - Global configuration access
    """
    def __init__(self):
        self.config = load_global_config()
        self.base_config = get_config_section("base_reasoning")
        self.confidence = self.base_config.get('confidence')
        self.is_supported = self.base_config.get('is_supported')
        self.validation_threshold = self.base_config.get('validation_threshold')
        self.similarity_threshold = self.base_config.get('similarity_threshold')

        self.evidence: List[Dict[str, Any]] = []
        self.active_hypotheses: List[str] = []
        self.validated_results: List[Any] = []

    def get_base_config_value(self, key: str, default=None):
        """
        Retrieve a value from the base reasoning configuration
        """
        return self.base_config.get(key, default)

    def log_step(self, message: str, level: str = "info"):
        """
        Unified logging method for reasoning steps
        """
        log_func = getattr(logger, level, logger.info)
        log_func(message)

    # @abstractmethod
    def perform_reasoning(self, input_data: any, context: dict = None) -> any:
        """
        Main reasoning method to be implemented by subclasses.
        """
        pass

    def print_reasoning(self, title: str, content: str):
        """
        Pretty-print reasoning output with formatted title
        """
        printer.print_header(title)
        printer.print_content(content)
        printer.print_footer()

    # -------------------------
    # Common Reasoning Methods
    # -------------------------
    
    def collect_evidence(self, sources: List[Any]) -> List[Dict]:
        """
        Collect and validate evidence from multiple sources
        """
        validated = []
        for source in sources:
            if self.validate_evidence(source, self.validation_threshold):
                self.evidence.append(source)
                validated.append(source)
                self.log_step(f"Evidence collected: {source.get('id', 'unknown')}")
        return validated

    def validate_evidence(self, evidence: Dict, min_confidence: float) -> bool:
        """
        Validate evidence based on confidence score and source reliability
        """
        confidence = evidence.get('confidence', 0.0)
        source = evidence.get('source', 'unknown')
        
        if confidence < min_confidence:
            self.log_step(f"Low confidence evidence rejected: {source} (score: {confidence})", "warning")
            return False
        return True

    def generate_hypotheses(self, observations: List[str]) -> List[str]:
        """
        Generate hypotheses based on observations
        """
        hypotheses = []
        for obs in observations:
            hypothesis = f"Potential explanation for: {obs}"
            hypotheses.append(hypothesis)
            self.active_hypotheses.append(hypothesis)
            self.log_step(f"Hypothesis generated: {hypothesis}")
        return hypotheses

    def test_hypothesis(self, hypothesis: str, test_conditions: Dict) -> Tuple[bool, float]:
        """
        Test a hypothesis against specific conditions
        """
        # Placeholder implementation - would be domain-specific
        confidence = self.confidence
        is_supported = self.is_supported
        self.log_step(f"Testing hypothesis: {hypothesis}")
        return is_supported, confidence

    def identify_causal_relationships(self, events: List[str]) -> List[Tuple[str, str]]:
        """
        Identify potential cause-effect relationships between events
        """
        relationships = []
        for i in range(len(events) - 1):
            cause = events[i]
            effect = events[i + 1]
            relationships.append((cause, effect))
            self.log_step(f"Identified causal relationship: {cause} -> {effect}")
        return relationships

    def decompose_whole(self, whole: Any, decomposition_strategy: str = "functional") -> List[Any]:
        """
        Break down a whole into constituent parts
        """
        # Placeholder implementation - would be domain-specific
        components = [f"Component_{i}" for i in range(3)]
        self.log_step(f"Decomposed {whole} into {len(components)} parts")
        return components

    def recognize_patterns(self, data_series: List[Any], pattern_type: str = "temporal") -> List[Dict]:
        """
        Recognize patterns in data series using statistical analysis and sequence detection
        Args:
            data_series: Sequential data points
            pattern_type: Type of pattern to recognize (temporal/spatial/behavioral)
        Returns:
            List of identified patterns with metadata
        """
        patterns = []
        
        # Temporal pattern detection (sequences, trends, cycles)
        if pattern_type == "temporal" and len(data_series) > 1:
            # Detect increasing sequences
            if all(data_series[i] < data_series[i+1] for i in range(len(data_series)-1)):
                patterns.append({
                    "type": "increasing_sequence",
                    "length": len(data_series),
                    "confidence": 0.9
                })
            
            # Detect periodic patterns
            if len(data_series) > 2:
                diffs = [data_series[i+1] - data_series[i] for i in range(len(data_series)-1)]
                if all(abs(diffs[i] - diffs[0]) < 0.1 * abs(diffs[0]) for i in range(1, len(diffs))):
                    patterns.append({
                        "type": "periodic",
                        "period": diffs[0],
                        "confidence": 0.85
                    })
        
        # Spatial pattern detection (clusters, distributions)
        elif pattern_type == "spatial" and len(data_series) > 2:
            # Simple cluster detection
            sorted_data = sorted(data_series)
            gaps = [sorted_data[i+1] - sorted_data[i] for i in range(len(sorted_data)-1)]
            avg_gap = sum(gaps) / len(gaps)
            
            clusters = []
            current_cluster = [sorted_data[0]]
            
            for i in range(1, len(sorted_data)):
                if sorted_data[i] - sorted_data[i-1] > 2 * avg_gap:
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
        
        # Fallback for unrecognized pattern types
        if not patterns:
            patterns.append({
                "type": "unknown",
                "message": f"No recognized patterns of type '{pattern_type}'",
                "confidence": 0.0
            })
        
        self.log_step(f"Recognized {len(patterns)} {pattern_type} patterns in data series")
        return patterns

    def draw_logical_conclusion(self, premises: List[str], logic_type: str = "deductive") -> str:
        """
        Draw logical conclusion from premises
        """
        conclusion = f"Conclusion based on {len(premises)} premises"
        self.log_step(f"Drew logical conclusion: {conclusion}")
        return conclusion

    def find_analogies(self, target: Any, domain: List[Any]) -> List[Any]:
        """
        Find structural and functional analogies between target and domain items
        Args:
            target: The subject to find analogies for
            domain: Collection of potential analogs
        Returns:
            List of analogous items with similarity scores
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
                    "matching_properties": [prop for prop in target_props if prop in item_props]
                })
        
        self.log_step(f"Found {len(analogs)} analogies for {target} (threshold: {self.similarity_threshold})")
        return analogs
    
    def _extract_properties(self, entity: Any) -> set:
        """Extract structural and functional properties of an entity"""
        if isinstance(entity, str):
            return set(entity.split())
        elif isinstance(entity, dict):
            props = set()
            for k, v in entity.items():
                props.add(k)
                if isinstance(v, list):
                    props.update(v)
                elif isinstance(v, str):
                    props.update(v.split())
                elif isinstance(v, dict):
                    props.update(v.keys())
            return props
        elif hasattr(entity, '__dict__'):
            return set(entity.__dict__.keys())
        return set()
    
    def _calculate_similarity(self, props1: set, props2: set) -> float:
        """Calculate Jaccard similarity between two property sets"""
        intersection = len(props1 & props2)
        union = len(props1 | props2)
        return intersection / union if union > 0 else 0.0

    def evaluate_probability(self, hypothesis: str, evidence: List[Dict]) -> float:
        """
        Evaluate probability of a hypothesis given evidence
        """
        # Placeholder implementation - would use Bayesian reasoning
        return 0.8

    def identify_contradictions(self, statements: List[str]) -> List[Tuple[str, str]]:
        """
        Identify logical contradictions using propositional logic analysis
        Args:
            statements: List of statements to check
        Returns:
            List of contradictory statement pairs with conflict type
        """
        contradictions = []
        # normalized = [self._normalize_statement(s) for s in statements]
        
        # Create statement dictionary with negation mapping
        stmt_dict = {}
        for i, stmt in enumerate(statements):
            stmt_dict[i] = {
                "original": statements[i],
                "normalized": stmt,
                "negation": self._get_negation(stmt)
            }
        
        # Check for direct contradictions (A vs ¬A)
        for i, stmt_data in stmt_dict.items():
            for j, other_data in stmt_dict.items():
                if i != j:
                    # Check if statements are direct negations
                    if stmt_data["normalized"] == other_data["negation"]:
                        contradictions.append((
                            f"{statements[i]} (contradicts)",
                            f"{statements[j]} (negation)",
                            "direct_negation"
                        ))
                    
                    # Check for mutual exclusivity (A∧B vs ¬A∨¬B)
                    elif (stmt_data["negation"] in other_data["normalized"] or 
                          other_data["negation"] in stmt_data["normalized"]):
                        contradictions.append((
                            f"{statements[i]} (exclusive with)",
                            f"{statements[j]}",
                            "mutual_exclusion"
                        ))
        
        # Remove duplicates
        unique_contradictions = list(set(contradictions))
        self.log_step(f"Identified {len(unique_contradictions)} logical contradictions")
        return unique_contradictions
    
    def _get_negation(self, statement: str) -> str:
        """Generate logical negation of a statement"""
        # Simple negation handling - would use more sophisticated NLP in real system
        negation_words = {"is": "is not", "are": "are not", "has": "has not", 
                         "can": "cannot", "will": "will not", "does": "does not"}
        
        for word, negation in negation_words.items():
            if word in statement:
                return statement.replace(word, negation)
        
        return "not " + statement
    
if __name__ == "__main__":
    print("\n=== Running Base Reasoning Types ===\n")
    printer.status("TEST", "Starting Base Reasoning Types tests", "info")

    base = BaseReasoning()
    base.similarity_threshold = 0.4
    printer.pretty("START", base, "success" if base else "error")

    print("\n* * * * * Phase 2 - Rules * * * * *\n")
    sources = [
        {"id": "E1", "confidence": 0.95, "source": "sensor_A"},
        {"id": "E2", "confidence": 0.4, "source": "sensor_B"}
    ]
    events = ["Event_A", "Event_B", "Event_C"]

    evidence = base.collect_evidence(sources=sources)
    causal = base.identify_causal_relationships(events=events)
    printer.pretty("evidence", evidence, "success" if evidence else "error")
    printer.pretty("causal", causal, "success" if causal else "error")

    print("\n* * * * * Phase 3 * * * * *\n")
    type1= "temporal"
    type2 = "deductive"
    data_series= [1.2, 1.4, 1.6, 1.8, 2.0]
    premise = [
        "The air conditioner is turned on",
        "The power consumption increases significantly",
        "Increased power usage is correlated with AC operation"
    ]
    target = {
        "name": "Smart Fridge",
        "features": ["cooling", "smart control", "energy saving"],
        "power_usage": "medium",
        "connectivity": "WiFi"
    }
    domain = [
        {
            "name": "Smart Washing Machine",
            "features": ["smart control", "energy saving", "load detection"],
            "power_usage": "high",
            "connectivity": "WiFi"
        },
        {
            "name": "Smart Oven",
            "features": ["heating", "WiFi", "touch control"],
            "power_usage": "high",
            "connectivity": "WiFi"
        },
        {
            "name": "Smart Thermostat",
            "features": ["cooling", "smart control", "energy saving"],
            "power_usage": "low",
            "connectivity": "WiFi"
        }
    ]

    patterns = base.recognize_patterns(pattern_type=type, data_series=data_series)
    conclusion = base.draw_logical_conclusion(premises=premise, logic_type=type2)
    analogies = base.find_analogies(target=target, domain=domain)

    printer.pretty("patterns", patterns, "success" if patterns else "error")
    printer.pretty("conclusion", conclusion, "success" if conclusion else "error")
    printer.pretty("analogies", analogies, "success" if analogies else "error")
    print("\n=== Successfully Ran Base Reasoning Types ===\n")
