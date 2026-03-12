
from typing import Any, Dict, List, Optional, Tuple

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from src.agents.reasoning.types.base_reasoning import BaseReasoning
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reasoning Abduction")
printer = PrettyPrinter

class ReasoningAbduction(BaseReasoning):
    """
    Implements abductive reasoning: Forming hypotheses to explain observations
    Process:
    1. Collect and validate evidence
    2. Generate plausible hypotheses
    3. Evaluate hypotheses using evidence
    4. Select the best explanation
    """
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.abduction_config = get_config_section("reasoning_abduction")
        self.min_confidence = self.abduction_config.get('min_confidence')
        self.max_hypotheses = self.abduction_config.get('max_hypotheses')
        self.explanatory_threshold = self.abduction_config.get('explanatory_threshold')

    def perform_reasoning(self, observations: any, context: dict = None) -> Dict[str, Any]:
        """
        Perform abductive reasoning on observations
        Args:
            observations: Data to explain (single item or list)
            context: Additional context for reasoning
        Returns:
            Best explanation with metadata
        """
        self.log_step("Starting abductive reasoning process")
        
        # Prepare input data
        obs_list = self._prepare_observations(observations)
        context = context or {}
        
        # Step 1: Collect and validate evidence
        evidence = self._collect_evidence(obs_list, context)
        
        # Step 2: Generate hypotheses
        hypotheses = self.generate_hypotheses(obs_list)
        
        # Step 3: Evaluate hypotheses
        evaluated_hypotheses = self.evaluate_hypotheses(hypotheses, evidence, context)
        
        # Step 4: Select best explanation
        best_hypothesis = self.select_best_hypothesis(evaluated_hypotheses)
        
        # Format and return results
        return self._format_results(
            best_hypothesis, 
            evaluated_hypotheses, 
            evidence
        )

    def _prepare_observations(self, observations: any) -> List[Any]:
        """Ensure observations are in consistent list format"""
        if not isinstance(observations, list):
            return [observations]
        return observations

    def _collect_evidence(self, observations: List[Any], context: Dict) -> List[Dict]:
        """Collect and validate evidence from multiple sources"""
        evidence_sources = context.get("evidence_sources", [])
        all_evidence = evidence_sources + [
            {"content": obs, "type": "observation", "confidence": 1.0}
            for obs in observations
        ]
        return self.collect_evidence(all_evidence)
    
    def generate_hypotheses(self, obs_list):
        """
        Generate plausible explanatory hypotheses for observations
        Args:
            observations: List of observed phenomena
        Returns:
            List of potential explanatory hypotheses
        """
        hypotheses = []
        
        # Generate simple observation-based hypotheses
        for obs in obs_list:
            # Use string representation for non-string observations
            obs_str = str(obs)
            hypotheses.append(f"Possible cause for {obs_str}")
            hypotheses.append(f"Potential explanation for {obs_str}")
        
        # Generate unified hypotheses for multiple observations
        if len(obs_list) > 1:
            combined_obs = " and ".join(str(obs) for obs in obs_list[:3])
            if len(obs_list) > 3:
                combined_obs += f" and {len(obs_list)-3} more"
            
            hypotheses.append(f"Common cause for {combined_obs}")
            hypotheses.append(f"Systemic issue explaining {combined_obs}")
        
        # Generate contextual hypotheses using domain knowledge
        domain_hypotheses = self._generate_domain_hypotheses(obs_list)
        hypotheses.extend(domain_hypotheses)
        
        # Limit to maximum hypotheses
        hypotheses = hypotheses[:self.max_hypotheses]
        self.log_step(f"Generated {len(hypotheses)} hypotheses for observations")
        return hypotheses
    
    def _generate_domain_hypotheses(self, observations: List[Any]) -> List[str]:
        """Generate domain-specific hypotheses based on observation patterns"""
        domain_hypotheses = []
        
        # Analyze observation types and patterns
        types = set()
        keywords = set()
        
        for obs in observations:
            if isinstance(obs, dict):
                types.update(obs.keys())
                for value in obs.values():
                    if isinstance(value, str):
                        keywords.update(value.split())
            elif isinstance(obs, str):
                keywords.update(obs.split())
        
        # Generate type-based hypotheses
        if types:
            type_list = ", ".join(sorted(types)[:3])
            domain_hypotheses.append(f"Configuration issue with {type_list} systems")
        
        # Generate keyword-based hypotheses
        if keywords:
            keyword_list = ", ".join(sorted(keywords)[:5])
            domain_hypotheses.append(f"Process failure involving {keyword_list}")
        
        # Temporal pattern hypotheses
        patterns = self.recognize_patterns(
            list(range(len(observations))),  # Using index as proxy for time
            pattern_type="temporal"
        )
        
        for pattern in patterns:
            if pattern["type"] == "increasing_sequence":
                domain_hypotheses.append("Escalating systemic failure")
            elif pattern["type"] == "periodic":
                domain_hypotheses.append("Cyclical process malfunction")
        
        return domain_hypotheses

    def evaluate_hypotheses(self, hypotheses: List[str], evidence: List[Dict], context: Dict) -> List[Dict[str, Any]]:
        """
        Evaluate hypotheses based on evidence and context
        Returns list of hypotheses with evaluation metrics
        """
        evaluated = []
        
        for hypothesis in hypotheses[:self.max_hypotheses]:
            self.log_step(f"Evaluating hypothesis: {hypothesis}")
            
            # Test against conditions
            test_conditions = {
                "evidence": evidence,
                "context": context
            }
            is_supported, confidence = self.test_hypothesis(hypothesis, test_conditions)
            
            # Calculate explanatory power
            explanatory_power = self.calculate_explanatory_power(hypothesis, evidence)
            
            # Calculate probability
            probability = self.evaluate_probability(hypothesis, evidence)
            
            evaluated.append({
                "hypothesis": hypothesis,
                "is_supported": is_supported,
                "confidence": confidence,
                "explanatory_power": explanatory_power,
                "probability": probability,
                "evidence_coverage": self.calculate_evidence_coverage(hypothesis, evidence)
            })
        
        return evaluated
    
    def test_hypothesis(self, hypothesis: str, test_conditions: Dict) -> Tuple[bool, float]:
        """
        Test hypothesis against evidence and context
        Args:
            hypothesis: Hypothesis to test
            test_conditions: Dictionary containing:
                - evidence: List of evidence items
                - context: Additional context information
        Returns:
            Tuple (is_supported, confidence_score)
        """
        evidence = test_conditions.get("evidence", [])
        context = test_conditions.get("context", {})
        
        self.log_step(f"Testing hypothesis: {hypothesis}")
        
        # Check for supporting evidence
        supporting, total = self._evaluate_evidence_support(hypothesis, evidence)
        support_ratio = supporting / total if total > 0 else 0
        
        # Check for contradictions
        contradictions = self._check_contradictions(hypothesis, evidence, context)
        contradiction_score = 1.0 - (len(contradictions) / total if total > 0 else 1.0)
        
        # Contextual consistency check
        context_consistency = self._check_context_consistency(hypothesis, context)
        
        # Calculate composite confidence
        confidence = min(1.0, 0.5 * support_ratio + 0.3 * contradiction_score + 0.2 * context_consistency)
        
        # Determine support status
        is_supported = confidence >= self.min_confidence and support_ratio > 0
        
        self.log_step(f"Hypothesis '{hypothesis[:30]}...' supported: {is_supported} (confidence: {confidence:.2f})")
        return is_supported, confidence
    
    def _evaluate_evidence_support(self, hypothesis: str, evidence: List[Dict]) -> Tuple[int, int]:
        """Count supporting evidence items for hypothesis"""
        supporting = 0
        total = 0
        hypothesis_keywords = set(hypothesis.lower().split())
        
        for item in evidence:
            content = str(item.get("content", "")).lower()
            item_keywords = set(content.split())
            
            # Calculate keyword overlap
            overlap = len(hypothesis_keywords & item_keywords)
            
            # Consider item supportive if >30% keywords match
            if overlap / len(hypothesis_keywords) > 0.3:
                supporting += 1
            total += 1
        
        return supporting, total
    
    def _check_contradictions(self, hypothesis: str, evidence: List[Dict], context: Dict) -> List[str]:
        """Identify evidence contradicting the hypothesis"""
        contradictions = []
        hypothesis_keywords = set(hypothesis.lower().split())
        
        for item in evidence:
            content = str(item.get("content", "")).lower()
            
            # Look for negation patterns
            if " not " in content or "n't " in content:
                item_keywords = set(content.split())
                overlap = len(hypothesis_keywords & item_keywords)
                
                # Consider contradictory if >50% keywords match with negation
                if overlap / len(hypothesis_keywords) > 0.5:
                    contradictions.append(content[:50] + "...")
        
        # Check context for known constraints
        constraints = context.get("constraints", [])
        for constraint in constraints:
            if " contradicts " in constraint and hypothesis in constraint:
                contradictions.append(constraint)
        
        return contradictions
    
    def _check_context_consistency(self, hypothesis: str, context: Dict) -> float:
        """Check consistency with contextual knowledge"""
        # Check against known facts in context
        known_facts = context.get("known_facts", [])
        matches = 0
        
        for fact in known_facts:
            if fact.lower() in hypothesis.lower():
                matches += 1
        
        # Calculate consistency score
        if known_facts:
            return matches / len(known_facts)
        return 1.0  # No facts = no inconsistency

    def calculate_explanatory_power(self, hypothesis: str, evidence: List[Dict]) -> float:
        """
        Measure how well the hypothesis explains the evidence
        Placeholder implementation - would be domain-specific
        """
        # Simple heuristic: Count of evidence items containing hypothesis keywords
        keywords = set(hypothesis.lower().split())
        explained_count = 0
        
        for item in evidence:
            content = str(item.get("content", "")).lower()
            if any(keyword in content for keyword in keywords):
                explained_count += 1
        
        coverage = explained_count / len(evidence) if evidence else 0.0
        return min(coverage, 1.0)  # Cap at 1.0
    
    def evaluate_probability(self, hypothesis: str, evidence: List[Dict]) -> float:
        """
        Evaluate Bayesian probability of hypothesis given evidence
        P(H|E) = P(E|H) * P(H) / P(E)
        """
        # Prior probability P(H) - base probability before evidence
        prior = self._calculate_prior_probability(hypothesis)
        
        # Likelihood P(E|H) - probability of evidence if hypothesis is true
        likelihood = self._calculate_likelihood(hypothesis, evidence)
        
        # Evidence probability P(E) - total probability of evidence
        evidence_prob = self._calculate_evidence_probability(evidence)
        
        # Avoid division by zero
        if evidence_prob <= 0:
            return 0.0
        
        # Bayes' theorem
        posterior = (likelihood * prior) / evidence_prob
        
        # Cap at 1.0
        return min(1.0, posterior)
    
    def _calculate_prior_probability(self, hypothesis: str) -> float:
        """Estimate prior probability based on hypothesis complexity"""
        # Simpler hypotheses have higher prior probability
        words = len(hypothesis.split())
        complexity = min(1.0, words / 20)  # Normalize to 0-1 range
        return 0.7 - (complexity * 0.3)  # Base 0.7 minus complexity penalty
    
    def _calculate_likelihood(self, hypothesis: str, evidence: List[Dict]) -> float:
        """Calculate P(E|H) - probability of evidence given hypothesis"""
        if not evidence:
            return 1.0  # No evidence doesn't contradict
        
        explained_count = 0
        hypothesis_keywords = set(hypothesis.lower().split())
        
        for item in evidence:
            content = str(item.get("content", "")).lower()
            item_keywords = set(content.split())
            overlap = len(hypothesis_keywords & item_keywords)
            
            # Consider evidence explained if >40% keywords match
            if overlap / len(item_keywords) > 0.4:
                explained_count += 1
        
        return explained_count / len(evidence)
    
    def _calculate_evidence_probability(self, evidence: List[Dict]) -> float:
        """Estimate total probability of evidence"""
        if not evidence:
            return 1.0
        
        total_confidence = sum(item.get("confidence", 0.5) for item in evidence)
        avg_confidence = total_confidence / len(evidence)
        
        # Adjust based on evidence diversity
        unique_sources = len(set(item.get("source", "unknown") for item in evidence))
        diversity_factor = min(1.0, unique_sources / 5)  # More sources = higher reliability
        
        return avg_confidence * diversity_factor

    def calculate_evidence_coverage(self, hypothesis: str, evidence: List[Dict]) -> List[Dict]:
        """Identify which evidence items are explained by the hypothesis"""
        coverage = []
        keywords = set(hypothesis.lower().split())
        
        for item in evidence:
            content = str(item.get("content", "")).lower()
            explained = any(keyword in content for keyword in keywords)
            coverage.append({
                "evidence": item,
                "explained": explained,
                "confidence": item.get("confidence", 0.0)
            })
        
        return coverage

    def select_best_hypothesis(self, evaluated_hypotheses: List[Dict]) -> Optional[Dict]:
        """
        Select the best hypothesis based on evaluation metrics
        """
        if not evaluated_hypotheses:
            return None
        
        # Filter supported hypotheses with minimum confidence
        valid_hypotheses = [
            h for h in evaluated_hypotheses
            if h["is_supported"] and h["confidence"] >= self.min_confidence
        ]
        
        if not valid_hypotheses:
            self.log_step("No valid hypotheses meet confidence threshold", "warning")
            return None
        
        # Select hypothesis with highest composite score
        best_hypothesis = max(
            valid_hypotheses,
            key=lambda h: (
                h["confidence"] * 0.4 +
                h["explanatory_power"] * 0.3 +
                h["probability"] * 0.3
            )
        )
        
        self.log_step(f"Selected best hypothesis: {best_hypothesis['hypothesis']}")
        return best_hypothesis

    def _format_results(
        self, 
        best_hypothesis: Optional[Dict], 
        all_hypotheses: List[Dict], 
        evidence: List[Dict]
    ) -> Dict[str, Any]:
        """Format final results with metadata"""
        return {
            "best_explanation": best_hypothesis,
            "alternative_hypotheses": [h for h in all_hypotheses if h != best_hypothesis],
            "evidence_used": evidence,
            "metrics": {
                "total_hypotheses": len(all_hypotheses),
                "valid_hypotheses": len([h for h in all_hypotheses if h["is_supported"]]),
                "evidence_count": len(evidence),
                "success": best_hypothesis is not None
            },
            "reasoning_type": "abduction"
        }

if __name__ == "__main__":
    print("\n=== Running Reasoning Abduction ===\n")
    printer.status("TEST", "Starting Reasoning Abduction tests", "info")

    abduction = ReasoningAbduction()

    result = abduction.perform_reasoning(
        observations="The grass is wet",
        context={
            "evidence_sources": [
                {"content": "It rained last night", "confidence": 0.9},
                {"content": "Sprinklers were on", "confidence": 0.7}
            ]
        }
    )
    
    # Complex observations
    result = abduction.perform_reasoning(
        observations=[
            "Temperature dropped 10°C",
            "Wind speed increased",
            "Barometric pressure falling"
        ],
        context={"location": "Seattle"}
    )
    
    printer.pretty("Abduction Result", result)

    print("\n=== Successfully Ran Reasoning Abduction ===\n")
