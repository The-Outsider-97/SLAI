
from typing import Dict, List, Type, Any, Optional, Set

import numpy as np

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from src.agents.reasoning.types.base_reasoning import BaseReasoning
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reasoning Inductive")
printer = PrettyPrinter

class ReasoningInductive(BaseReasoning):
    """
    Implements inductive reasoning: Deriving general principles from specific observations
    Process:
    1. Collect and validate observations
    2. Identify patterns and regularities
    3. Formulate generalized theories
    4. Validate theories through extrapolation
    5. Apply theories to make predictions
    """
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.inductive_config = get_config_section("reasoning_inductive")
        self.min_observations = self.inductive_config.get('min_observations')
        self.confidence_threshold = self.inductive_config.get('confidence_threshold')
        self.extrapolation_limit = self.inductive_config.get('extrapolation_limit')
        self.trend_analysis_weight = self.inductive_config.get('trend_analysis_weight')
        self.pattern_analysis_weight = self.inductive_config.get('pattern_analysis_weight')

    def perform_reasoning(self, observations: List[Any], context: dict = None) -> Dict[str, Any]:
        """
        Perform inductive reasoning on observations
        Args:
            observations: Specific instances to generalize from
            context: Additional context for reasoning
        Returns:
            General theory with validation and predictions
        """
        self.log_step("Starting inductive reasoning process")
        context = context or {}
        
        # Step 1: Collect and validate observations
        validated_obs = self._validate_observations(observations, context)
        
        if len(validated_obs) < self.min_observations:
            self.log_step(f"Insufficient observations ({len(validated_obs)}/{self.min_observations})", "warning")
            return self._format_results({}, validated_obs, context)
        
        # Step 2: Identify patterns
        patterns = self._identify_patterns(validated_obs, context)
        
        # Step 3: Formulate generalized theory
        theory = self._formulate_theory(patterns, validated_obs, context)
        
        # Step 4: Validate theory
        validation_result = self._validate_theory(theory, validated_obs, context)
        
        # Step 5: Make predictions
        predictions = self._make_predictions(theory, validated_obs, context)
        
        # Format and return results
        return self._format_results(
            theory, 
            validated_obs, 
            patterns, 
            validation_result, 
            predictions,
            context
        )

    def _validate_observations(self, observations: List[Any], context: Dict) -> List[Dict]:
        """
        Validate and standardize observations
        Returns list of observations with metadata
        """
        validated = []
        
        for i, obs in enumerate(observations):
            # Standardize different observation formats
            if isinstance(obs, dict):
                standardized = {
                    "id": f"obs_{i}",
                    "content": obs.get("content", obs),
                    "timestamp": obs.get("timestamp", i),
                    "confidence": obs.get("confidence", 1.0),
                    "source": obs.get("source", "unknown")
                }
            else:
                standardized = {
                    "id": f"obs_{i}",
                    "content": obs,
                    "timestamp": i,
                    "confidence": 1.0,
                    "source": "direct"
                }
            
            # Apply context filters
            if "observation_filter" in context:
                if not context["observation_filter"](standardized):
                    continue
            
            # Confidence threshold check
            if standardized["confidence"] >= self.confidence_threshold:
                validated.append(standardized)
        
        return validated

    def _identify_patterns(self, observations: List[Dict], context: Dict) -> List[Dict]:
        """
        Identify patterns across multiple dimensions
        Returns combined patterns from different analysis methods
        """
        patterns = []
        
        # Temporal patterns
        temporal_data = [obs["timestamp"] for obs in observations]
        temporal_patterns = self.recognize_patterns(
            temporal_data, 
            pattern_type="temporal"
        )
        patterns.extend(temporal_patterns)
        
        # Content-based patterns
        content_data = [str(obs["content"]) for obs in observations]
        content_patterns = self.recognize_patterns(
            content_data, 
            pattern_type="behavioral"
        )
        patterns.extend(content_patterns)
        
        # Attribute patterns (if available)
        if all("attributes" in obs for obs in observations):
            attribute_patterns = self._identify_attribute_patterns(observations)
            patterns.extend(attribute_patterns)
        
        # Contextual patterns
        if "pattern_generators" in context:
            for generator in context["pattern_generators"]:
                patterns.extend(generator(observations))
        
        # Filter by confidence
        return [p for p in patterns if p.get("confidence", 0) >= self.confidence_threshold]

    def _identify_attribute_patterns(self, observations: List[Dict]) -> List[Dict]:
        """Identify patterns across observation attributes"""
        patterns = []
        attribute_keys = set()
        
        # Collect all attribute keys
        for obs in observations:
            attribute_keys.update(obs["attributes"].keys())
        
        # Analyze patterns per attribute
        for key in attribute_keys:
            values = [obs["attributes"][key] for obs in observations if key in obs["attributes"]]
            
            if all(isinstance(v, (int, float)) for v in values):
                # Numeric attribute pattern
                num_patterns = self.recognize_patterns(values, pattern_type="temporal")
                for p in num_patterns:
                    p["attribute"] = key
                patterns.extend(num_patterns)
            else:
                # Categorical attribute pattern
                value_counts = {}
                for v in values:
                    value_counts[v] = value_counts.get(v, 0) + 1
                
                total = len(values)
                for value, count in value_counts.items():
                    frequency = count / total
                    if frequency > 0.7:  # Dominant value pattern
                        patterns.append({
                            "type": "dominant_value",
                            "attribute": key,
                            "value": value,
                            "frequency": frequency,
                            "confidence": min(1.0, frequency * 1.2)
                        })
        
        return patterns

    def _formulate_theory(self, patterns: List[Dict], observations: List[Dict], context: Dict) -> Dict[str, Any]:
        """
        Formulate a generalized theory from patterns
        Returns theory with scope and confidence
        """
        # Combine patterns into a cohesive theory
        theory_statements = []
        confidence_scores = []
        
        for pattern in patterns:
            statement = self._pattern_to_statement(pattern, observations, context)
            if statement:
                theory_statements.append(statement["statement"])
                confidence_scores.append(statement["confidence"])
        
        if not theory_statements:
            return {
                "theory": "No generalizable patterns found",
                "confidence": 0.0,
                "scope": "N/A"
            }
        
        # Calculate overall confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Determine theory scope based on observation diversity
        sources = set(obs["source"] for obs in observations)
        scope = "narrow"
        if len(sources) > 3:
            scope = "broad"
        elif len(sources) > 1:
            scope = "moderate"
        
        return {
            "theory": " AND ".join(theory_statements),
            "confidence": avg_confidence,
            "scope": scope,
            "supporting_patterns": patterns
        }

    def _pattern_to_statement(self, pattern: Dict, observations: List[Dict], context: Dict) -> Optional[Dict]:
        """Convert pattern to a generalized statement"""
        pattern_type = pattern["type"]
        
        # Temporal patterns
        if pattern_type == "increasing_sequence":
            first = observations[0]["content"]
            last = observations[-1]["content"]
            return {
                "statement": f"Values increase over time from {first} to {last}",
                "confidence": pattern.get("confidence", 0.8)
            }
        elif pattern_type == "periodic":
            return {
                "statement": f"Values follow a periodic cycle with period {pattern.get('period', 'unknown')}",
                "confidence": pattern.get("confidence", 0.75)
            }
        
        # Content patterns
        elif pattern_type == "dominant_value" and "attribute" in pattern:
            return {
                "statement": f"Most observations ({pattern['frequency']*100:.1f}%) have {pattern['attribute']} = {pattern['value']}",
                "confidence": pattern.get("confidence", 0.7)
            }
        
        # Attribute patterns
        elif pattern_type == "clusters":
            return {
                "statement": f"Observations form {pattern['count']} distinct clusters",
                "confidence": pattern.get("confidence", 0.7)
            }
        
        # Contextual pattern handling
        if "pattern_to_statement" in context:
            return context["pattern_to_statement"](pattern, observations)
        
        return None

    def _validate_theory(self, theory: Dict, observations: List[Dict], context: Dict) -> Dict[str, Any]:
        """
        Validate theory through multiple methods
        Returns validation metrics and status
        """
        if theory["confidence"] < self.confidence_threshold:
            return {
                "is_valid": False,
                "confidence": theory["confidence"],
                "tests": [],
                "reasons": ["Theory confidence below threshold"]
            }
        
        validation_tests = []
        
        # Test 1: Consistency with existing knowledge
        knowledge_consistency = self._check_knowledge_consistency(theory, context)
        validation_tests.append({
            "test": "knowledge_consistency",
            "result": knowledge_consistency["consistent"],
            "confidence": knowledge_consistency["confidence"]
        })
        
        # Test 2: Predictive accuracy on holdout data
        if "holdout_data" in context:
            predictive_accuracy = self._test_predictive_accuracy(theory, context["holdout_data"], context)
            validation_tests.append({
                "test": "predictive_accuracy",
                "result": predictive_accuracy["accuracy"] > 0.7,
                "confidence": predictive_accuracy["confidence"],
                "accuracy": predictive_accuracy["accuracy"]
            })
        
        # Test 3: Counterexample check
        counterexample_check = self._check_counterexamples(theory, observations, context)
        validation_tests.append({
            "test": "counterexample_check",
            "result": not counterexample_check["found"],
            "confidence": counterexample_check["confidence"]
        })
        
        # Calculate composite validation confidence
        test_confidences = [t["confidence"] for t in validation_tests]
        validation_confidence = sum(test_confidences) / len(test_confidences)
        
        # Determine overall validity
        is_valid = (validation_confidence >= self.confidence_threshold and 
                   all(t["result"] for t in validation_tests if "result" in t))
        
        return {
            "is_valid": is_valid,
            "confidence": validation_confidence,
            "tests": validation_tests,
            "overall_confidence": min(theory["confidence"], validation_confidence)
        }

    def _check_knowledge_consistency(self, theory: Dict, context: Dict) -> Dict[str, Any]:
        """Check consistency with existing knowledge base"""
        knowledge_base = context.get("knowledge_base", {})
        contradictions = []
        
        # Check for direct contradictions
        if "contradictions" in knowledge_base:
            for known_theory in knowledge_base["contradictions"]:
                if known_theory in theory["theory"]:
                    contradictions.append(known_theory)
        
        # Check supporting knowledge
        support_score = 0
        if "supporting_theories" in knowledge_base:
            for known_theory in knowledge_base["supporting_theories"]:
                if known_theory in theory["theory"]:
                    support_score += 0.3
        
        return {
            "consistent": len(contradictions) == 0,
            "confidence": min(1.0, 0.7 + support_score - (len(contradictions) * 0.3)),
            "contradictions": contradictions,
            "support_score": support_score
        }

    def _test_predictive_accuracy(self, theory: Dict, holdout_data: List[Any], context: Dict) -> Dict[str, Any]:
        """Test theory's predictive power on unseen data"""
        predictions = self._make_predictions(theory, holdout_data, context)
        correct = 0
        
        for prediction in predictions:
            if prediction["validated"] and prediction["confidence"] > self.confidence_threshold:
                correct += 1
        
        accuracy = correct / len(predictions) if predictions else 0.0
        confidence = min(1.0, accuracy * 1.2)  # Scale accuracy to confidence
        
        return {
            "accuracy": accuracy,
            "confidence": confidence,
            "correct_predictions": correct,
            "total_predictions": len(predictions)
        }

    def _check_counterexamples(
        self, 
        theory: Dict, 
        observations: List[Dict], 
        context: Dict
    ) -> Dict[str, Any]:
        """Search for counterexamples in observations"""
        counterexamples = []
        
        for obs in observations:
            if not self._observation_conforms(theory, obs, context):
                counterexamples.append(obs["id"])
        
        return {
            "found": len(counterexamples) > 0,
            "count": len(counterexamples),
            "confidence": max(0, 1.0 - (len(counterexamples) / len(observations))),
            "examples": counterexamples
        }

    def _observation_conforms(
        self, 
        theory: Dict, 
        observation: Dict, 
        context: Dict
    ) -> bool:
        """Check if observation conforms to theory"""
        # Placeholder implementation - would use domain-specific logic
        # In real system, this might use pattern matching or rule evaluation
        return "exception" not in str(observation["content"]).lower()

    def _make_predictions(self, theory: Dict, observations: List[Dict], context: Dict) -> List[Dict]:
        """
        Make predictions based on the theory
        Returns list of predictions with confidence
        """
        predictions = []
        
        # Identify trends for extrapolation
        trends = self._identify_trends(observations)
        
        # Make predictions based on theory and trends
        for trend in trends:
            prediction = self._extrapolate_trend(trend, context)
            if prediction:
                predictions.append(prediction)
        
        # Make categorical predictions
        categorical_preds = self._make_categorical_predictions(theory, observations, context)
        predictions.extend(categorical_preds)
        
        return predictions

    def _identify_trends(self, observations: List[Dict]) -> List[Dict]:
        """Identify trends in numerical observations"""
        trends = []
        
        # Try to extract numerical values
        numeric_values = []
        for obs in observations:
            if isinstance(obs["content"], (int, float)):
                numeric_values.append(obs["content"])
            elif isinstance(obs["content"], dict):
                # Find first numeric value in dict
                for v in obs["content"].values():
                    if isinstance(v, (int, float)):
                        numeric_values.append(v)
                        break
        
        if len(numeric_values) > 1:
            # Linear trend
            x = np.arange(len(numeric_values))
            slope, intercept = np.polyfit(x, numeric_values, 1)
            trends.append({
                "type": "linear",
                "slope": slope,
                "intercept": intercept,
                "confidence": 0.8
            })
            
            # Exponential trend
            try:
                log_values = np.log(numeric_values)
                exp_slope, exp_intercept = np.polyfit(x, log_values, 1)
                trends.append({
                    "type": "exponential",
                    "growth_rate": exp_slope,
                    "confidence": 0.7
                })
            except:
                pass
        
        return trends

    def _extrapolate_trend(self, trend: Dict, context: Dict) -> Dict:
        """Extrapolate trend to make predictions"""
        extrapolation_factor = context.get("extrapolation_factor", self.extrapolation_limit)
        
        if trend["type"] == "linear":
            future_value = trend["slope"] * extrapolation_factor + trend["intercept"]
            return {
                "type": "numerical",
                "prediction": f"Value will reach {future_value:.2f}",
                "confidence": trend["confidence"] * 0.9,
                "extrapolation_factor": extrapolation_factor
            }
        elif trend["type"] == "exponential":
            future_value = np.exp(trend["growth_rate"] * extrapolation_factor)
            return {
                "type": "numerical",
                "prediction": f"Value will reach {future_value:.2f}",
                "confidence": trend["confidence"] * 0.8,
                "extrapolation_factor": extrapolation_factor
            }
        return None

    def _make_categorical_predictions(self, theory: Dict, observations: List[Dict], context: Dict) -> List[Dict]:
        """Make categorical predictions based on theory patterns"""
        predictions = []
        
        # Look for dominant value patterns
        for pattern in theory.get("supporting_patterns", []):
            if pattern["type"] == "dominant_value":
                predictions.append({
                    "type": "categorical",
                    "prediction": f"Next observation will have {pattern['attribute']} = {pattern['value']}",
                    "confidence": pattern["confidence"] * 0.85,
                    "pattern": pattern
                })
        
        return predictions

    def _format_results(self, theory: Dict, observations: List[Dict], patterns: List[Dict], 
                        validation: Dict, predictions: List[Dict], context: Dict) -> Dict[str, Any]:
        """Format final results with metadata"""
        return {
            "theory": theory,
            "validation": validation,
            "predictions": predictions,
            "supporting_data": {
                "observations_used": observations,
                "patterns_identified": patterns,
                "context": context
            },
            "metrics": {
                "observations_count": len(observations),
                "patterns_count": len(patterns),
                "theory_confidence": theory.get("confidence", 0),
                "validation_confidence": validation.get("confidence", 0),
                "predictions_count": len(predictions),
                "success": validation.get("is_valid", False)
            },
            "reasoning_type": "inductive"
        }

if __name__ == "__main__":
    print("\n=== Running Reasoning Inductive ===\n")
    printer.status("TEST", "Starting Inductive Reasoning tests", "info")

    inductive = ReasoningInductive()

    # Create test observations
    observations = [
        {"content": {"temp": 22, "weather": "sunny"}, "source": "sensor_1"},
        {"content": {"temp": 24, "weather": "sunny"}, "source": "sensor_1"},
        {"content": {"temp": 26, "weather": "sunny"}, "source": "sensor_2"},
        {"content": {"temp": 28, "weather": "sunny"}, "source": "sensor_2"},
        {"content": {"temp": 30, "weather": "sunny"}, "source": "sensor_3"},
        {"content": {"temp": 28, "weather": "cloudy"}, "source": "sensor_3", "confidence": 0.9},
        {"content": {"temp": 26, "weather": "rainy"}, "source": "sensor_4", "confidence": 0.8}
    ]

    context = {
        "knowledge_base": {
            "contradictions": [
                "temperature decreases in summer"
            ],
            "supporting_theories": [
                "temperature increases during summer"
            ]
        },
        "extrapolation_factor": 2.0
    }

    result = inductive.perform_reasoning(
        observations=observations,
        context=context
    )
    
    printer.pretty("Inductive Reasoning Result", result)
    print("\n=== Successfully Ran Reasoning Inductive ===\n")
