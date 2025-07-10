
from typing import Any, Dict, List

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from src.agents.reasoning.types.base_reasoning import BaseReasoning
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reasoning Cause And Effect")
printer = PrettyPrinter

class ReasoningCauseAndEffect(BaseReasoning):
    """
    Implements cause-and-effect reasoning: Identifying causal relationships between events
    Process:
    1. Extract events and conditions from input
    2. Identify potential causal relationships
    3. Validate relationships through evidence and counterfactual analysis
    4. Model causal networks and predict outcomes
    """
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.cause_effect_config = get_config_section("reasoning_cause_effect")
        self.min_confidence = self.cause_effect_config.get('min_confidence')
        self.max_chain_length = self.cause_effect_config.get('max_chain_length')
        self.enable_counterfactual = self.cause_effect_config.get('enable_counterfactual')
        self.temporal_weight = self.cause_effect_config.get('temporal_weight')
        self.correlation_weight = self.cause_effect_config.get('correlation_weight')
        self.network_mode = self.cause_effect_config.get('network_mode')

    def perform_reasoning(
        self, 
        events: List[Any], 
        conditions: Dict[str, Any] = None, 
        context: dict = None
    ) -> Dict[str, Any]:
        """
        Perform cause-and-effect reasoning on events and conditions
        Args:
            events: Sequence of events to analyze
            conditions: Pre-existing conditions that might influence events
            context: Additional context for reasoning
        Returns:
            Causal analysis with identified relationships and predictions
        """
        self.log_step("Starting cause-and-effect reasoning")
        context = context or {}
        conditions = conditions or {}
        
        # Step 1: Extract and normalize events
        normalized_events = self._normalize_events(events, context)
        
        # Step 2: Identify potential causal relationships
        relationships = self.identify_causal_relationships(normalized_events)
        
        # Step 3: Validate relationships
        validated_relationships = []
        for rel in relationships:
            validated = self.validate_relationship(rel, conditions, context)
            if validated["confidence"] >= self.min_confidence:
                validated_relationships.append(validated)
        
        # Step 4: Build causal model
        causal_model = self.build_causal_model(validated_relationships, conditions)
        
        # Step 5: Predict outcomes
        predictions = self.predict_outcomes(causal_model, context)
        
        # Format and return results
        return self._format_results(
            validated_relationships, 
            causal_model, 
            predictions, 
            context
        )

    def _normalize_events(self, events: List[Any], context: Dict) -> List[Dict]:
        """Convert events to standardized format with temporal information"""
        normalized = []
        time_index = 0
        
        for event in events:
            # Handle different event representations
            if isinstance(event, dict):
                normalized_event = {
                    "id": event.get("id", f"event_{time_index}"),
                    "description": event.get("description", str(event)),
                    "timestamp": event.get("timestamp", time_index),
                    "attributes": event.get("attributes", {})
                }
            elif isinstance(event, str):
                normalized_event = {
                    "id": f"event_{time_index}",
                    "description": event,
                    "timestamp": time_index,
                    "attributes": {}
                }
            else:
                normalized_event = {
                    "id": f"event_{time_index}",
                    "description": str(event),
                    "timestamp": time_index,
                    "attributes": {}
                }
            
            # Apply contextual transformations
            if "event_normalizers" in context:
                for normalizer in context["event_normalizers"]:
                    normalized_event = normalizer(normalized_event)
            
            normalized.append(normalized_event)
            time_index += 1
        
        # Sort by timestamp
        return sorted(normalized, key=lambda e: e["timestamp"])

    def identify_causal_relationships(self, events: List[Dict]) -> List[Dict]:
        """
        Identify potential cause-effect relationships between events
        Enhanced with temporal analysis and pattern recognition
        """
        relationships = []
        
        # Basic sequential relationships
        for i in range(len(events) - 1):
            cause = events[i]
            effect = events[i + 1]
            relationships.append({
                "cause": cause,
                "effect": effect,
                "type": "sequential",
                "confidence": 0.7  # Base confidence for adjacent events
            })
        
        # Cross-event relationships (beyond immediate sequence)
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                cause = events[i]
                effect = events[j]
                
                # Check for matching patterns
                patterns = self.recognize_patterns(
                    [cause["description"], effect["description"]], 
                    pattern_type="behavioral"
                )
                
                if any(p["confidence"] > 0.6 for p in patterns):
                    relationships.append({
                        "cause": cause,
                        "effect": effect,
                        "type": "pattern_based",
                        "confidence": max(p["confidence"] for p in patterns)
                    })
        
        # Attribute-based relationships
        attribute_relationships = self._identify_attribute_relationships(events)
        relationships.extend(attribute_relationships)
        
        self.log_step(f"Identified {len(relationships)} potential causal relationships")
        return relationships

    def _identify_attribute_relationships(self, events: List[Dict]) -> List[Dict]:
        """Identify relationships based on shared attributes"""
        relationships = []
        attribute_map = {}
        
        # Build attribute index
        for event in events:
            for attr, value in event["attributes"].items():
                key = f"{attr}:{value}"
                if key not in attribute_map:
                    attribute_map[key] = []
                attribute_map[key].append(event)
        
        # Create relationships based on shared attributes
        for key, related_events in attribute_map.items():
            if len(related_events) > 1:
                # Create relationships between all pairs
                for i in range(len(related_events)):
                    for j in range(i + 1, len(related_events)):
                        # Earlier event as cause
                        if related_events[i]["timestamp"] < related_events[j]["timestamp"]:
                            relationships.append({
                                "cause": related_events[i],
                                "effect": related_events[j],
                                "type": "attribute_shared",
                                "attribute": key,
                                "confidence": 0.65
                            })
        
        return relationships

    def validate_relationship(self, relationship: Dict, conditions: Dict, context: Dict) -> Dict:
        """
        Validate a causal relationship using evidence and counterfactual analysis
        Returns enhanced relationship with validation metrics
        """
        cause = relationship["cause"]
        effect = relationship["effect"]
        rel_type = relationship["type"]
        
        self.log_step(f"Validating relationship: {cause['id']} -> {effect['id']} ({rel_type})")
        
        # Temporal validation
        temporal_score = self._validate_temporal(cause, effect, context)
        
        # Correlation validation
        correlation_score = self._validate_correlation(cause, effect, conditions, context)
        
        # Counterfactual analysis
        counterfactual_score = 0.0
        if self.enable_counterfactual:
            counterfactual_score = self._counterfactual_analysis(cause, effect, context)
        
        # Calculate composite confidence
        base_confidence = relationship.get("confidence", 0.5)
        validated_confidence = min(1.0, 
            base_confidence * (
                self.temporal_weight * temporal_score +
                self.correlation_weight * correlation_score +
                (1.0 - self.temporal_weight - self.correlation_weight) * counterfactual_score
            )
        )
        
        # Determine validation status
        is_valid = validated_confidence >= self.min_confidence
        
        return {
            **relationship,
            "validated_confidence": validated_confidence,
            "temporal_score": temporal_score,
            "correlation_score": correlation_score,
            "counterfactual_score": counterfactual_score,
            "is_valid": is_valid
        }

    def _validate_temporal(self, cause: Dict, effect: Dict, context: Dict) -> float:
        """Validate temporal relationship (cause must precede effect)"""
        time_gap = effect["timestamp"] - cause["timestamp"]
        
        # Perfect score if cause precedes effect with reasonable gap
        if time_gap > 0:
            max_gap = context.get("max_time_gap", 10)
            return max(0, 1.0 - (time_gap / max_gap))
        return 0.0

    def _validate_correlation(self, cause: Dict, effect: Dict, conditions: Dict, context: Dict) -> float:
        """Calculate correlation strength between cause and effect"""
        # Check for shared attributes
        shared_attributes = set(cause["attributes"].keys()) & set(effect["attributes"].keys())
        attribute_score = min(1.0, len(shared_attributes) * 0.3)
        
        # Check condition influence
        condition_influence = 0.0
        for cond_key, cond_value in conditions.items():
            if cond_key in cause["attributes"] and cond_key in effect["attributes"]:
                if cause["attributes"][cond_key] == cond_value and effect["attributes"][cond_key] == cond_value:
                    condition_influence += 0.2
        
        return min(1.0, attribute_score + condition_influence)

    def _counterfactual_analysis(self, cause: Dict, effect: Dict, context: Dict) -> float:
        """Evaluate relationship through counterfactual reasoning"""
        # Simulate absence of cause
        simulated_effect = self._simulate_absence(cause, effect, context)
        
        # If effect changes significantly without cause, relationship is strong
        if simulated_effect != effect["description"]:
            return 0.8  # High confidence in causality
        return 0.3  # Low confidence if no change

    def _simulate_absence(self, cause: Dict, effect: Dict, context: Dict) -> str:
        """Simulate what would happen without the cause"""
        # Placeholder - would use domain-specific simulation
        # In real implementation, this might use a simulation model
        if "rain" in cause["description"] and "wet" in effect["description"]:
            return "ground is dry"
        return effect["description"]  # Default no change

    def build_causal_model(self, relationships: List[Dict], conditions: Dict) -> Dict[str, Any]:
        """
        Build a causal model from validated relationships
        Supports both chain and network representations
        """
        model = {
            "nodes": [],
            "edges": [],
            "chains": [],
            "conditions": conditions
        }
        
        # Add nodes (events)
        node_ids = set()
        for rel in relationships:
            for role in ["cause", "effect"]:
                node = rel[role]
                if node["id"] not in node_ids:
                    model["nodes"].append(node)
                    node_ids.add(node["id"])
        
        # Add edges (relationships)
        for rel in relationships:
            model["edges"].append({
                "source": rel["cause"]["id"],
                "target": rel["effect"]["id"],
                "type": rel["type"],
                "confidence": rel["validated_confidence"]
            })
        
        # Build causal chains
        model["chains"] = self._find_causal_chains(model["edges"])
        
        # Add Bayesian network structure if enabled
        if self.network_mode == 'bayesian':
            model["network"] = self._build_bayesian_network(model)
        
        return model

    def _find_causal_chains(self, edges: List[Dict]) -> List[List[str]]:
        """Find causal chains up to max_chain_length"""
        chains = []
        # Create adjacency map
        graph = {}
        for edge in edges:
            if edge["source"] not in graph:
                graph[edge["source"]] = []
            graph[edge["source"]].append(edge["target"])
        
        # DFS to find chains
        def dfs(node, path, depth):
            if depth > self.max_chain_length:
                return
            new_path = path + [node]
            chains.append(new_path)
            if node in graph:
                for neighbor in graph[node]:
                    dfs(neighbor, new_path, depth + 1)
        
        for node in graph:
            dfs(node, [], 1)
        
        # Filter trivial chains
        return [chain for chain in chains if len(chain) > 1]

    def _build_bayesian_network(self, model: Dict) -> Dict:
        """Build Bayesian network representation"""
        network = {
            "nodes": {},
            "edges": model["edges"],
            "cpt": {}  # Conditional Probability Tables
        }
        
        # Initialize nodes
        for node in model["nodes"]:
            network["nodes"][node["id"]] = {
                "states": ["occurred", "not_occurred"],
                "description": node["description"]
            }
        
        # Build CPTs (simplified - would use data in real implementation)
        for edge in model["edges"]:
            source = edge["source"]
            target = edge["target"]
            
            # Create CPT for target given source
            cpt_key = f"{source}->{target}"
            network["cpt"][cpt_key] = {
                "source": source,
                "target": target,
                "probabilities": {
                    "occurred": {
                        "occurred": 0.8,  # P(target|source)
                        "not_occurred": 0.2
                    },
                    "not_occurred": {
                        "occurred": 0.3,
                        "not_occurred": 0.7
                    }
                }
            }
        
        return network

    def predict_outcomes(self, model: Dict, context: Dict) -> List[Dict]:
        """
        Predict outcomes based on causal model
        Args:
            model: Causal model built from relationships
            context: Context for prediction (e.g., interventions)
        Returns:
            List of predictions with confidence
        """
        predictions = []
        
        # Predict using causal chains
        for chain in model.get("chains", []):
            if len(chain) > 1:
                cause = chain[0]
                effect = chain[-1]
                predictions.append({
                    "cause": cause,
                    "effect": effect,
                    "chain": chain,
                    "type": "chain",
                    "confidence": 0.7  # Base confidence for chain prediction
                })
        
        # Predict using Bayesian network
        if self.network_mode == 'bayesian' and "network" in model:
            network_preds = self._predict_with_bayesian_network(model["network"], context)
            predictions.extend(network_preds)
        
        # Apply interventions from context
        interventions = context.get("interventions", {})
        for pred in predictions:
            if pred["cause"] in interventions:
                pred["confidence"] *= interventions[pred["cause"]].get("strength", 1.0)
                pred["intervention"] = interventions[pred["cause"]]
        
        return predictions

    def _predict_with_bayesian_network(self, network: Dict, context: Dict) -> List[Dict]:
        """Make predictions using Bayesian network (simplified)"""
        predictions = []
        
        # Find root causes (nodes with no incoming edges)
        root_causes = set(network["nodes"].keys())
        for edge in network["edges"]:
            if edge["target"] in root_causes:
                root_causes.remove(edge["target"])
        
        # Make predictions from root causes
        for cause in root_causes:
            # Find effects reachable from this cause
            for node_id in network["nodes"]:
                if node_id != cause:
                    # Simplified - would use actual inference in real implementation
                    predictions.append({
                        "cause": cause,
                        "effect": node_id,
                        "type": "bayesian",
                        "confidence": 0.75,
                        "path": [cause, node_id]  # Simplified path
                    })
        
        return predictions

    def _format_results(self, relationships: List[Dict], model: Dict, 
                        predictions: List[Dict], context: Dict) -> Dict[str, Any]:
        """Format final results with metadata"""
        valid_relationships = [r for r in relationships if r["is_valid"]]
        return {
            "valid_relationships": valid_relationships,
            "causal_model": model,
            "predictions": predictions,
            "context_used": context,
            "metrics": {
                "total_relationships": len(relationships),
                "valid_relationships": len(valid_relationships),
                "predictions_generated": len(predictions),
                "model_complexity": len(model["nodes"]) if "nodes" in model else 0,
                "success": len(valid_relationships) > 0
            },
            "reasoning_type": "cause_effect"
        }

if __name__ == "__main__":
    print("\n=== Running Reasoning Cause and Effect ===\n")
    printer.status("TEST", "Starting Cause and Effect tests", "info")

    cause_effect = ReasoningCauseAndEffect()

    # Create test events
    events = [
        {"description": "Heavy rainfall", "attributes": {"intensity": "high"}},
        {"description": "River water level rises", "attributes": {"level": "danger"}},
        {"description": "Flood warning issued", "attributes": {"severity": "high"}},
        {"description": "Residents evacuate", "attributes": {"area": "riverside"}}
    ]

    conditions = {
        "soil_saturation": "high",
        "drainage_capacity": "low"
    }

    context = {
        "max_time_gap": 5,
        "interventions": {
            "Heavy rainfall": {"action": "cloud seeding", "strength": 0.8}
        }
    }

    result = cause_effect.perform_reasoning(
        events=events,
        conditions=conditions,
        context=context
    )
    
    printer.pretty("Cause and Effect Result", result)
    print("\n=== Successfully Ran Reasoning Cause and Effect ===\n")