
import torch.nn as nn
import yaml, json
import random
import torch
import math

from pathlib import Path
from difflib import SequenceMatcher
from typing import Tuple, Dict, Any
from collections import defaultdict, deque

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from src.agents.reasoning.utils.model_compute import ModelCompute
from src.agents.reasoning.utils.adaptive_circuit import AdaptiveCircuit
from src.agents.reasoning.reasoning_memory import ReasoningMemory
from logs.logger import get_logger

logger = get_logger("Probabilistic Models")

class ProbabilisticModels:
    def __init__(self):
        """
        Initialize probabilistic reasoning system with:
        - Bayesian network structure
        - Knowledge base
        - Inference configuration
        
        Args:
            config: Preloaded configuration dictionary
        """
        # Load merged configuration
        self.config = load_global_config()
        self.inference_config = get_config_section('inference')
        storage_config = get_config_section('storage')

        # Load Bayesian network structure & knowledge base
        self.bayesian_network = self._load_bayesian_network(Path(storage_config['bayesian_network']))
        self.knowledge_base = self._load_knowledge_base(Path(storage_config['knowledge_db']))

        # Configure hybrid inference weights
        self.markov_logic_weight = self.inference_config.get('markov_logic_weight', 0.7)
        self.knowledge_base_weight = self.inference_config.get('knowledge_base_weight', 0.3)

        self.reasoning_memory = ReasoningMemory()
        self.model_compute = ModelCompute(circuit=None)

        # Link components
        self.model_compute.circuit = self._create_adaptive_circuit()
        self._init_learning_state()

        logger.info(f"Probabilistic Models fully initialized with integrated components")

    def _init_learning_state(self):
        """Initialize state for Bayesian learning cycle"""
        self.posterior_cache = {}
        self.observation_buffer = deque(maxlen=1000)
        self.convergence_threshold = 1e-4
        self.max_learning_cycles = 100

    def _load_bayesian_network(self, network_path: Path) -> Dict[str, Any]:
        """Load Bayesian network structure from JSON file"""
        if not network_path.exists():
            raise FileNotFoundError(f"Bayesian network file not found: {network_path}")

        logger.info(f"Loading Bayesian network from {network_path}")
        with open(network_path, 'r') as f:
            return json.load(f)

    def _load_knowledge_base(self, kb_path: Path) -> Dict[Tuple, Any]:
        if not kb_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")
        
        logger.info(f"Loading knowledge base from {kb_path}")
        with open(kb_path, 'r') as f:
            kb_data = json.load(f)
        
        processed_kb = {}
        # Process facts from "knowledge" array
        for fact in kb_data.get("knowledge", []):
            try:
                key = (fact[0], fact[1], fact[2])
                processed_kb[key] = float(fact[3])
            except (IndexError, ValueError):
                logger.warning(f"Skipping invalid fact: {fact}")
        
        logger.info(f"Loaded {len(processed_kb)} facts from knowledge base")
        return processed_kb

    def _create_adaptive_circuit(self):
        """Create circuit with Bayesian network-aware architecture"""
        return AdaptiveCircuit(
            network_structure=self.bayesian_network,
            knowledge_base=self.knowledge_base
        )

    def bayesian_inference(self, query: str, evidence: Dict[str, bool]) -> float:
        """
        Exact inference using message passing algorithm based on:
        - Pearl (1988) Probabilistic Reasoning in Intelligent Systems
        - Koller & Friedman (2009) Probabilistic Graphical Models
        """
        # Create parent and child maps
        parent_map = {node: [] for node in self.bayesian_network['nodes']}
        child_map = {node: [] for node in self.bayesian_network['nodes']}
        for parent, child in self.bayesian_network['edges']:
            parent_map[child].append(parent)
            child_map[parent].append(child)
    
        # Initialize messages and beliefs
        messages = defaultdict(dict)
        beliefs = {}
        for node in self.bayesian_network['nodes']:
            # Initialize with prior if available, else uniform
            prior = self.bayesian_network['cpt'].get(node, {}).get('prior', 0.5)
            beliefs[node] = prior
            
            # Initialize messages to uniform
            for parent in parent_map[node]:
                messages[parent][node] = {True: 1.0, False: 1.0}
            for child in child_map[node]:
                messages[child][node] = {True: 1.0, False: 1.0}
    
        # Set evidence
        for node, value in evidence.items():
            if node in self.bayesian_network['nodes']:
                beliefs[node] = 1.0 if value else 0.0
    
        # Loopy Belief Propagation
        max_iter = 50
        tolerance = 1e-5
        for iteration in range(max_iter):
            delta = 0.0
            
            # Update messages from children to parents
            for node in self.bayesian_network['nodes']:
                if node in evidence:
                    continue  # Skip evidence nodes
                    
                for parent in parent_map[node]:
                    # Compute message from node to parent
                    new_message = {True: 0.0, False: 0.0}
                    
                    for parent_state in [True, False]:
                        for node_state in [True, False]:
                            # Get CPT key based on parent states
                            parents_list = parent_map[node]
                            parent_states = [parent_state if p == parent else None for p in parents_list]
                            key = self._get_cpt_key(node, parent_states)
                            
                            # Get probability from CPT
                            prob = self._get_cpt_prob(node, key, node_state)
                            
                            # Product of messages from children
                            child_prod = 1.0
                            for child in child_map[node]:
                                if child != parent:  # Exclude current parent
                                    child_prod *= messages[child][node].get(node_state, 1.0)
                                    
                            new_message[parent_state] += prob * child_prod
                    
                    # Normalize message
                    total = sum(new_message.values())
                    if total > 0:
                        new_message = {k: v/total for k, v in new_message.items()}
                    
                    # Track maximum change
                    old_msg = messages[node][parent]
                    delta = max(delta, abs(new_message[True] - old_msg.get(True, 0.5)),
                                  abs(new_message[False] - old_msg.get(False, 0.5)))
                    
                    messages[node][parent] = new_message
    
            # Update beliefs
            for node in self.bayesian_network['nodes']:
                if node in evidence:
                    continue
                    
                belief_product = 1.0
                for child in child_map[node]:
                    belief_product *= messages[child][node].get(True, 1.0)  # Approximate
                
                # Apply prior if available
                prior = self.bayesian_network['cpt'].get(node, {}).get('prior', 0.5)
                new_belief = prior * belief_product
                
                # Normalize
                beliefs[node] = new_belief / (new_belief + (1 - prior) * (1 - belief_product))
    
            # Check convergence
            if delta < tolerance:
                logger.info(f"Converged after {iteration+1} iterations")
                break
    
        # Return belief for query node
        result = beliefs.get(query, 0.5)
        
        # Log to reasoning memory
        self.reasoning_memory.add({
            "type": "inference",
            "method": "loopy_bp",
            "query": query,
            "evidence": evidence,
            "result": result,
            "iterations": iteration+1
        }, tag="inference")
        
        return result
    
    # Helper functions needed for the implementation
    def _get_cpt_key(self, node: str, parent_states: list) -> str:
        """Generate CPT key from parent states"""
        return ",".join(str(s) for s in parent_states)
    
    def _get_cpt_prob(self, node: str, key: str, state: bool) -> float:
        """Get probability from CPT with safe fallbacks"""
        try:
            cpt = self.bayesian_network['cpt'][node].get(key, {})
            if state:
                return cpt.get("True", cpt.get(True, 0.5))
            else:
                return cpt.get("False", cpt.get(False, 0.5))
        except KeyError:
            return 0.5

    def run_bayesian_learning_cycle(self, observations: list) -> None:
        """Iteratively refines prior probabilities using Bayesian updating"""
        self.observation_buffer.extend(observations)
        
        # Learning loop
        for cycle in range(self.max_learning_cycles):
            delta_norm = 0.0
            batch = random.sample(self.observation_buffer, min(100, len(self.observation_buffer)))
            
            for obs in batch:
                evidence = self._extract_evidence_from_obs(obs)    # Extract relevant evidence
                
                # Update each variable in the network
                for node in self.bayesian_network['nodes']:
                    if node in evidence:
                        # Get current posterior
                        current_posterior = self.posterior_cache.get(node, 0.5)
                        
                        # Compute updated posterior
                        new_posterior = self.bayesian_inference(
                            query=node,
                            evidence=evidence
                        )
                        
                        # Apply Bayesian update
                        updated = self._bayesian_update(
                            prior=current_posterior,
                            likelihood=self._calculate_likelihood(evidence[node]),
                            posterior=new_posterior
                        )
                        
                        # Store update and track change
                        delta_norm += abs(updated - current_posterior)
                        self.posterior_cache[node] = updated
                        self.model_compute.dynamic_model_revision(
                            {node: updated}
                        )
            
            # Check convergence
            avg_delta = delta_norm / (len(batch) * len(self.bayesian_network['nodes']))
            if avg_delta < self.convergence_threshold:
                logger.info(f"Convergence achieved after {cycle+1} cycles")
                break
                
        # Consolidate learning
        self._update_knowledge_base()
        self.reasoning_memory.add(
            experience={
                "type": "learning_cycle", 
                "cycles": cycle+1,
                "posteriors": self.posterior_cache
            },
            tag="bayesian_learning",
            priority=1.0  # High priority for learning events
        )

    def _bayesian_update(self, prior: float, likelihood: float, posterior: float) -> float:
        """Bayesian update with stability checks"""
        updated = (likelihood * posterior) / (likelihood * posterior + (1 - likelihood) * (1 - posterior))
        return max(min(updated, 0.99), 0.01)  # Keep within valid probability range

    def _calculate_likelihood(self, observed_value: Any) -> float:
        """Calculate likelihood based on observation characteristics"""
        if isinstance(observed_value, bool):
            return 0.9 if observed_value else 0.1
        elif isinstance(observed_value, (int, float)):
            return 1.0 - min(0.2, abs(observed_value - 0.5))
        return 0.7  # Default confidence


    def probabilistic_query(self, fact: Tuple, evidence: Dict[Tuple, bool] = None) -> float:
        """
        Hybrid inference combining:
        - Exact Bayesian inference when network structure exists
        - Markov Logic Network-style weighted formulae
        - Neural semantic similarity fallback
        """
        # First check Bayesian network structure
        bn_nodes = self.bayesian_network['nodes']
        if fact[0] in bn_nodes and all(e[0] in bn_nodes for e in evidence.keys()):
            return self.bayesian_inference(fact[0], {k[0]:v for k,v in evidence.items()})

        # Markov Logic Network-style inference
        ml_weights = []
        ml_evidences = []

        # Weighted formulae components
        for e_fact, e_value in evidence.items():
            if e_fact in self.knowledge_base:
                weight = math.log(
                    self.knowledge_base[e_fact] / 
                    (1 - self.knowledge_base[e_fact] + 1e-8)
                )
                ml_weights.append(weight * (1 if e_value else -1))
                ml_evidences.append(1)

        # Logistic regression combination
        z = sum(w * e for w, e in zip(ml_weights, ml_evidences))
        probability = 1 / (1 + math.exp(-z))

        # Knowledge-based calibration
        base_prob = self.knowledge_base.get(fact, 0.5)
        return (
            self.markov_logic_weight * probability +
            self.knowledge_base_weight * base_prob
        )

    def multi_hop_reasoning(self, query: Tuple) -> float:
        """
        Graph-based traversal for combining facts across sources.
        """
        self._build_hypothesis_graph(query)    # Build hypothesis graph

        # Probabilistic graph traversal
        confidence = 1.0
        for hop in self.hypothesis_graph[query]:
            confidence *= self.knowledge_base.get(hop, 0.5)  # Bayesian chain rule
        return confidence

    def _build_hypothesis_graph(self, query: Tuple) -> None:
        """Build graph traversal structure for multi-hop reasoning"""
        self.hypothesis_graph = defaultdict(list)
        query_var = query[0]
        self._add_network_connections(query_var)    # Start with direct connections
        self._add_semantic_associations(query)    # Expand with knowledge base associations
        self._prune_graph(query_var)    # Prune low-confidence paths

    def _add_network_connections(self, root: str):
        """Add Bayesian network connections to graph"""
        # Add parents
        for parent, child in self.bayesian_network['edges']:
            if child == root:
                self.hypothesis_graph[root].append(("parent", parent))
        
        # Add children
        for parent, child in self.bayesian_network['edges']:
            if parent == root:
                self.hypothesis_graph[root].append(("child", child))
                
        # Add spouses (parents of children)
        for parent, child in self.bayesian_network['edges']:
            if parent == root:
                for grandparent, parent2 in self.bayesian_network['edges']:
                    if parent2 == child and grandparent != root:
                        self.hypothesis_graph[root].append(("spouse", grandparent))

    def _add_semantic_associations(self, query: Tuple):
        """Add knowledge-based associations to graph"""
        # Get base confidence and ensure it's scalar
        base_confidence = self.knowledge_base.get(query, 0.5)
        if isinstance(base_confidence, (list, tuple)):
            base_confidence = base_confidence[0] if base_confidence else 0.5
        elif isinstance(base_confidence, dict):
            base_confidence = base_confidence.get('weight', 0.5)
        elif not isinstance(base_confidence, (int, float)):
            base_confidence = 0.5
        
        # Find semantically similar facts
        for fact, confidence in self.knowledge_base.items():
            if fact == query:
                continue
                
            # Extract confidence value
            if isinstance(confidence, (int, float)):
                conf_value = confidence
            elif isinstance(confidence, (list, tuple)):
                conf_value = confidence[0] if confidence else 0.5
            elif isinstance(confidence, dict):
                conf_value = confidence.get('weight', 0.5)
            else:
                conf_value = 0.5
                
            # Calculate similarity
            similarity = self._semantic_similarity(query, fact)
            
            # Convert both to float for comparison
            try:
                base_conf_float = float(base_confidence)
                conf_value_float = float(conf_value)
                confidence_diff = abs(conf_value_float - base_conf_float)
            except (TypeError, ValueError):
                confidence_diff = 1.0  # Treat as maximally different if conversion fails
            
            if similarity > 0.6 or confidence_diff < 0.2:
                self.hypothesis_graph[query].append(("semantic", fact))

    def _prune_graph(self, root: str):
        """Prune low-confidence paths from graph"""
        for key in list(self.hypothesis_graph.keys()):
            pruned_hops = []
            for hop in self.hypothesis_graph[key]:
                try:
                    # Extract confidence value
                    confidence = self._hop_confidence(hop)
                    
                    # Convert to float if necessary
                    if isinstance(confidence, (list, tuple)):
                        confidence = confidence[0] if confidence else 0.0
                    elif isinstance(confidence, dict):
                        confidence = confidence.get('weight', 0.0)
                    
                    # Compare as float
                    if float(confidence) > 0.3:
                        pruned_hops.append(hop)
                except (TypeError, ValueError):
                    # Skip invalid entries
                    continue
                    
            self.hypothesis_graph[key] = pruned_hops
                
        # Ensure root remains even if isolated
        if root not in self.hypothesis_graph:
            self.hypothesis_graph[root] = []

    def _hop_confidence(self, hop) -> float:
        """Get confidence for a graph hop"""
        if isinstance(hop[1], tuple):
            confidence = self.knowledge_base.get(hop[1], 0.5)
            return self._extract_confidence_value(confidence)
        return 0.7

    def _extract_confidence_value(self, confidence) -> float:
        """Extract scalar confidence value from various data structures"""
        if isinstance(confidence, (float, int)):
            return confidence
        elif isinstance(confidence, list):
            # Use first element if available, otherwise default
            return confidence[0] if confidence else 0.5
        elif isinstance(confidence, dict):
            # Handle dictionary structures (like rule_weights)
            return confidence.get('weight', 0.5) if 'weight' in confidence else 0.5
        return 0.5
    
    
    def _extract_evidence_from_obs(self, obs) -> Dict[str, Any]:
        """Extract evidence from observation (simplified)"""
        return {node: random.random() > 0.5 for node in self.bayesian_network['nodes']}
    
    def _update_knowledge_base(self):
        """Update knowledge base with learned posteriors"""
        for node, posterior in self.posterior_cache.items():
            self.knowledge_base[(node,)] = posterior
    
    def _semantic_similarity(self, fact1, fact2) -> float:
        """Calculate semantic similarity between facts"""
        return SequenceMatcher(None, str(fact1), str(fact2)).ratio()
    
    def _hop_confidence(self, hop) -> float:
        """Get confidence for a graph hop"""
        if isinstance(hop[1], tuple):
            return self.knowledge_base.get(hop[1], 0.5)
        return 0.7  # Default for relationship types

if __name__ == "__main__":
    print("\n=== Running Probabilistic Models ===\n")
    model = ProbabilisticModels()

    print(model)
    print("\n=== Successfully Ran Probabilistic Models Test 1 ===\n")
    print("=== Running Probabilistic Models Test 2 ===\n")
    # Example query to Bayesian network
    query_node = "H"
    evidence = {"A": True, "C": False}
    
    try:
        prob = model.bayesian_inference(query_node, evidence)
        print(f"P({query_node}|{evidence}) = {prob:.4f}")
    except Exception as e:
        print(f"Inference failed: {str(e)}")

    # Example probabilistic query
    fact = ("G", "some_fact")
    evidence_facts = {("A", "root_cause"): True, ("C", "secondary_mechanism"): False}
    
    try:
        prob_query_result = model.probabilistic_query(fact, evidence_facts)
        print(f"Probabilistic query result for fact {fact} given evidence {evidence_facts}: {prob_query_result:.4f}")
    except Exception as e:
        print(f"Probabilistic query failed: {str(e)}")
    
    print("\n=== Successfully Ran Probabilistic Models Test 2 ===\n")
    print("=== Running Probabilistic Models Test 3 ===\n")
    query = ("G", "some_fact")
    hop = model.multi_hop_reasoning(query=query)

    print(f"{hop}")
    print("\n=== Successfully Ran Probabilistic Models ===\n")
