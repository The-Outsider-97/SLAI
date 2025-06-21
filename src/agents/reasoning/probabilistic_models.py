import math
import torch
import random
import yaml, json
import numpy as np
import torch.nn as nn

from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from typing import Tuple, Dict, Any, Optional
from collections import defaultdict, deque

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from src.agents.reasoning.utils.model_compute import ModelCompute
from src.agents.reasoning.utils.adaptive_circuit import AdaptiveCircuit
from src.agents.reasoning.utils.pgmpy_wrapper import PgmpyBayesianNetwork
from src.agents.reasoning.reasoning_memory import ReasoningMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Probabilistic Models")
printer = PrettyPrinter

class ProbabilisticModels(nn.Module):
    def __init__(self):
        """
        Initialize probabilistic reasoning system with:
        - Bayesian network structure
        - Knowledge base
        - Inference configuration
        
        Args:
            config: Preloaded configuration dictionary
        """
        super().__init__()

        # Load merged configuration
        self.config = load_global_config()
        self.semantic_frames_path = self.config.get('semantic_frames_path')
        self.contradiction_threshold = self.config.get('contradiction_threshold')
        self.markov_logic_weight = self.config.get('markov_logic_weight')

        self.inference_config = get_config_section('inference')
        self.convergence_threshold = self.inference_config.get('convergence_threshold')
        self.max_learning_cycles =  self.inference_config.get('max_learning_cycles')
        self.knowledge_base_weight = self.inference_config.get('knowledge_base_weight')
        self.structural_weights = self.inference_config.get('structural_weights')

        self.storage_config = get_config_section('storage')
        self.bayesian_network_path = self.storage_config.get('bayesian_network')
        self.knowledge_db_path = self.storage_config.get('knowledge_db')

        # Load network paths from config
        self.net_config = get_config_section('networks')
        for key, path in self.net_config.items():
            setattr(self, key, path)

        self.bayesian_network = self._load_bayesian_network(Path(self.bayesian_network_path))
        self.semantic_frames = self._load_semantic_frames(Path(self.semantic_frames_path))
        self.knowledge_base = self._load_knowledge_base(Path(self.knowledge_db_path))

        self.pgmpy_bn = PgmpyBayesianNetwork(self.bayesian_network)
        self.reasoning_memory = ReasoningMemory()
        self.model_compute = ModelCompute(circuit=None)
        self.adaptive_circuit = AdaptiveCircuit(
            network_structure=self.bayesian_network, # Pass the raw JSON here
            knowledge_base=self.knowledge_base
        )

        # Link components
        self.agent = None
        self.domain = None
        self.posterior_cache = {}
        self.similarity_cache = {}
        self.observation_buffer = deque(maxlen=1000)
        
        # Initialize network selection mappings
        self._initialize_network_selectors()
        
        logger.info(f"Probabilistic Models fully initialized with integrated components")

    def link_agent(self, agent):
        """Connect to parent ReasoningAgent"""
        self.agent = agent
        logger.info(f"Linked ProbabilisticModels to ReasoningAgent")

    def _initialize_network_selectors(self):
        """Initializes mappings for network selection."""
        # Mapping for Bayesian Networks: (task_type, complexity) -> network_key
        self.bn_selector_map = {
            ("simple_check", "low"): "bn2x2",
            ("sequential_inference", "low"): "bn3x3",
            ("common_cause_analysis", "low"): "bn4x4",
            ("contextual_reasoning", "medium"): "bn5x5",
            ("multi_source_fusion", "medium"): "bn6x6",
            ("hierarchical_inference", "medium"): "bn7x7",
            ("dual_process_modeling", "high"): "bn8x8",
            ("modular_diagnostics", "high"): "bn9x9",
            ("hybrid_reasoning", "high"): "bn10x10",
            ("scalability_test", "very_high"): "bn20x20",
            ("large_context", "very_high"): "bn32x32",
            ("stress_test", "extreme"): "bn64x64",
        }
        
        # Mapping for Grid Networks: complexity -> [dimensions]
        self.gn_selector_map = {
            "low": [2, 3, 4],
            "medium": [5, 6, 7, 8],
            "high": [9, 10, 20],
            "very_high": [32],
            "extreme": [64]
        }
        
    def is_bayesian_task(self, task_type: str) -> bool:
        """Determines if the task is better suited for Bayesian or Grid network."""
        grid_keywords = ["grid", "spatial", "image", "map", "layout", "pixel", "sensor", "topology"]
        bayesian_keywords = ["causal", "diagnostic", "logical", "inference", "reasoning", "probability"]
        
        task_lower = task_type.lower()
        
        # Check for explicit keywords
        if any(keyword in task_lower for keyword in grid_keywords):
            return False
        if any(keyword in task_lower for keyword in bayesian_keywords):
            return True
            
        # Heuristic for task structure
        if "network" in task_lower or "graph" in task_lower:
            return "bayes" in task_lower or "pgm" in task_lower
            
        # Default to Bayesian for abstract reasoning tasks
        return not ("adjacency" in task_lower or "neighbor" in task_lower)
    
    def select_network(self, task_type: str, complexity: str, speed_requirement: str) -> str:
        """
        High-level dispatcher that selects the most appropriate network file path
        with enhanced decision logic and fallback mechanisms.
        """
        task_type = task_type.lower()
        complexity = complexity.lower()
        speed_requirement = speed_requirement.lower()
    
        try:
            if self.is_bayesian_task(task_type):
                return self._select_bayesian_network(task_type, complexity, speed_requirement)
            return self._select_grid_network(task_type, complexity, speed_requirement)
        except Exception as e:
            logger.error(f"Network selection failed: {str(e)}. Using default")
            return self.bayesian_network_path
    
    def _select_grid_network(self, task_type: str, complexity: str, speed_requirement: str) -> str:
        """Selects grid network with task-aware dimension selection."""
        # Get available dimensions for this complexity
        dims = self.gn_selector_map.get(complexity)
        
        if not dims:
            # Complexity fallback logic
            if complexity in ["very_high", "extreme"]:
                dims = self.gn_selector_map.get("high", [8, 9, 10])
            elif complexity == "low":
                dims = self.gn_selector_map.get("medium", [4, 5, 6])
            else:
                dims = [8]  # Default to medium grid
            logger.warning(f"Unknown grid complexity '{complexity}'. Using fallback: {dims}")
    
        # Task-based dimension hints
        size_hint = None
        if "high_res" in task_type or "detailed" in task_type:
            size_hint = max(dims)
        elif "low_res" in task_type or "coarse" in task_type:
            size_hint = min(dims)
        
        # Speed-based selection
        if size_hint:
            dim = size_hint
        elif speed_requirement == "fast":
            dim = min(dims)
        elif speed_requirement == "accurate":
            dim = max(dims)
        else:  # balanced
            dim = sorted(dims)[len(dims) // 2]
        
        # Validate dimension exists
        network_key = f"gn{dim}x{dim}"
        if hasattr(self, network_key):
            return getattr(self, network_key)
        
        # Fallback to nearest available size
        available_dims = [int(key[2:-3]) for key in self.net_config if key.startswith("gn")]
        closest_dim = min(available_dims, key=lambda x: abs(x - dim))
        fallback_key = f"gn{closest_dim}x{closest_dim}"
        logger.warning(f"Grid network {dim}x{dim} not found. Using {closest_dim}x{closest_dim}")
        return getattr(self, fallback_key, self.gn8x8)
    
    def _select_bayesian_network(self, task_type: str, complexity: str, speed_requirement: str) -> str:
        """Selects Bayesian network with similarity-based fallback."""
        # Normalize inputs
        task_type = task_type.lower()
        complexity = complexity.lower()
        lookup_key = (task_type, complexity)
        
        # 1. Exact match
        if lookup_key in self.bn_selector_map:
            network_key = self.bn_selector_map[lookup_key]
            if hasattr(self, network_key):
                return getattr(self, network_key)
        
        # 2. Task similarity fallback
        candidate_tasks = [t for t, _ in self.bn_selector_map.keys() if t != task_type]
        if candidate_tasks:
            # Find most similar task
            best_match = max(candidate_tasks, 
                             key=lambda t: SequenceMatcher(None, task_type, t).ratio())
            similarity = SequenceMatcher(None, task_type, best_match).ratio()
            
            if similarity > 0.7:  # Threshold for usable similarity
                network_key = self.bn_selector_map[(best_match, complexity)]
                if hasattr(self, network_key):
                    logger.info(f"Using similar task '{best_match}' (similarity: {similarity:.2f})")
                    return getattr(self, network_key)
        
        # 3. Complexity-based fallback
        complexity_fallbacks = {
            "low": ["bn2x2", "bn3x3", "bn4x4"],
            "medium": ["bn5x5", "bn6x6", "bn7x7"],
            "high": ["bn8x8", "bn9x9", "bn10x10"],
            "very_high": ["bn20x20", "bn32x32"],
            "extreme": ["bn64x64"]
        }
        
        # Get candidates for this complexity level
        candidates = complexity_fallbacks.get(complexity, [])
        if not candidates:
            # Handle unknown complexity
            if complexity in ["very_high", "extreme"]:
                candidates = complexity_fallbacks["high"]
            else:
                candidates = complexity_fallbacks["medium"]
        
        # Speed-based selection
        if speed_requirement == "fast":
            network_key = candidates[0]
        elif speed_requirement == "accurate":
            network_key = candidates[-1]
        else:  # balanced
            network_key = candidates[len(candidates) // 2]
        
        return getattr(self, network_key, self.bayesian_network_path)

    def _load_bayesian_network(self, network_path: Path) -> Dict[str, Any]:
        """Load Bayesian network structure from JSON file"""
        if not network_path.exists():
            raise FileNotFoundError(f"Bayesian network file not found: {network_path}")

        logger.info(f"Loading Bayesian network from {network_path}")
        with open(network_path, 'r') as f:
            return json.load(f)

    def _load_semantic_frames(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            logger.error(f"Semantic frames file not found: {path}")
            return {}

        logger.info(f"Loading Semantic frames from {path}")
        with open(path, 'r') as f:
            return json.load(f)

    def _load_knowledge_base(self, kb_path: Path) -> Dict[Tuple, Any]:
        if not kb_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")
        logger.info(f"Loading knowledge base from {kb_path}")
        with open(kb_path, 'r') as f:
            kb_data = json.load(f)
        
        processed_kb = {}
        for fact in kb_data.get("knowledge", []):
            if isinstance(fact, dict):
                s = fact.get("subject")
                p = fact.get("predicate")
                o = fact.get("object")
                weight = fact.get("weight", 0.5)
            try:
                # Handle 4-element format: [subject, predicate, object, weight]
                if len(fact) == 4:
                    s, p, o, weight = fact
                    # Convert object to boolean if needed
                    if isinstance(o, str):
                        if o.lower() == 'true':
                            o_val = True
                        elif o.lower() == 'false':
                            o_val = False
                        else:
                            o_val = o
                    else:
                        o_val = o
                    key = (s, p, o_val)
                    processed_kb[key] = float(weight)
            except Exception as e:
                logger.warning(f"Skipping invalid fact in KB: {fact} | Error: {str(e)}")
        
        logger.info(f"Loaded {len(processed_kb)} facts from knowledge base")
        return processed_kb

    def bayesian_inference(self, query: str, evidence: Dict[str, bool]) -> float:
        """
        Exact inference using message passing algorithm based on:
        - Pearl (1988) Probabilistic Reasoning in Intelligent Systems
        - Koller & Friedman (2009) Probabilistic Graphical Models
        """
        if not self.pgmpy_bn:
            logger.error("Pgmpy Bayesian Network not available for inference. Returning 0.5")
            return 0.5 # Fallback probability

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
                    old_msg = messages[node].get(parent, {})
                    delta = max(delta, abs(new_message.get(True, 0.5) - old_msg.get(True, 0.5)),
                                  abs(new_message.get(False, 0.5) - old_msg.get(False, 0.5)))
                    
                    messages[node][parent] = new_message
    
            # Update beliefs
            for node in self.bayesian_network['nodes']:
                if node in evidence:
                    continue
                    
                belief_product = 1.0
                for child in child_map[node]:
                    belief_product *= messages[child].get(node, {}).get(True, 1.0)  # Approximate
                
                # Apply prior if available
                prior = self.bayesian_network['cpt'].get(node, {}).get('prior', 0.5)
                new_belief = prior * belief_product
                
                # Normalize
                current_belief = beliefs.get(node, 0.5)
                if (new_belief + (1 - prior) * (1 - belief_product)) > 0:
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
            "iterations": iteration+1 if 'iteration' in locals() else 0
        }, tag="inference")
        
        return result
    
    # Helper functions needed for the implementation
    def _get_cpt_key(self, node: str, parent_states: list) -> str:
        """Generate CPT key from parent states"""
        # A more robust key generation that handles None for unknown parent states
        key_parts = []
        for state in parent_states:
            if state is None:
                # This part is tricky. How to marginalize?
                # For now, let's assume a placeholder that won't match, forcing fallback.
                key_parts.append("UNKNOWN")
            else:
                key_parts.append(str(state))
        return ",".join(key_parts)

    
    def _get_cpt_prob(self, node: str, key: str, state: bool) -> float:
        """Get probability from CPT with safe fallbacks"""
        try:
            cpt = self.bayesian_network['cpt'][node]
            # Handle single-parent CPTs where key is 'True' or 'False'
            if key in cpt:
                prob_dict = cpt[key]
            else:
                # For multi-parent, the key is a comma-separated string
                prob_dict = cpt.get(key, {})

            if state:
                return prob_dict.get("True", 0.5)
            else:
                return prob_dict.get("False", 0.5)
        except KeyError:
            # Fallback for nodes with no CPT entries (or root nodes)
            if 'prior' in self.bayesian_network['cpt'].get(node, {}):
                prior_prob = self.bayesian_network['cpt'][node]['prior']
                return prior_prob if state else 1.0 - prior_prob
            return 0.5


    def run_bayesian_learning_cycle(self, observations: list) -> None: # Keep this method
        """Iteratively refines prior probabilities using Bayesian updating"""
        self.observation_buffer.extend(observations)
        
        learning_cycles_performed = 0
        avg_delta = 0.0
        nodes_updated_in_batch = 0

        for cycle in range(self.max_learning_cycles):
            learning_cycles_performed = cycle + 1
            delta_norm = 0.0
            
            if not self.observation_buffer:
                logger.info("Observation buffer empty, skipping learning cycle.")
                break
            
            batch_size = min(100, len(self.observation_buffer))
            if batch_size == 0 :
                break

            batch = random.sample(list(self.observation_buffer), batch_size) 
            
            nodes_updated_in_batch = 0
            for obs in batch:
                evidence = self._extract_evidence_from_obs(obs)    
                if not evidence: continue

                for node in self.bayesian_network.get('nodes', []):
                    if node in evidence:
                        nodes_updated_in_batch +=1
                        current_posterior = self.posterior_cache.get(node, self.bayesian_network['cpt'].get(node, {}).get('prior', 0.5))
                        
                        new_posterior = self.bayesian_inference(query=node, evidence={k: v['value'] for k, v in evidence.items() if isinstance(v, dict)})
                        
                        likelihood = self._calculate_likelihood(evidence[node]['value'] if isinstance(evidence[node], dict) else evidence[node])
                        updated = self._bayesian_update(
                            prior=current_posterior,
                            likelihood=likelihood, 
                            posterior=new_posterior 
                        )
                        
                        delta_norm += abs(updated - current_posterior)
                        self.posterior_cache[node] = updated
                        
                        if self.model_compute:
                             self.model_compute.dynamic_model_revision({node: updated})
            
            if nodes_updated_in_batch > 0 : 
                avg_delta = delta_norm / nodes_updated_in_batch

            if avg_delta < self.convergence_threshold and cycle > 0:
                logger.info(f"Bayesian learning converged after {learning_cycles_performed} cycles. Average delta: {avg_delta:.6f}")
                break
            elif cycle == self.max_learning_cycles - 1:
                logger.info(f"Bayesian learning reached max {self.max_learning_cycles} cycles. Average delta: {avg_delta:.6f}")
        
        if learning_cycles_performed > 0 :
            self._update_knowledge_base()
            self.reasoning_memory.add(
                experience={
                    "type": "bayesian_learning_cycle_completed", 
                    "cycles_run": learning_cycles_performed,
                    "final_avg_delta": avg_delta,
                    "num_posteriors_updated": len(self.posterior_cache)
                },
                tag="bayesian_learning",
                priority=1.0 
            )
        else:
            logger.info("No Bayesian learning cycles performed.")

    def _bayesian_update(self, prior: float, likelihood: float, posterior: float) -> float:
        """Bayesian update with stability checks"""
        # This seems to be a custom heuristic, not standard Bayesian update. Standard update is posterior = (likelihood * prior) / evidence.
        # The provided formula seems to be trying to blend a new posterior with an old one. Let's keep it but make it stable.
        denominator = (likelihood * posterior + (1 - likelihood) * (1 - posterior))
        if denominator == 0:
            return prior
        updated = (likelihood * posterior) / denominator
        return max(min(updated, 0.99), 0.01)

    def _calculate_likelihood(self, observed_value: Any) -> float:
        """Calculate likelihood based on observation characteristics"""
        if isinstance(observed_value, bool):
            return 0.95 if observed_value else 0.05
        elif isinstance(observed_value, (int, float)):
            return 1.0 - min(0.5, abs(observed_value - 0.5))
        return 0.75  # Default confidence for less structured data

    def probabilistic_query(self, fact: Tuple, evidence: Dict[Tuple, bool] = None) -> float:
        """
        Hybrid inference combining Bayesian inference, Markov Logic, and semantic similarity.
        """
        if evidence is None: evidence = {}
        bn_nodes = self.bayesian_network.get('nodes', [])
        
        # Check if it's a pure Bayesian Network query
        is_bn_query = (len(fact) == 1 and fact[0] in bn_nodes and
                       all(len(e) == 1 and e[0] in bn_nodes for e in evidence))

        if is_bn_query:
            bn_evidence = {e[0]: val for e, val in evidence.items()}
            return self.bayesian_inference(fact[0], bn_evidence)
            
        # Fallback to Markov Logic Network style reasoning
        total_weight = 0.0
        # Add the base belief of the query fact itself, if it exists
        if fact in self.knowledge_base:
            prob = self.knowledge_base[fact]
            base_weight = math.log(prob / (1 - prob)) if 0 < prob < 1 else (10 if prob >= 1 else -10)
            total_weight += base_weight

        # Incorporate evidence
        for e_fact, e_value in evidence.items():
            if e_fact in self.knowledge_base:
                kb_prob = self.knowledge_base[e_fact]
                weight = math.log(kb_prob / (1 - kb_prob)) if 0 < kb_prob < 1 else (10 if kb_prob >= 1 else -10)
                # If evidence is true, add the weight; if false, subtract it.
                total_weight += weight if e_value else -weight
        
        # Convert final summed log-odds back to probability
        return 1.0 / (1.0 + math.exp(-total_weight))

    def multi_hop_reasoning(self, query: Tuple) -> float:
        """
        Graph-based traversal for combining facts across sources.
        """
        if not isinstance(query, tuple) or len(query) < 1:
            return 0.0

        self._build_hypothesis_graph(query)

        if query not in self.hypothesis_graph:
            logger.debug(f"Query {query} not found in hypothesis graph after building.")
            return self._extract_confidence_value(self.knowledge_base.get(query, 0.0))

        # Use Dijkstra-like algorithm to find the most confident path/aggregation
        q = [(1.0, query, [query])] # (confidence, current_node, path_so_far)
        max_confidence = self._extract_confidence_value(self.knowledge_base.get(query, 0.0))
        visited = {query}

        while q:
            current_conf, current_node, path = q.pop(0)
            
            # Aggregate confidences, considering diminishing returns from longer paths
            path_len_penalty = 0.95 ** (len(path) - 1)
            aggregated_confidence = current_conf * path_len_penalty
            max_confidence = max(max_confidence, aggregated_confidence)

            if len(path) > 5: continue # Limit depth

            for hop_type, next_node in self.hypothesis_graph.get(current_node, []):
                if next_node not in visited:
                    visited.add(next_node)
                    hop_conf = self._hop_confidence((hop_type, next_node))
                    # Combine confidence using a product (assuming independence)
                    new_conf = current_conf * hop_conf
                    q.append((new_conf, next_node, path + [next_node]))
        
        return max_confidence

    def _build_hypothesis_graph(self, query: Tuple) -> None:
        """Build graph traversal structure for multi-hop reasoning"""
        self.hypothesis_graph = defaultdict(list) 
        if not query or not isinstance(query, tuple) or len(query) < 1:
            return

        q = deque([query])
        visited = {query}
        depth_limit = 3

        while q:
            current_fact = q.popleft()
            
            # Stop if depth is too high
            if len(self.hypothesis_graph[current_fact]) > depth_limit * 2: continue

            # Add structural connections from BN
            if len(current_fact) > 0 and current_fact[0] in self.bayesian_network.get('nodes', []):
                self._add_network_connections(current_fact)
            
            # Add semantic associations from KB
            self._add_semantic_associations(current_fact)

            for _, next_node in self.hypothesis_graph[current_fact]:
                if next_node not in visited:
                    visited.add(next_node)
                    q.append(next_node)

        self._prune_graph(query[0])

    def _add_network_connections(self, fact: tuple):
        """Add Bayesian network connections related to a fact's subject."""
        root = fact[0]
        nodes = self.bayesian_network.get('nodes', [])
        if root not in nodes:
            return

        for p, c in self.bayesian_network.get('edges', []):
            if c == root and p in nodes:
                # Representing this as a new fact tuple
                parent_fact = (p, 'is_parent_of', root)
                self.hypothesis_graph[fact].append(("parent", parent_fact))
            if p == root and c in nodes:
                child_fact = (c, 'is_child_of', root)
                self.hypothesis_graph[fact].append(("child", child_fact))

    def _add_semantic_associations(self, query: Tuple):
        """Add knowledge-based associations to graph"""
        base_confidence = self._extract_confidence_value(self.knowledge_base.get(query, 0.5))

        for fact, confidence in self.knowledge_base.items():
            if fact == query: continue
                
            conf_value = self._extract_confidence_value(confidence)
            similarity = self._semantic_similarity(query, fact)
            
            # Add link if semantically close or has similar confidence
            if similarity > 0.7 or abs(conf_value - base_confidence) < 0.15:
                self.hypothesis_graph[query].append(("semantic", fact))

    def _prune_graph(self, root: str):
        """Prune low-confidence paths from graph"""
        for key in list(self.hypothesis_graph.keys()):
            pruned_hops = []
            for hop in self.hypothesis_graph[key]:
                if self._hop_confidence(hop) > 0.2:  # Pruning threshold
                    pruned_hops.append(hop)
            self.hypothesis_graph[key] = pruned_hops

    def _hop_confidence(self, hop) -> float:
        """Get confidence for a graph hop"""
        hop_type, hop_target = hop
        base_conf = 0.5

        if hop_type == "semantic" and isinstance(hop_target, tuple):
            base_conf = self.knowledge_base.get(hop_target, 0.5)

            # Apply temporal decay
            if 'timestamp' in self.knowledge_base.get('_meta', {}).get(hop_target, {}):
                days_old = (datetime.now() - self.knowledge_base['_meta'][hop_target]['timestamp']).days
                decay = math.exp(-days_old / self.config.get('temporal_halflife', 180))
                base_conf *= decay

        # Structural hop (Bayesian network relationship)
        elif hop_type in ["parent", "child", "spouse"] and isinstance(hop_target, str):
            # Get relationship strength from CPT
            if hop_type == "parent":
                cpt = self.bayesian_network['cpt'].get(hop_target, {})
                base_conf = cpt.get('influence_strength', 0.7)
            else:
                base_conf = self.structural_weights.get(hop_type, 0.6)

        # Contextual boosting
        try:
            current_context = self.reasoning_memory.get_current_context()
        except AttributeError:
            current_context = []

        if 'high_uncertainty' in current_context:
            base_conf *= 0.8  # Reduce confidence in uncertain contexts
        elif 'crisis' in current_context:
            base_conf = min(1.0, base_conf * 1.2)  # Boost in crisis mode

        # Relationship-specific adjustments
        if hop_type == "causal":
            base_conf *= 1.1
        elif hop_type == "correlative":
            base_conf *= 0.9

            return self._extract_confidence_value(self.knowledge_base.get(hop_target, 0.1))
        elif hop_type in self.structural_weights:
            return self.structural_weights[hop_type]
        return max(0.1, min(0.99, base_conf))

    def _extract_confidence_value(self, confidence_data: Any) -> float: # Unchanged
        if isinstance(confidence_data, (float, int)):
            return float(confidence_data)
        elif isinstance(confidence_data, (list, tuple)):
            return float(confidence_data[0]) if confidence_data and isinstance(confidence_data[0], (float, int)) else 0.5
        elif isinstance(confidence_data, dict):
            return float(confidence_data.get('weight', 0.5)) if isinstance(confidence_data.get('weight'), (float, int)) else 0.5
        return 0.5

    def _extract_evidence_from_obs(self, obs) -> Dict[str, Any]:
        """
        Extracts structured evidence from observations for Bayesian inference.
        """
        evidence = {}
        nodes = self.bayesian_network.get('nodes', [])
        
        if isinstance(obs, dict):
            for node in nodes:
                if node in obs:
                    value = obs[node]
                    if isinstance(value, str):
                        if value.lower() in ['true', 'yes', 'positive']: evidence[node] = True
                        elif value.lower() in ['false', 'no', 'negative']: evidence[node] = False
                    else:
                        evidence[node] = bool(value)
        elif isinstance(obs, str):
            for frame, config in self.semantic_frames.items():
                if any(verb in obs.lower() for verb in config.get('verbs', [])):
                    for role, node_list in config.get('roles', {}).items():
                        for node in node_list:
                            if node in nodes:
                                evidence[node] = True # Simple activation

        # Handle time-series/sensor data
        elif isinstance(obs, (list, np.ndarray)):
            # Process sensor data streams
            sensor_mapping = self.config.get('sensor_node_mapping', {})
            for sensor_id, values in obs.items():
                if sensor_id in sensor_mapping:
                    node = sensor_mapping[sensor_id]
                    # Apply moving average for stability
                    window_size = min(5, len(values))
                    smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                    evidence[node] = smoothed[-1] > self.config.get('sensor_threshold', 0.7)

        # Convert to dictionary with confidence
        confident_evidence = {}
        for node, val in evidence.items():
            confident_evidence[node] = {'value': val, 'confidence': 0.9} # Default high confidence
            
        return confident_evidence

    def _update_knowledge_base(self) -> None:
        """Updates knowledge base with learned posterior probabilities from Bayesian learning."""
        if not self.posterior_cache: return
    
        update_count = 0
        conflict_count = 0

        for node, posterior in self.posterior_cache.items():
            # Create or update a fact representing the node's belief state
            fact = (node, "has_belief", True) # Represent belief as P(Node=True)
            current_value = self.knowledge_base.get(fact, 0.5)

            # Conflict resolution
            if abs(posterior - current_value) > self.config.get('knowledge_conflict_threshold', 0.3):
                conflict_count += 1
                # Resolve via weighted average
                posterior = (posterior * 0.7) + (current_value * 0.3)
                logger.info(f"Resolved knowledge conflict for {node}: {current_value:.2f} → {posterior:.2f}")

            # Update with versioning
            self.knowledge_base[fact] = posterior
            self.knowledge_base.setdefault('_versions', {})[fact] = self.knowledge_base.get('_versions', {}).get(fact, []) + [{
                'value': posterior,
                'timestamp': datetime.now().isoformat(),
                'source': 'bayesian_learning'
            }]
            update_count += 1

            # Propagate to related facts
            for rel_type in ['causal', 'correlative', 'hierarchical']:
                related = self.ontology.get_related_nodes(node, rel_type)
                for related_node in related:
                    rel_strength = self.ontology.relationship_strength(node, related_node)
                    rel_fact = (related_node, "has_belief", True)
                    current = self.knowledge_base.get(rel_fact, 0.5)
                    updated = current + (posterior - 0.5) * rel_strength * 0.3
                    self.knowledge_base[rel_fact] = max(0.01, min(0.99, updated))

        # Apply confidence decay
        current_time = datetime.now()
        for fact, value in list(self.knowledge_base.items()):
            if isinstance(fact, tuple) and fact[1] == "has_belief":
                last_update = max([v['timestamp'] for v in self.knowledge_base['_versions'].get(fact, [])], default=None)
                if last_update:
                    days_old = (current_time - datetime.fromisoformat(last_update)).days
                    decay_factor = math.exp(-days_old / self.config.get('knowledge_halflife', 90))
                    self.knowledge_base[fact] = value * decay_factor

        if update_count > 0:
            logger.info(f"Updated {update_count} facts in the knowledge base from posteriors.")
        
        # Clear the cache after updating to avoid reusing old posteriors
        self.posterior_cache.clear()
        self.reasoning_memory.save_checkpoint()

    def _semantic_similarity(self, fact1: Any, fact2: Any) -> float:
        """
        Computes semantic similarity between two facts using hybrid approach:
        1. Knowledge graph embeddings
        2. Contextual relationship analysis
        3. Lexical similarity as fallback
        """
        s1 = str(fact1)
        s2 = str(fact2)
        cache_key = tuple(sorted((s1, s2)))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        sim = SequenceMatcher(None, s1, s2).ratio()
        
        # Handle different tuple lengths
        if isinstance(fact1, tuple) and isinstance(fact2, tuple):
            n = min(len(fact1), len(fact2), 3)

            # Get available components
            components = []
            if n >= 1:
                components.append(("subject", fact1[0], fact2[0]))
            if n >= 2:
                components.append(("predicate", fact1[1], fact2[1]))
            if n >= 3:
                components.append(("object", fact1[2], fact2[2]))

            # Calculate similarities
            weights = self.config.get('similarity_weights', [0.4, 0.4, 0.2])
            total_sim = 0.0
            total_weight = 0.0

            for i, (comp_type, val1, val2) in enumerate(components):
                if comp_type == "predicate":
                    sim = self._predicate_similarity(val1, val2)
                else:
                    sim = self._entity_similarity(val1, val2)

                total_sim += weights[i] * sim
                total_weight += weights[i]

        self.similarity_cache[cache_key] = sim
        return total_sim / total_weight if total_weight > 0 else 0.0

    def _predicate_similarity(self, pred1: str, pred2: str) -> float:
        """Compute predicate similarity using semantic frames"""
        if not hasattr(self, 'semantic_frames') or not self.semantic_frames:
            return SequenceMatcher(None, pred1, pred2).ratio()

        frame1_keys = [k for k, v in self.semantic_frames.items() if pred1 in v.get('verbs', [])]
        frame2_keys = [k for k, v in self.semantic_frames.items() if pred2 in v.get('verbs', [])]
        
        if frame1_keys and frame2_keys:
            # Check if they belong to the same frame
            if frame1_keys[0] == frame2_keys[0]:
                return 1.0
            # Check for shared roles
            frame1 = self.semantic_frames[frame1_keys[0]]
            frame2 = self.semantic_frames[frame2_keys[0]]
            role_overlap = len(set(frame1['roles']) & set(frame2['roles'])) / len(set(frame1['roles']) | set(frame2['roles']))
            return 0.5 + 0.5 * role_overlap # Give partial credit for shared roles
        
        return SequenceMatcher(None, pred1, pred2).ratio()

    def _entity_similarity(self, ent1: str, ent2: str) -> float:
        """Compute entity similarity using knowledge graph embeddings"""
        # Check cache first
        cache_key = f"sim_{ent1}_{ent2}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Fallback to string similarity if components missing
        if not hasattr(self, 'knowledge_graph') or not hasattr(self, 'ontology'):
            sim = SequenceMatcher(None, ent1, ent2).ratio()
            self.similarity_cache[cache_key] = sim
            return sim
        
        # Get embeddings from knowledge graph
        emb1 = self.knowledge_graph.get_embedding(ent1)
        emb2 = self.knowledge_graph.get_embedding(ent2)
        
        if emb1 is not None and emb2 is not None:
            # Cosine similarity
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        else:
            # Fallback to path similarity in ontology
            sim = self.ontology.path_similarity(ent1, ent2) or 0.0
        
        # Apply domain-specific adjustments
        if self.domain == "medical":
            if "symptom" in ent1 and "symptom" in ent2:
                sim = min(1.0, sim * 1.2)
        
        self.similarity_cache[cache_key] = sim
        return sim

if __name__ == "__main__":
    print("\n=== Running Probabilistic Models ===\n")
    model = ProbabilisticModels()

    print(model)
    print("\n=== Successfully Ran Probabilistic Models Test 1 ===\n")
    print("=== Running Probabilistic Models Test 2 ===\n")

    # Test network selection
    print("--- Network Selection Tests ---")
    print(f"Task: simple_check, low, fast -> {model.select_network('simple_check', 'low', 'fast')}")
    print(f"Task: contextual_reasoning, medium, balanced -> {model.select_network('contextual_reasoning', 'medium', 'balanced')}")
    print(f"Task: large_context, very_high, accurate -> {model.select_network('large_context', 'very_high', 'accurate')}")
    print(f"Task: spatial_layout, high, fast -> {model.select_network('spatial_layout', 'high', 'fast')}")
    print(f"Task: grid_navigation, medium, accurate -> {model.select_network('grid_navigation', 'medium', 'accurate')}")
    print("---------------------------------\n")

    # Example query to Bayesian network
    query_node = "H"
    evidence = {"A": True, "C": False}
    
    try:
        prob = model.bayesian_inference(query_node, evidence)
        print(f"P({query_node}|{evidence}) = {prob:.4f}")
    except Exception as e:
        print(f"Inference failed: {str(e)}")

    # Example probabilistic query
    fact = ("G", "has_belief", True)
    evidence_facts = {("A", "is_active", True): True, ("C", "is_active", True): False}
    
    try:
        prob_query_result = model.probabilistic_query(fact, evidence_facts)
        print(f"Probabilistic query result for fact {fact} given evidence {evidence_facts}: {prob_query_result:.4f}")
    except Exception as e:
        print(f"Probabilistic query failed: {str(e)}")
    
    print("\n=== Successfully Ran Probabilistic Models Test 2 ===\n")
    print("=== Running Probabilistic Models Test 3 ===\n")
    query = ("A", "is_parent_of", "B")
    hop = model.multi_hop_reasoning(query=query)

    print(f"Multi-hop reasoning for {query}: {hop}")
    
    print("\n* * * * * Knowledge Base * * * * *\n")
    model._update_knowledge_base()
    printer.pretty("KB state after potential updates", model.knowledge_base, "success")
    print("\n=== Successfully Ran Probabilistic Models ===\n")
