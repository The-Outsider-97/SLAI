
from collections import deque
import json
import itertools

from typing import Dict, Any, List, Tuple

import numpy as np

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Pgmpy Wrapper")
printer = PrettyPrinter

class PgmpyBayesianNetwork:
    def __init__(self, network_definition: Dict[str, Any]):
        """
        Initializes the pgmpy Bayesian Network from a JSON-like dictionary definition.

        Args:
            network_definition (Dict[str, Any]): The network structure and CPTs,
                                                 similar to your bayesian_network.json.
        """
        self.network_def = network_definition
        self.model = DiscreteBayesianNetwork()
        self.inference_engine = None
        self._build_model()

    def _build_model(self):
        """
        Constructs the pgmpy BayesianNetwork model from the definition with
        enhanced handling for complex network structures and CPT formats.
        """
        nodes = self.network_def.get('nodes', [])
        edges = self.network_def.get('edges', [])
        cpt_data = self.network_def.get('cpt', {})

        if not nodes:
            raise ValueError("Network definition must contain 'nodes'.")

        # Create model structure
        self.model = DiscreteBayesianNetwork()
        self.model.add_nodes_from(nodes)
        self.model.add_edges_from(edges)
        
        # Build parent map and check for cycles
        parent_map = self._build_parent_map()
        self._validate_network_structure(parent_map)
        
        # Process nodes in topological order to ensure dependency satisfaction
        node_order = self._get_topological_order(parent_map)
        cpds = []
        missing_cpt_nodes = []
        
        for node in node_order:
            if node not in cpt_data:
                missing_cpt_nodes.append(node)
                continue
                
            try:
                if node in parent_map and parent_map[node]:
                    # Node with parents
                    cpd = self._build_child_cpd(node, parent_map[node], cpt_data[node])
                else:
                    # Root node
                    cpd = self._build_root_cpd(node, cpt_data[node])
                cpds.append(cpd)
            except Exception as e:
                logger.error(f"Error building CPD for node {node}: {str(e)}")
                raise

        if missing_cpt_nodes:
            logger.warning(f"Missing CPT data for nodes: {missing_cpt_nodes}. Using uniform distributions.")
            for node in missing_cpt_nodes:
                cpd = self._build_uniform_cpd(node, parent_map.get(node, []))
                cpds.append(cpd)

        self.model.add_cpds(*cpds)
        
        # Validate model structure and parameters
        validation_result = self._validate_model()
        if not validation_result:
            raise ValueError("Failed to build a valid pgmpy model.")
        
        self.inference_engine = VariableElimination(self.model)
        logger.info("pgmpy model built and validated successfully.")

    def _build_parent_map(self) -> Dict[str, List[str]]:
        """Build mapping of each node to its parents with cycle detection"""
        parent_map = {node: [] for node in self.model.nodes()}
        visited = set()
        recursion_stack = set()
    
        def dfs(current):
            visited.add(current)
            recursion_stack.add(current)
            for parent in self.model.get_parents(current):
                parent_map[current].append(parent)
                if parent not in visited:
                    if dfs(parent):
                        return True
                elif parent in recursion_stack:
                    raise ValueError(f"Cycle detected in network: {parent} -> ... -> {current}")
            recursion_stack.remove(current)
            return False
    
        for node in self.model.nodes():
            if node not in visited:
                if dfs(node):
                    raise ValueError("Cycle detected in Bayesian network structure")
    
        return parent_map

    def _validate_network_structure(self, parent_map: Dict[str, List[str]]):
        """Validate network structure and CPT compatibility"""
        cpt_data = self.network_def.get('cpt', {})
        for node, parents in parent_map.items():
            if node in cpt_data:
                node_cpt = cpt_data[node]
                # Check if parent states match CPT keys
                if parents:
                    expected_combinations = 2 ** len(parents)
                    actual_combinations = len(node_cpt)
                    if actual_combinations != expected_combinations:
                        logger.warning(
                            f"Node {node} has {len(parents)} parents ({expected_combinations} combinations) "
                            f"but CPT has {actual_combinations} entries"
                        )

    def _get_topological_order(self, parent_map: Dict[str, List[str]]) -> List[str]:
        """Get nodes in topological order using Kahn's algorithm"""
        in_degree = {node: 0 for node in self.model.nodes()}
        for node, parents in parent_map.items():
            for parent in parents:
                in_degree[node] += 1
                
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        order = []
        
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self.model.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
                    
        if len(order) != len(self.model.nodes()):
            raise ValueError("Network contains cycles, cannot perform topological sort")
            
        return order

    def _build_root_cpd(self, node: str, cpt_info: Dict) -> TabularCPD:
        """Build CPD for root node (no parents)"""
        if 'prior' in cpt_info:
            prior_true = float(cpt_info['prior'])
            if not 0 <= prior_true <= 1:
                raise ValueError(f"Prior probability for {node} must be between 0 and 1")
            return TabularCPD(
                variable=node,
                variable_card=2,
                values=[[1 - prior_true], [prior_true]]
            )
        
        # Handle alternative root node formats
        if 'values' in cpt_info:
            return TabularCPD(
                variable=node,
                variable_card=2,
                values=cpt_info['values']
            )
            
        raise ValueError(f"Invalid CPT format for root node {node}")

    def _build_child_cpd(self, node: str, parents: List[str], cpt_info: Dict) -> TabularCPD:
        """Build CPD for child node with parents"""
        evidence_card = [2] * len(parents)
        num_states = 2 ** len(parents)
        state_values = [[], []]  # [False, True] probabilities
        
        # Generate all possible parent state combinations
        for i, parent_states in enumerate(itertools.product([0, 1], repeat=len(parents))):
            # Convert states to string representation used in CPT keys
            state_key = ",".join(str(bool(state)) for state in parent_states)
            
            # Try different key formats
            cpt_entry = cpt_info.get(state_key)
            if not cpt_entry:
                # Try alternative representations
                alt_key = ",".join(str(state) for state in parent_states)
                cpt_entry = cpt_info.get(alt_key, cpt_info.get(str(parent_states)))
            
            if not cpt_entry:
                logger.warning(f"Missing CPT entry for {node} with parent states {state_key}")
                state_values[0].append(0.5)  # P(node=False)
                state_values[1].append(0.5)  # P(node=True)
                continue

            # Handle different CPT value formats
            if isinstance(cpt_entry, dict):
                norm = {str(k).lower(): float(v) for k, v in cpt_entry.items()}
                has_true = "true" in norm
                has_false = "false" in norm
            
                if has_true and has_false:
                    # Use both as-is, normalize
                    total = norm["true"] + norm["false"]
                    p_true = norm["true"] / total
                    p_false = norm["false"] / total
                elif has_true:
                    p_true = norm["true"]
                    p_false = 1.0 - p_true
                elif has_false:
                    p_false = norm["false"]
                    p_true = 1.0 - p_false
                else:
                    p_true = 0.5
                    p_false = 0.5
            elif isinstance(cpt_entry, list) and len(cpt_entry) == 2:
                p_false, p_true = map(float, cpt_entry)
            elif isinstance(cpt_entry, (int, float)):
                p_true = float(cpt_entry)
                p_false = 1.0 - p_true
            else:
                raise ValueError(f"Invalid CPT format for {node} at state {state_key}")

            # Validate probabilities
            if not (0 <= p_false <= 1 and 0 <= p_true <= 1):
                raise ValueError(f"Invalid probabilities for {node}: {p_false}, {p_true}")
                
            # Normalize if needed
            total = p_false + p_true
            if abs(total - 1.0) > 1e-5:
                p_false /= total
                p_true /= total
                
            state_values[0].append(p_false)
            state_values[1].append(p_true)
            
        return TabularCPD(
            variable=node,
            variable_card=2,
            values=state_values,
            evidence=parents,
            evidence_card=evidence_card
        )

    def _build_uniform_cpd(self, node: str, parents: List[str]) -> TabularCPD:
        """Create uniform distribution CPD for missing nodes"""
        if parents:
            evidence_card = [2] * len(parents)
            num_states = 2 ** len(parents)
            values = [[0.5] * num_states, [0.5] * num_states]
            return TabularCPD(
                variable=node,
                variable_card=2,
                values=values,
                evidence=parents,
                evidence_card=evidence_card
            )
        return TabularCPD(
            variable=node,
            variable_card=2,
            values=[[0.5], [0.5]]
        )

    def _validate_model(self) -> bool:
        """Perform comprehensive model validation"""
        # Check model structure
        if not self.model.check_model():
            logger.error("Model structure validation failed")
            return False
            
        # Validate each CPD
        for cpd in self.model.get_cpds():
            if not cpd.is_valid_cpd():
                logger.error(f"Invalid CPD for {cpd.variable}")
                return False
                
            # Get values as flattened array
            values = cpd.values
            flat_values = values.flatten()
            card_node = cpd.variable_card
            num_parent_configs = flat_values.size // card_node
            
            # Reshape to (card_node, num_parent_configs)
            reshaped = flat_values.reshape(card_node, num_parent_configs)
            
            # Check probability sums for each parent configuration
            for i in range(num_parent_configs):
                col = reshaped[:, i]
                prob_sum = float(np.sum(col))
                if abs(prob_sum - 1.0) > 1e-5:
                    logger.error(
                        f"CPD for {cpd.variable} doesn't sum to 1 for parent config index {i}: "
                        f"sum={prob_sum:.5f}, values={col}"
                    )
                    return False
                    
        logger.info("Model validation passed")
        return True

    def query(self, query_variable: str, evidence: Dict[str, bool]) -> float:
        """
        Performs a probabilistic query.

        Args:
            query_variable (str): The variable to query.
            evidence (Dict[str, bool]): Evidence dictionary (variable_name: True/False).

        Returns:
            float: The marginal probability P(query_variable=True | evidence).
        """
        if not self.inference_engine:
            raise RuntimeError("Inference engine not initialized. Build model first.")
        if query_variable not in self.model.nodes():
            raise ValueError(f"Query variable '{query_variable}' not in the model nodes.")

        # Convert boolean evidence to pgmpy's integer state representation (0 for False, 1 for True)
        pgmpy_evidence = {var: (1 if val else 0) for var, val in evidence.items()
                          if var in self.model.nodes()} # Filter out evidence for nodes not in model

        try:
            query_result = self.inference_engine.query(
                variables=[query_variable],
                evidence=pgmpy_evidence
            )
            # query_result is a Factor object. We need P(query_variable=True)
            # For a boolean variable, state 1 usually corresponds to True.
            prob_true = query_result.values[1] # Assuming states are [False, True] -> indices [0, 1]
            return float(prob_true)
        except Exception as e:
            logger.error(f"Error during pgmpy query for {query_variable} with evidence {evidence}: {e}")
            # Fallback or re-raise
            return 0.5 # Default to uncertain on error

    def get_risk_nodes(self) -> List[str]:
        """
        Return a list of nodes considered as 'risk' variables based on heuristics or metadata.
    
        Returns:
            List[str]: Names of nodes that are considered risk-related.
        """
        risk_keywords = ['risk', 'failure', 'loss', 'error', 'hazard', 'threat']
        risk_nodes = []
    
        for node in self.model.nodes():
            if any(kw in node.lower() for kw in risk_keywords):
                risk_nodes.append(node)
    
        return risk_nodes

if __name__ == "__main__":
    print("\n=== Running Pgmpy Bayesian Network ===\n")
    printer.status("TEST", "Starting Pgmpy Bayesian Network tests", "info")

    with open("src/agents/reasoning/networks/bayesian_network.json", "r") as f:
        network_definition = json.load(f)

    pbn = PgmpyBayesianNetwork(network_definition)

    print(pbn)

    print("\n* * * * * Phase 2 * * * * *\n")
    prob = pbn.query("H", {"A": True, "C": False})
    printer.pretty("QUERY P(H | A=True, C=False)", prob, "success")

    print("\n=== All tests completed successfully! ===\n")
