"""
Structural Causal Model Framework
Implements causal graph operations for counterfactual analysis through:
- Directed acyclic graph construction (Pearl, 2009)
- Backdoor/frontdoor adjustment sets (Shpitser et al., 2010)
- Doubly robust estimation (Bang & Robins, 2005)
"""

import statsmodels.formula.api as smf
import networkx as nx
import pandas as pd
import numpy as np
import subprocess
import itertools
import tempfile
import math
import json, os

from scipy.stats import pearsonr, norm
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import GraphicalLasso
from statsmodels.api import OLS
from statsmodels.formula.api import ols
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.sandbox.regression.gmm import IV2SLS # For Instrumental Variables

from src.agents.alignment.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Causal Model")
printer = PrettyPrinter

class CausalGraphBuilder:
    """
    Domain-aware causal structure learning implementing:
    - Constraint-based causal discovery (PC algorithm)
    - Score-based structure optimization (BIC scoring)
    - Confounder detection via latent variable analysis
    """

    def __init__(self):
        self.config = load_global_config()
        self.causal_config = get_config_section('causal_model')
        self.min_adjacency_confidence = self.causal_config.get('min_adjacency_confidence')
        self.max_parents = self.causal_config.get('max_parents')
        self.significance_level = self.causal_config.get('significance_level')
        self.forbidden_edges = self.causal_config.get('forbidden_edges')
        self.required_edges = self.causal_config.get('required_edges')
        self.latent_confounder_detection = self.causal_config.get('latent_confounder_detection', True)
        self.tetrad_path = self.causal_config.get('tetrad_path', '')
        self.fci_max_conditioning_set = self.causal_config.get('fci_max_conditioning_set', 5)
        self.pag = None

        self.graph = nx.DiGraph()
        self.nodes = []
        self.separating_sets = {} # Store separating sets found by PC

        logger.info(f"Causal Graph Builder succesfully initialized")

    def _partial_correlation(self, data: pd.DataFrame, i: str, j: str, conditioning_set: Set[str]) -> Tuple[float, float]:
        """
        Calculates the partial correlation between variables i and j, conditioned on the conditioning_set.
        Uses linear regression to partial out the effect of the conditioning set.
        Returns the partial correlation coefficient and the p-value.
        """
        if not conditioning_set:
            corr, p_value = pearsonr(data[i], data[j])
            return corr, p_value

        # Regress i on conditioning_set
        formula_i = f"{i} ~ {' + '.join(conditioning_set)}"
        model_i = ols(formula_i, data=data).fit()
        residuals_i = model_i.resid

        # Regress j on conditioning_set
        formula_j = f"{j} ~ {' + '.join(conditioning_set)}"
        model_j = ols(formula_j, data=data).fit()
        residuals_j = model_j.resid

        # Correlation of residuals
        corr, p_value = pearsonr(residuals_i, residuals_j)
        return corr, p_value


    def _fisher_z_test(self, data: pd.DataFrame, i: str, j: str, conditioning_set: Set[str]) -> bool:
        r"""
        Performs the Fisher-Z test for conditional independence.
        H0: i and j are independent given conditioning_set.
        Returns True if H0 is accepted (independent), False otherwise.

        The Fisher-Z transformation is $Z = 0.5 * \ln((1 + r) / (1 - r))$, where $r$ is the
        (partial) correlation coefficient.
        The test statistic is $z = Z * \sqrt{n - |S| - 3}$, where $n$ is the sample size
        and $|S|$ is the size of the conditioning set.
        Under H0, $z$ follows a standard normal distribution $N(0, 1)$.
        We reject H0 if $|z| > \Phi^{-1}(1 - \alpha / 2)$, where $\Phi^{-1}$ is the
        inverse CDF of the standard normal distribution and $\alpha$ is the significance level.
        """
        n = len(data)
        k = len(conditioning_set)

        if n <= k + 3:
             # Cannot perform test if sample size is too small relative to conditioning set
             logger.warning(f"Skipping Fisher-Z test for ({i}, {j}) | {conditioning_set}: n <= k + 3")
             return False # Assume dependence if test cannot be performed reliably

        partial_corr, _ = self._partial_correlation(data, i, j, conditioning_set)

        # Fisher-Z transformation
        # Avoid division by zero or log of non-positive number if partial_corr is +/- 1
        if abs(partial_corr) >= 1.0:
            # If correlation is perfect, they are dependent unless conditioning set explains it away perfectly
            # This case is complex; PC usually assumes faithfulness (no perfect correlations canceling out)
            # For simplicity, treat as dependent.
             return False

        z_transform = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr))

        # Test statistic
        test_statistic = abs(z_transform * np.sqrt(n - k - 3))

        # Critical value from standard normal distribution
        critical_value = norm.ppf(1 - self.significance_level / 2)

        # Check for independence
        is_independent = test_statistic < critical_value
        if is_independent:
            # Store the separating set
            self.separating_sets[(i, j)] = conditioning_set
            self.separating_sets[(j, i)] = conditioning_set

        return is_independent


    def construct_graph(self,
                       data: pd.DataFrame,
                       sensitive_attrs: List[str]) -> 'CausalModel':
        """
        Builds domain-constrained causal DAG with:
        1. Feature causal ordering
        2. Confounder identification
        3. Sensitive attribute positioning
        """
        self.nodes = list(data.columns)
        self.graph = nx.Graph() # Start with undirected graph for PC skeleton
        self.graph.add_nodes_from(self.nodes)
        self.separating_sets = {}

        logger.info("Running PC Algorithm Skeleton Phase...")
        self._run_pc_algorithm_skeleton(data)

        logger.info("Running PC Algorithm Orientation Phase...")
        self._run_pc_algorithm_orientation(data) # Orient edges based on v-structures and rules

        # Convert the potentially mixed graph (some undirected edges might remain) to Directed Acyclic Graph
        # A common approach is to break cycles by removing weakest links or using background knowledge.
        # For simplicity here. More robust methods exist.
        self.graph = self._orient_remaining_edges(self.graph)

        # Ensure it's a DAG after orientation
        if not nx.is_directed_acyclic_graph(self.graph):
             logger.warning("Graph contains cycles after PC orientation. Attempting to break cycles.")
             # Implement cycle breaking logic if necessary (e.g., remove edges based on CI test strength)
             # For now, we raise an error or return the graph with cycles noted.
             # A simple strategy: remove edges involved in cycles arbitrarily or based on weak confidence.
             # This part needs careful implementation depending on the desired robustness.
             # Here we just log and proceed, assuming downstream checks handle cycles if needed.
             pass # Placeholder for cycle breaking


        logger.info("Enforcing domain constraints...")
        self._enforce_graph_constraints() # Apply after PC

        logger.info("Optimizing structure with BIC score...")
        self._optimize_structure(data) # Fine-tune based on score

        # Final check for DAG property
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Final graph structure contains cycles after optimization and constraints.")

        logger.info("Detecting potential confounders...")
        # Note: Confounder detection based on graph structure is complex.
        # PC algorithm inherently tries to account for observed confounders.
        # This step might involve looking for specific patterns or using domain knowledge.
        self._detect_confounders(data, sensitive_attrs) # Placeholder

        logger.info("Causal graph construction complete.")
        return CausalModel(self.graph, data)

    def _run_pc_algorithm_skeleton(self, data: pd.DataFrame):
        """
        Constraint-based causal discovery - Skeleton phase.
        Starts with a fully connected undirected graph and removes edges
        based on conditional independence tests.
        """
        node_pairs = list(itertools.combinations(self.nodes, 2))
        # Initialize fully connected undirected graph
        for i, j in node_pairs:
            self.graph.add_edge(i, j)

        l = 0 # Size of conditioning set
        while True:
            edges_removed_in_iteration = False
            logger.info(f"PC Algorithm: Testing conditional independence with set size {l}")
            edges_to_remove = []
            # Iterate over edges present in the graph from the previous iteration
            current_edges = list(self.graph.edges())
            for i, j in current_edges:
                # Get neighbors of i excluding j
                neighbors_i = set(self.graph.neighbors(i)) - {j}
                if len(neighbors_i) >= l:
                    # Iterate through all conditioning sets S of size l from neighbors_i
                    for conditioning_set in itertools.combinations(neighbors_i, l):
                        cond_set = set(conditioning_set)
                        # Perform conditional independence test: Is i _||_ j | S ?
                        if self._fisher_z_test(data, i, j, cond_set):
                            if self.graph.has_edge(i, j):
                                edges_to_remove.append((i, j))
                                edges_removed_in_iteration = True
                                logger.debug(f"Removing edge ({i}, {j}) based on CI test with S={cond_set}")
                                # Store separating set
                                self.separating_sets[(i, j)] = cond_set
                                self.separating_sets[(j, i)] = cond_set
                            break # Found a separating set, no need to check others for this edge

            # Remove edges identified in this iteration
            for u, v in edges_to_remove:
                 # Ensure edge still exists before removing (might be removed by symmetric test)
                 if self.graph.has_edge(u, v):
                     self.graph.remove_edge(u, v)


            l += 1
            # Termination condition: No more edges removed or l exceeds max possible neighbors
            if not edges_removed_in_iteration or l > len(self.nodes) - 2:
                break

    def _run_pc_algorithm_orientation(self, data: pd.DataFrame):
         """
         Orient edges in the skeleton graph based on v-structures and orientation rules.
         Converts the undirected graph to a CPDAG (Completed Partially Directed Acyclic Graph).
         """
         # Create a directed graph copy to store orientations
         oriented_graph = nx.DiGraph()
         oriented_graph.add_nodes_from(self.graph.nodes())

         # 1. Identify v-structures (colliders): i -- k -- j where i and j are non-adjacent,
         #    and k is NOT in the separating set of i and j.
         for k in self.graph.nodes():
             neighbors_k = list(self.graph.neighbors(k))
             if len(neighbors_k) < 2:
                 continue
             for i, j in itertools.combinations(neighbors_k, 2):
                 # Check if i and j are non-adjacent in the skeleton
                 if not self.graph.has_edge(i, j):
                     # Check if k is NOT in the separating set S_ij
                     sep_set_ij = self.separating_sets.get((i, j))
                     if sep_set_ij is None or k not in sep_set_ij:
                         # Orient edges i -> k <- j
                         logger.debug(f"Orienting v-structure: {i} -> {k} <- {j}")
                         if not oriented_graph.has_edge(k, i): # Avoid double edges if already oriented opposite
                             oriented_graph.add_edge(i, k)
                         if not oriented_graph.has_edge(k, j):
                             oriented_graph.add_edge(j, k)

         # 2. Apply orientation rules iteratively until no more edges can be oriented:
         #    Rule R1: If i -> k and k -- j (undirected) and i, j are non-adjacent, orient k -> j. (Avoids new v-structure)
         #    Rule R2: If i -> k -> j, orient i -> j if i -- j is undirected. (Avoids cycle)
         #    Rule R3: If i -- k -> l and i -- j -> l and k -- j, orient k -> j. (Complex, avoids cycle/new v-structure)

         # Start with edges from v-structures
         # Keep track of undirected edges from the skeleton
         undirected_edges = set()
         for u, v in self.graph.edges():
              # If neither u->v nor v->u is in the oriented graph, it's currently undirected
              if not oriented_graph.has_edge(u, v) and not oriented_graph.has_edge(v, u):
                   undirected_edges.add(tuple(sorted((u, v))))


         changed = True
         while changed:
              changed = False
              edges_to_orient = [] # Store orientations found in this pass: (u, v) means orient u -> v

              # Convert set of tuples to list for iteration
              current_undirected = list(undirected_edges)

              for u, v in current_undirected:
                   # Check Rule R1: Search for w such that w -> u and w, v are non-adjacent
                   for w in oriented_graph.predecessors(u):
                        if not self.graph.has_edge(w, v) and not oriented_graph.has_edge(v,w): # Non-adjacent in original skeleton
                             if not oriented_graph.has_edge(v, u): # Ensure u->v is not already oriented
                                  edges_to_orient.append((u, v))
                                  logger.debug(f"Orientation Rule R1: {w} -> {u}, {u}--{v}, {w} not adj {v} => Orient {u} -> {v}")
                                  break # Orient u->v

                   if (u,v) in edges_to_orient or (v,u) in edges_to_orient: continue # Already decided

                   # Check Rule R1 symmetric: Search for w such that w -> v and w, u are non-adjacent
                   for w in oriented_graph.predecessors(v):
                        if not self.graph.has_edge(w, u) and not oriented_graph.has_edge(u,w): # Non-adjacent in original skeleton
                             if not oriented_graph.has_edge(u, v): # Ensure v->u is not already oriented
                                  edges_to_orient.append((v, u))
                                  logger.debug(f"Orientation Rule R1 (sym): {w} -> {v}, {u}--{v}, {w} not adj {u} => Orient {v} -> {u}")
                                  break # Orient v->u

                   if (u,v) in edges_to_orient or (v,u) in edges_to_orient: continue

                   # Check Rule R2: Search for path u -> w -> v
                   for w in oriented_graph.successors(u):
                        if oriented_graph.has_edge(w, v):
                             if not oriented_graph.has_edge(v, u): # Ensure u->v is not already oriented
                                 edges_to_orient.append((u, v))
                                 logger.debug(f"Orientation Rule R2: Path {u} -> {w} -> {v} => Orient {u} -> {v}")
                                 break # Orient u->v

                   if (u,v) in edges_to_orient or (v,u) in edges_to_orient: continue

                   # Check Rule R2 symmetric: Search for path v -> w -> u
                   for w in oriented_graph.successors(v):
                       if oriented_graph.has_edge(w, u):
                           if not oriented_graph.has_edge(u, v): # Ensure v->u is not already oriented
                               edges_to_orient.append((v, u))
                               logger.debug(f"Orientation Rule R2 (sym): Path {v} -> {w} -> {u} => Orient {v} -> {u}")
                               break # Orient v->u

                   if (u,v) in edges_to_orient or (v,u) in edges_to_orient: continue

                   # Check Rule R3: Find common neighbors w1, w2 of u, v such that
                   # w1 -- u -- w2, w1 -> v, w2 -> v and w1, w2 are non-adjacent
                   # This rule is more complex and less commonly implemented/needed. Skipping for brevity.

              # Apply orientations found in this pass
              if edges_to_orient:
                   changed = True
                   for u_orient, v_orient in edges_to_orient:
                        if not oriented_graph.has_edge(v_orient, u_orient): # Avoid creating cycles immediately
                           oriented_graph.add_edge(u_orient, v_orient)
                           # Remove from undirected set
                           undirected_edges.discard(tuple(sorted((u_orient, v_orient))))


         # After rules, add remaining undirected edges (from skeleton) without creating cycles
         # This step depends on the specific PC variant (e.g., PC-Stable handles this differently)
         # For now, we update the main graph attribute with the oriented edges found.
         self.graph = oriented_graph # Replace skeleton with the oriented graph (CPDAG)


    def _orient_remaining_edges(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Attempts to orient remaining undirected edges in the CPDAG to form a DAG.
        This is a placeholder for more sophisticated methods. A simple approach
        might be to orient based on some score or arbitrarily while avoiding cycles.
        """
        # This function needs a robust implementation. For now, it returns the graph as is.
        # A potential strategy: Iterate through undirected edges, tentatively orient one way,
        # check for cycles. If no cycle, keep orientation. If cycle, try other way.
        # If both create cycles, or neither, use another criterion (e.g., BIC score change).
        logger.warning("Orientation of remaining undirected edges is basic. Cycles might persist or orientation might be arbitrary.")
        # Example (very basic, likely insufficient):
        final_graph = graph.copy()
        undirected = [(u, v) for u, v in graph.to_undirected().edges() if not graph.has_edge(u, v) and not graph.has_edge(v, u)]

        for u, v in undirected:
             if not final_graph.has_edge(u,v) and not final_graph.has_edge(v,u):
                 # Try orienting u -> v
                 final_graph.add_edge(u, v)
                 if not nx.is_directed_acyclic_graph(final_graph):
                      final_graph.remove_edge(u, v)
                      # Try orienting v -> u
                      if not final_graph.has_edge(v,u) : # Check if already exists from previous step
                           final_graph.add_edge(v, u)
                           if not nx.is_directed_acyclic_graph(final_graph):
                                final_graph.remove_edge(v, u) # Cannot orient without cycle
                                logger.warning(f"Could not orient edge ({u}, {v}) without creating a cycle.")
                 # If u->v worked or v->u worked, the edge is now oriented.
        return final_graph


    def _enforce_graph_constraints(self):
        """Apply domain-specific structural constraints"""
        if not isinstance(self.graph, nx.DiGraph):
             logger.warning("Graph is not directed before enforcing constraints. Constraints might not apply correctly.")
             return # Or attempt conversion

        for (u, v) in self.forbidden_edges:
            if self.graph.has_edge(u, v):
                logger.info(f"Removing forbidden edge: {u} -> {v}")
                self.graph.remove_edge(u, v)

        for (u, v) in self.required_edges:
            if not self.graph.has_edge(u, v):
                 # Adding required edges might create cycles. Check required.
                 self.graph.add_edge(u, v)
                 if not nx.is_directed_acyclic_graph(self.graph):
                      logger.warning(f"Adding required edge {u} -> {v} created a cycle. Removing it.")
                      self.graph.remove_edge(u, v)
                      # Decide how to handle this conflict - maybe raise error or prioritize constraint vs DAG property
                 else:
                      logger.info(f"Adding required edge: {u} -> {v}")


    def _calculate_bic(self, data: pd.DataFrame, graph: nx.DiGraph) -> float:
        """
        Calculates the Bayesian Information Criterion (BIC) for a given DAG structure.
        BIC = sum over variables [log P(D_i | Pa(G, i), theta_i)] - 0.5 * log(N) * |Params|
        Assuming linear Gaussian models:
        log P(D_i | Pa(G, i), theta_i) = -N/2 * log(2*pi*sigma_i^2) - 1/(2*sigma_i^2) * RSS_i
        BIC = -N/2 * sum(log(sigma_i^2)) - 0.5 * log(N) * K
        Where N is sample size, K is total number of parameters (coefficients + variances).
        sigma_i^2 is the residual variance for variable i.
        """
        N = len(data)
        total_bic = 0
        total_params = 0

        for node in graph.nodes():
            parents = list(graph.predecessors(node))
            num_parents = len(parents)
            num_params_node = num_parents + 1 # Coefficients for parents + intercept/mean
            total_params += num_params_node + 1 # +1 for variance parameter

            if not parents:
                # Node with no parents (exogenous)
                mean = data[node].mean()
                variance = data[node].var(ddof=0) # Use population variance (ddof=0) for BIC consistency
                if variance <= 0: variance = np.finfo(float).eps # Avoid log(0)
                log_likelihood = -N/2 * np.log(2 * np.pi * variance) - N/2
            else:
                # Node with parents (endogenous)
                formula = f"{node} ~ {' + '.join(parents)}"
                try:
                    model = ols(formula, data=data).fit()
                    rss = np.sum(model.resid**2)
                    variance = rss / N # ML estimate of variance
                    if variance <= 0: variance = np.finfo(float).eps # Avoid log(0)
                    # Log-likelihood for Gaussian model (up to constants)
                    log_likelihood = -N/2 * np.log(variance) - N/2 * np.log(2*np.pi) - N/2

                except Exception as e:
                     logger.error(f"Error fitting OLS for BIC calculation on node {node} with parents {parents}: {e}")
                     # Penalize heavily if model fails
                     log_likelihood = -np.inf


            total_bic += log_likelihood

        bic_score = total_bic - 0.5 * np.log(N) * total_params
        # Higher BIC is better (less negative)
        return bic_score


    def _optimize_structure(self, data: pd.DataFrame):
        """
        Score-based structure optimization using greedy search (Hill Climbing) with BIC score.
        Starts from the structure learned by PC (or other initial graph) and iteratively
        adds, deletes, or reverses edges to improve the BIC score, while maintaining acyclicity.
        """
        if not nx.is_directed_acyclic_graph(self.graph):
             logger.warning("Initial graph for BIC optimization contains cycles. Skipping optimization.")
             return

        current_score = self._calculate_bic(data, self.graph)
        logger.info(f"Initial BIC score: {current_score}")

        nodes = list(self.graph.nodes())
        max_iterations = 100 # Limit iterations to prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            best_neighbor_graph = None
            best_neighbor_score = -np.inf # BIC is often negative, seek less negative
            operation_made = None # Track best operation: ('add', u, v), ('delete', u, v), ('reverse', u, v)

            # Consider all possible single-edge modifications
            for u, v in itertools.permutations(nodes, 2):
                 temp_graph = self.graph.copy()
                 operation_type = None

                 # 1. Try adding edge u -> v
                 if not temp_graph.has_edge(u, v) and not temp_graph.has_edge(v, u):
                      temp_graph.add_edge(u, v)
                      if nx.is_directed_acyclic_graph(temp_graph):
                           score = self._calculate_bic(data, temp_graph)
                           if score > best_neighbor_score:
                                best_neighbor_score = score
                                best_neighbor_graph = temp_graph.copy()
                                operation_made = ('add', u, v)
                      # Backtrack: remove the edge for the next potential modification
                      temp_graph.remove_edge(u, v)


                 # 2. Try deleting edge u -> v
                 if self.graph.has_edge(u, v): # Use self.graph to check original edge
                      temp_graph = self.graph.copy() # Start fresh from original graph
                      temp_graph.remove_edge(u, v)
                      # Deleting edges cannot create cycles if original graph was DAG
                      score = self._calculate_bic(data, temp_graph)
                      if score > best_neighbor_score:
                           best_neighbor_score = score
                           best_neighbor_graph = temp_graph.copy()
                           operation_made = ('delete', u, v)

                 # 3. Try reversing edge u -> v
                 if self.graph.has_edge(u, v): # Use self.graph to check original edge
                      temp_graph = self.graph.copy() # Start fresh
                      temp_graph.remove_edge(u, v)
                      temp_graph.add_edge(v, u)
                      if nx.is_directed_acyclic_graph(temp_graph):
                           score = self._calculate_bic(data, temp_graph)
                           if score > best_neighbor_score:
                                best_neighbor_score = score
                                best_neighbor_graph = temp_graph.copy()
                                operation_made = ('reverse', u, v)
                      # No need to backtrack here as we start fresh for each potential reversal

            # Check if the best modification improves the score
            if best_neighbor_score > current_score:
                 logger.info(f"Greedy BIC Step: Applying operation {operation_made} improving score from {current_score:.4f} to {best_neighbor_score:.4f}")
                 self.graph = best_neighbor_graph
                 current_score = best_neighbor_score
                 iteration += 1
            else:
                 logger.info(f"Greedy BIC Optimization: No further improvement found. Final BIC score: {current_score:.4f}")
                 break # Local optimum reached

        if iteration == max_iterations:
            logger.warning("Greedy BIC Optimization reached max iterations.")

    def _run_fci_algorithm(self, data: pd.DataFrame):
        """FCI algorithm implementation with latent variable handling"""
        # Step 1: FCI Skeleton Phase
        logger.info("Running FCI Skeleton Phase...")
        self._run_fci_skeleton(data)
        
        # Step 2: FCI Orientation Phase
        logger.info("Running FCI Orientation Phase...")
        self._run_fci_orientation()
        
        # Step 3: Store PAG
        self.pag = self.graph.copy()

    def _run_fci_skeleton(self, data: pd.DataFrame):
        """FCI skeleton phase with extended conditioning sets"""
        # Initialize complete undirected graph
        self.graph = nx.complete_graph(self.nodes, create_using=nx.Graph())
        l = 0
        
        while l <= self.fci_max_conditioning_set:
            edges_removed = False
            current_edges = list(self.graph.edges())
            
            for i, j in current_edges:
                neighbors_i = set(self.graph.neighbors(i)) - {j}
                neighbors_j = set(self.graph.neighbors(j)) - {i}
                possible_conditioning_sets = set()
                
                # Consider sets from both neighborhoods
                for k in range(0, min(l, len(neighbors_i)) + 1):
                    possible_conditioning_sets |= set(itertools.combinations(neighbors_i, k))
                
                for k in range(0, min(l, len(neighbors_j)) + 1):
                    possible_conditioning_sets |= set(itertools.combinations(neighbors_j, k))
                
                for cond_set in possible_conditioning_sets:
                    cond_set = set(cond_set)
                    if self._fisher_z_test(data, i, j, cond_set):
                        if self.graph.has_edge(i, j):
                            self.graph.remove_edge(i, j)
                            edges_removed = True
                            self.separating_sets[(i, j)] = cond_set
                            self.separating_sets[(j, i)] = cond_set
                            break
            
            if not edges_removed or l >= self.fci_max_conditioning_set:
                break
            l += 1

    def _run_fci_orientation(self):
        """FCI orientation rules for PAG construction"""
        # Initialize PAG with circle marks
        pag = nx.DiGraph()
        for u, v in self.graph.edges():
            pag.add_edge(u, v, mark='o')
            pag.add_edge(v, u, mark='o')
        
        # Rule 0: Orient colliders
        for node in pag.nodes():
            neighbors = list(pag.neighbors(node))
            if len(neighbors) < 2:
                continue
                
            for i, j in itertools.combinations(neighbors, 2):
                if not pag.has_edge(i, j):
                    sep_set = self.separating_sets.get((i, j), set())
                    if node not in sep_set:
                        # Orient i *-> node <-* j
                        pag[i][node]['mark'] = '>' if pag[i][node]['mark'] != '<' else '<'
                        pag[j][node]['mark'] = '>' if pag[j][node]['mark'] != '<' else '<'
        
        # Additional FCI orientation rules would be implemented here
        # (Rules 1-4 for further edge orientation)
        
        # Store orientation marks in graph
        self.graph = pag

    def _detect_confounders(self, data: pd.DataFrame, sensitive_attrs: List[str]):
        """Enhanced confounder detection with FCI and Tetrad integration"""
        if not self.latent_confounder_detection:
            logger.info("Latent confounder detection disabled in config")
            return
        
        # Save the original graph state
        original_graph = self.graph.copy()
        
        try:
            # Option 1: Use Tetrad if available
            if self.tetrad_path and self._run_tetrad_fci(data):
                self._analyze_tetrad_results(sensitive_attrs)
                return
                
            # Option 2: Internal FCI implementation
            self._run_fci_algorithm(data)
            self._analyze_pag(sensitive_attrs)
        except Exception as e:
            logger.error(f"Confounder detection failed: {e}")
        finally:
            # Always restore the original graph after confounder detection
            self.graph = original_graph

    def _run_tetrad_fci(self, data: pd.DataFrame) -> nx.DiGraph:
        tetrad_jar_path = self.tetrad_path
        if not os.path.isfile(tetrad_jar_path):
            logger.error(f"Tetrad JAR not found at path: {tetrad_jar_path}")
            raise FileNotFoundError(f"Tetrad JAR not found: {tetrad_jar_path}")
    
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                data.to_csv(tmp.name, index=False)
                tmp_path = tmp.name
    
            tetrad_cmd = [
                "java", "-jar", tetrad_jar_path,  # this is where the error likely originated
                "--algorithm", "fci",
                "--data", tmp_path
            ]
            logger.info(f"Running Tetrad with command: {' '.join(tetrad_cmd)}")
            result = subprocess.run(tetrad_cmd, capture_output=True, text=True, check=True)
            # ...parse result.stdout into a graph...
            return self._parse_tetrad_output(result.stdout)
    
        except Exception as e:
            logger.error(f"Confounder detection failed: {e}")
            return nx.DiGraph()

    def _analyze_tetrad_results(self, sensitive_attrs: List[str]):
        """Parse Tetrad output for latent confounders"""
        confounders = set()
        graph = self.tetrad_result['graph']
        
        for edge in graph['edges']:
            if edge['endpoint1'] == 'ARROW' and edge['endpoint2'] == 'ARROW':
                i, j = edge['node1'], edge['node2']
                confounders.add((i, j))
                if i in sensitive_attrs or j in sensitive_attrs:
                    logger.warning(f"Tetrad detected latent confounder involving sensitive attribute: {i} <-> {j}")

        self.graph.potential_latent_confounders = confounders
        logger.info(f"Tetrad detected {len(confounders)} potential latent confounders")

    def _analyze_pag(self, sensitive_attrs: List[str]):
        """Analyze PAG for latent confounders"""
        if not self.pag:
            return
            
        confounders = set()
        for u, v, data in self.pag.edges(data=True):
            if data.get('mark', '') == '<' and self.pag[v][u].get('mark', '') == '<':
                confounders.add((u, v))
                if u in sensitive_attrs or v in sensitive_attrs:
                    logger.warning(f"Detected latent confounder involving sensitive attribute: {u} <-> {v}")

        self.graph.potential_latent_confounders = confounders
        logger.info(f"Detected {len(confounders)} potential latent confounders")

    def _estimate_inverse_covariance(self, data: pd.DataFrame):
        """Estimate sparse inverse covariance for FCI tests"""
        try:
            cov_matrix = data.cov().values
            model = GraphicalLasso()
            model.fit(cov_matrix)
            return model.precision_
        except Exception as e:
            logger.error(f"Inverse covariance estimation failed: {str(e)}")
            return None

    # Update conditional independence test for FCI
    def _partial_correlation(self, data: pd.DataFrame, i: str, j: str, conditioning_set: Set[str]) -> Tuple[float, float]:
        """Enhanced with inverse covariance matrix option"""
        n = len(data)
        
        if self.causal_config.get('use_inverse_covariance', False) and n > 50:
            precision_matrix = self._estimate_inverse_covariance(data)
            if precision_matrix is not None:
                idx_i = data.columns.get_loc(i)
                idx_j = data.columns.get_loc(j)
                cond_indices = [data.columns.get_loc(c) for c in conditioning_set if c in data.columns]
                
                # Calculate partial correlation using precision matrix
                p_corr = -precision_matrix[idx_i, idx_j] / math.sqrt(
                    precision_matrix[idx_i, idx_i] * precision_matrix[idx_j, idx_j])
                
                # Fisher Z-transform for p-value
                n = len(data)
                z = 0.5 * math.log((1 + p_corr) / (1 - p_corr))
                se = 1 / math.sqrt(n - len(conditioning_set) - 3)
                z_score = abs(z / se)
                p_value = 2 * (1 - norm.cdf(z_score))
                return p_corr, p_value
            
        else:
            # Fallback to original OLS method
            return super()._partial_correlation(data, i, j, conditioning_set)


# ==================================================
# Causal Model Class (includes IV, Backdoor, etc.)
# ==================================================
class CausalModel:
    """
    Structural Causal Model implementing:
    - Potential outcome estimation
    - Backdoor adjustment
    - Counterfactual inference
    - Instrumental Variable estimation
    """

    def __init__(self, graph: nx.DiGraph, data: pd.DataFrame):
        if not isinstance(graph, nx.DiGraph):
             raise TypeError("Input graph must be a NetworkX DiGraph.")
        if not nx.is_directed_acyclic_graph(graph):
            # Attempt basic cycle breaking or raise error
             logger.warning("Input graph contains cycles. Attempting to resolve or raise error.")
             # For now, raise error. A robust solution would integrate cycle breaking earlier.
             raise ValueError("Graph must be a directed acyclic graph (DAG).")

        self.graph = graph
        # Ensure data contains all nodes in the graph
        missing_nodes = set(graph.nodes()) - set(data.columns)
        if missing_nodes:
            raise ValueError(f"Data is missing columns for nodes: {missing_nodes}")
        self.data = data.copy() # Work with a copy
        self.nodes = list(graph.nodes())
        self._validate_graph() # Redundant check, but good practice
        # Estimate SEM equations only if needed, can be computationally expensive
        self.structural_equations = None # Lazy initialization: self._estimate_structural_equations()

    def _validate_graph(self):
        """Ensure graph is a valid DAG"""
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Graph must be a directed acyclic graph")

    def _get_structural_equations(self) -> Dict[str, RegressionResultsWrapper]:
        """Estimate and return structural equations (memoized)."""
        if self.structural_equations is None:
            logger.info("Estimating structural equations...")
            equations = {}
            # Sort nodes topologically to ensure parents are processed before children if needed
            # Although OLS fitting doesn't strictly require this order here.
            try:
                sorted_nodes = list(nx.topological_sort(self.graph))
            except nx.NetworkXUnfeasible:
                raise ValueError("Graph contains cycles, cannot topologically sort for SEM estimation.")

            for var in sorted_nodes:
                parents = list(self.graph.predecessors(var))
                if not parents:
                    # Exogenous: Model might be just mean/variance, or skip if not predicting
                    # Store mean for simple baseline prediction if needed later
                    equations[var] = {'mean': self.data[var].mean(), 'type': 'exogenous'}
                    continue
                try:
                    X = self.data[parents]
                    X = add_constant(X, has_constant='add')
                    y = self.data[var]
                    # Handle missing data by dropping NaNs
                    valid_idx = X.dropna().index.intersection(y.dropna().index)
                    if len(valid_idx) == 0:
                        logger.warning(f"No valid data for {var} ~ {parents}. Skipping.")
                        continue
                    model = OLS(y.loc[valid_idx], X.loc[valid_idx]).fit()
                    equations[var] = model
                    if not pd.api.types.is_numeric_dtype(self.data[var]):
                        logger.warning(f"Outcome variable '{var}' for SEM is not numeric. Skipping equation estimation.")
                        continue
                    for p in parents:
                        if not pd.api.types.is_numeric_dtype(self.data[p]):
                            logger.warning(f"Parent variable '{p}' for node '{var}' is not numeric. Skipping equation estimation for '{var}'.")
                            raise TypeError("Non-numeric parent") # Break inner loop

                    # Check for sufficient data variance after NaN removal by OLS
                    if X.dropna().shape[0] < 2 or X.dropna().shape[1] + 1 > X.dropna().shape[0] : # N < k+1
                        logger.warning(f"Insufficient data points or high collinearity for OLS on {var} ~ {parents}. Skipping.")
                        continue
                    if y.loc[X.dropna().index].var() < 1e-9:
                        logger.warning(f"Outcome variable {var} has near-zero variance for OLS. Skipping.")
                        continue

                    model = OLS(y.loc[valid_idx], X.loc[valid_idx]).fit()
                    equations[var] = model # Store the fitted model object
                except TypeError: # Catch non-numeric parent error
                    continue # Skip this equation
                except Exception as e:
                    logger.error(f"Failed to estimate structural equation for {var} with parents {parents}: {e}")
                    # Optionally store None or raise error depending on desired robustness
                    equations[var] = None # Mark as failed

            self.structural_equations = equations
            logger.info("Structural equations estimated.")
        return self.structural_equations

    def estimate_effect(self,
                      treatment: str,
                      outcome: str,
                      method: str = "backdoor.linear_regression",
                      data: Optional[pd.DataFrame] = None,
                      instrument: Optional[str] = None) -> Union[pd.Series, float, None]:
        """
        Causal effect estimation using various methods.

        Args:
            treatment (str): Name of the treatment variable.
            outcome (str): Name of the outcome variable.
            method (str): Estimation method ('backdoor.linear_regression',
                          'backdoor.doubly_robust', 'iv.2sls', 'frontdoor.<method>').
            data (pd.DataFrame, optional): Data to use for estimation. Defaults to self.data.
            instrument (str, optional): Instrument variable name, required for 'iv' method.

        Returns:
            pd.Series or float or None: Effect estimate and stats (Series),
                                       just the estimate (float), or None if estimation fails.
                                       Format depends on the method.
        """
        if data is None:
            data = self.data

        # Validate inputs
        if treatment not in self.graph: raise ValueError(f"Treatment '{treatment}' not in graph.")
        if outcome not in self.graph: raise ValueError(f"Outcome '{outcome}' not in graph.")
        if treatment == outcome: raise ValueError("Treatment and outcome must be different.")

        if method.startswith("backdoor"):
            logger.info(f"Estimating ATE({treatment} -> {outcome}) using Backdoor Adjustment ({method})")
            return self._backdoor_adjustment(data, treatment, outcome, method)
        elif method.startswith("iv"):
            logger.info(f"Estimating ATE({treatment} -> {outcome}) using Instrumental Variable ({method})")
            if instrument is None:
                raise ValueError("Instrument variable must be provided for 'iv' method.")
            if instrument not in self.graph:
                raise ValueError(f"Instrument '{instrument}' not in graph.")
            # Allow different IV estimators, e.g., 'iv.2sls'
            iv_estimator = method.split('.')[-1] if '.' in method else '2sls' # Default to 2SLS
            return self._instrumental_variables(data, treatment, outcome, instrument, estimator=iv_estimator)
        elif method.startswith("frontdoor"):
             logger.info(f"Estimating ATE({treatment} -> {outcome}) using Frontdoor Adjustment ({method})")
             # Implementation requires finding mediating variables and applying the frontdoor formula
             # Placeholder: return self._frontdoor_adjustment(data, treatment, outcome, method)
             raise NotImplementedError("Frontdoor adjustment not yet implemented.")
        else:
            raise ValueError(f"Unknown estimation method: {method}")

    # --- Backdoor Adjustment Methods ---

    def _find_backdoor_paths(self, source: str, target: str) -> List[List[str]]:
        """
        Identifies all backdoor paths between source (treatment) and target (outcome).
        A backdoor path is a path from source to target that starts with an edge pointing
        into source (e.g., U -> source ... target, where U is unobserved or observed).
        In a DAG context, it means any path between source and target that is not a directed path
        from source to target, specifically paths containing an arrow pointing into source.

        Algorithm:
        1. Find all paths between source and target in the graph where edge directions are ignored (undirected graph).
        2. For each path, check if it's a backdoor path: it must contain an edge pointing into `source`.
           Specifically, the first edge on the path starting from `source` must be `X <- source`.

        Refined Definition: A path $p$ between $X$ (treatment) and $Y$ (outcome) is a backdoor path
        if it contains an arrow pointing into $X$. That is, $p = (..., W, X, ..., Y)$ where the
        edge between $W$ and $X$ is $W \to X$.

        Implementation Detail: We can find all paths in the undirected version and then filter.
        Or, more directly, search backwards from X. Any path from a node Z to Y, where Z is a parent of X,
        and the path does not go through X itself (except at Z->X), constitutes part of a backdoor path.
        Let's use the simpler definition check on all undirected paths first.
        """
        backdoor_paths = []
        undirected_graph = self.graph.to_undirected()

        # Check if source and target are connected at all
        if not nx.has_path(undirected_graph, source, target):
             return []

        # Limit path length reasonably? No, find all simple paths.
        for path in nx.all_simple_paths(undirected_graph, source=source, target=target):
            if len(path) >= 2:
                # Check the first step away from source: is it source -> path[1] or source <- path[1]?
                second_node = path[1]
                # If the directed graph has an edge pointing INTO source from the second node, it's a backdoor path.
                if self.graph.has_edge(second_node, source):
                    # We need to be careful here. A path is backdoor if it's not causal *and* creates confounding.
                    # Pearl's definition: A path containing an arrow into X.
                    # Let's stick to that: if the path starts X <- W ... Y
                     backdoor_paths.append(path)

        logger.debug(f"Found {len(backdoor_paths)} potential undirected paths starting with arrow into '{source}'.")
        # Further filter: Ensure path doesn't contain colliders that are blocked by conditioning? This is handled by adjustment set finding.

        # Alternative check: A path is backdoor if it doesn't start with source -> ...
        # This might be too broad. Let's stick to the "arrow into source" definition.

        return backdoor_paths # Return the list of nodes in each path

    def _is_blocked(self, path: List[str], conditioning_set: Set[str]) -> bool:
         r"""
         Checks if a path is blocked by a given conditioning set based on d-separation rules.
         A path is blocked if:
         1. It contains a chain X -> M -> Y where M is in the conditioning set.
         2. It contains a fork X <- M -> Y where M is in the conditioning set.
         3. It contains a collider X -> M <- Y where M is NOT in the conditioning set,
            and none of M's descendants are in the conditioning set.
         """
         if len(path) < 3: # Paths of length 0 or 1 are always blocked (or non-existent)
             return True

         for i in range(len(path) - 2):
             u, m, v = path[i], path[i+1], path[i+2]

             is_chain = self.graph.has_edge(u, m) and self.graph.has_edge(m, v)
             is_fork = self.graph.has_edge(m, u) and self.graph.has_edge(m, v)
             is_collider = self.graph.has_edge(u, m) and self.graph.has_edge(v, m)

             if (is_chain or is_fork) and m in conditioning_set:
                 return True # Blocked by chain or fork

             if is_collider:
                  # Check if the collider M or any of its descendants are in the conditioning set
                  descendants_m = nx.descendants(self.graph, m) | {m}
                  if not descendants_m.intersection(conditioning_set):
                       return True # Path is blocked by collider *not* conditioned on

         return False # Path is not blocked

    def _find_minimal_adjustment_set(self, treatment: str, outcome: str) -> Set[str]:
        r"""
        Identifies a minimal valid backdoor adjustment set using Pearl's backdoor criterion.
        A set Z satisfies the backdoor criterion relative to (X, Y) if:
        1. No node in Z is a descendant of X.
        2. Z blocks every backdoor path between X and Y (paths with an arrow into X).

        Algorithm (based on common heuristics, not guaranteed minimal in all cases but often sufficient):
        1. Find all backdoor paths from X to Y.
        2. Identify all nodes involved in these paths (excluding X and Y).
        3. A potential adjustment set could be the parents of X (if they aren't descendants of X).
           More generally, find a set Z that d-separates X from Y in G[V \ {Descendants(X)}]
           Alternatively, consider nodes on backdoor paths.

        A common constructive approach:
        1. Start with the set of parents of X: Pa(X).
        2. Remove any descendants of X from Pa(X).
        3. Add other nodes necessary to block remaining backdoor paths, typically ancestors of X or Y
           that are not descendants of X.

        Simpler approach (often works for DAGs from PC): Use parents of X.
        Pa(X) often satisfies the backdoor criterion if no element of Pa(X) is a descendant of X
        (which is true in a DAG if Pa(X) are direct parents).
        However, Pa(X) might not be minimal or block all paths if there's complex structure.

        Let's implement a more general method:
        Find all backdoor paths. Find all nodes on these paths. Try to find a minimal set of these nodes
        that blocks all paths, excluding descendants of X.
        Consider Ancestors(X union Y). Find subset Z of Ancestors that blocks paths.

        Let's try the Parent-based approach first, common in practice:
        """
        parents_of_treatment = set(self.graph.predecessors(treatment))

        # In a DAG, parents are not descendants. So condition 1 is met.
        # Now check if Pa(X) blocks all backdoor paths.
        adjustment_set = parents_of_treatment

        # Verify this set blocks all backdoor paths
        backdoor_paths = self._find_backdoor_paths(treatment, outcome)
        unblocked_paths = []
        for path in backdoor_paths:
            if not self._is_blocked(path, adjustment_set):
                 unblocked_paths.append(path)


        if not unblocked_paths:
             logger.info(f"Parents of treatment {parents_of_treatment} form a valid adjustment set.")
             return adjustment_set
        else:
             logger.warning(f"Parents of {treatment} do not block all backdoor paths: {unblocked_paths}. Need a more complex adjustment set finding method.")
             # Fallback / More Advanced Method Needed:
             # A more robust approach involves graph surgery (e.g., graph G_alpha removing outgoing edges from X)
             # and finding d-separation sets. Or analyzing paths directly.
             # Placeholder: Return parents, acknowledging limitation. Or implement full criterion.
             # For now, returning parents with a warning.
             # Consider implementing Shpitser's adjustment set algorithm if needed.
             return adjustment_set # Return parents as a common heuristic, but warn user.


    def _linear_adjustment(self,
                          data: pd.DataFrame,
                          treatment: str,
                          outcome: str,
                          adjustment_set: Set[str]) -> pd.Series:
        """Standard regression adjustment using statsmodels OLS."""
        if not adjustment_set and not self.graph.predecessors(treatment):
             logger.info("No adjustment set needed as treatment has no parents influencing it via backdoor paths.")
             formula = f"{outcome} ~ {treatment}"
        elif not adjustment_set and self.graph.predecessors(treatment):
             logger.warning("Adjustment set is empty, but treatment has parents. Backdoor paths might not be blocked.")
             formula = f"{outcome} ~ {treatment}" # Proceeding without adjustment
        else:
            # Ensure all adjustment variables are in the data
            missing_adj = adjustment_set - set(data.columns)
            if missing_adj:
                raise ValueError(f"Adjustment variables {missing_adj} not found in provided data.")
            # Filter out any adjustment variables that are the outcome itself (shouldn't happen with valid set)
            valid_adj_vars = list(adjustment_set - {outcome})
            # Construct the regression formula: Y ~ T + Z1 + Z2 + ...
            formula = f"{outcome} ~ {treatment}"
            if valid_adj_vars:
                formula += " + " + " + ".join(valid_adj_vars)

        logger.info(f"Fitting linear adjustment model: {formula}")
        try:
            # Fit OLS model using the provided data
            model = ols(formula, data=data, missing='drop').fit()

            # Extract treatment effect (coefficient of treatment variable)
            # Handle case where treatment variable might be categorical/transformed by patsy
            treatment_term = next((term for term in model.params.index if term.startswith(f"`{treatment}`")), None)
            if treatment_term is None:
                 # Try without backticks if lookup failed
                 treatment_term = next((term for term in model.params.index if term == treatment), None)

            if treatment_term is None:
                 logger.error(f"Treatment term '{treatment}' not found in model results. Terms: {model.params.index}")
                 # This might happen if treatment is perfectly collinear or has zero variance after NA drop
                 return pd.Series({
                     'effect': np.nan, 'std_error': np.nan, 'p_value': np.nan,
                     'ci_lower': np.nan, 'ci_upper': np.nan, 'n_obs': model.nobs, 'formula': formula
                 })


            effect = model.params[treatment_term]
            stderr = model.bse[treatment_term]
            p_value = model.pvalues[treatment_term]
            conf_int = model.conf_int().loc[treatment_term]

            # Return results as a Series
            return pd.Series({
                'effect': effect,
                'std_error': stderr,
                'p_value': p_value,
                'ci_lower': conf_int[0],
                'ci_upper': conf_int[1],
                'n_obs': model.nobs,
                'formula': formula # Include formula for inspection
            })
        except Exception as e:
             logger.error(f"Linear adjustment failed for formula '{formula}': {e}")
             # Return NaNs or raise error
             return pd.Series({
                 'effect': np.nan, 'std_error': np.nan, 'p_value': np.nan,
                 'ci_lower': np.nan, 'ci_upper': np.nan, 'n_obs': np.nan, 'formula': formula
             })


    def _doubly_robust_estimation(self,
                                 data: pd.DataFrame,
                                 treatment: str,
                                 outcome: str,
                                 adjustment_set: Set[str]) -> pd.Series:
        """
        Doubly robust estimator combining propensity score and outcome models.
        Provides unbiased estimate if *either* the propensity model *or* the outcome model is correctly specified.

        Formula (for binary treatment T):
        ATE = E[ (T * Y / PS(Z)) - ((1 - T) * Y / (1 - PS(Z))) ]
            + E[ (1 - T/PS(Z)) * E[Y | T=1, Z] ]
            + E[ (1 - (1-T)/(1-PS(Z))) * E[Y | T=0, Z] ]

        Simplified ATE estimation:
        E[Y(1)] = E[ T*Y/PS(Z) + (1 - T/PS(Z))*mu1(Z) ]
        E[Y(0)] = E[ (1-T)*Y/(1-PS(Z)) + (1 - (1-T)/(1-PS(Z)))*mu0(Z) ]
        ATE = E[Y(1)] - E[Y(0)]

        Where:
        - Y is the outcome.
        - T is the binary treatment (must be 0 or 1).
        - Z is the adjustment set.
        - PS(Z) = P(T=1 | Z) is the propensity score.
        - mu1(Z) = E[Y | T=1, Z] is the expected outcome under treatment, given Z.
        - mu0(Z) = E[Y | T=0, Z] is the expected outcome under control, given Z.
        """
        if not adjustment_set:
             logger.warning("Doubly robust estimation called with empty adjustment set. Results may be biased if confounding exists.")
             Z = pd.DataFrame(index=data.index) # Empty dataframe for covariates
        else:
             # Ensure adjustment set is valid
             missing_adj = adjustment_set - set(data.columns)
             if missing_adj: raise ValueError(f"Adjustment variables {missing_adj} not found in data.")
             Z = data[list(adjustment_set)]

        T = data[treatment]
        Y = data[outcome]

        # Check if treatment is binary (0 or 1)
        if not T.isin([0, 1]).all():
            raise ValueError("Doubly robust estimation currently requires a binary treatment variable (0 or 1).")

        logger.info("Fitting propensity score model (Logistic Regression)...")
        # Propensity score model P(T=1 | Z)
        # Add constant for Logistic Regression if Z is not empty
        Z_ps = add_constant(Z, has_constant='add') if not Z.empty else Z
        ps_model = LogisticRegression(solver='liblinear', C=1e6) # High C = less regularization
        try:
            ps_model.fit(Z_ps, T)
            # Predict probabilities P(T=1 | Z)
            propensity_scores = ps_model.predict_proba(Z_ps)[:, 1]
            # Clip scores to avoid division by zero/instability
            propensity_scores = np.clip(propensity_scores, 1e-6, 1 - 1e-6)
            logger.info("Propensity score model fitted.")
        except Exception as e:
             logger.error(f"Propensity score model fitting failed: {e}")
             return pd.Series({'effect': np.nan, 'std_error': np.nan}) # Cannot proceed


        logger.info("Fitting outcome models (Gradient Boosting Regressor)...")
        # Outcome model E[Y | T, Z]
        # We need E[Y | T=1, Z] (mu1) and E[Y | T=0, Z] (mu0)

        # Prepare data for outcome models
        XZ = Z.assign(**{treatment: T}) # Combine Z and T
        XZ_1 = Z.assign(**{treatment: 1}) # Data if everyone was treated
        XZ_0 = Z.assign(**{treatment: 0}) # Data if everyone was control

        # Use Gradient Boosting Regressor (or another flexible model)
        outcome_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        try:
            outcome_model.fit(XZ, Y)
            mu_hat = outcome_model.predict(XZ) # Predicted outcome E[Y|T,Z]
            mu1_hat = outcome_model.predict(XZ_1) # Predicted E[Y|T=1,Z] for all units
            mu0_hat = outcome_model.predict(XZ_0) # Predicted E[Y|T=0,Z] for all units
            logger.info("Outcome model fitted.")
        except Exception as e:
             logger.error(f"Outcome model fitting failed: {e}")
             return pd.Series({'effect': np.nan, 'std_error': np.nan}) # Cannot proceed


        # Calculate components for DR estimator
        dr_y1 = (T * Y / propensity_scores) + (1 - T / propensity_scores) * mu1_hat
        dr_y0 = ((1 - T) * Y / (1 - propensity_scores)) + (1 - (1 - T) / (1 - propensity_scores)) * mu0_hat

        # Estimate ATE
        ate_dr = np.mean(dr_y1 - dr_y0)

        # Estimate standard error (using influence functions or bootstrap)
        # Influence function approach:
        n = len(data)
        psi_dr = (dr_y1 - dr_y0) - ate_dr
        var_dr = np.mean(psi_dr**2) / n
        se_dr = np.sqrt(var_dr)

        logger.info(f"Doubly Robust ATE estimate: {ate_dr:.4f} (SE: {se_dr:.4f})")

        return pd.Series({
            'effect': ate_dr,
            'std_error': se_dr
            # P-value and CI can be calculated from estimate and SE assuming normality
        })

    def _backdoor_adjustment(self,
                            data: pd.DataFrame,
                            treatment: str,
                            outcome: str,
                            method: str) -> pd.Series:
        """Helper to route backdoor adjustments."""
        # Find valid adjustment set based on the graph structure
        try:
             adjustment_set = self._find_minimal_adjustment_set(treatment, outcome)
             logger.info(f"Identified adjustment set for ({treatment}, {outcome}): {adjustment_set}")
        except Exception as e:
             logger.error(f"Failed to find adjustment set for ({treatment}, {outcome}): {e}")
             # Return NaN series indicating failure
             return pd.Series({'effect': np.nan, 'std_error': np.nan, 'p_value': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan})


        if "linear_regression" in method:
            return self._linear_adjustment(data, treatment, outcome, adjustment_set)
        elif "doubly_robust" in method:
            return self._doubly_robust_estimation(data, treatment, outcome, adjustment_set)
        else:
            raise ValueError(f"Unknown backdoor method specified: {method}")

    # --- Instrumental Variable (IV) Methods ---
    def _check_iv_conditions(self, instrument: str, treatment: str, outcome: str) -> Dict[str, bool]:
        conditions = {
            'relevance_path': False,
            'exclusion': False,
            'independence': False
        }
        
        # 1. Relevance: Check for active paths between Z and X
        if nx.has_path(self.graph.to_undirected(), instrument, treatment):
            conditions['relevance_path'] = True
        
        # Create modified graph (remove outgoing edges from X)
        graph_X = self.graph.copy()
        for child in list(graph_X.successors(treatment)):
            graph_X.remove_edge(treatment, child)
        
        # 2. Exclusion: No directed paths from Z to Y in modified graph
        conditions['exclusion'] = not nx.has_path(graph_X, instrument, outcome)
        
        # 3. Independence: d-separation in modified graph
        conditions['independence'] = self._are_d_separated(
            graph_X, instrument, outcome, set()
        )
        return conditions
    
    def _are_d_separated(self, graph, node1, node2, conditioning_set) -> bool:
        """Check if node1 and node2 are d-separated given conditioning_set."""
        if node1 not in graph or node2 not in graph:
            return True
        
        # Check connectivity
        if not nx.has_path(graph.to_undirected(), node1, node2):
            return True
        
        # Examine all simple paths
        try:
            paths = nx.all_simple_paths(graph.to_undirected(), node1, node2)
        except nx.NodeNotFound:
            return True
        
        for path in paths:
            if not self._is_blocked_path(graph, path, conditioning_set):
                return False  # Active path exists → d-connected
        return True  # All paths blocked
    
    def _is_blocked_path(self, graph, path: List[str], conditioning_set: Set[str]) -> bool:
        """Check if a path is blocked by conditioning_set in a given graph."""
        if len(path) < 3:
            return True  # Trivially blocked
        
        for i in range(len(path) - 2):
            u, m, v = path[i], path[i+1], path[i+2]
            
            # Chain (→ m →) or Fork (← m →)
            if (graph.has_edge(u, m) and graph.has_edge(m, v)) or \
               (graph.has_edge(m, u) and graph.has_edge(m, v)):
                if m in conditioning_set:
                    return True  # Blocked by conditioning
            
            # Collider (→ m ←)
            elif graph.has_edge(u, m) and graph.has_edge(v, m):
                descendants = nx.descendants(graph, m) | {m}
                if not descendants & conditioning_set:  # No conditioning on collider/descendants
                    return True  # Path blocked
        return False  # Path not blocked
    
    # Update existing method to use new path-based check
    def _is_blocked(self, path: List[str], conditioning_set: Set[str]) -> bool:
        return self._is_blocked_path(self.graph, path, conditioning_set)

    def _instrumental_variables(self,
                               data: pd.DataFrame,
                               treatment: str,
                               outcome: str,
                               instrument: str,
                               estimator: str = '2sls') -> Union[float, pd.Series, None]:
        r"""
        Estimate causal effect using Instrumental Variables (IV).
        Requires a valid instrument satisfying relevance, exclusion, and independence.

        Common method: Two-Stage Least Squares (2SLS) for linear models.
        Stage 1: Regress treatment (X) on instrument (Z) and exogenous covariates (W):
                 $X = \delta_0 + \delta_1 Z + \delta_2 W + \epsilon$
                 Get predicted treatment: $\hat{X}$
        Stage 2: Regress outcome (Y) on predicted treatment ($\hat{X}$) and exogenous covariates (W):
                 $Y = \beta_0 + \beta_{IV} \hat{X} + \beta_2 W + \nu$
                 $\beta_{IV}$ is the IV estimate of the causal effect.

        Args:
            data (pd.DataFrame): Data for estimation.
            treatment (str): Endogenous treatment variable X.
            outcome (str): Outcome variable Y.
            instrument (str): Instrument variable Z.
            estimator (str): Specific IV estimator ('2sls').

        Returns:
             float or pd.Series or None: IV estimate of the effect (float), or Series with stats, or None if failed.
        """
        # 1. Check graphical conditions (optional but recommended)
        iv_conditions = self._check_iv_conditions(instrument, treatment, outcome)
        if not all(iv_conditions.values()):
            logger.warning(f"Instrument '{instrument}' may not satisfy all graphical conditions for IV estimation of {treatment} -> {outcome}. Proceeding with caution.")
            # Depending on severity, could return None or raise error.

        # 2. Identify exogenous covariates (W) to include in both stages.
        #    These are typically variables that affect Y but are not affected by T.
        #    In simplest case, W is empty. Often includes confounders Z-Y or X-Y if instrument is conditional.
        #    For simplicity, let's assume no *additional* covariates W for now. Advanced implementations would identify these from the graph.
        exog_covariates = [] # Placeholder: Identify based on graph if needed.

        # 3. Perform 2SLS estimation
        if estimator == '2sls':
            try:
                logger.info(f"Performing 2SLS: Stage 1 ({treatment} ~ {instrument}), Stage 2 ({outcome} ~ predicted {treatment})")
                # Prepare data for statsmodels IV2SLS
                # We need to specify the endogenous regressor (Treatment X) separately.

                endog = data[outcome]
                # Exogenous variables in the *structural* equation (Stage 2) - typically just intercept if no W
                exog_structural = pd.DataFrame({'Intercept': np.ones(len(data))})
                if exog_covariates:
                    exog_structural[exog_covariates] = data[exog_covariates]

                 # Endogenous variable(s) in the structural equation - the treatment X
                endog_regressor = data[[treatment]] # Needs to be DataFrame

                # Instruments = Instrument Z + any exogenous vars W included above
                instruments = data[[instrument]]
                if exog_covariates:
                    instruments[exog_covariates] = data[exog_covariates]
                # Add constant to instruments as well if including intercept in structural
                instruments_with_const = add_constant(instruments, has_constant='add')


                # Use statsmodels IV2SLS
                iv_model = IV2SLS(
                    endog=endog, 
                    exog=exog_structural.join(endog_regressor),  # Combine exogenous variables and treatment
                    instrument=instruments_with_const
                ).fit()

                logger.info(iv_model.summary())

                # Extract the coefficient for the treatment variable
                iv_effect = iv_model.params[treatment]
                iv_stderr = iv_model.bse[treatment]
                iv_pvalue = iv_model.pvalues[treatment]
                iv_conf_int = iv_model.conf_int().loc[treatment]

                return pd.Series({
                    'effect': iv_effect,
                    'std_error': iv_stderr,
                    'p_value': iv_pvalue,
                    'ci_lower': iv_conf_int[0],
                    'ci_upper': iv_conf_int[1],
                    'n_obs': iv_model.nobs,
                    'method': 'IV-2SLS',
                    'instrument': instrument
                })

            except Exception as e:
                logger.error(f"IV 2SLS estimation failed: {e}")
                # Check for common issues: weak instrument (low F-stat in stage 1), collinearity
                # Could try running stage 1 manually to check instrument strength.
                return None # Indicate failure
        else:
            raise NotImplementedError(f"IV estimator '{estimator}' not implemented.")


    # --- Counterfactual Methods ---

    def compute_counterfactual(self,
                             intervention: Dict[str, Union[float, int]],
                             observed_data_point: Optional[pd.Series] = None,
                             method: str = 'adjust') -> Union[pd.Series, pd.DataFrame]:
        r"""
        Compute counterfactual outcome(s) under a specific intervention using Pearl's 3-step process (Abduction, Action, Prediction)
        or simpler adjustment if structural equations are known.

        Args:
            intervention (Dict[str, Union[float, int]]): Dictionary specifying variables to intervene on and their fixed values. E.g., {'treatment': 1}.
            observed_data_point (pd.Series, optional): A single row of observed data representing the unit for which the counterfactual is computed. If None, computes expected counterfactual over the population.
            method (str): 'adjust' (uses SEM) or 'pearl3step' (requires modeling noise). Default 'adjust'.

        Returns:
            pd.Series or pd.DataFrame: Counterfactual outcomes. If observed_data_point is given, returns a Series with counterfactual values for that unit.
                                      Otherwise, returns a DataFrame representing the expected population distribution under intervention.
        """
        if method == 'adjust':
            # Simplified method: Modify graph (remove incoming to intervened), re-estimate/predict.
            # Assumes SEMs capture the essential structure.
            logger.info(f"Computing counterfactual using structural equation adjustment for intervention: {intervention}")

            # Ensure SEMs are estimated
            sems = self._get_structural_equations()
            if sems is None:
                raise RuntimeError("Structural equations must be estimated before computing counterfactuals with 'adjust' method.")


            if observed_data_point is not None:
                # Compute for a specific individual (requires abduction step for noise terms in full Pearl method)
                # Adjustment method is simpler: just substitute and predict down the chain.
                cf_data = observed_data_point.copy().to_frame().T # Make it DataFrame-like
                cf_data.index = ['counterfactual']
            else:
                # Compute expected counterfactual for the population
                cf_data = self.data.copy()


            # Process variables in topological order for prediction
            try:
                sorted_nodes = list(nx.topological_sort(self.graph))
            except nx.NetworkXUnfeasible:
                raise ValueError("Graph contains cycles, cannot topologically sort for counterfactual prediction.")


            for var in sorted_nodes:
                if var in intervention:
                    # Action: Set variable to intervened value
                    cf_data[var] = intervention[var]
                else:
                    # Prediction: Use SEM for non-intervened variables based on their parents' current/counterfactual values
                    parents = list(self.graph.predecessors(var))
                    model_info = sems.get(var)
                    # Handle exogenous/unmodeled vars first
                    is_exogenous_or_unmodeled = (
                        not parents or model_info is None or (isinstance(model_info, dict) and model_info.get('type') == 'exogenous'))
                    if is_exogenous_or_unmodeled: # Exogenous variable not intervened on keeps its original value(s)
                        if observed_data_point is not None:
                            if var in observed_data_point:
                                cf_data[var] = observed_data_point[var]
                        continue

                    if isinstance(model_info, RegressionResultsWrapper):
                        # Endogenous variable: Predict using its SEM
                        try:
                            # Prepare predictor data (current values of parents in cf_data)
                            X_pred_input = cf_data[parents].copy()
                            has_intercept = 'const' in model_info.model.exog_names
                            if has_intercept:
                                X_pred_input['const'] = 1.0
                            X_pred_aligned = X_pred_input.reindex(columns=model_info.model.exog_names, fill_value=0)
                            predictions = model_info.predict(X_pred_aligned)
                            cf_data[var] = predictions
                        except Exception as e:
                            logger.error(f"Prediction failed for variable '{var}' during counterfactual computation: {e}")
                            cf_data[var] = np.nan # Mark as failed
                    else:
                        logger.warning(f"No valid SEM found for endogenous variable '{var}'. Cannot predict counterfactual value.")
                        cf_data[var] = np.nan

            # Return the result
            if observed_data_point is not None:
                return cf_data.iloc[0] # Return Series for the individual
            else:
                return cf_data # Return DataFrame for the population


        elif method == 'pearl3step':
            # Requires modeling exogenous noise variables U, more complex.
            # Step 1: Abduction - Estimate U for the observed_data_point.
            # Step 2: Action - Modify structural equations based on intervention.
            # Step 3: Prediction - Compute counterfactual outcome using modified model and estimated U.
            raise NotImplementedError("Pearl's 3-step counterfactual method is not yet implemented.")
        else:
            raise ValueError(f"Unknown counterfactual method: {method}")


# Example Usage (Optional - Keep commented out or remove for final script)
if __name__ == '__main__':
    print("\n=== Running Causal Model ===\n")
    printer.status("Init", "Causal Model initialized", "success")
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    Z = np.random.normal(0, 1, n_samples)  # Instrument / Common Cause
    X = 0.6 * Z + np.random.normal(0, 0.8, n_samples) # Treatment influenced by Z
    Y = 1.5 * X + 0.8 * Z + np.random.normal(0, 1, n_samples) # Outcome influenced by X and Z (confounding)
    Y = 1.5 * X + np.random.normal(0, 1, n_samples) # Outcome influenced only by X (for simpler IV case)
    W = 0.4 * Z + np.random.normal(0, 1, n_samples) # Another variable correlated with Z

    data = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y, 'W': W})

    # --- Graph Learning ---
    builder = CausalGraphBuilder()
    # This will run PC, orient, BIC optimize etc.
    # In a real scenario, might need constraints.
    builder.forbidden_edges = [('Y', 'X')] # Example constraint
    learned_model = builder.construct_graph(data, sensitive_attrs=[]) # No sensitive attrs here

    print("\nLearned Graph Edges:")
    print(learned_model.graph.edges())

    # --- Effect Estimation ---
    print("\nEstimating Effect X -> Y using Backdoor Linear Regression:")
    try:
        backdoor_est = learned_model.estimate_effect(treatment='X', outcome='Y', method='backdoor.linear_regression')
        print(backdoor_est)
    except Exception as e:
        print(f"Backdoor estimation failed: {e}")


    print("\nEstimating Effect X -> Y using IV (Z as instrument):")
    try:
        iv_est = learned_model.estimate_effect(treatment='X', outcome='Y', method='iv.2sls', instrument='Z')
        if iv_est is not None:
            print(iv_est)
        else:
            print("IV estimation failed.")
    except Exception as e:
        print(f"IV estimation failed: {e}")


    # --- Counterfactual ---
    print("\nComputing Counterfactual E[Y | do(X=2)]:")
    try:
        intervention = {'X': 2.0}
        # Population counterfactual
        cf_population_df = learned_model.compute_counterfactual(intervention=intervention, method='adjust')
        mean_cf_outcome = cf_population_df['Y'].mean()
        print(f"Expected outcome Y under do(X=2): {mean_cf_outcome:.4f}")

        # Individual counterfactual (for the first data point)
        observed_point = data.iloc[0]
        cf_individual = learned_model.compute_counterfactual(intervention=intervention, observed_data_point=observed_point, method='adjust')
        print(f"\nCounterfactual for individual 0 (Observed Y={observed_point['Y']:.4f}):")
        print(cf_individual)

    except Exception as e:
        print(f"Counterfactual computation failed: {e}")
    print("\n=== Causal Model Test Completed ===\n")
