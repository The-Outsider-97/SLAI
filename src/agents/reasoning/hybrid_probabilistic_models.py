import json
import itertools

from pathlib import Path
from typing import Dict, Any, List, Tuple, Union

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Hybrid Models")
printer = PrettyPrinter()

class HybridProbabilisticModels:
    """
    A dedicated module for creating and managing hybrid Bayesian-Grid networks.
    It dynamically constructs new network structures by combining a semantic (causal)
    network with a spatial (grid) network based on specified connection strategies.
    """
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.net_config = get_config_section('networks')
        self.hybrid_networks_cache = {}
        logger.info("Hybrid Probabilistic Models initialized.")

    def select_hybrid_network(self, task_description: str) -> Dict[str, Any]:
        """
        Selects or creates a hybrid network based on a natural language task description.

        Args:
            task_description: A string describing the task, e.g.,
                              "Model a global weather forecast affecting a 4x4 region" or
                              "Aggregate sensor data from a 2x2 grid to a fire alarm".

        Returns:
            A dictionary representing the generated hybrid network.
        """
        # Simple keyword-based task analysis to determine the creation strategy
        task_lower = task_description.lower()

        if "global" in task_lower and "affecting" in task_lower:
            strategy = "global_to_local"
            # Example: "global weather affecting a 4x4 grid"
            base_bn_key = "bn2x2"  # A simple X->Y network
            base_grid_key = next((f"gn{i}x{i}" for i in [4, 3, 2] if f"{i}x{i}" in task_lower), "gn3x3")
            params = {'global_node': 'X'}
            logger.info(f"Selected 'global_to_local' strategy with BN '{base_bn_key}' and Grid '{base_grid_key}'.")

        elif "aggregate" in task_lower and "alarm" in task_lower:
            strategy = "local_to_global"
            # Example: "aggregate a 2x2 grid of smoke sensors to a main alarm"
            base_bn_key = "bn2x2"
            base_grid_key = next((f"gn{i}x{i}" for i in [2, 3, 4] if f"{i}x{i}" in task_lower), "gn2x2")
            params = {'aggregator_node_name': 'Agg_Sensor', 'target_node': 'Y'}
            logger.info(f"Selected 'local_to_global' strategy with Grid '{base_grid_key}' and BN '{base_bn_key}'.")

        elif "regional" in task_lower or "zone" in task_lower:
            strategy = "regional"
            # Example: "Activate irrigation zone (0,0) to (1,1) in a 4x4 grid"
            base_bn_key = "bn2x2"
            base_grid_key = next((f"gn{i}x{i}" for i in [4, 5, 3] if f"{i}x{i}" in task_lower), "gn4x4")
            params = {'source_node': 'X', 'region_coords': (0, 0, 1, 1)}
            logger.info(f"Selected 'regional' strategy with BN '{base_bn_key}' and Grid '{base_grid_key}'.")

        elif "multi-regional" in task_lower or "multiple zones" in task_lower:
            strategy = "multi-regional"
            # Example: "Control multiple irrigation zones (A and B) in a 6x6 field."
            base_bn_key = "bn6x6"  # Two colliders, but we'll use A and D as independent sources
            base_grid_key = next((f"gn{i}x{i}" for i in [6, 5] if f"{i}x{i}" in task_lower), "gn6x6")
            params = {
                'source_nodes': ['A', 'D'],
                'regions': [
                    (0, 0, 2, 2),  # Zone 1 for source A
                    (3, 3, 5, 5)   # Zone 2 for source D
                ]
            }
            logger.info(f"Selected 'multi-regional' strategy with BN '{base_bn_key}' and Grid '{base_grid_key}'.")

        elif "hierarchical" in task_lower and "aggregation" in task_lower:
            strategy = "hierarchical_aggregation"
            # Example: "Hierarchical aggregation from a 4x4 grid to regional hubs, then to a central alert."
            base_bn_key = "bn7x7"  # Perfect tree structure for this
            base_grid_key = "gn4x4"
            params = {
                # Grid quadrants will aggregate to these leaf nodes
                'leaf_nodes': ['D', 'E', 'F', 'G']
            }
            logger.info(f"Selected 'hierarchical_aggregation' strategy with Grid '{base_grid_key}' and BN '{base_bn_key}'.")
        
        elif "feedback" in task_lower:
            strategy = "feedback_loop"
            # Example: "Forecast causes grid changes, creating feedback for a final alert."
            base_bn_key = "bn3x3" # Simple X -> Y -> Z chain
            base_grid_key = "gn3x3"
            params = {
                'initial_cause': 'X',
                'aggregator_node_name': 'Grid_State_Aggregator',
                'intermediary_effect': 'Y',
                'final_effect': 'Z'
            }
            logger.info(f"Selected 'feedback_loop' strategy with BN '{base_bn_key}' and Grid '{base_grid_key}'.")

        elif "multi-regional" in task_lower or "multiple zones" in task_lower:
            strategy = "multi-regional"
            # Example: "Control multiple irrigation zones (A and D) in a 6x6 field."
            base_bn_key = "bn6x6"  # We'll use A and D as independent control sources.
            base_grid_key = next((f"gn{i}x{i}" for i in [6, 5] if f"{i}x{i}" in task_lower), "gn6x6")
            params = {
                'source_nodes': ['A', 'D'],
                'regions': [
                    (0, 0, 2, 5),  # Zone 1 for source A (top half)
                    (3, 0, 5, 5)   # Zone 2 for source D (bottom half)
                ]
            }
            logger.info(f"Selected 'multi-regional' strategy with BN '{base_bn_key}' and Grid '{base_grid_key}'.")

        elif "hierarchical" in task_lower and "aggregation" in task_lower:
            strategy = "hierarchical_aggregation"
            # Example: "Hierarchical aggregation from a 4x4 grid to regional hubs, then to a central alert."
            base_bn_key = "bn7x7"  # A perfect tree structure: B and C are hubs, A is the central alert.
            base_grid_key = "gn4x4"
            params = {
                'hub_nodes': ['B', 'C'], # The BN nodes that will receive aggregated data.
                'final_node': 'A'
            }
            logger.info(f"Selected 'hierarchical_aggregation' strategy with Grid '{base_grid_key}' and BN '{base_bn_key}'.")
        
        elif "feedback" in task_lower:
            strategy = "feedback_loop"
            # Example: "A forecast (X) causes grid changes, which provides feedback to an outcome (Y)."
            base_bn_key = "bn2x2" # Simple X -> Y chain
            base_grid_key = "gn2x2"
            params = {
                'initial_cause': 'X',
                'aggregator_node_name': 'Grid_State_Agg',
                'feedback_target': 'Y'
            }
            logger.info(f"Selected 'feedback_loop' strategy with BN '{base_bn_key}' and Grid '{base_grid_key}'.")

        elif "pathway" in task_lower or "conduction" in task_lower:
            strategy = "pathway_influence"
            # Example: "Model a wildfire spreading along a specific diagonal pathway in a 4x4 forest grid, driven by wind."
            base_bn_key = "bn2x2" # X represents 'Strong_Wind'
            base_grid_key = "gn4x4"
            params = {
                'source_node': 'X',
                'pathway_nodes': ['N00', 'N11', 'N22', 'N33'] # A diagonal path
            }
            logger.info(f"Selected 'pathway_influence' strategy with BN '{base_bn_key}' and Grid '{base_grid_key}'.")

        elif "robot" in task_lower and "navigation" in task_lower:
            strategy = "robotics_nav_feedback"
            # Example: "A robot's high-level task influences its grid navigation, and grid sensors provide feedback."
            base_bn_key = "bn3x3"  # Represents Task -> Action -> Status
            base_grid_key = "gn4x4" # Represents the robot's local map
            params = {
                'task_node': 'X',                  # High-level task (e.g., 'Go to Dock')
                'action_node': 'Y',                # Low-level action (e.g., 'Move Forward')
                'aggregator_node_name': 'LIDAR_Obstacle', # Summarizes sensor data
                'grid_sensor_prefix': 'Sensor_'    # To rename grid nodes for clarity
            }
            logger.info(f"Selected 'robotics_nav_feedback' strategy with BN '{base_bn_key}' and Grid '{base_grid_key}'.")

        elif "player" in task_lower and "influence" in task_lower:
            strategy = "player_influence_aoe"
            # Example: "A player at (3,3) on a 6x6 map influences AI within a radius of 2."
            base_bn_key = "bn4x4" # We'll use Y,Z,W as three independent AI agents
            base_grid_key = "gn6x6"
            params = {
                'player_pos': (3, 3),
                'influence_radius': 2,
                'target_nodes': ['Y', 'Z', 'W'] # AI agents to be influenced
            }
            logger.info(f"Selected 'player_influence_aoe' strategy with BN '{base_bn_key}' and Grid '{base_grid_key}'.")

        elif "procedural" in task_lower and "generation" in task_lower:
            strategy = "pcg_biome_generation"
            # Example: "Procedural generation of a 5x5 map based on global biome properties."
            base_bn_key = "bn4x4" # X is Biome (e.g., Forest/Desert), Y,Z,W are properties (e.g., Temperature, Humidity)
            base_grid_key = "gn5x5"
            params = {
                'global_property_nodes': ['Y', 'Z', 'W'], # Properties driven by the biome choice 'X'
                'grid_tile_prefix': 'Tile_'
            }
            logger.info(f"Selected 'pcg_biome_generation' strategy with BN '{base_bn_key}' and Grid '{base_grid_key}'.")

        else:
            logger.warning(f"Could not determine hybrid strategy from task: '{task_description}'. Using default.")
            return self.create_hybrid_network(
                base_bn_path=self.net_config['bn2x2'],
                base_grid_path=self.net_config['gn2x2'],
                connection_strategy="global_to_local",
                connection_params={'global_node': 'X'}
            )

        return self.create_hybrid_network(
            base_bn_path=self.net_config[base_bn_key],
            base_grid_path=self.net_config[base_grid_key],
            connection_strategy=strategy,
            connection_params=params
        )

    def create_hybrid_network(
        self,
        base_bn_path: str,
        base_grid_path: str,
        connection_strategy: str,
        connection_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Dynamically constructs a hybrid network by combining a base Bayesian Network
        with a base Grid Network using a specified connection strategy.

        Args:
            base_bn_path: Path to the semantic/causal network JSON.
            base_grid_path: Path to the spatial/grid network JSON.
            connection_strategy: One of ['global_to_local', 'local_to_global', 'regional'].
            connection_params: Dictionary of parameters for the chosen strategy.
                - For 'global_to_local': {'global_node': 'NodeName'}
                - For 'local_to_global': {'aggregator_node_name': 'NewName', 'target_node': 'NodeName'}
                - For 'regional': {'source_node': 'NodeName', 'region_coords': (x_start, y_start, x_end, y_end)}

        Returns:
            The complete hybrid network structure as a dictionary.
        """
        # Caching to avoid re-computing identical hybrids
        cache_key = (base_bn_path, base_grid_path, connection_strategy, json.dumps(connection_params, sort_keys=True))
        if cache_key in self.hybrid_networks_cache:
            logger.info("Returning cached hybrid network.")
            return self.hybrid_networks_cache[cache_key]

        # 1. Load base networks
        with open(base_bn_path, 'r') as f: base_bn = json.load(f)
        with open(base_grid_path, 'r') as f: base_grid = json.load(f)

        # 2. Initialize hybrid network by merging base structures
        hybrid_network = {
            'nodes': base_bn['nodes'] + base_grid['nodes'],
            'edges': base_bn['edges'] + base_grid['edges'],
            'cpt': {**base_bn['cpt'], **base_grid['cpt']},
            'description': f"Hybrid of '{base_bn.get('description', 'BN')}' and '{base_grid.get('description', 'Grid')}'",
            'metadata': {'hybrid_strategy': connection_strategy}
        }

        # 3. Apply connection strategy to create new edges and modify CPTs
        if connection_strategy == "global_to_local":
            global_node = connection_params['global_node']
            for grid_node in base_grid['nodes']:
                hybrid_network['edges'].append([global_node, grid_node])
                self._modify_cpt_for_new_parent(hybrid_network, grid_node, global_node, base_bn, base_grid)

        elif connection_strategy == "local_to_global":
            agg_node = connection_params['aggregator_node_name']
            target_node = connection_params['target_node']
            
            hybrid_network['nodes'].append(agg_node)
            grid_nodes_as_parents = base_grid['nodes']

            for grid_node in grid_nodes_as_parents:
                hybrid_network['edges'].append([grid_node, agg_node])
            hybrid_network['edges'].append([agg_node, target_node])
            
            # Create a functional CPT for the aggregator
            hybrid_network['cpt'][agg_node] = self._create_aggregator_cpt(grid_nodes_as_parents)
            self._modify_cpt_for_new_parent(hybrid_network, target_node, agg_node, base_bn, base_grid)

        elif connection_strategy == "regional":
            source_node = connection_params['source_node']
            x_start, y_start, x_end, y_end = connection_params['region_coords']
            grid_dim = int(len(base_grid['nodes'])**0.5)

            for r in range(y_start, y_end + 1):
                for c in range(x_start, x_end + 1):
                    # Node naming convention is assumed to be 'N<row><col>'
                    grid_node = f"N{r}{c}"
                    if grid_node in hybrid_network['nodes']:
                        hybrid_network['edges'].append([source_node, grid_node])
                        self._modify_cpt_for_new_parent(hybrid_network, grid_node, source_node, base_bn, base_grid)

        elif connection_strategy == "multi-regional":
            source_nodes = connection_params['source_nodes']
            regions = connection_params['regions']

            if len(source_nodes) != len(regions):
                raise ValueError("The number of source nodes must match the number of regions.")

            for i, source_node in enumerate(source_nodes):
                x_start, y_start, x_end, y_end = regions[i]
                for r in range(y_start, y_end + 1):
                    for c in range(x_start, x_end + 1):
                        grid_node = f"N{r}{c}"
                        if grid_node in hybrid_network['nodes']:
                            hybrid_network['edges'].append([source_node, grid_node])
                            self._modify_cpt_for_new_parent(hybrid_network, grid_node, source_node, base_bn, base_grid)
        
        elif connection_strategy == "hierarchical_aggregation":
            leaf_nodes = connection_params['leaf_nodes']
            grid_dim = int(len(base_grid['nodes'])**0.5)
            quadrant_size = grid_dim // 2
            
            # Divide grid into quadrants, one for each leaf node
            quadrants = [
                self._get_grid_nodes_in_region(grid_dim, (0, 0, quadrant_size - 1, quadrant_size - 1)),
                self._get_grid_nodes_in_region(grid_dim, (quadrant_size, 0, grid_dim - 1, quadrant_size - 1)),
                self._get_grid_nodes_in_region(grid_dim, (0, quadrant_size, quadrant_size - 1, grid_dim - 1)),
                self._get_grid_nodes_in_region(grid_dim, (quadrant_size, quadrant_size, grid_dim - 1, grid_dim - 1))
            ]

            for i, leaf_node in enumerate(leaf_nodes):
                quadrant_nodes = quadrants[i]
                for grid_node in quadrant_nodes:
                    if grid_node in hybrid_network['nodes']:
                        hybrid_network['edges'].append([grid_node, leaf_node])
                self._modify_cpt_for_new_parent(hybrid_network, leaf_node, quadrant_nodes[0], base_bn, base_grid) # Placeholder CPT modification


        elif connection_strategy == "feedback_loop":
            # X -> Grid -> Aggregator -> Y -> Z
            initial_cause = connection_params['initial_cause']
            agg_node = connection_params['aggregator_node_name']
            intermediary_effect = connection_params['intermediary_effect']

            # 1. Initial cause influences the entire grid
            for grid_node in base_grid['nodes']:
                hybrid_network['edges'].append([initial_cause, grid_node])
                self._modify_cpt_for_new_parent(hybrid_network, grid_node, initial_cause, base_bn, base_grid)

            # 2. Grid nodes aggregate to a new aggregator node
            hybrid_network['nodes'].append(agg_node)
            for grid_node in base_grid['nodes']:
                hybrid_network['edges'].append([grid_node, agg_node])
            hybrid_network['cpt'][agg_node] = self._create_aggregator_cpt(base_grid['nodes'])

            # 3. Rewire the original BN chain: remove Y's dependency on X and add dependency on the new aggregator
            hybrid_network['edges'] = [edge for edge in hybrid_network['edges'] if not (edge[0] == initial_cause and edge[1] == intermediary_effect)]
            hybrid_network['edges'].append([agg_node, intermediary_effect])
            self._modify_cpt_for_new_parent(hybrid_network, intermediary_effect, agg_node, base_bn, base_grid)

        elif connection_strategy == "multi-regional":
            source_nodes = connection_params['source_nodes']
            regions = connection_params['regions']
            if len(source_nodes) != len(regions):
                raise ValueError("The number of source nodes must match the number of regions.")

            for i, source_node in enumerate(source_nodes):
                region_nodes = self._get_grid_nodes_in_region(base_grid, regions[i])
                for grid_node in region_nodes:
                    if grid_node in hybrid_network['nodes']:
                        hybrid_network['edges'].append([source_node, grid_node])
                        self._modify_cpt_for_new_parent(hybrid_network, grid_node, source_node, base_bn, base_grid)
        
        elif connection_strategy == "hierarchical_aggregation":
            hub_nodes = connection_params['hub_nodes']
            grid_dim = int(len(base_grid['nodes'])**0.5)
            # Split the grid into as many regions as there are hub nodes
            num_hubs = len(hub_nodes)
            region_size = grid_dim // num_hubs
            
            for i, hub_node in enumerate(hub_nodes):
                # Define a vertical slice of the grid for each hub
                region_coords = (i * region_size, 0, (i + 1) * region_size - 1, grid_dim - 1)
                regional_grid_nodes = self._get_grid_nodes_in_region(base_grid, region_coords)
                
                # Each node in the region influences its corresponding hub
                for grid_node in regional_grid_nodes:
                    hybrid_network['edges'].append([grid_node, hub_node])
                
                # Modify the hub's CPT to account for all its new parents (from the grid)
                # Note: This is a placeholder; a real system would need a more robust CPT generation logic.
                # For demonstration, we'll make it dependent on the state of the first node in its quadrant.
                if regional_grid_nodes:
                    self._modify_cpt_for_new_parent(hybrid_network, hub_node, regional_grid_nodes[0], base_bn, base_grid)
        
        elif connection_strategy == "feedback_loop":
            initial_cause = connection_params['initial_cause']
            agg_node = connection_params['aggregator_node_name']
            feedback_target = connection_params['feedback_target']

            # 1. The initial abstract cause influences the entire grid
            for grid_node in base_grid['nodes']:
                hybrid_network['edges'].append([initial_cause, grid_node])
                self._modify_cpt_for_new_parent(hybrid_network, grid_node, initial_cause, base_bn, base_grid)

            # 2. The grid's state is summarized into a new aggregator node
            hybrid_network['nodes'].append(agg_node)
            for grid_node in base_grid['nodes']:
                hybrid_network['edges'].append([grid_node, agg_node])
            hybrid_network['cpt'][agg_node] = self._create_aggregator_cpt(base_grid['nodes'])

            # 3. This aggregated state provides "feedback" by influencing another node in the original BN
            # Remove the original dependency of the feedback target
            original_parents = self._get_parents(base_bn, feedback_target)
            hybrid_network['edges'] = [edge for edge in hybrid_network['edges'] if edge[1] != feedback_target or edge[0] not in original_parents]
            
            # Add the new feedback dependency
            hybrid_network['edges'].append([agg_node, feedback_target])
            self._modify_cpt_for_new_parent(hybrid_network, feedback_target, agg_node, base_bn, base_grid)

        elif connection_strategy == "robotics_nav_feedback":
            task_node = connection_params['task_node']
            action_node = connection_params['action_node']
            agg_node = connection_params['aggregator_node_name']
            
            # Rename grid nodes to represent sensors for clarity
            sensor_nodes = [f"{connection_params['grid_sensor_prefix']}{i}" for i in range(len(base_grid['nodes']))]
            node_map = dict(zip(base_grid['nodes'], sensor_nodes))
            hybrid_network['nodes'] = base_bn['nodes'] + sensor_nodes
            hybrid_network['edges'] = base_bn['edges'] + [[node_map[s], node_map[d]] for s, d in base_grid['edges']]
            hybrid_network['cpt'] = {**base_bn['cpt'], **{node_map[k]: v for k, v in base_grid['cpt'].items()}}
            
            # 1. High-level task influences the robot's action
            # This connection already exists in the base bn3x3 (X->Y)

            # 2. Grid sensor data is aggregated
            hybrid_network['nodes'].append(agg_node)
            for sensor_node in sensor_nodes:
                hybrid_network['edges'].append([sensor_node, agg_node])
            hybrid_network['cpt'][agg_node] = self._create_aggregator_cpt(sensor_nodes)

            # 3. Feedback: The aggregated sensor data influences the action node
            # Remove the original dependency (Task -> Action) and replace it with (Aggregator -> Action)
            hybrid_network['edges'] = [edge for edge in hybrid_network['edges'] if not (edge[0] == task_node and edge[1] == action_node)]
            hybrid_network['edges'].append([agg_node, action_node])
            # The action now depends on the aggregated sensor reading, not the high-level task
            self._modify_cpt_for_new_parent(hybrid_network, action_node, agg_node, base_bn, base_grid)

        elif connection_strategy == "player_influence_aoe":
            player_pos = connection_params['player_pos']
            radius = connection_params['influence_radius']
            target_nodes = connection_params['target_nodes'] # These are the AI agents
            
            # Get grid nodes within the player's influence radius
            nodes_in_radius = self._get_grid_nodes_in_region(base_grid, (
                player_pos[0] - radius, player_pos[1] - radius,
                player_pos[0] + radius, player_pos[1] + radius
            ))

            for ai_agent_node in target_nodes:
                # The AI agent's state (e.g., alert level) now depends on all grid cells in the radius
                for grid_node in nodes_in_radius:
                    if grid_node in hybrid_network['nodes']:
                         hybrid_network['edges'].append([grid_node, ai_agent_node])
                
                # The CPT becomes a large OR-gate: if any influential grid node is True, the AI becomes alert
                hybrid_network['cpt'][ai_agent_node] = self._create_or_gate_cpt(nodes_in_radius)

        elif connection_strategy == "pcg_biome_generation":
            prop_nodes = connection_params['global_property_nodes']
            
            # Rename grid nodes for clarity
            tile_nodes = [f"{connection_params['grid_tile_prefix']}{i}" for i in range(len(base_grid['nodes']))]
            node_map = dict(zip(base_grid['nodes'], tile_nodes))
            hybrid_network['nodes'] = base_bn['nodes'] + tile_nodes
            hybrid_network['edges'] = base_bn['edges'] + [[node_map[s], node_map[d]] for s, d in base_grid['edges']]
            hybrid_network['cpt'] = {**base_bn['cpt'], **{node_map[k]: v for k, v in base_grid['cpt'].items()}}
            
            # Each global property influences every tile on the grid
            for tile_node in tile_nodes:
                for prop_node in prop_nodes:
                    hybrid_network['edges'].append([prop_node, tile_node])
                
                # The CPT for each tile now depends on its neighbors AND the global properties
                self._modify_cpt_for_new_parent(hybrid_network, tile_node, prop_nodes, base_bn, base_grid)
        
        elif connection_strategy == "pathway_influence":
            source_node = connection_params['source_node']
            pathway_nodes = connection_params['pathway_nodes']
            
            for grid_node in pathway_nodes:
                if grid_node in hybrid_network['nodes']:
                    hybrid_network['edges'].append([source_node, grid_node])
                    self._modify_cpt_for_new_parent(hybrid_network, grid_node, source_node, base_bn, base_grid)

        self.hybrid_networks_cache[cache_key] = hybrid_network
        return hybrid_network

    def _get_parents(self, network: Dict, child_node: str) -> List[str]:
        """Utility to find the parents of a node in a network."""
        return [p for p, c in network['edges'] if c == child_node]

    def _modify_cpt_for_new_parent(
            self,
            hybrid_network: Dict,
            child: str,
            new_parents: Union[str, List[str]], # Can be a single node or a list
            base_bn: Dict,
            base_grid: Dict
        ):
            """
            Replaces the CPT of a child node to include one or more new parents.
            """
            if isinstance(new_parents, str):
                new_parents = [new_parents]
    
            original_parents = self._get_parents(base_bn, child) or self._get_parents(base_grid, child)
            all_parents = original_parents + new_parents
            new_cpt = {}
    
            original_cpt = hybrid_network['cpt'].get(child, {})
            
            parent_state_combinations = list(itertools.product([True, False], repeat=len(all_parents)))
    
            for combo in parent_state_combinations:
                cpt_key = ",".join(map(str, combo))
                
                # Slice the combo to get keys for the original parents and new parents
                original_combo = combo[:len(original_parents)]
                new_parents_combo = combo[len(original_parents):]
                
                original_combo_key = ",".join(map(str, original_combo))
    
                # Get base probability from the original CPT
                base_prob_true = 0.5
                if 'prior' in original_cpt:
                     base_prob_true = original_cpt['prior']
                elif original_combo_key in original_cpt:
                     base_prob_true = original_cpt.get(original_combo_key, {}).get('True', 0.5)
    
                # --- Placeholder Logic for combining influences ---
                # Sum the "influence" of the new parents being true.
                influence_score = sum(1 for state in new_parents_combo if state is True)
                
                # Adjust the base probability based on the influence score.
                # This is a simple heuristic; a real model would use a learned function.
                prob_true = base_prob_true + (influence_score * 0.1) - ( (len(new_parents) - influence_score) * 0.1)
                prob_true = max(0.01, min(0.99, prob_true)) # Clamp probability
                
                new_cpt[cpt_key] = {"True": prob_true, "False": 1.0 - prob_true}
                
            hybrid_network['cpt'][child] = new_cpt
            logger.debug(f"Modified CPT for node '{child}' to include new parents: {new_parents}.")

    def _create_aggregator_cpt(self, parent_nodes: List[str], threshold: float = 0.5) -> Dict[str, Any]:
        """
        Creates a CPT for an aggregator node based on a threshold rule.
        Note: This is computationally expensive for many parents and serves as a demonstration.
              A real system might use a functional or parameterized CPT.
        """
        if len(parent_nodes) > 10:
            logger.warning(f"Creating aggregator CPT for {len(parent_nodes)} parents. This is slow and memory-intensive.")
        
        new_cpt = {}
        parent_state_combinations = list(itertools.product([True, False], repeat=len(parent_nodes)))

        for combo in parent_state_combinations:
            cpt_key = ",".join(map(str, combo))
            # Rule: Aggregator is True if the proportion of True parents exceeds the threshold
            proportion_true = sum(combo) / len(combo)
            
            if proportion_true > threshold:
                new_cpt[cpt_key] = {"True": 0.95, "False": 0.05}
            else:
                new_cpt[cpt_key] = {"True": 0.05, "False": 0.95}
        
        return new_cpt
    
    def _get_grid_nodes_in_region(self, grid_dim: int, region_coords: Tuple[int, int, int, int]) -> List[str]:
            """Returns a list of node names within a specified rectangular region of a grid."""
            x_start, y_start, x_end, y_end = region_coords
            nodes_in_region = []
            for r in range(y_start, y_end + 1):
                for c in range(x_start, x_end + 1):
                    nodes_in_region.append(f"N{r}{c}")
            return nodes_in_region
    
    def _create_or_gate_cpt(self, parent_nodes: List[str]) -> Dict[str, Any]:
            """
            Creates a CPT that acts like a noisy OR-gate.
            The child is True if any parent is True.
            """
            if len(parent_nodes) > 10:
                logger.warning(f"Creating OR-gate CPT for {len(parent_nodes)} parents. This is slow.")
    
            new_cpt = {}
            parent_state_combinations = list(itertools.product([True, False], repeat=len(parent_nodes)))
    
            for combo in parent_state_combinations:
                cpt_key = ",".join(map(str, combo))
                # If any parent is True, the child is likely True
                if any(combo):
                    new_cpt[cpt_key] = {"True": 0.95, "False": 0.05}
                else: # All parents are False
                    new_cpt[cpt_key] = {"True": 0.01, "False": 0.99}
            
            return new_cpt

# --- Example Usage ---
if __name__ == "__main__":
    hybrid_builder = HybridProbabilisticModels()

    # --- Test 1: Global-to-Local Connection ---
    printer.section_header("Test 1: Global-to-Local Hybrid")
    task1 = "A global forecast influencing a 2x2 grid"
    global_hybrid = hybrid_builder.select_hybrid_network(task1)
    print(f"Task: '{task1}' -> Strategy: {global_hybrid['metadata']['hybrid_strategy']}")
    print("Modified CPT for N11:", list(global_hybrid['cpt']['N11'].keys())[0])

    # --- Test 2: Local-to-Global Connection ---
    printer.section_header("\nTest 2: Local-to-Global Hybrid")
    task2 = "Aggregate a 2x2 grid of sensors to trigger a main alarm"
    local_hybrid = hybrid_builder.select_hybrid_network(task2)
    print(f"Task: '{task2}' -> Strategy: {local_hybrid['metadata']['hybrid_strategy']}")
    print("New Nodes:", [n for n in local_hybrid['nodes'] if n not in ['X','Y','N00','N01','N10','N11']])
    print("Modified CPT for target Y:", list(local_hybrid['cpt']['Y'].keys()))

    # --- Test 3: Regional Connection ---
    printer.section_header("\nTest 3: Regional Hybrid")
    task3 = "Model an irrigation system for zone (0,0) to (1,1) in a 3x3 field"
    regional_hybrid = hybrid_builder.select_hybrid_network(task3)
    print(f"Task: '{task3}' -> Strategy: {regional_hybrid['metadata']['hybrid_strategy']}")
    new_edges = [edge for edge in regional_hybrid['edges'] if edge[0] == 'X' and edge[1].startswith('N')]
    print("New Regional Edges:", new_edges)
    
    # --- Test 4: Multi-Regional Connection ---
    printer.section_header("\nTest 4: Multi-Regional Hybrid")
    task4 = "Control multiple zones in a 6x6 field"
    multi_regional_hybrid = hybrid_builder.select_hybrid_network(task4)
    print(f"Task: '{task4}' -> Strategy: {multi_regional_hybrid['metadata']['hybrid_strategy']}")
    zone1_edges = [edge for edge in multi_regional_hybrid['edges'] if edge[0] == 'A']
    zone2_edges = [edge for edge in multi_regional_hybrid['edges'] if edge[0] == 'D']
    print(f"Zone 1 (Source A) controls {len(zone1_edges)-1} grid nodes.") # -1 for original edge to C
    print(f"Zone 2 (Source D) controls {len(zone2_edges)-1} grid nodes.") # -1 for original edge to E

    # --- Test 5: Hierarchical Aggregation ---
    printer.section_header("\nTest 5: Hierarchical Aggregation")
    task5 = "Hierarchical aggregation from a 4x4 grid"
    hierarchical_hybrid = hybrid_builder.select_hybrid_network(task5)
    print(f"Task: '{task5}' -> Strategy: {hierarchical_hybrid['metadata']['hybrid_strategy']}")
    parents_of_B = [edge[0] for edge in hierarchical_hybrid['edges'] if edge[1] == 'B']
    print(f"Hub node 'B' is now influenced by {len(parents_of_B)} grid nodes.")
    print("Original Root 'A' is now influenced by:", [edge[0] for edge in hierarchical_hybrid['edges'] if edge[1] == 'A'])

    # --- Test 6: Feedback Loop ---
    printer.section_header("\nTest 6: Feedback Loop")
    task6 = "Model grid feedback to a forecast"
    feedback_hybrid = hybrid_builder.select_hybrid_network(task6)
    print(f"Task: '{task6}' -> Strategy: {feedback_hybrid['metadata']['hybrid_strategy']}")
    print("Original edge X->Y removed:", ['X','Y'] not in feedback_hybrid['edges'])
    print("New Feedback Edge:", [edge for edge in feedback_hybrid['edges'] if edge[1] == 'Y'])

    # --- Test 7: Pathway Influence ---
    printer.section_header("\nTest 7: Pathway Influence")
    task7 = "Model wind guiding a fire along a pathway"
    pathway_hybrid = hybrid_builder.select_hybrid_network(task7)
    print(f"Task: '{task7}' -> Strategy: {pathway_hybrid['metadata']['hybrid_strategy']}")
    path_edges = [edge for edge in pathway_hybrid['edges'] if edge[0] == 'X' and edge[1].startswith('N')]
    print(f"Pathway edges from source 'X': {path_edges}")