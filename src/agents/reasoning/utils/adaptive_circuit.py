
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, List, Tuple, Set
from collections import deque

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Adaptive Circuit")
printer = PrettyPrinter

class AdaptiveCircuit(nn.Module):
    """Hybrid neural circuit informed by Bayesian network structure and knowledge base"""
    def __init__(
        self,
        network_structure: Dict[str, Any],
        knowledge_base: Dict[Tuple, Any]
    ):
        super().__init__()
        printer.section_header("Adaptive Circuit")
        self.config = load_global_config()
        self.circuit_config = get_config_section('adaptive_circuit')

        # Configuration parameters with defaults
        self.embedding_dim = self.circuit_config.get('embedding_dim', 64)
        self.hidden_dim = self.circuit_config.get('hidden_dim', 128)
        num_kb_embeddings = self.circuit_config.get('num_kb_embeddings', 1000)

        self.network_structure = network_structure
        self.knowledge_base = knowledge_base

        # Validate network structure
        if not network_structure or 'nodes' not in network_structure:
            raise ValueError("Network structure must contain 'nodes'")
        
        self.input_vars: List[str] = list(network_structure['nodes'])
        self.var_index: Dict[str, int] = {node: idx for idx, node in enumerate(self.input_vars)}
        self.output_vars: List[str] = self.input_vars
        self.num_bn_nodes = len(self.input_vars)

        # Knowledge Base Embedding Layer
        self.kb_entity_to_idx: Dict[str, int] = {}
        self.kb_embedding = None
        self._initialize_kb_embeddings(num_kb_embeddings)

        # Neural Network Layers
        fc1_input_dim = self.num_bn_nodes
        if self.kb_embedding:
            fc1_input_dim += self.embedding_dim

        self.fc1 = nn.Linear(fc1_input_dim, self.hidden_dim)
        self.bn_layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_bn_nodes)

        # Initialize with Bayesian priors
        self._initialize_with_priors()
        logger.info(f"AdaptiveCircuit initialized with input_dim={fc1_input_dim}")

    def _initialize_kb_embeddings(self, num_kb_embeddings: int):
        """Initialize knowledge base embeddings"""
        if not self.knowledge_base:
            logger.warning("No knowledge base provided")
            return

        kb_entities: Set[str] = set()
        for fact_tuple in self.knowledge_base.keys():
            if isinstance(fact_tuple, tuple):
                for item in fact_tuple:
                    kb_entities.add(str(item))

        # Create entity mapping
        sorted_entities = sorted(list(kb_entities))[:num_kb_embeddings]
        self.kb_entity_to_idx = {entity: i for i, entity in enumerate(sorted_entities)}
        
        if self.kb_entity_to_idx:
            self.kb_embedding = nn.Embedding(len(self.kb_entity_to_idx), self.embedding_dim)
            logger.info(f"Created KB embedding for {len(self.kb_entity_to_idx)} entities")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adaptive circuit"""
        # Validate input dimensions
        if x.dim() != 2 or x.size(1) != self.num_bn_nodes:
            raise ValueError(
                f"Input tensor must be (batch_size, {self.num_bn_nodes}), "
                f"got {tuple(x.shape)}"
            )
        
        # Generate KB features
        kb_features = self._get_kb_features_for_input(x)
        
        # Combine inputs
        if kb_features is not None:
            combined_input = torch.cat([x, kb_features], dim=1)
        else:
            combined_input = x
        
        # Neural network processing
        h = F.relu(self.fc1(combined_input))
        h = self.bn_layer_norm(h)
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        logits = self.fc3(h)
        return torch.sigmoid(logits)

    def _get_kb_features_for_input(self, x: torch.Tensor) -> torch.Tensor:
        """Generate KB-derived features for Bayesian Network nodes"""
        if not self.kb_embedding or not self.kb_entity_to_idx:
            return None
        
        batch_size = x.size(0)
        device = x.device
        
        # Find entities related to any BN node
        related_entities = set()
        for node_name in self.input_vars:
            for fact in self.knowledge_base.keys():
                if isinstance(fact, tuple) and node_name in fact:
                    for item in fact:
                        if item != node_name and item in self.kb_entity_to_idx:
                            related_entities.add(item)
        
        if not related_entities:
            return torch.zeros(batch_size, self.embedding_dim, device=device)
        
        # Get embeddings for related entities
        entity_indices = [self.kb_entity_to_idx[e] for e in related_entities]
        embeddings = self.kb_embedding(torch.tensor(entity_indices, device=device))
        
        # Aggregate embeddings (mean pooling)
        aggregated = embeddings.mean(dim=0, keepdim=True)
        return aggregated.expand(batch_size, -1)

    def _initialize_with_priors(self):
        """Initialize network weights using Bayesian network priors"""
        if 'cpt' not in self.network_structure:
            logger.warning("CPT not found. Using default initialization")
            return
    
        # Build parent map
        parent_map: Dict[str, List[str]] = {node: [] for node in self.network_structure['nodes']}
        for edge in self.network_structure.get('edges', []):
            if len(edge) == 2:
                parent, child = edge
                parent_map[child].append(parent)
        
        # Process nodes in topological order
        with torch.no_grad():
            for node_name in self._topological_sort():
                idx = self.var_index.get(node_name, -1)
                if idx == -1:
                    continue
                    
                node_cpt = self.network_structure['cpt'].get(node_name, {})
                
                # Initialize output layer bias
                if 'prior' in node_cpt:  # Root node
                    try:
                        prior = float(node_cpt['prior'])
                        clamped_prior = max(0.01, min(0.99, prior))
                        self.fc3.bias.data[idx] = math.log(clamped_prior / (1 - clamped_prior))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid prior for {node_name}")
                else:  # Child node
                    parent_probs = []
                    for parent in parent_map.get(node_name, []):
                        parent_idx = self.var_index.get(parent, -1)
                        if parent_idx != -1:
                            parent_bias = self.fc3.bias.data[parent_idx]
                            parent_prob = 1 / (1 + math.exp(-parent_bias))
                            parent_probs.append(parent_prob)
                    
                    if parent_probs:
                        avg_parent_prob = sum(parent_probs) / len(parent_probs)
                        clamped_prob = max(0.01, min(0.99, avg_parent_prob))
                        self.fc3.bias.data[idx] = math.log(clamped_prob / (1 - clamped_prob))
                
                # Initialize layer weights
                prior_confidence = node_cpt.get('confidence', 0.7)
                try:
                    prior_confidence = float(prior_confidence)
                except (ValueError, TypeError):
                    prior_confidence = 0.7
                    
                scale = math.sqrt(2.0 / (self.hidden_dim // 2 + self.num_bn_nodes))
                nn.init.normal_(
                    self.fc3.weight.data[idx], 
                    mean=0, 
                    std=scale * prior_confidence
                )
                
                input_dim = self.fc1.in_features
                scale = math.sqrt(2.0 / (input_dim + self.hidden_dim))
                nn.init.normal_(
                    self.fc1.weight.data[:, idx], 
                    mean=0, 
                    std=scale * prior_confidence
                )
    
        logger.info("Network weights initialized with Bayesian priors")

    def _topological_sort(self) -> List[str]:
        """Return nodes in topological order for dependency-aware initialization"""
        if 'edges' not in self.network_structure:
            return self.input_vars
            
        graph: Dict[str, List[str]] = {node: [] for node in self.network_structure['nodes']}
        in_degree: Dict[str, int] = {node: 0 for node in self.network_structure['nodes']}
        
        # Build graph
        for edge in self.network_structure['edges']:
            if len(edge) == 2:
                parent, child = edge
                if parent in graph and child in in_degree:
                    graph[parent].append(child)
                    in_degree[child] += 1
        
        # Initialize queue with root nodes
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        sorted_nodes = []
        
        # Perform topological sort
        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)
            for child in graph.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        # Handle cycles
        if len(sorted_nodes) != len(self.network_structure['nodes']):
            missing = set(self.network_structure['nodes']) - set(sorted_nodes)
            logger.warning(f"Bayesian network contains cycles. Missing nodes: {missing}")
            sorted_nodes += list(missing)
        
        return sorted_nodes

    def to_evidence_dict(self, input_tensor: torch.Tensor = None) -> Dict[str, Any]:
        """Convert circuit state to evidence dictionary"""
        evidence = {
            'metadata': {
                'model_type': 'AdaptiveCircuit',
                'input_vars': self.input_vars,
                'output_vars': self.output_vars,
                'num_parameters': sum(p.numel() for p in self.parameters())
            }
        }
        
        if input_tensor is not None:
            try:
                with torch.no_grad():
                    output = self.forward(input_tensor)
                    evidence['node_states'] = {
                        var: output[0, i].item()
                        for i, var in enumerate(self.output_vars)
                    }
                    evidence['timestamp'] = time.time()
            except Exception as e:
                logger.error(f"Evidence generation failed: {str(e)}")
                evidence['error'] = str(e)
        
        return evidence


if __name__ == "__main__":
    print("\n=== Running Adaptive Circuit ===")
    # Load Bayesian network from file
    bayesian_network_path = "src/agents/reasoning/networks/bayesian_network.json"
    with open(bayesian_network_path, 'r') as f:
        bayesian_network = json.load(f)
    
    # Load knowledge base from file
    knowledge_base_path = "src/agents/knowledge/templates/knowledge_db.json"
    with open(knowledge_base_path, 'r') as f:
        kb_data = json.load(f)
        # Process knowledge base into proper format
        knowledge_base = {}
        for fact in kb_data.get("knowledge", []):
            try:
                key = (fact[0], fact[1], fact[2])
                knowledge_base[key] = float(fact[3])
            except (IndexError, ValueError):
                pass

    circuit = AdaptiveCircuit(
        network_structure=bayesian_network,
        knowledge_base=knowledge_base
    )
    
    # Test forward pass
    test_input = torch.rand(1, len(circuit.input_vars))
    try:
        output = circuit(test_input)
        printer.status("Test", "Forward pass successful", "success")
        printer.status("Output", f"Shape: {tuple(output.shape)}")
    except Exception as e:
        printer.status("Error", f"Forward pass failed: {str(e)}", "error")
    
    # Test evidence generation
    try:
        evidence = circuit.to_evidence_dict(test_input)
        printer.status("Test", "Evidence generation successful", "success")
    except Exception as e:
        printer.status("Error", f"Evidence generation failed: {str(e)}", "error")
    print("\n=== Successfully ran Adaptive Circuit ===\n")
