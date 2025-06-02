"""
The probabilistic circuit operations are designed to be tractable while
supporting the complex reasoning required by ProbabilisticModels.
"""
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Any, List

#from src.agents.base.utils.math_science import (
#    sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative,
#    leaky_relu, leaky_relu_derivative, elu, elu_derivative, swish, swish_derivative,
#    cross_entropy, cross_entropy_derivative)
from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from src.agents.reasoning.utils.nodes import SumNode, ProductNode
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Model Compute")
printer = PrettyPrinter

#=====================================================
# Activation Functions and their Derivatives
#=====================================================

#ACTIVATION_FUNCTIONS: Dict[str, Tuple[Callable, Callable, bool]] = {
#    'sigmoid': (sigmoid, sigmoid_derivative, False),
#    'relu': (relu, relu_derivative, False),
#    'tanh': (tanh, tanh_derivative, False),
#    'leaky_relu': (leaky_relu, leaky_relu_derivative, True),
#    'elu': (elu, elu_derivative, True),
#    'swish': (swish, swish_derivative, False),
#    'entropy': (cross_entropy, cross_entropy_derivative, False),
#    'linear': (lambda x: x, lambda x: 1.0, False)
#}

class ModelCompute:
    """Probabilistic circuit operations manager"""
    def __init__(self, circuit: nn.Module = None):
        printer.section_header("Probabilistic Inference")
        self.config = load_global_config()
        self.validation_config = get_config_section('model_compute')

        # Initialize probabilistic circuit
        try:
            self._circuit = None
            self.optimizer = None
            self.loss_fn = nn.KLDivLoss(reduction='batchmean')
            self.circuit = circuit 
        except Exception as e:
            printer.status("ERROR", "ModelCompute initialized without circuit")
            raise

        # Epistemic state tracking
        self.belief_history = {}
        self.schema_version = 1.0

        logger.info(f"Model Compute succesfully initialized with: Schema V.{self.schema_version}")

    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, value):
        self._circuit = value
        if value is not None:
            self.optimizer = optim.Adam(
                value.parameters(), 
                lr=self.config.get('lr', 0.001)
            )
            logger.info("Optimizer initialized with circuit parameters")
        else:
            self.optimizer = None
            logger.warning("Circuit set to None - optimizer disabled")

    def _default_circuit(self) -> nn.Module:
        """Default probabilistic circuit architecture"""
        return nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            ProbabilisticLayer(256, 128),
            nn.Softmax(dim=1)
        )

    #=====================================================
    # Core Probabilistic Circuit Operations
    #=====================================================

    def compute_marginal_probability(self, variables: Tuple[str], evidence: Dict[str, Any]) -> float:
        """Compute marginal probability using circuit with evidence clamping"""
        if not self.circuit:
            raise ValueError("Circuit not initialized")
        
        # Convert evidence to tensor format
        input_tensor = self._evidence_to_tensor(evidence)
        
        # Clamp evidence variables
        for i, var in enumerate(self.circuit.input_vars):
            if var in evidence:
                input_tensor[:, i] = evidence[var]
        
        # Forward pass through circuit
        with torch.no_grad():
            output = self.circuit(input_tensor)
            var_indices = [self.circuit.var_index[v] for v in variables]
            result = torch.prod(output[:, var_indices]).item()
        
        return result

    def compute_map_estimate(self, evidence: Dict[str, Any]) -> Tuple[str, float]:
        """Compute MAP estimate using gradient ascent on log-probability"""
        try:
            # Convert evidence to differentiable tensor
            input_tensor = self._evidence_to_tensor(evidence).requires_grad_(True)
            
            # Optimization loop
            best_state, best_prob = None, -float('inf')
            for _ in range(100):  # Max optimization steps
                self.optimizer.zero_grad()
                
                # Forward pass
                probs = self.circuit(input_tensor)
                log_prob = torch.log(probs).sum()
                
                # Backward pass
                log_prob.backward()
                self.optimizer.step()
                
                # Track best state
                if log_prob.item() > best_prob:
                    best_prob = log_prob.item()
                    best_state = self._tensor_to_state(probs)
            
            return best_state, math.exp(best_prob)
        except Exception as e:
            printer.status("Error", f"MAP estimation failed: {str(e)}", "error")
            logger.exception("MAP estimation failed")
            raise

    def compute_moment(self, variable: str, order: int = 1) -> float:
        """Compute k-th moment using circuit sampling"""
        samples = self._sample_circuit(1000)  # 1000 samples
        var_values = [s[variable] for s in samples]
        return np.mean(np.power(var_values, order))

    def kullback_leibler_divergence(self, dist_p: Dict, dist_q: Dict) -> float:
        """Compute KL divergence between distributions"""
        p_tensor = torch.tensor([dist_p[k] for k in sorted(dist_p)])
        q_tensor = torch.tensor([dist_q[k] for k in sorted(dist_q)])
        return F.kl_div(
            q_tensor.log(), 
            p_tensor, 
            reduction='batchmean'
        ).item()

    def marginal_map_query(self, marginal_vars: Tuple[str], map_vars: Tuple[str], evidence: Dict[str, Any]) -> Tuple[str, float]:
        """Hybrid marginal-MAP inference using alternating optimization"""
        # Initialize with evidence
        state = evidence.copy()
        
        for _ in range(10):  # Iteration limit
            # Maximize over MAP variables
            for var in map_vars:
                if var not in state:
                    state[var] = self.compute_map_estimate(state)[0][var]
            
            # Marginalize over remaining variables
            for var in marginal_vars:
                if var not in state:
                    state[var] = self.compute_marginal_probability((var,), state)
        
        # Extract MAP values and confidence
        map_state = {var: state[var] for var in map_vars}
        confidence = self.compute_marginal_probability(map_vars, state)
        return map_state, confidence
    
    #=====================================================
    # Epistemological Operations
    #=====================================================
    def dynamic_model_revision(self, new_evidence: Dict[str, Any]) -> None:
        """Bayesian model updating with assimilation-accommodation"""
        if not self.circuit or not self.optimizer:
            logger.error("Cannot perform revision without initialized circuit and optimizer")
            return
            
        # Convert evidence to training data
        inputs, targets = self._evidence_to_training_data(new_evidence)
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training loop
        self.circuit.train()
        for epoch in range(5):  # Short refinement epochs
            for batch_in, batch_targ in loader:
                self.optimizer.zero_grad()
                outputs = self.circuit(batch_in)
                loss = self.loss_fn(outputs.log(), batch_targ)
                loss.backward()
                self.optimizer.step()
        
        # Schema version update
        self.schema_version += 0.1
        printer.status("Revision", f"Schema updated to V.{self.schema_version:.1f}", "success")

    def schema_equilibration(self) -> None:
        """Cognitive consistency maintenance via regularization"""
        # Add epistemic regularization term
        for param in self.circuit.parameters():
            param.grad += 0.01 * torch.sign(param.data)  # L1 regularization
        
        # Apply constraints for cognitive consistency
        self.enforce_circuit_constraints()

    #=====================================================
    # Circuit Structural Operations
    #=====================================================

    def check_decomposability(self) -> bool:
        """Verify decomposability property through scope analysis"""
        scopes = self.circuit.compute_scopes()
        for node in self.circuit.nodes:
            if isinstance(node, ProductNode):
                child_scopes = [scopes[child] for child in node.children]
                if any(s1.intersection(s2) for s1, s2 in zip(child_scopes, child_scopes[1:])):
                    return False
        return True

    def check_smoothness(self) -> bool:
        """Verify smoothness property through scope checking"""
        scopes = self.circuit.compute_scopes()
        for node in self.circuit.nodes:
            if isinstance(node, SumNode):
                child_scopes = [scopes[child] for child in node.children]
                if not all(s == child_scopes[0] for s in child_scopes[1:]):
                    return False
        return True

    def check_determinism(self) -> bool:
        """Verify determinism through activation patterns"""
        samples = self._sample_circuit(100)
        for sample in samples:
            activations = self.circuit.trace_activations(sample, input_tensor=None)
            for node in self.circuit.nodes:
                if isinstance(node, SumNode) and sum(activations[node]) > 1:
                    return False
        return True

    def enforce_circuit_constraints(self) -> None:
        """Apply structural constraints via parameter projection"""
        with torch.no_grad():
            # Enforce sum-to-one constraint
            for node in self.circuit.nodes:
                if isinstance(node, SumNode):
                    node.weights.data = F.normalize(node.weights.data, p=1, dim=0)
            
            # Enforce non-negativity
            for param in self.circuit.parameters():
                param.data = torch.clamp(param.data, min=0)

    #=====================================================
    # Interpretability and Explainability
    #=====================================================
    
    def explain_inference_path(self, query: str, evidence: Dict[str, Any]) -> str:
        """Generate human-readable inference explanation"""
        printer.section_header(f"Inference Explanation: {query}")
        input_tensor = self._evidence_to_tensor(evidence)
        activations = self.circuit.trace_activations(input_tensor, sample=None)
        
        explanation = f"Inference for '{query}' given evidence:\n"
        for node, act in activations.items():
            if act > 0.1:  # Significant activation
                explanation += f"- {node.description} (weight: {act:.2f})\n"
        
        # Add final confidence
        confidence = self.compute_marginal_probability((query,), evidence)
        explanation += f"\nConclusion: P({query}|evidence) = {confidence:.3f}"
        printer.code_block(explanation, language="text")

        return explanation

    def introspect_model_biases(self) -> Dict[str, float]:
        """Quantify model biases through parameter analysis"""
        printer.section_header("Model Bias Introspection")
        biases = {}
        for name, param in self.circuit.named_parameters():
            if 'bias' in name:
                biases[name] = param.data.mean().item()
        
        # Add epistemic bias metrics
        biases['schema_rigidity'] = self._calculate_schema_rigidity()
        headers = ["Bias Type", "Value"]
        rows = [[name, f"{value:.4f}"] for name, value in biases.items()]
        printer.table(headers, rows, "Model Biases")

        return biases

    def simulate_alternative_hypotheses(self, evidence: Dict[str, Any]) -> Dict[str, float]:
        """Generate counterfactual hypotheses by evidence perturbation"""
        printer.section_header(f"Simulate Alternative Hypotheses: {evidence}")
        base_prob = self.compute_marginal_probability(list(evidence.keys()), evidence)
        hypotheses = {}
        
        for var in evidence:
            # Create counterfactual evidence
            cf_evidence = evidence.copy()
            cf_evidence[var] = 1 - cf_evidence[var]  # Flip value
            
            # Compute probability difference
            cf_prob = self.compute_marginal_probability(list(cf_evidence.keys()), cf_evidence)
            hypotheses[var] = cf_prob - base_prob
        
        return hypotheses

    def belief_revision_trace(self, fact: Tuple, updates: List[Dict]) -> List[float]:
        """Track belief evolution through update sequence"""
        printer.section_header(f"Belief Revision Trace:\n{fact}\n{updates}")
        trace = []
        current_belief = self.knowledge_base.get(fact, 0.5)
        
        for update in updates:
            self.dynamic_model_revision(update)
            new_belief = self.compute_marginal_probability([fact[0]], {})
            trace.append(new_belief)
            current_belief = new_belief
        
        # Store history for introspection
        self.belief_history[fact] = trace
        return trace
    
    #=====================================================
    # Internal Utility Methods
    #=====================================================

    def _evidence_to_tensor(self, evidence: Dict[str, Any]) -> torch.Tensor:
        """Convert evidence dict to circuit input tensor"""
        if not self.circuit:
            raise ValueError("Circuit not initialized")
            
        return torch.tensor([
            evidence.get(var, 0.5) for var in self.circuit.input_vars
        ]).unsqueeze(0)

    def _sample_circuit(self, n_samples: int) -> List[Dict]:
        """Generate samples from the probabilistic circuit"""
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                sample = {}
                # Start from root and sample downward
                current = self.circuit.root
                while current.children:
                    if isinstance(current, SumNode):
                        # Sample child based on weights
                        child_idx = torch.multinomial(current.weights, 1).item()
                        current = current.children[child_idx]
                    elif isinstance(current, ProductNode):
                        # Process all children
                        for child in current.children:
                            # ... recursive sampling ...
                            pass
                samples.append(current.to_evidence_dict())
        return samples

    def _calculate_schema_rigidity(self) -> float:
        """Measure resistance to schema change"""
        if not self.belief_history:
            return 0.0
        return np.mean([
            np.std(trace) for trace in self.belief_history.values()
        ])

    def _evidence_to_training_data(self, evidence: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert evidence to supervised learning format"""
        if not self.circuit:
            raise ValueError("Circuit not initialized")
            
        inputs = []
        targets = []
        
        # Handle single evidence case
        if not isinstance(evidence, list):
            evidence = [evidence]
            
        for case in evidence:
            inp = [case.get(var, 0.5) for var in self.circuit.input_vars]
            target = [case.get(var, 0.5) for var in self.circuit.output_vars]
            inputs.append(inp)
            targets.append(target)
            
        return torch.tensor(inputs), torch.tensor(targets)


#=====================================================
# Supporting Circuit Components
#=====================================================

class ProbabilisticLayer(nn.Module):
    """Tractable probabilistic layer with structure constraints"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the adaptive circuit:
        1. Validates input dimensions
        2. Generates knowledge base features
        3. Combines input with KB features
        4. Processes through neural network layers
        5. Returns probability distribution over BN nodes
        
        Args:
            x: Input tensor representing states of Bayesian Network nodes.
               Shape: (batch_size, num_bn_nodes). Values in [0,1] range.
        
        Returns:
            Probability distribution over BN nodes.
            Shape: (batch_size, num_bn_nodes). Values in (0,1) range.
        """
        # Validate input dimensions
        if x.dim() != 2 or x.size(1) != self.num_bn_nodes:
            raise ValueError(f"Input tensor must be 2D with shape (batch_size, {self.num_bn_nodes}). "
                             f"Received shape: {tuple(x.shape)}")
        
        # Generate KB features (returns None if no KB embedding)
        kb_features = self._get_kb_features_for_input(x)
        
        # Combine BN node states with KB features
        if kb_features is not None:
            # kb_features shape: (batch_size, num_bn_nodes * embedding_dim)
            combined_input = torch.cat([x, kb_features], dim=1)
        else:
            combined_input = x
        
        # Neural network processing pipeline
        # Layer 1: Fully connected + ReLU
        h1 = self.fc1(combined_input)
        h1_activated = F.relu(h1)
        
        # Normalization and regularization
        h1_norm = self.bn_layer_norm(h1_activated)
        h1_dropped = self.dropout(h1_norm)
        
        # Layer 2: Fully connected + ReLU
        h2 = self.fc2(h1_dropped)
        h2_activated = F.relu(h2)
        
        # Output layer: Logits for each BN node
        logits = self.fc3(h2_activated)
        
        # Convert logits to probabilities
        probabilities = torch.sigmoid(logits)
        
        return probabilities

    def compute_scopes(self) -> Dict[str, set]:
        """
        Computes the scope of each node in the circuit by analyzing the computational graph.
        The scope of a node is the set of input variables that influence its output.
        
        Returns:
            Dictionary mapping layer names to sets of input variables they depend on
        """
        scopes = {}
        # Initialize scopes for input nodes
        for i, node_name in enumerate(self.input_vars):
            scopes[f'input_{node_name}'] = {node_name}
        
        # Create dependency graph
        graph = {
            'fc1': set(self.input_vars),
            'bn_layer_norm': {'fc1'},
            'dropout': {'bn_layer_norm'},
            'fc2': {'dropout'},
            'fc3': {'fc2'},
            'output': {'fc3'}
        }
        
        # Propagate scopes through the computational graph
        scopes['fc1'] = set(self.input_vars)
        scopes['bn_layer_norm'] = scopes['fc1']
        scopes['dropout'] = scopes['bn_layer_norm']
        scopes['fc2'] = scopes['dropout']
        scopes['fc3'] = scopes['fc2']
        scopes['output'] = scopes['fc3']
        
        # Add KB embedding scope if present
        if self.kb_embedding:
            kb_scope = set()
            for fact in self.knowledge_base.keys():
                if isinstance(fact, tuple):
                    kb_scope |= set(str(item) for item in fact)
            graph['kb_embedding'] = kb_scope
            scopes['kb_embedding'] = kb_scope
            scopes['fc1'] |= kb_scope
        
        logger.debug(f"Computed scopes for {len(scopes)} nodes")
        return scopes

    def trace_activations(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Traces activations through the network, capturing intermediate layer outputs.
        
        Args:
            input_tensor: Input to the circuit (batch_size, num_bn_nodes)
            
        Returns:
            Dictionary of layer names to their activation tensors
        """
        activations = {}
        with torch.no_grad():
            # Store input
            activations['input'] = input_tensor.clone()
            
            # KB features if available
            kb_features = self._get_kb_features_for_input(input_tensor)
            if kb_features is not None:
                activations['kb_features'] = kb_features.clone()
                combined_input = torch.cat([input_tensor, kb_features], dim=1)
            else:
                combined_input = input_tensor
                
            # Forward pass through layers
            h1 = self.fc1(combined_input)
            activations['fc1_pre_act'] = h1.clone()
            
            h1_relu = F.relu(h1)
            activations['relu1'] = h1_relu.clone()
            
            h1_norm = self.bn_layer_norm(h1_relu)
            activations['bn_layer_norm'] = h1_norm.clone()
            
            h1_dropped = self.dropout(h1_norm)
            activations['dropout'] = h1_dropped.clone()
            
            h2 = self.fc2(h1_dropped)
            activations['fc2_pre_act'] = h2.clone()
            
            h2_relu = F.relu(h2)
            activations['relu2'] = h2_relu.clone()
            
            logits = self.fc3(h2_relu)
            activations['fc3_logits'] = logits.clone()
            
            probabilities = torch.sigmoid(logits)
            activations['output_probabilities'] = probabilities.clone()
        
        logger.debug(f"Traced activations for {len(activations)} layers")
        return activations

    @property
    def nodes(self) -> List[Tuple[str, nn.Module]]:
        """
        Returns a list of named nodes in the circuit with their corresponding modules.
        
        Returns:
            List of (node_name, module) tuples
        """
        node_list = [
            ('fc1', self.fc1),
            ('bn_layer_norm', self.bn_layer_norm),
            ('dropout', self.dropout),
            ('fc2', self.fc2),
            ('fc3', self.fc3)
        ]
        
        if self.kb_embedding:
            node_list.insert(0, ('kb_embedding', self.kb_embedding))
        
        return node_list

    @property
    def root(self) -> nn.Module:
        """
        Returns the root module of the circuit (output layer).
        
        Returns:
            Output layer module
        """
        return self.fc3

    def to_evidence_dict(self, input_tensor: torch.Tensor = None) -> Dict[str, Any]:
        """
        Converts the circuit's state or output to an evidence dictionary.
        If input is provided, includes output probabilities.
        
        Args:
            input_tensor: Optional input to generate output evidence
            
        Returns:
            Evidence dictionary containing:
            - node_states: Current probability distributions
            - parameters: Model metadata
            - scopes: Variable dependencies
        """
        printer.status("Evidence", f"{input_tensor}")
        evidence = {
            'metadata': {
                'model_type': 'AdaptiveCircuit',
                'input_vars': self.input_vars,
                'output_vars': self.output_vars,
                'num_parameters': sum(p.numel() for p in self.parameters())
            },
            'scopes': self.compute_scopes()
        }
        
        # Add current state if input is provided
        if input_tensor is not None:
            with torch.no_grad():
                output = self.forward(input_tensor)
                evidence['node_states'] = {
                    var: output[0, i].item()
                    for i, var in enumerate(self.output_vars)
                }
                evidence['timestamp'] = time.time()
        
        # Add parameter summaries
        param_stats = {}
        for name, param in self.named_parameters():
            param_stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item()
            }
        evidence['parameter_stats'] = param_stats
        
        logger.debug("Generated evidence dictionary")
        return evidence

if __name__ == "__main__":
    print("\n=== Running Model Compute ===")
    printer.section_header("Model Compute Initialization")

    class TestCircuit(nn.Module):
        """Simple circuit for testing ModelCompute"""
        def __init__(self):
            super().__init__()
            self.input_vars = ['var_0', 'var_1', 'var_2']
            self.var_index = {v: i for i, v in enumerate(self.input_vars)}
            self.output_vars = self.input_vars
            self.net = nn.Sequential(
                nn.Linear(3, 10),
                nn.ReLU(),
                nn.Linear(10, 3),
                nn.Softmax(dim=1)
            )
            
        def forward(self, x):
            return self.net(x)

    test_circuit = TestCircuit()
    
    # Initialize ModelCompute with circuit
    model = ModelCompute(circuit=test_circuit)
    
    # Create test evidence
    evidence = {"var_0": 1.0, "var_1": 0.5}
    query = ("var_2",)
    
    # Run marginal probability calculation
    try:
        result = model.compute_marginal_probability(query, evidence)
        printer.status("Test", 
                      f"Marginal probability for {query}: {result:.4f}", 
                      "success")
    except Exception as e:
        printer.status("Error", f"Computation failed: {str(e)}", "error")
    print("\n=== Successfully Model Compute ===\n")
