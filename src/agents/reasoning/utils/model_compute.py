"""
The probabilistic circuit operations are designed to be tractable while
supporting the complex reasoning required by ProbabilisticModels.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from typing import Tuple, Dict, Any, Callable, List, Union
from torch.utils.data import DataLoader, TensorDataset

from src.agents.base.utils.math_science import (
    sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative,
    leaky_relu, leaky_relu_derivative, elu, elu_derivative, swish, swish_derivative,
    cross_entropy, cross_entropy_derivative)
from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Model Compute")
printer = PrettyPrinter

#=====================================================
# Activation Functions and their Derivatives
#=====================================================

ACTIVATION_FUNCTIONS: Dict[str, Tuple[Callable, Callable, bool]] = {
    'sigmoid': (sigmoid, sigmoid_derivative, False),
    'relu': (relu, relu_derivative, False),
    'tanh': (tanh, tanh_derivative, False),
    'leaky_relu': (leaky_relu, leaky_relu_derivative, True),
    'elu': (elu, elu_derivative, True),
    'swish': (swish, swish_derivative, False),
    'entropy': (cross_entropy, cross_entropy_derivative, False),
    'linear': (lambda x: x, lambda x: 1.0, False)
}

class ModelCompute(nn.Module):
    def __init__(self, circuit: nn.Module = None):
        super().__init__()
        printer.section_header("Probabilistic Inference")
        self.config = load_global_config()
        self.validation_config = get_config_section('model_compute')

        # Initialize probabilistic circuit
        self.circuit = circuit or DefaultCircuit()
        self.optimizer = optim.Adam(self.circuit.parameters(), lr=0.001)
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        
        # Epistemic state tracking
        self.belief_history = {}
        self.schema_version = 1.0

        logger.info(f"Model Compute succesfully initialized with: Schema Version {self.schema_version}")

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
        printer.status("Inference", f"Computing marginal for {variables} with evidence: {evidence}")
        with torch.no_grad():
            # Convert evidence to tensor format
            input_tensor = self._evidence_to_tensor(evidence)
            
            # Clamp evidence variables
            for i, var in enumerate(self.circuit.input_vars):  # FIX: Access circuit's input_vars
                if var in evidence:
                    input_tensor[:, i] = evidence[var]
            
            var_indices = [self.circuit.var_index[v] for v in variables]  # FIX: Access circuit's var_index
            
            # Forward pass through circuit
            output = self.circuit(input_tensor)
            result = torch.prod(output[:, var_indices]).item()
            printer.status("Result", f"P({variables}|evidence) = {result:.4f}", "success")
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
        printer.status("Revision", "Starting Bayesian model updating")
        # Convert evidence to training data
        inputs, targets = self._evidence_to_training_data(new_evidence)
        
        # Create DataLoader
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training loop
        self.circuit.train()
        for epoch in range(5):  # Short refinement epochs
            printer.progress_bar(epoch, 5, f"Epoch {epoch+1}/5")
            for batch_in, batch_targ in loader:
                self.optimizer.zero_grad()
                outputs = self.circuit(batch_in)
                loss = self.loss_fn(outputs.log(), batch_targ)
                loss.backward()
                self.optimizer.step()
        
        # Schema version update
        self.schema_version += 0.1
        printer.status("Revision", f"Schema updated to v{self.schema_version:.1f}", "success")

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
            activations = self.circuit.trace_activations(sample)
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
        activations = self.circuit.trace_activations(input_tensor)
        
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
        printer.section_header(f"Evidence to Tensor: {evidence}")
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
        inputs = []
        targets = []
        for case in evidence:
            inp = [case.get(var, 0.5) for var in self.circuit.input_vars]
            target = [case[var] for var in self.circuit.output_vars]
            inputs.append(inp)
            targets.append(target)
        return torch.tensor(inputs), torch.tensor(targets)

#=====================================================
# Supporting Circuit Components
#=====================================================

class DefaultCircuit(nn.Module):
    """Default probabilistic circuit with required attributes"""
    def __init__(self, num_vars=128):
        super().__init__()
        # Define variables and their indices
        self.input_vars = [f"var_{i}" for i in range(num_vars)]
        self.var_index = {var: i for i, var in enumerate(self.input_vars)}
        self.output_vars = self.input_vars  # Output variables same as input
        
        # Network architecture
        self.net = nn.Sequential(
            nn.Linear(num_vars, 256),
            nn.ReLU(),
            ProbabilisticLayer(256, num_vars),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.net(x)

class AdaptiveCircuit:
    def __init__(self, network_structure=None,
            knowledge_base=None):
        pass

class ProbabilisticLayer(nn.Module):
    """Tractable probabilistic layer with structure constraints"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        return F.softmax(x @ self.weights.t() + self.bias, dim=1)

class SumNode(nn.Module):
    """Weighted sum node for mixture distributions"""
    def __init__(self, children: List[nn.Module]):
        super().__init__()
        self.children = nn.ModuleList(children)
        self.weights = nn.Parameter(torch.ones(len(children)))
        
    def forward(self, x):
        outputs = [child(x) for child in self.children]
        weighted = [w * out for w, out in zip(F.softmax(self.weights, dim=0), outputs)]
        return sum(weighted)

class ProductNode(nn.Module):
    """Product node for factorized distributions"""
    def __init__(self, children: List[nn.Module]):
        super().__init__()
        self.children = nn.ModuleList(children)
        
    def forward(self, x):
        outputs = [child(x) for child in self.children]
        return torch.prod(torch.stack(outputs), dim=0)

if __name__ == "__main__":
    print("\n=== Running Model Compute ===")
    printer.section_header("Model Compute Initialization")

    model = ModelCompute()

    # Use existing variables from the circuit
    evidence = {"var_0": 1.0, "var_1": 0.5}
    query = ("var_2",)  # Query an existing variable

    result = model.compute_marginal_probability(query, evidence)
    printer.status("Test", f"Marginal probability for {query} completed: {result:.4f}",
                   "success" if result > 0 else "warning")
    print("\n=== Successfully Model Compute ===\n")
