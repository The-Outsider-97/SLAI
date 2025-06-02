
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict

class SumNode(nn.Module):
    """Weighted sum node for mixture distributions with structural constraints"""
    def __init__(self, children: List[nn.Module]):
        super().__init__()
        self.children = nn.ModuleList(children)
        # Initialize weights with uniform distribution
        self.weights = nn.Parameter(torch.ones(len(children)))
        self._register_constraints()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply softmax to ensure weights sum to 1
        normalized_weights = F.softmax(self.weights, dim=0)
        outputs = [child(x) for child in self.children]
        
        # Compute weighted sum while maintaining gradient flow
        result = torch.zeros_like(outputs[0])
        for weight, output in zip(normalized_weights, outputs):
            result += weight * output
            
        return result
    
    def compute_scope(self) -> set:
        """Compute scope as union of children's scopes"""
        scope = set()
        for child in self.children:
            if hasattr(child, 'compute_scope'):
                scope |= child.compute_scope()
        return scope
    
    def trace_activations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Trace activations through sum node and its children"""
        activations = {}
        normalized_weights = F.softmax(self.weights, dim=0).detach()
        
        # Collect child activations
        child_activations = []
        for i, child in enumerate(self.children):
            if hasattr(child, 'trace_activations'):
                child_act = child.trace_activations(x)
                child_activations.append(child_act)
                # Annotate with weight
                child_act['weight'] = normalized_weights[i]
            else:
                # Handle leaf nodes
                child_out = child(x)
                child_activations.append({
                    'output': child_out,
                    'weight': normalized_weights[i]
                })
                
        activations['children'] = child_activations
        activations['output'] = self.forward(x)
        return activations
    
    def _register_constraints(self):
        """Register parameters for constraint enforcement"""
        self._constraints = {
            'weights': lambda: F.normalize(self.weights.data, p=1, dim=0)
        }

class ProductNode(nn.Module):
    """Product node for factorized distributions with decomposability"""
    def __init__(self, children: List[nn.Module]):
        super().__init__()
        self.children = nn.ModuleList(children)
        self.scopes = [child.compute_scope() for child in children]
        self._validate_decomposability()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute element-wise product of children outputs
        result = self.children[0](x)
        for child in self.children[1:]:
            result = result * child(x)
        return result
    
    def compute_scope(self) -> set:
        """Compute scope as union of children's scopes"""
        scope = set()
        for child_scope in self.scopes:
            scope |= child_scope
        return scope
    
    def trace_activations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Trace activations through product node and its children"""
        activations = {}
        child_activations = []
        
        # Collect child activations
        for child in self.children:
            if hasattr(child, 'trace_activations'):
                child_act = child.trace_activations(x)
                child_activations.append(child_act)
            else:
                child_out = child(x)
                child_activations.append({'output': child_out})
                
        activations['children'] = child_activations
        activations['output'] = self.forward(x)
        return activations
    
    def _validate_decomposability(self):
        """Ensure children have disjoint scopes for valid decomposability"""
        for i, scope_i in enumerate(self.scopes):
            for j, scope_j in enumerate(self.scopes):
                if i != j and scope_i & scope_j:
                    raise ValueError(
                        f"Product node children have overlapping scopes: "
                        f"Child {i} scope {scope_i} overlaps with child {j} scope {scope_j}"
                    )
