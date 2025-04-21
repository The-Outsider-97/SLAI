import math

from typing import Tuple, Dict

class ProbabilisticModels:
    def __init__(self):
        pass

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
            
        # Fallback to Markov Logic Network-style inference
        ml_weights = []
        ml_evidences = []
        
        # Weighted formulae components
        for e_fact, e_value in evidence.items():
            if e_fact in self.knowledge_base:
                # Formula weight from KB confidence
                weight = math.log(self.knowledge_base[e_fact] / (1 - self.knowledge_base[e_fact] + 1e-8))
                ml_weights.append(weight * (1 if e_value else -1))
                ml_evidences.append(1)
                
        # Add semantic similarity component
        semantic_sim = sum(self.semantic_similarity(str(fact), str(e)) 
                        for e in evidence.keys()) / len(evidence)
        ml_weights.append(2.0)  # Fixed weight for semantic component
        ml_evidences.append(semantic_sim)
        
        # Logistic regression combination
        z = sum(w * e for w, e in zip(ml_weights, ml_evidences))
        probability = 1 / (1 + math.exp(-z))
        
        # Knowledge-based calibration
        base_prob = self.knowledge_base.get(fact, 0.5)
        return 0.7 * probability + 0.3 * base_prob  # Ensured to stay in [0,1]

    def neuro_symbolic_verify(self, fact: Tuple) -> float:
        """Updated to use your SLAILM components"""
        symbolic_score = self.knowledge_base.get(fact, 0.0)
        
        # Use SLAILM for neural validation
        neural_score = self.llm.validate_fact(
            fact, 
            self.knowledge_base
        )
        
        # Weighted combination from config
        return (self.inference_settings['neuro_symbolic_weight'] * neural_score + 
                (1 - self.inference_settings['neuro_symbolic_weight']) * symbolic_score)
    
    def multi_hop_reasoning(self, query: Tuple) -> float:
        """
        Graph-based traversal for combining facts across sources.
        """
        # Build hypothesis graph
        self._build_hypothesis_graph(query)
        
        # Probabilistic graph traversal
        confidence = 1.0
        for hop in self.hypothesis_graph[query]:
            confidence *= self.knowledge_base.get(hop, 0.5)  # Bayesian chain rule
        return confidence
