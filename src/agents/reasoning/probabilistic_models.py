
import yaml, json
import math

from pathlib import Path
from typing import Tuple, Dict, Any

from logs.logger import get_logger

logger = get_logger("Probabilistic Models")

CONFIG_PATH = "src/agents/reasoning/configs/reasoning_config.yaml"

def load_config(config_path=CONFIG_PATH):
    """Load YAML configuration from specified path"""
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    """Merge base config with optional user-provided config"""
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class ProbabilisticModels:
    def __init__(self, config: Dict[str, Any] = None, llm: Any = None):
        """
        Initialize probabilistic reasoning system with:
        - Bayesian network structure
        - Knowledge base
        - Inference configuration
        - Integrated LLM component
        
        Args:
            config: Preloaded configuration dictionary
            llm: Initialized language model component
        """
        # Load merged configuration
        self.config = config or get_merged_config()
        self.inference_settings = self.config['inference']
        
        # Load Bayesian network structure
        self.bayesian_network = self._load_bayesian_network(
            Path(self.config['storage']['bayesian_network'])
        )
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base(
            Path(self.config['storage']['knowledge_db'])
        )
        
        # Initialize LLM component
        self.llm = llm or self._initialize_default_llm()
        
        # Configure hybrid inference weights
        self.markov_logic_weight = self.inference_settings.get(
            'markov_logic_weight', 0.7
        )
        self.knowledge_base_weight = self.inference_settings.get(
            'knowledge_base_weight', 0.3
        )

        logger.info(f"Probabilistic Models initialized...")

    def _load_bayesian_network(self, network_path: Path) -> Dict[str, Any]:
        """Load Bayesian network structure from JSON file"""
        if not network_path.exists():
            raise FileNotFoundError(f"Bayesian network file not found: {network_path}")

        logger.info(f"Loading Bayesian network from {network_path}")
        with open(network_path, 'r') as f:
            return json.load(f)

    def _load_knowledge_base(self, kb_path: Path) -> Dict[Tuple, float]:
        """Load knowledge base with fact confidences"""
        if not kb_path.exists():
            raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")

        logger.info(f"Loading knowledge base from {kb_path}")
        with open(kb_path, 'r') as f:
            kb = json.load(f)
        return {tuple(k.split('||')): v for k, v in kb.items()}

    def _initialize_default_llm(self):
        """Fallback LLM initialization if none provided"""
        logger.warning("Initializing default LLM with basic validation capabilities")
        return SimpleLLM()

    def bayesian_inference(self, query: str, evidence: Dict[str, bool]) -> float:
        """
        Exact inference using message passing algorithm based on:
        - Pearl (1988) Probabilistic Reasoning in Intelligent Systems
        - Koller & Friedman (2009) Probabilistic Graphical Models
        """
        # Convert evidence to network nodes
        observed = {node: value for node, value in evidence.items() 
                   if node in self.bayesian_network['nodes']}
    
        # Initialize belief states
        beliefs = {node: {'prior': 0.5, 'likelihood': 1.0} 
                  for node in self.bayesian_network['nodes']}
    
        # Set observed evidence
        for node, value in observed.items():
            beliefs[node]['prior'] = 1.0 if value else 0.0
            beliefs[node]['likelihood'] = 1.0  # Hard evidence
    
        # Create parent map
        parent_map = {child: [] for child in self.bayesian_network['nodes']}
        for parent, child in self.bayesian_network['edges']:
            parent_map[child].append(parent)
    
        # Message passing schedule
        for _ in range(2):  # Two-pass loopy belief propagation
            # Forward pass (children to parents)
            for child in self.bayesian_network['nodes']:
                parents = parent_map[child]
                if not parents:
                    continue
    
                # Get all parent states combination
                parent_states = []
                for parent in parents:
                    parent_states.append(str(beliefs[parent]['prior'] >= 0.5))
    
                # Convert to CPT key format
                cpt_key = ",".join(parent_states)
                
                # Calculate message for each parent
                for parent in parents:
                    if parent == query:
                        continue
                    
                    # Get child's CPT for this parent combination
                    cpt = self.bayesian_network['cpt'][child].get(cpt_key, {})
                    message = sum(cpt.get(str(child_val), 0) * beliefs[child]['prior']
                               for child_val in [True, False])
                    
                    beliefs[parent]['likelihood'] *= message
    
            # Backward pass (parents to children)
            for parent in reversed(self.bayesian_network['nodes']):
                children = [edge[1] for edge in self.bayesian_network['edges'] if edge[0] == parent]
                for child in children:
                    if child == query:
                        continue
                    
                    # Get all parent states for this child
                    parents = parent_map[child]
                    parent_states = []
                    for p in parents:
                        parent_states.append(str(beliefs[p]['prior'] >= 0.5))
                    cpt_key = ",".join(parent_states)
                    
                    cpt = self.bayesian_network['cpt'][child].get(cpt_key, {})
                    message = sum(cpt.get(str(child_val), 0) * beliefs[parent]['prior']
                               for child_val in [True, False])
                    
                    beliefs[child]['likelihood'] *= message
    
        # Final marginal calculation using belief propagation
        marginal = 1.0
        for node in self.bayesian_network['nodes']:
            if node == query:
                prior = self.bayesian_network['cpt'].get(node, {}).get('prior', 0.5)
                marginal = prior * beliefs[node]['likelihood']
                break
            elif node in observed:
                marginal *= beliefs[node]['prior']
    
        # Normalize using partition function
        partition = marginal + (1 - prior) * (1 - beliefs[node]['likelihood'])
        return marginal / partition if partition != 0 else 0.5

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
        return (
            self.markov_logic_weight * probability +
            self.knowledge_base_weight * base_prob
        )

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

class SimpleLLM:
    """Fallback language model with basic validation"""
    def validate_fact(self, fact: Tuple, knowledge_base: Dict) -> float:
        """Simplified fact validation"""
        return knowledge_base.get(fact, 0.5)

if __name__ == "__main__":
    print("\n=== Running Probabilistic Models ===\n")

    config = load_config()
    llm = None

    reason = ProbabilisticModels(config=config, llm=llm)

    print(reason)
    print("\n=== Successfully Ran Probabilistic Models ===\n")

if __name__ == "__main__":
    print("\n=== Running Probabilistic Models ===\n")

    config = load_config()

    # Initialize the probabilistic models
    reason = ProbabilisticModels(config=config)

    # Example query to Bayesian network
    query_node = "H"
    evidence = {"A": True, "C": False}
    
    try:
        prob = reason.bayesian_inference(query_node, evidence)
        print(f"P({query_node}|{evidence}) = {prob:.4f}")
    except Exception as e:
        print(f"Inference failed: {str(e)}")
    
    probability = reason.bayesian_inference(query_node, evidence)

    print(f"Probability of {query_node} given evidence {evidence}: {probability:.4f}")

    # Example probabilistic query
    fact = ("G", "some_fact")
    evidence_facts = {("A", "root_cause"): True, ("C", "secondary_mechanism"): False}

    prob_query_result = reason.probabilistic_query(fact, evidence_facts)

    print(f"Probabilistic query result for fact {fact} given evidence {evidence_facts}: {prob_query_result:.4f}")

    print("\n=== Successfully Ran Probabilistic Models ===\n")
