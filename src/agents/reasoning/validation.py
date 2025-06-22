"""
Validation utilities for ReasoningAgent:
- Circular rule detection
- Contradiction/conflict detection
- Redundancy checking
"""
import yaml, json
import traceback
import time

from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional, Any

from sentence_transformers import util

from src.agents.reasoning.utils.config_loader import load_global_config, get_config_section
from src.agents.reasoning.utils.mln_rules import mln_rules
from src.agents.reasoning.reasoning_memory import ReasoningMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Validation Engine")
printer = PrettyPrinter

class ValidationEngine:
    def __init__(self):
        self.config = load_global_config()
        self.contradiction_threshold = self.config.get('contradiction_threshold')
        self.markov_logic_weight = self.config.get('markov_logic_weight')

        self.validation_config = get_config_section('validation')
        self.enable = self.validation_config.get('enable')
        self.redundancy_margin = self.validation_config.get('redundancy_margin')
        self.max_circular_depth = self.validation_config.get('max_circular_depth')
        self.validation_timeout = self.validation_config.get('validation_timeout')
        self.min_soundness_score = self.validation_config.get('min_soundness_score')
        self.max_validation_attempts = self.validation_config.get('max_validation_attempts')
        self.mln_rule_confidence_threshold = self.validation_config.get('mln_rule_confidence_threshold')

        self.storage_config = get_config_section('storage')
        self.knowledge_db_path = self.storage_config.get('knowledge_db')

        # Initialize core components
        self.knowledge_base = self._load_knowledge_base(Path(self.knowledge_db_path))

        self.reasoning_memory = ReasoningMemory()
        # Initialize semantic similarity model
        self.semantic_model = None
        if self.validation_config.get('enable_semantic_redundancy'):  # Safe get with default
            self._initialize_semantic_model()

        logger.info(f"Validation Initialized with {len(self.knowledge_base)} facts")

    def _load_knowledge_base(self, kb_path: Path) -> Dict[Tuple, float]:
        """Load knowledge base from configured path with robust parsing"""
        if not kb_path.exists():
            logger.error(f"Knowledge base file not found: {kb_path}")
            return {}
            
        with open(kb_path, 'r') as f:
            data = json.load(f)
            
        # Handle different KB formats
        if isinstance(data, dict) and 'knowledge' in data:
            facts = data['knowledge']
        elif isinstance(data, list):
            facts = data
        else:
            logger.error("Unsupported KB format")
            return {}
        
        processed_kb = {}
        for fact in facts:
            try:
                # Handle dictionary format
                if isinstance(fact, dict):
                    s = fact.get("subject")
                    p = fact.get("predicate")
                    o = fact.get("object")
                    weight = fact.get("weight", 0.5)
                    key = (s, p, o)
                    processed_kb[key] = float(weight)
                    
                # Handle list format
                elif isinstance(fact, list) and len(fact) >= 3:
                    s, p, o = fact[:3]
                    weight = fact[3] if len(fact) >= 4 else 0.5
                    key = (s, p, o)
                    processed_kb[key] = float(weight)
            except Exception as e:
                logger.warning(f"Skipping invalid fact: {fact} - {str(e)}")
                
        return processed_kb

    def _initialize_semantic_model(self):
        """Initialize semantic similarity model with error handling"""
        try:
            from src.agents.perception.modules.transformer import Transformer
            self.semantic_model = Transformer()
            logger.info("Initialized semantic similarity model for redundancy checks")
        except Exception as e:
            logger.error(f"Failed to initialize semantic model: {str(e)}")
            self.semantic_model = None

    def validate_all(self, rules: List[Tuple[str, callable, float]], new_facts: Dict[Tuple, float]) -> Dict:
        """Execute full validation pipeline"""
        start_time = time.time()
        results = {
            'circular_rules': [],
            'conflicts': [],
            'redundancies': [],
            'sound_rules': [],
            'validation_status': 'pending'
        }

        try:
            # Phase 1: Rule validation
            results['circular_rules'] = self.detect_circular_rules(rules)
            results['sound_rules'] = self.check_rule_soundness(rules)

            # Phase 2: Fact validation
            results['conflicts'] = self.detect_fact_conflicts(new_facts)
            results['redundancies'] = self.check_redundancies(new_facts)

            # Phase 3: KB consistency check
            combined_facts = {**self.knowledge_base, **new_facts}
            results['consistency'] = self.validate_knowledge_base_consistency(combined_facts)

            results['validation_status'] = 'success'
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            results['validation_status'] = f'failed: {str(e)}'

        results['execution_time'] = time.time() - start_time
        self.reasoning_memory.add(
            experience={
                "type": "validation_report",
                "results": results,
                "rules": [name for name, _, _ in rules],
                "new_facts_count": len(new_facts)
            },
            tag="validation",
            priority=0.9  # High priority for validation results
        )
        return results

    def detect_circular_rules(self, rules: List[Tuple[str, callable, float]]) -> List[str]:
        """Enhanced circular dependency detection with depth tracking"""
        rule_graph = defaultdict(set)
        rule_names = {name for name, _, _ in rules}
    
        # Build rule dependency graph only using known rule names
        for name, rule, _ in rules:
            try:
                outputs = rule({})
                for fact in outputs:
                    if isinstance(fact, tuple) and len(fact) == 3 and fact[2] in rule_names:
                        rule_graph[name].add(fact[2])  # Only connect to other rule names
            except Exception as e:
                logger.warning(f"Rule {name} failed execution: {str(e)}")
                continue
    
        # Initialize in-degree with all rule names
        in_degree = {node: 0 for node in rule_names}
        for node, deps in rule_graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
    
        # Topological sort
        queue = [node for node, degree in in_degree.items() if degree == 0]
        topological_order = []
    
        while queue:
            node = queue.pop(0)
            topological_order.append(node)
            for neighbor in rule_graph.get(node, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
    
        # Remaining nodes with in-degree > 0 are part of a cycle
        circular_nodes = [node for node in in_degree if in_degree[node] > 0]
        return circular_nodes

    def detect_fact_conflicts(self, new_facts: Dict[Tuple, float]) -> List[Tuple]:
        """Multi-dimensional conflict detection"""
        conflicts = []
        threshold = self.contradiction_threshold

        # Direct contradictions
        for (s, p, o), conf in new_facts.items():
            if conf < threshold:
                continue
            inverse_fact = (s, p, f"not_{o}")
            if inverse_fact in self.knowledge_base:
                kb_conf = self.knowledge_base[inverse_fact]
                if abs(conf - kb_conf) > threshold:
                    conflicts.append(((s, p, o), inverse_fact))

        # Semantic contradictions using embeddings
        if self.semantic_model:
            semantic_conflicts = self._detect_semantic_conflicts(new_facts)
            conflicts.extend(semantic_conflicts)

        return conflicts

    def check_redundancies(self, new_facts: Dict[Tuple, float]) -> List[Tuple]:
        """Multi-modal redundancy detection"""
        redundancies = []
        margin = self.redundancy_margin

        # Exact match redundancies
        for fact, conf in new_facts.items():
            if fact in self.knowledge_base:
                if abs(conf - self.knowledge_base[fact]) <= margin:
                    redundancies.append(fact)

        # Semantic redundancies
        if self.semantic_model:
            semantic_redundancies = self._detect_semantic_redundancies(new_facts)
            redundancies.extend(semantic_redundancies)

        return list(set(redundancies))

    def check_rule_soundness(self, rules: List[Tuple[str, callable, float]]) -> Dict:
        """Validate rules against current knowledge base"""
        sound_rules = []
        unsound_rules = []
        min_score = self.min_soundness_score

        for name, rule, weight in rules:
            try:
                inferred = rule(self.knowledge_base)
                match_score = self._calculate_rule_match(inferred)
                if match_score >= min_score:
                    sound_rules.append((name, match_score))
                else:
                    unsound_rules.append((name, match_score))
            except Exception as e:
                logger.warning(f"Rule {name} validation failed: {str(e)}")
                unsound_rules.append((name, 0.0))

        return {
            'sound': sorted(sound_rules, key=lambda x: -x[1]),
            'unsound': sorted(unsound_rules, key=lambda x: -x[1])
        }

    def validate_knowledge_base_consistency(self, kb: Dict[Tuple, float]) -> Dict:
        """Full KB consistency check with retry logic"""
        attempts = 0
        max_attempts = self.max_validation_attempts
        timeout = self.validation_timeout

        while attempts < max_attempts:
            try:
                return self._perform_consistency_check(kb)
            except Exception as e:
                logger.warning(f"Consistency check attempt {attempts+1} failed: {str(e)}")
                attempts += 1
                time.sleep(timeout)
        
        raise RuntimeError(f"Consistency check failed after {max_attempts} attempts")

    def _perform_consistency_check(self, kb: Dict[Tuple, Any]) -> Dict:
        """Core consistency validation logic with type safety"""
        consistency_report = {
            'total_facts': len(kb),
            'contradictions': [],
            'redundancies': [],
            'confidence_violations': [],
            'markov_violations': []
        }

        # Confidence boundary checks with type validation
        for fact, conf in kb.items():
            try:
                conf_val = float(conf)
                if not (0.0 <= conf_val <= 1.0):
                    consistency_report['confidence_violations'].append(fact)
            except (TypeError, ValueError):
                consistency_report['confidence_violations'].append(fact)

        # Cross-validation with Markov logic network
        mlw = self.markov_logic_weight
        consistency_report['markov_violations'] = self._validate_with_markov_logic(kb, mlw)

        return consistency_report

    def _detect_semantic_conflicts(self, new_facts: Dict[Tuple, float]) -> List[Tuple]:
        """Use semantic similarity to detect non-literal conflicts"""
        conflicts = []
        fact_embeddings = {}

        # Create embeddings for all facts
        for fact in new_facts:
            text = ' '.join(fact)
            fact_embeddings[fact] = self.semantic_model.encode(text)

        # Compare against KB facts
        for new_fact, new_emb in fact_embeddings.items():
            for kb_fact, kb_emb in self.knowledge_base.items():
                similarity = util.pytorch_cos_sim(new_emb, kb_emb).item()
                if similarity > 0.8 and abs(new_facts[new_fact] - self.knowledge_base[kb_fact]) > 0.3:
                    conflicts.append((new_fact, kb_fact))

        return conflicts

    def _detect_semantic_redundancies(self, new_facts: Dict[Tuple, float]) -> List[Tuple]:
        """Find semantically similar facts using embeddings"""
        redundancies = []
        threshold = 0.95

        for fact in new_facts:
            fact_text = ' '.join(fact)
            fact_embedding = self.semantic_model.encode(fact_text)
            
            for kb_fact in self.knowledge_base:
                kb_text = ' '.join(kb_fact)
                kb_embedding = self.semantic_model.encode(kb_text)
                similarity = util.pytorch_cos_sim(fact_embedding, kb_embedding).item()
                
                if similarity > threshold:
                    redundancies.append(fact)
                    break

        return redundancies

    def _calculate_rule_match(self, inferred: Dict) -> float:
        """Calculate how well rule outputs match existing knowledge"""
        if not inferred:
            return 0.0
        
        total_score = 0.0
        for fact, confidence in inferred.items():
            kb_confidence = self.knowledge_base.get(fact, 0.0)
            total_score += 1 - abs(confidence - kb_confidence)
        
        return total_score / len(inferred)

    def _validate_with_markov_logic(self, kb: Dict[Tuple, Any], weight: float) -> List[Tuple]:
        """Validate using imported MLN-style soft rules with type safety"""
        violations = []
        confidence_threshold = self.mln_rule_confidence_threshold
    
        # Use the imported mln_rules list
        for rule_def in mln_rules:
            rule_id = rule_def.get("id", "UnknownRule")
            description = rule_def.get("description", "No description")
            lambda_func = rule_def.get("lambda_rule")
    
            if not callable(lambda_func):
                logger.warning(f"Rule {rule_id} does not have a callable lambda_rule. Skipping.")
                continue
    
            try:
                if lambda_func(kb, confidence_threshold):
                    violations.append((f"MLN_VIOLATION ({rule_id})", description))
            except Exception as e:
                logger.warning(f"Error executing MLN rule check: {str(e)}")
    
        # Soft logic: additionally penalize any contradictory logic below threshold
        for (s, p, o), conf in kb.items():
            try:
                conf_val = float(conf)
                if o.startswith("not_"):
                    positive_form = (s, p, o.replace("not_", ""))
                    if positive_form in kb:
                        pos_conf = float(kb[positive_form])
                        if abs(pos_conf - conf_val) > self.contradiction_threshold:
                            violations.append(((s, p, o), f"Conflicting with {positive_form}"))
            except (TypeError, ValueError):
                continue  # Skip invalid confidence values
    
        return violations

if __name__ == "__main__":
    print("\n=== Running Validation Engine ===\n")
    # Create sample knowledge base for testing
    sample_kb = {
        # Facts for circular rule testing
        ("RuleA", "depends_on", "RuleB"): 0.9,
        ("RuleB", "depends_on", "RuleC"): 0.8,
        ("RuleC", "depends_on", "RuleA"): 0.7,  # Circular dependency
        
        # Facts for contradiction testing
        ("Socrates", "is_alive", "True"): 0.9,
        ("Socrates", "is_alive", "False"): 0.95,  # Direct contradiction
        ("Water", "state_is", "Liquid"): 0.8,
        ("Water", "state_is", "Solid"): 0.85,     # Semantic contradiction
        
        # Facts for redundancy testing
        ("Earth", "shape", "Round"): 0.99,
        ("Earth", "shape", "Spherical"): 0.98,    # Semantic redundancy
        ("Pi", "value", "3.14"): 0.95,
        
        # Facts for MLN rule testing
        ("Bob", "is_married_to", "Alice"): 0.9,
        ("Bob", "is_married_to", "Carol"): 0.85,  # Bigamy violation
        ("Engine", "is_part_of", "Car"): 1.0,
        ("Car", "is_part_of", "Engine"): 0.8,     # Part-whole violation
    }

    # Create sample rules for testing
    def rule1(kb):
        return {("Earth", "shape", "Round"): 0.95}
    
    def rule2(kb):
        return {("RuleA", "depends_on", "RuleB"): 0.9}
    
    def circular_rule(kb):
        return {("RuleC", "depends_on", "RuleA"): 0.8}
    
    sample_rules = [
        ("Rule1", rule1, 0.9),
        ("Rule2", rule2, 0.8),
        ("CircularRule", circular_rule, 0.7)
    ]

    # Initialize validation engine
    validation = ValidationEngine()
    print("Validation Engine initialized with test knowledge base\n")

    # Test 1: Circular Rule Detection
    print("=== Testing Circular Rule Detection ===")
    circular_rules = validation.detect_circular_rules(sample_rules)
    print(f"Circular rules detected: {circular_rules}\n")
    
    # Test 2: Fact Conflict Detection
    print("=== Testing Fact Conflict Detection ===")
    new_facts = {
        ("Socrates", "is_alive", "True"): 0.91,  # Contradicts existing fact
        ("Water", "state_is", "Gas"): 0.75       # New fact
    }
    conflicts = validation.detect_fact_conflicts(new_facts)
    print("Conflicts found:")
    for conflict in conflicts:
        print(f"- {conflict[0]} vs {conflict[1]}")
    print()
    
    # Test 3: Redundancy Checking
    print("=== Testing Redundancy Checking ===")
    redundancies = validation.check_redundancies({
        ("Earth", "shape", "Round"): 0.96,      # Nearly identical
        ("Pi", "value", "3.14159"): 0.9,        # New fact
        ("Earth", "shape", "OblateSpheroid"): 0.7  # Semantic similarity
    })
    print("Redundant facts found:")
    for redundant in redundancies:
        print(f"- {redundant}")
    print()
    
    # Test 4: Rule Soundness Checking
    print("=== Testing Rule Soundness Checking ===")
    soundness = validation.check_rule_soundness(sample_rules)
    print("Sound rules:")
    for rule, score in soundness['sound']:
        print(f"- {rule} (score: {score:.2f})")
    print("\nUnsound rules:")
    for rule, score in soundness['unsound']:
        print(f"- {rule} (score: {score:.2f})")
    print()
    
    # Test 5: Full Validation Pipeline
    print("=== Testing Full Validation Pipeline ===")
    full_results = validation.validate_all(sample_rules, new_facts)
    print("Validation Results:")
    print(f"- Status: {full_results['validation_status']}")
    print(f"- Execution Time: {full_results['execution_time']:.2f}s")
    print("- Circular Rules:", full_results['circular_rules'])
    print("- Conflicts:", len(full_results['conflicts']), "found")
    print("- Redundancies:", len(full_results['redundancies']), "found")
    print("- Sound Rules:", len(full_results['sound_rules']['sound']))
    print("- KB Consistency Issues:", len(full_results['consistency'].get('markov_violations', [])))

    # Test 6: MLN Rule Validation
    print("\n=== Testing MLN Rule Validation ===")
    mln_violations = validation._validate_with_markov_logic(sample_kb, 0.7)
    print("MLN Rule Violations Found:")
    for violation in mln_violations:
        print(f"- {violation[0]}: {violation[1]}")

    print(validation)
    print("\n=== Successfully Validation Engine ===\n")
