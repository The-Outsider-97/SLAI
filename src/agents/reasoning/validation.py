"""
Validation utilities for ReasoningAgent:
- Circular rule detection
- Contradiction/conflict detection
- Redundancy checking
"""

from typing import List, Tuple, Dict, Set


def detect_circular_rules(rules: List[Tuple[str, callable, float]], max_depth: int = 3) -> List[str]:
    """
    Detect circular dependencies between rules by simulating dependency chains.
    
    Args:
        rules: List of (rule_name, rule_func, weight) tuples
        max_depth: Maximum recursion depth to check

    Returns:
        List of rule names involved in circular logic
    """
    rule_dependencies: Dict[str, Set[str]] = {}

    # Extract rule output dependencies
    for name, rule, _ in rules:
        try:
            inferred = rule({})
            for fact in inferred:
                if isinstance(fact, tuple) and len(fact) == 3:
                    subj, pred, obj = fact
                    if name not in rule_dependencies:
                        rule_dependencies[name] = set()
                    rule_dependencies[name].add(obj)
        except Exception:
            continue

    def has_cycle(start: str, visited: Set[str], depth: int) -> bool:
        if depth > max_depth:
            return False
        visited.add(start)
        for dep in rule_dependencies.get(start, []):
            if dep in visited or has_cycle(dep, visited.copy(), depth + 1):
                return True
        return False

    circular_rules = [rule for rule in rule_dependencies if has_cycle(rule, set(), 0)]
    return circular_rules


def detect_fact_conflicts(facts: Dict[Tuple, float], threshold: float = 0.5) -> List[Tuple]:
    """
    Identify conflicting facts with high confidence.

    Args:
        facts: Dictionary of (subject, predicate, object): confidence
        threshold: Minimum confidence to consider a fact contradictory

    Returns:
        List of contradictory fact tuples
    """
    inverse_facts = {}
    conflicts = []

    for (subj, pred, obj), conf in facts.items():
        if conf < threshold:
            continue
        inverse = (subj, pred, f"not_{obj}")
        if inverse in facts and facts[inverse] >= threshold:
            conflicts.append(((subj, pred, obj), (subj, pred, f"not_{obj}")))

    return conflicts


def redundant_fact_check(derived: Dict[Tuple, float], stored: Dict[Tuple, float], margin: float = 0.01) -> List[Tuple]:
    """
    Check for redundant or overly similar facts.

    Args:
        derived: Newly inferred facts
        stored: Existing facts in KB
        margin: Confidence difference below which they are considered redundant

    Returns:
        List of fact tuples that are redundant
    """
    redundant = []
    for fact, conf in derived.items():
        if fact in stored:
            if abs(conf - stored[fact]) <= margin:
                redundant.append(fact)
    return redundant
