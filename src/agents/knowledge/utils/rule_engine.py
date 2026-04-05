import time
import os
import json
import yaml
import signal
import traceback
import multiprocessing

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, Optional, List, Tuple, Any
from functools import partial

from src.agents.knowledge.utils.knowledge_errors import RuleTimeoutError
from src.agents.knowledge.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Rule Engine")
printer = PrettyPrinter


# -----------------------------------------------------------------------------
# Worker function – runs in a separate process (must be at module level for pickling)
# -----------------------------------------------------------------------------
def _run_rule_code(rule_code: str, knowledge_base: Dict[Tuple[str, str, str], float]) -> Dict:
    """
    Execute rule code in a restricted environment.
    Returns inferred facts dict.
    """
    safe_globals = {
        '__builtins__': {
            'abs': abs, 'all': all, 'any': any, 'bool': bool,
            'dict': dict, 'enumerate': enumerate, 'float': float,
            'int': int, 'len': len, 'list': list, 'max': max,
            'min': min, 'range': range, 'round': round, 'set': set,
            'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip,
            'True': True, 'False': False, 'None': None,
        },
        'kb': knowledge_base,
    }
    local_ns = {'inferred': {}}
    exec(rule_code, safe_globals, local_ns)
    result = local_ns.get('inferred', {})
    if not isinstance(result, dict):
        raise TypeError(f"Rule returned {type(result)}, expected dict")
    return result


# -----------------------------------------------------------------------------
# Rule Engine - Production Ready (no daemonic nesting)
# -----------------------------------------------------------------------------
class RuleEngine:

    SECTORS = {"civic", "medical", "economic", "scientific", "philosophical", "technological"}

    def __init__(self):
        self.config = load_global_config()
        self.enabled = self.config.get('enabled', True)

        self.rule_config = get_config_section('rule_engine')
        self.verbose_logging = self.rule_config.get('verbose_logging', False)
        self.auto_discover = self.rule_config.get('auto_discover', True)
        self.min_rule_confidence = self.rule_config.get('min_rule_confidence', 0.6)
        self.slow_rule_threshold = self.rule_config.get('slow_rule_threshold', 0.5)
        self.rule_timeout = self.rule_config.get('rule_timeout_seconds', 1.0)
        self.max_concurrent_rules = self.rule_config.get('max_concurrent_rules', 4)
        self.rule_sources = self.rule_config.get('rule_sources', [])
        self.save_inferred = self.rule_config.get('save_inferred', False)
        self.rules_dir = self.rule_config.get('rules_dir', 'src/agents/knowledge/templates/')
        self.max_facts_per_rule = self.rule_config.get('max_facts_per_rule', 10)

        # Internal data structures
        self.rules: List[Dict] = []
        self.sector_rules: Dict[str, List[Dict]] = defaultdict(list)
        self.category_rules: Dict[str, List[Dict]] = defaultdict(list)

        # Failure tracking
        self.rule_failure_counts: Dict[str, int] = defaultdict(int)
        self.rule_timeout_counts: Dict[str, int] = defaultdict(int)
        self.rule_last_error: Dict[str, str] = {}

        # Process pool (non‑daemonic workers)
        self._executor = None

        # Load rules
        self.load_all_sectors()

    def _get_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.max_concurrent_rules)
        return self._executor

    def load_all_sectors(self):
        """Loads all sector rules from JSON files in rules_dir."""
        if not os.path.isdir(self.rules_dir):
            logger.warning(f"[RuleEngine] Rules directory not found: {self.rules_dir}")
            return

        loaded_count = 0
        for filename in os.listdir(self.rules_dir):
            if filename.endswith("_rules.json"):
                sector = filename.replace("_rules.json", "").lower()
                if sector in self.SECTORS:
                    path = os.path.join(self.rules_dir, filename)
                    try:
                        self.load_rules_from_json(path)
                        logger.info(f"[RuleEngine] Loaded {sector} rules from: {filename}")
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"[RuleEngine] Failed to load {sector} rules: {e}")
        if loaded_count == 0:
            logger.warning("[RuleEngine] No sector rule files loaded.")

    def load_rules_from_json(self, path: str):
        """Load rules from a JSON file and add them to the engine."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            rules = json.load(f)

        for rule_def in rules:
            if "implementation" not in rule_def:
                note = rule_def.get("symbolic_ai_implementation_note")
                if note:
                    rule_def["implementation"] = f"# symbolic placeholder\n# {note}"
                else:
                    logger.warning(f"Skipping rule with missing implementation: {rule_def.get('name', 'Unnamed')}")
                    continue

            name = rule_def["name"]
            code = rule_def["implementation"]
            weight = rule_def.get("weight", 1.0)
            tags = rule_def.get("tags", [])
            metadata = {
                "description": rule_def.get("description", ""),
                "source": rule_def.get("source", ""),
                "type": rule_def.get("type", "heuristic")
            }

            self.add_rule(
                name=name,
                rule_code=code,
                weight=weight,
                tags=tags,
                metadata=metadata
            )

    def add_rule(self,
                 name: str,
                 rule_code: str,
                 weight: float = 1.0,
                 tags: Optional[List[str]] = None,
                 metadata: Optional[Dict] = None):
        """Add a rule to the engine. rule_code is a string containing Python code."""
        if weight <= 0:
            raise ValueError("Weight must be positive")
        if not isinstance(rule_code, str):
            raise TypeError("rule_code must be a string")

        rule = {
            "name": name,
            "code": rule_code,
            "weight": weight,
            "tags": tags or [],
            "meta": metadata or {}
        }
        self.rules.append(rule)

        # Index by tags (sectors, categories)
        for tag in rule["tags"]:
            tag_lower = tag.lower()
            if tag_lower in self.SECTORS:
                self.sector_rules[tag_lower].append(rule)
            self.category_rules[tag_lower].append(rule)

        logger.debug(f"Added rule: {name} with weight {weight}")

    def get_rules_by_category(self, category: str) -> List[Dict]:
        return self.category_rules.get(category.lower(), [])

    def smart_apply(self, knowledge_base: dict, verbose=False) -> dict:
        """Applies rules from the most relevant sector, or all if none matches."""
        printer.status("ENGINE", "Applying rules", "info")
        sector = self.infer_sector(knowledge_base)

        if sector != "general" and sector in self.sector_rules and self.sector_rules[sector]:
            logger.info(f"[RuleEngine] Smart Apply: Using sector-specific rules for: {sector}")
            rules_to_use = self.sector_rules[sector]
        else:
            logger.info(f"[RuleEngine] Smart Apply: No specific sector match. Applying all rules.")
            rules_to_use = self.rules

        return self._apply_rules(rules_to_use, knowledge_base, verbose)

    def apply(self, knowledge_base: dict, verbose=False) -> dict:
        """Apply all rules in the engine."""
        return self._apply_rules(self.rules, knowledge_base, verbose)

    def apply_by_sector(self, knowledge_base: dict, sector: str, verbose=False) -> dict:
        """Apply rules belonging to a specific sector."""
        printer.status("ENGINE", "Applying by sectors", "info")
        rules = self.sector_rules.get(sector.lower(), [])
        return self._apply_rules(rules, knowledge_base, verbose, sector_tag=sector)

    def _apply_rules(self,
                     rules: List[Dict],
                     knowledge_base: dict,
                     verbose: bool = False,
                     sector_tag: Optional[str] = None) -> dict:
        """
        Execute a list of rules with sandboxing, timeouts, and failure tracking.
        Returns inferred facts dict (and optionally traces if verbose).
        """
        inferred = {}
        traces = []
        executor = self._get_executor()

        # Submit all rules in parallel (but still enforce per‑rule timeout)
        futures = []
        for rule in rules:
            future = executor.submit(_run_rule_code, rule["code"], knowledge_base)
            futures.append((rule, future, time.perf_counter()))

        for rule, future, start_time in futures:
            try:
                results = future.result(timeout=self.rule_timeout)
                exec_time = time.perf_counter() - start_time
                if exec_time > self.slow_rule_threshold:
                    logger.debug(f"Slow rule {rule['name']}: {exec_time:.2f}s")

                # Apply weight and confidence threshold
                for fact, conf in results.items():
                    weighted_conf = conf * rule["weight"]
                    if weighted_conf < self.min_rule_confidence:
                        continue
                    if fact not in inferred or weighted_conf > inferred[fact]:
                        inferred[fact] = weighted_conf
                        if verbose or self.verbose_logging:
                            traces.append({
                                "fact": fact,
                                "confidence": weighted_conf,
                                "rule": rule["name"],
                                "source": rule["meta"].get("source", "unknown"),
                                "sector": sector_tag
                            })

                # Reset failure counter on success
                self.rule_failure_counts[rule["name"]] = 0

            except FutureTimeoutError:
                self.rule_timeout_counts[rule["name"]] += 1
                self.rule_failure_counts[rule["name"]] += 1
                self.rule_last_error[rule["name"]] = f"Timeout after {self.rule_timeout}s"
                logger.warning(f"[RuleEngine] Rule {rule['name']} timed out after {self.rule_timeout}s")
                # Cancel the future if still running (best effort)
                future.cancel()

            except Exception as e:
                self.rule_failure_counts[rule["name"]] += 1
                self.rule_last_error[rule["name"]] = f"{type(e).__name__}: {e}"
                logger.warning(f"[RuleEngine] Rule {rule['name']} failed: {e}\n{traceback.format_exc()}")

        # Limit total inferred facts
        if self.max_facts_per_rule > 0 and len(inferred) > self.max_facts_per_rule:
            inferred = dict(sorted(inferred.items(), key=lambda x: x[1], reverse=True)[:self.max_facts_per_rule])

        return (inferred, traces) if verbose else inferred

    def infer_sector(self, knowledge_base: dict) -> str:
        """Determine the most likely sector based on keywords in the knowledge base."""
        printer.status("ENGINE", "Inferring sectors", "info")

        keywords = {
            "civic": {"law", "governance", "citizen", "vote", "policy", "courtroom", "privacy"},
            "medical": {"diagnosis", "symptom", "treatment", "patient", "disease", "medicine"},
            "economic": {"market", "inflation", "currency", "gdp", "employment", "money", "dollar", "recession"},
            "scientific": {"experiment", "hypothesis", "equation", "physics", "biology", "research"},
            "philosophical": {"ethics", "metaphysics", "ontology", "logic", "thought", "consciousness"},
            "technological": {"ai", "robot", "algorithm", "neural", "software", "agents", "data", "code"}
        }

        all_terms = set()
        for s, p, o in knowledge_base.keys():
            all_terms.add(str(s).lower())
            all_terms.add(str(p).lower())
            all_terms.add(str(o).lower())

        sector_scores = defaultdict(int)
        for sector, terms in keywords.items():
            for term in terms:
                if term in all_terms:
                    sector_scores[sector] += 1

        if not sector_scores:
            return "general"
        return max(sector_scores, key=sector_scores.get)

    def save_rules(self, path: str):
        """Export rule metadata (not code) to JSON for inspection."""
        with open(path, "w") as f:
            json.dump([{
                "name": r["name"],
                "weight": r["weight"],
                "tags": r["tags"],
                "meta": r["meta"]
            } for r in self.rules], f, indent=2)

    def get_failure_stats(self) -> Dict[str, Any]:
        """Return failure and timeout statistics for all rules."""
        return {
            "failures": dict(self.rule_failure_counts),
            "timeouts": dict(self.rule_timeout_counts),
            "last_errors": self.rule_last_error
        }

    def close(self):
        """Shut down the process pool."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self):
        self.close()


# -----------------------------------------------------------------------------
# Backward compatibility
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Rule Engine ===\n")
    printer.status("Init", "Rule Engine initialized", "success")

    engine = RuleEngine()
    print(f"{engine}")

    print("\n* * * * * Phase 2 - Rules * * * * *\n")
    knowledge_base = {
        ("AI", "is_a", "Technology"): 1.0,
        ("Socrates", "is_a", "Human"): 0.95,
    }
    smart_result = engine.smart_apply(knowledge_base=knowledge_base, verbose=False)
    sector_result = engine.apply_by_sector(knowledge_base=knowledge_base, sector="technology", verbose=False)

    printer.pretty("APPLY", smart_result, "success" if smart_result else "error")
    printer.pretty("SECTOR", sector_result, "success" if sector_result else "error")

    print("\n=== Successfully ran the Rule Engine ===\n")
    engine.close()
