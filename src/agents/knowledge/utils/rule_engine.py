
import time, os
import yaml, json

from collections import defaultdict
from typing import Dict, Union, Callable, Optional, List, Tuple

from src.agents.knowledge.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Rule Engine")
printer = PrettyPrinter

class RuleEngine:
    def __init__(self):
        self.config = load_global_config()
        self.enabled = self.config.get('enabled')

        self.rule_config = get_config_section('rule_engine')
        self.verbose_logging = self.rule_config.get('verbose_logging')
        self.auto_discover = self.rule_config.get('auto_discover')
        self.min_rule_confidence = self.rule_config.get('min_rule_confidence')
        self.slow_rule_threshold = self.rule_config.get('slow_rule_threshold')
        self.rule_sources = self.rule_config.get('rule_sources')
        self.save_inferred = self.rule_config.get('save_inferred')
        self.rules_dir = self.rule_config.get('rules_dir')
        self.max_facts_per_rule = self.rule_config.get('max_facts_per_rule')

        if 'slow_rule_threshold' not in self.rule_config:
            self.rule_config['slow_rule_threshold'] = 0.5  # Default 500ms
        self.sector_rules = defaultdict(list)
        self.rules = []
        self.sector_rules = {
            "civic": [],
            "medical": [],
            "economic": [],
            "scientific": [],
            "philosophical": [],
            "technological": []
        }
        self.load_all_sectors()

    SECTORS = {"civic", "medical", "economic", "scientific", "philosophical", "technological"}

    def load_all_sectors(self):
        """
        Loads all sector rules from the templates folder based on file naming conventions.
        E.g. civic_rules.json → sector 'civic'
        """
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

    def load_rules_from_json(self, path):
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

            def make_func(code_str):
                def rule(kb):
                    inferred = {}
                    exec(code_str, {}, {"kb": kb, "inferred": inferred})
                    return inferred
                return rule

            self.add_rule(
                name=name,
                rule_func=make_func(code),
                weight=weight,
                tags=tags,
                metadata=metadata
            )

    def add_rule(self,
                 name: str,
                 rule_func: Callable[[dict], Dict[str, float]],
                 weight: float = 1.0,
                 tags: Optional[List[str]] = None,
                 metadata: Optional[Dict] = None
                 ):
        # Validate inputs
        if weight <= 0:
            raise ValueError("Weight must be positive")
        if not callable(rule_func):
            raise TypeError("rule_func must be callable")
        
        rule = {
            "name": name,
            "func": rule_func,
            "weight": weight,
            "tags": tags or [],
            "meta": metadata or {}
        }
        self.rules.append(rule)
        logger.debug(f"Added rule: {name} with weight {weight} and tags {tags}")

        for tag in rule["tags"]:
            if (sector := tag.lower()) in self.SECTORS:
                self.sector_rules[sector].append(rule)

    def smart_apply(self, knowledge_base: dict, verbose=False) -> dict:
        """Applies rules from the most relevant sector or all if no specific match."""
        printer.status("ENGINE", "Applying rules", "info")

        sector = self.infer_sector(knowledge_base)
        rules_to_use = []
        
        if sector != "general" and sector in self.sector_rules and self.sector_rules[sector]:
            logger.info(f"[RuleEngine] Smart Apply: Using sector-specific rules for: {sector}")
            rules_to_use = self.sector_rules[sector]
        else:
            logger.info(f"[RuleEngine] Smart Apply: No specific sector match or no rules for sector '{sector}'. Applying all general rules.")
            rules_to_use = self.rules # Fallback to all rules

        original_rules = self.rules
        self.rules = rules_to_use
        result = self.apply(knowledge_base, verbose)
        self.rules = original_rules # Restore original rules
        return result

    def infer_sector(self, knowledge_base: dict) -> str:
        printer.status("ENGINE", "Inferring sectors", "info")

        keywords = {
            "civic": {"law", "governance", "citizen", "vote", "policy", "courtroom", "privacy"},
            "medical": {"diagnosis", "symptom", "treatment", "patient", "disease", "medicine"},
            "economic": {"market", "inflation", "currency", "gdp", "employment", "money", "dollar", "recession"},
            "scientific": {"experiment", "hypothesis", "equation", "physics", "biology", "research"},
            "philosophical": {"ethics", "metaphysics", "ontology", "logic", "thought", "consciousness"},
            "technological": {"ai", "robot", "algorithm", "neural", "software", "agents", "data", "code"}
        }
        # Consider predicates and objects as well for better sector inference
        all_terms_in_kb = set()
        for s, p, o in knowledge_base.keys():
            all_terms_in_kb.add(str(s).lower())
            all_terms_in_kb.add(str(p).lower())
            all_terms_in_kb.add(str(o).lower())

        sector_scores = defaultdict(int)
        for sector, terms in keywords.items():
            for term in terms:
                if term in all_terms_in_kb:
                    sector_scores[sector] +=1
        
        if not sector_scores:
            return "general"
            
        return max(sector_scores, key=sector_scores.get)

    def apply(self, knowledge_base: dict, verbose=False) -> dict:
        inferred = {}
        traces = []

        for rule in self.rules:
            start_time = time.perf_counter()
            exec_time = time.perf_counter() - start_time
            if exec_time > self.slow_rule_threshold:
                logger.debug(f"Slow rule {rule['name']}: {exec_time:.2f}s")
            try:
                results = rule["func"](knowledge_base)
                exec_time = time.perf_counter() - start_time
                if exec_time > self.slow_rule_threshold:
                    logger.debug(f"Slow rule {rule['name']}: {exec_time:.2f}s")
                for fact, conf in results.items():
                    weighted_conf = conf * rule["weight"]
                    if fact not in knowledge_base or weighted_conf > knowledge_base.get(fact, 0):
                        inferred[fact] = weighted_conf
                        if verbose or self.verbose_logging:
                            traces.append({
                                "fact": fact,
                                "confidence": weighted_conf,
                                "rule": rule["name"],
                                "source": rule["meta"].get("source", "unknown")
                            })
            except Exception as e:
                logger.warning(f"[RuleEngine] Rule {rule['name']} failed: {e}")

        return (inferred, traces) if verbose else inferred

    def apply_by_sector(self, knowledge_base: dict, sector: str, verbose=False) -> dict:
        printer.status("ENGINE", "Applying by sectors", "info")

        inferred = {}
        traces = []

        rules = self.sector_rules.get(sector.lower(), [])
        for rule in rules:
            try:
                results = rule["func"](knowledge_base)
                for fact, conf in results.items():
                    weighted_conf = conf * rule["weight"]
                    if fact not in knowledge_base or weighted_conf > knowledge_base.get(fact, 0):
                        inferred[fact] = weighted_conf
                        if verbose or self.verbose_logging:
                            traces.append({
                                "fact": fact,
                                "confidence": weighted_conf,
                                "rule": rule["name"],
                                "source": rule["meta"].get("source", "unknown"),
                                "sector": sector
                            })
            except Exception as e:
                logger.warning(f"[RuleEngine] [{sector}] Rule {rule['name']} failed: {e}")

        return (inferred, traces) if verbose else inferred

    def _should_log(self, verbose: bool) -> bool:
        return verbose or self.verbose_logging

    def save_rules(self, path):
        with open(path, "w") as f:
            json.dump([{
                "name": r["name"],
                "weight": r["weight"],
                "tags": r["tags"],
                "meta": r["meta"]
            } for r in self.rules], f, indent=2)

    def _human_mortality_rule_impl(self, kb: dict, rule_weight: float) -> Dict[Tuple[str,str,str], float]:
        """If (X is_a Human), then infer (X is_mortal True)."""
        inferred_facts = {}
        for (s, p, o), conf in kb.items():
            if p == "is_a" and o == "Human":
                new_fact = (s, "is_mortal", "True") # Represent boolean as string for consistency
                new_confidence = conf * rule_weight
                if new_confidence > inferred_facts.get(new_fact, -1.0) and new_confidence > self.min_rule_confidence:
                   inferred_facts[new_fact] = new_confidence
        return inferred_facts

if __name__ == "__main__":
    print("\n=== Running Rule Engine ===\n")
    printer.status("Init", "Rule Engine initialized", "success")

    engine = RuleEngine()
    print(f"{engine}")

    print("\n* * * * * Phase 2 - Rules * * * * *\n")
    knowledge_base={
        ("AI", "is_a", "Technology"): 1.0,
        ("Socrates", "is_a", "Human"): 0.95,
    }
    verbose=False
    smart = engine.smart_apply(knowledge_base=knowledge_base, verbose=False)
    sectors = engine.apply_by_sector(knowledge_base=knowledge_base, sector="technology", verbose=False)

    printer.pretty("APPLY", smart, "success" if smart else "error")
    printer.pretty("SECTOR", sectors, "success" if sectors else "error")

    print("\n=== Successfully ran the Rule Engine ===\n")
