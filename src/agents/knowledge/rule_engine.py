
import time, os
import yaml, json
import traceback

from collections import defaultdict
from typing import Dict, Union, Callable, Optional, List
from types import SimpleNamespace

from logs.logger import get_logger

logger = get_logger("Rule Engine")

CONFIG_PATH = "src/agents/knowledge/configs/knowledge_config.yaml"

def dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d

def get_config_section(section: Union[str, Dict], config_file_path: str):
    if isinstance(section, dict):
        return dict_to_namespace(section)
    
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    if section not in config:
        raise KeyError(f"Section '{section}' not found in config file: {config_file_path}")
    return dict_to_namespace(config[section])

class RuleEngine:
    def __init__(self,
                 config_section_name: str = "rule_engine",
                 config_file_path: str = CONFIG_PATH
                 ):
        self.config = get_config_section(config_section_name, config_file_path)
        if not hasattr(self.config, 'slow_rule_threshold'):
            self.config.slow_rule_threshold = 0.5  # Default 500ms
        self.sector_rules = defaultdict(list)
        self.rules = []
        self.sector_rules = {
            "civic": [],         # src/agents/knowledge/templates/civic_rules.json
            "medical": [],       # src/agents/knowledge/templates/medical_rules.json
            "economic": [],      # src/agents/knowledge/templates/economic_rules.json
            "scientific": [],    # src/agents/knowledge/templates/scientific_rules.json
            "philosophical": [], # src/agents/knowledge/templates/philosophical_rules.json
            "technological": []  # src/agents/knowledge/templates/tech_rules.json AI-centered
        }
        self.load_all_sectors()

    SECTORS = {"civic", "medical", "economic", "scientific", "philosophical", "technological"}

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

        for tag in rule["tags"]:
            if (sector := tag.lower()) in self.SECTORS:
                self.sector_rules[sector].append(rule)

    def smart_apply(self, knowledge_base: dict, verbose=False) -> dict:
        """
        Applies rules from the most relevant sector based on the context in the KB.
        Falls back to all rules if sector is unknown or has no specific matches.
        """
        sector = self.infer_sector(knowledge_base)
        if sector in self.sector_rules and self.sector_rules[sector]:
            logger.info(f"[RuleEngine] Applying sector-specific rules for: {sector}")
            return self.apply_by_sector(knowledge_base, sector, verbose=verbose)
        else:
            logger.info(f"[RuleEngine] No specific sector match. Applying all rules.")
            return self.apply(knowledge_base, verbose=verbose)

    def infer_sector(self, knowledge_base: dict) -> str:
        """
        Attempt to infer which sector is most relevant based on keys or tags in the knowledge base.
        Returns sector name or 'general' if unknown.
        """
        keywords = {
            "civic": {"law", "governance", "citizen", "vote", "policy", "courtroom", "privacy"},
            "medical": {"diagnosis", "symptom", "treatment", "patient", "disease", "medicine"},
            "economic": {"market", "inflation", "currency", "gdp", "employment", "moneny", "dollar", "recession"},
            "scientific": {"experiment", "hypothesis", "equation", "physics", "biology"},
            "philosophical": {"ethics", "metaphysics", "ontology", "logic", "thought"},
            "technological": {"ai", "robot", "algorithm", "neural", "software", "agents"}
        }

        kb_keys = set(map(str.lower, knowledge_base.keys()))
        for sector, terms in keywords.items():
            if kb_keys & terms:
                return sector

        return "general"

    def apply_by_sector(self, knowledge_base: dict, sector: str, verbose=False) -> dict:
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
                        if verbose or self.config.verbose_logging:
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
        return verbose or self.config.verbose_logging

    def apply(self, knowledge_base: dict, verbose=False) -> dict:
        inferred = {}
        traces = []

        for rule in self.rules:
            start_time = time.perf_counter()
            try:
                results = rule["func"](knowledge_base)
                exec_time = time.perf_counter() - start_time
                if exec_time > self.config.slow_rule_threshold:
                    logger.debug(f"Slow rule {rule['name']}: {exec_time:.2f}s")
                for fact, conf in results.items():
                    weighted_conf = conf * rule["weight"]
                    if fact not in knowledge_base or weighted_conf > knowledge_base.get(fact, 0):
                        inferred[fact] = weighted_conf
                        if verbose or self.config.verbose_logging:
                            traces.append({
                                "fact": fact,
                                "confidence": weighted_conf,
                                "rule": rule["name"],
                                "source": rule["meta"].get("source", "unknown")
                            })
            except Exception as e:
                logger.warning(f"[RuleEngine] Rule {rule['name']} failed: {e}")

        return (inferred, traces) if verbose else inferred

    def save_rules(self, path):
        with open(path, "w") as f:
            json.dump([{
                "name": r["name"],
                "weight": r["weight"],
                "tags": r["tags"],
                "meta": r["meta"]
            } for r in self.rules], f, indent=2)

    def load_all_sectors(self, rules_dir="src/agents/knowledge/templates"):
        """
        Loads all sector rules from the templates folder based on file naming conventions.
        E.g. civic_rules.json → sector 'civic'
        """
        loaded_count = 0
        for filename in os.listdir(rules_dir):
            if filename.endswith("_rules.json"):
                sector = filename.replace("_rules.json", "").lower()
                if sector in self.SECTORS:
                    path = os.path.join(rules_dir, filename)
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



if __name__ == "__main__":
    import readline

    engine = RuleEngine()
    print("\n🧠 Rule Engine Interactive Console")
    print("Commands: load <path>, kb <json>, apply, sector <sector>, smart, reset, list, exit\n")

    kb = {}

    while True:
        try:
            cmd = input(">> ").strip()

            if cmd.startswith("load "):
                path = cmd.split(" ", 1)[1].strip()
                engine.load_rules_from_json(path)
                print(f"✅ Loaded rules from {path}")

            elif cmd == "list":
                print("\n📜 Registered Rules:")
                for rule in engine.rules:
                    print(f"- {rule['name']} ({', '.join(rule['tags'])})")
                print()

            elif cmd.startswith("kb "):
                kb_input = cmd[3:].strip()
                try:
                    kb = json.loads(kb_input)
                    print(f"🧾 Loaded KB with {len(kb)} entries.")
                except Exception as e:
                    print(f"⚠️ Invalid JSON: {e}")

            elif cmd == "apply":
                inferred, traces = engine.apply(kb, verbose=True)
                print(f"\n✅ Inferred: {json.dumps(inferred, indent=2)}")
                print(f"📈 Traces:")
                for t in traces:
                    print(f"- {t['rule']} → {t['fact']} ({t['confidence']:.2f})")

            elif cmd.startswith("sector "):
                sector = cmd.split(" ", 1)[1].strip().lower()
                if sector not in engine.SECTORS:
                    print(f"⚠️ Unknown sector: {sector}")
                    continue
                inferred, traces = engine.apply_by_sector(kb, sector, verbose=True)
                print(f"\n✅ Inferred ({sector}): {json.dumps(inferred, indent=2)}")
                print(f"📈 Traces:")
                for t in traces:
                    print(f"- {t['rule']} → {t['fact']} ({t['confidence']:.2f})")

            elif cmd == "smart":
                inferred, traces = engine.smart_apply(kb, verbose=True)
                print(f"\n✅ Smart Inference: {json.dumps(inferred, indent=2)}")
                print(f"📈 Traces:")
                for t in traces:
                    print(f"- {t['rule']} → {t['fact']} ({t['confidence']:.2f})")

            elif cmd == "reset":
                kb = {}
                print("🧼 KB reset.")

            elif cmd == "exit":
                print("👋 Goodbye.")
                break

            else:
                print("❓ Unknown command. Try: load, kb, apply, sector, smart, list, reset, exit")

        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break
