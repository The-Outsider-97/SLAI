class RuleEngine:
    def __init__(self):
        self.rules = []

    def add_rule(self, rule_func, name: str, weight: float = 1.0):
        self.rules.append((name, rule_func, weight))

    def apply(self, knowledge_base: dict) -> dict:
        inferred = {}
        for name, rule, weight in self.rules:
            try:
                results = rule(knowledge_base)
                for fact, conf in results.items():
                    combined_conf = conf * weight
                    if fact not in knowledge_base or combined_conf > knowledge_base[fact]:
                        inferred[fact] = combined_conf
            except Exception as e:
                print(f"[RuleEngine] Rule {name} failed: {e}")
        return inferred
