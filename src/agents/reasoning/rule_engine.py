import itertools

from typing import Dict, Any
from collections import defaultdict

from src.agents.language.resource_loader import ResourceLoader

class RuleEngine:
    def _load_pragmatic_heuristics(self) -> Dict[str, Any]:
        """
        Loads pragmatic heuristics derived from discourse theory and conversational principles.
        Includes:
        - Gricean Maxims - Sentiment polarity cues - Discourse markers - Politeness strategies
        """
        sentiment_lexicon = ResourceLoader.get_sentiment_lexicon()
        return {
            "gricean_maxims": {
                "quantity": lambda utterance: len(utterance.split()) < 50,
                "quality": lambda utterance: "probably" not in utterance.lower(),
                "relation": lambda utterance: any(word in utterance.lower() for word in ["so", "because", "therefore"]),
                "manner": lambda utterance: not any(word in utterance for word in ["uh", "um", "like"])
            },
            "sentiment_polarity": {
                "positive": set(sentiment_lexicon["positive"].keys()),
                "negative": set(sentiment_lexicon["negative"].keys())
            },
            "discourse_markers": ["however", "in contrast", "furthermore", "therefore", "meanwhile"],
            "politeness_strategies": {
                "hedging": ["perhaps", "maybe", "I think"],
                "boosters": ["clearly", "absolutely", "definitely"]
            }
        }

    def _create_dependency_rules(self) -> Dict[str, list]:
        """Head-modifier grammar inspired by Universal Dependencies"""
        return {
            'nsubj': ['NN', 'VB'],
            'dobj': ['VB', 'NN'],
            'amod': ['NN', 'JJ'],
            'advmod': ['VB', 'RB'],
            'prep': ['IN', 'NN']
        }

    def _discover_new_rules(self, min_support: float = 0.3, min_confidence: float = 0.7):
        """
        Association rule mining with Apriori algorithm adaptation.
        Implements:
        - Agrawal & Srikant (1994) Fast Algorithms for Mining Association Rules
        - Confidence-weighted support from Lê et al. (2014) Fuzzy Association Rules
        """
        # Convert knowledge base to transaction-style format
        transactions = []
        for fact, conf in self.knowledge_base.items():
            if conf > 0.5:  # Consider facts with at least 50% confidence
                transactions.append(fact)

        # Generate frequent itemsets with confidence-weighted support
        itemsets = defaultdict(float)
        for fact in transactions:
            for element in itertools.chain.from_iterable(
                itertools.combinations(fact, r) for r in range(1, 4)
            ):
                itemsets[element] += self.knowledge_base.get(fact, 0.5)

        # Filter by minimum support
        freq_itemsets = {k: v/len(transactions) for k, v in itemsets.items() 
                        if v/len(transactions) >= min_support}

        # Generate candidate rules
        candidate_rules = []
        for itemset in freq_itemsets:
            if len(itemset) < 2:
                continue
            for i in range(1, len(itemset)):
                antecedent = itemset[:i]
                consequent = itemset[i:]
                candidate_rules.append((antecedent, consequent))

        # Calculate rule confidence
        valid_rules = []
        for ant, cons in candidate_rules:
            ant_support = sum(self.knowledge_base.get(fact, 0.0) 
                            for fact in transactions 
                            if set(ant).issubset(fact)) / len(transactions)
            
            rule_support = sum(self.knowledge_base.get(fact, 0.0)
                            for fact in transactions
                            if set(ant+cons).issubset(fact)) / len(transactions)
            
            if ant_support > 0:
                confidence = rule_support / ant_support
                if confidence >= min_confidence:
                    valid_rules.append({
                        'antecedent': ant,
                        'consequent': cons,
                        'confidence': confidence,
                        'support': rule_support
                    })

        # Convert to executable rules
        for rule in valid_rules:
            ant = rule['antecedent']
            cons = rule['consequent']
            
            def rule_func(kb, antecedents=ant, consequents=cons, conf=rule['confidence']):
                matches = [fact for fact in kb if all(e in fact for e in antecedents)]
                return {consequents: conf * len(matches)/(len(kb)+1e-8)}  # Prevent division by zero
                
            rule_name = f"LearnedRule_{hash(frozenset(ant+cons))}"
            self.add_rule(rule_func, rule_name, weight=rule['confidence'])
