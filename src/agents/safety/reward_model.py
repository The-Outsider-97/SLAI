
import re, sys
import yaml, json
import numpy as np

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QApplication
from datetime import timedelta, datetime
from typing import Dict, List, Callable, Union

from src.agents.evaluators.utils.report import get_visualizer
from src.agents.safety.secure_memory import SecureMemory
from logs.logger import get_logger

logger = get_logger("Security Reward Model")

CONFIG_PATH = "src/agents/safety/configs/secure_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class RewardModel:
    def __init__(self, config):
        config = load_config() or {}
        self.config = config.get('reward_model', {})
        memory = SecureMemory(config)
        self.memory = memory

        self.rules = {
            "alignment": lambda x: 1 - x.count("harm") / max(1, len(x)),
            "helpfulness": lambda x: x.count("assist") / max(1, len(x))
        }

        logger.info(f"Succesfully initialize Secure Reward Model with:")
        logger.info(f"\n{self.rules}")

        # Initialize rule-based and learned components
        self.rule_based = self._init_rule_based_system()
        self.learned_model = self._init_learned_model()
        self.rule_weights = self._load_rule_weights()
        
        logger.info("Security Reward Model initialized with:")
        logger.info(f"Rule-based components: {list(self.rule_based.keys())}")
        logger.info(f"Rule weights: {self.rule_weights}")

    def _init_rule_based_system(self) -> Dict[str, Callable]:
        """Initialize predefined security rules with secure storage"""
        rules = {
            "alignment": self._alignment_score,
            "helpfulness": self._helpfulness_score,
            "privacy": self._privacy_score,
            "safety": self._safety_score,
            "truthfulness": self._truthfulness_score
        }

        # Store rule definitions in secure memory
        for name, func in rules.items():
            self.memory.add(
                {"name": name, "definition": func.__doc__},
                tags=["reward_model", "rule_definition"],
                sensitivity=0.7
            )

        return rules

    def _init_learned_model(self):
        """Initialize simple regression model for human feedback"""
        self.regression_model = None
        self.feature_names = list(self.rule_based.keys())
        return self._predict_learned  # Return prediction function

    def _predict_learned(self, rule_scores: Dict[str, float]) -> float:
        """Predict score using regression model"""
        if not self.regression_model:
            return 0.0

        features = [rule_scores[name] for name in self.feature_names]
        return self.regression_model.predict([features])[0]

    def retrain_model(self, training_data: List[Dict]):
        """Retrain with human feedback data"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("scikit-learn not available for retraining")
            return

        # Prepare training data
        X, y = [], []
        for sample in training_data:
            if "model_scores" in sample and "human_rating" in sample:
                try:
                    features = [sample["model_scores"][k] for k in self.feature_names]
                    X.append(features)
                    y.append(sample["human_rating"])
                except KeyError:
                    continue

        if len(X) < 10:
            logger.warning(f"Insufficient training data: {len(X)} samples")
            return

        # Train regression model
        X = np.array(X)
        y = np.array(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)

        self.regression_model = model
        logger.info(f"Retrained reward model with {len(X)} samples")
        logger.debug(f"Model coefficients: {model.coef_}")

    def _load_rule_weights(self) -> Dict[str, float]:
        """Load rule weights from config or secure memory"""
        # Check if weights exist in memory
        weight_entries = self.memory.recall(tag="rule_weights", top_k=1)
        if weight_entries:
            return weight_entries[0]['data']  # Return the weights value

        # Fallback to config defaults
        default_weights = self.config.get("default_weights", {
            "alignment": 0.4,
            "helpfulness": 0.3,
            "privacy": 0.15,
            "safety": 0.1,
            "truthfulness": 0.05
        })

        # Store default weights in memory
        self.memory.add(
            default_weights,
            tags=["reward_model", "rule_weights"],
            sensitivity=0.5
        )

        return default_weights

    def evaluate(self, text: str, context: Dict = None) -> Dict[str, float]:
        """Evaluate text against all security reward components"""
        scores = {}

        # Calculate rule-based scores
        for name, rule in self.rule_based.items():
            scores[name] = rule(text)

        # Calculate learned score
        scores["learned"] = self.learned_model(scores)

        # Calculate composite score
        composite = sum(scores[name] * self.rule_weights.get(name, 0) 
                     for name in self.rule_based)
        composite += scores["learned"] * self.rule_weights.get("learned", 0)
        scores["composite"] = composite

        # Store evaluation in secure memory
        if self.config.get("store_evaluations", True):
            evaluation_record = {
                "text": text,
                "scores": scores,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
            self.memory.add(
                evaluation_record,
                tags=["evaluation", "reward_model"],
                sensitivity=0.6
            )

        # Add attention-based scoring if available
        if context and "attention_analysis" in context:
            attn_report = context["attention_analysis"]
            scores["attention_quality"] = 1.0 - min(attn_report.get("anomaly_score", 0), 1.0)
            scores["attention_stability"] = 1.0 - min(attn_report.get("uniformity", 0) / 2.0, 1.0)

            # Add to composite score
            attention_factor = 0.1  # 10% weight to attention quality
            composite += scores["attention_quality"] * attention_factor

        return scores

    def update_rule_weights(self, new_weights: Dict[str, float]):
        """Update rule weights and store in secure memory"""
        # Validate and normalize weights
        total = sum(new_weights.values())
        if total == 0:
            logger.warning("Cannot update weights: sum of weights is zero")
            return
            
        normalized = {k: v/total for k, v in new_weights.items()}
        
        # Update in-memory weights
        self.rule_weights = normalized
        
        # Store updated weights in secure memory
        self.memory.add(
            normalized,
            tags=["reward_model", "rule_weights"],
            sensitivity=0.7
        )
        logger.info(f"Updated rule weights: {normalized}")

    # Rule-based scoring functions =============================================
    
    def _alignment_score(self, text: str) -> float:
        """Measure alignment with ethical guidelines (0-1 scale)"""
        harmful_terms = ["harm", "hate", "discriminate", "illegal"]
        harmful_count = sum(text.lower().count(term) for term in harmful_terms)
        return 1.0 - min(harmful_count / max(1, len(text.split()) * 5), 1.0)

    def _helpfulness_score(self, text: str) -> float:
        """Measure helpfulness and constructive content (0-1 scale)"""
        helpful_terms = ["help", "assist", "support", "guide", "answer"]
        helpful_count = sum(text.lower().count(term) for term in helpful_terms)
        return min(helpful_count / max(1, len(text.split()) * 3), 1.0)

    def _privacy_score(self, text: str) -> float:
        """Detect privacy risks and PII exposure (0-1 scale)"""
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4} \d{4} \d{4} \d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
        pii_count = sum(len(re.findall(pattern, text)) for pattern in pii_patterns)
        return 1.0 - min(pii_count * 0.5, 1.0)  # 0.5 penalty per PII

    def _safety_score(self, text: str) -> float:
        """Detect safety risks and dangerous content (0-1 scale)"""
        risk_terms = ["dangerous", "unsafe", "risk", "hazard", "toxic"]
        risk_count = sum(text.lower().count(term) for term in risk_terms)
        return 1.0 - min(risk_count / max(1, len(text.split())) * 2, 1.0)

    def _truthfulness_score(self, text: str) -> float:
        """Measure factual accuracy and truthfulness (0-1 scale)"""
        # Placeholder - would integrate with fact-checking modules
        return 0.8  # Default neutral score

    # Analytics and reporting ==================================================
    
    def get_evaluation_history(self, time_range: str = "7d") -> List[Dict]:
        """Retrieve evaluation history from secure memory"""
        # Convert time range to filter
        time_map = {"1d": 1, "7d": 7, "30d": 30}
        days = time_map.get(time_range, 7)
        cutoff = datetime.now() - timedelta(days=days)
        
        # Create timestamp filter
        filters = {"timestamp": {"$gt": cutoff.isoformat()}}  # Simplified for concept
        
        entries = self.memory.recall(tag="evaluation", top_k=1000)
        cutoff = datetime.now() - timedelta(days=days)
        
        # Filter and sort manually
        filtered = [
            e for e in entries
            if datetime.fromisoformat(e['meta']['timestamp']) > cutoff
        ]
        filtered.sort(key=lambda e: e['meta']['timestamp'], reverse=False)
        return filtered

    def generate_report(self, metrics: Dict[str, float]) -> Dict:
        """Generate security reward metrics report"""
        evaluations = self.get_evaluation_history("30d")
        report = []

        visualizer = get_visualizer()
        if not evaluations:
            return {}
            
        # Calculate average scores
        avg_scores = {}
        for key in self.rule_based.keys():
            avg_scores[key] = np.mean([e['data']['scores'][key] for e in evaluations])

        # Header Section
        report.append(f"\n# Security Reward Model Report\n")
        report.append(f"**Generated**: {datetime.now().isoformat()}\n")
        chart = visualizer.render_temporal_chart(QSize(600, 400), 'success_rate')
        report.append(f"![Reward Chart](data:image/png;base64,{visualizer._chart_to_base64(chart)})")
        if 'composite_score' in metrics:
            report.append(f"- **Composite Score**: {metrics['composite_score']:.3f}")
            report.append("\n### Metric Weight Impact")
            for metric, weighted in metrics.get('weighted_breakdown', {}).items():
                report.append(f"  - {metric}: {weighted:.3f}")

        report.append(f"\n---\n*Report generated by {self.__class__.__name__}*")   # Footer with system info

        return {
            "evaluation_count": len(evaluations),
            "average_scores": avg_scores,
            "weighted_composite": np.mean([e['data']['scores']['composite'] for e in evaluations]),
            "rule_weights": self.rule_weights
        }

if __name__ == "__main__":
    print("\n=== Running Security Reward Model ===\n")
    app = QApplication(sys.argv)
    config = load_config()

    reward = RewardModel(config=config)

    logger.info(f"{reward}")
    print(f"\n* * * * * Phase 2 * * * * *\n")    
    # Example evaluation
    sample_text = "I'm here to help you safely navigate this process without any harm"
    scores = reward.evaluate(sample_text)
    print(f"Security Scores:\n" + json.dumps(scores, indent=2))


    # Update weights example
    new_weights = {
        "alignment": 0.5,
        "helpfulness": 0.3,
        "privacy": 0.1,
        "safety": 0.1,
        "truthfulness": 0.0
    }
    reward.update_rule_weights(new_weights)
    
    # Generate report
    report = reward.generate_report(scores)
    print(json.dumps(report, indent=2))
    print("\n=== Successfully Ran Security Reward Model ===\n")
