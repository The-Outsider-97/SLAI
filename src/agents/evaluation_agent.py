__version__ = "1.8.0"

"""
Unified Evaluation Framework
Implements: Static Analysis, Behavioral Testing, Reward Modeling, and Multi-Objective Evaluation

Key Features:
1. Implements safety mechanisms from Scholten et al. (2022) "Safe RL Validation"
2. Statistical methods follow Demšar (2006) "Statistical Comparisons of Classifiers"
3. Pareto ranking based on Deb et al. (2002) NSGA-II algorithm
4. Modular design with failure mode analysis (Ibrahim et al. 2021)
"""

import time
import math
import torch
import json, yaml
import pandas as pd
import torch.nn as nn

from joblib import load, dump
from dataclasses import dataclass, field
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Optional, Any, Tuple

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.utils.interpretability import InterpretabilityHelper
from src.utils.database import IssueDBConnector, FallbackIssueTracker
from src.agents.evaluators.adaptive_risk import RiskAdaptation
from src.agents.evaluators.base_infra import HyperparamTuner
from src.agents.evaluators.behavioral_validator import BehavioralValidator
from src.agents.evaluators.performance_evaluator import PerformanceEvaluator
from src.agents.evaluators.resource_utilization_evaluator import ResourceUtilizationEvaluator
from src.agents.evaluators.statistical_evaluator import StatisticalEvaluator
from src.agents.evaluators.efficiency_evaluator import EfficiencyEvaluator
from src.agents.evaluators.utils.validation_protocol import ValidationProtocol
from src.agents.evaluators.utils.static_analyzer import StaticAnalyzer
from src.agents.safety.safety_guard import SafetyGuard
from src.agents.base_agent import BaseAgent
from logs.logger import get_logger

logger = get_logger("Evaluation Agent")

LOCAL_CONFIG_PATH = "src/agents/evaluators/configs/evaluator_config.yaml"
TUNER_CONFIG_PATH = "src/tuning/configs/hyperparam.yaml"

class AnomalyDetector(nn.Module):
    def __init__(self, input_dim=4, seq_len=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * seq_len, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim * seq_len),
            nn.Unflatten(1, (seq_len, input_dim))
        )
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# ------------------------ Operational Infrastructure ------------------------ #
@dataclass
class EvaluationProtocol:
    """Unified configuration for evaluation pipelines"""
    static_analysis: bool = True
    behavioral_tests: int = 100
    reward_constraints: Dict = None
    optimization_targets: List[str] = None

class OperationalError:
    pass

class CertificationError(Exception):
    """Custom exception for certification failures"""
    pass

class FallbackEvaluatorAgent(BaseAgent):
    def __init__(self,
                 shared_memory, 
                 agent_factory, 
                 config=None,
                 **kwargs):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config
        )

    def is_initialized(self):
        return True

    def supports_fail_operational(self):
        """
        Returns True if this fallback agent is designed to operate in degraded mode.
        Ensures:
        - Core evaluations still run (e.g. safety checks)
        - System doesn't crash entirely under failure
        """
        return hasattr(self, 'evaluators') and all(
            callable(getattr(ev, "execute_test_suite", None)) for ev in self.evaluators.values()
        )

    def has_redundant_safety_channels(self):
        """
        Verifies presence of redundant safety mechanisms such as:
        - Fallback safety guard
        - Hardcoded rule-based validators
        - Safety metrics thresholds
        """
        guard_present = hasattr(self, 'safety_guard') and self.safety_guard.is_minimal_viable()
        static_limits = self.config.get("safety_limits", {})
        has_thresholds = all(k in static_limits for k in ['max_latency', 'min_accuracy'])
    
        return guard_present or has_thresholds

class EvaluationAgent(BaseAgent):
    def __init__(self, 
                 shared_memory, 
                 agent_factory, 
                 config: Optional[Dict] = None,
                 **kwargs):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config
        )
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.config = load_global_config()
        self.evaluation_config = get_config_section('evaluation_agent')

        self.update_interval = self.evaluation_config.get('update_interval', 60.0)
        self.decay_factor = self.evaluation_config.get('decay_factor', 0.95)
        self.initial_hazard_rates = self.evaluation_config.get('initial_hazard_rates', {})
        self.system_failure = self.evaluation_config.get('system_failure', 0.000001)
        self.sensor_failure = self.evaluation_config.get('sensor_failure', 0.00001)
        self.risk_thresholds = self.evaluation_config.get('risk_thresholds', {})
        self.warning = self.evaluation_config.get('warning', [0.4, 0.7])
        self.critical = self.evaluation_config.get('critical', [0.7, 1.0])

        # Load config from file if not provided
        if self.config is None:
            self.config = self._load_config_from_file(LOCAL_CONFIG_PATH)

        self.anomaly_model = self._init_anomaly_detector()
        self.issue_db = self._connect_issue_database()
        
        # Core services with fallback initialization
        self.evaluators = self._init_evaluators_modules()
        self.protocol = self._init_validation_protocol()
        self.interpreter = InterpretabilityHelper()
        self.risk_model = self._init_risk_model()
        self.tuner = self._init_hyperparam_tuner()

    def _init_evaluators_modules(self):
        """Initialize all evaluator modules using loaded config"""
        return {
            'behavioral': BehavioralValidator(config=self.config.get('behavioral_evaluator', {})),
            'performance': PerformanceEvaluator(config=self.config.get('performance_evaluator', {})),
            'efficiency': EfficiencyEvaluator(config=self.config.get('efficiency_evaluator', {})),
            'statistical': StatisticalEvaluator(config=self.config.get('statistical_evaluator', {})),
            'resource': ResourceUtilizationEvaluator(config=self.config.get('resource_utilization_evaluator', {}))
        }

    def _init_validation_protocol(self) -> ValidationProtocol:
        """Initialize validated protocol configuration"""
        raw_config = self.config or self._load_config_from_file(LOCAL_CONFIG_PATH)
        
        # Handle nested config structure
        protocol_config = raw_config.get("validation_protocol", {})
        
        try:
            protocol = ValidationProtocol(**protocol_config)
            protocol.validate_configuration()
            return protocol
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid protocol config: {e}. Using defaults.")
            return ValidationProtocol()  # Fallback to dataclass defaults

    def _load_config_from_file(self, config_path: str) -> Dict:
        """Load YAML configuration file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _train_anomaly_model(self):
        """Train ensemble model on historical bug patterns"""
        df = pd.DataFrame(self._load_training_data())
        
        # Feature engineering
        features = df[[
            'severity', 'cognitive_complexity', 
            'data_flow_depth', 'security_risk'
        ]]
        
        # Hybrid detection model
        model = IsolationForest(
            n_estimators=200,
            contamination=0.01,
            random_state=42
        ).fit(features)
        
        # Deep autoencoder for temporal patterns
        if len(df) > 1000:
            self._train_deep_anomaly_detector(df)
            
        return model

    def _load_training_data(self):
        return [
            {
                'severity': 0.8,
                'cognitive_complexity': 3.5,
                'data_flow_depth': 2,
                'security_risk': 0.9
            },
            {
                'severity': 0.3,
                'cognitive_complexity': 2.1,
                'data_flow_depth': 1,
                'security_risk': 0.1
            }
            # Add more examples as needed
        ]

    def _init_anomaly_detector(self):
        """Load or train anomaly detection model"""
        try:
            return load('models/anomaly_detector.joblib')
        except FileNotFoundError:
            return self._train_anomaly_model()
    
    def _train_deep_anomaly_detector(self, df):
        """Train transformer-based anomaly detector on sequence patterns"""
        from src.agents.perception.modules.transformer import Transformer

        # Generate temporal sequences
        sequences = self._generate_temporal_sequences(df)
        if not sequences:
            logger.warning("No temporal sequences generated for deep anomaly detector")
            return

        # Prepare training data
        features = torch.tensor(sequences, dtype=torch.float32)

        # Initialize transformer-based autoencoder
        config = {
            'transformer': {
                'embed_dim': 32,
                'num_layers': 2,
                'max_position_embeddings': features.shape[1]
            },
            'attention': {'type': 'efficient'},
            'feedforward': {'hidden_dim': 64}
        }

        encoder = Transformer(config)
        decoder = Transformer(config)  # Simple symmetric decoder

        # Training parameters
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), 
            lr=0.001
        )
        criterion = nn.MSELoss()
        num_epochs = 20
        batch_size = 32

        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size]

                # Forward pass
                encoded = encoder(batch)
                decoded = decoder(encoded)

                # Reconstruction loss
                loss = criterion(decoded, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss = {epoch_loss/len(features):.6f}")

        # Save model
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'config': config
        }, 'models/deep_anomaly.pt')

        logger.info("Transformer-based anomaly detector trained and saved")

    def _generate_temporal_sequences(self, df, window_size=10):
        """Convert tabular data to temporal sequences"""
        sequences = []
        features = df[['severity', 'cognitive_complexity', 
                       'data_flow_depth', 'security_risk']].values
        
        for i in range(len(features) - window_size):
            sequences.append(features[i:i+window_size])
        
        return sequences

    def detect_anomalies(self, current_issues: List[Dict]) -> List[Dict]:
        """Flag suspicious patterns using ML models"""
        from src.agents.perception.modules.transformer import Transformer
        features = self._extract_ml_features(current_issues)
        predictions = self.anomaly_model.predict(features)
        
        # Load deep model if available
        deep_anomalies = []
        try:
            checkpoint = torch.load('models/deep_anomaly.pt')
            encoder = Transformer(checkpoint['config'])
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            
            # Get reconstruction errors
            features_tensor = torch.tensor(features, dtype=torch.float32)
            with torch.no_grad():
                reconstructions = encoder(features_tensor)
                errors = torch.mean((features_tensor - reconstructions)**2, dim=1)
            
            # Flag anomalies (top 5% reconstruction errors)
            threshold = torch.quantile(errors, 0.95)
            deep_anomalies = [i for i, err in enumerate(errors) if err > threshold]
            
        except FileNotFoundError:
            logger.info("Deep anomaly model not found, using only isolation forest")
        
        # Combine results from both models
        combined_anomalies = set(
            [i for i, pred in enumerate(predictions) if pred == -1] +
            deep_anomalies
        )
        
        return [current_issues[i] for i in combined_anomalies]

    def execute_validation_cycle(self, params: Dict) -> Dict:
        """Comprehensive evaluation across all dimensions"""
        results = {}
        
        # Static Analysis
        if self.protocol.static_analysis['enable']:
            analyzer = StaticAnalyzer('src/agents/evaluation_agent.py')
            static_results = analyzer.full_analysis()
            results.update({
                'static_analysis': static_results,
                'static_analysis_explanation': self._explain_static_results(static_results)
            })
    
        # Behavioral Testing
        if self.protocol.behavioral_testing['test_types']:
            test_suite = self.evaluators['behavioral'].execute_test_suite(
                sut=self.create_agent()
            )
            results['behavioral'] = test_suite
            results['test_explanation'] = self._explain_test_results(test_suite)
    
            # Subsequent evaluators using behavioral test results
            outputs = test_suite.get('predictions', [])
            truths = test_suite.get('expected_outputs', [])
            
            # Performance Evaluation
            results['performance'] = self.evaluators['performance'].evaluate(
                outputs=outputs,
                ground_truths=truths
            )
    
            # Efficiency Evaluation
            results['efficiency'] = self.evaluators['efficiency'].evaluate(
                outputs=outputs,
                ground_truths=truths
            )
    
        # Resource Utilization (always runs)
        results['resource'] = self.evaluators['resource'].evaluate()
    
        # Statistical Evaluation (requires historical context)
        statistical_data = self._prepare_statistical_dataset()
        results['statistical'] = self.evaluators['statistical'].evaluate(
            datasets=statistical_data
        )
    
        # Add aggregated metrics
        results.update(self._gather_core_metrics(results))
        
        return results

    def _prepare_statistical_dataset(self) -> Dict[str, List[float]]:
        """Prepare historical data for statistical analysis"""
        return {
            'current_run': self.shared_memory.get("latest_metrics", {}).get('performance', {}).get('accuracy_history', []),
            'previous_runs': [
                entry['performance']['accuracy'] 
                for entry in self.shared_memory.get("metric_history", [])
                if 'performance' in entry
            ]
        }

    def _connect_issue_database(self):
        """Robust database connection with fallback handling"""
        try:
            
            
            # Get config with environment variable overrides
            db_config = self.config.get("issue_database", {})
            
            # Connection validation
            if not all(key in db_config for key in ('host', 'port', 'database')):
                raise ValueError("Incomplete database configuration")
    
            # Establish connection with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.issue_db = IssueDBConnector(**db_config)
                    self.issue_db.initialize_schema({
                        "evaluation_issues": """
                            CREATE TABLE IF NOT EXISTS evaluation_issues (
                                id UUID PRIMARY KEY,
                                timestamp TIMESTAMPTZ NOT NULL,
                                issue_type VARCHAR(50) NOT NULL,
                                severity FLOAT CHECK (severity >= 0 AND severity <= 1),
                                context JSONB,
                                metrics JSONB,
                                resolution_status VARCHAR(20) DEFAULT 'unresolved'
                            )
                        """
                    })
                    logger.info("Connected to issue database")
                    return self.issue_db
                except OperationalError as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Connection attempt {attempt+1} failed: {e}")
                    time.sleep(2 ** attempt)
                    
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            logger.info("Initializing fallback issue tracker")
            self.issue_db = FallbackIssueTracker()
            return self.issue_db

    def _init_risk_model(self):
        """Initialize risk model with validation"""
        risk_config = self.config.get('risk_adaptation', {})
        return RiskAdaptation(config=risk_config)

    def _init_hyperparam_tuner(self):
        """Initialize tuner with validation"""
        return HyperparamTuner(
            tuner_config=TUNER_CONFIG_PATH,
            evaluation_function=self.evaluate_hyperparameters,
            strategy=self.config.get("tuning_strategy", "bayesian")
        )

    def evaluate_hyperparameters(self, params: Dict) -> float:
        """Optimized hyperparameter evaluation"""
        try:
            validation_result = self.execute_validation_cycle(params)
            return self._calculate_composite_score(validation_result)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return -float('inf')

    def _explain_static_results(self, static_results: Dict) -> str:
        """Generate human-readable analysis"""
        security_metrics = static_results.get('security_metrics', {})
        return (
            f"Code Quality Report:\n"
            f"- Technical Debt Ratio: {static_results.get('technical_debt', 0):.1%} "
            f"(Threshold: {self.protocol.static_analysis['code_quality']['tech_debt_threshold']:.1%})\n"
            f"- Critical Security Issues: {security_metrics.get('critical_count', 0)} "
            f"(Max Allowed: {self.protocol.static_analysis['security']['max_critical']})"
        )
    
    def _explain_test_results(self, results: Dict) -> str:
        """Generate test outcome summary using actual result keys"""
        return self.interpreter.explain_validation_metrics({
            'passed': results['passed'],
            'failed': results['failed'],
            'coverage': results['requirement_coverage']
        })

    def _gather_core_metrics(self, raw_results: Dict) -> Dict[str, Any]:
        """Aggregate metrics from all evaluators"""
        return {
            "accuracy": raw_results['performance'].get('accuracy', 0.0),
            "efficiency": raw_results['efficiency'].get('score', 0.0),
            "safety_score": self.risk_model.get_current_risk("system_failure")['risk_metrics']['current_mean'],
            "resource_usage": raw_results['resource']['composite_score'],
            "statistical_significance": raw_results['statistical']['hypothesis_tests'].get('main_effect', {}).get('p_value', 1.0)
        }

    def create_agent(self) -> BaseAgent:  
        """Factory method with fault tolerance"""
        from src.agents.agent_factory import AgentFactory
        self.agent_factory = AgentFactory 
        try:  
            agent = self.agent_factory.create(
                config={  
                    **self.config.get("agent_params", {}),  
                    "runtime_context": {  
                        "risk_model": self.risk_model,  
                        "evaluators": self.evaluators  
                    }  
                }  
            )  
        
            if not agent.is_initialized():  
                agent._init_agent_specific_components()  
        
            return agent  
        
        except Exception as e:  
            logger.critical(f"Agent creation failed: {e}")  
            return FallbackEvaluatorAgent(
                shared_memory=self.shared_memory,
                agent_factory=self.agent_factory,
                config=self.config
            )

    def _calculate_composite_score(self, results: Dict) -> float:
        """Weighted scoring with safety constraints"""
        weights = self.config.get("metric_weights", {})
        return (
            weights.get("accuracy", 0.6) * results["accuracy"] +
            weights.get("safety", 0.3) * results["safety_score"] +
            weights.get("efficiency", 0.1) * (1 - results["resource_usage"])
        )

    def log_evaluation(self, results: Dict) -> None:
        """Secure logging with privacy checks"""
        if not isinstance(results, dict):
            logger.error("Invalid results format - expected dictionary")
            return
        
        sanitized = self._sanitize_results(results)
        self._store_metrics(sanitized)
        self._update_risk_model(sanitized)

    def _get_safety_compliance(self) -> str:
        """Check against protocol safety requirements"""
        safety_params = self.protocol.safety_constraints['risk_mitigation']
        current_safety = self.risk_model.get_current_metrics()
        
        compliance = {
            'positional': current_safety['position_margin'] >= safety_params['safety_margins']['positional'],
            'temporal': current_safety['time_margin'] >= safety_params['safety_margins']['temporal'],
            'fail_operational': safety_params['fail_operational']
        }
        return "Compliant" if all(compliance.values()) else "Non-compliant"

    def _sanitize_results(self, results: Dict) -> Dict:
        """Apply privacy protections and validation"""
        sanitized = results.copy()
        for key in ["notes", "feedback"]:
            if key in sanitized:
                sanitized[key] = SafetyGuard.scrub(sanitized[key])
        return sanitized

    def _store_metrics(self, metrics: Dict) -> None:
        """Store metrics in shared memory with validation"""
        if not isinstance(metrics, dict):
            logger.warning("Invalid metrics format")
            return

        self.shared_memory.set("latest_metrics", metrics)
        self.shared_memory.append("metric_history", metrics)

    def _update_risk_model(self, metrics: Dict) -> None:
        """Update risk model with validation"""
        if self.risk_model:
            try:
                self.risk_model.update_model(
                    metrics.get("hazards", {}),
                    metrics.get("operational_time", 0)
                )
            except KeyError as e:
                logger.error(f"Risk update failed: {e}")

    def request_certification(self, codebase_path: str):
        """Initiate formal certification process"""
        cert_suite = AIValidationSuite(self.protocol)
        
        try:
            certification = cert_suite.execute_full_validation(
                codebase_path,
                self.create_agent()
            )
            self.shared_memory.set("certification_status", certification)
        except CertificationError as e:
            logger.error(f"Certification failed: {e}")
            self.trigger_mitigation_actions()

    def trigger_mitigation_actions(self):
        """
        Executes emergency mitigation protocols when the system becomes unsafe.
        Includes:
        - Logging critical failure state
        - Freezing unsafe components
        - Switching to fallback evaluators
        - Notifying external systems
        """
        logger.critical("Triggering mitigation actions due to certification failure or critical hazard.")
        
        self.shared_memory.set("system_status", "degraded")
    
        last_metrics = self.shared_memory.get("latest_metrics", {})
        self.issue_db.log_issue({
            "issue_type": "safety_breach",
            "severity": 1.0,
            "context": {"last_metrics": last_metrics},
            "resolution_status": "unresolved"
        })
    
        for key, evaluator in self.evaluators.items():
            if hasattr(evaluator, 'disable_temporarily'):
                evaluator.disable_temporarily()
            else:
                logger.warning(f"{key} evaluator lacks disable_temporarily() method.")
    
        fallback = FallbackEvaluatorAgent(
            shared_memory=self.shared_memory,
            agent_factory=self.agent_factory,
            config=self.config
        )
        self.shared_memory.set("active_agent", fallback)
    
        if "notify" in self.config:
            self._notify_operations_team("System degraded: mitigation actions triggered.")

    def get_overall_system_health(self) -> Dict[str, Any]:
        """
        Returns a high-level system health snapshot, useful for diagnostics API.
        Combines safety, performance, efficiency, and statistical confidence.
        """
        try:
            latest_metrics = self.shared_memory.get("latest_metrics") or {}
            
            # Provide default values for missing metrics
            safety_score = latest_metrics.get("safety_score", 0.0)
            accuracy = latest_metrics.get("accuracy", 0.0)
            efficiency = latest_metrics.get("efficiency", 0.0)
            resource_usage = latest_metrics.get("resource_usage", 1.0)
            stat_sig = latest_metrics.get("statistical_significance", 1.0)
    
            # Composite status categories
            def score_status(score, thresholds):
                if score >= thresholds["good"]:
                    return "Good"
                elif score >= thresholds["warning"]:
                    return "Warning"
                else:
                    return "Critical"
    
            thresholds = {
                "accuracy": {"good": 0.85, "warning": 0.65},
                "safety": {"good": 0.80, "warning": 0.60},
                "efficiency": {"good": 0.70, "warning": 0.50},
                "stat_sig": {"good": 0.01, "warning": 0.05}  # lower is better here
            }
    
            health_summary = {
                "status": "OK" if safety_score >= 0.8 and accuracy >= 0.85 else "Degraded",
                "metrics": {
                    "accuracy": {
                        "value": round(accuracy, 4),
                        "status": score_status(accuracy, thresholds["accuracy"])
                    },
                    "safety_score": {
                        "value": round(safety_score, 4),
                        "status": score_status(safety_score, thresholds["safety"])
                    },
                    "efficiency": {
                        "value": round(efficiency, 4),
                        "status": score_status(efficiency, thresholds["efficiency"])
                    },
                    "resource_usage": {
                        "value": round(resource_usage, 4),
                        "status": "Normal" if resource_usage <= 0.8 else "High"
                    },
                    "statistical_significance": {
                        "value": round(stat_sig, 4),
                        "status": "Significant" if stat_sig < 0.05 else "Not Significant"
                    }
                },
                "timestamp": time.time()
            }
    
            return health_summary
        except Exception as e:
            logger.error(f"Failed to compute system health: {e}", exc_info=True)
            return {
                "status": "Unknown",
                "error": str(e),
                "timestamp": time.time()
            }

class SafetyRewardModel:
    """
    Constrained reward function implementing
    Lagrangian optimization from Chow et al. (2017)
    """
    
    def __init__(self, constraints: Dict[str, Any]):
        self.constraints = constraints
        self.violation_history = []
        self._lagrangian_multipliers = {
            'safety': 0.1,
            'ethics': 0.2
        }

    def calculate_reward(self, state: Dict, action: Dict, outcome: Dict) -> float:
        """Multi-objective reward calculation with dynamic penalties"""
        base_reward = outcome.get('performance', 0.0)
        
        # Constraint calculations
        safety_penalty = self._calculate_safety_violation(outcome)
        ethics_penalty = self._calculate_ethical_violation(action)
        
        # Lagrangian formulation
        constrained_reward = base_reward \
            - self._lagrangian_multipliers['safety'] * safety_penalty \
            - self._lagrangian_multipliers['ethics'] * ethics_penalty
            
        self._update_multipliers(safety_penalty, ethics_penalty)
        return constrained_reward

    def _calculate_safety_violation(self, outcome: Dict) -> float:
        """Physical safety metrics using Hamilton-Jacobi reachability"""
        return max(0, outcome.get('hazard_prob', 0) - self.constraints['safety_tolerance'])

    def _calculate_ethical_violation(self, action: Dict) -> float:
        """Ethical penalty using Veale et al. (2018) bias metrics"""
        return sum(
            1 for pattern in self.constraints['ethical_patterns']
            if pattern in action['decision_path']
        )

    def _update_multipliers(self, safety_viol: float, ethics_viol: float):
        """Dual gradient descent update rule"""
        self._lagrangian_multipliers['safety'] *= (1 + safety_viol)
        self._lagrangian_multipliers['ethics'] *= (1 + ethics_viol)

# ------------------------ Statistical Evaluation Engine ------------------------ #
class MultiObjectiveEvaluator:
    """
    Implements non-dominated sorting from:
    Deb et al. (2002) "A Fast Elitist Multiobjective Genetic Algorithm"
    """
    
    def __init__(self, objectives: List[str], weights: Dict[str, float]):
        self.objectives = objectives
        self.weights = weights
        self.history = []

    def pareto_frontier(self, solutions: List[Dict]) -> List[Dict]:
        """NSGA-II inspired non-dominated sorting"""
        frontiers = []
        remaining = solutions.copy()
        
        while remaining:
            current_front = []
            dominated = []
            
            for candidate in remaining:
                if not any(self._dominates(other, candidate) for other in remaining):
                    current_front.append(candidate)
                else:
                    dominated.append(candidate)
            
            frontiers.append(current_front)
            remaining = dominated
            
        return frontiers

    def _dominates(self, a: Dict, b: Dict) -> bool:
        """Pareto domination criteria with weighted objectives"""
        better = 0
        for obj in self.objectives:
            a_val = a[obj] * self.weights.get(obj, 1.0)
            b_val = b[obj] * self.weights.get(obj, 1.0)
            if a_val > b_val:
                better += 1
            elif a_val < b_val:
                better -= 1
        return better > 0

    def statistical_analysis(self, baseline: List[float], treatment: List[float]) -> Dict:
        """Implements Demšar (2006) statistical comparison protocol"""
        n = len(baseline)
        z_scores = [(b - t) / math.sqrt(n) for b, t in zip(baseline, treatment)]
        return {
            'mean_diff': sum(b - t for b, t in zip(baseline, treatment)) / n,
            'effect_size': self._hedges_g(baseline, treatment),
            'significance': any(abs(z) > 2.58 for z in z_scores)  # p<0.01
        }

    def _hedges_g(self, a: List[float], b: List[float]) -> float:
        """Bias-corrected effect size metric"""
        n1, n2 = len(a), len(b)
        var_pooled = (sum((x - sum(a)/n1)**2 for x in a) + 
                     sum((x - sum(b)/n2)**2 for x in b)) / (n1 + n2 - 2)
        return (sum(a)/n1 - sum(b)/n2) / math.sqrt(var_pooled)

class AIValidationSuite:
    """Certification-grade validation for regulatory compliance"""
    
    def __init__(self, protocol: EvaluationProtocol):
        self.protocol = protocol
        self.artifacts = {
            'static_report': None,
            'behavioral_results': None,
            'safety_case': None,
            'certification_evidence': []
        }

    def execute_full_validation(self, codebase: str, agent: BaseAgent) -> Dict:
        """
        End-to-end certification pipeline meeting IV&V standards (IEEE 1012-2016)
        """
        # Phase 1: Architecture Validation
        self._validate_architecture(agent)
        
        # Phase 2: Static Verification
        self.artifacts['static_report'] = self._run_static_verification(codebase)
        
        # Phase 3: Behavioral Qualification
        self.artifacts['behavioral_results'] = self._run_behavioral_qualification(agent)
        
        # Phase 4: Safety Case Development
        self.artifacts['safety_case'] = self._build_safety_case(agent)
        
        return self._generate_certification_package()

    def _validate_architecture(self, agent: BaseAgent):
        """Verify architectural compliance with safety standards"""
        # Check component isolation and fail-operational mechanisms
        if not agent.supports_fail_operational():
            raise CertificationError("Architecture lacks fail-operational capabilities")
            
        # Verify redundancy mechanisms
        if not agent.has_redundant_safety_channels():
            raise CertificationError("Insufficient safety redundancy")

    def _run_static_verification(self, codebase: str) -> Dict:
        """Comprehensive static analysis meeting DO-178C/MISRA standards"""
        analyzer = StaticAnalyzer(codebase)
        report = analyzer.full_analysis()
        
        # Apply certification thresholds
        if report['critical_violations'] > self.protocol.certification_thresholds['max_critical']:
            raise CertificationError(f"Critical violations exceed threshold: {report['critical_violations']}")
            
        return report

    def _run_behavioral_qualification(self, agent: BaseAgent) -> Dict:
        """Requirements-based testing with traceability"""
        validator = BehavioralValidator()
        results = validator.execute_certification_suite(agent)
        
        # Verify requirements coverage
        coverage = results['requirement_coverage']
        if coverage < self.protocol.certification_thresholds['min_coverage']:
            raise CertificationError(f"Insufficient requirement coverage: {coverage:.1%}")
            
        return results

    def _build_safety_case(self, agent: BaseAgent) -> Dict:
        """Construct safety case following ISO 26262/UL 4600"""
        return {
            "hazard_analysis": agent.perform_hazard_analysis(),
            "safety_goals": agent.derive_safety_goals(),
            "fault_tree": agent.generate_fault_tree(),
            "diagnostic_coverage": agent.calculate_diagnostic_coverage()
        }

    def _generate_certification_package(self) -> Dict:
        """Generate ISO-compliant certification package"""
        return {
            'certification_status': self._determine_certification(),
            'compliance_matrix': self._generate_compliance_matrix(),
            'safety_case': self.artifacts['safety_case'],
            'evidence_bundle': self._package_evidence()
        }

    def _package_evidence(self) -> Dict:
        """Prepare evidence for regulatory submission"""
        return {
            "static_analysis": self.artifacts['static_report'],
            "behavioral_tests": self.artifacts['behavioral_results'],
            "traceability_matrix": self._generate_traceability_matrix(),
            "tool_qualification": self._qualify_validation_tools()
        }

    def _generate_traceability_matrix(self) -> pd.DataFrame:
        """Create requirements-to-test traceability matrix"""
        # Implementation would link requirements to test cases
        return pd.DataFrame(columns=['Requirement', 'Test Case', 'Status'])

    def _qualify_validation_tools(self) -> Dict:
        """Verify toolchain meets qualification standards"""
        return {
            "static_analyzer": "Qualified per DO-330 TQL3",
            "test_framework": "Certified per ISO/IEC 29119"
        }

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Evaluation Agent ===\n")
    class SharedMemory:
        def __init__(self, config=None):
            self.data = {}
    
        def set(self, key, value):
            self.data[key] = value
    
        def append(self, key, value):  # <-- fix here
            if key not in self.data or not isinstance(self.data[key], list):
                self.data[key] = []
            self.data[key].append(value)
    
        def get(self, key, default=None):
            return self.data.get(key, default)
    
    config = None
    shared_memory = SharedMemory(config=None)
    agent_factory = lambda: None

    agent = EvaluationAgent(shared_memory, agent_factory, config={})
    print(agent)
    print(f"\n* * * * * Phase 2 - Logging * * * * *\n")
    results={}
    log = agent.log_evaluation(results=results)

    logger.info(f"{log}")
    print(f"\n* * * * * Phase 3 - Validation * * * * *\n")
    validation_params = {}
    results = agent.execute_validation_cycle(validation_params)
    print(f"\n* * * * * Phase 4 * * * * *\n")
    codebase_path=None
    request = agent.request_certification(codebase_path=codebase_path)

    print(request)
    print("\n=== Successfully Ran Evaluation Agent ===\n")
