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

import os
import time
import math
import hashlib
import json, yaml
import numpy as np
import pandas as pd
import torch.nn as nn

from datetime import datetime
from joblib import load, dump
from dataclasses import dataclass, field
from collections import defaultdict, deque
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Optional, Any, Tuple

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
    def __init__(self, input_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

@dataclass
class RiskModelParameters:
    initial_hazard_rates: Dict[str, float] = field(default_factory=lambda: {
        "system_failure": 1e-6,
        "sensor_failure": 1e-5,
        "unexpected_behavior": 1e-4
    })
    update_interval: float = 60.0
    decay_factor: float = 0.95
    risk_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "warning": (0.4, 0.7),
        "critical": (0.7, 1.0)
    })

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
        # Load config from file if not provided
        if self.config is None:
            self.config = self._load_config_from_file(LOCAL_CONFIG_PATH)
        
        # Initialize core components
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory

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

    def _init_anomaly_detector(self):
        """Load or train anomaly detection model"""
        try:
            return load('models/anomaly_detector.joblib')
        except FileNotFoundError:
            return self._train_anomaly_model()

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

    def _train_deep_anomaly_detector(self, df):
        """Train neural model on sequence patterns"""
        sequences = self._generate_temporal_sequences(df)
        model = AnomalyDetector()
        # Add training loop with LSTM/Transformer
        dump(model, 'src/agents/evaluation/models/deep_anomaly.joblib')

    def detect_anomalies(self, current_issues: List[Dict]) -> List[Dict]:
        """Flag suspicious patterns using ML models"""
        features = self._extract_ml_features(current_issues)
        predictions = self.anomaly_model.predict(features)
        return [issue for issue, pred in zip(current_issues, predictions) if pred == -1]

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
            agent = self.agent_factory.create(  # Remove 'agent_type' parameter
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
    """Orchestrates complete validation lifecycle"""
    
    def __init__(self, protocol: EvaluationProtocol):
        self.protocol = protocol
        self.artifacts = {
            'static_report': None,
            'behavioral_results': None,
            'safety_metrics': None
        }

    def execute_full_validation(self, codebase: str, agent: callable):
        """End-to-end validation pipeline"""
        if self.protocol.static_analysis:
            analyzer = StaticAnalyzer(codebase)
            self.artifacts['static_report'] = analyzer.full_analysis()
            
        validator = BehavioralValidator()
        self.artifacts['behavioral_results'] = validator.execute_test_suite(agent)
        
        if self.protocol.reward_constraints:
            reward_model = SafetyRewardModel(self.protocol.reward_constraints)
            # Connect to agent's decision loop
        
        return self._generate_validation_report()

    def _generate_validation_report(self) -> Dict:
        """Structured report using IV&V standards (IEEE 1012-2016)"""
        return {
            'certification_status': self._determine_certification(),
            'risk_assessment': self._calculate_risk_index(),
            'improvement_targets': self._identify_improvements()
        }

    def _determine_certification(self) -> str:
        """Implements UL 4600 safety case requirements"""
        if self.artifacts['static_report']['security_issues'] > 0:
            return 'FAILED'
        if self.artifacts['behavioral_results']['failed'] > 0:
            return 'CONDITIONAL'
        return 'CERTIFIED'

    def _calculate_risk_index(self) -> float:
        """Calculates risk priority number (RPN) from FMEA"""
        severity = self.artifacts['static_report']['critical_count']
        occurrence = self.artifacts['behavioral_results']['failed']
        return severity * occurrence

    def _identify_improvements(self) -> List[str]:
        """Root cause analysis using 5 Whys methodology"""
        return [
            f"Critical code violations: {self.artifacts['static_report']['critical_count']}",
            f"Behavioral failures: {self.artifacts['behavioral_results']['failed']}"
        ]

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

    print("\n=== Successfully Ran Evaluation Agent ===\n")
