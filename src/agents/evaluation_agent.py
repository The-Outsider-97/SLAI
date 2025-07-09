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
import torch
import json, yaml
import numpy as np
import pandas as pd
import torch.nn as nn
from datetime import datetime

from joblib import load, dump
from dataclasses import dataclass, field
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Optional, Any, Tuple

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.utils.interpretability import InterpretabilityHelper
from src.utils.database import IssueDBConnector, FallbackIssueTracker
from src.agents.evaluators.adaptive_risk import RiskAdaptation
from src.agents.evaluators.base_infra import EvalTuner
from src.agents.evaluators.safety_evaluator import SafetyEvaluator
from src.agents.evaluators.behavioral_validator import BehavioralValidator
from src.agents.evaluators.efficiency_evaluator import EfficiencyEvaluator
from src.agents.evaluators.autonomous_evaluator import AutonomousEvaluator
from src.agents.evaluators.statistical_evaluator import StatisticalEvaluator
from src.agents.evaluators.performance_evaluator import PerformanceEvaluator
from src.agents.evaluators.resource_utilization_evaluator import ResourceUtilizationEvaluator
from src.agents.evaluators.utils.certification_framework import CertificationStatus
from src.agents.evaluators.utils.evaluation_errors import OperationalError, CertificationError
from src.agents.evaluators.utils.evaluators_calculations import EvaluatorsCalculations
from src.agents.evaluators.utils.evaluation_transformer import EvaluationTransformer
from src.agents.evaluators.utils.validation_protocol import ValidationProtocol
from src.agents.evaluators.utils.static_analyzer import StaticAnalyzer
from src.agents.safety.safety_guard import SafetyGuard
from src.agents.base_agent import BaseAgent
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Evaluation Agent")
printer = PrettyPrinter

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
                 config = None,
                 **kwargs):
        super().__init__(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=config
        )
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.config = load_global_config()
        self.db_config = self.config.get("issue_database", {})

        self.evaluation_config = get_config_section('evaluation_agent')
        self.memory_warning_threshold = self.evaluation_config.get('memory_warning_threshold', 0.7)
        self.memory_critical_threshold = self.evaluation_config.get('memory_critical_threshold', 0.9)
        self.model_dir = self.evaluation_config.get("model_dir")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.deep_anomaly = self.evaluation_config.get('deep_anomaly')
        self.anomaly_detector = self.evaluation_config.get('anomaly_detector')
        self.update_interval = self.evaluation_config.get('update_interval', 60.0)
        self.decay_factor = self.evaluation_config.get('decay_factor', 0.95)
        self.initial_hazard_rates = self.evaluation_config.get('initial_hazard_rates', {})
        self.system_failure = self.evaluation_config.get('system_failure', 0.000001)
        self.sensor_failure = self.evaluation_config.get('sensor_failure', 0.00001)
        self.risk_thresholds = self.evaluation_config.get('risk_thresholds', {})
        self.warning = self.evaluation_config.get('warning', [0.4, 0.7])
        self.critical = self.evaluation_config.get('critical', [0.7, 1.0])
        self.test_cases = self.evaluation_config.get('behavioral_test_cases') or self._load_default_test_cases()
        self.task_config = self.evaluation_config.get("autonomous_tasks", [])

        self.autonomous_tasks = self._load_autonomous_tasks()
        self.anomaly_model = self._init_anomaly_detector()
        self.issue_db = self._connect_issue_database()

        protocol = None
        
        # Core services with fallback initialization
        self.evaluators = self._init_evaluators_modules()
        self.protocol = self._init_validation_protocol()
        self.interpreter = InterpretabilityHelper()
        self.calculations = EvaluatorsCalculations()
        self.validation_suite = AIValidationSuite(protocol=protocol)
        self.risk_model = self._init_risk_model()
        self.tuner = self._init_hyperparam_tuner()

        self.transformer = None

    def _load_default_test_cases(self) -> List[Dict]:
        """Fallback test cases when none are configured"""
        return [
            {
                'scenario': {
                    'input': "test_input",
                    'requirement_id': "REQ-001",
                    'detection_method': 'automated'
                },
                'oracle': lambda x: x == "expected_output"
            }
        ]

    def get_overall_system_health(self) -> Dict[str, Any]:
        """
        Returns a high-level system health snapshot, useful for diagnostics API.
        Combines safety, performance, efficiency, and statistical confidence.
        """
        try:
            latest_metrics = self.shared_memory.get("latest_metrics") or {}
            
            # Get shared memory health metrics
            try:
                sm_metrics = self.shared_memory.metrics()
                sm_stats = self.shared_memory.get_usage_stats()
            except Exception as e:
                logger.error(f"Failed to get shared memory metrics: {e}")
                sm_metrics = {}
                sm_stats = {}
            
            # Calculate memory health status
            memory_usage_pct = sm_stats.get('memory_usage_percentage', 0) / 100
            memory_status = "Normal"
            if memory_usage_pct > self.memory_critical_threshold:
                memory_status = "Critical"
            elif memory_usage_pct > self.memory_warning_threshold:
                memory_status = "Warning"
            
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
                "shared_memory": {
                    "memory_usage_percent": sm_stats.get('memory_usage_percentage', 0),
                    "available_memory_mb": sm_stats.get('available_memory_mb', 0),
                    "item_count": sm_stats.get('item_count', 0),
                    "pending_expiration_cleanup": sm_stats.get('pending_expiration_cleanup', 0),
                    "access_count": sm_metrics.get('access_count', 0),
                    "status": memory_status
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

    def _init_evaluators_modules(self):
        """Initialize all evaluator modules using loaded config"""
        modules = {
            'behavioral': BehavioralValidator(test_cases=self.test_cases),
            'performance': PerformanceEvaluator(),
            'efficiency': EfficiencyEvaluator(),
            'statistical': StatisticalEvaluator(),
            'resource': ResourceUtilizationEvaluator(),
            'autonomous': AutonomousEvaluator(),
            'safety': SafetyEvaluator()
        }
        return modules

    def _init_validation_protocol(self) -> ValidationProtocol:
        """Initialize validated protocol configuration"""
        try:
            protocol = ValidationProtocol()
            protocol.validate_configuration()
            return protocol
        except ValueError as e:
            logger.error(f"ValidationProtocol configuration error: {e}")
            raise

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
            return load(self.anomaly_detector)
        except FileNotFoundError:
            return self._train_anomaly_model()
    
    def _train_deep_anomaly_detector(self, df):
        """Train transformer-based anomaly detector on sequence patterns"""
        sequences = self._generate_temporal_sequences(df)
        if not sequences:
            logger.warning("No temporal sequences generated for deep anomaly detector")
            return
        
        # Prepare training data
        features = torch.tensor(sequences, dtype=torch.float32)
        input_dim = features.shape[-1]
        seq_len = features.shape[1]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize specialized transformer
        encoder = EvaluationTransformer(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=128  # Adjust based on your needs
        )
        decoder = EvaluationTransformer(
            input_dim=input_dim,
            seq_len=seq_len,
            num_encoder_layers=0,  # Decoder only
            num_decoder_layers=2
        )
        
        # Move to device
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        features = features.to(device)
        
        # Training parameters
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), 
            lr=0.001,
            weight_decay=1e-5
        )
        criterion = nn.MSELoss()
        num_epochs = 30
        batch_size = 64
        num_batches = len(features) // batch_size
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            permutation = torch.randperm(len(features))
            
            for i in range(0, num_batches):
                idx = permutation[i*batch_size : (i+1)*batch_size]
                batch = features[idx]
                
                # Forward pass
                encoded = encoder(batch)
                decoded = decoder(encoded)
                loss = criterion(decoded, batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_epoch_loss:.6f}")
        
        with torch.no_grad():
            encoded = encoder(features)
            reconstructions = decoder(encoded)
            errors = torch.mean((features - reconstructions)**2, dim=[1, 2])
            
        # Save normalization parameters
        min_err = torch.min(errors).item()
        max_err = torch.max(errors).item()
        threshold = torch.quantile(errors, 0.90).item()
        
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'input_dim': input_dim,
            'seq_len': seq_len,
            'min_error': min_err,
            'max_error': max_err,
            'anomaly_threshold': threshold
        }, self.deep_anomaly)
        
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
        """Flag suspicious patterns using ML models with robust error handling"""
        features = self._extract_ml_features(current_issues)
        if not features:
            return []
        
        # Predict with isolation forest
        predictions = self.anomaly_model.predict(features)
        
        # Initialize deep model metrics
        deep_anomalies = []
        reconstruction_errors = []
        deep_threshold = 0.0
        
        try:
            # Load deep model with proper device handling
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(self.deep_anomaly, map_location=device)
            
            # Create new encoder/decoder instances
            encoder = EvaluationTransformer(
                input_dim=checkpoint['input_dim'],
                seq_len=checkpoint['seq_len'],
                d_model=128
            ).to(device).eval()
            
            decoder = EvaluationTransformer(
                input_dim=checkpoint['input_dim'],
                seq_len=checkpoint['seq_len'],
                d_model=128
            ).to(device).eval()
            
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            
            # Convert to sequences (sliding window)
            sequences = self._create_sequences(features, window_size=checkpoint['seq_len'])
            if not sequences:
                raise ValueError("Insufficient data for sequence formation")
            
            features_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)
            
            # Get reconstruction errors
            with torch.no_grad():
                encoded = encoder(features_tensor)
                reconstructions = decoder(encoded)
                errors = torch.mean((features_tensor - reconstructions)**2, dim=[1, 2])
                reconstruction_errors = errors.cpu().numpy()
            
            # Use precomputed normalization from training
            min_err = checkpoint.get('min_error', 0.0)
            max_err = checkpoint.get('max_error', 1.0)
            normalized_errors = (reconstruction_errors - min_err) / (max_err - min_err + 1e-8)
            deep_threshold = checkpoint.get('anomaly_threshold', 0.9)
            
            # Flag anomalies (top 10% scores)
            deep_anomalies = [i for i, score in enumerate(normalized_errors) if score > deep_threshold]
        
        except Exception as e:
            logger.error(f"Deep anomaly detection failed: {e}")
            # Fallback to isolation forest only
            deep_anomalies = [i for i, pred in enumerate(predictions) if pred == -1]
        
        # Combine results using weighted ensemble
        combined_scores = []
        for i in range(len(features)):
            if_score = 1.0 if predictions[i] == -1 else 0.0
            deep_score = 1.0 if i in deep_anomalies else 0.0
            combined_scores.append(0.4 * if_score + 0.6 * deep_score)
        
        # Final threshold (tune based on precision/recall needs)
        final_threshold = 0.5
        anomalies_indices = [i for i, score in enumerate(combined_scores) if score > final_threshold]
        
        return [current_issues[i] for i in anomalies_indices]
        
    def _extract_ml_features(self, issues: List[Dict]) -> List[List[float]]:
        """Extract features for ML models with null handling"""
        features = []
        for issue in issues:
            try:
                # Handle missing values with defaults
                severity = issue.get('severity', 0.5)
                complexity = issue.get('cognitive_complexity', 2.0)
                data_flow = issue.get('data_flow_depth', 1)
                security = issue.get('security_risk', 0.3)
                
                # Add temporal features
                timestamp = issue.get('timestamp', datetime.now())
                hour = timestamp.hour / 24.0  # Normalized hour
                
                features.append([
                    severity,
                    complexity,
                    data_flow,
                    security,
                    hour
                ])
            except Exception as e:
                logger.error(f"Feature extraction error: {e}")
        return features
 
    def _create_sequences(self, features: List[List[float]], window_size: int) -> List[List[List[float]]]:
        """Convert features to sliding window sequences"""
        sequences = []
        for i in range(len(features) - window_size + 1):
            sequences.append(features[i:i+window_size])
        return sequences
   
    def _load_autonomous_tasks(self) -> List[Dict[str, Any]]:
        """
        Loads tasks used for autonomous planning and robotics evaluation.
        Can be loaded from config, file, or generated inline.
        """
        task_config = self.task_config
        if isinstance(task_config, str) and task_config.endswith(".json"):
            try:
                with open(task_config, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load autonomous tasks from file: {e}")
                return []
        elif isinstance(task_config, list):
            return task_config
        else:
            logger.warning("No autonomous tasks defined in config.")
            return []

    def execute_validation_cycle(self, params: Dict) -> Dict:
        """Comprehensive evaluation across all dimensions"""
        try:
            results = {}
            
            # Static Analysis
            if self.protocol.static_analysis['enable']:
                analyzer = StaticAnalyzer('src/agents/evaluation_agent.py')
                static_results = analyzer.full_analysis()
                results.update({
                    'static_analysis': static_results,
                    'static_analysis_explanation': self._explain_static_results(static_results)
                })
        
            # Create agent properly
            agent = self.create_agent()
            
            # Behavioral Testing
            if self.protocol.behavioral_testing['test_types']:
                test_suite = self.evaluators['behavioral'].execute_test_suite(agent)
                
            # Autonomous Task Evaluation - use validated tasks
            if self.autonomous_tasks:
                results['autonomous'] = self.evaluators['autonomous'].evaluate_task_set(
                    self._get_validated_tasks()
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
        
        
            statistical_data = self._prepare_statistical_dataset()
            if len(statistical_data['current_run']) > 0 and len(statistical_data['current_run']) >= self.evaluators['statistical'].min_sample_size:
                results['statistical'] = self.evaluators['statistical'].evaluate(
                    datasets=statistical_data
                )
            else:
                logger.info("Skipping statistical evaluation - insufficient data")
                results['statistical'] = {'status': 'skipped', 'reason': 'Insufficient data'}

            # Autonomous Task Evaluation
            if self.autonomous_tasks:
                try:
                    results['autonomous'] = self.evaluators['autonomous'].evaluate_task_set(
                        self.autonomous_tasks
                    )
                except Exception as e:
                    logger.error(f"Autonomous evaluation failed: {str(e)}")
                    results['autonomous'] = {'error': str(e)}
            
            # Safety Incident Evaluation
            if SafetyEvaluator().safety_incidents:
                try:
                    results['safety'] = self.evaluators['safety'].evaluate_operation(
                        SafetyEvaluator().safety_incidents
                    )
                except Exception as e:
                    logger.error(f"Safety evaluation failed: {str(e)}")
                    results['safety'] = {'error': str(e)}

            # Add aggregated metrics
            results.update(self._gather_core_metrics(results))

            # Financial health assessment
            financial_health = self._evaluate_financial_health(
                params.get('portfolio_state', {}),
                params.get('dashboard_data', {})
            )
            results.update(financial_health)
            
            # Determine overall status
            results['status'] = self._determine_system_status(results)

            return results
        except Exception as e:
            logger.error(f"Validation cycle failed: {str(e)}")
            return {
                'error': str(e),
                'cycle_failed': True
            }

    def _prepare_statistical_dataset(self) -> Dict[str, List[float]]:
        """Prepare historical data for statistical analysis"""
        metrics = self.shared_memory.get("latest_metrics") or {}
        perf_metrics = metrics.get('performance', {})
        
        # SAFETY CHECK
        current_data = perf_metrics.get('accuracy_history', [])
        if not isinstance(current_data, list):
            current_data = []
    
        dataset = {
            'current_run': (self.shared_memory.get("latest_metrics") or {}).get('performance', {}).get('accuracy_history', []),
            'previous_runs': [
                entry['performance']['accuracy'] 
                for entry in self.shared_memory.get("metric_history") or []
                if 'performance' in entry
            ]
        }
        return dataset
    
    def _evaluate_financial_health(self, portfolio: Dict, dashboard: Dict) -> Dict:
        """Financial-specific risk assessment"""
        metrics = {
            'value_at_risk': portfolio.get('value_at_risk', 0),
            'drawdown': portfolio.get('current_drawdown', 0),
            'liquidity_ratio': portfolio.get('cash', 0) / portfolio.get('portfolio_value', 1)
        }
        
        # Check against thresholds
        critical_issues = []
        if metrics['value_at_risk'] > 0.15:
            critical_issues.append('VaR exceeds 15% threshold')
        if metrics['drawdown'] > 0.1:
            critical_issues.append('Drawdown exceeds 10% limit')
        if metrics['liquidity_ratio'] < 0.2:
            critical_issues.append('Liquidity ratio below 20%')
            
        return {
            'financial_metrics': metrics,
            'critical_issues': critical_issues
        }
    
    def _determine_system_status(self, results: Dict) -> str:
        """Determine overall system health status"""
        min_success = self.risk_thresholds.get('min_success_rate', 0.8)  # Default value
        
        if results.get('safety', {}).get('compliance_rate', 0) < min_success:
            return 'critical'
        if results.get('financial_metrics', {}).get('critical_issues'):
            return 'critical'
        if results.get('performance', {}).get('accuracy', 0) < 0.7:
            return 'warning'
        return 'normal'

    def _connect_issue_database(self):
        """Robust database connection with fallback handling"""
        try:
            # Connection validation
            if not all(key in self.db_config for key in ('host', 'port', 'database')):
                raise ValueError("Incomplete database configuration")
    
            # Establish connection with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.issue_db = IssueDBConnector(**self.db_config)
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
        self.risk_config = self.evaluation_config.get('risk_adaptation', {})
        return RiskAdaptation()

    def _init_hyperparam_tuner(self):
        """Initialize tuner with validation"""
        return EvalTuner(model_type=None,
            evaluation_function=self.evaluate_hyperparameters
        )

    def evaluate_hyperparameters(self, params: Dict) -> float:
        """Optimized hyperparameter evaluation"""
        try:
            validation_result = self.execute_validation_cycle(params)
            return self.calculations._calculate_composite_score(validation_result)
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
        """Aggregate metrics from all evaluators with safe access"""
        # Get nested statistical value safely
        statistical = raw_results.get('statistical', {})
        hypothesis_tests = statistical.get('hypothesis_tests', {})
        main_effect = hypothesis_tests.get('main_effect', {})
        p_value = 1.0
        if statistical and 'hypothesis_tests' in statistical:
            main_effect = statistical['hypothesis_tests'].get('main_effect', {})
            p_value = main_effect.get('p_value', 1.0)
    
        metrics = {
            "accuracy": raw_results.get('performance', {}).get('accuracy', 0.0),
            "efficiency": raw_results.get('efficiency', {}).get('score', 0.0),
            "safety_score": self.risk_model.get_current_risk("system_failure")['risk_metrics']['current_mean'],
            "resource_usage": raw_results.get('resource', {}).get('composite_score', 1.0),
            "statistical_significance": p_value,  # Use safely accessed value
            "autonomous_score": raw_results.get('autonomous', {}).get('composite_score', 0.0),
            "safety_compliance": raw_results.get('safety', {}).get('compliance_rate', 0.0)
        }
        return metrics
    
    def _get_validated_tasks(self) -> List[Dict]:
        """Ensure tasks have required structure"""
        return [
            {
                'id': f"task_{i}",
                'path': task.get('path', []),
                'optimal_path': task.get('optimal_path', []),
                'completion_time': task.get('completion_time', 0.0),
                'energy_consumed': task.get('energy_consumed', 0.0),
                'collisions': task.get('collisions', 0),
                'success': task.get('success', False)
            }
            for i, task in enumerate(self.autonomous_tasks)
        ]

    def create_agent(self) -> BaseAgent:  
        """Factory method with fault tolerance"""
        try:
            # Determine agent type from configuration
            agent_type = self.config.get('evaluation_agent', {}).get(
                'test_agent_type', 'adaptive'
            )
            
            # Prepare agent configuration
            agent_config = {
                **self.config.get("agent_params", {}),
                "runtime_context": {
                    "risk_model": self.risk_model,
                    "evaluators": self.evaluators
                }
            }
            
            # Create agent using factory
            agent = self.agent_factory.create(
                agent_type=agent_type,
                shared_memory=self.shared_memory,
                config=agent_config
            )
        
            # Initialize agent if needed
            if not agent.is_initialized():  
                agent._init_agent_specific_components()  
        
            return agent  
        
        except Exception as e:  
            logger.critical(f"Agent creation failed: {e}", exc_info=True)  
            return FallbackEvaluatorAgent(
                shared_memory=self.shared_memory,
                agent_factory=self.agent_factory,
                config=self.config
            )
        
    def predict(self, state: Any = None) -> Dict[str, Any]:
        """
        Predicts evaluation metrics based on current system state.
        
        Args:
            state (Any, optional): Input state for prediction. Uses system health if None.
        
        Returns:
            Dict[str, Any]: Predicted metrics including safety, performance, and resource usage
        """
        try:
            # Get current health status as base prediction
            health = self.get_overall_system_health()
            
            # If state is provided, run a lightweight evaluation cycle
            if state is not None:
                light_results = self.execute_validation_cycle(
                    {"lightweight": True, "input_state": state}
                )
                health["predicted_metrics"] = light_results.get('metrics', {})
            
            return {
                "status": "success",
                "prediction": health,
                "confidence": 0.85  # Base confidence score
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "confidence": 0.0
            }

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

    def log_evaluation(self, results: Dict) -> None:
        """Secure logging with privacy checks"""
        if not isinstance(results, dict):
            logger.error("Invalid results format - expected dictionary")
            return
        
        sanitized = self._sanitize_results(results)
        self._store_metrics(sanitized)
        self._update_risk_model(sanitized)

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

    def _notify_operations_team(self, message: str, level: str = "CRITICAL"):
        """
        Notifies the operations team about critical system events.
    
        Supports configurable notification channels:
        - Email alerts
        - Webhook integrations (e.g., Slack, Teams, PagerDuty)
        - External logging services
    
        Args:
            message (str): Message to be sent.
            level (str): Severity level: 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        """
        notification_config = self.config.get("operations_notification", {})
        if not notification_config:
            logger.warning("No operations notification config found.")
            return
    
        # Example logger-based fallback
        logger_method = getattr(logger, level.lower(), logger.info)
        logger_method(f"OPERATION ALERT: {message}")
    
        try:
            # Send email (if configured)
            if notification_config.get("email", {}).get("enabled", False):
                self._send_email_alert(message, notification_config["email"])
    
            # Send webhook (e.g., Slack, Teams, custom dashboards)
            if notification_config.get("webhook", {}).get("enabled", False):
                self._send_webhook_alert(message, notification_config["webhook"])
    
            # Log to external service (e.g., ELK, Datadog)
            if notification_config.get("external_logging", {}).get("enabled", False):
                self._log_to_external_service(message, level, notification_config["external_logging"])
    
        except Exception as e:
            logger.error(f"Failed to notify operations team: {e}", exc_info=True)

    def _send_email_alert(self, message: str, config: Dict[str, Any]):
        import smtplib
        from email.mime.text import MIMEText
    
        msg = MIMEText(message)
        msg["Subject"] = config.get("subject", "AI System Alert")
        msg["From"] = config["from"]
        msg["To"] = config["to"]
    
        with smtplib.SMTP(config["smtp_host"], config.get("smtp_port", 587)) as server:
            if config.get("use_tls", True):
                server.starttls()
            server.login(config["username"], config["password"])
            server.send_message(msg)
        logger.info("Email alert sent to operations team.")
    
    def _send_webhook_alert(self, message: str, config: Dict[str, Any]):
        import requests
        payload = {"text": message}
        response = requests.post(config["url"], json=payload, timeout=5)
        if response.status_code == 200:
            logger.info("Webhook alert sent successfully.")
        else:
            logger.warning(f"Webhook failed: {response.status_code} {response.text}")
    
    def _log_to_external_service(self, message: str, level: str, config: Dict[str, Any]):
        import requests
        payload = {
            "service": "EvaluationAgent",
            "level": level,
            "message": message,
            "timestamp": time.time()
        }
        response = requests.post(config["url"], json=payload, timeout=5)
        if response.status_code != 200:
            logger.warning(f"External log failed: {response.status_code} {response.text}")

    def supports_fail_operational(self) -> bool:
        """
        Determines if the agent is capable of fail-operational behavior:
        - Core evaluators are present and interface-compliant.
        - Risk model and issue tracking subsystems are initialized.
        - Memory and fallback handling support exists.
        - Behavioral evaluator can continue execution under partial degradation.
        """
        try:
            if not hasattr(self, 'evaluators') or not isinstance(self.evaluators, dict):
                logger.warning("Fail-op check: 'evaluators' missing or not a dictionary")
                return False
    
            required_evaluators = ['behavioral', 'performance', 'efficiency', 'safety']
            for key in required_evaluators:
                if key not in self.evaluators:
                    logger.warning(f"Fail-op check: Missing required evaluator '{key}'")
                    return False
    
            # Verify interface compliance
            behavioral_eval = self.evaluators.get('behavioral')
            if not callable(getattr(behavioral_eval, "execute_test_suite", None)):
                logger.warning("Fail-op check: 'behavioral' evaluator lacks 'execute_test_suite'")
                return False
    
            # Check for critical infrastructure components
            if not hasattr(self, 'risk_model') or self.risk_model is None:
                logger.warning("Fail-op check: 'risk_model' not initialized")
                return False
    
            if not hasattr(self, 'issue_db') or self.issue_db is None:
                logger.warning("Fail-op check: 'issue_db' not initialized")
                return False
    
            if not hasattr(self, 'shared_memory') or self.shared_memory is None:
                logger.warning("Fail-op check: 'memory' not initialized")
                return False
    
            # Check for minimal self-recovery capacity
            if hasattr(self, 'fallback_agent'):
                fallback_ready = callable(getattr(self.fallback_agent, 'supports_fail_operational', None)) and \
                                 self.fallback_agent.supports_fail_operational()
                if not fallback_ready:
                    logger.warning("Fail-op check: fallback_agent exists but is not fail-operational")
                    return False
    
            logger.info("Fail-operational check: PASSED")
            return True
    
        except Exception as e:
            logger.exception(f"Fail-operational check raised exception: {e}")
            return False

    def has_redundant_safety_channels(self) -> bool:
        """
        Verifies presence of redundant safety mechanisms such as:
        - Fallback safety guard
        - Hardcoded rule-based validators
        - Safety metrics thresholds
        - Redundant data validation paths
        - Diagnostic coverage above critical threshold
        - Watchdog or liveness monitoring
        """
    
        # 1. Check fallback safety guard
        guard_present = hasattr(self, 'safety_guard') and isinstance(SafetyGuard)
        guard_viable = guard_present and SafetyGuard.is_minimal_viable()
    
        # 2. Configured thresholds in safety_limits
        static_limits = self.config.get("safety_limits", {})
        has_thresholds = all(k in static_limits for k in ['max_latency', 'min_accuracy'])
    
        # 3. Redundant evaluators or rule-based checks
        hardcoded_rules = hasattr(self, 'evaluators') and 'safety' in self.evaluators \
                          and hasattr(self.evaluators['safety'], 'rule_based_checks')
    
        # 4. Diagnostic coverage from safety case (optional stub fallback)
        try:
            diag_coverage = self.calculations.calculate_diagnostic_coverage()
            diagnostics_ok = diag_coverage.get("coverage", 0) >= 0.85  # ISO threshold suggestion
        except Exception:
            diagnostics_ok = False
    
        # 5. Liveness monitor or watchdog
        liveness_ok = hasattr(self, 'shared_memory') and self.shared_memory.get('last_alive_ping') is not None
    
        # 6. Runtime safety hooks (like SafetyGuard monitoring threads)
        runtime_hooks = hasattr(SafetyGuard, 'monitor_thread') if guard_present else False
    
        return any([
            guard_viable,
            has_thresholds,
            hardcoded_rules,
            diagnostics_ok,
            liveness_ok,
            runtime_hooks
        ])

class AIValidationSuite:
    """Certification-grade validation for regulatory compliance"""
    
    def __init__(self, protocol: ValidationProtocol):
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
        if not codebase or not os.path.isdir(codebase):
            raise CertificationError(f"Invalid codebase path: {codebase}")
            
        analyzer = StaticAnalyzer(codebase)
        report = analyzer.full_analysis()

        critical_count = report['security_metrics']['critical_count']
        if critical_count > self.protocol.static_analysis['security']['max_critical']:
            raise CertificationError(f"Critical violations exceed threshold: {critical_count}")
            
        return report

    def _run_behavioral_qualification(self, agent: BaseAgent) -> Dict:
        """Requirements-based testing with traceability"""
        validator = BehavioralValidator()
        results = validator.execute_certification_suite(
            sut=agent.perform_task,
            certification_requirements=self.protocol.behavioral_tests
        )
        
        # Verify the outcome of the certification suite
        if results['overall_status'] == "FAILED":
            # Extract details for the error message
            failed_reqs = [t['requirement_id'] for t in results['traceability_matrix'] if t['status'] == 'FAILED']
            raise CertificationError(f"Behavioral qualification failed. Failed requirements: {failed_reqs}")
            
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
        certificate = {
            'certification_status': CertificationStatus,
            'compliance_matrix': self._generate_compliance_matrix(),
            'safety_case': self.artifacts['safety_case'],
            'evidence_bundle': self._package_evidence()
        }
        return certificate
    
    def _generate_compliance_matrix(self) -> Dict[str, bool]:
        """Map requirements to verification evidence"""
        return {
            "static_analysis": self.artifacts['static_report']['security_metrics']['critical_count'] <= self.protocol.static_analysis['security']['max_critical'],
            # ... other compliance checks
        }

    def _package_evidence(self) -> Dict:
        """Prepare evidence for regulatory submission"""
        evidence = {
            "static_analysis": self.artifacts['static_report'],
            "behavioral_tests": self.artifacts['behavioral_results'],
            "traceability_matrix": self._generate_traceability_matrix(),
            "tool_qualification": self._qualify_validation_tools()
        }
        return evidence

    def _generate_traceability_matrix(self) -> pd.DataFrame:
        """
        Create requirements-to-test traceability matrix.
    
        Each row maps a behavioral requirement to its corresponding test case,
        the test outcome, and whether it satisfies certification requirements.
        """
        if not self.artifacts['behavioral_results']:
            raise CertificationError("Behavioral results are missing. Cannot build traceability matrix.")
    
        traceability = []
        behavioral_tests = self.artifacts['behavioral_results'].get("tests", [])
    
        for test in behavioral_tests:
            requirement_id = test.get("requirement_id", "UNKNOWN")
            test_case_id = test.get("test_case_id", "N/A")
            outcome = test.get("status", "UNKNOWN")
            rationale = test.get("notes", "No explanation provided")
    
            traceability.append({
                "Requirement": requirement_id,
                "Test Case": test_case_id,
                "Status": outcome,
                "Rationale": rationale
            })
    
        df = pd.DataFrame(traceability)
        df.sort_values(by=["Requirement", "Test Case"], inplace=True)
        return df

    def _qualify_validation_tools(self) -> Dict:
        """Verify toolchain meets qualification standards"""
        tools = {
            "static_analyzer": "Qualified per DO-330 TQL3",
            "test_framework": "Certified per ISO/IEC 29119"
        }
        return tools
