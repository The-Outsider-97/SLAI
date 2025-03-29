"""
Enhanced Collaborative Agent with Safety Monitoring and Task Coordination

Features:
1. Comprehensive safety monitoring with Bayesian risk assessment
2. Multi-agent task coordination with optimization
3. Thread-safe shared memory operations
4. Configuration management
5. Serialization/deserialization support
6. Performance tracking and metrics
"""

import logging
import numpy as np
import yaml
import random
import json
import threading
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from abc import ABC, abstractmethod
import unittest
from datetime import datetime

# Scientific computing import with error handling
try:
    from scipy.stats import beta
    SCI_AVAILABLE = True
except ImportError:
    SCI_AVAILABLE = False
    logging.warning("SciPy not available - Bayesian features will be limited")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('collaborative_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk assessment levels with auto-generated values"""
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    CRITICAL = auto()

@dataclass
class SafetyAssessment:
    """Safety assessment with serialization support"""
    risk_score: float
    risk_level: RiskLevel
    recommended_action: str
    confidence: float = 1.0
    affected_agents: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def serialize(self) -> str:
        """Convert to JSON string"""
        return json.dumps({
            **asdict(self),
            'risk_level': self.risk_level.name,
            'timestamp': self.timestamp
        })
    
    @classmethod
    def deserialize(cls, json_str: str) -> 'SafetyAssessment':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls(
            risk_score=data['risk_score'],
            risk_level=RiskLevel[data['risk_level']],
            recommended_action=data['recommended_action'],
            confidence=data['confidence'],
            affected_agents=data['affected_agents'],
            timestamp=data.get('timestamp', datetime.now().timestamp())
        )

class ThreadSafeSharedMemory:
    """Thread-safe key-value store with expiration support"""
    def __init__(self):
        self._data = {}
        self._lock = threading.RLock()
        self._expirations = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._expirations and self._expirations[key] < datetime.now().timestamp():
                del self._data[key]
                del self._expirations[key]
                return default
            return self._data.get(key, default)
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            self._data[key] = value
            if ttl:
                self._expirations[key] = datetime.now().timestamp() + ttl
    
    def update(self, updates: Dict[str, Any]) -> None:
        with self._lock:
            self._data.update(updates)
    
    def clear_expired(self) -> None:
        with self._lock:
            now = datetime.now().timestamp()
            expired = [k for k, v in self._expirations.items() if v < now]
            for key in expired:
                del self._data[key]
                del self._expirations[key]

class BayesianThresholdAdapter:
    """Adaptive threshold calculator using Bayesian methods"""
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.alpha = prior_alpha
        self.beta = prior_beta
    
    def update(self, successes: int, failures: int) -> None:
        """Update with new success/failure data"""
        self.alpha += successes
        self.beta += failures
    
    def get_threshold(self, percentile: float = 0.9) -> float:
        """Calculate current threshold"""
        if not SCI_AVAILABLE:
            return max(0.1, min(0.9, self.alpha / (self.alpha + self.beta)))
        
        if self.alpha + self.beta <= 1:
            return 0.5  # Default prior
        return beta.ppf(percentile, self.alpha, self.beta)

class TaskScheduler(ABC):
    """Abstract scheduler interface"""
    @abstractmethod
    def schedule(self, tasks: List[Dict], agents: Dict[str, Any]) -> Dict:
        pass

class DeadlineAwareScheduler(TaskScheduler):
    """Earliest Deadline First scheduler with capability matching"""
    def schedule(self, tasks: List[Dict], agents: Dict[str, Any]) -> Dict:
        # Validate input
        if not tasks or not agents:
            return {}
            
        # Sort by deadline then priority
        sorted_tasks = sorted(
            tasks,
            key=lambda x: (x.get('deadline', float('inf')), 
            -x.get('priority', 0))
        )
        
        assignments = {}
        agent_loads = {agent: 0.0 for agent in agents}
        
        for task in sorted_tasks:
            task_id = task.get('id', str(hash(str(task))))
            best_agent, best_score = None, -1
            best_requirements = []
            
            for agent, details in agents.items():
                # Skip overloaded agents
                if agent_loads[agent] >= 1.0:
                    continue
                
                # Calculate capability match
                capabilities = set(details.get('capabilities', []))
                requirements = set(task.get('requirements', []))
                matched = list(capabilities & requirements)
                match_score = len(matched) / max(1, len(requirements))
                
                # Adjust for load balancing
                load_factor = 1 - details.get('current_load', 0)
                total_score = match_score * load_factor
                
                if total_score > best_score:
                    best_score = total_score
                    best_agent = agent
                    best_requirements = matched
            
            if best_agent:
                # Calculate timing considering dependencies
                duration = task.get('estimated_duration', 0)
                dependencies = task.get('dependencies', [])
                dep_end_times = [assignments[dep]['end_time'] for dep in dependencies if dep in assignments]
                start_time = max(
                    agent_loads[best_agent],
                    max(dep_end_times, default=0)
                )
                end_time = start_time + duration
                
                assignments[task_id] = {
                    'agent': best_agent,
                    'start_time': start_time,
                    'end_time': end_time,
                    'risk_score': task.get('estimated_risk', 0.5),
                    'requirements_met': best_requirements,
                    'task_type': task.get('type', 'unknown')
                }
                agent_loads[best_agent] = end_time
        
        return assignments

class CollaborativeAgent:
    """Main collaborative agent implementation"""
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 shared_memory: Optional[Any] = None, 
                 risk_threshold: float = 0.2,
                 agent_network: Optional[Dict[str, Any]] = None):
        """
        Initialize collaborative agent.
        
        Args:
            config_path: Path to YAML config file
            shared_memory: Optional shared memory instance
            risk_threshold: Initial risk threshold (0.0-1.0)
            agent_network: Predefined agent network dictionary
        """
        self.name = "CollaborativeAgent"
        self.shared_memory = shared_memory or ThreadSafeSharedMemory()
        self.risk_threshold = max(0.0, min(1.0, risk_threshold))
        
        # Load configuration
        self.agent_network = self._load_config(config_path) if config_path else agent_network or {}
        
        # Initialize risk model
        self.risk_model = {
            'task_risks': defaultdict(list),
            'agent_risks': defaultdict(list),
            'thresholds': defaultdict(BayesianThresholdAdapter)
        }
        self.risk_model['thresholds']['default'] = BayesianThresholdAdapter()
        
        # Initialize components
        self.scheduler = DeadlineAwareScheduler()
        self._metrics_lock = threading.Lock()
        self._init_performance_metrics()
        
        logger.info(f"Initialized {self.name} with {len(self.agent_network)} agents")

    def _init_performance_metrics(self) -> None:
        """Initialize performance tracking metrics"""
        self.performance_metrics = {
            'assessments_completed': 0,
            'interventions': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'tasks_coordinated': 0,
            'coordination_failures': 0
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load agent network configuration from YAML file"""
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config.get('agent_network', {})
        except Exception as e:
            logger.error(f"Config load error: {e}")
            return {}

    def _validate_risk_score(self, score: float) -> None:
        """Validate risk score is between 0.0 and 1.0"""
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Risk score {score} out of bounds [0.0, 1.0]")

    def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute task with safety monitoring.
        
        Args:
            task_data: Dictionary containing:
                - policy_risk_score: Required risk score (0.0-1.0)
                - task_type: Task category
                - source_agent: Originating agent
                - action_details: Action metadata
                
        Returns:
            Dictionary with assessment results
        """
        try:
            # Validate input
            if 'policy_risk_score' not in task_data:
                return self._error_response("Missing policy_risk_score")
            
            risk_score = task_data['policy_risk_score']
            self._validate_risk_score(risk_score)
            
            # Perform assessment
            assessment = self.assess_risk(
                risk_score=risk_score,
                task_type=task_data.get('task_type', 'general'),
                source_agent=task_data.get('source_agent', 'unknown'),
                action_details=task_data.get('action_details')
            )
            
            # Update knowledge base
            self._update_shared_knowledge(assessment, task_data)
            
            # Prepare response
            return {
                'status': 'success',
                'assessment': assessment.__dict__,
                'performance_metrics': self._init_performance_metrics()
            }
            
        except ValueError as e:
            return self._error_response(f"Validation error: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error in execute()")
            return self._error_response(f"Internal error: {str(e)}")

    def assess_risk(self, 
                   risk_score: float, 
                   task_type: str = "general",
                   source_agent: str = "unknown",
                   action_details: Optional[Dict] = None) -> SafetyAssessment:
        """
        Perform comprehensive risk assessment.
        
        Args:
            risk_score: Computed risk score (0.0-1.0)
            task_type: Task category
            source_agent: Originating agent
            action_details: Action metadata
            
        Returns:
            SafetyAssessment object
        """
        try:
            self._validate_risk_score(risk_score)
            
            threshold = self._get_risk_threshold(task_type, source_agent)
            risk_level = self._calculate_risk_level(risk_score, threshold)
            
            recommendation = self._generate_recommendation(
                risk_score, risk_level, source_agent, action_details
            )
            
            # Update metrics
            with self._metrics_lock:
                self.performance_metrics['assessments_completed'] += 1
                if risk_level.value >= RiskLevel.HIGH.value:
                    self.performance_metrics['interventions'] += 1
            
            return SafetyAssessment(
                risk_score=risk_score,
                risk_level=risk_level,
                recommended_action=recommendation,
                confidence=self._calculate_confidence(risk_score, threshold),
                affected_agents=self._identify_affected_agents(source_agent, action_details)
            )
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            return SafetyAssessment(
                risk_score=1.0,
                risk_level=RiskLevel.CRITICAL,
                recommended_action="halt_immediately",
                confidence=0.0,
                affected_agents=[]
            )

    def coordinate_tasks(self, tasks, available_agents, optimization_goals=None):
        """
        Coordinates tasks among agents and returns scheduling, safety, and metrics.
        """
        safety_checks = {}
        optimized_schedule = {}
        # Your task coordination logic should populate safety_checks and optimized_schedule
        return {
            'status': 'success',
            'assignments': optimized_schedule,
            'safety_checks': {k: v.__dict__ for k, v in safety_checks.items()},
            'metadata': self._calculate_coordination_metrics(
                optimized_schedule,
                safety_checks,
                optimization_goals
            )
        }
    def _apply_optimizations(
        self,
        schedule: Dict[str, Any],
        tasks: List[Dict[str, Any]],
        agents: Dict[str, Any],
        goals: List[str],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply optimization goals to the schedule.
        """
        optimized_schedule = schedule.copy()
    
        # Apply max concurrent tasks constraint
        if 'max_concurrent_tasks' in constraints:
            agent_counts = defaultdict(int)
            for task_id, assignment in optimized_schedule.items():
                agent = assignment['agent']
                agent_counts[agent] += 1
                if agent_counts[agent] > constraints['max_concurrent_tasks']:
                    optimized_schedule[task_id]['status'] = 'rejected_overload'
    
        # Score assignments based on optimization goals
        for task_id, assignment in optimized_schedule.items():
            if assignment.get('status') in ['rejected_high_risk', 'rejected_overload']:
                continue
            
            agent = assignment['agent']
            agent_details = agents[agent]
            task = next(t for t in tasks if t['id'] == task_id)
        
            score = 0
            weight = 1.0 / len(goals)  # Equal weighting
        
            for goal in goals:
                if goal == 'minimize_risk':
                    score += weight * (1 - assignment['risk_score'])
                elif goal == 'maximize_throughput':
                    score += weight * (agent_details['throughput'] / 100)
                elif goal == 'balance_load':
                    score += weight * (1 - agent_details['current_load'])
                elif goal == 'minimize_makespan':
                    score += weight * (1 / (assignment['end_time'] + 0.001))
        
            optimized_schedule[task_id]['optimization_score'] = score
    
        # For tasks with multiple possible agents, select best based on score
        # (Implementation depends on your scheduler's flexibility)
    
    
    
    def train_risk_model(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Enhanced training with Bayesian threshold adaptation
        """
        logger.info(f"Training risk model with {len(training_data)} samples")
        
        # Process training data
        for sample in training_data:
            try:
                self._validate_risk_score(sample['risk_score'])
                task_type = sample.get('task_type', 'general')
                agent = sample.get('agent', 'default')
                
                # Store risk data
                self.risk_model['task_risks'][task_type].append(sample['risk_score'])
                self.risk_model['agent_risks'][agent].append(sample['risk_score'])
                
                # Update Bayesian adapters
                threshold_key = f"{task_type}_{agent}"
                adapter = self.risk_model['thresholds'].get(threshold_key, BayesianThresholdAdapter())
                
                if 'outcome' in sample:
                    if sample['outcome']:
                        adapter.update(1, 0)  # Success
                    else:
                        adapter.update(0, 1)  # Failure
                    
                    # Update performance metrics
                    with self._metrics_lock:
                        if not sample['outcome'] and sample['risk_score'] < self._get_risk_threshold(task_type, agent):
                            self.performance_metrics['false_negatives'] += 1
                        elif sample['outcome'] and sample['risk_score'] > self._get_risk_threshold(task_type, agent):
                            self.performance_metrics['false_positives'] += 1
                
                self.risk_model['thresholds'][threshold_key] = adapter
                
            except ValueError as e:
                logger.warning(f"Invalid training sample skipped: {e}")
        
        logger.info("Risk model training completed")

    def _get_risk_threshold(self, task_type: str, agent: str = "default") -> float:
        """
        Get threshold considering both task and agent factors using Bayesian adaptation
        """
        # Try combined task-agent threshold first
        threshold_key = f"{task_type}_{agent}"
        if threshold_key in self.risk_model['thresholds']:
            return self.risk_model['thresholds'][threshold_key].get_threshold()
        
        # Fall back to task-specific threshold
        if task_type in self.risk_model['thresholds']:
            return self.risk_model['thresholds'][task_type].get_threshold()
            
        # Final fallback to default threshold
        return self.risk_model['thresholds']['default'].get_threshold()
        
    def _calculate_risk_level(self, risk_score: float, threshold: float) -> RiskLevel:
        """
        Calculate risk level based on score and threshold.
        
        Args:
            risk_score: Computed risk score
            threshold: Current risk threshold
            
        Returns:
            Appropriate RiskLevel enum
        """
        if risk_score < threshold * 0.5:
            return RiskLevel.LOW
        elif risk_score < threshold:
            return RiskLevel.MODERATE
        elif risk_score < threshold * 1.5:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _generate_recommendation(self,
                               risk_score: float,
                               risk_level: RiskLevel,
                               source_agent: str,
                               action_details: Optional[Dict]) -> str:
        """
        Generate appropriate recommendation based on risk assessment.
        
        Args:
            risk_score: Computed risk score
            risk_level: Determined risk level
            source_agent: Originating agent
            action_details: Details of proposed action
            
        Returns:
            Recommended action string
        """
        if risk_level == RiskLevel.LOW:
            return "proceed"
        elif risk_level == RiskLevel.MODERATE:
            return "proceed_with_caution"
        elif risk_level == RiskLevel.HIGH:
            return self._generate_mitigation_strategy(source_agent, action_details)
        else:  # CRITICAL
            return "halt_immediately"

    def _generate_mitigation_strategy(self, 
                                    source_agent: str,
                                    action_details: Optional[Dict]) -> str:
        """
        Generate specific mitigation strategy for high-risk actions.
        
        Args:
            source_agent: Originating agent
            action_details: Details of proposed action
            
        Returns:
            Specific mitigation strategy
        """
        # Default mitigation strategies
        strategies = [
            "reduce_action_entropy",
            "add_safety_constraints",
            "require_human_approval",
            "switch_to_safer_agent"
        ]
        
        # Agent-specific strategies
        if source_agent in self.agent_network:
            if 'rl' in self.agent_network[source_agent].get('type', ''):
                return "reduce_exploration_rate"
            elif 'planning' in self.agent_network[source_agent].get('type', ''):
                return "add_precondition_checks"
        
        # Action-specific strategies
        if action_details:
            if 'physical_action' in action_details.get('tags', []):
                return "enable_physical_safeguards"
            elif 'data_modification' in action_details.get('tags', []):
                return "create_backup_first"
        
        return random.choice(strategies)  # Fallback

    def _update_shared_knowledge(self, 
                               assessment: SafetyAssessment,
                               task_data: Dict[str, Any]) -> None:
        """
        Update shared knowledge base with assessment results.
        
        Args:
            assessment: SafetyAssessment object
            task_data: Original task data
        """
        if self.shared_memory:
            knowledge_update = {
                'risk_assessment': assessment.__dict__,
                'task_metadata': {
                    'type': task_data.get('task_type', 'general'),
                    'source': task_data.get('source_agent', 'unknown'),
                    'timestamp': task_data.get('timestamp')
                }
            }
            self.shared_memory.set(
                f"risk_assessment_{task_data.get('task_id')}",
                knowledge_update
            )

    def _select_optimal_agent(self, 
                            task: Dict[str, Any],
                            available_agents: List[str]) -> Optional[str]:
        """
        Select the best agent for a given task based on capabilities and risk profile.
        
        Args:
            task: Task description dictionary
            available_agents: List of available agent names
            
        Returns:
            Name of selected agent or None if no suitable agent found
        """
        if not available_agents:
            return None
            
        # Score agents based on capability matching and risk profile
        agent_scores = []
        for agent in available_agents:
            if agent not in self.agent_network:
                continue
                
            # Capability matching
            capability_match = sum(
                1 for req in task.get('requirements', [])
                if req in self.agent_network[agent].get('capabilities', [])
            ) / max(1, len(task.get('requirements', [])))
            
            # Risk adjustment
            agent_risk = np.mean(self.risk_model['agent_risks'].get(agent, [0.5]))
            risk_factor = 1 - min(agent_risk, 0.9)  # Never go below 0.1
            
            agent_scores.append((agent, capability_match * risk_factor))
        
        if not agent_scores:
            return None
            
        return max(agent_scores, key=lambda x: x[1])[0]

    def _calculate_confidence(self, risk_score: float, threshold: float) -> float:
        """Calculate confidence score for assessment (0.0-1.0)."""
        distance = abs(risk_score - threshold)
        return max(0.0, 1.0 - (distance * 2))  # 1.0 when exactly at threshold

    def _identify_affected_agents(self,
                                source_agent: str,
                                action_details: Optional[Dict]) -> List[str]:
        """Identify other agents that might be affected by this action."""
        if not action_details or 'affected_components' not in action_details:
            return []
            
        return [
            agent for agent in self.agent_network
            if any(
                comp in self.agent_network[agent].get('components', [])
                for comp in action_details['affected_components']
            )
        ]

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Generate standardized error response."""
        logger.error(message)
        return {
            'status': 'error',
            'error': message,
            'assessment': None,
            'recommendations': []
        }

class TestCollaborativeAgent(unittest.TestCase):
    """Comprehensive unit tests for CollaborativeAgent"""
    
    def setUp(self):
        self.agent = CollaborativeAgent(risk_threshold=0.3)
        
    def test_risk_score_validation(self):
        with self.assertRaises(ValueError):
            self.agent._validate_risk_score(-0.1)
        with self.assertRaises(ValueError):
            self.agent._validate_risk_score(1.1)
        
        # Should not raise
        self.agent._validate_risk_score(0.0)
        self.agent._validate_risk_score(0.5)
        self.agent._validate_risk_score(1.0)
    

    def _calculate_coordination_metrics(self, assignments, safety_checks):
        total_duration = 0
        total_risk = 0
        compliant_tasks = 0

        for assignment in assignments.values():
            duration = assignment.get("end_time", 0) - assignment.get("start_time", 0)
            total_duration += duration

        for task_id, check in safety_checks.items():
            risk = check.get("risk_score", 1.0)
            total_risk += risk
            if check.get("risk_level", "") in ["low", "moderate"]:
                compliant_tasks += 1

        num_tasks = len(assignments) or 1
        avg_risk = total_risk / num_tasks
        safety_compliance = compliant_tasks / num_tasks

        return {
            "total_duration": round(total_duration, 2),
            "average_risk_score": round(avg_risk, 2),
            "safety_compliance": round(safety_compliance, 2),
            "resource_utilization": f"{round(num_tasks / len(self.agent_network) * 100, 2)}%"
        }
    def test_safety_assessment_serialization(self):
        assessment = SafetyAssessment(
            risk_score=0.7,
            risk_level=RiskLevel.HIGH,
            recommended_action="halt",
            confidence=0.9,
            affected_agents=["agent1", "agent2"]
        )
        
        serialized = assessment.serialize()
        deserialized = SafetyAssessment.deserialize(serialized)
        
        self.assertEqual(assessment.risk_score, deserialized.risk_score)
        self.assertEqual(assessment.risk_level, deserialized.risk_level)
        self.assertEqual(assessment.recommended_action, deserialized.recommended_action)

# Example usage
if __name__ == "__main__":
    # Initialize with complete agent network
    agent = CollaborativeAgent(
        agent_network={
            # Safety and Coordination Agents
            'SafeAI_Agent': {
                'type': 'safety',
                'capabilities': ['risk_assessment', 'threshold_adjustment'],
                'components': ['safety_module']
            },
            
            # Reinforcement Learning Agents
            'DQN_Agent': {
                'type': 'dqn',
                'capabilities': ['deep_q_learning', 'experience_replay'],
                'components': ['replay_buffer', 'q_network']
            },
            
            # Specialized RL Agents
            'MultiTaskRL_Agent': {
                'type': 'multitask',
                'capabilities': ['task_embedding', 'shared_representation'],
                'components': ['task_encoder']
            },
            'MAML_Agent': {
                'type': 'meta_learning',
                'capabilities': ['few_shot_adaptation', 'gradient_based_meta'],
                'components': ['meta_policy']
            },
            
            # Financial/Technical Agents
            'RSI_Agent': {
                'type': 'technical',
                'capabilities': ['market_analysis', 'trend_detection'],
                'components': ['technical_indicators']
            },
            
            # Cognitive Agents
            'NL_Agent': {
                'type': 'nlp',
                'capabilities': ['language_understanding', 'text_generation'],
                'components': ['language_model']
            },
            'Reasoning_Agent': {
                'type': 'knowledge',
                'capabilities': ['logical_inference', 'rule_based_reasoning'],
                'components': ['knowledge_graph']
            },
            
            # Planning/Execution Agents
            'Planning_Agent': {
                'type': 'planning',
                'capabilities': ['task_decomposition', 'resource_scheduling'],
                'components': ['planner_module']
            },
            'Execution_Agent': {
                'type': 'action',
                'capabilities': ['operation_execution', 'environment_interaction'],
                'components': ['actuator_interface']
            },
            
            # Perception Agents
            'Perception_Agent': {
                'type': 'sensors',
                'capabilities': ['multimodal_fusion', 'feature_extraction'],
                'components': ['sensor_processor']
            },
            
            # Adaptive Agents
            'Adaptive_Agent': {
                'type': 'self_optimizing',
                'capabilities': ['dynamic_parameter_adjustment', 'context_awareness'],
                'components': ['adaptation_engine']
            },
            
            # Evaluation Agents
            'Evaluation_Agent': {
                'type': 'metrics',
                'capabilities': ['performance_analysis', 'statistical_testing'],
                'components': ['metrics_dashboard']
            }
        },
        risk_threshold=0.35
    )

    # Complex safety assessment scenario
    task_data = {
        'policy_risk_score': 0.62,
        'task_type': 'cross_agent_coordination',
        'source_agent': 'Adaptive_Agent',
        'action_details': {
            'type': 'system_parameter_update',
            'tags': ['multitask', 'exploration_adjustment'],
            'affected_components': ['dqn', 'meta_policy', 'actuator_interface'],
            'parameters': {
                'exploration_rate': 0.25,
                'learning_rate': 0.001,
                'safety_margin': 0.15
            }
        }
    }
    
    # Execute assessment with full agent context
    result = agent.execute(task_data)
    print("Multi-Agent Safety Assessment Result:")
    print(f"Risk Level: {result['assessment']['risk_level']}")
    print(f"Recommendation: {result['assessment']['recommended_action']}")
    print(f"Affected Agents: {result['assessment']['affected_agents']}")
    print("\nDetailed Assessment:")
    print(json.dumps(result, indent=2))

    # Demonstrate cross-agent coordination
    print("\nPerformance Metrics Overview:")
    print(f"Total Assessments: {result['performance_metrics']['assessments_completed']}")
    print(f"System Interventions: {result['performance_metrics']['interventions']}")
    print(f"Current False Positive Rate: {result['performance_metrics']['false_positives']/result['performance_metrics']['assessments_completed']:.2%}")

# Enhanced Task Coordination Example
    # Define a complex set of tasks with dependencies and constraints
    tasks = [
        {
            'id': 't1',
            'type': 'real_time_decision',
            'requirements': ['low_latency', 'high_accuracy', 'safety_critical'],
            'deadline': 0.5,  # seconds
            'priority': 9,
            'estimated_duration': 0.3,
            'dependencies': []
        },
        {
            'id': 't2',
            'type': 'long_term_planning',
            'requirements': ['strategic_thinking', 'resource_optimization', 'multi_agent'],
            'deadline': 5.0,
            'priority': 7,
            'estimated_duration': 2.5,
            'dependencies': ['t4']
        },
        {
            'id': 't3',
            'type': 'risk_assessment',
            'requirements': ['safety_analysis', 'real_time_monitoring'],
            'deadline': 1.0,
            'priority': 8,
            'estimated_duration': 0.8,
            'dependencies': ['t1']
        },
        {
            'id': 't4',
            'type': 'data_processing',
            'requirements': ['large_scale_processing', 'pattern_recognition'],
            'deadline': 3.0,
            'priority': 6,
            'estimated_duration': 1.2,
            'dependencies': []
        }
    ]

    # Available agents with their capabilities and current workload
    available_agents = {
        'SafeAI_Agent': {
            'capabilities': ['safety_analysis', 'risk_assessment'],
            'current_load': 0.4,
            'throughput': 5  # tasks/hour
        },
        'DQN_Agent': {
            'capabilities': ['low_latency', 'high_accuracy'],
            'current_load': 0.7,
            'throughput': 20
        },
        'MultiTaskRL_Agent': {
            'capabilities': ['multi_agent', 'resource_optimization'],
            'current_load': 0.3,
            'throughput': 15
        },
        'Planning_Agent': {
            'capabilities': ['strategic_thinking', 'long_term_planning'],
            'current_load': 0.5,
            'throughput': 10
        },
        'Perception_Agent': {
            'capabilities': ['pattern_recognition', 'real_time_monitoring'],
            'current_load': 0.6,
            'throughput': 18
        }
    }

    # Execute coordination with optimization parameters
    coordination_result = agent.coordinate_tasks(
        tasks=tasks,
        available_agents=available_agents,
        optimization_goals=['minimize_risk', 'maximize_throughput'],
        constraints={
            'max_concurrent_tasks': 3,
            'safety_threshold': 0.4
        }
    )

    # Print structured results
    print("\n=== Task Coordination Results ===")
    print(f"Total Tasks: {len(tasks)}")
    print(f"Successfully Assigned: {len(coordination_result['assignments'])}")
    print(f"High Risk Tasks: {coordination_result['metadata']['high_risk_tasks']}")
    
    print("\nDetailed Assignments:")
    for task_id, assignment in coordination_result['assignments'].items():
        task = next(t for t in tasks if t['id'] == task_id)
        risk_level = coordination_result['safety_checks'][task_id]['risk_level']
        print(f"\nTask {task_id} ({task['type']}):")
        print(f"  Assigned to: {assignment['agent']}")
        print(f"  Start Time: {assignment['start_time']:.1f}s")
        print(f"  End Time: {assignment['end_time']:.1f}s")
        print(f"  Risk Assessment: {risk_level}")
        print(f"  Requirements Met: {', '.join(assignment['requirements_met'])}")
    print(f"Total Estimated Duration: {coordination_result['metadata']['total_duration']:.1f}s")
    print(f"Resource Utilization: {coordination_result['metadata']['resource_utilization']:.1%}")
    print(f"Safety Compliance: {coordination_result['metadata']['safety_compliance']:.1%}")

    # Visualize schedule
    print("\nGantt Chart Representation:")
    for task_id, assignment in coordination_result['assignments'].items():
        task = next(t for t in tasks if t['id'] == task_id)
        duration = assignment['schedule']['end_time'] - assignment['schedule']['start_time']
        bar = '█' * int(20 * duration / coordination_result['metadata']['total_duration'])
        print(f"{task_id} [{bar}] {assignment['schedule']['start']:.1f}-{assignment['schedule']['end_time']:.1f}s ({assignment['agent']})")


if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        # Dummy data simulation for visualization
        coordination_result = {
            "assignments": {
                0: {"agent": "Planning_Agent", "start_time": 0.0, "end_time": 5.5},
                1: {"agent": "NL_Agent", "start_time": 5.5, "end_time": 10.0},
                2: {"agent": "Execution_Agent", "start_time": 10.0, "end_time": 15.0},
                3: {"agent": "Evaluation_Agent", "start_time": 15.0, "end_time": 18.0}
            }
        }
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab10.colors
        agent_color_map = {}
        for i, (task_id, assignment) in enumerate(coordination_result["assignments"].items()):
            agent = assignment["agent"]
            start = assignment["start_time"]
            end = assignment["end_time"]
            duration = end - start
            if agent not in agent_color_map:
                agent_color_map[agent] = colors[len(agent_color_map) % len(colors)]
            ax.barh(y=task_id, width=duration, left=start, height=0.5, color=agent_color_map[agent])
            ax.text(start + duration / 2, task_id, agent, va="center", ha="center", color="white", fontsize=9)
        ax.set_yticks([task_id for task_id in coordination_result["assignments"]])
        ax.set_yticklabels([f"Task {task_id}" for task_id in coordination_result["assignments"]])
        ax.set_xlabel("Time (s)")
        ax.set_title("SLAI Agent Task Assignment Timeline")
        legend_handles = [mpatches.Patch(color=color, label=agent) for agent, color in agent_color_map.items()]
        ax.legend(handles=legend_handles, title="Agents", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig("/mnt/data/slai_gantt_chart_from_script.png")
        print("✅ Gantt chart saved to slai_gantt_chart_from_script.png")
    except Exception as e:
        print("Gantt chart generation failed:", str(e))
