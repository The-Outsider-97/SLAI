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
                'performance_metrics': self._get_performance_metrics()
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

    def coordinate_tasks(
        self,
        tasks: List[Dict[str, Any]],
        available_agents: List[str],
        optimization_goals: List[str] = None,
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Coordinate task assignments with optimizations - Single Corrected Implementation.
        
        Args:
            tasks: List of tasks to assign
            available_agents: Available agent names
            optimization_goals: List of optimization priorities
            constraints: Task assignment constraints
            
        Returns:
            Dictionary with assignments and metrics
        """
        try:
            # Validate inputs
            if not tasks:
                return self._error_response("No tasks provided")
            if not available_agents:
                return self._error_response("No agents available")
                
            # Set defaults
            optimization_goals = optimization_goals or ['minimize_risk']
            constraints = constraints or {}
            
            # Filter eligible agents
            eligible_agents = {
                agent: details 
                for agent, details in self.agent_network.items()
                if agent in available_agents and 
                   self._meets_constraints(agent, details, constraints)
            }
            
            if not eligible_agents:
                return self._error_response("No agents meet constraints")
            
            # Generate and optimize schedule
            schedule = self.scheduler.schedule(tasks, eligible_agents)
            optimized_schedule = self._apply_optimizations(
                schedule, tasks, eligible_agents, optimization_goals, constraints
            )
            
            # Perform safety checks
            safety_checks = {}
            for task_id, assignment in optimized_schedule.items():
                task = next(t for t in tasks if t.get('id', str(hash(str(t)))) == task_id)
                assessment = self.assess_risk(
                    risk_score=assignment['risk_score'],
                    task_type=assignment.get('task_type', 'unknown'),
                    source_agent=assignment['agent'],
                    action_details=task
                )
                safety_checks[task_id] = assessment
                
                # Apply safety threshold
                if 'safety_threshold' in constraints:
                    if assessment.risk_score > constraints['safety_threshold']:
                        optimized_schedule[task_id]['status'] = 'rejected_high_risk'
            
            # Update metrics
            with self._metrics_lock:
                self.performance_metrics['tasks_coordinated'] += len(optimized_schedule)
                if len(optimized_schedule) < len(tasks):
                    self.performance_metrics['coordination_failures'] += 1
            
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
            
        except Exception as e:
            logger.exception("Task coordination failed")
            return self._error_response(f"Coordination error: {str(e)}")

    def _meets_constraints(self, agent: str, details: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """
        Check if agent meets all given constraints.
        
        Args:
            agent: Agent name
            details: Agent capabilities/details
            constraints: Constraints to check
            
        Returns:
            bool: True if agent meets all constraints
        """
        # Check required capabilities
        if 'required_capabilities' in constraints:
            if not all(cap in details.get('capabilities', []) 
                      for cap in constraints['required_capabilities']):
                return False
                
        # Check maximum load
        if 'max_load' in constraints:
            if details.get('current_load', 0) > constraints['max_load']:
                return False
                
        return True

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get a thread-safe copy of performance metrics"""
        with self._metrics_lock:
            return self.performance_metrics.copy()

    def _calculate_coordination_metrics(
        self,
        assignments: Dict[str, Any],
        safety_checks: Dict[str, SafetyAssessment],
        optimization_goals: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive coordination metrics.
        
        Args:
            assignments: Task assignments
            safety_checks: Safety assessments
            optimization_goals: List of optimization goals
            
        Returns:
            Dictionary of calculated metrics
        """
        total_duration = 0.0
        total_risk = 0.0
        compliant_tasks = 0
        num_tasks = len(assignments)

        for assignment in assignments.values():
            duration = assignment.get("end_time", 0) - assignment.get("start_time", 0)
            total_duration += duration

        for task_id, check in safety_checks.items():
            risk = check.risk_score
            total_risk += risk
            if check.risk_level.value < RiskLevel.HIGH.value:
                compliant_tasks += 1

        avg_risk = total_risk / num_tasks if num_tasks > 0 else 0.0
        safety_compliance = compliant_tasks / num_tasks if num_tasks > 0 else 0.0

        return {
            "total_tasks": num_tasks,
            "assigned_tasks": num_tasks,
            "high_risk_tasks": num_tasks - compliant_tasks,
            "total_duration": round(total_duration, 2),
            "average_risk": round(avg_risk, 2),
            "safety_compliance": round(safety_compliance, 2),
            "optimization_goals": optimization_goals,
            "resource_utilization": f"{round(num_tasks / len(self.agent_network) * 100, 1)}%" 
                                   if self.agent_network else "0%"
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
    max_duration = coordination_result['metadata']['total_duration']
    for task_id, assignment in coordination_result['assignments'].items():
        duration = assignment['end_time'] - assignment['start_time']
        bar_length = int(20 * duration / max_duration) if max_duration > 0 else 0
        bar = '█' * max(1, bar_length)  # Ensure at least one character
        print(f"{task_id} [{bar}] {assignment['start_time']:.1f}-{assignment['end_time']:.1f}s ({assignment['agent']})")
