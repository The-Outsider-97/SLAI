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

import os, sys
import logging
import numpy as np
import yaml
import random
import json
import threading
import unittest

from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from enum import Enum, auto
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime

from src.utils.agent_factory import AgentFactory
from src.agents.language.grammar_processor import GrammarProcessor

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.WARNING)

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
        """Convert to JSON string without timestamp duplication"""
        data = asdict(self)
        data['risk_level'] = self.risk_level.name
        return json.dumps(data)

    @classmethod
    def deserialize(cls, json_str: str) -> 'SafetyAssessment':
        data = json.loads(json_str)
        if 'timestamp' not in data:
            logger.warning("Deserializing assessment without timestamp")
        return cls(**data)

class ThreadSafeSharedMemory:
    """Thread-safe key-value store with expiration support"""
    def __init__(self):
        self.agent_cache = {}
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
    
    def __init__(self, shared_memory, agent_factory: AgentFactory,
                 config: Optional[Dict] = None,
                 risk_threshold=0.2, agent_network: Optional[Dict] = None,
                 config_path: Optional[str] = None):
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.risk_threshold = max(0.0, min(1.0, risk_threshold))
        self.agents = {}
        self.shared_resources = {
            "shared_memory": shared_memory,
            "task_router": {}
        }

        config = config or {}
        
        # Load config if needed
        if config_path and not agent_network:
            self.agent_network = self._load_config(config_path)
        else:
            self.agent_network = agent_network or {}

        # Prefer manual config if valid
        if isinstance(config, dict) and ("agents" in config or "agent-network" in config):
            for agent_type, agent_cfg in config["agents"].items():
                try:
                    agent = self.factory.create(agent_type, agent_cfg)
                    self.agents[agent_type] = agent
                except Exception as e:
                    logging.warning(f"Failed to initialize {agent_type}: {e}")

        elif self.agent_network:
            for agent_name, agent_info in self.agent_network.items():
                try:
                    agent = self.agent_factory.create(agent_name, agent_info)
                    self.agents[agent_name] = agent
                except Exception as e:
                    logging.warning(f"Failed to load {agent_name}: {e}")
        else:
            logging.warning(f"CollaborativeAgent: Ignored invalid config type: {type(config)}")

        logger.info(f"Initialized CollaborativeAgent with {len(self.agents)} agents")

        # Core components
        self.name = "CollaborativeAgent"
        self.shared_resources = {
            "shared_memory": {},  # Use actual shared components if needed
            "task_router": {}
        }
        self.factory = AgentFactory(config=config, shared_resources=self.shared_resources)
        from src.agents.learning_agent import LearningAgent, SLAIEnv
        self.factory.register("learning",
                              lambda **kwargs: LearningAgent(
                                  env=SLAIEnv(),
                                  config=kwargs,
                                  shared_memory=self.shared_memory,
                                  args=kwargs.get("args", ()),
                                  kwargs=kwargs.get("kwargs", {})
                                  ))

        self.grammar = GrammarProcessor(lang='en')
        self.risk_model = {
            'task_risks': defaultdict(list),
            'agent_risks': defaultdict(list),
            'thresholds': defaultdict(BayesianThresholdAdapter)
        }
        self.risk_model['thresholds']['default'] = BayesianThresholdAdapter()
        self.scheduler = DeadlineAwareScheduler()
        self._init_performance_metrics()
        self.learning_subsystems = {}

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
        failures = self.shared_memory.get("agent_stats", {}).get(self.name, {}).get("errors", [])
        for err in failures:
            if self.is_similar(task_data, err["data"]):
                self.logger.info("Recognized a known problematic case, applying workaround.")
                return self.alternative_execute(task_data)

        errors = self.shared_memory.get(f"errors:{self.name}", [])

        # Check if current task_data has caused errors before
        for error in errors:
            if self.is_similar(task_data, error['task_data']):
                self.handle_known_issue(task_data, error)
                return

        # Proceed with normal execution
        try:
            result = self.perform_task(task_data)
            self.shared_memory.set(f"results:{self.name}", result)
        except Exception as e:
            # Log the failure in shared memory
            error_entry = {'task_data': task_data, 'error': str(e)}
            errors.append(error_entry)
            self.shared_memory.set(f"errors:{self.name}", errors)
            raise

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
                'performance_metrics': self.performance_metrics
            }
            
        except ValueError as e:
            return self._error_response(f"Validation error: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error in execute()")
            return self._error_response(f"Internal error: {str(e)}")

    def alternative_execute(self, task_data):
        """
        Fallback logic when normal execution fails or matches a known failure pattern.
        Attempts to simplify, sanitize, or reroute the input for safer processing.
        """
        try:
            # Step 1: Sanitize task data (remove noise, normalize casing, trim tokens)
            if isinstance(task_data, str):
                clean_data = task_data.strip().lower().replace('\n', ' ')
            elif isinstance(task_data, dict) and "text" in task_data:
                clean_data = task_data["text"].strip().lower()
            else:
                clean_data = str(task_data).strip()

            # Step 2: Apply a safer, simplified prompt or fallback logic
            fallback_prompt = f"Can you try again with simplified input:\n{clean_data}"
            if hasattr(self, "llm") and callable(getattr(self.llm, "generate", None)):
                return self.llm.generate(fallback_prompt)

            # Step 3: If the agent wraps another processor (e.g. GrammarProcessor, LLM), reroute
            if hasattr(self, "grammar") and callable(getattr(self.grammar, "compose_sentence", None)):
                facts = {"event": "fallback", "value": clean_data}
                return self.grammar.compose_sentence(facts)

            # Step 4: Otherwise just echo the cleaned input as confirmation
            return f"[Fallback response] I rephrased your input: {clean_data}"

        except Exception as e:
            # Final fallback — very safe and generic
            return "[Fallback failure] Unable to process your request at this time."        

    def is_similar(self, task_data, past_task_data):
        """
        Compares current task with past task to detect similarity.
        Uses key overlap and value resemblance heuristics.
        """
        if type(task_data) != type(past_task_data):
            return False
    
        # Handle simple text-based tasks
        if isinstance(task_data, str) and isinstance(past_task_data, str):
            return task_data.strip().lower() == past_task_data.strip().lower()
    
        # Handle dict-based structured tasks
        if isinstance(task_data, dict) and isinstance(past_task_data, dict):
            shared_keys = set(task_data.keys()) & set(past_task_data.keys())
            similarity_score = 0
            for key in shared_keys:
                if isinstance(task_data[key], str) and isinstance(past_task_data[key], str):
                    if task_data[key].strip().lower() == past_task_data[key].strip().lower():
                        similarity_score += 1
            # Consider similar if 50% or more keys match closely
            return similarity_score >= (len(shared_keys) / 2)
    
        return False
    
    def handle_known_issue(self, task_data, error):
        """
        Attempt to recover from known failure patterns.
        Could apply input transformation or fallback logic.
        """
        self.logger.warning(f"Handling known issue from error: {error.get('error')}")
    
        # Fallback strategy #1: remove problematic characters
        if isinstance(task_data, str):
            cleaned = task_data.replace("🧠", "").replace("🔥", "")
            self.logger.info(f"Retrying with cleaned input: {cleaned}")
            return self.perform_task(cleaned)
    
        # Fallback strategy #2: modify specific fields in structured input
        if isinstance(task_data, dict):
            cleaned_data = task_data.copy()
            for key, val in cleaned_data.items():
                if isinstance(val, str) and "emoji" in error.get("error", ""):
                    cleaned_data[key] = val.encode("ascii", "ignore").decode()
            self.logger.info("Retrying task with cleaned structured data.")
            return self.perform_task(cleaned_data)
    
        # Fallback strategy #3: return a graceful degradation response
        self.logger.warning("Returning fallback response for unresolvable input.")
        return {"status": "failed", "reason": "Repeated known issue", "fallback": True}
    
    def perform_task(self, task_data):
        """
        Simulated execution method — replace with actual agent logic.
        This is where core functionality would happen.
        """
        self.logger.info(f"Executing task with data: {task_data}")
    
        if isinstance(task_data, str) and "fail" in task_data.lower():
            raise ValueError("Simulated failure due to blacklisted word.")
    
        if isinstance(task_data, dict):
            # Simulate failure on missing required keys
            required_keys = ["input", "context"]
            for key in required_keys:
                if key not in task_data:
                    raise KeyError(f"Missing required key: {key}")
    
        # Simulate result
        return {"status": "success", "result": f"Processed: {task_data}"}

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

    def coordinate_tasks(self, tasks, available_agents, optimization_goals=None, constraints=None):
        """        Full coordination pipeline with safety checks        """
    
        # Validate inputs
        if not tasks or not available_agents:
            return {'status': 'error', 'error': 'Invalid input'}
        
        # Generate initial schedule
        schedule = self.scheduler.schedule(tasks, available_agents)

        # Apply optimizations and safety checks
        optimized = self._apply_optimizations(
            schedule,
            tasks,
            available_agents,
            optimization_goals or [],
            constraints or {}
        )
        
        # Generate safety assessments
        safety_checks = {
            task_id: self.assess_risk(assignment['risk_score'])
            for task_id, assignment in optimized.items()
        }

        # Collect fairness metrics (Added after safety checks)
        from src.utils.metrics_utils import FairnessMetrics, PerformanceMetrics, MetricBridge
        self.metric_bridge = MetricBridge(self.factory)
        
        metrics = {
            'demographic_parity_violations': FairnessMetrics.demographic_parity(
                sensitive_groups=list(self.agent_network.keys()),
                positive_rates={agent: len(assignments.get(agent, []))/len(tasks) 
                                for agent in self.agent_network}
            )[0],
            'calibration_error': PerformanceMetrics.calibration_error(
                y_true=np.array([t.get('priority', 0.5) for t in tasks]),
                probs=np.array([a.get('optimization_score', 0.5) 
                                for a in assignments.values()])
            )
        }
        
        if hasattr(self, 'metric_bridge'):
            self.metric_bridge.submit_metrics(metrics)
    
        # Your task coordination logic should populate safety_checks and optimized_schedule
        return {
            'status': 'success',
            'assignments': optimized,
            'safety_checks': safety_checks,
            'metadata': self._calculate_coordination_metrics(
                optimized,
                safety_checks)
        }

    def _apply_optimizations(
        self,
        schedule: Dict[str, Any],
        tasks: List[Dict[str, Any]],
        agents: Dict[str, Any],
        goals: List[str],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Full optimization pipeline with agent reassignment logic"""
        optimized = schedule.copy()
        agent_loads = defaultdict(float)
        task_map = {t['id']: t for t in tasks}
        max_concurrent = constraints.get('max_concurrent_tasks', 3)

        # Phase 1: Initial scoring
        for task_id, assignment in optimized.items():
            task = task_map[task_id]
            agent = assignment['agent']
            agent_loads[agent] = assignment['end_time']

        # Phase 2: Optimization passes
        for optimization_pass in range(2):  # Two-pass system
            sorted_tasks = sorted(
                optimized.items(),
                key=lambda x: (
                    -x[1].get('optimization_score', 0),
                    x[1].get('start_time', 0)
                ),
                reverse=True
            )

            for task_id, assignment in sorted_tasks:
                task = task_map[task_id]
                current_agent = assignment['agent']
                current_score = assignment.get('optimization_score', 0)
                best_agent = current_agent
                best_score = current_score
                best_timing = (assignment['start_time'], assignment['end_time'])

                # Skip rejected tasks
                if assignment.get('status', '') in ['rejected_high_risk', 'rejected_overload']:
                    continue

                # Find potential better agents
                for agent_name, agent_details in agents.items():
                    if agent_name == current_agent:
                        continue

                    # Capability check
                    if not self._agent_has_capabilities(agent_details, task):
                        continue

                    # Load check
                    current_load = agent_loads[agent_name]
                    if current_load >= 1.0:
                        continue

                    # Calculate new timing considering dependencies
                    dep_end_times = [
                        optimized[dep]['end_time']
                        for dep in task.get('dependencies', [])
                        if dep in optimized
                    ]
                    proposed_start = max(current_load, max(dep_end_times, default=0))
                    proposed_end = proposed_start + task.get('estimated_duration', 0)

                    # Calculate new score
                    score = self._calculate_agent_score(
                        agent_details,
                        task,
                        goals,
                        proposed_start,
                        proposed_end
                    )

                    if score > best_score:
                        best_score = score
                        best_agent = agent_name
                        best_timing = (proposed_start, proposed_end)

                # Apply reassignment if improvement found
                if best_agent != current_agent:
                    # Update agent loads
                    agent_loads[current_agent] = optimized[task_id]['start_time']
                    agent_loads[best_agent] = best_timing[1]

                    # Update assignment
                    optimized[task_id].update({
                        'agent': best_agent,
                        'start_time': best_timing[0],
                        'end_time': best_timing[1],
                        'optimization_score': best_score,
                        'requirements_met': agents[best_agent]['capabilities']
                    })

                    # Update dependent tasks
                    self._update_dependent_tasks(task_id, optimized, task_map)

        # Final constraint enforcement
        for task_id, assignment in optimized.items():
            agent = assignment['agent']
            if agent_loads[agent] > 1.0:
                optimized[task_id]['status'] = 'rejected_overload'
                
        return optimized

    def _agent_has_capabilities(self, agent_details: Dict, task: Dict) -> bool:
        """Check if agent meets all task requirements"""
        required = set(task.get('requirements', []))
        available = set(agent_details.get('capabilities', []))
        return required.issubset(available)

    def _calculate_agent_score(self, agent_details: Dict, task: Dict,
                            goals: List[str], start_time: float,
                            end_time: float) -> float:
        """Multi-factor scoring with temporal considerations"""
        score = 0.0
        duration = end_time - start_time
        task_risk = task.get('estimated_risk', 0.5)
        
        # Weighted goal scoring
        goal_weights = {
            'minimize_risk': 0.4,
            'maximize_throughput': 0.3,
            'balance_load': 0.2,
            'minimize_makespan': 0.1
        }
        
        for goal in goals:
            if goal == 'minimize_risk':
                score += (1 - task_risk) * goal_weights.get(goal, 0.4)
            elif goal == 'maximize_throughput':
                score += (agent_details['throughput'] / 100) * goal_weights.get(goal, 0.3)
            elif goal == 'balance_load':
                load_factor = 1 - agent_details['current_load']
                score += load_factor * goal_weights.get(goal, 0.2)
            elif goal == 'minimize_makespan':
                time_factor = 1 / (duration + 0.001)  # Prevent division by zero
                score += time_factor * goal_weights.get(goal, 0.1)
        
        # Temporal penalty for delayed tasks
        deadline = task.get('deadline', float('inf'))
        if end_time > deadline:
            score *= max(0.1, 1 - (end_time - deadline)/10)  # Linear decay
        
        return score

    def _update_dependent_tasks(self, task_id: str, schedule: Dict, task_map: Dict):
        """Propagate timing changes to dependent tasks"""
        for dependent_id, dependent in schedule.items():
            if task_id in task_map[dependent_id].get('dependencies', []):
                # Recalculate timing for dependent task
                dep_timing = [
                    schedule[dep]['end_time']
                    for dep in task_map[dependent_id].get('dependencies', [])
                ]
                new_start = max(
                    schedule[dependent_id]['start_time'],
                    max(dep_timing, default=0)
                )
                
                if new_start != schedule[dependent_id]['start_time']:
                    duration = task_map[dependent_id]['estimated_duration']
                    schedule[dependent_id]['start_time'] = new_start
                    schedule[dependent_id]['end_time'] = new_start + duration
                    
                    self._update_dependent_tasks(dependent_id, schedule, task_map)
    
    
    def train_risk_model(self, training_data: List[Dict[str, Any]]) -> None:
        """Add learning subsystem integration"""
        super().train_risk_model(training_data)
        
        # Update RSI learner with new risk data
        [d['risk_score'] for d in training_data]
        
    
    def _update_shared_knowledge(self, assessment: SafetyAssessment, task_data: Dict):
        """Enhanced knowledge sharing with learning subsystem"""
        super()._update_shared_knowledge(assessment, task_data)
        
        # Update learning subsystems
        task_data.get('action_details', {}),
        assessment.risk_level.value
        

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

    def generate(self, shared_memory, task_data: Union[str, dict]) -> str:
        from models.slai_lm import SLAILM, get_shared_slailm

        llm = get_shared_slailm(shared_memory)
        agent = self.factory.create("language", {"llm": llm})

        try:
            if isinstance(task_data, str):
                task_data = {"task_type": "language", "prompt": task_data}

            task_type = task_data.get("task_type", "general")
            prompt = task_data.get("prompt", "").strip()[:1000]

            if not prompt:
                raise ValueError("Empty input")

            if "language" not in self.agents:
                llm = SLAILM(shared_memory)
                self.agents["language"] = self.factory.create(
                    "language", {
                        "llm": llm,
                        "history": [],
                        "summary": "",
                        "memory_limit": 1000,
                        "enable_summarization": True,
                        "summarizer": None
                    }
                )

            lang_agent = self.agents["language"]
            lang_agent.dialogue_context.add(prompt)

            intent = lang_agent.parse_intent(prompt) or {"type": "unknown"}
            if not isinstance(intent, dict):
                logger.warning("Intent is not a dict; applying fallback.")
                intent = {"type": str(intent), "confidence": 0.5}

            if intent.get("type", "").lower() in ["train", "learning", "learn"]:
                agent_config = self._build_learning_config(intent)
                agent = self.factory.create("learning", agent_config)
                reward = agent.train_episode()
                raw_facts = {
                    'agent': 'SLAI Learning Agent',
                    'event': 'training_complete',
                    'metric': 'reward',
                    'value': round(reward, 2)
                }
                return self.grammar.compose_sentence(raw_facts)

            return lang_agent.generate(prompt)

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}", exc_info=True)
            return f"[System Error] Generation failed: {str(e)}"


    def _build_learning_config(self, config: Optional[Dict] = None) -> Dict:
        base_config = {
            "algorithm": "ppo",
            "policy_network": "transformer",
            "buffer_size": 10000,
            "priority_alpha": 0.6,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
            "learning_schedule": "exponential_decay",
            "args": (),
            "kwargs": {}
        }
        print("Received config for learning agent:", config)

        # Only update if config is a proper dictionary
        if isinstance(config, dict):
            base_config.update(config)
        else:
            logging.warning(f"CollaborativeAgent: Ignored invalid config type: {type(config)}")

        return base_config

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

    def _mock_agent_response(self, prompt: str, agent_name: str) -> str:
        """Improved agent routing with learning subsystem support"""
        try:
            agent_info = self.agent_network.get(agent_name, {})
            
            # Handle learning subsystem separately
            if "learning" in agent_name.lower():
                learner = self.learning_subsystems.get(agent_name.lower())
                if not learner:
                    return f"[Error] Learning agent {agent_name} not found"
                
                # Parse learning parameters from prompt
                params = self._parse_learning_parameters(prompt)
                return learner.execute_learning_task(params)
            else:
                # Unified interface for other agents
                agent = self.factory.create(
                    agent_name.split('_')[0].lower(),  # Extract base agent type
                    {"shared_resources": self.shared_resources}
                )
                
                # Execute planning agent workflow with visualization
                if "planning" in agent_name.lower():
                    plan = agent.create_plan(prompt)
                    return f"Generated Plan:\n{plan.to_markdown()}\n\n{agent.validate_plan(plan)}"
                
                return agent.execute(prompt)
        
        except Exception as e:
            logger.error(f"Agent response error: {str(e)}")
            return f"[System Error] Agent communication failed"

    def agent_name(self, leaarner, params, plan, prompt):
        """Specialized handler for learning subsystem"""  # <-- This causes syntax error
        learner = self.learning_subsystems.get(agent_name.lower())
        if not learner:
            return f"[Error] Learning agent {agent_name} not found"
        
        # Parse learning parameters from prompt
        params = self._parse_learning_parameters(prompt)
        return learner.execute_learning_task(params)

        """Execute planning agent workflow with visualization"""
        plan = agent.create_plan(prompt)
        return f"Generated Plan:\n{plan.to_markdown()}\n\n{agent.validate_plan(plan)}"

    def _handle_safety_agent(self, agent, prompt):
        """Full safety assessment pipeline"""
        risk_report = agent.assess_context(prompt)
        return (
            f"Safety Assessment:\n"
            f"Risk Level: {risk_report.risk_level.name}\n"
            f"Confidence: {risk_report.confidence:.2%}\n"
            f"Recommendations: {risk_report.recommendations}"
        )

    def _handle_technical_agent(self, agent, prompt):
        """Technical analysis with visualization support"""
        analysis = agent.analyze(prompt)
        return (
            f"Technical Report:\n"
            f"{analysis.summary}\n\n"
            f"Key Indicators: {', '.join(analysis.indicators)}\n"
            f"Visualization: {agent.generate_chart(analysis)}"
        )

    def _handle_learning_input(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch to specific learning agent based on task input"""
        policy_type = task_input.get("policy", "").lower()

        # Get the base learning agent (already initialized with all backends if needed)
        agent = self._get_agent("learning")

        # Dispatch internally based on policy
        if policy_type == "maml":
            if hasattr(agent, "maml"):
                result = agent.maml.execute(task_input)
            else:
                raise AttributeError("LearningAgent is missing 'maml' backend.")
        else:
            raise ValueError(f"Unsupported policy type: {policy_type}")

        return {
            "learning_result": result,
            "used_policy": policy_type
        }

class TestCollaborativeAgent(unittest.TestCase):
    """Comprehensive unit tests for CollaborativeAgent"""
    
    def setUp(self):
        self.agent = CollaborativeAgent(
            agent_network={
                'TestAgent': {
                    'type': 'testing',
                    'capabilities': ['testing'],
                    'components': ['test_module']
                }
            },
            risk_threshold=0.3
        )

    def test_full_coordination_flow(self):
        tasks = [
            {
                'id': 'task_1',
                'type': 'test_type',
                'requirements': ['testing'],
                'deadline': 5.0,
                'priority': 5,
                'estimated_duration': 1.0,
                'dependencies': [],
                'estimated_risk': 0.2
            }
        ]
        agents = {
            'TestAgent': {
                'capabilities': ['testing'],
                'current_load': 0.1,
                'throughput': 10
            }
        }

        result = self.agent.coordinate_tasks(tasks, agents)
        self.assertEqual(result['status'], 'success')
        self.assertIn('assignments', result)
        self.assertIn('task_1', result['assignments'])
        self.assertEqual(result['assignments']['task_1']['agent'], 'TestAgent')
        
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

class MetricBridge:
    def __init__(self, agent_factory):
        self.factory = agent_factory
        self.metrics_history = []
        
    def submit_metrics(self, metrics: dict):
        """Implements exponential moving average from Brown's smoothing (1956)"""
        self.metrics_history.append(metrics)
        
        # Weighted average (α=0.2)
        decay_factor = 0.2
        avg_metrics = {}
        for key in metrics.keys():
            series = [m.get(key, 0) for m in self.metrics_history[-10:]]
            avg_metrics[key] = sum(
                decay_factor*(1-decay_factor)**i * val 
                for i, val in enumerate(reversed(series))
            )
            
        # Thresholds from ISO/IEC 24029-1:2021 AI quality standards
        if avg_metrics.get('demographic_parity_violations', 0) > 0.1:
            self.factory.adapt_from_metrics(avg_metrics)

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
