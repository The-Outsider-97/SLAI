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
from datetime import datetime

from src.utils.agent_factory import AgentFactory
from src.agents.language.grammar_processor import GrammarProcessor
from src.agents.base_agent import BaseAgent
from src.agents.planning.task_scheduler import DeadlineAwareScheduler
from models.slai_lm_registry import SLAILMManager
from logs.logger import get_logger

logger = get_logger(__name__)

try:
    from scipy.stats import beta
    SCI_AVAILABLE = True
except ImportError:
    SCI_AVAILABLE = False
    logger.warning("SciPy not available - Bayesian features will be limited")

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


class CollaborativeAgent(BaseAgent):
    from src.utils.system_optimizer import SystemOptimizer
    """Main collaborative agent implementation"""
    
    def __init__(self, shared_memory, agent_factory: AgentFactory,
                 config: Optional[Dict] = None,
                 risk_threshold=0.2, agent_network: Optional[Dict] = None,
                 config_path: Optional[str] = None):
        super().__init__("CollaborativeAgent", shared_memory)
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.system_optimizer = self.SystemOptimizer()
        self.risk_threshold = max(0.0, min(1.0, risk_threshold))

        # SIMPLIFIED AGENT ACCESS
        self.agents = {}  # Now populated on-demand via factory
 
        self.grammar = GrammarProcessor(lang='en')
        self.risk_model = {
            'task_risks': defaultdict(list),
            'agent_risks': defaultdict(list),
            'thresholds': defaultdict(BayesianThresholdAdapter)
        }
        self.risk_model['thresholds']['default'] = BayesianThresholdAdapter()
        self.scheduler = DeadlineAwareScheduler()
        self.learning_subsystems = {}

    def _init_performance_metrics(self) -> None:
        """Initialize performance tracking metrics"""
        self.performance_metrics.update({
            'assessments_completed': 0,
            'interventions': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'tasks_coordinated': 0,
            'coordination_failures': 0
        })

    def _validate_risk_score(self, score: float) -> None:
        """Validate risk score is between 0.0 and 1.0"""
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Risk score {score} out of bounds [0.0, 1.0]")

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
        from src.utils.metrics_utils import FairnessMetrics, PerformanceMetrics, MetricBridge
        from src.utils.system_optimizer import SystemOptimizer

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
        self.metric_bridge = MetricBridge(self.factory, self.system_optimizer)
        agent_task_map = defaultdict(list)
        for task_id, assignment in optimized.items():
            agent_task_map[assignment['agent']].append(task_id)
        
        positive_rates = {
            agent: len(agent_task_map[agent]) / len(tasks)
            for agent in self.agent_network
        }
        
        metrics = {
            'demographic_parity_violations': FairnessMetrics.demographic_parity(
                sensitive_groups=list(self.agent_network.keys()),
                positive_rates=positive_rates
            )[0],
            'calibration_error': PerformanceMetrics.calibration_error(
                y_true=np.array([t.get('priority', 0.5) for t in tasks]),
                probs=np.array([a.get('optimization_score', 0.5) 
                                for a in optimized.values()])
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
            "resource_utilization": f"{round(num_tasks / len(self.agent_factory.registry) * 100, 2)}%"
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
        if "language" not in self.agents:
            lang_agent = self.agent_factory.get("language")
            self.agents["language"] = lang_agent  # CACHE FOR REUSE

        prompt = task_data.get("prompt", "").strip()[:1000]
        lang_agent = self.agents["language"]
        lang_agent.dialogue_context.add(prompt)

        intent = lang_agent.parse_intent(prompt) or {"type": "unknown"}
        if not isinstance(intent, dict):
            logger.warning("Intent is not a dict; applying fallback.")
            intent = {"type": str(intent), "confidence": 0.5}

        if intent.get("type", "").lower() in ["train", "learning", "learn"]:
            agent_config = self._build_learning_config(intent)
            agent = self.factory.get("learning", agent_config)
            reward = agent.train_episode()
            raw_facts = {
                'agent': 'SLAI Learning Agent',
                'event': 'training_complete',
                'metric': 'reward',
                'value': round(reward, 2)
            }
            return self.grammar.compose_sentence(raw_facts)

        frame = lang_agent.build_frame(prompt)
        context = lang_agent.dialogue_context            
        return lang_agent.generate(frame, prompt, context)

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
            logger.warning(f"CollaborativeAgent: Ignored invalid config type: {type(config)}")

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
        if self.agent_factory.has_agent(source_agent):
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
        agent_scores = []
        for agent in available_agents:
            if not self.agent_factory.has_agent(agent):
                continue
                
            capabilities = self.agent_factory.get_agent_capabilities(agent)
            capability_match = sum(
                1 for req in task.get('requirements', [])
                if req in capabilities
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
                agent = self.factory.get(
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

    def handle_agent_task(self, agent_type: str, prompt: str) -> str:
        """
        Dispatch agent task by type (e.g., 'learning', 'planning').

        Args:
            agent_type (str): Type of agent ('learning', 'planning', etc.)
            prompt (str): Natural language task description

        Returns:
            str: Response or result
        """
        try:
            agent_key = agent_type.lower()

            if "learning" in agent_key:
                learner = self.learning_subsystems.get(agent_key)
                if not learner:
                    return f"[Error] Learning agent '{agent_key}' not found"
                params = self._parse_learning_parameters(prompt)
                return learner.execute_learning_task(params)

            elif "planning" in agent_key:
                agent = self.agents.get(agent_key)
                if not agent:
                    agent = self.factory.get(agent_key, {"shared_resources": self.shared_resources})
                plan = agent.create_plan(prompt)
                validation = agent.validate_plan(plan)
                return f"Generated Plan:\n{plan.to_markdown()}\n\n{validation}"

            else:
                return f"[Error] Unknown agent type: {agent_type}"

        except Exception as e:
            logger.error(f"Failed to execute agent task: {str(e)}", exc_info=True)
            return f"[System Error] Task execution failed: {str(e)}"

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

    def get_agent(self, agent_name: str) -> Any:
        """Get agent instance with thread-safe pooling"""
        try:
            return self.agent_pool.get(agent_name)
        except KeyError:
            with self.factory._get_agent_context(agent_name) as agent:
                self.agent_pool.put(agent_name, agent)
                return agent

    def _get_agent_safe(self, agent_type: str) -> Any:
        """Thread-safe factory access with error handling"""
        try:
            return self.agent_factory.get_agent(agent_type)
        except Exception as e:
            logger.error(f"Failed to retrieve {agent_type}: {str(e)}")
            return None

class TestCollaborativeAgent(unittest.TestCase):
    """Comprehensive unit tests for CollaborativeAgent"""
    
    def setUp(self):
        shared_memory = self.SharedMemory()
        factory = AgentFactory(config={
            'agent-network': {
                'TestAgent': {
                    'class': 'src.agents.testing_agent.TestAgent',
                    'init_args': {'capabilities': ['testing']}
                }
            }
        })
        self.agent = CollaborativeAgent(
            shared_memory=shared_memory,
            agent_factory=factory,
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
