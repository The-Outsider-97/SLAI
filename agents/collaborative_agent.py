"""
Enhanced Collaborative Agent with Safety Monitoring and Task Coordination

This implementation provides a comprehensive collaborative agent that:
1. Monitors and evaluates safety of other agents' actions
2. Coordinates tasks between multiple agents
3. Maintains shared knowledge and risk assessments
4. Provides adaptive safety recommendations

Academic References:
- Wooldridge (2009) "An Introduction to MultiAgent Systems"
- Leibo et al. (2017) "Multi-agent Reinforcement Learning in Sequential Social Dilemmas"
- Amodei et al. (2016) "Concrete Problems in AI Safety"
"""

import logging
import numpy as np
import yaml
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum

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
    """Enumeration for risk assessment levels"""
    LOW = 0
    MODERATE = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class SafetyAssessment:
    """Data structure for safety assessment results"""
    risk_score: float
    risk_level: RiskLevel
    recommended_action: str
    confidence: float = 1.0
    affected_agents: List[str] = None

class CollaborativeAgent:
    """
    Advanced collaborative agent for multi-agent systems with safety monitoring.
    
    Features:
    - Real-time risk assessment
    - Adaptive safety thresholds
    - Inter-agent coordination
    - Knowledge sharing
    - Performance monitoring
    """
    
    def __init__(self, 
                 shared_memory: Optional[Any] = None, 
                 risk_threshold: float = 0.2,
                 agent_network: Optional[Dict[str, Any]] = None):
        """
        Initialize the collaborative agent.
        
        Args:
            shared_memory: Shared memory object for inter-agent communication
            risk_threshold: Initial risk threshold (0.0-1.0)
            agent_network: Dictionary of known agents and their capabilities
        """
        self.name = "CollaborativeAgent"
        self.shared_memory = shared_memory
        self.risk_threshold = risk_threshold
        self.agent_network = agent_network or {}
        
        # Safety model components
        self.risk_model = {
            'task_risks': defaultdict(list),  # Historical risk data per task type
            'agent_risks': defaultdict(list), # Risk profiles per agent
            'thresholds': defaultdict(float)  # Learned thresholds
        }
        
        # Performance tracking
        self.performance_metrics = {
            'assessments_completed': 0,
            'interventions': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        # Initialize with default threshold
        self.risk_model['thresholds']['default'] = risk_threshold
        
        logger.info(f"Initialized {self.name} with risk threshold: {risk_threshold}")

    def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute collaborative task with safety monitoring.
        
        Args:
            task_data: Dictionary containing:
                - 'policy_risk_score': Computed risk score (0.0-1.0)
                - 'task_type': Category of task being performed
                - 'source_agent': Originating agent (optional)
                - 'action_details': Details of proposed action (optional)
                
        Returns:
            Dictionary containing safety assessment and recommendations
        """
        # Validate input
        if 'policy_risk_score' not in task_data:
            return self._error_response("Missing required field: policy_risk_score")
            
        risk_score = task_data['policy_risk_score']
        task_type = task_data.get('task_type', 'general')
        source_agent = task_data.get('source_agent', 'unknown')
        
        # Perform safety assessment
        assessment = self.assess_risk(
            risk_score=risk_score,
            task_type=task_type,
            source_agent=source_agent,
            action_details=task_data.get('action_details')
        )
        
        # Update shared knowledge
        self._update_shared_knowledge(assessment, task_data)
        
        # Prepare response
        response = {
            'status': 'assessed',
            'assessment': assessment.__dict__,
            'performance_metrics': self.performance_metrics,
            'recommended_actions': self._generate_recommendations(assessment)
        }
        
        logger.info(f"Safety assessment completed for {source_agent}: {response}")
        return response

    def assess_risk(self, 
                   risk_score: float, 
                   task_type: str = "general",
                   source_agent: str = "unknown",
                   action_details: Optional[Dict] = None) -> SafetyAssessment:
        """
        Comprehensive risk assessment with contextual analysis.
        
        Args:
            risk_score: Computed risk score (0.0-1.0)
            task_type: Category of task being performed
            source_agent: Originating agent
            action_details: Details of proposed action
            
        Returns:
            SafetyAssessment object
        """
        # Determine risk level
        threshold = self._get_risk_threshold(task_type, source_agent)
        risk_level = self._calculate_risk_level(risk_score, threshold)
        
        # Generate recommendations
        recommendation = self._generate_recommendation(
            risk_score, 
            risk_level,
            source_agent,
            action_details
        )
        
        # Track performance
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

    def coordinate_tasks(self, 
                        tasks: List[Dict[str, Any]],
                        available_agents: List[str]) -> Dict[str, Any]:
        """
        Coordinate tasks among available agents based on capabilities and risk profiles.
        
        Args:
            tasks: List of tasks to be assigned
            available_agents: List of agent names available for assignment
            
        Returns:
            Dictionary containing task assignments and safety assessments
        """
        assignments = {}
        safety_checks = {}
        
        for task in tasks:
            best_agent = self._select_optimal_agent(task, available_agents)
            if best_agent:
                # Perform pre-assignment safety check
                assessment = self.assess_risk(
                    risk_score=task.get('estimated_risk', 0.5),
                    task_type=task['type'],
                    source_agent=best_agent,
                    action_details=task
                )
                
                assignments[task['id']] = best_agent
                safety_checks[task['id']] = assessment.__dict__
                
        return {
            'assignments': assignments,
            'safety_checks': safety_checks,
            'metadata': {
                'total_tasks': len(tasks),
                'assigned_tasks': len(assignments),
                'high_risk_tasks': sum(
                    1 for a in safety_checks.values() 
                    if a['risk_level'] >= RiskLevel.HIGH.value
                )
            }
        }

    def train_risk_model(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Train the risk assessment model with historical data.
        
        Args:
            training_data: List of dictionaries containing:
                - 'task_type': Task category
                - 'risk_score': Observed risk score
                - 'outcome': Whether the task succeeded (True/False)
                - 'agent': Source agent (optional)
        """
        logger.info(f"Training risk model with {len(training_data)} samples")
        
        # Process training data
        for sample in training_data:
            task_type = sample.get('task_type', 'general')
            agent = sample.get('agent', 'default')
            
            # Store risk data
            self.risk_model['task_risks'][task_type].append(sample['risk_score'])
            self.risk_model['agent_risks'][agent].append(sample['risk_score'])
            
            # Update performance metrics based on outcomes
            if 'outcome' in sample:
                if sample['outcome'] is False and sample['risk_score'] < self._get_risk_threshold(task_type, agent):
                    self.performance_metrics['false_negatives'] += 1
                elif sample['outcome'] is True and sample['risk_score'] > self._get_risk_threshold(task_type, agent):
                    self.performance_metrics['false_positives'] += 1
        
        # Update thresholds based on 90th percentile of historical risks
        for task_type, risks in self.risk_model['task_risks'].items():
            if len(risks) >= 10:  # Only update with sufficient data
                self.risk_model['thresholds'][task_type] = np.percentile(risks, 90)
                
        for agent, risks in self.risk_model['agent_risks'].items():
            if len(risks) >= 10 and agent != 'default':
                self.risk_model['thresholds'][f"agent_{agent}"] = np.percentile(risks, 85)
                
        logger.info("Risk model training completed")

    def _get_risk_threshold(self, task_type: str, agent: str = "default") -> float:
        """
        Get the appropriate risk threshold considering both task and agent factors.
        
        Args:
            task_type: Task category
            agent: Source agent identifier
            
        Returns:
            Appropriate risk threshold (0.0-1.0)
        """
        # Try task-specific threshold first
        threshold = self.risk_model['thresholds'].get(task_type)
        
        # Fall back to agent-specific threshold
        if threshold is None and agent != "default":
            threshold = self.risk_model['thresholds'].get(f"agent_{agent}")
            
        # Final fallback to default threshold
        return threshold or self.risk_model['thresholds']['default']

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

# Example usage
if __name__ == "__main__":
    # Initialize with complete agent network
    agent = CollaborativeAgent(
        agent_network={
            # Safety and Coordination Agents
            'SafeAI_Agent': {
                'type': 'safety',
                'capabilities': ['risk_assessment', 'threshold_adjustment']
            },
            
            # Reinforcement Learning Agents
            'DQN_Agent': {
                'type': 'dqn',
                'capabilities': ['deep_q_learning', 'experience_replay']
            },
            
            # Specialized RL Agents
            'MultiTaskRL_Agent': {
                'type': 'multitask',
                'capabilities': ['task_embedding', 'shared_representation']
            },
            'MAML_Agent': {
                'type': 'meta_learning',
                'capabilities': ['few_shot_adaptation', 'gradient_based_meta']
            },
            
            # Financial/Technical Agents
            'RSI_Agent': {
                'type': 'technical',
                'capabilities': ['market_analysis', 'trend_detection']
            },
            
            # Cognitive Agents
            'NL_Agent': {
                'type': 'nlp',
                'capabilities': ['language_understanding', 'text_generation']
            },
            'Reasoning_Agent': {
                'type': 'knowledge',
                'capabilities': ['logical_inference', 'rule_based_reasoning']
            },
            
            # Planning/Execution Agents
            'Planning_Agent': {
                'type': 'planning',
                'capabilities': ['task_decomposition', 'resource_scheduling']
            },
            'Execution_Agent': {
                'type': 'action',
                'capabilities': ['operation_execution', 'environment_interaction']
            },
            
            # Perception Agents
            'Perception_Agent': {
                'type': 'sensors',
                'capabilities': ['multimodal_fusion', 'feature_extraction']
            },
            
            # Adaptive Agents
            'Adaptive_Agent': {
                'type': 'self_optimizing',
                'capabilities': ['dynamic_parameter_adjustment', 'context_awareness']
            },
            
            # Evaluation Agents
            'Evaluation_Agent': {
                'type': 'metrics',
                'capabilities': ['performance_analysis', 'statistical_testing']
            }
        }
    )
    
    # Example safety assessment
    task_data = {
        'policy_risk_score': 0.45,
        'task_type': 'physical_manipulation',
        'source_agent': 'RL_Agent1',
        'action_details': {
            'type': 'grasp_object',
            'tags': ['physical_action'],
            'affected_components': ['arm1', 'camera2']
        }
    }
    
    result = agent.execute(task_data)
    print("Full Network Safety Assessment Result:\n", result)

    # Example task coordination
    tasks = [
        {'id': 't1', 'type': 'real_time_decision', 'requirements': ['low_latency', 'high_accuracy']},
        {'id': 't2', 'type': 'long_term_planning', 'requirements': ['strategic_thinking', 'resource_optimization']}
    ]
    
    coordination_result = agent.coordinate_tasks(
        tasks=tasks,
        available_agents=['SafeAI_Agent', 'DQN_Agent', 'MultiTaskRL_Agent', 'MAML_Agent', 'RSI_Agent', 'NL_Agent', 'Reasoning_Agent', 'Planning_Agent', 'Execution_Agent', 'Adaptive_Agent', 'Perception_Agent', 'Evaluation_Agent', ]
    )
    print("\nTask Coordination Result:\n", coordination_result)
