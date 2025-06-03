import torch.nn as nn
import time
import uuid
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from typing import Callable
from collections import deque

from src.agents.adaptive.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger

logger = get_logger("SLAI Task Router")

FALLBACK_PLANS = {
    "TranslateAndSummarize": ["Translate", "Summarize"],
    "AnalyzeData": ["PreprocessData", "Analyze"],
    "ExplainConcept": ["RetrieveFact", "Summarize"]
}

CONFIG_PATH = "src/agents/alignment/configs/alignment_config.yaml"
MODEL_PATH = "src/agents/collaborative/models/trained_model.pkl"
SCALER_PATH = "src/agents/collaborative/models/scaler.pkl"

def model_predict_func(df: pd.DataFrame) -> np.ndarray:
    """Enhanced prediction function with proper preprocessing"""
    try:
        # 1. Load trained artifacts
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # 2. Preprocess input data
        numerical_data = df.select_dtypes(include=[np.number])
        
        # Handle missing values (example)
        numerical_data.fillna(numerical_data.mean(), inplace=True)
        
        # Apply same scaling as training
        scaled_data = scaler.transform(numerical_data)
        
        # 3. Make predictions (example using sklearn model)
        predictions = model.predict_proba(scaled_data)[:, 1]  # Probability of positive class
        
        # 4. Add fairness-aware postprocessing
        sensitive_mask = df['gender'].isin(['Female'])
        if predictions[sensitive_mask].mean() < 0.5:  # Example fairness correction
            predictions[sensitive_mask] *= 1.15
            
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        # Fallback to simple heuristic
        return np.where(df['age'] > df['age'].median(), 1, 0).astype(np.float32)

class TaskRouter:
    FALLBACK_PLANS = {
        "train_model": ["retry_simple_trainer", "notify_human"],
        "data_audit": ["emergency_data_cleaner"]
    }

    def __init__(self):
        self.config = load_global_config()
        self.router_config = get_config_section('task_routing')

        #self.adaptive_router = AdaptiveRouter()

        logger.info(f"Rask Router succesfully intialized.")

    def route(self, task_type, task_data):
        eligible_agents = self.registry.get_agents_by_task(task_type)
        context = self.shared_memory.get('task_context', {})

        if not eligible_agents:
            raise Exception(f"No agents found for task type '{task_type}'")

        # Step 1: Rank agents by success history or priority
        sorted_agents = self._rank_agents(eligible_agents)

        # Step 2: Try each agent in order until success
        for agent_name, agent, _ in sorted_agents:
            try:
                # Increment active tasks BEFORE execution
                agent_stats = self.shared_memory.get("agent_stats") or {}
                current_tasks = agent_stats.get(agent_name, {}).get("active_tasks", 0)
                agent_stats[agent_name]["active_tasks"] = current_tasks + 1  # Thread-safe via SharedMemory

                self.shared_memory.set("agent_stats", agent_stats)
                logger.info(f"Routing task '{task_type}' to agent: {agent_name}")
                result = agent.execute(task_data)

                # Decrement on SUCCESS
                agent_stats = self.shared_memory.get("agent_stats") or {}
                agent_stats[agent_name]["active_tasks"] = max(0, current_tasks - 1)
                self.shared_memory.set("agent_stats", agent_stats)

                # Step 3: Log success to shared memory
                self._record_success(agent_name)
                return result

            # Error handling
            except Exception as e:
                # Decrement on FAILURE
                agent_stats = self.shared_memory.get("agent_stats") or {}
                agent_stats[agent_name]["active_tasks"] = max(0, current_tasks - 1)
                self.shared_memory.set("agent_stats", agent_stats)
                logger.exception(f"Agent '{agent_name}' failed...")
                self._record_failure(agent_name)
                
                #Fallback logic
                if task_type in self.FALLBACK_PLANS:
                    for subtask in self.FALLBACK_PLANS[task_type]:
                        logger.info(f"Attempting fallback: {subtask}")
                        try:
                            return self.route(subtask, task_data)  # Recursive retry
                        except Exception:
                            raise RuntimeError(f"No agents found for task type '{task_type}'")

        # If all fail
        raise Exception(f"All agents failed for task type '{task_type}'")

    def _rank_agents(self, agents):
        ranked = []
        agent_stats = self.shared_memory.get("agent_stats") or {}
        
        for name, agent in agents.items():
            meta = agent_stats.get(name, {})
            success = meta.get("successes", 0)
            failures = meta.get("failures", 0)
            total = success + failures
            
            # Dynamic weighting
            success_rate = success / total if total > 0 else 1.0  # Favor new agents
            priority = meta.get("priority", 0) * 0.2  # Configurable weight
            load = meta.get("active_tasks", 0) * 0.3  # Penalize busy agents
            
            score = success_rate + priority - load
            ranked.append((name, agent, score))
        
        # Sort by score descending
        return sorted(ranked, key=lambda x: x[2], reverse=True)

    def _record_success(self, agent_name):
        stats = self.shared_memory.get("agent_stats", {})
        entry = stats.get(agent_name, {"successes": 0, "failures": 0, "priority": 0})
        entry["successes"] += 1
        self.shared_memory.put("agent_stats", stats)

    def _record_failure(self, agent_name):
        stats = self.shared_memory.setdefault("agent_stats", {})
        entry = stats.setdefault(agent_name, {"successes": 0, "failures": 0, "priority": 0})
        entry["failures"] += 1
        self.shared_memory.put("agent_stats", stats)

class AdaptiveRouter:
    def __init__(self):
        self.config = load_global_config()
        self.router_config = get_config_section('task_routing')
        num_handlers = self.router_config.get('num_handlers', 2)
        
        self.policy = nn.Sequential(
            nn.Linear(self.router_config.get('state_dim', 64), 64),
            nn.ReLU(),
            nn.Linear(64, num_handlers)
        )
        self.experience_buffer = deque(maxlen=1000)

        self._init_planner()

        logger.info(f"Adaptive Router succesfully intialized with:\n{self.experience_buffer}")

    def _init_planner(self):
        from src.agents.planning.task_scheduler import DeadlineAwareScheduler

        risk_threshold = self.router_config.get('risk_threshold', 0.7)
        retry_policy = self.router_config.get('retry_policy', {'max_attempts': 3, 'backoff_factor': 1.5})
        
        self.task_scheduler = DeadlineAwareScheduler(
            # risk_threshold,
            # retry_policy
        )

    def _init_alignment(self):
        from src.agents.alignment.alignment_monitor import AlignmentMonitor
        from src.agents.alignment.auditors.causal_model import CausalModel, CausalGraphBuilder
        graph_builder = CausalGraphBuilder() 
        raw_data = pd.read_csv("data/users.csv", sep=';')
        data = raw_data.select_dtypes(include=[np.number])
        sensitive_attrs = config.get('sensitive_attributes', ['age', 'gender'])

        causal_model_instance = graph_builder.construct_graph(data, sensitive_attrs)
        self.alignment_monitor = AlignmentMonitor(
            sensitive_attributes=config.get('sensitive_attributes', ['age', 'gender']),
            model_predict_func=model_predict_func,
            causal_model=causal_model_instance,
            config_file_path=config.get('config_file_path', CONFIG_PATH)
        )

    def route_message(self, message, routing_table: dict):
        """Enhanced routing with task scheduling"""
        for condition, handler in routing_table.items():
            if condition in message:
                return handler(message)
        # Convert message to task format
        task = self._message_to_task(message)
        
        # Get available handlers from agent factory
        handlers = self.agent_factory.get_available_agents()
        
        # Schedule using risk-aware scheduler
        schedule = self.task_scheduler.schedule(
            tasks=[task],
            agents=handlers,
            risk_assessor=self.alignment_monitor.assess_risk
        )
        
        if not schedule:
            logger.warning("Routing failed, using fallback strategy")
            return super().route_message(message)

        # Extract best handler from schedule
        handler_id = next(iter(schedule.values()))['agent_id']
        reward = self._calculate_routing_reward(message, handler_id)
        
        # Store experience with priority
        self._store_experience(message, handler_id, reward)
        
        return handler_id

    def _calculate_routing_reward(self, message, handler_id):
        """Calculate reward based on agent performance and task risk."""
        agents = self.agent_factory.get_available_agents()
        agent_details = agents.get(handler_id, {})
        
        # Get agent metrics
        successes = agent_details.get("successes", 1)
        failures = agent_details.get("failures", 0)
        success_rate = successes / (successes + failures + 1e-6)  # Avoid division by zero
        current_load = agent_details.get("current_load", 0)
        load_penalty = current_load * 0.3  # Penalize busy agents
        
        # Get task risk
        task_risk = message.get('risk_score', 0.5)
        
        # Combine factors
        reward = success_rate - load_penalty - (task_risk * 0.2)
        return reward

    def _store_experience(self, message, handler_id, reward):
        """Store routing experience for policy training."""
        state = self._message_to_state(message)
        self.experience_buffer.append({
            'state': state,
            'action': handler_id,
            'reward': reward
        })
        logger.debug(f"Stored experience: {self.experience_buffer[-1]}")

    def _message_to_state(self, message):
        """Convert message to a state vector for the policy network."""
        task = self._message_to_task(message)
        return np.array([
            task['deadline'] - time.time(),  # Time remaining
            task['risk_score'],              # Task risk
            len(task['requirements'])        # Complexity
        ], dtype=np.float32)

    def _message_to_task(self, message):
        """Convert message to scheduler task format"""
        return {
            'id': message.get('message_id', str(uuid.uuid4())),
            'requirements': self._extract_requirements(message),
            'deadline': time.time() + message.get('ttl', 60),
            'risk_score': self.alignment_monitor.assess_risk(message).get('risk_score', 0.5),
            'metadata': message.get('metadata', {})
        }

    def _extract_requirements(self, message):
        """Extract task requirements from message content"""
        content = message.get('content', '').lower()
        requirements = []
        if 'translate' in content:
            requirements.append('Translate')
        if 'summarize' in content:
            requirements.append('Summarize')
        if 'analyze' in content:
            requirements.append('Analyze')
        return requirements


if __name__ == "__main__":
    print("\n=== Running Task Router ===\n")
    router1 = TaskRouter()
    print(router1)
    print("\n=== Successfully ran the Task Router ===\n")

if __name__ == "__main__":
    # Mock configuration
    config = {
        'risk_threshold': 0.7,
        'retry_policy': {'max_attempts': 3, 'backoff_factor': 1.5},
        'state_dim': 3,
        'num_handlers': 2,
        'sensitive_attributes': ['age', 'gender'],
        'config_file_path': CONFIG_PATH
    }
    # Mock agent factory with test agents
    class MockAgentFactory:
        def get_available_agents(self):
            return {
                'translator_agent': {
                    'capabilities': ['Translate'],
                    'current_load': 0.2,
                    'successes': 15,
                    'failures': 1
                },
                'analyzer_agent': {
                    'capabilities': ['Analyze'],
                    'current_load': 0.5,
                    'successes': 10,
                    'failures': 3
                }
            }
    
    # Initialize router with mocked dependencies
    router2 = AdaptiveRouter()
    router2.agent_factory = MockAgentFactory()
    
    # Test messages
    messages = [
        {   # Low-risk translation task
            'message_id': 'test_1',
            'content': 'Translate "Hello World" to French',
            'ttl': 30,
            'metadata': {'priority': 2},
            'risk_score': 0.3
        },
        {   # High-risk analysis task
            'message_id': 'test_2',
            'content': 'Analyze sales data',
            'ttl': 60,
            'metadata': {'priority': 1},
            'risk_score': 0.8
        }
    ]
    
    # Test routing
    for idx, msg in enumerate(messages, 1):
        print(f"\nRouting Message {idx}:")
        try:
            handler = router2.route_message(msg, routing_table={})
            print(f"Assigned Handler: {handler}")
        except Exception as e:
            print(f"Routing failed: {str(e)}")
    
    # Inspect experience buffer
    print("\nExperience Buffer Contents:")
    for exp in router2.experience_buffer:
        print(f"State: {exp['state']}, Action: {exp['action']}, Reward: {exp['reward']:.2f}")
