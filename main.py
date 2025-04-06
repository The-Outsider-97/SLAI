# ===== START OF main.py =====
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from torch.utils.data import DataLoader, TensorDataset

import os, sys
import yaml
import torch
import queue
import logging
import threading
import subprocess


# Add project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from models.slai_lm import SLAILM

# === Logger Setup ===
from logs.logger import get_logger, get_log_queue
from src.utils.logger import setup_logger

logger = setup_logger("SLAI", level=logging.DEBUG)

try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Failed to load config.yaml: {e}")
    sys.exit(1)

log_queue = get_log_queue()
metric_queue = queue.Queue()

# === UI Imports ===
from frontend.startup_screen import StartupScreen
from frontend.main_window import MainWindow  # <- REAL MainWindow from frontend

def launch_ui():
    app = QApplication(sys.argv)

    def show_main_window():
        app.main_window = MainWindow(log_queue=log_queue, metric_queue=metric_queue)
        app.main_window.show()

    app.splash_screen = StartupScreen(on_ready_to_proceed=show_main_window)
    app.splash_screen.show()

    QTimer.singleShot(2000, app.splash_screen.notify_launcher_ready)

    sys.exit(app.exec_())

# === Entry Point ===
if __name__ == "__main__":
    logger.info("Launching SLAI UI...")
    launch_ui()
# ===== END OF main.py =====


# ===== START OF main_autotune.py =====
import os
import sys
import yaml
import json
import time
import torch
import shutil
import logging
import subprocess
from logs.logs_parser import LogsParser
from src.utils.agent_factory import create_agent
from src.utils.logger import setup_logger
from src.alignment.bias_detection import BiasDetection
from src.alignment.ethical_constraints import EthicalConstraints
from src.alignment.fairness_evaluator import FairnessEvaluator
from src.agents.evaluation_agent import BehavioralValidator, SafetyRewardModel, StaticAnalyzer
from deployment.git.rollback_handler import RollbackHandler
from src.tuning.bayesian_search import BayesianSearch
from src.tuning.grid_search import GridSearch
from src.tuning.tuner import HyperParamTuner
from logging.handlers import RotatingFileHandler

os.makedirs("models/", exist_ok=True)

file_handler = RotatingFileHandler('logs/run.log', maxBytes=10*1024*1024, backupCount=5)

logger = setup_logger("RLAgent", level=logging.INFO)

try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Failed to load config.yaml: {e}")
    sys.exit(1)

logger.info("Launching Recursive Learning Agent...")

# Strategy selection for hyperparam tuning
strategy = config["tuning"]["strategy"]

if strategy == "grid":
    config_file = config["configs"]["grid_config"]
else:
    config_file = config["configs"]["bayesian_config"]

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Create logger
logger = logging.getLogger('AutoTuneOrchestrator')
logger.setLevel(logging.INFO)

# Console handler (stdout)
console_handler = logging.StreamHandler(stream=sys.stdout)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# File handler (run.log)
file_handler = logging.FileHandler('logs/run.log', mode='a', encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add both handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def main():
    logger.info("[AutoTune] Starting agent configuration...")

    config = {
        "learning_rate": 0.01,
        "num_layers": 2,
        "activation_function": "relu"
    }

    agent = create_agent(agent_name="rl_agent", config=config)
    logger.info("[AutoTune] Agent initialized.")

    task_data = {
        "hyperparameters": {
            "lr": 0.005,
            "batch_size": 64,
            "gamma": 0.99
        }
    }

    logger.info("[AutoTune] Executing agent task...")
    result = agent.execute(task_data)
    logger.info(f"[AutoTune] Execution result: {result}")

class AutoTuneOrchestrator:
    """
    Orchestrator for training, evaluation, behavioral testing, reward-based evaluation,
    static code analysis, and automated corrective actions including rollback and retrain.
    """

    def __init__(self):
        # Configs
        self.bias_threshold = 0.1
        self.reward_threshold = 70.0
        self.max_retries = 5

        # Core Handlers
        self.rollback_handler = RollbackHandler(
            models_dir='models/',
            backup_dir='backups/'
        )        
        self.hyperparam_tuner = HyperParamTuner(
            config_path='config.yaml',
            evaluation_function=self.rl_agent_evaluation
        )

        # Components
        self.logs_parser = LogsParser(
            log_dir='logs/',
            bias_threshold=self.bias_threshold,
            reward_threshold=self.reward_threshold,
            rollback_handler=self.rollback_handler,
            hyperparam_tuner=self.hyperparam_tuner
        )

        self.behavioral_tests = BehavioralValidator(
            rollback_handler=self.rollback_handler,
            hyperparam_tuner=self.hyperparam_tuner
        )

        self.reward_function = SafetyRewardModel(
            rollback_handler=self.rollback_handler,
            hyperparam_tuner=self.hyperparam_tuner,
            safety_thresholds={
                'negative_reward_limit': -50,
                'alignment_violation_limit': 3
            }
        )

        self.static_analyzer = StaticAnalyzer(
            codebase_path='src/',
            rollback_handler=self.rollback_handler,
            hyperparam_tuner=self.hyperparam_tuner,
            thresholds={
                'max_warnings': 5,
                'critical_issues': True
            }
        )
        self.bayesian_optimizer = BayesianSearch(
            config_file=config["configs"]["bayesian_config"],
            evaluation_function=self.rl_agent_evaluation,
            n_calls=10,
            n_random_starts=2
        )

        self.grid_optimizer = GridSearch(
            config_file='hyperparam_tuning/example_grid_config.json',
            evaluation_function=self.rl_agent_evaluation
        )

        self._init_behavioral_tests()

    def _init_behavioral_tests(self):
        """
        Define behavioral test cases for the agent.
        """
        def validate_greet(response):
            return response == "Hello!"

        def validate_farewell(response):
            return response == "Goodbye!"

        self.behavioral_tests.add_test_case("greet", "Agent should greet politely", validate_greet)
        self.behavioral_tests.add_test_case("farewell", "Agent should say goodbye", validate_farewell)

    def run_training_pipeline(self):
        """
        Full training pipeline with evaluation, reward monitoring, static code analysis,
        and corrective actions.
        """
        logger.info(" Starting AutoTune Training Pipeline")

        retry_count = 0
        while retry_count < self.max_retries:
            logger.info(f" Training Run {retry_count + 1} / {self.max_retries}")

            # STEP 1: Run Static Code Analysis before training
            logger.info("🛠️ Running Static Code Analysis...")
            self.static_analyzer.run_static_analysis()

            # STEP 2: Train the AI agent
            self.train_agent()

            # STEP 3: Parse Logs & Evaluate Metrics
            report = self.logs_parser.parse_logs()

            # STEP 4: Behavioral Tests
            logger.info(" Running Behavioral Tests...")
            self.behavioral_tests.run_tests(self.simulated_agent_function)

            # STEP 5: Evaluate Reward Function
            logger.info(" Evaluating Reward Function...")
            state = {'user': 'UserABC'}
            action = 'recommend_product'
            outcome = {
                'reward': 10,
                'harm': False,
                'bias_detected': False,
                'discrimination_detected': False
            }
            reward = self.reward_function.compute_reward(state, action, outcome)
            logger.info(f"Reward Function Computed Reward: {reward}")

            # STEP 6: Hyperparameter Optimization
            logger.info("🔎 Triggering Hyperparameter Tuning from CLI...")

            # STEP 7: Run Grid Hyperparameter Optimization (after agent evaluation cycle)
            logger.info("🔎 Running Bayesian Hyperparameter Search...")
            best_params, best_score, _ = self.bayesian_optimizer.run_search()
            logger.info(f"Best hyperparameters from Bayesian optimization: {best_params}") 
            
            # STEP 8: Run Bayesian Hyperparameter Optimization (after agent evaluation cycle)
            logger.info("🔎 Running Grid Hyperparameter Search...")
            best_grid_params = self.grid_optimizer.run_search()
            logger.info(f"Best hyperparameters from GridSearch: {best_grid_params}")
            
            # STEP 9: Decide on Retraining or Rollback
            action_taken = self.decision_policy(report)

            if not action_taken:
                logger.info(" Model passed all checks. Ending pipeline.")
                break

            retry_count += 1

        if retry_count >= self.max_retries:
            logger.warning("⚠️ Max retries reached. Manual intervention recommended.")

    def train_agent(self):
        """
        Placeholder for the actual agent training logic.
        """
        logger.info(" Training AI Agent...")
        time.sleep(2)
        logger.info(" Training complete. Logs and metrics generated.")

    def simulated_agent_function(self, input_data):
        """
        Simulates the agent's response for testing purposes.
        """
        if input_data == "greet":
            return "Hello!"
        elif input_data == "farewell":
            return "Goodbye!"
        else:
            return "I don't understand."

    def rl_agent_evaluation(self, params):
        """
        Dummy evaluation function that simulates RL agent performance
        (used by Bayesian optimizer).
        """
        learning_rate = params['learning_rate']
        num_layers = params['num_layers']
        activation_function = params['activation']

        logger.info(f"Evaluating RL agent with: lr={learning_rate}, layers={num_layers}, activation={activation_function}")

        # Dummy score based on distance from ideal params
        score = -((learning_rate - 0.01) ** 2 + (num_layers - 3) ** 2)
        return score

    def decision_policy(self, report=None):
        """
        Evaluate parsed logs and decide whether to retrain or rollback.
        """
        logger.info("🔎 Evaluating performance and alignment metrics...")

        try:
            with open('logs/parsed_metrics.json', 'r') as f:
                report = json.load(f)
        except Exception as e:
            logger.error(f" Failed to load parsed metrics report: {e}")
            return False

        parity_diff = abs(report.get('statistical_parity', {}).get('parity_difference', 0.0))
        tpr_diff = abs(report.get('equal_opportunity', {}).get('tpr_difference', 0.0))
        reward_score = report.get('performance', {}).get('best_reward', 0.0)

        corrective_action = False

        if parity_diff > self.bias_threshold or tpr_diff > self.bias_threshold:
            logger.warning(f"⚠️ Bias thresholds breached: ParityDiff={parity_diff}, TPRDiff={tpr_diff}")
            self.rollback_handler.rollback_model()
            corrective_action = True

        if reward_score < self.reward_threshold:
            logger.warning(f"⚠️ Reward threshold breached: {reward_score}")
            self.hyperparam_tuner.run_tuning_pipeline()
            corrective_action = True

        return corrective_action

if __name__ == "__main__":
    orchestrator = AutoTuneOrchestrator()

    tuner = HyperParamTuner(
        config_path=config_file,
        evaluation_function=orchestrator.rl_agent_evaluation,
        strategy=strategy,
        n_calls=config["tuning"]["n_calls"],
        n_random_starts=config["tuning"]["n_random_starts"]
    )

    orchestrator.run_training_pipeline()

# ===== END OF main_autotune.py =====


# ===== START OF main_collaborative.py =====
import os
import sys
import logging
import time
import yaml
import json
import pickle
import queue
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple
from multiprocessing import Pool, cpu_count
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

# Vector DB imports (using FAISS for performance)
try:
    import faiss
except ImportError:
    logging.error("FAISS not installed. Please install with: pip install faiss-cpu")
    sys.exit(1)

# Set up logging
logger = logging.getLogger("SLAI")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

@dataclass
class Task:
    id: str
    type: str
    data: Dict[str, Any]
    priority: float = 0.5
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    def __reduce__(self):
        """Ensure proper pickling of Task objects"""
        return (self.__class__, (
            self.id,
            self.type,
            self.data,
            self.priority,
            self.status,
            self.created_at,
            self.dependencies,
            self.assigned_agent
        ))

class VectorDB:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []
        self.id_map = {}
        self.next_id = 0
    
    def add(self, vector: np.ndarray, metadata: Dict[str, Any]):
        if len(vector) != self.dim:
            raise ValueError(f"Vector dimension {len(vector)} != {self.dim}")
        
        vector = vector.astype('float32').reshape(1, -1)
        self.index.add(vector)
        self.metadata.append(metadata)
        self.id_map[self.next_id] = len(self.metadata) - 1
        self.next_id += 1
    
    def search(self, query_vector: np.ndarray, k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
        query_vector = query_vector.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i in range(k):
            if indices[0][i] >= 0:  # FAISS returns -1 for invalid indices
                metadata = self.metadata[self.id_map[indices[0][i]]]
                results.append((float(distances[0][i]), metadata))
        
        return results
    
    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.meta", "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "id_map": self.id_map,
                "next_id": self.next_id
            }, f)
    
    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.meta", "rb") as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            self.id_map = data["id_map"]
            self.next_id = data["next_id"]

class SharedMemory:
    def __init__(self, persist_path: str = "slai_memory"):
        self.memory = {}
        self.vector_db = VectorDB(dim=128)
        self.task_history = []
        self.persist_path = persist_path
        self.load()
    
    def set(self, key: str, value: Any):
        self.memory[key] = value
    
    def get(self, key: str, default=None):
        return self.memory.get(key, default)
    
    def keys(self):
        return self.memory.keys()
    
    def store_embedding(self, vector: np.ndarray, metadata: Dict[str, Any]):
        self.vector_db.add(vector, metadata)
    
    def retrieve_similar(self, query_vector: np.ndarray, k: int = 3):
        return self.vector_db.search(query_vector, k)
    
    def save(self):
        with open(f"{self.persist_path}.state", "wb") as f:
            pickle.dump({
                "memory": self.memory,
                "task_history": self.task_history
            }, f)
        self.vector_db.save(self.persist_path)
    
    def load(self):
        try:
            with open(f"{self.persist_path}.state", "rb") as f:
                data = pickle.load(f)
                self.memory = data["memory"]
                self.task_history = data["task_history"]
            self.vector_db.load(self.persist_path)
            logger.info("Loaded previous state from disk")
        except (FileNotFoundError, EOFError):
            logger.info("No previous state found, starting fresh")

class RLTaskRouter:
    def __init__(self, shared_memory: SharedMemory):
        self.shared_memory = shared_memory
        self.task_queue = []
        self.agent_q_values = defaultdict(dict)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
    
    def add_task(self, task: Task):
        self.task_queue.append(task)
        self._reprioritize()
    
    def _reprioritize(self):
        self.task_queue.sort(key=lambda x: (-x.priority, x.created_at))
    
    def pop_next_task(self) -> Optional[Task]:
        return self.task_queue.pop(0) if self.task_queue else None
    
    def select_agent(self, task_type: str, available_agents: List[str]) -> Optional[str]:
        if np.random.random() < self.exploration_rate:
            return np.random.choice(available_agents) if available_agents else None
        
        best_agent, best_q = None, -float('inf')
        for agent in available_agents:
            q = self.agent_q_values[agent].get(task_type, 1.0)
            if q > best_q:
                best_q = q
                best_agent = agent
        return best_agent
    
    def update_q_values(self, agent: str, task_type: str, reward: float, next_state_value: float):
        current_q = self.agent_q_values[agent].get(task_type, 1.0)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_state_value - current_q)
        self.agent_q_values[agent][task_type] = new_q
    
    def create_child_tasks(self, parent_task: Task, results: Dict[str, Any]) -> List[Task]:
        new_tasks = []
        if "requires_optimization" in results:
            new_tasks.append(Task(
                id=f"optimize_{parent_task.id}",
                type="optimize",
                data={"parameters": results["requires_optimization"]},
                dependencies=[parent_task.id],
                priority=parent_task.priority * 0.9
            ))
        if "safety_concerns" in results:
            new_tasks.append(Task(
                id=f"safety_check_{parent_task.id}",
                type="safety",
                data={"risk_factors": results["safety_concerns"]},
                dependencies=[parent_task.id],
                priority=1.0
            ))
        for task in new_tasks:
            self.add_task(task)
        return new_tasks

class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(task_history: List[Task], agent_performance: Dict[str, Any]) -> Dict[str, Any]:
        metrics = {
            "throughput": defaultdict(int),
            "success_rate": defaultdict(float),
            "task_times": defaultdict(list),
            "agent_utilization": defaultdict(float),
            "priority_efficiency": defaultdict(float)
        }
        
        agent_task_counts = defaultdict(int)
        agent_success_counts = defaultdict(int)
        total_time = 0
        
        for task in task_history:
            agent_task_counts[task.assigned_agent] += 1
            metrics["throughput"][task.type] += 1
            metrics["task_times"][task.type].append(time.time() - task.created_at)
            
            if task.status == "completed":
                agent_success_counts[task.assigned_agent] += 1
                metrics["success_rate"][task.type] += 1
                total_time += time.time() - task.created_at
        
        for task_type in metrics["throughput"]:
            if metrics["throughput"][task_type] > 0:
                metrics["success_rate"][task_type] /= metrics["throughput"][task.type]
                metrics["priority_efficiency"][task.type] = (
                    metrics["success_rate"][task.type] * np.mean(metrics["task_times"][task.type]))
        
        for agent in agent_performance:
            metrics["agent_utilization"][agent] = agent_task_counts.get(agent, 0) / max(1, len(task_history))
        
        metrics["average_task_time"] = total_time / max(1, len(task_history))
        return metrics
    
    @staticmethod
    def visualize_metrics(metrics: Dict[str, Any], save_path: str = None):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        task_types = list(metrics["throughput"].keys())
        plt.bar(task_types, [metrics["throughput"][t] for t in task_types])
        plt.title("Task Throughput by Type")
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        plt.bar(task_types, [metrics["success_rate"][t] for t in task_types])
        plt.title("Success Rate by Task Type")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        agents = list(metrics["agent_utilization"].keys())
        plt.bar(agents, [metrics["agent_utilization"][a] for a in agents])
        plt.title("Agent Utilization")
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        plt.bar(task_types, [metrics["priority_efficiency"][t] for t in task_types])
        plt.title("Priority Efficiency")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

class CollaborationManager:
    def __init__(self, shared_memory: Optional[SharedMemory] = None, num_workers: int = None):
        self.shared_memory = shared_memory or SharedMemory()
        self.registry = AgentRegistry(self.shared_memory)
        self.router = RLTaskRouter(self.shared_memory)
        self.num_workers = min(num_workers or cpu_count() - 1, 8)
        self.metrics = PerformanceMetrics()
    
    def register_agent(self, name: str, agent, capabilities: List[str]):
        self.registry.register(name, agent, capabilities)
        logger.info(f"Registered agent {name} with capabilities: {capabilities}")
    
    def _execute_task(self, task: Task, agent_name: str) -> Tuple[Task, Dict[str, Any]]:
        try:
            agent_data = self.registry.get_agent(agent_name)
            if not agent_data:
                raise ValueError(f"Agent {agent_name} not found")
                
            agent = agent_data["instance"]
            result = agent.execute(task.type, task.data)
            task.status = "completed"
            return task, result
        except Exception as e:
            logger.error(f"Task {task.id} failed: {str(e)}")
            task.status = "failed"
            return task, {"error": str(e)}
    
    def execute_task_cycle(self):
        tasks_to_execute = []
        available_agents = [name for name in self.registry.list_agents() 
                          if self.registry.get_agent_status(name) == "idle"]
        
        while available_agents and (task := self.router.pop_next_task()):
            agent_name = self.router.select_agent(task.type, available_agents)
            if agent_name:
                self.registry.update_status(agent_name, "busy")
                task.assigned_agent = agent_name
                tasks_to_execute.append((task, agent_name))
                available_agents.remove(agent_name)
        
        if tasks_to_execute:
            with Pool(processes=self.num_workers) as pool:
                results = pool.starmap(self._execute_task, tasks_to_execute)
                for task, result in results:
                    self._process_task_result(task, result)
            return True
        return False
    
    def _process_task_result(self, task: Task, result: Dict[str, Any]):
        self.shared_memory.task_history.append(task)
        
        if task.status == "completed":
            embedding = np.random.rand(128)  # Replace with actual embedding
            self.shared_memory.store_embedding(
                vector=embedding,
                metadata={
                    "task": task.type,
                    "agent": task.assigned_agent,
                    "timestamp": time.time(),
                    "result": result
                }
            )
            
            performance = (0.4 * result.get("performance", 0.5) + 
                         0.3 * 1.0 +  # Success
                         0.2 * result.get("safety_score", 1.0) + 
                         0.1 * (1 - (time.time() - task.created_at)/100))
            
            self.router.update_q_values(
                task.assigned_agent,
                task.type,
                performance,
                performance * 0.9
            )
            
            self.router.create_child_tasks(task, result)
        
        self.registry.update_status(task.assigned_agent, "idle")
    
    def save_state(self, path: str = "slai_state"):
        self.shared_memory.save()
        with open(f"{path}.router", "wb") as f:
            pickle.dump(self.router.agent_q_values, f)
        self.registry.save(f"{path}.registry")
        logger.info(f"System state saved to {path}")
    
    def load_state(self, path: str = "slai_state"):
        try:
            with open(f"{path}.router", "rb") as f:
                self.router.agent_q_values = pickle.load(f)
            self.registry.load(f"{path}.registry")
            logger.info("Loaded system state from disk")
        except FileNotFoundError:
            logger.warning("No previous system state found")

class AgentRegistry:
    def __init__(self, shared_memory: SharedMemory):
        self.agents = {}
        self.shared_memory = shared_memory
    
    def register(self, name: str, agent, capabilities: List[str]):
        self.agents[name] = {
            "instance": agent,
            "capabilities": capabilities,
            "status": "idle",
            "performance": 0.0,
            "task_count": 0
        }
    
    def get_agent(self, name: str) -> Optional[Dict[str, Any]]:
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        return list(self.agents.keys())
    
    def get_agent_status(self, name: str) -> str:
        return self.agents.get(name, {}).get("status", "unknown")
    
    def update_status(self, name: str, status: str):
        if name in self.agents:
            self.agents[name]["status"] = status
    
    def update_performance(self, name: str, score: float):
        if name in self.agents:
            self.agents[name]["performance"] = score
            self.agents[name]["task_count"] += 1
    
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.agents, f)
    
    def load(self, path: str):
        try:
            with open(path, "rb") as f:
                self.agents = pickle.load(f)
        except FileNotFoundError:
            logger.warning("No previous registry state found")

class SafeAIAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_threshold = config.get("risk_threshold", 0.3)
    
    def execute(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if task_type == "safety":
            risk_score = data.get("policy_risk_score", 0)
            if risk_score > self.risk_threshold:
                return {
                    "status": "unsafe",
                    "risk_score": risk_score,
                    "actions": ["halt_execution", "notify_operator"],
                    "safety_score": 0.1,
                    "performance": 0.8
                }
            else:
                return {
                    "status": "safe",
                    "risk_score": risk_score,
                    "safety_score": 0.9,
                    "performance": 1.0
                }
        return {"status": "unhandled_task_type", "performance": 0.0}

class MultiTaskAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tasks = config.get("tasks", [])
        self.current_task = 0
    
    def execute(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if task_type == "multi_task_learning":
            performance = 0.7 + np.random.normal(0, 0.1)
            return {
                "performance": performance,
                "task_complexity": 0.5,
                "current_task": self.tasks[self.current_task],
                "safety_score": 0.8
            }
        return {"status": "unhandled_task_type", "performance": 0.0}

def initialize_system():
    shared_memory = SharedMemory()
    collab_mgr = CollaborationManager(shared_memory, num_workers=4)
    
    collab_mgr.register_agent(
        "safe_ai",
        SafeAIAgent({"risk_threshold": 0.25}),
        ["safety", "risk_management"]
    )
    
    collab_mgr.register_agent(
        "multitask",
        MultiTaskAgent({"tasks": ["CartPole", "MountainCar"]}),
        ["multi_task_learning", "adaptation"]
    )
    
    initial_tasks = [
        Task(id="initial_safety", type="safety", data={"policy_risk_score": 0.2}),
        Task(id="initial_mtl", type="multi_task_learning", data={"domains": ["CartPole"]}),
        Task(id="optimize_1", type="optimize", data={"parameters": {"lr": 0.01}}),
        Task(id="meta_1", type="meta_learning", data={"tasks": ["few_shot"]})
    ]
    
    for task in initial_tasks:
        collab_mgr.router.add_task(task)
    
    return collab_mgr

def main():
    logger.info("Starting SLAI v2.0 - Enhanced Collaborative Agent System")
    collab_mgr = initialize_system()
    
    try:
        for cycle in range(20):
            logger.info(f"\n=== Execution Cycle {cycle + 1} ===")
            collab_mgr.execute_task_cycle()
            
            if cycle % 5 == 0:
                collab_mgr.save_state()
                metrics = PerformanceMetrics.calculate_metrics(
                    collab_mgr.shared_memory.task_history,
                    collab_mgr.registry.agents
                )
                PerformanceMetrics.visualize_metrics(metrics, f"metrics_cycle_{cycle}.png")
        
        collab_mgr.save_state()
        metrics = PerformanceMetrics.calculate_metrics(
            collab_mgr.shared_memory.task_history,
            collab_mgr.registry.agents
        )
        PerformanceMetrics.visualize_metrics(metrics, "final_metrics.png")
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        collab_mgr.save_state()
    
    logger.info("System execution completed")

if __name__ == "__main__":
    main()

# ===== END OF main_collaborative.py =====


# ===== START OF main_knowledge.py =====
import os
import json
import math
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import pickle

class KnowledgeAgent:
    def __init__(self, knowledge_base_dir: str, persist_file: str = None):
        """
        Initialize the Knowledge Agent with enhanced capabilities.
        
        Args:
            knowledge_base_dir: Path to directory containing knowledge documents
            persist_file: Optional file path to save/load knowledge state
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.persist_file = persist_file
        self.documents: Dict[str, str] = {}
        self.vocabulary: Dict[str, int] = {}
        self.inverted_index: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.document_norms: Dict[str, float] = {}
        self.document_count = 0
        
        if persist_file and os.path.exists(persist_file):
            self._load_state()
        else:
            self._load_documents()
            self._build_index()

    def _load_documents(self):
        """Load documents from various file formats."""
        for filename in os.listdir(self.knowledge_base_dir):
            filepath = os.path.join(self.knowledge_base_dir, filename)
            try:
                if filename.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as file:
                        self.documents[filename] = file.read()
                elif filename.endswith('.json'):
                    with open(filepath, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        if isinstance(data, dict):
                            self.documents[filename] = json.dumps(data)
                        else:
                            self.documents[filename] = str(data)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        
        self.document_count = len(self.documents)

    def _tokenize(self, text: str) -> List[str]:
        """Basic tokenizer with stemming and stop word removal."""
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = text.split()
        
        # Simple stemming (Porter stemmer would be better)
        stemmed = []
        for token in tokens:
            if len(token) > 4:
                if token.endswith('ing'):
                    token = token[:-3]
                elif token.endswith('ly'):
                    token = token[:-2]
            stemmed.append(token)
        
        return stemmed

    def _compute_tfidf(self, term: str, document: str) -> float:
        """Compute TF-IDF score for a term in a document."""
        tf = document.count(term) / len(document.split())
        idf = math.log(self.document_count / (1 + sum(1 for d in self.documents.values() if term in d)))
        return tf * idf

    def _build_index(self):
        """Build inverted index with TF-IDF weights."""
        for doc_name, content in self.documents.items():
            tokens = self._tokenize(content)
            unique_terms = set(tokens)
            
            # Update vocabulary
            for term in unique_terms:
                if term not in self.vocabulary:
                    self.vocabulary[term] = len(self.vocabulary)
                
                # Compute TF-IDF
                self.inverted_index[term][doc_name] = self._compute_tfidf(term, content)
            
            # Precompute document norm for cosine similarity
            self.document_norms[doc_name] = math.sqrt(sum(
                weight**2 for weight in self.inverted_index[term][doc_name].values()
            ))

    def _cosine_similarity(self, query_weights: Dict[str, float], doc_name: str) -> float:
        """Compute cosine similarity between query and document."""
        dot_product = 0.0
        query_norm = math.sqrt(sum(w**2 for w in query_weights.values()))
        
        for term, q_weight in query_weights.items():
            if term in self.inverted_index and doc_name in self.inverted_index[term]:
                dot_product += q_weight * self.inverted_index[term][doc_name]
        
        doc_norm = self.document_norms.get(doc_name, 1e-10)
        return dot_product / (query_norm * doc_norm)

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve relevant documents using vector space model."""
        query_terms = self._tokenize(query)
        query_weights = {}
        
        # Compute query vector weights
        for term in set(query_terms):
            tf = query_terms.count(term) / len(query_terms)
            idf = math.log(self.document_count / (1 + sum(1 for d in self.documents.values() if term in d)))
            query_weights[term] = tf * idf
        
        # Score documents
        scores = []
        for doc_name in self.documents:
            score = self._cosine_similarity(query_weights, doc_name)
            scores.append((doc_name, score))
        
        # Return top K results
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    def augment_generation(self, query: str, max_length: int = 2000) -> str:
        """Generate response with context from relevant documents."""
        relevant_docs = self.retrieve_documents(query)
        response = ["Based on retrieved information:"]
        
        for doc_name, score in relevant_docs:
            content = self.documents[doc_name]
            snippet = content[:500] + "..." if len(content) > 500 else content
            response.append(f"\nFrom {doc_name} (relevance: {score:.2f}):\n{snippet}")
        
        # Ensure response doesn't exceed max length
        full_response = "\n".join(response)
        return full_response[:max_length] + "..." if len(full_response) > max_length else full_response

    def update_knowledge_base(self, new_document_path: str):
        """Add new document incrementally to the knowledge base."""
        filename = os.path.basename(new_document_path)
        with open(new_document_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        self.documents[filename] = content
        self.document_count += 1
        
        # Incremental update of inverted index
        tokens = self._tokenize(content)
        unique_terms = set(tokens)
        
        for term in unique_terms:
            if term not in self.vocabulary:
                self.vocabulary[term] = len(self.vocabulary)
            self.inverted_index[term][filename] = self._compute_tfidf(term, content)
        
        # Update document norm
        self.document_norms[filename] = math.sqrt(sum(
            weight**2 for weight in self.inverted_index[term][filename].values()
        ))

    def save_state(self):
        """Persist the current state to disk."""
        if self.persist_file:
            with open(self.persist_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'vocabulary': self.vocabulary,
                    'inverted_index': self.inverted_index,
                    'document_norms': self.document_norms,
                    'document_count': self.document_count
                }, f)

    def _load_state(self):
        """Load persisted state from disk."""
        with open(self.persist_file, 'rb') as f:
            state = pickle.load(f)
            self.documents = state['documents']
            self.vocabulary = state['vocabulary']
            self.inverted_index = state['inverted_index']
            self.document_norms = state['document_norms']
            self.document_count = state['document_count']

# ===== END OF main_knowledge.py =====


# ===== START OF main_learning.py =====
"""
Continual Meta-Learning System with Self-Improving Capabilities

Key Academic References:
1. Meta-Learning: Finn et al. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. PMLR.
2. Deep Q-Learning: Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature.
3. RSI Strategies: Sutton & Barto (2018). Reinforcement Learning: An Introduction. MIT Press.
4. Continual Learning: Parisi et al. (2019). Continual Lifelong Learning with Neural Networks. IEEE TCDS.
"""

import numpy as np
import random
from collections import deque
from pathlib import Path

# Local imports following project structure
from src.utils.agent_factory import create_agent, validate_config
from src.collaborative.shared_memory import SharedMemory

class ContinualLearner:
    """
    Core system integrating multiple learning strategies with self-improvement
    
    Implements the Self-Improving AI Architecture from:
    Schmidhuber (2013). PowerPlay: Training an Increasingly General Problem Solver
    """
    
    def __init__(self, env, config_path="config.yaml"):
        self.env = env
        self.shared_memory = SharedMemory()
        self.feedback_buffer = deque(maxlen=1000)
        self.demo_buffer = deque(maxlen=500)
        
        # Initialize agents through factory
        self.agents = {
            "dqn": self._init_dqn(),
            "maml": self._init_maml(),
            "rsi": self._init_rsi()
        }
        
        # Continual learning parameters
        self.meta_update_interval = 100
        self.strategy_weights = np.ones(3)  # [DQN, MAML, RSI]
        self.performance_history = []
        
        # Academic-inspired parameters
        self.curiosity_beta = 0.2  # From Pathak et al. (2017) Curiosity-driven Exploration
        self.ewc_lambda = 0.4  # Elastic Weight Consolidation (Kirkpatrick et al., 2017)
        
    def _init_dqn(self):
        """Initialize DQN agent with evolutionary capabilities"""
        config = {
            "state_size": self.env.observation_space.shape[0],
            "action_size": self.env.action_space.n,
            "hidden_size": 128,
            "gamma": 0.99,
            "epsilon_decay": 0.995
        }
        validate_config("dqn", config)
        return create_agent("dqn", config)
    
    def _init_maml(self):
        """Initialize MAML agent for fast adaptation"""
        config = {
            "state_size": self.env.observation_space.shape[0],
            "action_size": self.env.action_space.n,
            "hidden_size": 64,
            "meta_lr": 0.001,
            "inner_lr": 0.01
        }
        validate_config("maml", config)
        return create_agent("maml", config)
    
    def _init_rsi(self):
        """Initialize RSI agent with shared memory"""
        return create_agent("rsi", {
            "state_size": self.env.observation_space.shape[0],
            "action_size": self.env.action_space.n,
            "shared_memory": self.shared_memory
        })
    
    def run_episode(self, agent_type="dqn", train=True):
        """
        Execute one environment episode with selected agent
        
        Implements Hybrid Reward Architecture from:
        van Seijen et al. (2017). Hybrid Reward Architecture for Reinforcement Learning
        """
        state = self.env.reset()
        total_reward = 0
        episode_data = []
        
        while True:
            # Select action using current strategy
            action = self._select_action(state, agent_type)
            
            # Environment step
            next_state, reward, done, _ = self.env.step(action)
            
            # Store experience
            episode_data.append((state, action, reward, next_state, done))
            
            # Process immediate feedback
            self._process_feedback(state, action, reward)
            
            state = next_state
            total_reward += reward
            
            if done: break
        
        if train:
            self._train_on_episode(episode_data, agent_type)
        
        return total_reward
    
    def _select_action(self, state, agent_type):
        """Action selection with exploration strategy"""
        # Epsilon-greedy with decaying exploration
        if random.random() < self._current_epsilon():
            return self.env.action_space.sample()
            
        return self.agents[agent_type].act(state)
    
    def _current_epsilon(self):
        """Decaying exploration rate with minimum floor"""
        return max(0.01, 0.1 * (0.98 ** len(self.performance_history)))
    
    def _train_on_episode(self, episode_data, agent_type):
        """
        Train on episode data using selected agent
        
        Implements Experience Replay with:
        Lin (1992). Self-Improving Reactive Agents Based On Reinforcement Learning
        """
        agent = self.agents[agent_type]
        
        # Convert to agent-specific training format
        if agent_type == "dqn":
            for transition in episode_data:
                agent.store_transition(*transition)
            loss = agent.train()
            
        elif agent_type == "maml":
            loss = agent.meta_update([(self.env, None)])
            
        elif agent_type == "rsi":
            agent.remember(episode_data)
            loss = agent.train()
        
        # Update strategy weights based on performance
        self._update_strategy_weights(loss)
        
        # Consolidate knowledge (EWC)
        self._elastic_weight_consolidation()
    
    def _update_strategy_weights(self, recent_loss):
        """
        Dynamic strategy weighting using:
        Yin et al. (2020). Learn to Combine Strategies in Reinforcement Learning
        """
        # Normalized inverse loss weighting
        losses = np.array([recent_loss, 0.1, 0.1])  # Placeholder
        self.strategy_weights = 1 / (losses + 1e-8)
        self.strategy_weights /= self.strategy_weights.sum()
    
    def _elastic_weight_consolidation(self):
        """Mitigate catastrophic forgetting using EWC"""
        # Implementation adapted from:
        # Kirkpatrick et al. (2017). Overcoming catastrophic forgetting in neural networks
        for agent in self.agents.values():
            if hasattr(agent, 'consolidate_weights'):
                agent.consolidate_weights(self.ewc_lambda)
    
    def process_demonstration(self, demo_data):
        """
        Learn from human/expert demonstrations
        
        Implements Dagger algorithm:
        Ross et al. (2011). A Reduction of Imitation Learning to Structured Prediction
        """
        self.demo_buffer.extend(demo_data)
        
        # Train all agents on demonstration data
        for agent in self.agents.values():
            if hasattr(agent, 'learn_from_demo'):
                agent.learn_from_demo(self.demo_buffer)
    
    def _process_feedback(self, state, action, reward):
        """
        Process real-time feedback using:
        Knox & Stone (2009). Interactively shaping agents via human reinforcement
        """
        self.feedback_buffer.append((state, action, reward))
        
        # Update agents with recent feedback
        for agent in self.agents.values():
            if hasattr(agent, 'incorporate_feedback'):
                agent.incorporate_feedback(self.feedback_buffer)
    
    def meta_learn(self, num_tasks=10):
        """
        Meta-learning phase using:
        Wang et al. (2020). Automating Reinforcement Learning with Meta-Learning
        """
        print("Starting meta-learning phase...")
        
        for task in range(num_tasks):
            # Generate new task variation
            task_env = self._modify_environment()
            
            # Fast adaptation
            adapted_agent = self.agents["maml"].adapt(task_env)
            
            # Evaluate and update strategy
            reward = self._evaluate_adapted(adapted_agent, task_env)
            self._update_meta_knowledge(reward)
    
    def _modify_environment(self):
        """Create new task variation for meta-learning"""
        # Placeholder - implement environment parameter randomization
        return self.env
    
    def _evaluate_adapted(self, agent, env):
        """Evaluate adapted agent on new task"""
        total_reward = 0
        state = env.reset()
        
        for _ in range(1000):
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done: break
            
        return total_reward
    
    def _update_meta_knowledge(self, reward):
        """Update meta-parameters based on adaptation success"""
        # Update strategy weights
        self.strategy_weights[1] *= (1 + reward/1000)  # MAML index
        
        # Update curiosity parameter
        self.curiosity_beta = max(0.1, self.curiosity_beta * (1 + reward/500))
    
    def continual_learning_loop(self, num_episodes=1000):
        """
        Main continual learning loop implementing:
        Ring (1997). CHILD: A First Step Towards Continual Learning
        """
        for episode in range(num_episodes):
            # Select strategy using multi-armed bandit
            agent_type = self._select_strategy()
            
            # Run episode with selected strategy
            reward = self.run_episode(agent_type)
            self.performance_history.append(reward)
            
            # Meta-learning updates
            if episode % self.meta_update_interval == 0:
                self.meta_learn(num_tasks=3)
                
            # Demonstration learning
            if episode % 50 == 0 and len(self.demo_buffer) > 0:
                self.process_demonstration(random.sample(self.demo_buffer, 10))
            
            # System self-evaluation
            if episode % 100 == 0:
                self._system_self_diagnostic()
    
    def _select_strategy(self):
        """Strategy selection using Thompson sampling"""
        # Implement multi-armed bandit strategy
        sampled_weights = np.random.normal(self.strategy_weights, 0.1)
        return ["dqn", "maml", "rsi"][np.argmax(sampled_weights)]
    
    def _system_self_diagnostic(self):
        """Comprehensive system health check"""
        # Check agent performance
        recent_perf = np.mean(self.performance_history[-100:])
        print(f"Recent average performance: {recent_perf:.2f}")
        
        # Memory diagnostics
        print(f"Feedback buffer: {len(self.feedback_buffer)} samples")
        print(f"Demo buffer: {len(self.demo_buffer)} samples")
        
        # Strategy distribution
        print("Current strategy weights:", self.strategy_weights)

class SimpleEnv:
    """Simplified environment for demonstration"""
    def __init__(self):
        self.observation_space = self.ObservationSpace(4)
        self.action_space = self.ActionSpace(2)
    
    def reset(self):
        return np.random.randn(4)
    
    def step(self, action):
        return np.random.randn(4), random.random(), random.random() < 0.2, {}
    
    class ObservationSpace:
        def __init__(self, dim):
            self.shape = (dim,)
    
    class ActionSpace:
        def __init__(self, n):
            self.n = n
        
        def sample(self):
            return random.randint(0, self.n-1)

if __name__ == "__main__":
    # Initialize components
    env = SimpleEnv()
    learner = ContinualLearner(env)
    
    # Run continual learning
    print("Starting continual learning process...")
    learner.continual_learning_loop(num_episodes=1000)
    
    # Final evaluation
    print("\nFinal system evaluation:")
    learner._system_self_diagnostic()

# ===== END OF main_learning.py =====


# ===== START OF main_reasoning.py =====
"""
main_reasoning.py - Entry point for the Cognitive Hybrid Reasoning Intelligent Agent System (CHRIS)
"""

import sys
import time
from src.utils.agent_factory import AgentFactory
from config.settings import AGENT_CONFIG, ENVIRONMENT_CONFIG

class ReasoningSystem:
    def __init__(self):
        """Initialize the reasoning system with agent factory and environment"""
        self.agent_factory = AgentFactory()
        self.agents = {}
        self.environment = None  # Would be initialized with environment interface
        
    def initialize_system(self):
        """Initialize all system components"""
        print("Initializing CHRIS Reasoning System...")
        
        # Initialize environment
        self._initialize_environment()
        
        # Create agents based on configuration
        for agent_name, agent_config in AGENT_CONFIG.items():
            agent_type = agent_config['type']
            self.agents[agent_name] = self.agent_factory.create_agent(
                agent_type, agent_config, self.environment
            )
            print(f"Created {agent_type} agent: {agent_name}")
    
    def _initialize_environment(self):
        """Initialize the environment interface"""
        # This would connect to the actual environment (e.g., Unreal Tournament)
        # For now we'll just create a mock environment
        self.environment = {
            'name': ENVIRONMENT_CONFIG['name'],
            'state': {},
            'sensors': {}
        }
        print(f"Initialized environment: {self.environment['name']}")
    
    def run(self):
        """Main execution loop for the reasoning system"""
        print("Starting reasoning system main loop...")
        
        try:
            while True:
                # Update environment state
                self._update_environment_state()
                
                # Process each agent through the reasoning stages
                for agent_name, agent in self.agents.items():
                    self._process_agent(agent)
                
                # Small delay to prevent CPU overload
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nShutting down reasoning system...")
            self.shutdown()
    
    def _update_environment_state(self):
        """Update the environment state from sensors"""
        # In a real implementation, this would read actual sensor data
        # For now we'll just simulate some state changes
        self.environment['state']['timestamp'] = time.time()
        
        # Simulate some environment changes
        if 'counter' not in self.environment['state']:
            self.environment['state']['counter'] = 0
        self.environment['state']['counter'] += 1
    
    def _process_agent(self, agent):
        """Process an agent through the cognitive hybrid reasoning stages"""
        # Observation Stage
        agent.observe(self.environment)
        
        # Orientation Stage
        agent.orient()
        
        # Decision Stage
        agent.decide()
        
        # Action Stage
        actions = agent.act()
        
        # Learning Stage (if applicable)
        if hasattr(agent, 'learn'):
            agent.learn(self.environment)
        
        # Execute actions in environment
        if actions:
            self._execute_actions(agent, actions)
    
    def _execute_actions(self, agent, actions):
        """Execute the agent's actions in the environment"""
        # In a real implementation, this would interface with the environment
        print(f"Agent {agent.name} executing actions: {actions}")
        
        # Update environment based on actions
        for action in actions:
            if action['type'] == 'movement':
                self.environment['state'][f'{agent.name}_position'] = action['target']
            elif action['type'] == 'communication':
                # Handle inter-agent communication
                pass
    
    def shutdown(self):
        """Cleanup system resources"""
        print("Cleaning up resources...")
        for agent in self.agents.values():
            if hasattr(agent, 'cleanup'):
                agent.cleanup()
        print("System shutdown complete.")

def main():
    """Main entry point for the reasoning system"""
    reasoning_system = ReasoningSystem()
    reasoning_system.initialize_system()
    reasoning_system.run()

if __name__ == "__main__":
    main()

# ===== END OF main_reasoning.py =====


# ===== START OF main_safe_ai.py =====
import os
import sys
import yaml
import torch
import logging

from logs.logger import get_logger
from src.utils.logger import setup_logger, cleanup_logger
from src.research.evaluator import Evaluator
from src.research.experiment_manager import ExperimentManager
from src.tuning.tuner import HyperparamTuner
from src.utils.agent_factory import create_agent
from src.collaborative.shared_memory import SharedMemory
from modules.data_handler import DataHandler
from modules.model_trainer import ModelTrainer
from modules.security_manager import SecurityManager
from modules.monitoring import Monitoring
from modules.compliance_auditor import ComplianceAuditor
from deployment.git.rollback_handler import RollbackHandler

logger = setup_logger("SafeAIAgent", level=logging.INFO)

logger.info("Training started")
logger.warning("Risk score too high")

try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Failed to load config.yaml: {e}")
    sys.exit(1)

logger.info("Initializing Safe AI Agent...")

try:
    agent = create_agent(agent_name="safe_ai", shared_memory=shared_memory, config={"risk_threshold": 0.2})
    agent.run()
    logger.info("Safe AI Agent execution completed.")
except Exception as e:
    logger.error(f"Safe AI agent encountered an error: {e}", exc_info=True)

# Configure logging
os.makedirs(config['run']['output_dir'], exist_ok=True)

def main():
    logger.info(" Starting Safe AI Pipeline...")

    # Initialize Components
    shared_memory = SharedMemory()
    data_handler = DataHandler(shared_memory=shared_memory)
    model_trainer = ModelTrainer(shared_memory=shared_memory)
    security_manager = SecurityManager(shared_memory=shared_memory)
    monitoring = Monitoring(shared_memory=shared_memory)
    compliance_auditor = ComplianceAuditor()
    rollback_handler = RollbackHandler(models_dir=config['run']['output_dir'], backup_dir=config['rollback']['backup_dir'])

    try:
        # Step 1: Data Handling & Fairness Check
        logger.info("Loading and preprocessing data...")
        raw_data = data_handler.load_data(config['paths']['data_source'])

        if config['fairness'].get('enforce_fairness', False):
            data_handler.check_data_fairness(raw_data)

        clean_data = data_handler.preprocess_data(raw_data)

        # Step 2: Hyperparameter Tuning
        tuner = HyperparamTuner(
            agent_class=SafeAI_Agent,
            search_space={
                "risk_threshold": [0.3, 0.2, 0.1],
                "compliance_weight": [0.5, 1.0]  # Optional if supported
            },
            base_task={
                "policy_risk_score": 0.27,
                "task_type": "reinforcement_learning"
            },
            shared_memory=shared_memory,
            max_trials=6
        )
        best_tune = tuner.run_grid_search()
        logger.info(f"Best Tuning Result: {best_tune}")

        # Step 3: Experiment Management
        logger.info("Running multiple SafeAI experiments...")
        manager = ExperimentManager(shared_memory=shared_memory)
        results = manager.run_experiments(
            agent_configs=[
                {
                    "agent_class": SafeAI_Agent,
                    "init_args": {"shared_memory": shared_memory, "risk_threshold": 0.2},
                    "name": "safe_v1"
                },
                {
                    "agent_class": SafeAI_Agent,
                    "init_args": {"shared_memory": shared_memory, "risk_threshold": 0.1},
                    "name": "safe_strict"
                }
            ],
            task_data={
                "policy_risk_score": 0.27,
                "task_type": "reinforcement_learning"
            }
        )
        top = manager.summarize_results(sort_key="risk_score", minimize=True)[0]
        logger.info(f"\U0001F3C6 Best Agent: {top['agent']} with score {top['result']['risk_score']}")

        # Step 4: Model Training
        logger.info("Training model...")
        model = model_trainer.train_model(clean_data)

        # Step 5: Security Hardening
        if config['security'].get('encrypt_models', False):
            logger.info("Applying model security...")
            security_manager.secure_model(model)

        if config['security'].get('enable_threat_detection', False):
            security_manager.check_for_threats()

        # Step 6: Compliance Audit
        if config['compliance'].get('enable_audit', False):
            logger.info("⚖ Running compliance audit...")
            compliance_auditor.run_audit()

        # Step 7: Monitoring
        if config['monitoring'].get('enable_monitoring', False):
            logger.info("Starting monitoring...")
            monitoring.start(model, data_handler)

        # Step 8: Evaluation
        logger.info("Evaluating SafeAI agent...")
        evaluator = Evaluator(shared_memory=shared_memory, monitoring=monitoring)
        eval_result = evaluator.evaluate_agent(
            agent=SafeAI_Agent(shared_memory=shared_memory),
            task_data={
                "policy_risk_score": 0.32,
                "task_type": "meta_learning"
            },
            metadata={"experiment": "baseline_risk_check"}
        )
        logger.info(f"SafeAI Evaluation Result: {eval_result}")

        logger.info(" Safe AI Pipeline completed successfully!")

    except Exception as e:
        logger.error(f" Pipeline error: {e}", exc_info=True)
        if config['rollback'].get('enabled', False):
            logger.info(" Rolling back...")
            rollback_handler.rollback_model()

    finally:
        logger.info("Pipeline finished.")

if __name__ == "__main__":
    main()

print("Safe AI Pipeline completed successfully!")

# ===== END OF main_safe_ai.py =====
