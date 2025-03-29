import os
import sys
import logging
import time
import yaml
import json
import pickle
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from multiprocessing import Pool, cpu_count
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

# Vector DB imports (using FAISS for performance)
try:
    import faiss
except ImportError:
    logger.error("FAISS not installed. Please install with: pip install faiss-cpu")
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
    created_at: float = time.time()
    dependencies: List[str] = None
    assigned_agent: str = None

    def to_dict(self):
        return asdict(self)

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
        # Save memory state
        with open(f"{self.persist_path}.state", "wb") as f:
            pickle.dump({
                "memory": self.memory,
                "task_history": self.task_history
            }, f)
        
        # Save vector DB
        self.vector_db.save(self.persist_path)
    
    def load(self):
        try:
            # Load memory state
            with open(f"{self.persist_path}.state", "rb") as f:
                data = pickle.load(f)
                self.memory = data["memory"]
                self.task_history = data["task_history"]
            
            # Load vector DB
            self.vector_db.load(self.persist_path)
            logger.info("Loaded previous state from disk")
        except (FileNotFoundError, EOFError):
            logger.info("No previous state found, starting fresh")

class RLTaskRouter:
    """Reinforcement Learning based Task Router"""
    def __init__(self, shared_memory: SharedMemory):
        self.shared_memory = shared_memory
        self.task_queue = []
        self.agent_q_values = defaultdict(dict)  # agent -> task_type -> Q-value
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
    
    def add_task(self, task: Task):
        self.task_queue.append(task)
        self._reprioritize()
    
    def _reprioritize(self):
        # Sort by priority and creation time
        self.task_queue.sort(key=lambda x: (-x.priority, x.created_at))
    
    def get_next_task(self) -> Optional[Task]:
        if not self.task_queue:
            return None
        return self.task_queue[0]
    
    def select_agent(self, task_type: str, available_agents: List[str]) -> Optional[str]:
        # Exploration: choose random agent
        if np.random.random() < self.exploration_rate:
            return np.random.choice(available_agents) if available_agents else None
        
        # Exploitation: choose agent with highest Q-value for this task type
        best_agent, best_q = None, -float('inf')
        for agent in available_agents:
            q = self.agent_q_values[agent].get(task_type, 1.0)  # Default to 1.0
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
                priority=parent_task.priority * 0.9  # Slightly lower priority
            ))
        
        if "safety_concerns" in results:
            new_tasks.append(Task(
                id=f"safety_check_{parent_task.id}",
                type="safety",
                data={"risk_factors": results["safety_concerns"]},
                dependencies=[parent_task.id],
                priority=1.0  # Safety tasks get highest priority
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
        
        # Calculate basic metrics
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
        
        # Calculate derived metrics
        for task_type in metrics["throughput"]:
            if metrics["throughput"][task_type] > 0:
                metrics["success_rate"][task_type] /= metrics["throughput"][task_type]
                metrics["priority_efficiency"][task_type] = (
                    metrics["success_rate"][task_type] * np.mean(metrics["task_times"][task_type])
        
        for agent in agent_performance:
            total_tasks = agent_task_counts.get(agent, 1)
            metrics["agent_utilization"][agent] = agent_task_counts.get(agent, 0) / max(1, len(task_history))
        
        metrics["average_task_time"] = total_time / max(1, len(task_history))
        
        return metrics
    
    @staticmethod
    def visualize_metrics(metrics: Dict[str, Any], save_path: str = None):
        plt.figure(figsize=(15, 10))
        
        # Throughput by task type
        plt.subplot(2, 2, 1)
        task_types = list(metrics["throughput"].keys())
        counts = [metrics["throughput"][t] for t in task_types]
        plt.bar(task_types, counts)
        plt.title("Task Throughput by Type")
        plt.xticks(rotation=45)
        
        # Success rate by task type
        plt.subplot(2, 2, 2)
        success_rates = [metrics["success_rate"][t] for t in task_types]
        plt.bar(task_types, success_rates)
        plt.title("Success Rate by Task Type")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Agent utilization
        plt.subplot(2, 2, 3)
        agents = list(metrics["agent_utilization"].keys())
        utilizations = [metrics["agent_utilization"][a] for a in agents]
        plt.bar(agents, utilizations)
        plt.title("Agent Utilization")
        plt.xticks(rotation=45)
        
        # Priority efficiency
        plt.subplot(2, 2, 4)
        efficiencies = [metrics["priority_efficiency"][t] for t in task_types]
        plt.bar(task_types, efficiencies)
        plt.title("Priority Efficiency (Success Rate * Speed)")
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
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.pool = Pool(processes=self.num_workers)
        self.metrics = PerformanceMetrics()
    
    def register_agent(self, name: str, agent, capabilities: List[str]):
        self.registry.register(name, agent, capabilities)
        logger.info(f"Registered agent {name} with capabilities: {capabilities}")
    
    def _execute_task(self, task: Task, agent_name: str) -> Tuple[Task, Dict[str, Any]]:
        """Task execution function for multiprocessing"""
        agent_data = self.registry.get_agent(agent_name)
        agent = agent_data["instance"]
        
        try:
            result = agent.execute(task.type, task.data)
            task.status = "completed"
            return task, result
        except Exception as e:
            logger.error(f"Task {task.id} failed in worker: {str(e)}")
            task.status = "failed"
            return task, {"error": str(e)}
    
    def execute_task_cycle(self):
        """Execute one full cycle of task processing with parallel execution"""
        tasks_to_execute = []
        available_agents = []
        
        # Get available agents
        for agent_name in self.registry.list_agents():
            if self.registry.get_agent_status(agent_name) == "idle":
                available_agents.append(agent_name)
        
        # Assign tasks to available agents
        while available_agents and (task := self.router.get_next_task()):
            task = self.task_queue.pop(0)
            agent_name = self.router.select_agent(task.type, available_agents)
            
            if agent_name:
                self.registry.update_status(agent_name, "busy")
                task.assigned_agent = agent_name
                tasks_to_execute.append((task, agent_name))
                available_agents.remove(agent_name)
        
        # Execute tasks in parallel
        if tasks_to_execute:
            results = self.pool.starmap(self._execute_task, tasks_to_execute)
            
            for task, result in results:
                # Update task status and store results
                self.shared_memory.task_history.append(task)
                
                if task.status == "completed":
                    # Generate embedding for the result
                    embedding = self._generate_result_embedding(result)
                    self.shared_memory.store_embedding(
                        vector=embedding,
                        metadata={
                            "task": task.type,
                            "agent": task.assigned_agent,
                            "timestamp": time.time(),
                            "result": result
                        }
                    )
                    
                    # Update RL router with performance
                    performance = self._evaluate_performance(task, result)
                    self.router.update_q_values(
                        agent=task.assigned_agent,
                        task_type=task.type,
                        reward=performance,
                        next_state_value=performance * 0.9  # Estimated next state value
                    )
                    
                    # Create follow-up tasks
                    self.router.create_child_tasks(task, result)
                
                # Mark agent as available again
                self.registry.update_status(task.assigned_agent, "idle")
            
            return True
        
        return False
    
    def _generate_result_embedding(self, result: Dict[str, Any]) -> np.ndarray:
        """Convert result to embedding vector"""
        # In production, use a proper embedding model
        return np.random.rand(128)  # Placeholder
    
    def _evaluate_performance(self, task: Task, result: Dict[str, Any]) -> float:
        """Evaluate task performance with multiple factors"""
        time_taken = time.time() - task.created_at
        success = 1.0 if task.status == "completed" else 0.0
        quality = result.get("performance", 0.5)
        safety = result.get("safety_score", 1.0)
        
        # Weighted combination of factors
        return (0.4 * quality + 0.3 * success + 0.2 * safety + 0.1 * (1 - time_taken/100))
    
    def save_state(self, path: str = "slai_state"):
        """Save the entire system state"""
        # Save shared memory
        self.shared_memory.save()
        
        # Save router state
        with open(f"{path}.router", "wb") as f:
            pickle.dump(self.router.agent_q_values, f)
        
        # Save registry state
        self.registry.save(f"{path}.registry")
        
        logger.info(f"System state saved to {path}")
    
    def load_state(self, path: str = "slai_state"):
        """Load system state from disk"""
        try:
            # Load router state
            with open(f"{path}.router", "rb") as f:
                self.router.agent_q_values = pickle.load(f)
            
            # Load registry state
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
    
    def find_agents_for_task(self, task_type: str) -> List[str]:
        return [name for name, data in self.agents.items() 
                if task_type in data["capabilities"] and data["status"] == "idle"]
    
    def save(self, path: str):
        """Save registry state to disk"""
        with open(path, "wb") as f:
            pickle.dump(self.agents, f)
    
    def load(self, path: str):
        """Load registry state from disk"""
        try:
            with open(path, "rb") as f:
                self.agents = pickle.load(f)
        except FileNotFoundError:
            logger.warning("No previous registry state found")

# Example Agent Implementations
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
                    "performance": 0.8  # High performance for catching risks
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
            # Simulate task learning with some variability
            performance = 0.7 + np.random.normal(0, 0.1)
            return {
                "performance": performance,
                "task_complexity": 0.5,
                "current_task": self.tasks[self.current_task],
                "safety_score": 0.8
            }
        return {"status": "unhandled_task_type", "performance": 0.0}

def initialize_system():
    """Initialize the SLAI system with core components"""
    shared_memory = SharedMemory()
    collab_mgr = CollaborationManager(shared_memory, num_workers=4)
    
    # Register core agents
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
    
    # Initialize with some tasks
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
        # Main execution loop
        for cycle in range(20):  # Run for 20 cycles
            logger.info(f"\n=== Execution Cycle {cycle + 1} ===")
            collab_mgr.execute_task_cycle()
            
            # Periodically save state and show metrics
            if cycle % 5 == 0:
                collab_mgr.save_state()
                
                # Calculate and visualize metrics
                metrics = PerformanceMetrics.calculate_metrics(
                    collab_mgr.shared_memory.task_history,
                    collab_mgr.registry.agents
                )
                PerformanceMetrics.visualize_metrics(metrics, f"metrics_cycle_{cycle}.png")
        
        # Final metrics and state save
        collab_mgr.save_state()
        metrics = PerformanceMetrics.calculate_metrics(
            collab_mgr.shared_memory.task_history,
            collab_mgr.registry.agents
        )
        PerformanceMetrics.visualize_metrics(metrics, "final_metrics.png")
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        collab_mgr.save_state()
    finally:
        collab_mgr.pool.close()
        collab_mgr.pool.join()
    
    logger.info("System execution completed")

if __name__ == "__main__":
    main()
