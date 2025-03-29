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
