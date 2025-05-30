import threading
import json, yaml
import time
import requests

from requests.exceptions import RequestException
from abc import ABC, abstractmethod
from typing import Dict, List, Union
from queue import PriorityQueue
from dataclasses import dataclass
from types import SimpleNamespace
from threading import Lock

from src.agents.planning.planning_types import Task, TaskType
from logs.logger import get_logger

logger = get_logger("Safety Planning")

CONFIG_PATH = "src/agents/planning/configs/planning_config.yaml"

def dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d

def get_config_section(section: Union[str, Dict], config_file_path: str):
    if isinstance(section, dict):
        return dict_to_namespace(section)
    
    with open(config_file_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if section not in config:
        raise KeyError(f"Section '{section}' not found in config file: {config_file_path}")
    return dict_to_namespace(config[section])

# Data Structures
@dataclass
class ResourceProfile:
    gpu: int = 0
    ram: int = 0  # In GB
    specialized_hardware: List[str] = None

class ClusterResources(SimpleNamespace):
    gpu_total: int
    ram_total: int
    specialized_hardware_available: List[str]
    current_allocations: Dict[str, ResourceProfile]

# Custom Exceptions
class AdjustmentError(Exception):
    """Exception for invalid plan adjustments."""
    def __init__(self, message: str, adjustment: Dict, conflict_details: Dict = None):
        super().__init__(message)
        self.adjustment = adjustment
        self.conflict_details = conflict_details or {}
        self.timestamp = time.time()

class ReplanningError(Exception):
    """Exception for failures in recovery planning processes."""
    def __init__(self, message: str, failed_task: Task, candidates: List = None):
        super().__init__(message)
        self.failed_task = failed_task
        self.candidates = candidates or []
        self.metadata = {
            "error_type": "replanning_failure",
            "timestamp": time.time()
        }

class TemporalViolation(Exception):
    """Base class for temporal constraint violations"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message)
        self.violation_details = kwargs

class ResourceViolation(Exception):
    """Exception for unmet resource constraints"""
    def __init__(self, message: str, required: Dict, available: Dict):
        super().__init__(message)
        self.required = required
        self.available = available
        self.metadata = self._generate_metadata()

    def _generate_metadata(self) -> Dict:
        return {
            "violation_type": "resource_constraint",
            "timestamp": time.time(),
            "resolution_strategies": [
                "resource_scaling",
                "task_repurposing",
                "priority_reallocation"
            ]
        }

class SafetyMarginError(ResourceViolation):
    """Exception for safety margin violations"""
    def __init__(self, message: str, resource_type: str, buffer_amount: float):
        super().__init__(message, required={}, available={})
        self.resource_type = resource_type
        self.buffer_amount = buffer_amount
        self.metadata['violation_type'] = 'safety_margin'

class SafetyPlanning:
    """Core module for safe distributed planning with interactive adjustments"""
    
    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        self.adjustment_queue = PriorityQueue()
        self.resource_monitor = ResourceMonitor()
        self.distributed_orchestrator = DistributedOrchestrator(config)
        self.current_plan = []  # Add internal plan storage
        self.task_library = []  # Add task library if needed

    def interactive_adjustment_handler(self, adjustment: Dict) -> None:
        """Handle real-time plan modifications from UI/API"""
        with self.lock:
            try:
                self._validate_adjustment(adjustment)
                adjusted_plan = self._apply_adjustment(
                    self.current_plan,
                    adjustment
                )
                if adjusted_plan and self.safety_check(adjusted_plan):
                    self.current_plan = adjusted_plan  # Store internally
                    self.log_adjustment(adjustment)
            except AdjustmentError as e:
                self.handle_adjustment_failure(e, adjustment)

    def _apply_adjustment(self, current_plan: List[Task], adjustment: Dict) -> List[Task]:
        """Dummy implementation - should be replaced with actual adjustment logic"""
        # Return a copy of current plan if no adjustment logic exists
        return [task for task in current_plan] if current_plan else []

    def _validate_adjustment(self, adjusment):
        pass

    def handle_adjustment_failure(self, e, adjusment):
        pass

    def distributed_decomposition(self, task: Task) -> List[Task]:
        """Distribute task across available resources"""
        if not self.resource_monitor.check_global_resources(task):
            raise ResourceViolation(
                "Global resources insufficient for distributed execution",
                task.resource_requirements,
                self.resource_monitor.cluster_resources
            )

        return self.distributed_orchestrator.decompose_and_distribute(
            task,
            self.agent.task_library,
            self.resource_monitor.current_loads
        )

    def safety_check(self, plan: List[Task]) -> bool:
        """Comprehensive safety validation for plans"""
        for task in plan:
            self._validate_equipment_constraints(task)
            self._validate_temporal_constraints(task)
            self._validate_safety_margins(task)
        return True

    def log_adjustment(self, adjusment):
        pass

    def _validate_temporal_constraints(self, task: Task) -> None:
        """Validate task timing against current plan and system state."""
        current_time = time.time()
        
        # Hard deadline validation
        if task.deadline < current_time:
            raise ValueError(f"Task {task.id} deadline {task.deadline} is in the past")
            
        # Schedule feasibility check
        estimated_duration = self._calculate_estimated_duration(task)
        if (current_time + estimated_duration) > task.deadline:
            raise TemporalViolation(
                f"Insufficient time to complete task {task.id}. "
                f"Required: {estimated_duration}s, Available: {task.deadline - current_time:.1f}s",
                required_duration=estimated_duration,
                available_window=task.deadline - current_time
            )

        # Dependency chain validation
        if task.dependencies:
            latest_prereq_end = max(
                t.end_time for t in self.current_plan
                if t.id in task.dependencies
            )
            if task.start_time < latest_prereq_end:
                raise TemporalViolation(
                    f"Task {task.id} starts before dependency completion",
                    dependency_end=latest_prereq_end,
                    scheduled_start=task.start_time
                )

    def _validate_safety_margins(self, task: Task) -> None:
        """Ensure resource allocations maintain safety buffers."""
        config = get_config_section('safety_margins', CONFIG_PATH)
        available = self.resource_monitor.get_available_resources()
        
        # Calculate safety buffers
        gpu_buffer = available.gpu * config.resource_buffers.gpu
        ram_buffer = available.ram * config.resource_buffers.ram
        specialized_buffer = config.resource_buffers.specialized_hardware
        
        # Check GPU safety margin
        if (available.gpu - task.resource_requirements.gpu) < gpu_buffer:
            raise SafetyMarginError(
                f"GPU allocation violates safety buffer. "
                f"Available: {available.gpu}, Requested: {task.resource_requirements.gpu}, "
                f"Minimum buffer: {gpu_buffer}",
                resource_type='gpu',
                buffer_amount=gpu_buffer
            )
            
        # Check specialized hardware availability
        required_specialized = set(task.resource_requirements.specialized_hardware)
        reserved_hardware = set(available.specialized_hardware) - set(specialized_buffer)
        if not required_specialized.issubset(reserved_hardware):
            missing = required_specialized - reserved_hardware
            raise SafetyMarginError(
                f"Specialized hardware buffer violated. Missing: {missing}",
                resource_type='specialized_hardware',
                buffer_amount=specialized_buffer
            )

        # Temporal safety buffer (minimum idle time between tasks)
        if task.duration < config.temporal.min_task_duration:
            raise SafetyMarginError(
                f"Task duration too short for safe execution. "
                f"Current: {task.duration}s, Minimum: {config.temporal.min_task_duration}s",
                resource_type='temporal',
                buffer_amount=config.temporal.min_task_duration
            )

    def _validate_equipment_constraints(self, task: Task) -> None:
        """Enhanced equipment validation with dynamic resource checking"""
        req = task.resource_requirements
        available = self.resource_monitor.get_available_resources()

        if req.gpu > available.gpu:
            raise ResourceViolation(
                f"GPU required: {req.gpu}, available: {available.gpu}",
                req.gpu,
                available.gpu
            )
        if req.specialized_hardware and not available.specialized_hardware:
            missing = set(req.specialized_hardware) - \
                     set(available.specialized_hardware)
            raise ResourceViolation(
                f"Missing specialized hardware: {missing}",
                req.specialized_hardware,
                available.specialized_hardware
            )

    # Additional Core Methods
    def dynamic_replanning_pipeline(self, failed_task: Task) -> List[Task]:
        """Full safety-aware replanning workflow"""
        with self.lock:
            try:
                candidates = self._generate_repair_candidates(failed_task)
                validated = [
                    c for c in candidates
                    if self.safety_check(c.repaired_plan)
                ]
                return self._select_optimal_repair(validated)
            except ReplanningError as e:
                logger.error(f"Emergency shutdown triggered: {str(e)}")
                self._emergency_shutdown_procedure()

    def _emergency_shutdown_procedure(self):
        """Internal emergency handling"""
        logger.critical("Initiating emergency shutdown sequence")
        self.current_plan = []

    def update_allocations(self, requirements: ResourceProfile):
        """Update resource monitor with new allocations"""
        self.resource_monitor.allocate_resources(requirements)

class DistributedOrchestrator:
    """Manages distributed execution across multiple agents/nodes"""
    
    def __init__(self, shared_memory):
        self.shared_memory = shared_memory
        self.decomposition_strategies = {
            'horizontal': self.horizontal_split,
            'vertical': self.vertical_split,
            'hybrid': self.hybrid_split
        }
        self.config = get_config_section('decomposition', CONFIG_PATH)

    def horizontal_split(self, task: Task) -> List[Task]:
        """Data-parallel decomposition with dynamic chunk sizing"""
        try:
            # Calculate optimal split count based on data size and cluster resources
            data_size = len(task.input_data)
            chunk_size = max(
                self.config.min_chunk_size,
                data_size // self.shared_memory.cluster_size
            )
            
            subtasks = []
            for i in range(0, data_size, chunk_size):
                chunk = task.input_data[i:i+chunk_size]
                subtask = Task(
                    id=f"{task.id}_h{i//chunk_size}",
                    input_data=chunk,
                    resource_requirements=task.resource_requirements.copy(),
                    parent_task=task.id,
                    dependencies=task.dependencies,
                    deadline=task.deadline
                )
                # Reduce memory requirements for smaller chunks
                subtask.resource_requirements.ram = max(
                    task.resource_requirements.ram // 2,
                    self.config.min_ram_per_chunk
                )
                subtasks.append(subtask)
            
            logger.info(f"Split task {task.id} into {len(subtasks)} horizontal chunks")
            return subtasks
            
        except (AttributeError, TypeError) as e:
            logger.error(f"Horizontal split failed for {task.id}: {str(e)}")
            return [task]

    def vertical_split(self, task: Task) -> List[Task]:
        """Functional decomposition into processing stages"""
        stages = getattr(self.config.vertical_stages, str(task.task_type), [])
        if not stages:
            logger.warning(f"No vertical stages defined for {task.task_type}")
            return [task]

        subtasks = []
        prev_stage = None
        for stage_num, stage_config in enumerate(stages):
            subtask = Task(
                name=f"{task.name}_v{stage_num}",
                processing_stage = stage_config.name,
                resource_requirements=ResourceProfile(
                    gpu = getattr(stage_config, 'gpu', 0),
                    ram = getattr(stage_config, 'ram', task.resource_requirements.ram),
                    specialized_hardware = getattr(stage_config, 'specialized_hw', [])
                ),
                parent_task=task.name,
                dependencies=[prev_stage.name] if prev_stage else task.dependencies,
                deadline=task.deadline - (len(stages) - stage_num) * self.config.stage_time_buffer
            )
            if prev_stage:
                prev_stage.output_target = subtask.name
            subtasks.append(subtask)
            prev_stage = subtask

        logger.info(f"Vertically decomposed {task.id} into {len(subtasks)} stages")
        return subtasks

    def hybrid_split(self, task: Task) -> List[Task]:
        """Combined vertical and horizontal decomposition"""
        vertical_stages = self.vertical_split(task)
        hybrid_tasks = []
        
        for stage in vertical_stages:
            if stage.resource_requirements.gpu > 0:  # Only split GPU-heavy stages
                horizontal_subtasks = self.horizontal_split(stage)
                hybrid_tasks.extend(horizontal_subtasks)
            else:
                hybrid_tasks.append(stage)
                
        # Rebuild dependencies between stages
        for i in range(1, len(hybrid_tasks)):
            if hybrid_tasks[i].parent_task != hybrid_tasks[i-1].parent_task:
                hybrid_tasks[i].dependencies.append(hybrid_tasks[i-1].id)
                
        return hybrid_tasks

class ResourceMonitor:
    """Real-time cluster resource tracking with failure resilience"""
    
    def __init__(self):
        self.cluster_resources = ClusterResources()
        self.update_interval = 5  # seconds
        self.node_query_timeout = 2  # seconds
        self._lock = Lock()
        self._service_discovery_config = get_config_section('service_discovery', CONFIG_PATH)
        self._init_monitoring_thread()
        self._node_cache = {}  # Cache last known good states

    def _init_monitoring_thread(self):
        def monitor_loop():
            while True:
                self._update_resource_map()
                time.sleep(self.update_interval)

        threading.Thread(
            target=monitor_loop,
            daemon=True
        ).start()

    def _discover_cluster_nodes(self):
        """Discover nodes through configured service discovery backend"""
        try:
            if self._service_discovery_config.mode == 'consul':
                return self._query_consul_cluster()
            elif self._service_discovery_config.mode == 'k8s':
                return self._query_kubernetes_cluster()
            else:  # Default to static config
                return self._service_discovery_config.static_nodes
        except Exception as e:
            logger.error(f"Service discovery failed: {str(e)}")
            return list(self._node_cache.keys())  # Fallback to cached nodes

    def _query_consul_cluster(self):
        """Query Consul service discovery"""
        try:
            response = requests.get(
                f"{self._service_discovery_config.consul_url}/v1/catalog/nodes",
                timeout=self.node_query_timeout
            )
            response.raise_for_status()
            return [node['Node'] for node in response.json()]
        except RequestException as e:
            logger.error(f"Consul query failed: {str(e)}")
            return []

    def _query_kubernetes_cluster(self):
        """Query Kubernetes API for nodes"""
        try:
            headers = {"Authorization": f"Bearer {self._service_discovery_config.k8s_token}"}
            response = requests.get(
                f"{self._service_discovery_config.k8s_api}/api/v1/nodes",
                headers=headers,
                timeout=self.node_query_timeout
            )
            response.raise_for_status()
            return [item['metadata']['name'] for item in response.json()['items']]
        except RequestException as e:
            logger.error(f"Kubernetes API error: {str(e)}")
            return []

    def _query_node_resources(self, node_id):
        """Make RPC call to node's resource endpoint"""
        try:
            # First check cache for recent data
            if node_id in self._node_cache:
                cached = self._node_cache[node_id]
                if time.time() - cached['timestamp'] < self.update_interval * 2:
                    return cached['data']

            # Make HTTP call to node's metrics endpoint
            response = requests.get(
                f"http://{node_id}:{self._service_discovery_config.node_port}/metrics",
                timeout=self.node_query_timeout
            )
            response.raise_for_status()
            
            # Parse response
            metrics = response.json()
            resource_data = {
                'gpu_available': metrics['gpu']['free'],
                'ram_available': metrics['memory']['free'],
                'specialized_hw': metrics.get('specialized_hw', []),
                'gpu_allocated': metrics['gpu']['total'] - metrics['gpu']['free'],
                'ram_allocated': metrics['memory']['total'] - metrics['memory']['free'],
                'specialized_allocated': metrics.get('specialized_allocated', [])
            }
            
            # Update cache
            self._node_cache[node_id] = {
                'data': resource_data,
                'timestamp': time.time()
            }
            
            return resource_data
            
        except RequestException as e:
            logger.warning(f"Resource query failed for {node_id}: {str(e)}")
            # Return cached data if available
            if node_id in self._node_cache:
                return self._node_cache[node_id]['data']
            return None

    def _update_resource_map(self):
        """Thread-safe resource map update"""
        with self._lock:
            new_resources = ClusterResources(
                gpu_total=0,
                ram_total=0,
                specialized_hardware_available=[],
                current_allocations={}
            )
            
            nodes = self._discover_cluster_nodes()
            seen_hardware = set()
    
            if not nodes:  # Add check for empty node list
                logger.warning("No cluster nodes discovered")
                return
            
            for node_id in nodes:
                node_data = self._query_node_resources(node_id)
                if not node_data:
                    continue
                
                # Update totals
                new_resources.gpu_total += node_data['gpu_available']
                new_resources.ram_total += node_data['ram_available']
                
                # Track specialized hardware
                for hw in node_data['specialized_hw']:
                    if hw not in seen_hardware:
                        new_resources.specialized_hardware_available.append(hw)
                        seen_hardware.add(hw)
                
                # Store node allocation
                new_resources.current_allocations[node_id] = ResourceProfile(
                    gpu=node_data['gpu_allocated'],
                    ram=node_data['ram_allocated'],
                    specialized_hardware=node_data['specialized_allocated']
                )
            
            # Only update if significant changes occur
            if new_resources != self.cluster_resources:
                self.cluster_resources = new_resources
                logger.info("Cluster resource map updated")

    def allocate_resources(self, requirements: ResourceProfile):
        """Track resource consumption"""
        with self._lock:
            self.cluster_resources.gpu_total -= requirements.gpu
            self.cluster_resources.ram_total -= requirements.ram
            self.cluster_resources.specialized_hardware_available = [
                hw for hw in self.cluster_resources.specialized_hardware_available 
                if hw not in requirements.specialized_hardware
            ]

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Safety Planning ===\n")
    adjustment = None
    shared_memory = {}
    task = Task(
        name="test_task",
        task_type=TaskType.ABSTRACT,  # or TaskType.PRIMITIVE depending on context
    )
    task.resource_requirements = ResourceProfile(gpu=1, ram=16, specialized_hardware=[])
    task.deadline = time.time() + 3600  # Optional: Set a deadline
    task.dependencies = []  

    safety = SafetyPlanning(config=get_config_section('safety_margins', CONFIG_PATH))
    safety.current_plan = []
    orchestrator = DistributedOrchestrator(
        shared_memory=SimpleNamespace(
            cluster_size=4,
            vertical_stages=get_config_section('decomposition', CONFIG_PATH).vertical_stages
        )
    )

    safety.interactive_adjustment_handler({})
    orchestrator.vertical_split(task)

    print(f"Selected action: {safety}")
    print(orchestrator)

    print("\n* * * * * Phase 2 * * * * *\n")
    monitor = ResourceMonitor()
    monitor._init_monitoring_thread()

    print(monitor)
    print("\n=== Successfully Ran Safety Planning ===\n")
