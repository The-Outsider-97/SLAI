import threading
import json, yaml
import time, copy
import requests

from requests.exceptions import RequestException
from abc import ABC, abstractmethod
from typing import Dict, List, Union
from queue import PriorityQueue
from dataclasses import dataclass, field
from types import SimpleNamespace
from threading import Lock

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.utils.planning_errors import (AdjustmentError, ReplanningError, TemporalViolation,
                                                       SafetyMarginError, ResourceViolation)
from src.agents.planning.planning_types import Task, TaskType
from src.agents.planning.planning_memory import PlanningMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Safety Planning")
printer = PrettyPrinter

# Data Structures
@dataclass
class ResourceProfile:
    gpu: int = 0
    ram: int = 0  # In GB
    specialized_hardware: List[str] = None

@dataclass
class ClusterResources:
    gpu_total: int = 0
    ram_total: int = 0
    specialized_hardware_available: List[str] = field(default_factory=list)
    current_allocations: Dict[str, ResourceProfile] = field(default_factory=dict)

class SafetyPlanning:
    """Core module for safe distributed planning with interactive adjustments"""
    
    def __init__(self):
        self.config = load_global_config()
        self.safety_config = get_config_section('safety_planning')
        self.margins_config = get_config_section('safety_margins')
        self.resource_buffers = self.margins_config.get('resource_buffers', {
            'gpu', 'ram', 'specialized_hardware'
        })
        self.temporal = self.margins_config.get('temporal', {
            'min_task_duration', 'max_concurrent', 'time_buffer'
        })
        self.lock = threading.RLock()
        self.memory = PlanningMemory()
        self.adjustment_queue = PriorityQueue()
        self.resource_monitor = ResourceMonitor()
        self.distributed_orchestrator = DistributedOrchestrator()
        self.current_plan = []  # Add internal plan storage
        self.task_library = []  # Add task library if needed

    @property
    def safety_margins(self):
        return self.margins_config

    def interactive_adjustment_handler(self, adjustment: Dict) -> None:
        """Handle real-time plan modifications from UI/API"""
        printer.status("INIT", "Interactive handler succesfully initialized", "info")

        if not adjustment or not adjustment.get("type"):
            printer.status("ADJUST-REJECT", "Empty or invalid adjustment received", "warning")
            return  # Prevent recursion or undefined behavior

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
        """
        Applies a real-time plan adjustment safely by modifying the current task list.
        
        Supported adjustment types:
        - "modify_task": updates parameters like deadline, resources, or duration
        - "add_task": appends a new validated task
        - "remove_task": deletes a task (and possibly dependent tasks)
        
        Raises:
            AdjustmentError if the adjustment is invalid or unsafe.
        """
        adjusted_plan = current_plan.copy()
        adjustment_type = adjustment.get("type")
        task_id = adjustment.get("task_id")
    
        if adjustment_type == "modify_task":
            updated = False
            for i, task in enumerate(adjusted_plan):
                if task.id == task_id:
                    for key, value in adjustment.get("updates", {}).items():
                        if hasattr(task, key):
                            setattr(task, key, value)
                    updated = True
                    break
            if not updated:
                raise AdjustmentError(f"Task {task_id} not found for modification", adjustment)
    
        elif adjustment_type == "add_task":
            new_task = adjustment.get("task")
            if not new_task:
                raise AdjustmentError("Missing 'task' in add_task adjustment", adjustment)
            adjusted_plan.append(new_task)
    
        elif adjustment_type == "remove_task":
            adjusted_plan = [t for t in adjusted_plan if t.id != task_id]
            # Optional: also remove dependents if needed
    
        else:
            raise AdjustmentError(f"Unsupported adjustment type: {adjustment_type}", adjustment)
    
        return adjusted_plan

    def _validate_adjustment(self, adjustment: Dict) -> None:
        """Comprehensive validation of adjustment requests with safety checks"""
        printer.status("VALIDATE", f"Validating adjustment: {adjustment.get('type')}", "info")
        
        # Basic structural validation
        if not isinstance(adjustment, dict):
            raise AdjustmentError("Adjustment must be a dictionary", adjustment)
        
        adj_type = adjustment.get("type")
        if adj_type not in ["modify_task", "add_task", "remove_task"]:
            raise AdjustmentError(f"Invalid adjustment type: {adj_type}", adjustment)
        
        # Type-specific validation
        if adj_type == "modify_task":
            self._validate_modification(adjustment)
        elif adj_type == "add_task":
            self._validate_addition(adjustment)
        elif adj_type == "remove_task":
            self._validate_removal(adjustment)
        
        # Temporal validation (deadline consistency)
        if 'deadline' in adjustment.get('updates', {}):
            new_deadline = adjustment['updates']['deadline']
            if new_deadline < time.time():
                raise TemporalViolation(
                    f"Adjusted deadline {new_deadline} is in the past",
                    scheduled_time=new_deadline,
                    current_time=time.time()
                )
        
        # Resource validation
        if 'resource_requirements' in adjustment.get('updates', {}):
            new_req = adjustment['updates']['resource_requirements']
            self._validate_equipment_constraints(Task(resource_requirements=new_req))
            self._validate_safety_margins(Task(resource_requirements=new_req))
        
        printer.status("VALIDATE-SUCCESS", "Adjustment validation passed", "success")
    
    def _validate_modification(self, adjustment: Dict) -> None:
        """Validate task modification requests"""
        task_id = adjustment.get("task_id")
        updates = adjustment.get("updates", {})
        
        if not task_id:
            raise AdjustmentError("Missing task_id for modification", adjustment)
        
        if not updates:
            raise AdjustmentError("No updates specified for task modification", adjustment)
        
        # Verify task exists in current plan
        task_exists = any(t.id == task_id for t in self.current_plan)
        if not task_exists:
            raise AdjustmentError(f"Task {task_id} not found in current plan", adjustment)
        
        # Validate update keys
        valid_keys = {'deadline', 'priority', 'resource_requirements', 'dependencies'}
        invalid_keys = set(updates.keys()) - valid_keys
        if invalid_keys:
            raise AdjustmentError(f"Invalid update keys: {invalid_keys}", adjustment)
    
    def _validate_addition(self, adjustment: Dict) -> None:
        """Validate new task addition requests"""
        new_task = adjustment.get("task")
        
        if not new_task:
            raise AdjustmentError("Missing task object for addition", adjustment)
        
        # Basic task validation
        if not hasattr(new_task, 'id'):
            raise AdjustmentError("New task missing ID field", adjustment)
        
        # Check for duplicate IDs
        if any(t.id == new_task.id for t in self.current_plan):
            raise AdjustmentError(f"Task ID {new_task.id} already exists", adjustment)
        
        # Validate dependencies
        for dep_id in getattr(new_task, 'dependencies', []):
            if not any(t.id == dep_id for t in self.current_plan):
                raise AdjustmentError(f"Dependency {dep_id} not found", adjustment)
    
    def _validate_removal(self, adjustment: Dict) -> None:
        """Validate task removal requests"""
        task_id = adjustment.get("task_id")
        
        if not task_id:
            raise AdjustmentError("Missing task_id for removal", adjustment)
        
        # Verify task exists
        if not any(t.id == task_id for t in self.current_plan):
            raise AdjustmentError(f"Task {task_id} not found in current plan", adjustment)
        
        # Check for dependent tasks
        if getattr(adjustment, 'cascade', False):
            return  # Allow removal with dependents if cascade is specified
        
        dependents = [t.id for t in self.current_plan 
                     if task_id in getattr(t, 'dependencies', [])]
        if dependents:
            raise AdjustmentError(
                f"Cannot remove task {task_id} with dependents: {dependents}",
                adjustment,
                conflict_details={'dependents': dependents}
            )

    def handle_adjustment_failure(self, e: AdjustmentError, adjustment: Dict) -> None:
        """Comprehensive failure handling with fallback strategies"""
        printer.status("ADJUST-FAIL", f"Adjustment failed: {str(e)}", "error")
        
        # Log detailed diagnostics
        failure_data = {
            'timestamp': time.time(),
            'adjustment': adjustment,
            'error_type': type(e).__name__,
            'message': str(e),
            'current_plan': [t.id for t in self.current_plan],
            'resource_status': self.resource_monitor.get_available_resources().__dict__
        }
        
        if isinstance(e, ResourceViolation):
            self._handle_resource_failure(e, adjustment)
        elif isinstance(e, TemporalViolation):
            self._handle_temporal_failure(e, adjustment)
        else:
            self._handle_generic_failure(e, adjustment)
        
        # Notify monitoring systems
        self._send_alert_notification(failure_data)
        
        # Create diagnostic checkpoint
        self.memory.save_checkpoint(
            label=f"adjust_fail_{adjustment.get('type')}",
            metadata=failure_data
        )
    
    def _handle_resource_failure(self, e: ResourceViolation, adjustment: Dict) -> None:
        """Resource-specific failure handling"""
        printer.status("RESOURCE-FAIL", f"Resource violation: {str(e)}", "warning")
        
        # Attempt to scale resources
        if self._attempt_resource_scaling(e.required, e.available):
            printer.status("RESOURCE-RECOVER", "Resource scaled successfully", "success")
            # Retry adjustment after scaling
            self.interactive_adjustment_handler(adjustment)
            return
        
        # Fallback to task decomposition
        if adjustment.get("type") == "add_task":
            task = adjustment.get("task")
            try:
                subtasks = self.distributed_decomposition(task)
                if subtasks:
                    # Replace single task with decomposed subtasks
                    adjustment['type'] = "add_subtasks"
                    adjustment['subtasks'] = subtasks
                    adjustment.pop('task', None)
                    printer.status("DECOMPOSE-FALLBACK", "Using task decomposition", "warning")
                    self.interactive_adjustment_handler(adjustment)
                    return
            except Exception as decomp_error:
                logger.error(f"Decomposition failed: {str(decomp_error)}")
        
        # Final fallback: queue for later execution
        self._queue_for_later_execution(adjustment)
        printer.status("QUEUED", "Adjustment queued for later execution", "info")
    
    def _handle_temporal_failure(self, e: TemporalViolation, adjustment: Dict) -> None:
        """Temporal constraint failure handling"""
        printer.status("TIME-FAIL", f"Temporal violation: {str(e)}", "warning")
        
        # Attempt to reprioritize
        if 'deadline' in adjustment.get('updates', {}):
            original_deadline = adjustment['updates']['deadline']
            extended = original_deadline + self.temporal.time_buffer * 2
            adjustment['updates']['deadline'] = extended
            
            try:
                printer.status("DEADLINE-EXTEND", f"Extending deadline to {extended}", "info")
                self.interactive_adjustment_handler(adjustment)
                return
            except Exception:
                # Reset to original if extension fails
                adjustment['updates']['deadline'] = original_deadline
        
        # Fallback to resource reallocation
        if self._reallocate_resources_for_priority(adjustment):
            return
        
        # Final fallback: partial execution
        self._enable_partial_execution(adjustment)
    
    def _handle_generic_failure(self, e: Exception, adjustment: Dict) -> None:
        """Generic failure handling strategy"""
        printer.status("GENERIC-FAIL", f"Generic failure: {str(e)}", "error")
        
        # Attempt simple retry with backoff
        max_retries = self.config.get('adjustment_retries', 5)
        retry_count = adjustment.get('_retry_count', 0)
        
        if retry_count < max_retries:
            adjustment['_retry_count'] = retry_count + 1
            backoff = 2 ** retry_count
            printer.status("RETRY", f"Retrying in {backoff}s (attempt {retry_count+1}/{max_retries})", "info")
            time.sleep(backoff)
            self.interactive_adjustment_handler(adjustment)
            return
        
        # Fallback to human intervention
        self._escalate_to_human_operator(adjustment, str(e))
    
    # Helper methods for failure handling
    def _attempt_resource_scaling(self, required: Dict, available: Dict) -> bool:
        """Attempt to scale cluster resources dynamically"""
        printer.status("SCALING", "Attempting resource scaling", "info")
        # Implementation would interface with cloud provider APIs
        # or cluster orchestration systems
        return False  # Placeholder
    
    def _reallocate_resources_for_priority(self, adjustment: Dict) -> bool:
        """Reallocate resources from lower-priority tasks"""
        task_id = adjustment.get("task_id")
        if not task_id:
            return False
        
        # Find lower priority tasks that could yield resources
        candidate_tasks = [
            t for t in self.current_plan 
            if t.priority > adjustment.get('priority', 1)
            and t.id != task_id
        ]
        
        if not candidate_tasks:
            return False
        
        # Try pausing lowest priority task
        candidate_tasks.sort(key=lambda x: x.priority)
        task_to_pause = candidate_tasks[0]
        
        try:
            pause_adjustment = {
                'type': 'modify_task',
                'task_id': task_to_pause.id,
                'updates': {'status': 'paused'}
            }
            self.interactive_adjustment_handler(pause_adjustment)
            
            # Retry original adjustment
            self.interactive_adjustment_handler(adjustment)
            return True
        except Exception:
            # Revert pausing if fails
            resume_adjustment = {
                'type': 'modify_task',
                'task_id': task_to_pause.id,
                'updates': {'status': 'active'}
            }
            self.interactive_adjustment_handler(resume_adjustment)
            return False
    
    def _queue_for_later_execution(self, adjustment: Dict) -> None:
        """Add adjustment to pending queue for later execution"""
        # Create prioritized queue item
        priority = adjustment.get('priority', 3)
        queue_item = {
            'adjustment': adjustment,
            'priority': priority,
            'timestamp': time.time(),
            'retry_count': 0
        }
        
        # Add to priority queue (thread-safe)
        with self.lock:
            self.adjustment_queue.put((-priority, time.time(), queue_item))
        
        logger.info(f"Queued adjustment for later execution (priority: {priority})")
    
    def _enable_partial_execution(self, adjustment: Dict) -> None:
        """Enable partial execution mode for time-constrained tasks"""
        task_id = adjustment.get("task_id")
        if not task_id:
            return
        
        # Find task in current plan
        for task in self.current_plan:
            if task.id == task_id:
                # Enable partial execution mode
                if hasattr(task, 'execution_modes') and 'partial' in task.execution_modes:
                    adjustment['updates'] = adjustment.get('updates', {})
                    adjustment['updates']['execution_mode'] = 'partial'
                    try:
                        self.interactive_adjustment_handler(adjustment)
                        printer.status("PARTIAL-EXEC", "Enabled partial execution mode", "warning")
                    except Exception:
                        logger.error("Failed to enable partial execution")
                break
    
    def _escalate_to_human_operator(self, adjustment: Dict, error_msg: str) -> None:
        """Escalate failure to human operators"""
        alert_payload = {
            'type': 'adjustment_failure',
            'adjustment': adjustment,
            'error': error_msg,
            'timestamp': time.time(),
            'resource_status': self.resource_monitor.get_available_resources().__dict__,
            'plan_status': [t.id for t in self.current_plan]
        }
        
        # Send to monitoring dashboard
        self._send_alert_notification(alert_payload)
        printer.status("HUMAN-ESCALATE", "Adjustment requires human intervention", "critical")
    
    def _send_alert_notification(self, payload: Dict) -> None:
        """Send alert to monitoring system (stub implementation)"""
        # Actual implementation would use:
        # - Slack/Teams webhooks
        # - PagerDuty API
        # - Custom dashboard integration
        logger.error(f"SAFETY ALERT: {json.dumps(payload, indent=2)}")

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

    def log_adjustment(self, adjustment: Dict) -> None:
        """Comprehensive adjustment logging with contextual metadata"""
        if not adjustment:
            return
        
        log_entry = {
            'timestamp': time.time(),
            'type': adjustment.get('type'),
            'origin': adjustment.get('origin', 'api'),
            'status': 'applied',
            'adjustment': copy.deepcopy(adjustment),
            'plan_snapshot': {
                'task_ids': [t.id for t in self.current_plan],
                'resource_utilization': self._get_resource_utilization()
            },
            'perf_metrics': self._collect_performance_metrics()
        }
        
        # Remove sensitive data if present
        if 'credentials' in log_entry['adjustment']:
            log_entry['adjustment']['credentials'] = 'REDACTED'
        
        # Store in execution history
        with self.lock:
            self.base_state['execution_history'].append(log_entry)
        
        # Publish to monitoring stream
        self._publish_adjustment_event(log_entry)
        
        # Create audit checkpoint periodically
        if len(self.base_state['execution_history']) % self.config.audit_interval == 0:
            self.memory.save_checkpoint(
                label=f"audit_{int(time.time())}",
                metadata={'last_adjustment': log_entry}
            )
        
        logger.info(f"Logged adjustment: {adjustment.get('type')}")
    
    def _get_resource_utilization(self) -> Dict:
        """Capture resource utilization metrics"""
        resources = self.resource_monitor.get_available_resources()
        return {
            'gpu_utilization': f"{resources.gpu_allocated}/{resources.gpu_total}",
            'ram_utilization': f"{resources.ram_allocated}GB/{resources.ram_total}GB",
            'specialized_hardware': resources.specialized_hardware_available
        }
    
    def _collect_performance_metrics(self) -> Dict:
        """Collect real system performance metrics"""
        metrics = {
            'timestamp': time.time(),
            'system_load': self._get_system_load(),
            'network_latency': self._measure_network_latency(),
            'service_health': self._check_service_health(),
            'plan_execution_rate': len(self.base_state['execution_history']) / 60  # tasks/min
        }
        return metrics
    
    def _get_system_load(self) -> float:
        """Get system load average using psutil if available"""
        try:
            import psutil
            return psutil.getloadavg()[0]  # 1-minute load average
        except ImportError:
            # Fallback to dummy value
            return 0.75
    
    def _measure_network_latency(self) -> float:
        """Measure network latency to a control server"""
        try:
            target = self.config.get('monitoring', {}).get('latency_target', '8.8.8.8')
            start = time.perf_counter()
            requests.get(f"http://{target}", timeout=0.5)
            return (time.perf_counter() - start) * 1000  # ms
        except RequestException:
            return -1  # Error indicator
    
    def _check_service_health(self) -> Dict[str, str]:
        """Check health of critical services"""
        services = {
            'resource_monitor': self.resource_monitor,
            'distributed_orchestrator': self.distributed_orchestrator,
            'planning_memory': self.memory
        }
        return {name: "OK" if service else "DOWN" for name, service in services.items()}
    
    def _publish_adjustment_event(self, log_entry: Dict) -> None:
        """Publish adjustment event to monitoring systems"""
        event_type = f"adjustment.{log_entry['type']}"
        payload = {
            'event_type': event_type,
            'data': log_entry,
            'metadata': {
                'service': 'safety_planning',
                'version': self.config.version
            }
        }
        
        # Elasticsearch integration
        if self.config.get('elasticsearch', {}).get('enabled', False):
            try:
                from elasticsearch import Elasticsearch
                es = Elasticsearch(self.config['elasticsearch']['hosts'])
                es.index(
                    index=self.config['elasticsearch']['index'],
                    body=payload
                )
            except ImportError:
                logger.error("Elasticsearch client not available")
            except Exception as e:
                logger.error(f"Elasticsearch indexing failed: {str(e)}")
        
        # Kafka integration
        if self.config.get('kafka', {}).get('enabled', False):
            try:
                from kafka import KafkaProducer
                producer = KafkaProducer(
                    bootstrap_servers=self.config['kafka']['bootstrap_servers'],
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                producer.send(self.config['kafka']['topic'], payload)
            except ImportError:
                logger.error("Kafka producer not available")
            except Exception as e:
                logger.error(f"Kafka publish failed: {str(e)}")
        
        # Prometheus metrics
        if self.config.get('prometheus', {}).get('enabled', False):
            try:
                from prometheus_client import CollectorRegistry, Counter, push_to_gateway
                registry = CollectorRegistry()
                c = Counter('adjustment_events_total', 'Total adjustment events', 
                            ['event_type'], registry=registry)
                c.labels(event_type=event_type).inc()
                push_to_gateway(
                    self.config['prometheus']['pushgateway'],
                    job='safety_planning',
                    registry=registry
                )
            except ImportError:
                logger.error("Prometheus client not available")
            except Exception as e:
                logger.error(f"Prometheus push failed: {str(e)}")
        
        # Original debug logging
        if self.config.get('enable_event_logging'):
            logger.debug(f"PUBLISHED EVENT: {json.dumps(payload, indent=2)}")

    def update_allocations(self, task: Task) -> None:
        """Update resource allocations with safety checks and atomic operations"""
        with self.lock:
            requirements = task.resource_requirements
            available = self.resource_monitor.get_available_resources()
            
            # Validate before allocation
            self._validate_equipment_constraints(task)
            self._validate_safety_margins(task)
            
            # Perform atomic allocation
            try:
                # Deduct from global pool
                self.resource_monitor.cluster_resources.gpu_total -= requirements.gpu
                self.resource_monitor.cluster_resources.ram_total -= requirements.ram
                
                # Remove specialized hardware
                self.resource_monitor.cluster_resources.specialized_hardware_available = [
                    hw for hw in self.resource_monitor.cluster_resources.specialized_hardware_available 
                    if hw not in requirements.specialized_hardware
                ]
                
                # Track per-task allocation
                self.resource_monitor.current_allocations[task.id] = requirements
                logger.info(f"Allocated resources for task {task.id}: {requirements}")
                
            except Exception as e:
                logger.error(f"Resource allocation failed for {task.id}: {str(e)}")
                # Rollback changes
                self.resource_monitor.cluster_resources.gpu_total += requirements.gpu
                self.resource_monitor.cluster_resources.ram_total += requirements.ram
                raise ResourceViolation(
                    f"Allocation rollback for {task.id}",
                    requirements,
                    available
                )

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
        available = self.resource_monitor.get_available_resources()
        
        # Calculate safety buffers
        gpu_buffer = available.gpu * self.resource_buffers.gpu
        ram_buffer = available.ram * self.resource_buffers.ram
        specialized_buffer = self.resource_buffers.specialized_hardware
        
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
        if task.duration < self.temporal.min_task_duration:
            raise SafetyMarginError(
                f"Task duration too short for safe execution. "
                f"Current: {task.duration}s, Minimum: {self.temporal.min_task_duration}s",
                resource_type='temporal',
                buffer_amount=self.temporal.min_task_duration
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

class DistributedOrchestrator:
    """Manages distributed execution across multiple agents/nodes"""
    
    def __init__(self):
        self.config = load_global_config()
        self.do_config = get_config_section('decomposition')
        self.min_chunk_size = self.do_config.get('min_chunk_size')
        self.max_parallel_chunks = self.do_config.get('max_parallel_chunks')
        self.min_ram_per_chunk = self.do_config.get('min_ram_per_chunk')
        self.stage_time_buffer = self.do_config.get('stage_time_buffer')
        self.vertical_stages = self.do_config.get('vertical_stages', {
            'ABSTRACT', 'processing_pipeline'
        })

        self.memory = PlanningMemory()
        self.decomposition_strategies = {
            'horizontal': self.horizontal_split,
            'vertical': self.vertical_split,
            'hybrid': self.hybrid_split
        }

        logger.info(f"Distributed Orchestrator succesfully initialized with: {self.decomposition_strategies}")

    def horizontal_split(self, task: Task) -> List[Task]:
        """Data-parallel decomposition with dynamic chunk sizing"""
        try:
            # Calculate optimal split count based on data size and cluster resources
            data_size = len(task.input_data)
            chunk_size = max(
                self.min_chunk_size,
                data_size // self.memory.cluster_size
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
                    self.min_ram_per_chunk
                )
                subtasks.append(subtask)
            
            logger.info(f"Split task {task.id} into {len(subtasks)} horizontal chunks")
            return subtasks
            
        except (AttributeError, TypeError) as e:
            logger.error(f"Horizontal split failed for {task.id}: {str(e)}")
            return [task]

    def vertical_split(self, task: Task) -> List[Task]:
        """Functional decomposition into processing stages"""
        stages = getattr(self.vertical_stages, str(task.task_type), [])
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
                deadline=task.deadline - (len(stages) - stage_num) * self.stage_time_buffer
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
        self._node_cache = {}  # Cache last known good states
        self.config = load_global_config()
        self.resource_config = get_config_section('service_discovery')
        self.mode = self.resource_config.get('mode')
        self.static_nodes = self.resource_config.get('static_nodes')
        self.consul_url = self.resource_config.get('consul_url')
        self.k8s_api = self.resource_config.get('k8s_api')
        self.k8s_token = self.resource_config.get('k8s_token')
        self.node_port = self.resource_config.get('node_port')

        self.cluster_resources = ClusterResources()
        self.update_interval = 5  # seconds
        self.node_query_timeout = 2  # seconds
        self._lock = Lock()
        self._init_monitoring_thread()
        self.allocations = {}
        self.resource_graph = {}

    def get_available_resources(self) -> ClusterResources:
        """Return current available cluster resources after subtracting allocations."""
        with self._lock:
            allocated_gpu = 0
            allocated_ram = 0
            allocated_hw = set()
    
            for profile in self.cluster_resources.current_allocations.values():
                allocated_gpu += profile.gpu
                allocated_ram += profile.ram
                if profile.specialized_hardware:
                    allocated_hw.update(profile.specialized_hardware)
    
            available_gpu = self.cluster_resources.gpu_total - allocated_gpu
            available_ram = self.cluster_resources.ram_total - allocated_ram
            available_hw = [
                hw for hw in self.cluster_resources.specialized_hardware_available
                if hw not in allocated_hw
            ]
    
            return ClusterResources(
                gpu_total=available_gpu,
                ram_total=available_ram,
                specialized_hardware_available=available_hw,
                current_allocations={}  # Optional: omit or keep original for full state
            )

    def _init_monitoring_thread(self):
        def monitor_loop():
            time.sleep(0.1)  # short delay to ensure full object init
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
            if self.mode == 'consul':
                return self._query_consul_cluster()
            elif self.mode == 'k8s':
                return self._query_kubernetes_cluster()
            else:  # Default to static config
                return self.static_nodes
        except Exception as e:
            logger.error(f"Service discovery failed: {str(e)}")
            return list(self._node_cache.keys())  # Fallback to cached nodes

    def _query_consul_cluster(self):
        """Query Consul service discovery"""
        try:
            response = requests.get(
                f"{self.consul_url}/v1/catalog/nodes",
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
            headers = {"Authorization": f"Bearer {self.k8s_token}"}
            response = requests.get(
                f"{self.k8s_api}/api/v1/nodes",
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
                f"http://{node_id}:{self.node_port}/metrics",
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
    task = Task()
    task.resource_requirements = ResourceProfile(gpu=1, ram=16, specialized_hardware=[])
    task.deadline = time.time() + 3600  # Optional: Set a deadline
    task.dependencies = []  

    safety = SafetyPlanning()
    safety.current_plan = []
    orchestrator = DistributedOrchestrator()

    safety.interactive_adjustment_handler({})
    orchestrator.vertical_split(task)

    print(f"Selected action: {safety}")
    print(orchestrator)

    print("\n* * * * * Phase 2 * * * * *\n")
    monitor = ResourceMonitor()
    monitor._init_monitoring_thread()

    print(monitor)
    print("\n=== Successfully Ran Safety Planning ===\n")
