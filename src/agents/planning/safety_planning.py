
import random
import threading
import json, yaml
import time, copy
import requests

from dataclasses import field
from queue import PriorityQueue
from collections import defaultdict
from requests.exceptions import RequestException
from typing import Any, Callable, Dict, List, Optional, Union

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from src.agents.planning.utils.planning_errors import (AdjustmentError, ReplanningError, TemporalViolation,
                                                       SafetyMarginError, ResourceViolation)
from src.agents.planning.utils.planning_calculations import PlanningCalculations
from src.agents.planning.utils.resource_monitor import ResourceMonitor
from src.agents.planning.planning_types import (Task, TaskType, TaskStatus, ResourceProfile,
                                                ClusterResources, RepairCandidate, SafetyViolation)
from src.agents.planning.planning_memory import PlanningMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Safety Planning")
printer = PrettyPrinter

class SafetyPlanning:
    """Core module for safe distributed planning with interactive adjustments"""
    safety_margins: Dict[str, float] = field(default_factory=dict)
    violation_history: List[SafetyViolation] = field(default_factory=list)
    current_violations: List[SafetyViolation] = field(default_factory=list)
    safety_policies: Dict[str, Callable] = field(default_factory=dict)
    resource_monitor: ResourceMonitor = field(default_factory=ResourceMonitor)

    def __init__(self):
        self.config = load_global_config()
        self.ram_limit = self.config.get('ram_limit')
        self.gpu_limit = self.config.get('gpu_limit')

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
        self.calculations = PlanningCalculations()
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
                if task.name == task_id:
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

        #return self.distributed_orchestrator.decompose_and_distribute(
        #    task, self.agent.task_library, self.resource_monitor.current_loads)
        return self.distributed_orchestrator.decompose_and_distribute(task)

    def _reset_temporal_attributes(self, task: Task) -> None:
        """Reset temporal attributes to current time + buffer"""
        current_time = time.time()
        buffer = 60  # 1 minute buffer
        
        # Always reset start time to current time + buffer
        task.start_time = current_time + buffer
        
        # Reset deadline if it's invalid
        if getattr(task, 'deadline', 0) < task.start_time:
            if hasattr(task, 'duration'):
                task.deadline = task.start_time + task.duration
            else:
                task.deadline = task.start_time + 300  # Default 5 minutes

    def safety_check(self, plan: List[Task]) -> bool:
        """Comprehensive safety validation for plans"""
        # resource_usage = defaultdict(float)
        current_time = time.time()

        # Process each task in the plan
        for task in plan:

            # Handle missing temporal attributes
            if not hasattr(task, 'start_time') or task.start_time is None:
                task.start_time = current_time + 60  # Default 1min buffer

            if not hasattr(task, 'deadline') or task.deadline is None:
                task.deadline = task.start_time + 3600  # Default 1hr

            # Get safety margin values
            gpu_margin = self.margins_config.get('gpu_buffer', 0.15)
            ram_margin = self.margins_config.get('ram_buffer', 0.20)
            
            # GPU buffer check
            req = task.resource_requirements
            if req.gpu > self.gpu_limit * (1 - gpu_margin):
                violation = SafetyViolation(
                    violation_type="ResourceExceeded",
                    resource="gpu",
                    measured_value=req.gpu,
                    threshold=self.gpu_limit * (1 - gpu_margin),
                    task_id=task.id,
                    severity="high",
                    corrective_action="Reduce GPU usage"
                )
                self.current_violations.append(violation)
                return False
            
            # RAM buffer check
            if req.ram > self.ram_limit * (1 - ram_margin):
                violation = SafetyViolation(
                    violation_type="ResourceExceeded",
                    resource="ram",
                    measured_value=req.ram,
                    threshold=self.ram_limit * (1 - ram_margin),
                    task_id=task.id,
                    severity="high",
                    corrective_action="Reduce RAM usage"
                )
                self.current_violations.append(violation)
                return False
    
            # Validate temporal constraints for this task
            if task.start_time < current_time:
                task.start_time = current_time + 10
                raise TemporalViolation(f"Task {task.name} starts in past")
            if task.deadline and task.deadline < task.start_time:
                task.deadline = task.start_time + 300  # Default 5min duration
                raise TemporalViolation(f"Task {task.name} has invalid deadline")
        
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
                self.resource_monitor.current_allocations[task.name] = requirements  # Changed task.id -> task.name
                logger.info(f"Allocated resources for task {task.name}: {requirements}")  # Changed task.id -> task.name
                
            except Exception as e:
                logger.error(f"Resource allocation failed for {task.name}: {str(e)}")  # Changed task.id -> task.name
                # Rollback changes
                self.resource_monitor.cluster_resources.gpu_total += requirements.gpu
                self.resource_monitor.cluster_resources.ram_total += requirements.ram
                raise ResourceViolation(
                    f"Allocation rollback for {task.name}",  # Changed task.id -> task.name
                    requirements,
                    available
                )

    def _validate_temporal_constraints(self, task: Task) -> None:
        """Comprehensive temporal validation with dependency resolution and timeline projection"""
        try:
            # Initialize with safe defaults if attributes are missing
            current_time = time.time()
            task_deadline = getattr(task, 'deadline', current_time + 3600)  # Default 1hr deadline
            task_start = getattr(task, 'start_time', current_time + 60)  # Default 1min buffer
            task_duration = getattr(task, 'duration', 300)  # Default 5min duration
            dependencies = getattr(task, 'dependencies', [])
    
            # Timeline validation
            if task_start and task_start < time.time():
                raise TemporalViolation(
                    f"Task {getattr(task, 'name', 'unknown')} scheduled in past",
                    scheduled_time=task_start,
                    current_time=current_time
                )
            
            if task_deadline < (task_start + task_duration):
                raise TemporalViolation(
                    f"Insufficient time for task {getattr(task, 'name', 'unknown')}",
                    required_duration=task_duration,
                    available_window=task_deadline - task_start
                )
            
            # Dependency validation
            if dependencies:
                dependency_end_times = []
                for dep_id in dependencies:
                    dep_task = next((t for t in self.current_plan if getattr(t, 'id', None) == dep_id), None)
                    if dep_task:
                        dep_end = getattr(dep_task, 'end_time', getattr(dep_task, 'start_time', 0) + getattr(dep_task, 'duration', 0))
                        dependency_end_times.append(dep_end)
                    else:
                        logger.warning(f"Dependency task {dep_id} not found in current plan")
                
                if dependency_end_times:
                    latest_dep_end = max(dependency_end_times)
                    if task_start < latest_dep_end:
                        raise TemporalViolation(
                            f"Task {getattr(task, 'name', 'unknown')} starts before dependency completion",
                            dependency_end=latest_dep_end,
                            scheduled_start=task_start
                        )
            
            # Schedule conflict detection
            concurrent_tasks = [t for t in self.current_plan 
                               if getattr(t, 'start_time', 0) <= task_start <= getattr(t, 'end_time', 0)]
            if len(concurrent_tasks) > self.temporal.get('max_concurrent', 5):
                raise TemporalViolation(
                    f"Too many concurrent tasks at start time",
                    max_allowed=self.temporal.get('max_concurrent', 5),
                    scheduled_count=len(concurrent_tasks)
                )
                
        except Exception as e:
            logger.error(f"Temporal validation failed: {str(e)}")
            raise
    
    def _validate_safety_margins(self, task: Task) -> None:
        """Resource buffer validation with dynamic margin calculation"""
        try:
            # Get available resources with fallback
            available = self.resource_monitor.get_available_resources() if self.resource_monitor else ClusterResources()
            
            # Handle and convert resource requirements
            if not hasattr(task, 'resource_requirements') or not isinstance(task.resource_requirements, ResourceProfile):
                logger.warning(f"Task {getattr(task, 'name', 'unknown')} has invalid resource requirements, converting to ResourceProfile")
                task.resource_requirements = ResourceProfile()
                
            req = task.resource_requirements
            
            # Calculate safety buffers
            gpu_buffer = self.resource_buffers.get('gpu', 0.1) * getattr(available, 'gpu_total', 1)
            ram_buffer = self.resource_buffers.get('ram', 0.1) * getattr(available, 'ram_total', 8)
            hw_buffer = self.resource_buffers.get('specialized_hardware', [])
            
            # GPU margin validation
            if getattr(req, 'gpu', 0) > 0:
                available_gpu = getattr(available, 'gpu_total', 0) - getattr(available, 'gpu_allocated', 0)
                if (available_gpu - req.gpu) < gpu_buffer:
                    raise SafetyMarginError(
                        f"GPU allocation violates safety buffer",
                        resource_type='gpu',
                        requested=req.gpu,
                        available=available_gpu,
                        buffer_needed=gpu_buffer
                    )
            
            # RAM margin validation
            if getattr(req, 'ram', 0) > 0:
                available_ram = getattr(available, 'ram_total', 0) - getattr(available, 'ram_allocated', 0)
                if (available_ram - req.ram) < ram_buffer:
                    raise SafetyMarginError(
                        f"RAM allocation violates safety buffer",
                        resource_type='ram',
                        requested=req.ram,
                        available=available_ram,
                        buffer_needed=ram_buffer
                    )
            
            # Specialized hardware validation
            required_hw = set(getattr(req, 'specialized_hardware', []) or [])
            if required_hw:
                available_hw = set(getattr(available, 'specialized_hardware_available', []) or [])
                reserved_hw = available_hw - set(hw_buffer)
                
                if not required_hw.issubset(reserved_hw):
                    missing = required_hw - reserved_hw
                    raise SafetyMarginError(
                        f"Specialized hardware buffer violated",
                        resource_type='specialized_hardware',
                        missing_hardware=list(missing),
                        buffer_hardware=hw_buffer
                    )
            
            # Temporal safety buffer
            min_duration = self.temporal.get('min_task_duration', 10)
            if getattr(task, 'duration', 0) < min_duration:
                raise SafetyMarginError(
                    f"Task duration too short for safe execution",
                    resource_type='temporal',
                    current_duration=getattr(task, 'duration', 0),
                    minimum_duration=min_duration
                )
                
        except Exception as e:
            logger.error(f"Safety margin validation failed: {str(e)}")
            raise
    
    def _validate_equipment_constraints(self, task: Task) -> None:
        """Resource availability validation with comprehensive error handling"""
        try:
            # Handle and convert resource requirements
            if not hasattr(task, 'resource_requirements') or not isinstance(task.resource_requirements, ResourceProfile):
                logger.warning(f"Task {getattr(task, 'name', 'unknown')} has invalid resource requirements, converting to ResourceProfile")
                task.resource_requirements = ResourceProfile()
                
            req = task.resource_requirements
            available = self.resource_monitor.get_available_resources() if self.resource_monitor else ClusterResources()
            
            # GPU validation
            if getattr(req, 'gpu', 0) > 0:
                available_gpu = getattr(available, 'gpu_total', 0) - getattr(available, 'gpu_allocated', 0)
                if req.gpu > available_gpu:
                    raise ResourceViolation(
                        f"Insufficient GPU resources",
                        'gpu',
                        req.gpu,
                        available_gpu
                    )
            
            # RAM validation
            if getattr(req, 'ram', 0) > 0:
                available_ram = getattr(available, 'ram_total', 0) - getattr(available, 'ram_allocated', 0)
                if req.ram > available_ram:
                    raise ResourceViolation(
                        f"Insufficient RAM resources",
                        'ram',
                        req.ram,
                        available_ram
                    )
            
            # Specialized hardware validation
            required_hw = set(getattr(req, 'specialized_hardware', []) or [])
            if required_hw:
                available_hw = set(getattr(available, 'specialized_hardware_available', []) or [])
                missing_hw = required_hw - available_hw
                
                if missing_hw:
                    raise ResourceViolation(
                        f"Missing specialized hardware",
                        'specialized_hardware',
                        list(missing_hw),
                        list(available_hw)
                    )
                    
            # Edge case: No resource requirements
            if not any([req.gpu, req.ram, required_hw]):
                logger.info(f"Task {getattr(task, 'name', 'unknown')} has no resource requirements")
                
        except Exception as e:
            logger.error(f"Equipment validation failed: {str(e)}")
            raise

    # Additional Core Methods
    def dynamic_replanning_pipeline(self, failed_task: Task) -> List[Task]:
        """Full safety-aware replanning workflow"""
        alternatives = []
        
        # Strategy 1: Reduce resource requirements
        reduced_resource_task = failed_task.copy()
        reduced_resource_task.resource_requirements.gpu *= 0.8
        reduced_resource_task.resource_requirements.ram *= 0.8
        self._reset_temporal_attributes(reduced_resource_task)  # Reset temporal attributes
        if self.safety_check([reduced_resource_task]):
            alternatives.append([reduced_resource_task])
            
        # Strategy 2: Alternative method with lower requirements
        if failed_task.task_type == TaskType.ABSTRACT:
            for method_idx in range(len(failed_task.methods)):
                alt_task = failed_task.copy()
                alt_task.selected_method = method_idx
                self._reset_temporal_attributes(alt_task)  # Reset temporal attributes
                if self.safety_check([alt_task]):
                    alternatives.append(alt_task.get_subtasks())

        with self.lock:
            try:
                candidates = self._generate_repair_candidates(failed_task)
                validated = [
                    c for c in candidates
                    if self.safety_check(c.repaired_plan)
                ]
                if not validated:
                    return None
                return self._select_optimal_repair(validated)
            except ReplanningError as e:
                logger.error(f"Emergency shutdown triggered: {str(e)}")
                self._emergency_shutdown_procedure()
        
        return None or alternatives

    def _generate_repair_candidates(self, failed_task: Task) -> List['RepairCandidate']:
        """
        Generates multiple repair candidates for a failed task using various strategies.
        Implements academic principles from task repair literature and distributed systems.
        """
        candidates = []
        logger.info(f"Generating repair candidates for failed task: {failed_task.name}")

        # Strategy 1: Simple Retry (with resource boost)
        try:
            retry_task = copy.deepcopy(failed_task)
            retry_task.status = TaskStatus.PENDING
            self._reset_temporal_attributes(retry_task) 

            # Apply resource boost for retry attempts
            available = self.resource_monitor.get_available_resources()
            retry_task.resource_requirements.gpu = min(
                retry_task.resource_requirements.gpu * 1.5,
                available.gpu_total
            )
            retry_task.resource_requirements.ram = min(
                retry_task.resource_requirements.ram * 1.2,
                available.ram_total
            )
            
            repair_plan = self._create_repair_plan(failed_task, [retry_task])
            candidates.append(RepairCandidate(
                strategy="retry_with_boost",
                repaired_plan=repair_plan,
                estimated_cost=self._estimate_repair_cost(retry_task),
                risk_assessment=self._assess_repair_risk(retry_task, "retry")
            ))

        except Exception as e:
            logger.error(f"Retry candidate generation failed: {str(e)}", exc_info=True)

        # Strategy 2: Alternative Method Execution
        try:
            if failed_task.task_type == TaskType.ABSTRACT and failed_task.methods:
                for method_idx in range(len(failed_task.methods)):
                    if method_idx == failed_task.selected_method:
                        continue  # Skip the failed method
                    
                    subtasks = failed_task.get_subtasks(method_idx)
                    repair_plan = self._create_repair_plan(failed_task, subtasks)
                    
                    candidates.append(RepairCandidate(
                        strategy=f"alt_method_{method_idx}",
                        repaired_plan=repair_plan,
                        estimated_cost=self._estimate_repair_cost(subtasks),
                        risk_assessment=self._assess_repair_risk(subtasks, "alt_method")
                    ))
        except Exception as e:
            logger.error(f"Alternative method candidate failed: {str(e)}")

        # Strategy 3: Task Decomposition
        try:
            decomposed = self.distributed_orchestrator.decompose_and_distribute(failed_task)
            if decomposed:
                # Reset temporal attributes for all subtasks
                for task in decomposed:
                    self._reset_temporal_attributes(task)
                repair_plan = self._create_repair_plan(failed_task, decomposed)
                candidates.append(RepairCandidate(
                    strategy="distributed_decomposition",
                    repaired_plan=repair_plan,
                    estimated_cost=self._estimate_repair_cost(decomposed),
                    risk_assessment=self._assess_repair_risk(decomposed, "decomposition")
                ))
        except Exception as e:
            logger.error(f"Decomposition candidate failed: {str(e)}", exc_info=True)

        # Strategy 4: Partial Execution Mode
        try:
            if 'partial' in getattr(failed_task, 'execution_modes', []):
                partial_task = copy.deepcopy(failed_task)
                partial_task.execution_mode = 'partial'
                self._reset_temporal_attributes(partial_task)
                repair_plan = self._create_repair_plan(failed_task, [partial_task])
                
                candidates.append(RepairCandidate(
                    strategy="partial_execution",
                    repaired_plan=repair_plan,
                    estimated_cost=self._estimate_repair_cost(partial_task),
                    risk_assessment=self._assess_repair_risk(partial_task, "partial")
                ))
        except Exception as e:
            logger.error(f"Partial execution candidate failed: {str(e)}", exc_info=True)

        # Strategy 5: Resource Reallocation
        try:
            if self._can_reallocate_resources(failed_task):
                reallocated_task = copy.deepcopy(failed_task)
                self._reset_temporal_attributes(reallocated_task)
                victims = self._find_resource_victims(failed_task)
                
                repair_plan = self._create_repair_plan(failed_task, [reallocated_task], victims)
                
                candidates.append(RepairCandidate(
                    strategy="resource_reallocation",
                    repaired_plan=repair_plan,
                    estimated_cost=self._estimate_repair_cost(reallocated_task, victims),
                    risk_assessment=self._assess_repair_risk(reallocated_task, "reallocation", victims)
                ))
        except Exception as e:
            logger.error(f"Resource reallocation candidate failed: {str(e)}", exc_info=True)

        # Fallback strategy: Skip and Replan
        try:
            pruned_plan = self._create_pruned_plan(self.current_plan, failed_task)
            candidates.append(RepairCandidate(
                strategy="skip_and_replan",
                repaired_plan=pruned_plan,
                estimated_cost=0,  # Cost will come from later replanning
                risk_assessment={
                    "risk_score": 0.9,
                    "risk_level": "high",
                    "details": "Task skipped, requires manual review"
                }
            ))
        except Exception as e:
            logger.error(f"Skip candidate failed: {str(e)}", exc_info=True)

        logger.info(f"Generated {len(candidates)} repair candidates")
        return candidates
    
    def _create_pruned_plan(self, original_plan: List[Task], failed_task: Task) -> List[Task]:
        """
        Generates a pruned version of the original plan by removing the failed task and its dependents.
        """
        if failed_task not in original_plan:
            logger.warning(f"Failed task '{failed_task.name}' not in original plan")
            return original_plan
    
        # Build a dependency graph if needed
        dependents = self._get_dependent_tasks(original_plan, failed_task)
    
        pruned_plan = [t for t in original_plan if t != failed_task and t not in dependents]
    
        logger.info(f"Pruned plan: removed {1 + len(dependents)} tasks")
        return pruned_plan
    
    def _get_dependent_tasks(self, plan: List[Task], task: Task) -> List[Task]:
        """
        Recursively finds all tasks in the plan that depend on the given task.
        """
        dependents = []
        for t in plan:
            if task.id in t.dependencies:
                dependents.append(t)
                dependents += self._get_dependent_tasks(plan, t)
        return dependents

    def _create_repair_plan(self, failed_task: Task, replacement_tasks: List[Task], 
                               victims: List[Task] = None) -> List[Task]:
            """Creates a repaired plan by replacing the failed task with new tasks"""
            new_plan = []
            replaced = False
            victims = victims or []
            
            for task in self.current_plan:
                if task.id == failed_task.id:  # or task.name == failed_task.name: 
                    new_plan.extend(replacement_tasks)
                    replaced = True
                elif task in victims:
                    # Pause victim tasks for resource reallocation
                    paused_task = copy.deepcopy(task)
                    paused_task.status = TaskStatus.PENDING
                    paused_task.priority = min(paused_task.priority, 1)  # Demote priority
                    new_plan.append(paused_task)
                else:
                    # Copy existing tasks, updating dependencies if needed
                    new_task = copy.deepcopy(task)
                    
                    # Update dependencies if they point to the failed task
                    if failed_task.id in new_task.dependencies:
                        new_task.dependencies = [
                            dep for dep in new_task.dependencies if dep != failed_task.id
                        ] + [t.id for t in replacement_tasks]
                    
                    new_plan.append(new_task)
            
            if not replaced:
                logger.warning(f"Failed task {failed_task.id} not in current plan. Creating new plan from replacements.")
                return replacement_tasks
            
            return new_plan

    def _select_optimal_repair(self, candidates: List['RepairCandidate']) -> List[Task]:
        """
        Selects the optimal repair candidate using a multi-criteria decision analysis approach.
        Considers cost, risk, resource usage, and temporal constraints.
        """
        if not candidates:
            raise ReplanningError("No valid repair candidates available", None)
        
        logger.info(f"Selecting optimal repair from {len(candidates)} candidates")
        
        # Score each candidate using weighted factors
        scored_candidates = []
        weights = self.config.get('replanning_weights', {
            'cost': 0.4,
            'risk': 0.3,
            'time': 0.2,
            'resource': 0.1
        })
        
        for candidate in candidates:
            # Normalize scores (lower is better for cost and risk)
            cost_score = self._normalize_cost(candidate.estimated_cost)
            risk_score = self._normalize_risk(candidate.risk_assessment)
            time_score = self._estimate_time_efficiency(candidate.repaired_plan)
            resource_score = self._assess_resource_efficiency(candidate.repaired_plan)
            
            # Calculate composite score
            composite_score = (
                weights['cost'] * cost_score +
                weights['risk'] * risk_score +
                weights['time'] * time_score +
                weights['resource'] * resource_score
            )
            
            scored_candidates.append((composite_score, candidate))
        
        # Select candidate with highest composite score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_candidate = scored_candidates[0][1]
        
        logger.info(f"Selected repair strategy: {best_candidate.strategy} "
                   f"(score: {scored_candidates[0][0]:.2f})")
        
        return best_candidate.repaired_plan

    # Helper methods for candidate evaluation
    def _estimate_repair_cost(self, tasks: Union[Task, List[Task]], victims: List[Task] = None) -> float:
        """Estimates the cost of a repair operation"""
        if not isinstance(tasks, list):
            tasks = [tasks]
        
        victim_cost = sum(v.cost for v in (victims or [])) if victims else 0
        task_cost = sum(t.cost for t in tasks)
        
        # Add overhead cost based on strategy complexity
        overhead = self.config.get('repair_overheads', {
            'retry': 0.1,
            'alt_method': 0.2,
            'decomposition': 0.3,
            'partial': 0.4,
            'reallocation': 0.5
        }).get(tasks[0].strategy if hasattr(tasks[0], 'strategy') else 'unknown', 0.3)
        
        return (task_cost + victim_cost) * (1 + overhead)

    def _assess_repair_risk(self, tasks: Union[Task, List[Task]], strategy: str, 
                           victims: List[Task] = None) -> Dict[str, Any]:
        """Assesses the risk of a repair strategy using historical data"""
        risk_factors = {
            'success_rate': self.memory.get_method_success_rate(strategy),
            'resource_margin': self.calculations.calculate_resource_margin(tasks),
            'temporal_margin': self.calculations.calculate_temporal_margin(tasks),
            'dependency_risk': self.calculations.calculate_dependency_risk(tasks),
            'victim_impact': len(victims) if victims else 0
        }
        
        # Calculate composite risk score (0-1, higher is riskier)
        risk_score = min(1.0, (
            0.4 * (1 - risk_factors['success_rate']) +
            0.3 * risk_factors['resource_margin'] +
            0.2 * risk_factors['temporal_margin'] +
            0.1 * risk_factors['dependency_risk'] +
            0.05 * min(1.0, risk_factors['victim_impact'] / 5.0)
        ))

        try:
            success_rate = self.memory.get_method_success_rate(strategy)
        except AttributeError:
            success_rate = 0.5
        
        return {
            'risk_score': risk_score,
            'factors': risk_factors,
            'risk_level': 'low' if risk_score < 0.3 else 'medium' if risk_score < 0.6 else 'high'
        }

    def _can_reallocate_resources(self, task: Task) -> bool:
        """Determines if resource reallocation is possible"""
        required_gpu = task.resource_requirements.gpu
        required_ram = task.resource_requirements.ram
        
        # Check if lower priority tasks exist that could free resources
        return any(
            t for t in self.current_plan 
            if t.priority > task.priority and 
            t.id != task.id and
            t.status == TaskStatus.EXECUTING and
            (t.resource_requirements.gpu >= required_gpu or 
             t.resource_requirements.ram >= required_ram)
        )

    def _find_resource_victims(self, task: Task) -> List[Task]:
        """Finds lower priority tasks to pause for resource reallocation"""
        required_gpu = task.resource_requirements.gpu
        required_ram = task.resource_requirements.ram
        candidates = []
        
        # Find suitable candidates to pause
        for t in self.current_plan:
            if (t.priority > task.priority and 
                t.id != task.id and
                t.status == TaskStatus.EXECUTING):
                
                # Check if this task could free sufficient resources
                if (t.resource_requirements.gpu >= required_gpu and
                    t.resource_requirements.ram >= required_ram):
                    return [t]  # Found a single victim
                
                # Otherwise collect partial matches
                if (t.resource_requirements.gpu > 0 or 
                    t.resource_requirements.ram > 0):
                    candidates.append(t)
        
        # Try to find a combination of victims
        gpu_total = 0
        ram_total = 0
        selected = []
        
        for t in sorted(candidates, key=lambda x: x.priority):
            if gpu_total < required_gpu or ram_total < required_ram:
                gpu_total += t.resource_requirements.gpu
                ram_total += t.resource_requirements.ram
                selected.append(t)
        
        if gpu_total >= required_gpu and ram_total >= required_ram:
            return selected
        
        return []  # No suitable combination found

    # Helper methods for optimal selection
    def _normalize_cost(self, cost: float) -> float:
        """Normalizes cost to a 0-1 scale (1 is best)"""
        max_cost = self.config.get('replanning_max_cost', 100.0)
        return max(0, 1 - min(cost / max_cost, 1.0))

    def _normalize_risk(self, risk_assessment: Dict) -> float:
        """Normalizes risk to a 0-1 scale (1 is best)"""
        return 1 - risk_assessment.get('risk_score', 1.0)

    def _estimate_time_efficiency(self, plan: List[Task]) -> float:
        """Estimates the time efficiency of a plan (0-1, higher is better)"""
        total_time = sum(t.duration for t in plan if hasattr(t, 'duration'))
        min_possible = sum(self.memory.get_min_duration(t.name) for t in plan)
        
        if min_possible == 0:
            return 0.5  # Neutral score if no data
        
        return min(1.0, min_possible / total_time)

    def _assess_resource_efficiency(self, plan: List[Task]) -> float:
        """Assesses resource utilization efficiency (0-1, higher is better)"""
        total_gpu = sum(t.resource_requirements.gpu for t in plan)
        total_ram = sum(t.resource_requirements.ram for t in plan)
        
        # Get available cluster resources
        cluster = self.resource_monitor.get_available_resources()
        max_gpu = cluster.gpu_total
        max_ram = cluster.ram_total
        
        if max_gpu == 0 or max_ram == 0:
            return 0.5  # Neutral score if no resource data
        
        gpu_efficiency = total_gpu / max_gpu
        ram_efficiency = total_ram / max_ram
        
        # Handle division by zero cases
        if gpu_efficiency == 0 and ram_efficiency == 0:
            return 0.0
        if gpu_efficiency == 0 or ram_efficiency == 0:
            return max(gpu_efficiency, ram_efficiency)
        
        # Use harmonic mean for balanced efficiency
        return 2 * (gpu_efficiency * ram_efficiency) / (gpu_efficiency + ram_efficiency) 

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
    
    def decompose_and_distribute(self, task: Task) -> Optional[List[Task]]:
        """
        Decomposes a complex task and distributes its components across available agents or nodes.
        """
        if not task.methods:
            logger.warning(f"Using default decomposition for {task.name}")
            return [task]
    
        for method in task.methods:
            try:
                method_idx = 0
                #subtasks = method.get_subtasks(method_idx)
                #logger.info(f"Decomposed '{task.name}' into {len(subtasks)} subtasks")

                subtasks = method_list
                if not subtasks:
                    continue

                logger.info(f"Decomposed '{task.name}' into {len(subtasks)} subtasks using one of its methods.")
    
                # Distribute subtasks across compute nodes (this is pseudo-coded for orchestration)
                distributed = []
                for subtask_template in subtasks:
                    #assigned_node = self.resource_allocator.select_node(subtask)
                    #subtask.assigned_node = assigned_node
                    new_task = subtask_template.copy()
                    new_task.id = f"{subtask_template.name}_{int(time.time()*1000)}_{random.randint(0,999)}"
                    new_task.parent = task
                    distributed.append(new_task)

                return distributed
            except Exception as e:
                logger.error(f"Decomposition failed: {str(e)}")
                continue
    
        logger.error(f"Failed to decompose and distribute task '{task.name}'")
        return None

# ====================== Usage Example ======================
if __name__ == "__main__":
    print("\n=== Running Safety Planning ===\n")
    adjustment = None
    task = Task()
    task.resource_requirements = ResourceProfile(gpu=1, ram=16, specialized_hardware=[])
    task.deadline = time.time() + 3600
    task.dependencies = []  

    safety = SafetyPlanning()
    safety.current_plan = []
    orchestrator = DistributedOrchestrator()

    safety.interactive_adjustment_handler({})
    orchestrator.vertical_split(task)

    print(f"Selected action: {safety}")
    print(orchestrator)

    print("\n* * * * * Phase 2 * * * * *\n") 
    failed_task = Task(
        name="evacuate_building",
        id="task_evacuate",
        task_type=TaskType.PRIMITIVE,
        status=TaskStatus.FAILED,
        resource_requirements=ResourceProfile(gpu=1, ram=4),
        start_time=time.time() + 60,  # Start in 1 minute
        deadline=time.time() + 300,   # 5 minutes from now
        duration=200
    )

    safety.current_plan = [failed_task]
    
    pipeline = safety.dynamic_replanning_pipeline(failed_task=failed_task)

    printer.pretty("REPLAN", pipeline, "success" if pipeline else "error")
    print("\n=== Successfully Ran Safety Planning ===\n")
