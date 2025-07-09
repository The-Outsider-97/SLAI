
import numpy as np
import time
import yaml, json

from typing import Dict, Any, List, Optional, Callable, Union
from abc import ABC, abstractmethod
from collections import defaultdict, deque

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Task Scheduler")
printer = PrettyPrinter

class TaskScheduler(ABC):
    """Abstract scheduler interface"""
    def __init__(self):
        super().__init__()
        logger.info(f"Abstract scheduler interface succesfully initialized")
        
    @abstractmethod
    def schedule(self,
                 tasks: List[Dict],
                 agents: Dict[str, Any],
                 risk_assessor: Optional[Callable] = None,
                 state: Optional[Dict] = None) -> Dict:
        return

class DeadlineAwareScheduler(TaskScheduler):
    """Earliest Deadline First scheduler with capability matching"""
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.task_config = get_config_section('task_scheduler')

        self.risk_threshold = self.task_config.get('risk_threshold')
        self.base_duration_per_requirement = self.task_config.get('base_duration_per_requirement')
        self.efficiency_attribute = self.task_config.get('efficiency_attribute')
        self.retry_policy = self.task_config.get('retry_policy', {
            'max_retries', 'max_attempts', 'backoff_factor', 'delay'
        })
        self.agent = {}
        self.task_history = defaultdict(list)

        logger.info(f"Task Scheduler succesfully initialized with: {self.task_history}")

    def schedule(self,
                 tasks: List[Dict],
                 agents: Dict[str, Any],
                 risk_assessor: Optional[Callable] = None,
                 state: Optional[Dict] = None) -> Dict:

        """Main scheduling workflow with integrated risk checks"""
        printer.status("INIT", "Schedule succesfully initialized", "info")

        self.agents = agents
        self.state = state
        self.task_history["current"] = tasks
        validated = self._validate_inputs(tasks, agents)
        if not validated:
            return {}

        # Phase 1: Risk assessment and prioritization
        prioritized = self._prioritize_tasks(tasks, risk_assessor)

        # Phase 2: Agent capability matching
        candidate_map = self._map_capabilities(prioritized, agents)

        # Phase 3: Temporal scheduling with dependencies
        schedule = self._create_schedule(candidate_map, agents, state)

        # Phase 4: Risk mitigation and fallback planning
        return self._apply_risk_mitigation(schedule, risk_assessor)

    def _validate_inputs(self, tasks, agents):
        """Robust validation with diagnostics, type checks, and safe defaults"""
        printer.status("VALIDATION", "Input validation initialized", "info")
        errors = []
        warnings = []

        # Top-level structure validation
        if not isinstance(tasks, list):
            errors.append("Tasks must be a list")
        if not isinstance(agents, dict):
            errors.append("Agents must be a dictionary")
        if errors:
            logger.error(f"Validation failed: {'; '.join(errors)}")
            return False

        # Task validation with detailed diagnostics
        valid_task_ids = set()
        for idx, task in enumerate(tasks):
            task_errors = []
            task_warnings = []
            task_id = task.get('id', f"Unidentified task at index {idx}")

            # ID validation
            if 'id' not in task:
                task_errors.append("Missing 'id' field")
            elif not isinstance(task['id'], str):
                task_errors.append("'id' must be a string")
            elif not task['id'].strip():
                task_errors.append("'id' cannot be empty")
            elif task['id'] in valid_task_ids:
                task_errors.append(f"Duplicate task ID: {task['id']}")
            else:
                valid_task_ids.add(task['id'])

            # Requirements validation
            if 'requirements' not in task:
                task_warnings.append("Missing 'requirements', using default []")
                task['requirements'] = []
            elif not isinstance(task['requirements'], list):
                task_errors.append("'requirements' must be a list")
            else:
                for i, req in enumerate(task['requirements']):
                    if not isinstance(req, str):
                        task_errors.append(f"Requirement {i} must be string")
                    elif not req.strip():
                        task_errors.append(f"Requirement {i} cannot be empty")

            # Deadline validation
            current_time = time.time()
            if 'deadline' not in task:
                task_warnings.append("Missing 'deadline', using default (now + 300s)")
                task['deadline'] = current_time + 300
            elif not isinstance(task['deadline'], (int, float)):
                task_errors.append("'deadline' must be numeric")
            else:
                if task['deadline'] < current_time:
                    task_warnings.append("Deadline is in the past")
                elif task['deadline'] < current_time + 10:
                    task_warnings.append("Deadline is too imminent (<10s)")

            # Dependency validation
            if 'dependencies' in task:
                if not isinstance(task['dependencies'], list):
                    task_errors.append("'dependencies' must be a list")
                else:
                    for dep in task['dependencies']:
                        if not isinstance(dep, str):
                            task_errors.append("Dependency must be string")
                        elif dep not in valid_task_ids:
                            task_warnings.append(f"Unknown dependency: {dep}")

            # Collect diagnostics
            if task_errors:
                errors.append(f"Task {task_id}: {'; '.join(task_errors)}")
            if task_warnings:
                warnings.append(f"Task {task_id}: {'; '.join(task_warnings)}")

        # Agent validation with capability checks
        valid_agent_ids = set()
        for agent_id, details in agents.items():
            agent_errors = []
            agent_warnings = []

            if not isinstance(details, dict):
                agent_errors.append("Agent details must be a dictionary")
            else:
                # Capability validation
                if 'capabilities' not in details:
                    agent_warnings.append("Missing 'capabilities', using default []")
                    details['capabilities'] = []
                elif not isinstance(details['capabilities'], list):
                    agent_errors.append("'capabilities' must be a list")
                else:
                    for i, cap in enumerate(details['capabilities']):
                        if not isinstance(cap, str):
                            agent_errors.append(f"Capability {i} must be string")
                        elif not cap.strip():
                            agent_errors.append(f"Capability {i} cannot be empty")

                # Load validation
                if 'current_load' not in details:
                    agent_warnings.append("Missing 'current_load', using default 0.0")
                    details['current_load'] = 0.0
                elif not isinstance(details['current_load'], (int, float)):
                    agent_errors.append("'current_load' must be numeric")
                elif details['current_load'] < 0:
                    agent_warnings.append("Negative load reset to 0.0")
                    details['current_load'] = max(0.0, details['current_load'])
                elif details['current_load'] > 1.5:
                    agent_warnings.append("Extremely high load (>1.5)")

                # Performance metrics
                for metric in ['successes', 'failures']:
                    if metric in details and not isinstance(details[metric], int):
                        agent_errors.append(f"'{metric}' must be integer")

                # Efficiency validation
                eff_attr = self.efficiency_attribute
                if eff_attr in details and not isinstance(details[eff_attr], (int, float)):
                    agent_errors.append(f"'{eff_attr}' must be numeric")

            # Collect diagnostics
            if agent_errors:
                errors.append(f"Agent {agent_id}: {'; '.join(agent_errors)}")
            if agent_warnings:
                warnings.append(f"Agent {agent_id}: {'; '.join(agent_warnings)}")
            else:
                valid_agent_ids.add(agent_id)

        # Final availability check
        if not valid_task_ids:
            errors.append("No valid tasks after validation")
        if not valid_agent_ids:
            errors.append("No valid agents after validation")

        # Diagnostic reporting
        for warning in warnings:
            logger.warning(warning)
        for error in errors:
            logger.error(error)

        # Create validation report
        report = {
            'valid_tasks': len(valid_task_ids),
            'valid_agents': len(valid_agent_ids),
            'errors': len(errors),
            'warnings': len(warnings)
        }
        printer.status("VALIDATION", f"Validation report: {report}", 
                      "success" if not errors else "error")

        return not errors and valid_task_ids and valid_agent_ids

    def _prioritize_tasks(self, tasks, risk_assessor):
        """Risk-aware prioritization using collaborative agent's assessment"""
        printer.status("INIT", "Risk-aware prioritization succesfully initialized", "info")

        prioritized = []
        for task in tasks:
            risk_score = 0.5
            if risk_assessor:
                assessment = risk_assessor(task) if risk_assessor else {} # or (task.get('risk_score', 0.5))
                risk_score = assessment.get('risk_score', 0.5)
                task['risk_assessment'] = assessment
                
            priority = self._calculate_priority(
                task['deadline'],
                task.get('priority', 3),
                risk_score
            )
            prioritized.append((priority, task))
            
        return [t for _, t in sorted(prioritized, reverse=True)]

    def _calculate_priority(self, deadline, base_priority, risk_score):
        """Hybrid priority calculation"""
        printer.status("INIT", "Priority calculation succesfully initialized", "info")

        time_criticality = 1 / (deadline - time.time() + 0.000001)
        risk_penalty = np.clip(risk_score - self.risk_threshold, 0, 1)
        return (0.6 * base_priority + 
                0.3 * time_criticality - 
                0.1 * risk_penalty)

    def _map_capabilities(self, tasks, agents):
        """Capability matching with load awareness"""
        printer.status("INIT", "Map capabilities succesfully initialized", "info")

        candidate_map = defaultdict(list)
        for task in tasks:
            for agent_id, details in agents.items():
                if self._agent_is_eligible(agent_id, task, details):
                    score = self._calculate_agent_score(agent_id, task, details)
                    candidate_map[task['id']].append((agent_id, score))
        return candidate_map

    def _apply_risk_mitigation(self, schedule, risk_assessor):
        """Apply collaborative agent's safety recommendations"""
        printer.status("INIT", "Risk mitigation succesfully initialized", "info")

        mitigated = {}
        for task_id, assignment in schedule.items():
            if risk_assessor and assignment['risk_score'] > self.risk_threshold:
                alt = self._find_alternative(
                    task_id, 
                    assignment,
                    schedule,
                    risk_assessor
                )
                if alt:
                    mitigated[task_id] = alt
                    continue
            mitigated[task_id] = assignment
        return mitigated
    
    def _find_alternative(self, task_id, assignment, schedule, risk_assessor):
        """Find safer alternatives for high-risk tasks through reassignment or decomposition"""
        printer.status("ALT", f"Seeking alternatives for high-risk task {task_id}", "warning")
        
        # 1. Try agent reassignment first
        original_agent = assignment['agent_id']
        task = next(t for t in self.task_history["current"] if t["id"] == task_id)
        
        # Get all eligible agents including previously rejected ones
        eligible_agents = self._map_capabilities([task], self.agents)[task_id]
        
        # Filter and rank agents by risk-adjusted score
        candidates = []
        for agent_id, raw_score in eligible_agents:
            if agent_id == original_agent:
                continue  # Skip original assignment
                
            # Calculate risk-adjusted score
            agent_details = self.agents[agent_id]
            risk_factor = self._calculate_agent_risk(task, agent_details)
            adjusted_score = raw_score * (1 - risk_factor)
            
            candidates.append((adjusted_score, agent_id, agent_details))
        
        # Sort candidates by descending risk-adjusted score
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Select best viable alternative
        for score, agent_id, details in candidates:
            if score > 0:  # Viable candidate
                new_assignment = self._create_assignment(
                    task_id, agent_id, details, 
                    details["current_load"], self.state
                )
                new_assignment["mitigation_strategy"] = f"Reassigned from {original_agent}"
                printer.status("ALT-SUCCESS", 
                              f"Found safer agent {agent_id} (risk: {new_assignment['risk_score']:.2f})",
                              "success")
                return new_assignment
        
        # 2. Fallback to task decomposition
        printer.status("ALT-FALLBACK", 
                      f"No agents found, attempting task decomposition for {task_id}", 
                      "warning")
        subtasks = self._decompose_task(task)
        if subtasks:
            printer.status("ALT-DECOMPOSE", 
                          f"Decomposed {task_id} into {len(subtasks)} subtasks", 
                          "success")
            return {
                "task_id": task_id,
                "subtasks": subtasks,
                "mitigation_strategy": "Task decomposition",
                "risk_score": assignment['risk_score'] * 0.7  # Reduced risk
            }
        
        # 3. Final fallback - delay with mitigation
        printer.status("ALT-DELAY", 
                      f"Using delay mitigation for {task_id}", 
                      "warning")
        return {
            "task_id": task_id,
            "agent_id": original_agent,
            "start_time": assignment["start_time"] + self.retry_policy["delay"],
            "end_time": assignment["end_time"] + self.retry_policy["delay"],
            "risk_score": assignment['risk_score'],
            "mitigation_strategy": f"Delayed by {self.retry_policy['delay']}s with enhanced monitoring"
        }
    
    # Helper methods for risk assessment and decomposition
    def _calculate_agent_risk(self, task, agent_details):
        """Calculate agent-specific risk factor (0-1 scale)"""
        printer.status("INIT", "Calculate agent risk", "info")

        # Agent capability gap risk
        gap_risk = 1 - (len(set(agent_details['capabilities']) & set(task['requirements'])) / len(task['requirements']))
        
        # Historical performance risk
        perf_risk = agent_details.get("failures", 0) / (agent_details.get("successes", 1) + agent_details.get("failures", 0) + 1e-6)
        
        # Composite risk factor
        return 0.6 * gap_risk + 0.4 * perf_risk
    
    def _decompose_task(self, task):
        """Break complex tasks into simpler subtasks (stub implementation)"""
        printer.status("INIT", "Task desomposer succesfully initialized", "info")

        # Implementation would use domain-specific decomposition logic
        # Example: Medical task -> [triage, treatment, evacuation]
        return None  # Return None for this example

    def _agent_is_eligible(self, agent_id, task, details):
        """Check agent capabilities and availability"""
        printer.status("INIT", "Agent eligibility succesfully initialized", "info")

        capabilities = set(details.get('capabilities', []))
        requirements = set(task['requirements'])
        return (
            capabilities.issuperset(requirements) and
            details['current_load'] < 1.0 and
            agent_id not in task.get('blacklisted_agents', [])
        )

    def _calculate_agent_score(self, agent_id, task, agent_details):
        """Score agents based on capability match, current load, efficiency, and specialization."""
        printer.status("INIT", "Agent score succesfully initialized", "info")

        # 1. Success rate (reliability factor)
        successes = agent_details.get("successes", 1)
        failures = agent_details.get("failures", 0)
        success_rate = successes / (successes + failures + 1e-6)
        
        # 2. Efficiency factor (higher = better)
        efficiency = agent_details.get(self.efficiency_attribute, 1.0)
        
        # 3. Load penalty (current utilization)
        load_penalty = agent_details["current_load"] * 0.3
        
        # 4. Capability specialization bonus
        capabilities = set(agent_details.get('capabilities', []))
        requirements = set(task['requirements'])
        overlap = capabilities & requirements
        specialization = len(overlap) / len(requirements) if requirements else 1.0
        
        # 5. Deadline proximity factor (urgency bonus)
        time_factor = max(0, 1 - (task['deadline'] - time.time()) / 3600)  # 1hr normalization
        
        # Composite score calculation with weighted factors
        return (
            0.4 * success_rate +
            0.3 * efficiency +
            0.2 * specialization +
            0.1 * time_factor -
            load_penalty
        )

    def _create_schedule(self, candidate_map, agents, state):
        """Temporal scheduling with plan optimization"""
        printer.status("INIT", "Schedule creator succesfully initialized", "info")

        schedule = {}
        agent_loads = {a: 0.0 for a in agents}
        dependency_graph = self._build_dependency_graph(state)

        for task_id in self._order_by_dependencies(candidate_map, dependency_graph):
            candidates = candidate_map.get(task_id, [])
            best_agent, best_score = None, -np.inf
            
            for agent_id, score in candidates:
                current_load = agent_loads[agent_id]
                load_penalty = np.exp(current_load)
                adjusted_score = score / load_penalty
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_agent = agent_id
                    
            if best_agent:
                agent_details = agents[best_agent]
                schedule[task_id] = self._create_assignment(
                    task_id, best_agent, agent_details, agent_loads[best_agent], state
                )
                agent_loads[best_agent] = schedule[task_id]['end_time']
                
        return schedule

    def _build_dependency_graph(self, state):
        """Build dependency graph from task dependencies in the state"""
        printer.status("INIT", "dependencies succesfully initialized", "info")

        if state and 'dependency_graph' in state:
            return state['dependency_graph']
        
        # Fallback: Construct from tasks if not in state
        dependency_graph = defaultdict(list)
        tasks = state.get('tasks', []) if state else []
        
        for task in tasks:
            task_id = task.get('id')
            if not task_id:
                continue
            # Add edges from dependencies to this task
            for dep in task.get('dependencies', []):
                dependency_graph[dep].append(task_id)
        
        return dependency_graph
    
    def _order_by_dependencies(self, candidate_map, dependency_graph):
        """Topological sort using Kahn's algorithm for task dependencies"""
        printer.status("INIT", "Topological sorter succesfully initialized", "info")

        task_ids = list(candidate_map.keys())
        in_degree = defaultdict(int)
        adj = defaultdict(list)
        
        # Build adjacency list and in-degree counts
        for src, dests in dependency_graph.items():
            for dest in dests:
                if src in task_ids and dest in task_ids:
                    adj[src].append(dest)
                    in_degree[dest] += 1
        
        # Initialize queue with tasks having no dependencies
        queue = deque([t for t in task_ids if in_degree.get(t, 0) == 0])
        topo_order = []
        
        # Process nodes
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Handle cycles or missing dependencies
        if len(topo_order) != len(task_ids):
            remaining = [t for t in task_ids if t not in topo_order]
            topo_order.extend(remaining)
            logger.warning(f"Cyclic dependencies detected. Unordered tasks: {remaining}")
        
        return topo_order

    def _create_assignment(self, task_id: str, agent_id: str, agent_details: Dict, current_load: float, state: Dict) -> Dict:
        """Create task assignment with duration based on requirements and agent efficiency."""
        printer.status("INIT", "Assignment succesfully initialized", "info")

        task = next((t for t in (state.get('tasks', [])) if t.get('id') == task_id), None)
        num_requirements = len(task['requirements']) if task else 1
        base_duration = self.task_config.get('base_duration_per_requirement', 5.0) * num_requirements
        efficiency = max(agent_details.get(self.task_config.get('efficiency_attribute', 'efficiency'), 1.0), 0.1)
        task_duration = base_duration / efficiency
        
        return {
            'task_id': task_id,
            'agent_id': agent_id,
            'start_time': current_load,
            'end_time': current_load + task_duration,
            'risk_score': task.get('risk_assessment', {}).get('risk_score', 0.5) if task else 0.5
        }

if __name__ == "__main__":
    print("\n=== Running Task Scheduler Test ===\n")
    printer.status("Init", "Task Scheduler initialized", "success")

    scheduler = DeadlineAwareScheduler()
    print(scheduler)
    print("\n* * * * * Phase 2 * * * * *\n")
    tasks = [{
        "id": "task1",
        "requirements": ["leave_prep", "keys", "door_access"],
        "deadline": time.time() + 30,  # deadline 30 seconds from now
    }]
    agents = {
        "agent1": {
            "capabilities": ["leave_prep", "keys", "door_access"],
            "current_load": 0.2,
            "successes": 5,
            "failures": 1,
            "efficiency": 1.2
        },
        "agent2": {
            "capabilities": ["keys", "navigation"],
            "current_load": 0.4,
            "successes": 3,
            "failures": 2,
            "efficiency": 0.9
        }
    }
    state = {
        "tasks": tasks
    }

    plan = scheduler.schedule(tasks=tasks, agents=agents, risk_assessor = None, state=state)
    task = scheduler._prioritize_tasks(tasks=tasks, risk_assessor = None)
    map = scheduler._map_capabilities(tasks=tasks, agents=agents)
    printer.pretty("Planner", plan, "success")
    printer.pretty("task", task, "success")
    printer.pretty("task", map, "success")
    print("\n=== Successfully Ran Task Scheduler ===\n")
