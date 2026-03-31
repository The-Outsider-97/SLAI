"""
Task Scheduler – Deadline‑aware, risk‑sensitive scheduling with capability matching.

Provides an abstract scheduler interface and a concrete DeadlineAwareScheduler
that prioritizes tasks by deadline and risk, matches agent capabilities, and
offers fallback strategies for high‑risk tasks.
"""

import time
import threading
import numpy as np

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Task Scheduler")
printer = PrettyPrinter

class TaskScheduler(ABC):
    """Abstract base class for all task schedulers."""

    @abstractmethod
    def schedule(
        self,
        tasks: List[Dict],
        agents: Dict[str, Any],
        risk_assessor: Optional[Callable] = None,
        state: Optional[Dict] = None,
    ) -> Dict:
        """Schedule tasks to agents."""
        pass

class DeadlineAwareScheduler(TaskScheduler):
    """
    Earliest Deadline First scheduler with capability matching and risk assessment.

    The scheduler proceeds in four phases:
    1. Risk assessment and prioritisation.
    2. Agent capability matching.
    3. Temporal scheduling with dependencies.
    4. Risk mitigation and fallback planning.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = load_global_config()
        self.task_config = get_config_section("task_scheduler")

        # Configuration defaults
        self.risk_threshold = self.task_config.get("risk_threshold", 0.7)
        self.base_duration_per_requirement = self.task_config.get(
            "base_duration_per_requirement", 5.0
        )
        self.efficiency_attribute = self.task_config.get("efficiency_attribute", "efficiency")
        self.retry_policy = self.task_config.get(
            "retry_policy",
            {
                "max_retries": 3,
                "max_attempts": 3,
                "backoff_factor": 1.5,
                "delay": 10,
            },
        )

        # Runtime state
        self.agents: Dict[str, Any] = {}
        self.state: Optional[Dict] = None
        self.task_history: Dict[str, Any] = defaultdict(list)  # "current" key holds tasks

        self._lock = threading.RLock()
        logger.info("DeadlineAwareScheduler initialized")

    # -------------------------------------------------------------------------
    # Public scheduling entry point
    # -------------------------------------------------------------------------
    def schedule(
        self,
        tasks: List[Dict],
        agents: Dict[str, Any],
        risk_assessor: Optional[Callable] = None,
        state: Optional[Dict] = None,
    ) -> Dict:
        """
        Produce a schedule mapping task IDs to agent assignments.

        Args:
            tasks: List of task dictionaries, each must contain 'id', 'requirements',
                   'deadline', and optional 'priority', 'dependencies'.
            agents: Dictionary mapping agent IDs to agent details (capabilities,
                    current_load, successes, failures, efficiency).
            risk_assessor: Optional callable that returns a risk assessment dict
                           for a task. If provided, it must take a task dict and
                           return a dict with at least 'risk_score'.
            state: Optional global state containing e.g. dependency graph.

        Returns:
            Dictionary of scheduled assignments: {task_id: assignment_dict}.
            If validation fails, returns an empty dict.
        """
        with self._lock:
            self.agents = agents
            self.state = state
            self.task_history["current"] = tasks

            # Phase 0: Input validation
            if not self._validate_inputs(tasks, agents):
                logger.error("Input validation failed – cannot schedule")
                return {}

            # Phase 1: Risk assessment and prioritisation
            prioritized = self._prioritize_tasks(tasks, risk_assessor)

            # Phase 2: Capability matching
            candidate_map = self._map_capabilities(prioritized, agents)

            # Phase 3: Temporal scheduling with dependencies
            schedule = self._create_schedule(candidate_map, agents, state)

            # Phase 4: Risk mitigation
            return self._apply_risk_mitigation(schedule, risk_assessor)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    def _validate_inputs(self, tasks: List[Dict], agents: Dict[str, Any]) -> bool:
        """
        Perform comprehensive validation of inputs.

        Returns:
            True if validation passes, False otherwise.
        """
        errors = []
        warnings = []

        # Top‑level structure checks
        if not isinstance(tasks, list):
            errors.append("Tasks must be a list")
        if not isinstance(agents, dict):
            errors.append("Agents must be a dictionary")

        if errors:
            logger.error(f"Validation failed: {'; '.join(errors)}")
            return False

        # Task validation
        valid_task_ids = set()
        for idx, task in enumerate(tasks):
            task_id = task.get("id", f"Unidentified task at index {idx}")
            task_errors = []
            task_warnings = []

            # ID
            if "id" not in task:
                task_errors.append("Missing 'id' field")
            elif not isinstance(task["id"], str):
                task_errors.append("'id' must be a string")
            elif not task["id"].strip():
                task_errors.append("'id' cannot be empty")
            elif task["id"] in valid_task_ids:
                task_errors.append(f"Duplicate task ID: {task['id']}")
            else:
                valid_task_ids.add(task["id"])

            # Requirements
            if "requirements" not in task:
                task_warnings.append("Missing 'requirements', using default []")
                task["requirements"] = []
            elif not isinstance(task["requirements"], list):
                task_errors.append("'requirements' must be a list")
            else:
                for i, req in enumerate(task["requirements"]):
                    if not isinstance(req, str):
                        task_errors.append(f"Requirement {i} must be string")
                    elif not req.strip():
                        task_errors.append(f"Requirement {i} cannot be empty")

            # Deadline
            current_time = time.time()
            if "deadline" not in task:
                task_warnings.append("Missing 'deadline', using default (now + 300s)")
                task["deadline"] = current_time + 300
            elif not isinstance(task["deadline"], (int, float)):
                task_errors.append("'deadline' must be numeric")
            else:
                if task["deadline"] < current_time:
                    task_warnings.append("Deadline is in the past")
                elif task["deadline"] < current_time + 10:
                    task_warnings.append("Deadline is too imminent (<10s)")

            # Dependencies (optional)
            if "dependencies" in task:
                if not isinstance(task["dependencies"], list):
                    task_errors.append("'dependencies' must be a list")
                else:
                    for dep in task["dependencies"]:
                        if not isinstance(dep, str):
                            task_errors.append("Dependency must be string")
                        elif dep not in valid_task_ids:
                            task_warnings.append(f"Unknown dependency: {dep}")

            if task_errors:
                errors.append(f"Task {task_id}: {'; '.join(task_errors)}")
            if task_warnings:
                warnings.append(f"Task {task_id}: {'; '.join(task_warnings)}")

        # Agent validation
        valid_agent_ids = set()
        for agent_id, details in agents.items():
            agent_errors = []
            agent_warnings = []

            if not isinstance(details, dict):
                agent_errors.append("Agent details must be a dictionary")
            else:
                # Capabilities
                if "capabilities" not in details:
                    agent_warnings.append("Missing 'capabilities', using default []")
                    details["capabilities"] = []
                elif not isinstance(details["capabilities"], list):
                    agent_errors.append("'capabilities' must be a list")
                else:
                    for i, cap in enumerate(details["capabilities"]):
                        if not isinstance(cap, str):
                            agent_errors.append(f"Capability {i} must be string")
                        elif not cap.strip():
                            agent_errors.append(f"Capability {i} cannot be empty")

                # Current load
                if "current_load" not in details:
                    agent_warnings.append("Missing 'current_load', using default 0.0")
                    details["current_load"] = 0.0
                elif not isinstance(details["current_load"], (int, float)):
                    agent_errors.append("'current_load' must be numeric")
                elif details["current_load"] < 0:
                    agent_warnings.append("Negative load reset to 0.0")
                    details["current_load"] = 0.0
                elif details["current_load"] > 1.5:
                    agent_warnings.append("Extremely high load (>1.5)")

                # Success/failure counts (optional)
                for metric in ["successes", "failures"]:
                    if metric in details and not isinstance(details[metric], int):
                        agent_errors.append(f"'{metric}' must be integer")

                # Efficiency (optional)
                eff_attr = self.efficiency_attribute
                if eff_attr in details and not isinstance(details[eff_attr], (int, float)):
                    agent_errors.append(f"'{eff_attr}' must be numeric")

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

        # Report warnings and errors
        for w in warnings:
            logger.warning(w)
        for e in errors:
            logger.error(e)

        report = {
            "valid_tasks": len(valid_task_ids),
            "valid_agents": len(valid_agent_ids),
            "errors": len(errors),
            "warnings": len(warnings),
        }
        printer.status("VALIDATION", f"Validation report: {report}", "success" if not errors else "error")
        return not errors and valid_task_ids and valid_agent_ids

    # -------------------------------------------------------------------------
    # Prioritisation
    # -------------------------------------------------------------------------
    def _prioritize_tasks(self, tasks: List[Dict], risk_assessor: Optional[Callable]) -> List[Dict]:
        """Risk‑aware prioritisation – sort by descending priority."""
        prioritized = []
        for task in tasks:
            risk_score = 0.5
            if risk_assessor:
                assessment = risk_assessor(task) if risk_assessor else {}
                risk_score = assessment.get("risk_score", 0.5)
                task["risk_assessment"] = assessment

            priority = self._calculate_priority(
                task["deadline"],
                task.get("priority", 3),
                risk_score,
            )
            prioritized.append((priority, task))

        # Sort descending (higher priority first)
        prioritized.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in prioritized]

    def _calculate_priority(self, deadline: float, base_priority: int, risk_score: float) -> float:
        """
        Hybrid priority: 60% base priority, 30% time criticality, -10% risk penalty.
        Time criticality = 1/(time remaining + epsilon).
        """
        remaining = deadline - time.time()
        if remaining <= 0:
            time_criticality = 1e6  # extremely high if deadline already passed
        else:
            time_criticality = 1.0 / remaining

        risk_penalty = np.clip(risk_score - self.risk_threshold, 0, 1)
        return 0.6 * base_priority + 0.3 * time_criticality - 0.1 * risk_penalty

    # -------------------------------------------------------------------------
    # Capability mapping
    # -------------------------------------------------------------------------
    def _map_capabilities(self, tasks: List[Dict], agents: Dict[str, Any]) -> Dict[str, List[Tuple[str, float]]]:
        """
        For each task, return a list of (agent_id, score) pairs of eligible agents.
        Eligibility: agent capabilities superset of task requirements, load < 1.0,
                     and not blacklisted.
        """
        candidate_map = defaultdict(list)
        for task in tasks:
            for agent_id, details in agents.items():
                if self._agent_is_eligible(agent_id, task, details):
                    score = self._calculate_agent_score(agent_id, task, details)
                    candidate_map[task["id"]].append((agent_id, score))
        return candidate_map

    def _agent_is_eligible(self, agent_id: str, task: Dict, details: Dict) -> bool:
        """Check capability, load, and blacklist conditions."""
        capabilities = set(details.get("capabilities", []))
        requirements = set(task.get("requirements", []))
        return (
            capabilities.issuperset(requirements)
            and details.get("current_load", 0.0) < 1.0
            and agent_id not in task.get("blacklisted_agents", [])
        )

    def _calculate_agent_score(self, agent_id: str, task: Dict, details: Dict) -> float:
        """
        Score an agent for a given task based on:
        - success rate (40%)
        - efficiency (30%)
        - capability specialization (20%)
        - deadline proximity (10%)
        - load penalty (current_load * 0.3 subtracted)
        """
        successes = details.get("successes", 1)
        failures = details.get("failures", 0)
        success_rate = successes / (successes + failures + 1e-6)

        efficiency = details.get(self.efficiency_attribute, 1.0)

        requirements = set(task.get("requirements", []))
        if requirements:
            overlap = set(details.get("capabilities", [])) & requirements
            specialization = len(overlap) / len(requirements)
        else:
            specialization = 1.0

        remaining = task["deadline"] - time.time()
        time_factor = max(0.0, min(1.0, 1.0 - remaining / 3600))  # normalize over 1 hour

        load_penalty = details.get("current_load", 0.0) * 0.3

        return (
            0.4 * success_rate
            + 0.3 * efficiency
            + 0.2 * specialization
            + 0.1 * time_factor
            - load_penalty
        )

    # -------------------------------------------------------------------------
    # Temporal scheduling
    # -------------------------------------------------------------------------
    def _create_schedule(
        self, candidate_map: Dict[str, List[Tuple[str, float]]], agents: Dict[str, Any], state: Optional[Dict]
    ) -> Dict:
        """
        Create a schedule by assigning tasks to agents respecting dependencies.
        Uses a simple load‑based assignment order (topological order if dependencies exist).
        """
        schedule = {}
        agent_loads = {aid: 0.0 for aid in agents}
        dependency_graph = self._build_dependency_graph(state)

        task_order = self._order_by_dependencies(candidate_map, dependency_graph)

        for task_id in task_order:
            candidates = candidate_map.get(task_id, [])
            if not candidates:
                logger.warning(f"No eligible agent for task {task_id}")
                continue

            # Choose best agent based on score / load penalty
            best_agent = None
            best_adjusted = -np.inf
            for agent_id, score in candidates:
                current_load = agent_loads.get(agent_id, 0.0)
                load_penalty = np.exp(current_load)  # exponential load penalty
                adjusted = score / load_penalty
                if adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_agent = agent_id

            if best_agent:
                schedule[task_id] = self._create_assignment(
                    task_id, best_agent, agents[best_agent], agent_loads[best_agent], state
                )
                agent_loads[best_agent] = schedule[task_id]["end_time"]

        return schedule

    def _build_dependency_graph(self, state: Optional[Dict]) -> Dict[str, List[str]]:
        """Build a graph of dependencies from the state if present."""
        if state and "dependency_graph" in state:
            return state["dependency_graph"]

        # Fallback: construct from task dependencies if tasks are in state
        dep_graph = defaultdict(list)
        tasks = state.get("tasks", []) if state else []
        for task in tasks:
            task_id = task.get("id")
            if not task_id:
                continue
            for dep in task.get("dependencies", []):
                dep_graph[dep].append(task_id)
        return dep_graph

    def _order_by_dependencies(
        self, candidate_map: Dict[str, List[Tuple[str, float]]], dependency_graph: Dict[str, List[str]]
    ) -> List[str]:
        """
        Topological sort (Kahn's algorithm) on task IDs.
        If cycles exist, returns a partial order with remaining tasks appended.
        """
        task_ids = list(candidate_map.keys())
        in_degree = defaultdict(int)
        adj = defaultdict(list)

        # Build adjacency from dependencies
        for src, dests in dependency_graph.items():
            for dest in dests:
                if src in task_ids and dest in task_ids:
                    adj[src].append(dest)
                    in_degree[dest] += 1

        # Queue of tasks with no dependencies
        queue = deque([t for t in task_ids if in_degree.get(t, 0) == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(task_ids):
            remaining = [t for t in task_ids if t not in order]
            order.extend(remaining)
            logger.warning(f"Cyclic dependencies detected. Unordered tasks: {remaining}")

        return order

    def _create_assignment(
        self, task_id: str, agent_id: str, agent_details: Dict, current_load: float, state: Optional[Dict]
    ) -> Dict:
        """
        Create an assignment dictionary with start_time, end_time, and risk_score.
        Duration = (number of requirements * base_duration) / agent efficiency.
        """
        # Locate the task in state if available
        task = None
        if state and "tasks" in state:
            for t in state["tasks"]:
                if t.get("id") == task_id:
                    task = t
                    break
        if task is None:
            # Use dummy task
            task = {"requirements": [], "risk_assessment": {"risk_score": 0.5}}

        num_requirements = len(task.get("requirements", []))
        base_duration = self.base_duration_per_requirement * max(num_requirements, 1)
        efficiency = max(agent_details.get(self.efficiency_attribute, 1.0), 0.1)
        duration = base_duration / efficiency

        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "start_time": current_load,
            "end_time": current_load + duration,
            "risk_score": task.get("risk_assessment", {}).get("risk_score", 0.5),
        }

    # -------------------------------------------------------------------------
    # Risk mitigation
    # -------------------------------------------------------------------------
    def _apply_risk_mitigation(self, schedule: Dict, risk_assessor: Optional[Callable]) -> Dict:
        """Attempt to replace high‑risk assignments with safer alternatives."""
        mitigated = {}
        for task_id, assignment in schedule.items():
            if risk_assessor and assignment.get("risk_score", 0.0) > self.risk_threshold:
                alt = self._find_alternative(task_id, assignment, schedule, risk_assessor)
                if alt:
                    mitigated[task_id] = alt
                    continue
            mitigated[task_id] = assignment
        return mitigated

    def _find_alternative(self, task_id: str, assignment: Dict, schedule: Dict, risk_assessor: Callable) -> Optional[Dict]:
        """
        Find a safer alternative for a high‑risk task.
        Strategies:
        1. Reassign to a different agent.
        2. Decompose the task (stub).
        3. Delay the task.
        """
        logger.warning(f"Seeking alternatives for high‑risk task {task_id}")

        # 1. Try reassignment
        original_agent = assignment["agent_id"]
        task = next((t for t in self.task_history.get("current", []) if t["id"] == task_id), None)
        if not task:
            return None

        # Get all eligible agents (including previously rejected ones)
        eligible = self._map_capabilities([task], self.agents)[task_id]
        candidates = []
        for agent_id, raw_score in eligible:
            if agent_id == original_agent:
                continue
            agent_details = self.agents[agent_id]
            risk_factor = self._calculate_agent_risk(task, agent_details)
            adjusted_score = raw_score * (1 - risk_factor)
            candidates.append((adjusted_score, agent_id, agent_details))

        candidates.sort(reverse=True, key=lambda x: x[0])
        for score, agent_id, details in candidates:
            if score > 0:
                new_assignment = self._create_assignment(
                    task_id, agent_id, details, details["current_load"], self.state
                )
                new_assignment["mitigation_strategy"] = f"Reassigned from {original_agent}"
                logger.info(f"Found safer agent {agent_id} for task {task_id}")
                return new_assignment

        # 2. Fallback to task decomposition (stub – would need full task decomposition logic)
        subtasks = self._decompose_task(task)
        if subtasks:
            logger.info(f"Decomposed {task_id} into {len(subtasks)} subtasks")
            return {
                "task_id": task_id,
                "subtasks": subtasks,
                "mitigation_strategy": "Task decomposition",
                "risk_score": assignment.get("risk_score", 0.5) * 0.7,
            }

        # 3. Delay
        delay = self.retry_policy.get("delay", 10)
        logger.info(f"Using delay mitigation for {task_id} (+{delay}s)")
        return {
            "task_id": task_id,
            "agent_id": original_agent,
            "start_time": assignment.get("start_time", 0) + delay,
            "end_time": assignment.get("end_time", 0) + delay,
            "risk_score": assignment.get("risk_score", 0.5),
            "mitigation_strategy": f"Delayed by {delay}s",
        }

    def _calculate_agent_risk(self, task: Dict, agent_details: Dict) -> float:
        """
        Calculate an agent‑specific risk factor (0–1) based on:
        - capability gap (60%)
        - historical performance (40%)
        """
        requirements = set(task.get("requirements", []))
        capabilities = set(agent_details.get("capabilities", []))
        if requirements:
            gap_risk = 1 - len(capabilities & requirements) / len(requirements)
        else:
            gap_risk = 0.0

        successes = agent_details.get("successes", 1)
        failures = agent_details.get("failures", 0)
        total = successes + failures
        perf_risk = failures / total if total > 0 else 0.0

        return 0.6 * gap_risk + 0.4 * perf_risk

    def _decompose_task(self, task: Dict) -> Optional[List[Dict]]:
        """
        Stub for task decomposition. Override in subclasses or with domain logic.
        Returns None (no decomposition) by default.
        """
        return None


# -------------------------------------------------------------------------
# Test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Running Task Scheduler Test ===\n")
    printer.status("Init", "Task Scheduler initialized", "success")

    scheduler = DeadlineAwareScheduler()
    tasks = [
        {
            "id": "task1",
            "requirements": ["leave_prep", "keys", "door_access"],
            "deadline": time.time() + 30,
        }
    ]
    agents = {
        "agent1": {
            "capabilities": ["leave_prep", "keys", "door_access"],
            "current_load": 0.2,
            "successes": 5,
            "failures": 1,
            "efficiency": 1.2,
        },
        "agent2": {
            "capabilities": ["keys", "navigation"],
            "current_load": 0.4,
            "successes": 3,
            "failures": 2,
            "efficiency": 0.9,
        },
    }
    state = {"tasks": tasks}
    plan = scheduler.schedule(tasks=tasks, agents=agents, risk_assessor=None, state=state)
    printer.pretty("Planner", plan, "success")
    print("\n=== Successfully Ran Task Scheduler ===\n")