
import numpy as np
import time
import yaml, json

from typing import Dict, Any, List, Optional, Callable, Union
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from types import SimpleNamespace

from logs.logger import get_logger

logger = get_logger("Task Scheduler")

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

class TaskScheduler(ABC):
    """Abstract scheduler interface"""
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def schedule(self,
                 tasks: List[Dict],
                 agents: Dict[str, Any],
                 risk_assessor: Optional[Callable] = None,
                 state: Optional[Dict] = None) -> Dict:
        return

class DeadlineAwareScheduler(TaskScheduler):
    """Earliest Deadline First scheduler with capability matching"""
    def __init__(self, agent=None, risk_threshold: Optional[float] = None,
                 config_section_name: str = "task_scheduler",
                 config_file_path: str = CONFIG_PATH,
                 retry_policy: Optional[Dict] = None):
        self.config = get_config_section(config_section_name, config_file_path)
        self.agent = agent
        self.risk_threshold = risk_threshold if risk_threshold is not None else self.config.risk_threshold
        self.retry_policy = retry_policy if retry_policy is not None else self.config.retry_policy
        self.task_history = defaultdict(list)
        #self.agent.route_message({
        #    'message_id': '123',
        #    'content': 'urgent_task',
        #    'ttl': 30,  # 30 second deadline
        #    'metadata': {'priority': 2}
        #})

    def schedule(self,
                 tasks: List[Dict],
                 agents: Dict[str, Any],
                 risk_assessor: Optional[Callable] = None,
                 state: Optional[Dict] = None) -> Dict:

        """Main scheduling workflow with integrated risk checks"""
        
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
        """Consistent validation with planning_agent patterns"""
        if not tasks or not agents:
            logger.warning("Scheduling failed: empty tasks or agents")
            return False
            
        required_keys = {'id', 'requirements', 'deadline'}
        if not all(required_keys.issubset(t) for t in tasks):
            logger.error("Invalid task structure")
            return False
            
        return True

    def _prioritize_tasks(self, tasks, risk_assessor):
        """Risk-aware prioritization using collaborative agent's assessment"""
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
        time_criticality = 1 / (deadline - time.time() + 1e-6)
        risk_penalty = np.clip(risk_score - self.risk_threshold, 0, 1)
        return (0.6 * base_priority + 
                0.3 * time_criticality - 
                0.1 * risk_penalty)

    def _map_capabilities(self, tasks, agents):
        """Capability matching with load awareness"""
        candidate_map = defaultdict(list)
        for task in tasks:
            for agent_id, details in agents.items():
                if self._agent_is_eligible(agent_id, task, details):
                    score = self._calculate_agent_score(agent_id, task, details)
                    candidate_map[task['id']].append((agent_id, score))
        return candidate_map

    def _agent_is_eligible(self, agent_id, task, details):
        """Check agent capabilities and availability"""
        capabilities = set(details.get('capabilities', []))
        requirements = set(task['requirements'])
        return (
            capabilities.issuperset(requirements) and
            details['current_load'] < 1.0 and
            agent_id not in task.get('blacklisted_agents', [])
        )

    def _calculate_agent_score(self, agent_id, task, agent_details):
        """Score agents based on capability match and current load."""
        # Example: Higher success rate and lower load = better score
        success_rate = agent_details.get("successes", 1) / (agent_details.get("failures", 0) + agent_details.get("successes", 1) + 1e-6)
        load_penalty = agent_details["current_load"] * 0.3
        return success_rate - load_penalty

    def _create_schedule(self, candidate_map, agents, state):
        """Temporal scheduling with plan optimization"""
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
        task = next((t for t in (state.get('tasks', [])) if t.get('id') == task_id), None)
        num_requirements = len(task['requirements']) if task else 1
        base_duration = self.config.base_duration_per_requirement * num_requirements
        efficiency = max(agent_details.get(self.config.efficiency_attribute, 1.0), 0.1)
        task_duration = base_duration / efficiency
        
        return {
            'task_id': task_id,
            'agent_id': agent_id,
            'start_time': current_load,
            'end_time': current_load + task_duration,
            'risk_score': task.get('risk_assessment', {}).get('risk_score', 0.5) if task else 0.5
        }

    def _apply_risk_mitigation(self, schedule, risk_assessor):
        """Apply collaborative agent's safety recommendations"""
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

if __name__ == "__main__":
    print("")
    print("\n=== Running Task Scheduler ===")
    print("")
    from unittest.mock import Mock
    mock_agent = Mock()
    mock_agent.shared_memory = {}
    scheduler = DeadlineAwareScheduler(agent=mock_agent)
    print("")
    print("\n=== Successfully Ran Task Scheduler ===\n")

if __name__ == "__main__":
    print("\n=== Emergency Response Simulation ===")
    from datetime import datetime, timedelta
    import time

    # Create mock agents with different capabilities
    agents = {
        "medic_team_1": {
            "capabilities": ["medical", "triage", "evacuation"],
            "current_load": 0.0,
            "efficiency": 1.2,
            "successes": 15,
            "failures": 2
        },
        "fire_team_alpha": {
            "capabilities": ["firefighting", "search_rescue", "hazmat"],
            "current_load": 0.3,
            "efficiency": 0.9,
            "successes": 20,
            "failures": 1
        },
        "drone_swarm": {
            "capabilities": ["mapping", "surveillance"],
            "current_load": 0.1,
            "efficiency": 2.5,
            "successes": 45,
            "failures": 5
        }
    }

    # Create emergency response tasks with dependencies
    base_time = time.time()
    tasks = [
        {
            "id": "t1",
            "requirements": ["medical"],
            "deadline": base_time + 1800,  # 30 minutes
            "priority": 1,
            "description": "Immediate first aid"
        },
        {
            "id": "t2",
            "requirements": ["firefighting"],
            "deadline": base_time + 3600,  # 1 hour
            "priority": 2,
            "dependencies": ["t1"],
            "description": "Contain chemical fire"
        },
        {
            "id": "t3",
            "requirements": ["mapping"],
            "deadline": base_time + 1200,  # 20 minutes
            "priority": 1,
            "description": "Map disaster area"
        },
        {
            "id": "t4",
            "requirements": ["search_rescue"],
            "deadline": base_time + 2400,  # 40 minutes
            "priority": 1,
            "dependencies": ["t3"],
            "description": "Search for trapped civilians"
        },
        {
            "id": "t5",
            "requirements": ["hazmat"],
            "deadline": base_time + 7200,  # 2 hours
            "priority": 3,
            "description": "Chemical spill containment"
        }
    ]

    # Simple risk assessment function
    def risk_assessor(task):
        complexity = len(task.get("requirements", []))
        return {"risk_score": min(0.2 * complexity + 0.1 * task.get("priority", 1), 1.0)}

    # Initialize scheduler
    from unittest.mock import Mock
    mock_agent = Mock()
    mock_agent.shared_memory = {"dependency_graph": {"t3": ["t4"]}}
    
    scheduler = DeadlineAwareScheduler(agent=mock_agent)
    
    # Run scheduling
    print("\nInitializing emergency response scheduling...")
    schedule = scheduler.schedule(
        tasks=tasks,
        agents=agents,
        risk_assessor=risk_assessor,
        state={"tasks": tasks}
    )

    # Display results
    print("\n=== Final Emergency Schedule ===")
    print(f"{'Task ID':<8}{'Agent':<18}{'Start':<8}{'End':<8}{'Risk':<6}{'Description'}")
    for task_id, assignment in schedule.items():
        task = next(t for t in tasks if t["id"] == task_id)
        print(f"{task_id:<8}{assignment['agent_id']:<18}"
              f"{assignment['start_time']:.1f}\t{assignment['end_time']:.1f}\t"
              f"{assignment['risk_score']:.2f}\t{task['description']}")

    # Timeline visualization
    print("\n=== Timeline ===")
    for agent_id, details in agents.items():
        agent_tasks = [t for t in schedule.values() if t["agent_id"] == agent_id]
        agent_tasks.sort(key=lambda x: x["start_time"])
        timeline = " | ".join(f"[{t['task_id']} {t['start_time']:.1f}-{t['end_time']:.1f}]" 
                   for t in agent_tasks)
        print(f"{agent_id}: {timeline}")
