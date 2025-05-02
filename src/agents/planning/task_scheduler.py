
import numpy as np
import time

from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
from collections import defaultdict, deque

from logs.logger import get_logger

logger = get_logger(__name__)

class TaskScheduler(ABC):
    """Abstract scheduler interface"""
    @abstractmethod
    def schedule(self,
                 tasks: List[Dict],
                 agents: Dict[str, Any],
                 risk_assessor: Optional[Callable] = None,
                 state: Optional[Dict] = None) -> Dict:
        pass

class DeadlineAwareScheduler(TaskScheduler):
    """Earliest Deadline First scheduler with capability matching"""
    def __init__(self, risk_threshold: float = 0.7, retry_policy: Dict = None):
        self.risk_threshold = risk_threshold
        self.retry_policy = retry_policy or {
            'max_attempts': 3,
            'backoff_factor': 1.5
        }
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
                assessment = risk_assessor(task.get('risk_score', 0.5))
                risk_score = assessment.risk_score
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
                schedule[task_id] = self._create_assignment(
                    task_id, best_agent, agent_loads[best_agent], state
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
