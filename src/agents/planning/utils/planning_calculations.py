
import math
import time

from typing import List, Dict, Union

from src.agents.planning.planning_types import Task, ResourceProfile, ClusterResources
from src.agents.planning.utils.resource_monitor import ResourceMonitor

class PlanningCalculations:
    """Centralized service for safety-critical planning calculations"""

    def __init__(self):
        self.resource_monitor = ResourceMonitor()

    def calculate_resource_margin(self, tasks: Union[Task, List[Task]]) -> float:
        """
        Calculates resource margin for task(s) (0-1 scale, 1=best)
        Implements NASA-inspired margin calculation with cluster awareness
        """
        if not tasks:
            return 1.0  # No tasks = best possible margin
            
        if not isinstance(tasks, list):
            tasks = [tasks]
        
        # Get current resource availability
        if not self.resource_monitor:
            return 0.7  # Default safety margin if no monitor
        
        available = self.resource_monitor.get_available_resources()
        total_req = ResourceProfile()
        
        # Aggregate resource requirements
        for task in tasks:
            req = task.resource_requirements
            total_req.gpu += req.gpu
            total_req.ram += req.ram
            total_req.specialized_hardware = list(
                set(total_req.specialized_hardware) | set(req.specialized_hardware)
            )
        
        # Calculate margin components
        gpu_margin = self._calculate_component_margin(
            total_req.gpu, available.gpu_total, 'gpu'
        )
        ram_margin = self._calculate_component_margin(
            total_req.ram, available.ram_total, 'ram'
        )
        hw_margin = self._calculate_hardware_margin(
            total_req.specialized_hardware,
            available.specialized_hardware_available
        )
        
        # Geometric mean provides balanced view of all margins
        return (gpu_margin * ram_margin * hw_margin) ** (1/3)
    
    def _calculate_component_margin(self, required: float, available: float, 
                                  resource_type: str) -> float:
        """Calculates margin for a single resource component"""
        if required <= 0:
            return 1.0
            
        if available <= 0:
            return 0.0  # No available resources
            
        utilization = required / available
        margin = 1 - utilization
        
        # Apply safety buffer based on resource type
        if resource_type == 'gpu':
            buffer = 0.15  # GPU buffer
        else:
            buffer = 0.2   # RAM buffer
            
        return max(0, min(1, margin - buffer))
    
    def _calculate_hardware_margin(self, required: List[str], available: List[str]) -> float:
        """Calculates margin for specialized hardware"""
        if not required:
            return 1.0
            
        if not available:
            return 0.0
            
        # Calculate coverage ratio
        coverage = len(set(required) & set(available)) / len(required)
        buffer = 0.1  # Hardware buffer
        return max(0, min(1, coverage - buffer))

    def calculate_temporal_margin(self, tasks: Union[Task, List[Task]]) -> float:
        """
        Calculates temporal margin for task(s) (0-1 scale, 1=best)
        Uses critical path analysis with PERT-style estimation
        """
        if not tasks:
            return 1.0
            
        if not isinstance(tasks, list):
            tasks = [tasks]
        
        current_time = time.time()
        total_duration = 0
        max_deadline = 0
        
        for task in tasks:
            # Handle missing attributes gracefully
            duration = getattr(task, 'duration', 300)
            deadline = getattr(task, 'deadline', current_time + 3600)
            
            total_duration += duration
            if deadline > max_deadline:
                max_deadline = deadline
        
        # Calculate available time window
        available_time = max(0, max_deadline - current_time)
        
        if available_time <= 0:
            return 0.0
            
        # Calculate temporal margin with buffer
        utilization = total_duration / available_time
        margin = 1 - utilization
        buffer = 0.25  # Time buffer
        
        return max(0, min(1, margin - buffer))

    def calculate_dependency_risk(self, tasks: Union[Task, List[Task]]) -> float:
        """
        Calculates dependency risk for task(s) (0-1 scale, 1=best)
        Implements graph complexity analysis based on dependencies
        """
        if not tasks:
            return 1.0
            
        if not isinstance(tasks, list):
            tasks = [tasks]
        
        # Build dependency graph
        graph = {}
        for task in tasks:
            deps = getattr(task, 'dependencies', [])
            if deps:
                graph[task.id] = deps
        
        # Calculate complexity metrics
        num_nodes = len(graph)
        if num_nodes == 0:
            return 1.0  # No dependencies = lowest risk
            
        num_edges = sum(len(deps) for deps in graph.values())
        
        # Calculate criticality factor (longest path in dependency graph)
        criticality = self._find_criticality(graph)
        
        # Normalize metrics
        edge_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        normalized_criticality = criticality / num_nodes if num_nodes > 0 else 0
        
        # Combine factors with weights
        risk_score = (0.6 * normalized_criticality) + (0.4 * edge_density)
        return max(0, min(1, 1 - risk_score))  # Convert to margin (1 - risk)
    
    def _find_criticality(self, graph: Dict[str, List[str]]) -> int:
        """Finds longest path in dependency graph using topological sort"""
        in_degree = {node: 0 for node in graph}
        for deps in graph.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Initialize distances
        dist = {node: 0 for node in graph}
        queue = [node for node in graph if in_degree[node] == 0]
        
        while queue:
            node = queue.pop(0)
            for neighbor in graph.get(node, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if dist[neighbor] < dist[node] + 1:
                        dist[neighbor] = dist[node] + 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return max(dist.values()) if dist else 0