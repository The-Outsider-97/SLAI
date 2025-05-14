
import yaml
import psutil
import time

from collections import deque, defaultdict
from typing import Dict, Union, List
from types import SimpleNamespace

from src.agents.planning.planning_types import Task, TaskType, TaskStatus
from src.agents.planning.planning_metrics import PlanningMetrics
from src.agents.planning.decision_tree_heuristic import DecisionTreeHeuristic
from src.agents.planning.gradient_boosting_heuristic import GradientBoostingHeuristic
from src.agents.planning.task_scheduler import DeadlineAwareScheduler
from logs.logger import get_logger

logger = get_logger("Planning Monitor")

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

class PlanningMonitor:
    """Monitors planning system health and operational metrics"""
    def __init__(self, agent=None,
                 config_section_name: str = "planning_monitor",
                 config_file_path: str = CONFIG_PATH):
        self.config = get_config_section(config_section_name, config_file_path)
        self.agent = agent
        self._reset_tracking()

    def _reset_tracking(self):
        """Initialize all metric tracking structures"""
        self.plan_history = deque(maxlen=self.config.metrics_window)
        self.method_performance = defaultdict(lambda: {
            'success': 0, 'total': 0, 'last_used': 0
        })
        self.resource_history = deque(maxlen=3600)  # 1 hour at 1-sec resolution
        self.last_full_scan = time.time()

    def track_plan_start(self, plan: List[Task]):
        """Record initial plan state"""
        plan_meta = {
            'start_time': time.time(),
            'task_count': len(plan),
            'abstract_count': sum(1 for t in plan if t.task_type == TaskType.ABSTRACT),
            'status': None,
            'metrics': None
        }
        self.plan_history.append(plan_meta)
        return plan_meta

    def track_plan_completion(self, plan_meta: Dict, final_status: TaskStatus):
        """Finalize plan tracking with execution results"""
        plan_meta['status'] = final_status
        plan_meta['duration'] = time.time() - plan_meta['start_time']
        
        # Record primitive task success rates
        for task in self.agent.current_plan:
            if task.task_type == TaskType.PRIMITIVE:
                key = (task.name, 'primitive')
                self.method_performance[key]['total'] += 1
                if task.status == TaskStatus.SUCCESS:
                    self.method_performance[key]['success'] += 1

        self._perform_interval_checks()

    def update_method_stats(self, method_key: tuple, success: bool):
        """Update statistics for HTN decomposition methods"""
        self.method_performance[method_key]['total'] += 1
        self.method_performance[method_key]['last_used'] = time.time()
        if success:
            self.method_performance[method_key]['success'] += 1

    def _perform_interval_checks(self):
        """Execute configured health checks"""
        self._check_resource_limits()
        
        if len(self.plan_history) % self.config.check_intervals.plan_execution == 0:
            self._analyze_planning_trends()
            self._identify_method_anomalies()

    def _check_resource_limits(self):
        """Monitor system resource utilization"""
        current_cpu = psutil.cpu_percent()
        current_mem = psutil.Process().memory_info().rss / 1024 / 1024
        
        self.resource_history.append({
            'timestamp': time.time(),
            'cpu': current_cpu,
            'memory': current_mem
        })

        if current_cpu > self.config.anomaly_thresholds.cpu_peak:
            logger.warning(f"CPU threshold breached: {current_cpu}%")
            
        if current_mem > self.config.anomaly_thresholds.memory_peak:
            logger.warning(f"Memory threshold breached: {current_mem:.1f}MB")

        if time.time() - self.last_full_scan > self.config.check_intervals.resource_scan:
            self._perform_full_system_scan()
            self.last_full_scan = time.time()

    def _analyze_planning_trends(self):
        """Calculate success rates and efficiency metrics"""
        recent_plans = list(self.plan_history)[-self.config.check_intervals.plan_execution:]
        success_count = sum(1 for p in recent_plans if p['status'] == TaskStatus.SUCCESS)
        success_rate = success_count / len(recent_plans) if recent_plans else 0
        
        if success_rate < self.config.anomaly_thresholds.success_rate:
            logger.error(f"Success rate alert: {success_rate:.1%} below threshold")

    def _identify_method_anomalies(self):
        """Detect underperforming decomposition methods"""
        methods = sorted(
            self.method_performance.items(),
            key=lambda x: x[1]['total'],
            reverse=True
        )[:self.config.method_analysis_depth]

        for (name, type_), stats in methods:
            if stats['total'] == 0:
                continue
            success_rate = stats['success'] / stats['total']
            if success_rate < self.config.anomaly_thresholds.success_rate:
                logger.warning(f"Underperforming method: {name} ({type_}) "
                              f"Success rate: {success_rate:.1%}")

    def _perform_full_system_scan(self):
        """Comprehensive system health check"""
        logger.info("Performing full planning system scan...")
        
        # Check method usage distribution
        total_method_calls = sum(m['total'] for m in self.method_performance.values())
        if total_method_calls > 0:
            top_method = max(self.method_performance.items(),
                            key=lambda x: x[1]['total'])
            logger.debug(f"Most used method: {top_method[0][0]} ({top_method[1]['total']} uses)")

        # Analyze resource trends
        if self.resource_history:
            avg_cpu = sum(r['cpu'] for r in self.resource_history) / len(self.resource_history)
            avg_mem = sum(r['memory'] for r in self.resource_history) / len(self.resource_history)
            logger.info(f"Resource averages - CPU: {avg_cpu:.1f}%, Memory: {avg_mem:.1f}MB")

    def generate_diagnostics(self) -> Dict:
        """Return current system state for external monitoring"""
        return {
            'recent_success_rate': self._calculate_recent_success_rate(),
            'active_methods': len(self.method_performance),
            'resource_stats': self._current_resource_usage(),
            'pending_plans': len(self.agent.current_plan) if self.agent else 0
        }
    
    def monitor_planning_metrics(self, plan: List[Task], final_status: TaskStatus, planning_start: float, planning_end: float) -> Dict:
        """Evaluate full set of IPC-style planning metrics."""
        metrics = PlanningMetrics(name="MonitorMetrics", agent=self.agent)
        results = metrics.calculate_all_metrics(plan, planning_start, planning_end, final_status)
        
        logger.info(f"Plan metrics summary: {results}")
        return results
    
    def monitor_decision_tree_heuristic(self, dt_heuristic: DecisionTreeHeuristic) -> Dict:
        """Track feature usage and decision accuracy of decision tree."""
        if not dt_heuristic.trained:
            logger.warning("DecisionTreeHeuristic is not trained.")
            return {"trained": False}
    
        importances = dt_heuristic.feature_importances_
        named = list(zip(dt_heuristic.feature_names, importances))
        top_features = sorted(named, key=lambda x: x[1], reverse=True)
    
        logger.info("Top features in DecisionTreeHeuristic:")
        for fname, score in top_features[:5]:
            logger.info(f"- {fname}: {score:.3f}")
    
        return {
            "trained": True,
            "top_features": top_features[:5],
            "full_feature_importances": named
        }
    
    def monitor_gradient_boosting_heuristic(self, gb_heuristic: GradientBoostingHeuristic) -> Dict:
        """Analyze training state and feature relevance of GB classifier."""
        if not gb_heuristic.trained:
            logger.warning("GradientBoostingHeuristic is not trained.")
            return {"trained": False}
    
        importances = gb_heuristic.feature_importances_
        named = list(zip(gb_heuristic.feature_names, importances))
        top_features = sorted(named, key=lambda x: x[1], reverse=True)
    
        logger.info("Top features in GradientBoostingHeuristic:")
        for fname, score in top_features[:5]:
            logger.info(f"- {fname}: {score:.3f}")
    
        return {
            "trained": True,
            "top_features": top_features[:5],
            "full_feature_importances": named
        }
    
    def monitor_deadline_scheduler(self, scheduler: DeadlineAwareScheduler) -> Dict:
        """Summarize task assignment quality and fairness."""
        total_assignments = sum(len(history) for history in scheduler.task_history.values())
        agent_utilization = {
            agent: len(tasks)
            for agent, tasks in scheduler.task_history.items()
        }
    
        most_loaded = max(agent_utilization.items(), key=lambda x: x[1], default=("none", 0))
        least_loaded = min(agent_utilization.items(), key=lambda x: x[1], default=("none", 0))
    
        logger.info(f"Total assignments made: {total_assignments}")
        logger.info(f"Most loaded agent: {most_loaded[0]} ({most_loaded[1]} tasks)")
        logger.info(f"Least loaded agent: {least_loaded[0]} ({least_loaded[1]} tasks)")
    
        return {
            "total_assignments": total_assignments,
            "agent_utilization": agent_utilization,
            "most_loaded": most_loaded,
            "least_loaded": least_loaded
        }

    def _calculate_recent_success_rate(self) -> float:
        recent = list(self.plan_history)[-self.config.metrics_window:]
        successes = sum(1 for p in recent if p['status'] == TaskStatus.SUCCESS)
        return successes / len(recent) if recent else 0.0

    def _current_resource_usage(self) -> Dict:
        return {
            'cpu': psutil.cpu_percent(),
            'memory': psutil.Process().memory_info().rss / 1024 / 1024,
            'timestamp': time.time()
        }

if __name__ == "__main__":
    print("")
    print("\n=== Running Planning Monitor ===")
    print("")
    from unittest.mock import Mock
    mock_agent = Mock()
    mock_agent.shared_memory = {}
    monitor = PlanningMonitor(agent=mock_agent)

    # Example stubs:
    dt_heuristic = DecisionTreeHeuristic(agent=mock_agent)
    gb_heuristic = GradientBoostingHeuristic(agent=mock_agent)
    scheduler = DeadlineAwareScheduler(agent=mock_agent)

    monitor.monitor_decision_tree_heuristic(dt_heuristic)
    monitor.monitor_gradient_boosting_heuristic(gb_heuristic)
    monitor.monitor_deadline_scheduler(scheduler)
    print("")
    print("\n=== Successfully Ran Planning Monitor ===\n")
