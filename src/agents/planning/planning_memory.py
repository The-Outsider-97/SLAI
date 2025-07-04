
import os
import time
import copy
import json, yaml

from typing import Optional, Union, Dict, Any
from collections import deque, defaultdict

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Planning Memory")
printer = PrettyPrinter

class PlanningMemory:
    """Maintains planning state checkpoints and statistical memory"""
    def __init__(self):
        self.config = load_global_config()
        self.monitor_snapshot = self.config.get('monitor_snapshot')

        self.memory_config = get_config_section('planning_memory')
        self.checkpoints_dir = self.memory_config.get('checkpoints_dir')
        self.max_checkpoints = self.memory_config.get('max_checkpoints')
        self.history_window = self.memory_config.get('history_window')
        self.retention_days = self.memory_config.get('retention_days')
        self.compression = self.memory_config.get('compression')
        self.auto_save_interval = self.memory_config.get('auto_save_interval')

        self.agent = {}
        self.checkpoints = deque(maxlen=self.max_checkpoints)
        self._init_default_state()
        self.ensure_checkpoint_dir()
        self.load_latest_checkpoint()

    def _init_default_state(self):
        """Initialize empty state containers"""
        printer.status("INIT", "Default state succesfully initialized", "info")

        self.base_state = {
            'task_library': {},
            'method_stats': defaultdict(lambda: {'success': 0, 'total': 0, 'avg_cost': 0.0}),
            'world_state': {},
            'execution_history': deque(maxlen=self.history_window),
            'plan_metrics': defaultdict(list)
        }

    def ensure_checkpoint_dir(self):
        """Ensures the checkpoint directory exists."""
        try:
            if not os.path.exists(self.checkpoints_dir):
                os.makedirs(self.checkpoints_dir, exist_ok=True)
                logger.info(f"Created checkpoint directory: {self.checkpoints_dir}")
        except Exception as e:
            logger.error(f"Failed to create checkpoint directory: {e}")
            raise

    def load_latest_checkpoint(self):
        """Load most recent checkpoint on initialization"""
        try:
            if self.checkpoints:
                self.load_checkpoint(-1)
        except Exception as e:
            logger.error(f"Checkpoint load failed: {e}")

    def get_task_outcome(self, task_id: str) -> Optional[bool]:
        """Retrieve actual outcome of a task (True=success, False=failure)"""
        if task_id in self.base_state['execution_history']:
            entry = self.base_state['execution_history'][task_id]
            return entry.get('status') == 'success'
        return None

    def get_method_success_rate(self, method_name: str) -> float:
        """Get historical success rate for a method"""
        stats = self.base_state['method_stats'].get(method_name)
        if stats and stats['total'] > 0:
            return stats['success'] / stats['total']
        return 0.0
    
    def get_min_duration(self, task_name: str) -> float:
        """Retrieve the minimum observed duration for a given task"""
        durations = []
    
        for entry in self.base_state['execution_history']:
            if entry.get('name') == task_name:
                start = entry.get('start_time')
                end = entry.get('end_time')
    
                if start is not None and end is not None and end > start:
                    durations.append(end - start)
    
        if durations:
            return min(durations)
    
        logger.warning(f"No valid duration data found for task '{task_name}'. Returning infinity.")
        return float('inf')

    def is_sequential_task(self, task: Dict[str, Any], min_length: int) -> bool:
        """Check task sequence using execution history"""
        task_chain = self._get_task_ancestry(task)
        
        # Check memory for sequential patterns
        for entry in self.base_state['execution_history']:
            if entry.get('task_chain') == task_chain[-min_length:]:
                return True
        return False

    def _get_task_ancestry(self, task: Dict[str, Any]) -> list:
        """Get task hierarchy chain"""
        chain = []
        current = task
        while current:
            if 'name' not in current:
                logger.error(f"Missing 'name' in task: {current}")
                break
            chain.append(current['name'])
            current = current.get('parent')
        return chain[::-1]  # Root-first order


    def save_checkpoint(self, label: str = None, metadata: dict = None):
        """Capture current planning state with timestamp"""
        checkpoint = {
            'timestamp': time.time(),
            'label': label or "auto_save",
            'state': self._capture_agent_state(),
            'metadata': metadata or {}
        }

        self.checkpoints.append(checkpoint)
        logger.info(f"Saved planning checkpoint '{checkpoint['label']}'")
        return checkpoint

    def load_checkpoint(self, index: int = -1):
        """Restore planning state from checkpoint"""
        if not self.checkpoints:
            logger.warning("No checkpoints available to load")
            return False

        try:
            checkpoint = self.checkpoints[index]
            self._restore_agent_state(checkpoint['state'])
            logger.info(f"Loaded checkpoint '{checkpoint['label']}' from {checkpoint['timestamp']}")
            return True
        except IndexError:
            logger.error(f"Invalid checkpoint index: {index}")
            return False

    def ensure_checkpoint_access(self, path) -> str:
        """
        Ensures the checkpoint directory exists and the file is accessible.
        If the directory doesn't exist, it is created.
        If the file doesn't exist, an empty one is created.

        Returns:
            str: The absolute path to the checkpoint file.
        """
        try:
            dir_path = os.path.dirname(path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created checkpoint directory: {dir_path}")

            if not os.path.exists(path):
                with open(path, 'w', encoding='utf-8') as f:
                    f.write('')  # create empty file
                logger.info(f"Created new checkpoint file: {path}")
            else:
                logger.debug(f"Checkpoint file already exists: {path}")

            return os.path.abspath(path)

        except Exception as e:
            logger.error(f"Failed to access or create checkpoint file: {e}")
            raise

    def _capture_agent_state(self):
        """Deep copy relevant agent state components"""
        if not self.agent:
            return {}

        return {
            'task_library': copy.deepcopy(self.agent.task_library),
            'method_stats': copy.deepcopy(self.agent.method_stats),
            'world_state': copy.deepcopy(self.agent.world_state),
            'execution_history': copy.deepcopy(self.agent.execution_history),
            'plan_metrics': copy.deepcopy(getattr(self.agent, 'plan_metrics', {}))
        }

    def _restore_agent_state(self, state: dict):
        """Restore state to agent instance"""
        if not self.agent:
            return

        self.agent.task_library = state['task_library']
        self.agent.method_stats = state['method_stats']
        self.agent.world_state = state['world_state']
        self.agent.execution_history = deque(state['execution_history'], maxlen=self.history_window)

        if 'plan_metrics' in state:
            self.agent.plan_metrics = state['plan_metrics']

    def prune_old_checkpoints(self):
        """Remove checkpoints older than retention period"""
        if not self.retention_days:
            return

        cutoff = time.time() - (self.retention_days * 86400)
        initial_count = len(self.checkpoints)
        self.checkpoints = deque(
            [cp for cp in self.checkpoints if cp['timestamp'] > cutoff],
            maxlen=self.max_checkpoints
        )
        logger.debug(f"Pruned {initial_count - len(self.checkpoints)} old checkpoints")

    def to_json(self, file_path: str = None):
        """Serialize checkpoints to JSON format"""
        data = {
            'config': self.config,
            'checkpoints': list(self.checkpoints)
        }
        return json.dumps(data, indent=2) if not file_path else json.dump(data, open(file_path, 'w'))

    @classmethod
    def from_json(cls, json_data: Union[str, dict], agent=None):
        """Reconstruct planning memory from JSON"""
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        memory = cls(agent=agent)
        memory.checkpoints = deque(data.get('checkpoints', []), maxlen=memory.max_checkpoints)
        return memory

    def get_memory_usage(self) -> dict:
        """Return memory consumption statistics"""
        return {
            'checkpoints': len(self.checkpoints),
            'task_library': len(self.base_state['task_library']),
            'method_stats': len(self.base_state['method_stats']),
            'history_items': len(self.base_state['execution_history'])
        }

if __name__ == "__main__":
    print("\n=== Running Planning Memory Test ===\n")
    printer.status("Init", "Planning Memory initialized", "success")

    memory = PlanningMemory()
    print(memory)
    print("\n=== Successfully Ran Planning Memory ===\n")
