
import os
import time
import copy
import json, yaml

from typing import Union, Dict, List, Deque
from collections import deque, defaultdict
from types import SimpleNamespace

from logs.logger import get_logger

logger = get_logger("Planning Memory")

CONFIG_PATH = "src/agents/planning/configs/planning_config.yaml"
CHECKPOINT_PATH = "src/agents/planning/checkpoint/checkpoint.txt"

def dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d

def get_config_section(section: Union[str, Dict], config_file_path: str = CONFIG_PATH):
    if isinstance(section, dict):
        return dict_to_namespace(section)
    
    with open(config_file_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if section not in config:
        raise KeyError(f"Section '{section}' not found in config file: {config_file_path}")
    return dict_to_namespace(config[section])

class PlanningMemory:
    """Maintains planning state checkpoints and statistical memory"""
    def __init__(self, agent=None,
                 config_section_name: str = "planning_memory",
                 config_file_path: str = CONFIG_PATH):
        self.config = get_config_section(config_section_name, config_file_path)
        self.agent = agent
        self.checkpoints = deque(maxlen=self.config.max_checkpoints)
        self._init_default_state()

    def _init_default_state(self):
        """Initialize empty state containers"""
        self.base_state = {
            'task_library': {},
            'method_stats': defaultdict(lambda: {'success': 0, 'total': 0, 'avg_cost': 0.0}),
            'world_state': {},
            'execution_history': deque(maxlen=self.config.history_window),
            'plan_metrics': defaultdict(list)
        }

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

    def ensure_checkpoint_access(self, path: str = CHECKPOINT_PATH) -> str:
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
        self.agent.execution_history = deque(state['execution_history'], maxlen=self.config.history_window)
        
        if 'plan_metrics' in state:
            self.agent.plan_metrics = state['plan_metrics']

    def prune_old_checkpoints(self):
        """Remove checkpoints older than retention period"""
        if not self.config.retention_days:
            return

        cutoff = time.time() - (self.config.retention_days * 86400)
        initial_count = len(self.checkpoints)
        self.checkpoints = deque(
            [cp for cp in self.checkpoints if cp['timestamp'] > cutoff],
            maxlen=self.config.max_checkpoints
        )
        logger.debug(f"Pruned {initial_count - len(self.checkpoints)} old checkpoints")

    def to_json(self, file_path: str = None):
        """Serialize checkpoints to JSON format"""
        data = {
            'config': vars(self.config),
            'checkpoints': list(self.checkpoints)
        }
        return json.dumps(data, indent=2) if not file_path else json.dump(data, open(file_path, 'w'))

    @classmethod
    def from_json(cls, json_data: Union[str, dict], agent=None):
        """Reconstruct planning memory from JSON"""
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        memory = cls(agent=agent)
        memory.checkpoints = deque(data.get('checkpoints', []), maxlen=memory.config.max_checkpoints)
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
    print("")
    print("\n=== Running Planning Memory ===")
    print("")
    from unittest.mock import Mock
    mock_agent = Mock()
    mock_agent.shared_memory = {}
    memory = PlanningMemory(agent=mock_agent)
    print("")
    print("\n=== Successfully Ran Planning Memory ===\n")
