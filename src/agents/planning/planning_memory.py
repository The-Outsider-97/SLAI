
import gzip
import copy
import os
import time
import copy
import json, yaml
import threading

from typing import Any, Deque, Dict, List, Optional, Union
from collections import deque, defaultdict

from src.agents.planning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Planning Memory")
printer = PrettyPrinter

class PlanningMemory:
    """Maintains planning state checkpoints and statistical memory with disk persistence."""

    # File naming constants
    CHECKPOINT_PREFIX = "checkpoint"
    MANIFEST_FILE = "checkpoints_manifest.json"

    def __init__(self, agent: Optional[Any]=None):
        self.config = load_global_config()
        self.monitor_snapshot = self.config.get('monitor_snapshot')
        self.memory_config = get_config_section('planning_memory')

        self.checkpoints_dir = self.memory_config.get('checkpoints_dir')
        self.max_checkpoints = self.memory_config.get('max_checkpoints')
        self.history_window = self.memory_config.get('history_window')
        self.retention_days = self.memory_config.get('retention_days')
        self.compression = self.memory_config.get('compression')
        self.auto_save_interval = self.memory_config.get('auto_save_interval')

        self.agent = agent
        self._base_state = self._init_base_state() if agent is None else None

        # In‑memory cache of recent checkpoints (most recent first)
        self.checkpoints: Deque[Dict[str, Any]] = deque(maxlen=self.max_checkpoints)

        # Thread safety
        self._lock = threading.RLock()

        # Auto‑save timer
        self._auto_save_timer: Optional[threading.Timer] = None
        self._last_auto_save_time: float = 0.0

        # Ensure directories and load latest checkpoint
        self.ensure_checkpoint_dir()
        self._load_manifest()
        self.load_latest_checkpoint()

        # Start auto‑save if interval > 0
        if self.auto_save_interval > 0:
            self._start_auto_save()

    def _init_base_state(self) -> Dict[str, Any]:
        """Create an empty base state for use when no agent is provided."""
        return {
            "task_library": {},
            "method_stats": defaultdict(lambda: {"success": 0, "total": 0, "avg_cost": 0.0}),
            "world_state": {},
            "execution_history": deque(maxlen=self.history_window),
            "plan_metrics": defaultdict(list),
        }

    @property
    def _state(self) -> Dict[str, Any]:
        """Return the active state (agent’s if available, else base)."""
        if self.agent is not None:
            return {
                "task_library": getattr(self.agent, "task_library", {}),
                "method_stats": getattr(self.agent, "method_stats", defaultdict()),
                "world_state": getattr(self.agent, "world_state", {}),
                "execution_history": getattr(self.agent, "execution_history", deque()),
                "plan_metrics": getattr(self.agent, "plan_metrics", defaultdict(list)),
            }
        return self._base_state

    # -------------------------------------------------------------------------
    # Directory & file management
    # -------------------------------------------------------------------------
    def ensure_checkpoint_dir(self) -> None:
        """Create the checkpoint directory if it does not exist."""
        try:
            os.makedirs(self.checkpoints_dir, exist_ok=True)
            logger.info(f"Checkpoint directory ready: {self.checkpoints_dir}")
        except Exception as e:
            logger.error(f"Failed to create checkpoint directory: {e}")
            raise

    def _get_checkpoint_path(self, timestamp: float) -> str:
        """Return the filesystem path for a checkpoint with the given timestamp."""
        filename = f"{self.CHECKPOINT_PREFIX}_{timestamp:.6f}.json"
        if self.compression:
            filename += ".gz"
        return os.path.join(self.checkpoints_dir, filename)

    def _get_manifest_path(self) -> str:
        """Return the path to the manifest file."""
        return os.path.join(self.checkpoints_dir, self.MANIFEST_FILE)

    def _read_manifest(self) -> List[Dict[str, Any]]:
        """Read the manifest file and return its content (list of checkpoint entries)."""
        path = self._get_manifest_path()
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if not raw:
                logger.warning("Manifest file was empty. Reinitializing manifest: %s", path)
                self._write_manifest([])
                return []
            payload = json.loads(raw)
            if not isinstance(payload, list):
                logger.warning("Manifest content was invalid (expected list). Reinitializing: %s", path)
                self._write_manifest([])
                return []
            return payload
        except Exception as e:
            logger.error(f"Failed to read manifest: {e}")
            return []

    def _write_manifest(self, entries: List[Dict[str, Any]]) -> None:
        """Write the manifest to disk."""
        path = self._get_manifest_path()
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception as e:
            logger.error(f"Failed to write manifest: {e}")
            raise

    # -------------------------------------------------------------------------
    # Checkpoint persistence
    # -------------------------------------------------------------------------
    def _save_checkpoint_to_disk(self, checkpoint: Dict[str, Any]) -> None:
        """
        Persist a checkpoint dictionary to a file.
        Handles compression if enabled.
        """
        path = self._get_checkpoint_path(checkpoint["timestamp"])
        try:
            # Serialize
            data = json.dumps(checkpoint, indent=2, default=self._json_serializer).encode("utf-8")

            if self.compression:
                with gzip.open(path, "wb") as f:
                    f.write(data)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(data.decode("utf-8"))

            logger.debug(f"Checkpoint saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint to {path}: {e}")
            raise

    def _load_checkpoint_from_disk(self, path: str) -> Dict[str, Any]:
        """
        Load a checkpoint from a file (supports compressed files).
        """
        try:
            if self.compression and path.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {path}: {e}")
            raise

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for objects that are not JSON serializable."""
        if isinstance(obj, deque):
            return list(obj)
        if isinstance(obj, defaultdict):
            return dict(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _prune_old_checkpoints(self) -> None:
        """
        Remove checkpoint files and manifest entries older than retention_days
        and keep only the most recent max_checkpoints.
        """
        entries = self._read_manifest()
        if not entries:
            return

        # Sort by timestamp descending (most recent first)
        entries.sort(key=lambda e: e["timestamp"], reverse=True)

        # Apply retention days filter
        cutoff = time.time() - (self.retention_days * 86400) if self.retention_days else 0
        to_keep = []
        for entry in entries:
            if self.retention_days and entry["timestamp"] < cutoff:
                # too old
                try:
                    os.remove(entry["path"])
                    logger.debug(f"Removed old checkpoint: {entry['path']}")
                except OSError:
                    pass
                continue
            to_keep.append(entry)

        # Apply max_checkpoints limit (keep most recent)
        if len(to_keep) > self.max_checkpoints:
            extra = to_keep[self.max_checkpoints :]
            for entry in extra:
                try:
                    os.remove(entry["path"])
                    logger.debug(f"Removed excess checkpoint: {entry['path']}")
                except OSError:
                    pass
            to_keep = to_keep[: self.max_checkpoints]

        # Write back manifest
        self._write_manifest(to_keep)

    def _update_manifest(self, checkpoint: Dict[str, Any]) -> None:
        """
        Add a checkpoint to the manifest and optionally prune old ones.
        """
        entries = self._read_manifest()
        # Remove any existing entry with the same timestamp (should not happen)
        entries = [e for e in entries if e["timestamp"] != checkpoint["timestamp"]]
        # Add new
        entries.append(
            {
                "timestamp": checkpoint["timestamp"],
                "label": checkpoint["label"],
                "path": self._get_checkpoint_path(checkpoint["timestamp"]),
                "metadata": checkpoint["metadata"],
            }
        )
        self._write_manifest(entries)

        # Prune after adding new
        self._prune_old_checkpoints()

    # -------------------------------------------------------------------------
    # Public checkpoint API
    # -------------------------------------------------------------------------
    def save_checkpoint(self, label: Optional[str] = None, metadata: Optional[dict] = None) -> Dict[str, Any]:
        """
        Capture current planning state and save it to disk and memory.

        Args:
            label: Optional identifier for the checkpoint.
            metadata: Additional user data to store.

        Returns:
            The checkpoint dictionary.
        """
        with self._lock:
            checkpoint = {
                "timestamp": time.time(),
                "label": label or "auto_save",
                "state": self._capture_agent_state(),
                "metadata": metadata or {},
            }

            # Save to disk
            self._save_checkpoint_to_disk(checkpoint)
            self._update_manifest(checkpoint)

            # Update in‑memory deque (most recent first)
            self.checkpoints.appendleft(checkpoint)

            logger.info(f"Saved planning checkpoint '{checkpoint['label']}'")
            self._last_auto_save_time = checkpoint["timestamp"]
            return checkpoint

    def load_checkpoint(self, index: int = -1) -> bool:
        """
        Restore planning state from a checkpoint in the in‑memory deque.

        Args:
            index: Index into the deque (0 = most recent, -1 = oldest).

        Returns:
            True if successful, False otherwise.
        """
        with self._lock:
            if not self.checkpoints:
                logger.warning("No checkpoints available to load")
                return False

            try:
                checkpoint = self.checkpoints[index]
                self._restore_agent_state(checkpoint["state"])
                logger.info(f"Loaded checkpoint '{checkpoint['label']}' from {checkpoint['timestamp']}")
                return True
            except (IndexError, KeyError) as e:
                logger.error(f"Failed to load checkpoint index {index}: {e}")
                return False

    def load_latest_checkpoint(self) -> bool:
        """
        Load the most recent checkpoint from the in‑memory deque.

        Returns:
            True if a checkpoint was loaded, False otherwise.
        """
        with self._lock:
            if not self.checkpoints:
                logger.info("No checkpoints available to load.")
                return False
            checkpoint = self.checkpoints[0]  # most recent first
            self._restore_agent_state(checkpoint["state"])
            logger.info(f"Loaded latest checkpoint '{checkpoint['label']}'")
            return True

    def _load_manifest(self) -> None:
        """
        Load all checkpoints from disk into the in‑memory deque.
        This is called once during initialization.
        """
        entries = self._read_manifest()
        if not entries:
            return
        # Sort descending (most recent first)
        entries.sort(key=lambda e: e["timestamp"], reverse=True)
        # Keep only up to max_checkpoints
        entries = entries[: self.max_checkpoints]
        self.checkpoints.clear()
        for entry in entries:
            try:
                cp = self._load_checkpoint_from_disk(entry["path"])
                self.checkpoints.append(cp)  # append in order (most recent first)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint from {entry['path']}: {e}")
        logger.info(f"Loaded {len(self.checkpoints)} checkpoints from manifest.")

    def _capture_agent_state(self) -> Dict[str, Any]:
        """Deep copy relevant agent state components."""
        state = self._state
        return {
            "task_library": copy.deepcopy(state["task_library"]),
            "method_stats": copy.deepcopy(state["method_stats"]),
            "world_state": copy.deepcopy(state["world_state"]),
            "execution_history": copy.deepcopy(state["execution_history"]),
            "plan_metrics": copy.deepcopy(state["plan_metrics"]),
        }

    def _restore_agent_state(self, state: Dict[str, Any]) -> None:
        """Restore state to the agent (or base state)."""
        if self.agent is not None:
            # Update agent attributes
            self.agent.task_library = state["task_library"]
            self.agent.method_stats = state["method_stats"]
            self.agent.world_state = state["world_state"]
            self.agent.execution_history = deque(state["execution_history"], maxlen=self.history_window)
            self.agent.plan_metrics = state["plan_metrics"]
        else:
            # Update base state
            self._base_state["task_library"] = state["task_library"]
            self._base_state["method_stats"] = state["method_stats"]
            self._base_state["world_state"] = state["world_state"]
            self._base_state["execution_history"] = deque(state["execution_history"], maxlen=self.history_window)
            self._base_state["plan_metrics"] = state["plan_metrics"]

    # -------------------------------------------------------------------------
    # Auto‑save
    # -------------------------------------------------------------------------
    def _start_auto_save(self) -> None:
        """Start a background timer that triggers auto‑saves periodically."""
        if self._auto_save_timer:
            self._auto_save_timer.cancel()
        self._auto_save_timer = threading.Timer(self.auto_save_interval, self._auto_save_worker)
        self._auto_save_timer.daemon = True
        self._auto_save_timer.start()

    def _auto_save_worker(self) -> None:
        """Worker called by the timer to perform an auto‑save and reschedule."""
        try:
            self.auto_save()
        except Exception as e:
            logger.error(f"Auto‑save failed: {e}")
        finally:
            # Reschedule
            if self.auto_save_interval > 0:
                self._start_auto_save()

    def auto_save(self) -> None:
        """
        Perform an automatic save if enough time has passed since the last auto‑save.
        """
        now = time.time()
        if now - self._last_auto_save_time >= self.auto_save_interval:
            self.save_checkpoint(label="auto_save")
            self._last_auto_save_time = now

    def stop_auto_save(self) -> None:
        """Stop the auto‑save timer."""
        if self._auto_save_timer:
            self._auto_save_timer.cancel()
            self._auto_save_timer = None

    # -------------------------------------------------------------------------
    # State query methods (consistent with the original API)
    # -------------------------------------------------------------------------
    def get_task_outcome(self, task_id: str) -> Optional[bool]:
        """
        Retrieve the actual outcome of a task.

        Args:
            task_id: Identifier of the task.

        Returns:
            True if the task succeeded, False if failed, None if not found.
        """
        history = self._state["execution_history"]
        for entry in history:
            if entry.get("task_id") == task_id:
                return entry.get("status") == "success"
        return None

    def get_method_success_rate(self, method_name: str) -> float:
        """
        Get historical success rate for a method.

        Args:
            method_name: Name of the method.

        Returns:
            Success rate between 0 and 1, or 0.0 if no data.
        """
        stats = self._state["method_stats"].get(method_name)
        if stats and stats["total"] > 0:
            return stats["success"] / stats["total"]
        return 0.0

    def get_min_duration(self, task_name: str) -> float:
        """
        Retrieve the minimum observed duration for a given task.

        Args:
            task_name: Name of the task.

        Returns:
            Minimum duration in seconds, or infinity if no data.
        """
        durations = []
        history = self._state["execution_history"]
        for entry in history:
            if entry.get("name") == task_name:
                start = entry.get("start_time")
                end = entry.get("end_time")
                if start is not None and end is not None and end > start:
                    durations.append(end - start)

        if durations:
            return min(durations)

        logger.warning(f"No valid duration data found for task '{task_name}'. Returning infinity.")
        return float("inf")

    def is_sequential_task(self, task: Dict[str, Any], min_length: int) -> bool:
        """
        Check if the task is part of a sequential pattern of length >= min_length.

        Args:
            task: Task dictionary with "name" and optionally "parent".
            min_length: Minimum sequence length to consider.

        Returns:
            True if the task appears in a chain of at least min_length tasks.
        """
        task_chain = self._get_task_ancestry(task)
        # Check memory for sequential patterns
        history = self._state["execution_history"]
        for entry in history:
            entry_chain = entry.get("task_chain", [])
            # Compare the last min_length elements of the chain
            if len(entry_chain) >= min_length and entry_chain[-min_length:] == task_chain[-min_length:]:
                return True
        return False

    def _get_task_ancestry(self, task: Dict[str, Any]) -> List[str]:
        """Get task hierarchy chain (root to leaf)."""
        chain = []
        current = task
        while current:
            name = current.get("name")
            if not name:
                logger.error(f"Missing 'name' in task: {current}")
                break
            chain.append(name)
            current = current.get("parent")
        return chain[::-1]  # root-first

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------
    def to_json(self, file_path: Optional[str] = None) -> Union[str, None]:
        """
        Serialize the current in‑memory checkpoints to JSON.

        Args:
            file_path: If provided, write to file; otherwise return JSON string.

        Returns:
            JSON string if file_path is None, otherwise None.
        """
        data = {"config": self.config, "checkpoints": list(self.checkpoints)}
        json_str = json.dumps(data, indent=2, default=self._json_serializer)
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            return None
        return json_str

    @classmethod
    def from_json(cls, json_data: Union[str, dict], agent: Optional[Any] = None) -> "PlanningMemory":
        """
        Reconstruct a PlanningMemory from JSON data.

        Args:
            json_data: JSON string or dict.
            agent: Agent to associate with the new memory.

        Returns:
            A new PlanningMemory instance.
        """
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        memory = cls(agent=agent)
        memory.checkpoints = deque(data.get("checkpoints", []), maxlen=memory.max_checkpoints)
        # Also restore the manifest? Not needed, but we could update.
        return memory

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Return memory consumption statistics.

        Returns:
            Dictionary with counts of various items.
        """
        state = self._state
        return {
            "checkpoints": len(self.checkpoints),
            "task_library": len(state["task_library"]),
            "method_stats": len(state["method_stats"]),
            "history_items": len(state["execution_history"]),
        }


if __name__ == "__main__":
    print("\n=== Running Planning Memory Test ===\n")
    printer.status("Init", "Planning Memory initialized", "success")

    memory = PlanningMemory()
    print(memory.get_memory_usage())
    print("\n=== Successfully Ran Planning Memory ===\n")
