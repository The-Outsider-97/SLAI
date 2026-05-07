import gzip
import copy
import os
import time
import json
import threading
import tempfile

from typing import Any, Deque, Dict, List, Optional, Union
from collections import deque, defaultdict

from .utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Planning Memory")
printer = PrettyPrinter

class PlanningMemory:
    """Maintains planning state checkpoints and statistical memory with disk persistence."""

    # File naming constants
    CHECKPOINT_PREFIX = "checkpoint"
    MANIFEST_FILE = "checkpoints_manifest.json"

    def __init__(self, agent: Optional[Any] = None):
        self.config = load_global_config()
        self.monitor_snapshot = self.config.get('monitor_snapshot')
        self.memory_config = get_config_section('planning_memory')

        # ----- Safe extraction of config values with defaults and type conversion -----
        # checkpoints_dir: must be a non-empty string
        raw_checkpoints_dir = self.memory_config.get('checkpoints_dir')
        if raw_checkpoints_dir is None or not isinstance(raw_checkpoints_dir, str):
            self.checkpoints_dir = "src/agents/planning/checkpoints/"  # fallback
            logger.warning("Missing or invalid 'checkpoints_dir' in planning_memory config. Using default.")
        else:
            self.checkpoints_dir = raw_checkpoints_dir

        # max_checkpoints: must be int > 0
        raw_max = self.memory_config.get('max_checkpoints')
        if raw_max is None or not isinstance(raw_max, int) or raw_max <= 0:
            self.max_checkpoints = 100
            logger.warning("Missing or invalid 'max_checkpoints'. Using default 100.")
        else:
            self.max_checkpoints = raw_max

        # history_window: must be int >= 0
        raw_history = self.memory_config.get('history_window')
        if raw_history is None or not isinstance(raw_history, int) or raw_history < 0:
            self.history_window = 1000
            logger.warning("Missing or invalid 'history_window'. Using default 1000.")
        else:
            self.history_window = raw_history

        # retention_days: can be None or int (None means no age-based pruning)
        raw_retention = self.memory_config.get('retention_days')
        if raw_retention is None:
            self.retention_days = None
        elif isinstance(raw_retention, int) and raw_retention > 0:
            self.retention_days = raw_retention
        else:
            self.retention_days = None
            logger.warning("Invalid 'retention_days' (must be positive int). Disabling age-based pruning.")

        # compression: bool
        raw_compression = self.memory_config.get('compression')
        self.compression = bool(raw_compression) if raw_compression is not None else False

        # auto_save_interval: float or int > 0, or 0/None to disable
        raw_interval = self.memory_config.get('auto_save_interval')
        if raw_interval is None:
            self.auto_save_interval = 0.0  # disabled
        else:
            try:
                interval = float(raw_interval)
                self.auto_save_interval = interval if interval > 0 else 0.0
            except (ValueError, TypeError):
                self.auto_save_interval = 0.0
                logger.warning("Invalid 'auto_save_interval' (must be numeric). Auto-save disabled.")

        # default_duration_fallback: not in config originally, but used in get_min_duration
        raw_fallback = self.memory_config.get('default_duration_fallback')
        if raw_fallback is None:
            self.default_duration_fallback = 300.0
        else:
            try:
                self.default_duration_fallback = float(raw_fallback)
            except (ValueError, TypeError):
                self.default_duration_fallback = 300.0

        self._missing_duration_warned: set[str] = set()

        self.agent = agent
        # Always initialize _base_state as a dict (even if agent is provided, we keep it as fallback)
        self._base_state = self._init_base_state()

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
        """Create a base state dictionary (used when no agent is provided or as fallback)."""
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
        if not self.checkpoints_dir:
            logger.error("Checkpoint directory path is empty or None. Cannot create.")
            raise ValueError("checkpoints_dir is not configured properly.")
        try:
            os.makedirs(self.checkpoints_dir, exist_ok=True)
            logger.info(f"Checkpoint directory ready: {self.checkpoints_dir}")
        except Exception as e:
            logger.error(f"Failed to create checkpoint directory: {e}")
            raise

    def _get_checkpoint_path(self, timestamp: float) -> str:
        """Return the filesystem path for a checkpoint with the given timestamp."""
        if not self.checkpoints_dir:
            raise RuntimeError("checkpoints_dir is not set.")
        filename = f"{self.CHECKPOINT_PREFIX}_{timestamp:.6f}.json"
        if self.compression:
            filename += ".gz"
        return os.path.join(self.checkpoints_dir, filename)

    def _get_manifest_path(self) -> str:
        """Return the path to the manifest file."""
        if not self.checkpoints_dir:
            raise RuntimeError("checkpoints_dir is not set.")
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
        path_dir = os.path.dirname(path) or "."
        max_retries = 5
        retry_delay = 0.05
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path_dir,
                prefix=f"{os.path.basename(path)}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmp_path = f.name
                json.dump(entries, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            for attempt in range(max_retries):
                try:
                    os.replace(tmp_path, path)
                    break
                except PermissionError:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(retry_delay * (attempt + 1))
        except Exception as e:
            logger.error(f"Failed to write manifest: {e}")
            raise
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # -------------------------------------------------------------------------
    # Checkpoint persistence
    # -------------------------------------------------------------------------
    def _save_checkpoint_to_disk(self, checkpoint: Dict[str, Any]) -> None:
        """Persist a checkpoint dictionary to a file. Handles compression if enabled."""
        path = self._get_checkpoint_path(checkpoint["timestamp"])
        try:
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
        """Load a checkpoint from a file (supports compressed files)."""
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

        # Apply retention days filter (if retention_days is set)
        cutoff = time.time() - (self.retention_days * 86400) if self.retention_days is not None else 0
        to_keep = []
        for entry in entries:
            if self.retention_days is not None and entry["timestamp"] < cutoff:
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
            extra = to_keep[self.max_checkpoints:]
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
        """Add a checkpoint to the manifest and optionally prune old ones."""
        entries = self._read_manifest()
        # Remove any existing entry with the same timestamp (should not happen)
        entries = [e for e in entries if e["timestamp"] != checkpoint["timestamp"]]
        entries.append(
            {
                "timestamp": checkpoint["timestamp"],
                "label": checkpoint["label"],
                "path": self._get_checkpoint_path(checkpoint["timestamp"]),
                "metadata": checkpoint["metadata"],
            }
        )
        self._write_manifest(entries)
        self._prune_old_checkpoints()

    # -------------------------------------------------------------------------
    # Public checkpoint API
    # -------------------------------------------------------------------------
    def save_checkpoint(self, label: Optional[str] = None, metadata: Optional[dict] = None) -> Dict[str, Any]:
        """Capture current planning state and save it to disk and memory."""
        with self._lock:
            checkpoint = {
                "timestamp": time.time(),
                "label": label or "auto_save",
                "state": self._capture_agent_state(),
                "metadata": metadata or {},
            }
            self._save_checkpoint_to_disk(checkpoint)
            self._update_manifest(checkpoint)
            self.checkpoints.appendleft(checkpoint)
            logger.info(f"Saved planning checkpoint '{checkpoint['label']}'")
            self._last_auto_save_time = checkpoint["timestamp"]
            return checkpoint

    def load_checkpoint(self, index: int = -1) -> bool:
        """Restore planning state from a checkpoint in the in‑memory deque."""
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
        """Load the most recent checkpoint from the in‑memory deque."""
        with self._lock:
            if not self.checkpoints:
                logger.info("No checkpoints available to load.")
                return False
            checkpoint = self.checkpoints[0]  # most recent first
            self._restore_agent_state(checkpoint["state"])
            logger.info(f"Loaded latest checkpoint '{checkpoint['label']}'")
            return True

    def _load_manifest(self) -> None:
        """Load all checkpoints from disk into the in‑memory deque."""
        entries = self._read_manifest()
        if not entries:
            return
        entries.sort(key=lambda e: e["timestamp"], reverse=True)
        entries = entries[: self.max_checkpoints]
        self.checkpoints.clear()
        for entry in entries:
            try:
                cp = self._load_checkpoint_from_disk(entry["path"])
                self.checkpoints.append(cp)
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
            # Update base state (always a dict)
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
        # auto_save_interval is guaranteed to be a float > 0 here (checked in __init__)
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
            # Reschedule only if auto_save_interval is still > 0
            if self.auto_save_interval > 0:
                self._start_auto_save()

    def auto_save(self) -> None:
        """Perform an automatic save if enough time has passed since the last auto‑save."""
        if self.auto_save_interval <= 0:
            return
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
        """Retrieve the actual outcome of a task."""
        history = self._state["execution_history"]
        for entry in history:
            if entry.get("task_id") == task_id:
                return entry.get("status") == "success"
        return None

    def get_method_success_rate(self, method_name: str) -> float:
        """Get historical success rate for a method."""
        stats = self._state["method_stats"].get(method_name)
        if stats and stats["total"] > 0:
            return stats["success"] / stats["total"]
        return 0.0

    def get_min_duration(self, task_name: str) -> float:
        """Retrieve the minimum observed duration for a given task."""
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
        if task_name not in self._missing_duration_warned:
            logger.warning(
                "No valid duration data found for task '%s'. Using fallback %.1fs.",
                task_name,
                self.default_duration_fallback,
            )
            self._missing_duration_warned.add(task_name)
        return self.default_duration_fallback

    def is_sequential_task(self, task: Dict[str, Any], min_length: int) -> bool:
        """Check if the task is part of a sequential pattern of length >= min_length."""
        task_chain = self._get_task_ancestry(task)
        history = self._state["execution_history"]
        for entry in history:
            entry_chain = entry.get("task_chain", [])
            if len(entry_chain) >= min_length and entry_chain[-min_length:] == task_chain[-min_length:]:
                return True
        return False

    def _get_task_ancestry(self, task: Dict[str, Any]) -> List[str]:
        """Get task hierarchy chain (root to leaf)."""
        chain = []
        current = task
        visited = set()

        while isinstance(current, dict):
            current_id = id(current)
            if current_id in visited:
                logger.warning("Detected cycle in task ancestry; stopping traversal.")
                break
            visited.add(current_id)

            name = current.get("name")
            if isinstance(name, str) and name.strip():
                chain.append(name)
            else:
                logger.debug("Skipping unnamed task node in ancestry traversal: %s", current)

            current = current.get("parent")

        return chain[::-1]  # root-first

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------
    def to_json(self, file_path: Optional[str] = None) -> Union[str, None]:
        """Serialize the current in‑memory checkpoints to JSON."""
        data = {"config": self.config, "checkpoints": list(self.checkpoints)}
        json_str = json.dumps(data, indent=2, default=self._json_serializer)
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            return None
        return json_str

    @classmethod
    def from_json(cls, json_data: Union[str, dict], agent: Optional[Any] = None) -> "PlanningMemory":
        """Reconstruct a PlanningMemory from JSON data."""
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
        memory = cls(agent=agent)
        memory.checkpoints = deque(data.get("checkpoints", []), maxlen=memory.max_checkpoints)
        return memory

    def get_memory_usage(self) -> Dict[str, int]:
        """Return memory consumption statistics."""
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
