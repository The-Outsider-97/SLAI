import copy
import time

from collections import deque
from typing import Any, Dict, List, Optional

from src.agents.handler.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Handler Memory")
printer = PrettyPrinter

class HandlerMemory:
    """Checkpoint + telemetry memory for HandlerAgent decisions."""

    def __init__(self):
        self.config = load_global_config()

        self.memory_config = get_config_section('memory')
        self.max_checkpoints = self.memory.get("max_checkpoints", 100)
        self.max_telemetry_events = self.memory.get("max_telemetry_events", 1000)
        self._checkpoints = deque(maxlen=self.max_checkpoints)
        self._telemetry = deque(maxlen=self.max_telemetry_events)

        logger.info(f"Handler memory succesfully initialized")

    def save_checkpoint(self, label: str, state: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        checkpoint_id = f"{label}:{int(time.time() * 1000)}"
        self._checkpoints.append(
            {
                "id": checkpoint_id,
                "label": label,
                "created": time.time(),
                "state": copy.deepcopy(state),
                "metadata": metadata or {},
            }
        )
        return checkpoint_id

    def find_checkpoints(self, label: Optional[str] = None, max_age: Optional[float] = None) -> List[Dict[str, Any]]:
        now = time.time()
        results = []
        for cp in reversed(self._checkpoints):
            if label and cp["label"] != label:
                continue
            if max_age is not None and (now - cp["created"] > max_age):
                continue
            results.append(cp)
        return results

    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        for cp in reversed(self._checkpoints):
            if cp["id"] == checkpoint_id:
                return copy.deepcopy(cp["state"])
        return None

    def append_telemetry(self, event: Dict[str, Any]) -> None:
        self._telemetry.append(event)

    def recent_telemetry(self, limit: int = 100) -> List[Dict[str, Any]]:
        return list(self._telemetry)[-limit:]

if __name__ == "__main__":
    memory = HandlerMemory(config={"max_checkpoints": 2, "max_telemetry_events": 3})
    checkpoint_id = memory.save_checkpoint(
        label="smoke",
        state={"step": 1, "status": "ok"},
        metadata={"source": "__main__"},
    )
    restored = memory.restore_checkpoint(checkpoint_id)
    memory.append_telemetry({"event": "smoke_test", "ok": bool(restored)})

    print("HandlerMemory smoke test")
    print(f"checkpoint_id={checkpoint_id}")
    print(f"found_checkpoints={len(memory.find_checkpoints(label='smoke'))}")
    print(f"restored={restored}")
    print(f"recent_telemetry={memory.recent_telemetry(limit=1)}")
