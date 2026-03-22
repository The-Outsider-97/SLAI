import os
import json
import hashlib

from datetime import datetime
from typing import Any, Dict, Optional

from src.agents.reader.utils.config_loader import get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reader Memory")
printer = PrettyPrinter


class ReaderMemory:
    """Small memory utility for checkpointing and conversion cache."""

    def __init__(self):
        memory_cfg = get_config_section("reader_memory")
        self.checkpoint_dir = memory_cfg.get("checkpoint_dir", "tmp/reader/checkpoints")
        self.cache_dir = memory_cfg.get("cache_dir", "tmp/reader/cache")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        logger.info(f"Reader Memory successfully Initialized")

    def _cache_key(self, payload: Dict[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def write_checkpoint(self, step: str, data: Dict[str, Any]) -> str:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(self.checkpoint_dir, f"{ts}_{step}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def set_cache(self, key_payload: Dict[str, Any], value: Dict[str, Any]) -> str:
        key = self._cache_key(key_payload)
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(value, f)
        return cache_path

    def get_cache(self, key_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._cache_key(key_payload)
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        if not os.path.exists(cache_path):
            return None
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
