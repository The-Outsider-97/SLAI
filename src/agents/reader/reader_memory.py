from __future__ import annotations

import hashlib
import json

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.agents.reader.utils.config_loader import get_config_section
from src.agents.reader.utils.reader_error import PersistenceError
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Reader Memory")
printer = PrettyPrinter


class ReaderMemory:
    """Persistent checkpoint + cache store for Reader workflows."""

    def __init__(self):
        memory_cfg = get_config_section("reader_memory")
        self.checkpoint_dir = Path(memory_cfg.get("checkpoint_dir", "tmp/reader/checkpoints")).expanduser()
        self.cache_dir = Path(memory_cfg.get("cache_dir", "tmp/reader/cache")).expanduser()
        self.cache_ttl_seconds = int(memory_cfg.get("cache_ttl_seconds", 0))

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, payload: Dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            tmp_path.replace(path)
        except Exception as exc:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise PersistenceError("Failed to persist reader memory payload", {"path": str(path)}, exc) from exc

    def write_checkpoint(self, step: str, data: Dict[str, Any], run_id: Optional[str] = None) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        checkpoint_id = f"{timestamp}_{step}"
        if run_id:
            checkpoint_id = f"{run_id}_{checkpoint_id}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"

        payload = {
            "checkpoint_id": checkpoint_id,
            "step": step,
            "timestamp": timestamp,
            "payload": data,
        }
        self._atomic_write_json(checkpoint_path, payload)
        return str(checkpoint_path)

    def set_cache(self, key_payload: Dict[str, Any], value: Dict[str, Any]) -> str:
        key = self._cache_key(key_payload)
        cache_path = self.cache_dir / f"{key}.json"
        payload = {
            "key": key,
            "created_at": datetime.now(timezone.utc).timestamp(),
            "value": value,
        }
        self._atomic_write_json(cache_path, payload)
        return str(cache_path)

    def get_cache(self, key_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._cache_key(key_payload)
        cache_path = self.cache_dir / f"{key}.json"
        if not cache_path.exists():
            return None

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise PersistenceError("Failed to read cache payload", {"path": str(cache_path)}, exc) from exc

        created_at = float(payload.get("created_at", 0.0))
        if self.cache_ttl_seconds > 0 and created_at > 0:
            age = datetime.now(timezone.utc).timestamp() - created_at
            if age > self.cache_ttl_seconds:
                try:
                    cache_path.unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed removing stale cache file: %s", cache_path)
                return None

        return payload.get("value")
