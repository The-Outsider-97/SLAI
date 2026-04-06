"""
Knowledge Synchronization System for SLAI
- Maintains consistency between memory and external sources
- Implements conflict resolution strategies
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import requests
import yaml

from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from difflib import SequenceMatcher

try:  # Optional dependency in some environments.
    import psycopg2
    from psycopg2 import sql
except Exception:  # pragma: no cover - exercised in dependency-light environments.
    psycopg2 = None
    sql = None

from src.agents.knowledge.utils.config_loader import load_global_config, get_config_section
from src.agents.knowledge.knowledge_memory import KnowledgeMemory
from src.agents.knowledge.utils.rule_engine import RuleEngine
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Knowledge Synchronizer")
printer = PrettyPrinter


class KnowledgeSynchronizer:
    """Orchestrates knowledge consistency across components with version control."""

    def __init__(
        self,
        knowledge_memory: Optional[KnowledgeMemory] = None,
        rule_engine: Optional[RuleEngine] = None,
        autostart: bool = False,
    ):
        self.config = load_global_config()
        self.enabled = bool(self.config.get("enabled", True))
        self.sync_config = get_config_section("knowledge_sync") or {}

        self.auto_sync = self._normalize_auto_sync(self.sync_config.get("auto_sync"))
        self.conflict_resolution = self._normalize_conflict_resolution(
            self.sync_config.get("conflict_resolution")
        )
        self.versioning = self._normalize_versioning(self.sync_config.get("versioning"))
        self.retry_policy = self._normalize_retry_policy(self.sync_config.get("retry_policy"))
        self.source_timeout_policy = self._normalize_source_timeout_policy(
            self.sync_config.get("source_timeouts")
        )
        self.external_fetch_workers = self._coerce_positive_int(
            self.sync_config.get("external_fetch_workers", 4), default=4
        )

        external_config = get_config_section("external_sources")
        self.external_config = self._normalize_external_sources(external_config)

        self.knowledge_memory = knowledge_memory if knowledge_memory is not None else KnowledgeMemory()
        self.rule_engine = rule_engine if rule_engine is not None else RuleEngine()

        self.version_history: "OrderedDict[str, deque]" = OrderedDict()
        self.sync_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.sync_thread: Optional[threading.Thread] = None
        self._sleep = time.sleep
        self._source_state: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"consecutive_failures": 0, "circuit_open_until": 0.0}
        )
        self._last_external_stats: Dict[str, int] = {
            "external_fetches": 0,
            "external_errors": 0,
            "external_successes": 0,
            "external_skipped": 0,
        }

        self.conflict_strategies = {
            "timestamp": self._resolve_by_timestamp,
            "confidence": self._resolve_by_confidence,
            "semantic": self._resolve_by_semantics,
            "governance": self._resolve_by_rules,
            "rule_based": self._resolve_by_rules,
        }

        if self.enabled and self.auto_sync.get("enabled") and autostart:
            self.start_sync()

        logger.info(
            "Knowledge Synchronizer initialized with auto_sync=%s, retry_policy=%s, workers=%s",
            self.auto_sync,
            self.retry_policy,
            self.external_fetch_workers,
        )

    def _normalize_auto_sync(self, value: Any) -> Dict[str, Any]:
        defaults = {"enabled": False, "interval": 300}
        if not isinstance(value, dict):
            return defaults
        return {
            "enabled": bool(value.get("enabled", defaults["enabled"])),
            "interval": self._coerce_positive_int(value.get("interval"), defaults["interval"]),
        }

    def _normalize_conflict_resolution(self, value: Any) -> Dict[str, Any]:
        defaults = {
            "strategy": "timestamp",
            "similarity_threshold": 0.85,
            "auto_quarantine": True,
        }
        if not isinstance(value, dict):
            return defaults
        return {
            "strategy": str(value.get("strategy", defaults["strategy"])).lower(),
            "similarity_threshold": self._coerce_float(
                value.get("similarity_threshold"), defaults["similarity_threshold"]
            ),
            "auto_quarantine": bool(value.get("auto_quarantine", defaults["auto_quarantine"])),
        }

    def _normalize_versioning(self, value: Any) -> Dict[str, Any]:
        defaults = {"enabled": True, "max_versions": 10}
        if not isinstance(value, dict):
            return defaults
        return {
            "enabled": bool(value.get("enabled", defaults["enabled"])),
            "max_versions": self._coerce_positive_int(
                value.get("max_versions"), defaults["max_versions"]
            ),
        }

    def _normalize_retry_policy(self, value: Any) -> Dict[str, Any]:
        defaults = {
            "max_attempts": 3,
            "base_delay_seconds": 0.5,
            "max_delay_seconds": 8.0,
            "circuit_breaker_failures": 3,
            "circuit_breaker_reset_seconds": 300.0,
        }
        if not isinstance(value, dict):
            return defaults
        normalized = dict(defaults)
        normalized["max_attempts"] = self._coerce_positive_int(
            value.get("max_attempts"), defaults["max_attempts"]
        )
        normalized["base_delay_seconds"] = self._coerce_float(
            value.get("base_delay_seconds"), defaults["base_delay_seconds"]
        )
        normalized["max_delay_seconds"] = self._coerce_float(
            value.get("max_delay_seconds"), defaults["max_delay_seconds"]
        )
        normalized["circuit_breaker_failures"] = self._coerce_positive_int(
            value.get("circuit_breaker_failures"), defaults["circuit_breaker_failures"]
        )
        normalized["circuit_breaker_reset_seconds"] = self._coerce_float(
            value.get("circuit_breaker_reset_seconds"), defaults["circuit_breaker_reset_seconds"]
        )
        return normalized

    def _normalize_source_timeout_policy(self, value: Any) -> Dict[str, float]:
        defaults = {"file": 5.0, "api": 10.0, "database": 15.0, "inline": 1.0}
        if not isinstance(value, dict):
            return defaults
        normalized = dict(defaults)
        for key in defaults:
            normalized[key] = self._coerce_float(value.get(key), defaults[key])
        return normalized

    def _normalize_external_sources(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [value]
        return [value]

    def _coerce_positive_int(self, value: Any, default: int) -> int:
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return default
        return coerced if coerced > 0 else default

    def _coerce_float(self, value: Any, default: float) -> float:
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return float(default)
        return coerced if coerced > 0 else float(default)

    def start_sync(self) -> bool:
        """Explicit orchestration barrier for background synchronization."""
        if not (self.enabled and self.auto_sync.get("enabled")):
            logger.info("Background sync not started because synchronization is disabled")
            return False

        if self.sync_thread and self.sync_thread.is_alive():
            return True

        self.stop_event.clear()

        def sync_loop():
            logger.info("Background sync thread started")
            while not self.stop_event.is_set():
                try:
                    start_time = time.time()
                    stats = self.full_sync()
                    duration = time.time() - start_time
                    logger.info(
                        "Sync completed in %.2fs. Updates=%s Conflicts=%s ExternalErrors=%s",
                        duration,
                        stats.get("memory_updates", 0),
                        stats.get("memory_conflicts", 0),
                        stats.get("external_errors", 0),
                    )
                except Exception as exc:  # pragma: no cover - defensive runtime protection
                    logger.error(f"Sync failed: {exc}", exc_info=True)

                self.stop_event.wait(self.auto_sync.get("interval", 300))

            logger.info("Background sync thread stopped")

        self.sync_thread = threading.Thread(target=sync_loop, daemon=True, name="knowledge-sync")
        self.sync_thread.start()
        return True

    def _start_sync_thread(self):
        """Backward-compatible wrapper for explicit thread start."""
        return self.start_sync()

    def resolve_conflict(self, key: str, *versions: Dict[str, Any]) -> Any:
        """Apply configured conflict resolution strategy."""
        printer.status("SYNC", "Apply configured conflict resolution strategy", "info")
        valid_versions = [version for version in versions if isinstance(version, dict)]
        if not valid_versions:
            raise ValueError(f"No valid versions supplied for conflict resolution on key='{key}'")

        strategy_name = self.conflict_resolution.get("strategy", "timestamp")
        strategy = self.conflict_strategies.get(strategy_name, self._resolve_by_timestamp)
        return strategy(key, *valid_versions)

    def _resolve_by_timestamp(self, key: str, *versions: Dict[str, Any]) -> Any:
        """Select most recent version based on timestamp."""
        printer.status("SYNC", "Resolving conflicts by timestamp", "info")
        return max(versions, key=lambda version: version.get("metadata", {}).get("timestamp", 0))

    def _resolve_by_confidence(self, key: str, *versions: Dict[str, Any]) -> Any:
        """Select version with highest confidence score."""
        printer.status("SYNC", "Resolving conflicts with highest confidence score", "info")
        return max(versions, key=lambda version: version.get("metadata", {}).get("confidence", 0))

    def _resolve_by_semantics(self, key: str, *versions: Dict[str, Any]) -> Any:
        """Resolve conflicts using semantic similarity consensus."""
        printer.status("SYNC", "Resolving conflicts using semantic similarity", "info")

        best_match = None
        highest_score = -1.0
        for version in versions:
            text = str(version.get("text", ""))
            other_texts = [str(candidate.get("text", "")) for candidate in versions if candidate is not version]
            avg_score = (
                sum(SequenceMatcher(None, text, other_text).ratio() for other_text in other_texts)
                / max(1, len(other_texts))
            )
            if avg_score > highest_score:
                highest_score = avg_score
                best_match = version

        return best_match or versions[0]

    def _resolve_by_rules(self, key: str, *versions: Dict[str, Any]) -> Any:
        """Resolve conflicts using rule engine inference."""
        printer.status("SYNC", "Resolving conflicts using rule engine inference", "info")

        kb = {
            f"version_{index}": {
                "text": version.get("text", ""),
                "metadata": version.get("metadata", {}),
            }
            for index, version in enumerate(versions)
        }

        try:
            inferred = self.rule_engine.smart_apply(kb)
        except Exception as exc:
            logger.warning(f"Rule-based conflict resolution failed for key='{key}': {exc}")
            return self._resolve_by_timestamp(key, *versions)

        best_version_key = max(inferred, key=lambda candidate: inferred[candidate], default=None)
        if best_version_key and best_version_key.startswith("version_"):
            try:
                version_index = int(best_version_key.split("_")[-1])
                if 0 <= version_index < len(versions):
                    return versions[version_index]
            except ValueError:
                logger.warning(f"Invalid inferred version key '{best_version_key}' for key='{key}'")

        return self._resolve_by_timestamp(key, *versions)

    def full_sync(self, components: Optional[List[str]] = None) -> Dict[str, int]:
        """Perform complete synchronization across specified components."""
        printer.status("SYNC", "Performing complete synchronization", "info")

        stats: Dict[str, int] = defaultdict(int)
        components = components or ["memory", "external", "rules"]

        with self.sync_lock:
            if "memory" in components:
                stats.update(self._sync_memory_with_external())
            if "rules" in components:
                stats.update(self._sync_rule_engine())
            if "external" in components:
                stats.update(self._sync_with_external_sources())
            if self.versioning.get("enabled", False):
                self._create_version_snapshot()

        return dict(stats)

    def _sync_memory_with_external(self) -> Dict[str, int]:
        """Synchronize core memory with external knowledge sources."""
        stats = {"memory_updates": 0, "memory_conflicts": 0}
        external_data, external_stats = self._fetch_external_data(include_stats=True)
        stats.update(external_stats)

        for key, external_value in external_data.items():
            memory_rows = self.knowledge_memory.recall(key=key)
            if not memory_rows:
                self.knowledge_memory.update(key, external_value)
                stats["memory_updates"] += 1
                continue

            current_value = memory_rows[0][0]
            if self._detect_conflict(current_value, external_value):
                stats["memory_conflicts"] += 1
                resolved = self.resolve_conflict(key, current_value, external_value)
                self.knowledge_memory.update(key, resolved)

        return stats

    def _sync_rule_engine(self) -> Dict[str, int]:
        """Refresh rule engine with latest rules."""
        stats = {"rules_loaded": 0, "errors": 0}
        try:
            self.rule_engine.load_all_sectors()
            stats["rules_loaded"] = len(getattr(self.rule_engine, "rules", []))
            logger.info("Reloaded %s rules into RuleEngine", stats["rules_loaded"])
        except Exception as exc:
            logger.error(f"Failed to sync rule engine: {exc}", exc_info=True)
            stats["errors"] = 1
        return stats

    def _sync_with_external_sources(self) -> Dict[str, int]:
        """Synchronize with configured external knowledge sources."""
        _, stats = self._fetch_external_data(include_stats=True)
        return stats

    def _fetch_external_data(
        self, include_stats: bool = False
    ) -> Tuple[Dict[str, dict], Dict[str, int]] | Dict[str, dict]:
        """Fetch and merge knowledge entries from configured external sources."""
        merged_data: Dict[str, dict] = {}
        stats = {
            "external_fetches": 0,
            "external_errors": 0,
            "external_successes": 0,
            "external_skipped": 0,
        }

        sources = list(self.external_config)
        if not sources:
            self._last_external_stats = stats
            return (merged_data, stats) if include_stats else merged_data

        max_workers = min(self.external_fetch_workers, max(1, len(sources)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._fetch_source_with_retry, source): source for source in sources
            }
            for future in as_completed(future_map):
                source = future_map[future]
                try:
                    source_data, source_stats = future.result()
                except Exception as exc:  # pragma: no cover - defensive safety
                    logger.error(f"Failed to load external source {source}: {exc}", exc_info=True)
                    stats["external_fetches"] += 1
                    stats["external_errors"] += 1
                    continue

                for metric_name, metric_value in source_stats.items():
                    stats[metric_name] = stats.get(metric_name, 0) + metric_value

                for key, value in source_data.items():
                    if not isinstance(value, dict):
                        value = {
                            "text": str(value),
                            "metadata": {"timestamp": time.time(), "confidence": 0.5},
                        }
                    value.setdefault("metadata", {})
                    value["metadata"]["source"] = self._source_identifier(source)
                    merged_data[key] = value

        self._last_external_stats = stats
        return (merged_data, stats) if include_stats else merged_data

    def _fetch_source_with_retry(self, source: Any) -> Tuple[Dict[str, dict], Dict[str, int]]:
        source_id = self._source_identifier(source)
        stats = {
            "external_fetches": 0,
            "external_errors": 0,
            "external_successes": 0,
            "external_skipped": 0,
        }

        if self._is_circuit_open(source_id):
            logger.warning(f"Skipping external source while circuit is open: {source_id}")
            stats["external_skipped"] = 1
            return {}, stats

        max_attempts = self._source_max_attempts(source)
        base_delay = self._source_base_delay_seconds(source)
        max_delay = self.retry_policy["max_delay_seconds"]

        for attempt in range(1, max_attempts + 1):
            stats["external_fetches"] += 1
            try:
                raw_data = self._fetch_source_once(source)
                processed = self._process_external_data(raw_data, source)
                self._register_source_success(source_id)
                stats["external_successes"] = 1
                return processed, stats
            except Exception as exc:
                logger.warning(
                    "Attempt %s/%s failed for source %s: %s",
                    attempt,
                    max_attempts,
                    source_id,
                    exc,
                )
                if attempt >= max_attempts:
                    self._register_source_failure(source_id)
                    stats["external_errors"] = 1
                    return {}, stats

                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                self._sleep(delay)

        stats["external_errors"] = 1
        return {}, stats

    def _source_identifier(self, source: Any) -> str:
        if isinstance(source, str):
            return source
        if isinstance(source, dict):
            for key in ("name", "endpoint", "connection_string", "path", "type"):
                value = source.get(key)
                if value:
                    return str(value)
        return repr(source)

    def _is_circuit_open(self, source_id: str) -> bool:
        state = self._source_state[source_id]
        return state["circuit_open_until"] > time.time()

    def _register_source_success(self, source_id: str) -> None:
        state = self._source_state[source_id]
        state["consecutive_failures"] = 0
        state["circuit_open_until"] = 0.0

    def _register_source_failure(self, source_id: str) -> None:
        state = self._source_state[source_id]
        state["consecutive_failures"] += 1
        if state["consecutive_failures"] >= self.retry_policy["circuit_breaker_failures"]:
            state["circuit_open_until"] = time.time() + self.retry_policy[
                "circuit_breaker_reset_seconds"
            ]

    def _source_max_attempts(self, source: Any) -> int:
        if isinstance(source, dict):
            retry_config = source.get("retry_policy")
            if isinstance(retry_config, dict):
                return self._coerce_positive_int(
                    retry_config.get("max_attempts"), self.retry_policy["max_attempts"]
                )
        return self.retry_policy["max_attempts"]

    def _source_base_delay_seconds(self, source: Any) -> float:
        if isinstance(source, dict):
            retry_config = source.get("retry_policy")
            if isinstance(retry_config, dict):
                return self._coerce_float(
                    retry_config.get("base_delay_seconds"),
                    self.retry_policy["base_delay_seconds"],
                )
        return self.retry_policy["base_delay_seconds"]

    def _source_timeout_seconds(self, source: Any, source_type: str) -> float:
        default_timeout = self.source_timeout_policy.get(source_type, 10.0)
        if isinstance(source, dict):
            return self._coerce_float(source.get("timeout_seconds"), default_timeout)
        return default_timeout

    def _fetch_source_once(self, source: Any) -> Dict[str, Any]:
        if isinstance(source, str):
            return self._fetch_from_file(source)

        if not isinstance(source, dict):
            raise TypeError(f"Unknown external source format: {source}")

        source_type = str(source.get("type", "inline")).lower()
        if source_type == "api":
            return self._fetch_from_api(source, timeout_seconds=self._source_timeout_seconds(source, "api"))
        if source_type == "database":
            return self._fetch_from_database(
                source, timeout_seconds=self._source_timeout_seconds(source, "database")
            )
        if source_type == "inline":
            data = source.get("data", {})
            if not isinstance(data, dict):
                raise TypeError("Inline source data must be a dictionary")
            return data
        if source_type == "file":
            path = source.get("path")
            if not path:
                raise ValueError("File source missing 'path'")
            return self._fetch_from_file(path)

        raise ValueError(f"Unknown source type: {source_type}")

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        config_path = self.config.get("__config_path__")
        search_roots: List[Path] = [Path.cwd()]
        if config_path:
            config_root = Path(config_path).resolve().parent
            search_roots.extend([config_root, config_root.parent])
        module_root = Path(__file__).resolve().parent
        search_roots.extend([module_root, module_root.parent, module_root.parent.parent])

        for root in search_roots:
            resolved = (root / path).resolve()
            if resolved.exists():
                return resolved

        return candidate

    def _fetch_from_file(self, path: str) -> dict:
        """Fetch data from local file source."""
        resolved = self._resolve_path(path)
        if str(resolved).endswith((".yaml", ".yml")):
            with open(resolved, "r", encoding="utf-8") as file_handle:
                return yaml.safe_load(file_handle) or {}
        if str(resolved).endswith(".json"):
            with open(resolved, "r", encoding="utf-8") as file_handle:
                return json.load(file_handle)
        logger.warning(f"Unsupported file type: {resolved}")
        return {}

    def _fetch_from_api(self, source_config: dict, timeout_seconds: float = 10.0) -> dict:
        """Fetch data from API endpoint."""
        endpoint = source_config.get("endpoint")
        if not endpoint:
            raise ValueError("API source missing endpoint")

        headers = {}
        auth_type = source_config.get("auth_type")
        if auth_type == "bearer_token":
            token = source_config.get("token")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        params = source_config.get("params", {})
        response = requests.get(endpoint, headers=headers, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError("API responses must be JSON objects for synchronization")
        return data

    def _fetch_from_database(self, source_config: dict, timeout_seconds: float = 15.0) -> dict:
        """Fetch data from database source."""
        conn_str = source_config.get("connection_string")
        tables = source_config.get("tables", [])
        if not conn_str or not tables:
            raise ValueError("Database source missing connection string or tables")
        if psycopg2 is None or sql is None:
            raise RuntimeError("psycopg2 is required for database synchronization")

        data: Dict[str, dict] = {}
        connect_kwargs = source_config.get("connect_kwargs", {})
        if not isinstance(connect_kwargs, dict):
            connect_kwargs = {}
        connect_kwargs.setdefault("connect_timeout", int(timeout_seconds))

        with psycopg2.connect(conn_str, **connect_kwargs) as connection:
            with connection.cursor() as cursor:
                for table in tables:
                    query = sql.SQL("SELECT * FROM {}") .format(sql.Identifier(table))
                    cursor.execute(query)
                    columns = [description[0] for description in cursor.description]
                    for row in cursor.fetchall():
                        key = f"{table}_{row[0]}"
                        data[key] = dict(zip(columns, row))
        return data

    def _process_external_data(self, data: dict, source: Any) -> dict:
        """Process and validate data from external sources."""
        if not isinstance(data, dict):
            raise TypeError(f"External source {source} returned {type(data).__name__}, expected dict")

        logger.info(f"Processing data from source: {source}, {len(data)} items")
        processed = {}
        for key, value in data.items():
            try:
                if not key or value is None:
                    logger.warning(f"Skipping invalid entry: key={key}, value={value}")
                    continue

                if not isinstance(value, dict):
                    value = {"text": str(value)}
                value.setdefault("metadata", {})
                value["metadata"].setdefault("timestamp", time.time())
                value["metadata"].setdefault("confidence", 0.5)
                processed[str(key)] = value
            except Exception as exc:
                logger.warning(f"Error processing item {key}: {exc}")

        logger.info(f"Processed {len(processed)} valid items from source: {source}")
        return processed

    def _create_version_snapshot(self) -> str:
        """Create versioned snapshot of current knowledge state and return version ID."""
        memory_state = {}
        for key in self.knowledge_memory.keys():
            rows = self.knowledge_memory.recall(key=key)
            if rows:
                memory_state[key] = rows[0][0]

        rule_state = [rule.get("name", "") for rule in getattr(self.rule_engine, "rules", [])]
        snapshot = {"memory": memory_state, "rules": rule_state, "timestamp": time.time()}

        version_payload = json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")
        version_id = hashlib.sha256(version_payload).hexdigest()[:16]
        self.version_history[version_id] = deque([snapshot], maxlen=1)
        self.version_history.move_to_end(version_id)

        max_versions = self.versioning.get("max_versions", 10)
        while len(self.version_history) > max_versions:
            self.version_history.popitem(last=False)

        return version_id

    def rollback_version(self, version_id: str, confirm: bool = False) -> bool:
        """Safe version rollback with confirmation and backup."""
        if version_id not in self.version_history:
            logger.error(f"Rollback failed: Version {version_id} not found")
            return False
        if not confirm:
            logger.warning("Rollback requires explicit confirmation")
            return False

        try:
            backup_id = self._create_version_snapshot()
            logger.info(f"Created backup version: {backup_id}")
            snapshot = self.version_history[version_id][-1]

            with self.sync_lock:
                self.knowledge_memory.clear()
                for key, value in snapshot["memory"].items():
                    self.knowledge_memory.update(key, value)
                self.rule_engine.load_all_sectors()

            logger.info(f"Successfully rolled back to version: {version_id}")
            return True
        except Exception as exc:  # pragma: no cover - defensive runtime protection
            logger.error(f"Rollback failed: {exc}", exc_info=True)
            return False

    def stop_sync(self):
        """Stop background synchronization."""
        self.stop_event.set()
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=5)

    def _detect_conflict(self, existing: Dict[str, Any], new: Dict[str, Any]) -> bool:
        """Robust conflict detection with multiple strategies."""
        if existing == new:
            return False

        similarity = self._calculate_content_similarity(existing, new)
        threshold = self.conflict_resolution.get("similarity_threshold", 0.7)

        existing_conf = existing.get("metadata", {}).get("confidence", 0.5)
        new_conf = new.get("metadata", {}).get("confidence", 0.5)
        confidence_diff = abs(existing_conf - new_conf)

        existing_time = existing.get("metadata", {}).get("timestamp", 0)
        new_time = new.get("metadata", {}).get("timestamp", 0)
        time_diff = abs(existing_time - new_time)

        if similarity < threshold:
            logger.debug(f"Conflict detected by similarity: {similarity:.2f} < {threshold}")
            return True
        if confidence_diff > 0.3:
            logger.debug(f"Conflict detected by confidence diff: {confidence_diff:.2f}")
            return True
        if time_diff > 86400 * 30:
            logger.debug(f"Conflict detected by time diff: {time_diff / 86400:.1f} days")
            return True
        return False

    def _calculate_content_similarity(self, item1: dict, item2: dict) -> float:
        """Calculate content similarity using multiple strategies."""
        text1 = str(item1.get("text", ""))
        text2 = str(item2.get("text", ""))
        if text1 and text2:
            return SequenceMatcher(None, text1, text2).ratio()

        keys = set(item1.keys()) | set(item2.keys())
        similarities = []
        for key in keys:
            if key == "metadata":
                continue
            value1 = str(item1.get(key, ""))
            value2 = str(item2.get(key, ""))
            if value1 and value2:
                similarities.append(SequenceMatcher(None, value1, value2).ratio())

        if similarities:
            return sum(similarities) / len(similarities)
        return 1.0 if item1 == item2 else 0.0


if __name__ == "__main__":  # pragma: no cover
    print("\n=== Knowledge Synchronizer Test ===")
    synchronizer = KnowledgeSynchronizer()
    printer.status("Initial sync:", synchronizer)
    printer.status("SYNC", synchronizer.start_sync(), "success")
    print("\n=== Synchronization Test Completed ===\n")
