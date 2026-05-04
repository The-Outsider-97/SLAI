import hashlib
import json
import threading
import time
import yaml

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.agents.knowledge.utils.config_loader import get_config_section, load_global_config
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Knowledge Monitor")
printer = PrettyPrinter


class KnowledgeMonitor:
    """Knowledge integrity monitor with lazy dependency wiring and explicit lifecycle control."""

    def __init__(
        self,
        agent: Any = None,
        knowledge_cache: Any = None,
        rule_engine: Any = None,
        governor: Any = None,
        perform_action: Any = None,
        autostart: bool = False,
    ):
        self.agent = agent
        self.config = load_global_config()
        self.monitor_config = get_config_section("knowledge_monitor") or {}
        self.enabled = bool(self.monitor_config.get("enabled", True))

        self._component_lock = threading.RLock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._knowledge_cache = knowledge_cache
        self._rule_engine = rule_engine
        self._governor = governor
        self._perform_action = perform_action

        self.academic_sources = self._load_academic_sources()
        self.integrity_hashes: Dict[str, Dict[str, str]] = {"papers": {}, "datasets": {}}
        self.validation_context: Dict[str, Any] = {"invalid_entries": [], "trust_scores": {}}
        self.last_run_at: Optional[float] = None

        if autostart and self.enabled:
            self.start_monitoring()

    @property
    def knowledge_cache(self):
        with self._component_lock:
            if self._knowledge_cache is not None:
                return self._knowledge_cache
            cache = getattr(self.agent, "knowledge_cache", None)
            if self._is_cache_like(cache):
                self._knowledge_cache = cache
                return self._knowledge_cache
            from src.agents.knowledge.knowledge_cache import KnowledgeCache

            self._knowledge_cache = KnowledgeCache()
            return self._knowledge_cache

    @property
    def rule_engine(self):
        with self._component_lock:
            if self._rule_engine is not None:
                return self._rule_engine
            engine = getattr(self.agent, "rule_engine", None)
            if self._is_rule_engine_like(engine):
                self._rule_engine = engine
                return self._rule_engine
            from src.agents.knowledge.utils.rule_engine import RuleEngine

            self._rule_engine = RuleEngine()
            return self._rule_engine

    @property
    def governor(self):
        with self._component_lock:
            if self._governor is not None:
                return self._governor
            governor = getattr(self.agent, "governor", None)
            if self._is_governor_like(governor):
                self._governor = governor
                return self._governor
            try:
                from src.agents.knowledge.governor import Governor

                self._governor = Governor(knowledge_agent=self.agent)
                if hasattr(self._governor, "agent"):
                    self._governor.agent = self.agent
            except Exception as exc:  # pragma: no cover
                logger.warning("Governor initialization skipped: %s", exc)
                self._governor = None
            return self._governor

    @property
    def perform_action(self):
        with self._component_lock:
            if self._perform_action is not None:
                return self._perform_action
            action_handler = getattr(self.agent, "perform_action", None)
            if self._is_action_handler_like(action_handler):
                self._perform_action = action_handler
                return self._perform_action
            try:
                from src.agents.knowledge.perform_action import PerformAction

                self._perform_action = PerformAction()
            except Exception as exc:  # pragma: no cover
                logger.warning("PerformAction initialization skipped: %s", exc)
                self._perform_action = None
            return self._perform_action

    @property
    def monitoring_active(self) -> bool:
        return self._monitor_thread is not None and self._monitor_thread.is_alive()

    def _is_mock_like(self, value: Any) -> bool:
        return value is not None and type(value).__module__.startswith("unittest.mock")

    def _is_cache_like(self, cache: Any) -> bool:
        if cache is None or self._is_mock_like(cache):
            return False
        return all(callable(getattr(cache, attr, None)) for attr in ("get", "set", "hash_query"))

    def _is_rule_engine_like(self, engine: Any) -> bool:
        if engine is None or self._is_mock_like(engine):
            return False
        rules = getattr(engine, "rules", None)
        return isinstance(rules, list) and (callable(getattr(engine, "apply", None)) or callable(getattr(engine, "smart_apply", None)) or rules == [])

    def _is_governor_like(self, governor: Any) -> bool:
        if governor is None or self._is_mock_like(governor):
            return False
        return callable(getattr(governor, "record_violations", None)) or callable(getattr(governor, "handle_emergency_alert", None))

    def _is_action_handler_like(self, action_handler: Any) -> bool:
        if action_handler is None or self._is_mock_like(action_handler):
            return False
        return callable(action_handler) or callable(getattr(action_handler, "perform", None)) or callable(getattr(action_handler, "execute", None))

    def _safe_size(self, value: Any) -> int:
        try:
            return len(value)
        except (TypeError, AttributeError):
            return 0


    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path

        candidates = [Path.cwd() / path]
        config_path = self.config.get("__config_path__") if isinstance(self.config, dict) else None
        if config_path:
            config_dir = Path(config_path).resolve().parent
            candidates.extend([config_dir / path, config_dir.parent / path, config_dir.parent.parent / path])

        module_dir = Path(__file__).resolve().parent
        candidates.append(module_dir / path)

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_academic_sources(self) -> Dict[str, Any]:
        sources = {
            "domains": {str(domain).strip().lower() for domain in self.monitor_config.get("allowed_academic_domains", []) if str(domain).strip()},
            "papers": [],
            "datasets": [],
        }

        for path_value in self.monitor_config.get("academic_source_paths", []):
            path = self._resolve_path(str(path_value))
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    if path.suffix.lower() in {".yaml", ".yml"}:
                        data = yaml.safe_load(handle) or {}
                    else:
                        data = json.load(handle) or {}
            except FileNotFoundError:
                logger.warning("Academic source file not found: %s", path)
                continue
            except (yaml.YAMLError, json.JSONDecodeError, OSError) as exc:
                logger.error("Academic source loading error from %s: %s", path, exc)
                continue

            if not isinstance(data, dict):
                logger.warning("Academic source file must contain a mapping: %s", path)
                continue

            for domain in data.get("domains", []):
                domain_text = str(domain).strip().lower()
                if domain_text:
                    sources["domains"].add(domain_text)

            sources["papers"].extend(self._normalize_papers(data.get("papers", []), source_path=str(path)))
            sources["datasets"].extend(self._normalize_datasets(data.get("datasets", []), source_path=str(path)))

        return sources

    def _normalize_papers(self, papers: List[Any], source_path: str) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in papers:
            if not isinstance(item, dict):
                logger.warning("Skipping malformed paper entry from %s: %r", source_path, item)
                continue
            normalized.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "source": str(item.get("source", "")).strip(),
                    "published": str(item.get("published", "")).strip(),
                    "doi": str(item.get("doi", "")).strip() or None,
                    "metadata": dict(item.get("metadata", {})) if isinstance(item.get("metadata", {}), dict) else {},
                    "source_path": source_path,
                }
            )
        return normalized

    def _normalize_datasets(self, datasets: List[Any], source_path: str) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in datasets:
            if not isinstance(item, dict):
                logger.warning("Skipping malformed dataset entry from %s: %r", source_path, item)
                continue
            normalized.append(
                {
                    "name": str(item.get("name", "")).strip(),
                    "version": str(item.get("version", "")).strip(),
                    "source": str(item.get("source", "")).strip(),
                    "metadata": dict(item.get("metadata", {})) if isinstance(item.get("metadata", {}), dict) else {},
                    "source_path": source_path,
                }
            )
        return normalized

    def start_monitoring(self) -> bool:
        if not self.enabled:
            logger.info("KnowledgeMonitor is disabled; monitoring thread not started")
            return False
        if self.monitoring_active:
            return False

        self._stop_event.clear()

        def monitor_loop():
            logger.info("Knowledge monitor thread started")
            while not self._stop_event.is_set():
                try:
                    self.run_monitoring_cycle()
                except Exception as exc:  # pragma: no cover
                    logger.error("Monitoring cycle failed: %s", exc, exc_info=True)
                self._stop_event.wait(max(int(self.monitor_config.get("check_interval", 3600)), 1))
            logger.info("Knowledge monitor thread stopped")

        self._monitor_thread = threading.Thread(target=monitor_loop, name="knowledge-monitor", daemon=True)
        self._monitor_thread.start()
        return True

    def _start_monitoring_thread(self):
        return self.start_monitoring()

    def stop_monitoring(self, timeout: float = 5.0) -> bool:
        self._stop_event.set()
        thread = self._monitor_thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        return thread is None or not thread.is_alive()

    def run_monitoring_cycle(self) -> Dict[str, Any]:
        compliance = self.check_academic_compliance()
        integrity = self.verify_data_integrity()
        self.last_run_at = time.time()
        return {"timestamp": self.last_run_at, "violations": compliance, "integrity": integrity}

    def check_academic_compliance(self) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []
        max_age_years = max(int(self.monitor_config.get("max_source_age", 5)), 0)
        max_age_delta = timedelta(days=365 * max_age_years)

        for entry in self.academic_sources.get("papers", []):
            source = entry.get("source", "")
            title = entry.get("title", "")
            if source and not self._is_valid_source(source):
                violations.append({"type": "invalid_domain", "entry": title, "source": source})

            published_text = entry.get("published")
            published_at = self._parse_date(published_text)
            if published_at is None and published_text:
                violations.append({"type": "invalid_publication_date", "entry": title, "published": published_text})
            elif published_at and (datetime.now(timezone.utc) - published_at) > max_age_delta:
                violations.append(
                    {
                        "type": "outdated_source",
                        "entry": title,
                        "age_years": max((datetime.now(timezone.utc) - published_at).days // 365, 0),
                    }
                )

            if self.monitor_config.get("crossref_checking", True) and entry.get("doi") and not self._is_valid_doi(entry["doi"]):
                violations.append({"type": "invalid_doi", "entry": title, "doi": entry["doi"]})

        if violations:
            logger.warning("Academic compliance violations detected: %s", len(violations))
            self._handle_violations(violations)
        return violations

    def _is_valid_source(self, url: str) -> bool:
        domain = urlparse(url).netloc.lower().strip()
        if not domain:
            return False
        allowed_domains = self.academic_sources.get("domains") or {
            str(domain).strip().lower() for domain in self.monitor_config.get("allowed_academic_domains", []) if str(domain).strip()
        }
        return any(domain == approved or domain.endswith(f".{approved}") for approved in allowed_domains)

    def verify_data_integrity(self) -> Dict[str, Any]:
        if not self.monitor_config.get("enable_data_integrity_checks", True):
            return {"enabled": False, "breaches": []}

        current_hashes = {
            "papers": {p.get("title", ""): self._generate_hash(p) for p in self.academic_sources.get("papers", []) if p.get("title")},
            "datasets": {d.get("name", ""): self._generate_hash(d) for d in self.academic_sources.get("datasets", []) if d.get("name")},
        }

        breaches: List[Dict[str, Any]] = []
        for asset_type in ["papers", "datasets"]:
            previous = self.integrity_hashes.get(asset_type, {})
            for name, new_hash in current_hashes[asset_type].items():
                old_hash = previous.get(name)
                if old_hash and new_hash != old_hash:
                    logger.critical("Data integrity breach detected in %s: %s", asset_type[:-1], name)
                    breaches.append({"type": "data_tampering", "asset_type": asset_type[:-1], "entry": name})

        self.integrity_hashes = current_hashes
        if breaches:
            self._handle_violations(breaches)
        return {
            "enabled": True,
            "breaches": breaches,
            "papers_checked": len(current_hashes["papers"]),
            "datasets_checked": len(current_hashes["datasets"]),
        }

    def _generate_hash(self, data: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def _handle_violations(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        action = str(self.monitor_config.get("violation_policy", "log")).lower()
        timestamp = time.time()
        violation_records = [
            {
                **violation,
                "timestamp": timestamp,
                "component": "KnowledgeMonitor",
                "severity": self._calculate_severity(violation.get("type", "")),
            }
            for violation in violations
        ]

        self._store_violation_records(violation_records)

        governor = self.governor
        if governor is not None and hasattr(governor, "record_violations"):
            try:
                governor.record_violations(violation_records)
            except Exception as exc:  # pragma: no cover
                logger.warning("Governor violation recording failed: %s", exc)

        if action == "log":
            for violation in violations:
                logger.info("Compliance issue: %s - %s", violation.get("type"), violation.get("entry", ""))
        elif action == "quarantine":
            self._quarantine_assets(violations)
        elif action == "alert":
            self._trigger_system_alert(violation_records)
        else:
            logger.warning("Unknown violation policy '%s'; defaulting to log", action)

        self._invalidate_affected_knowledge(violations)
        return violation_records

    def _store_violation_records(self, violation_records: List[Dict[str, Any]]) -> None:
        shared_memory = self._get_shared_memory()
        if isinstance(shared_memory, dict):
            shared_memory.setdefault("violations", []).extend(violation_records)
            return

        if shared_memory is not None and hasattr(shared_memory, "set") and hasattr(shared_memory, "get"):
            existing = shared_memory.get("violations") or []
            shared_memory.set("violations", existing + violation_records)

    def _calculate_severity(self, violation_type: str) -> int:
        thresholds = self.monitor_config.get("severity_thresholds", {}) or {}
        severity_map = {
            "invalid_domain": thresholds.get("critical", 3),
            "outdated_source": thresholds.get("high", 2),
            "data_tampering": thresholds.get("critical", 3),
            "invalid_publication_date": thresholds.get("medium", 1),
            "invalid_doi": thresholds.get("medium", 1),
        }
        return int(severity_map.get(violation_type, thresholds.get("medium", 1)))

    def _quarantine_assets(self, violations: List[Dict[str, Any]]) -> None:
        quarantined = []
        for violation in violations:
            entry_name = violation.get("entry")
            if violation.get("type") == "invalid_domain" and violation.get("source"):
                self.academic_sources["papers"] = [
                    paper for paper in self.academic_sources.get("papers", []) if paper.get("source") != violation["source"]
                ]
                if entry_name:
                    quarantined.append(entry_name)

            if entry_name:
                key = self.knowledge_cache.hash_query(entry_name)
                self.knowledge_cache.set(key, {"status": "quarantined", "entry": entry_name})

        if quarantined:
            logger.warning("Quarantined %s assets", len(quarantined))

    def _trigger_system_alert(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        shared_memory = self._get_shared_memory() or {}
        cache_store = getattr(self.knowledge_cache, "cache", None)
        rules = getattr(self.rule_engine, "rules", None)
        alert_report = {
            "trigger": "knowledge_violation",
            "violations": violations,
            "system_status": {
                "memory_usage": self._shared_get(shared_memory, "memory_usage"),
                "cache_health": self._safe_size(cache_store),
                "active_rules": self._safe_size(rules),
            },
        }

        governor = self.governor
        if governor is not None and hasattr(governor, "handle_emergency_alert"):
            try:
                governor.handle_emergency_alert(alert_report)
            except Exception as exc:  # pragma: no cover
                logger.warning("Governor emergency alert failed: %s", exc)

        if self.monitor_config.get("auto_flush_on_alert", True):
            self.knowledge_cache.flush_flagged_entries()
        return alert_report

    def _invalidate_affected_knowledge(self, violations: List[Dict[str, Any]]) -> None:
        invalid_entries = [violation["entry"] for violation in violations if violation.get("entry")]

        for entry in invalid_entries:
            key = self.knowledge_cache.hash_query(entry)
            cached = self.knowledge_cache.get(key)
            if isinstance(cached, dict):
                cached["validation_status"] = "suspect"
                self.knowledge_cache.set(key, cached)

        self.validation_context = {
            "invalid_entries": invalid_entries,
            "trust_scores": {entry: 0.2 for entry in invalid_entries},
        }
        if hasattr(self.rule_engine, "update_validation_context"):
            try:
                self.rule_engine.update_validation_context(**self.validation_context)
            except Exception as exc:  # pragma: no cover
                logger.warning("Rule engine validation-context update failed: %s", exc)

    def generate_academic_report(self) -> Dict[str, Any]:
        return {
            "timestamp": time.time(),
            "sources_checked": len(self.academic_sources.get("papers", [])) + len(self.academic_sources.get("datasets", [])),
            "valid_domains": sorted(self.academic_sources.get("domains", [])),
            "integrity_checksum": self._generate_hash(self.integrity_hashes),
            "monitoring_active": self.monitoring_active,
            "last_run_at": self.last_run_at,
            "validation_context": dict(self.validation_context),
        }

    def _get_shared_memory(self) -> Any:
        return getattr(self.agent, "shared_memory", None)

    def _shared_get(self, shared_memory: Any, key: str, default: Any = None) -> Any:
        if isinstance(shared_memory, dict):
            return shared_memory.get(key, default)
        if shared_memory is not None and hasattr(shared_memory, "get"):
            try:
                value = shared_memory.get(key)
                return default if value is None else value
            except Exception:
                return default
        return default

    def _parse_date(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y"):
            try:
                parsed = datetime.strptime(value, fmt)
                return parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _is_valid_doi(self, doi: str) -> bool:
        doi = doi.strip()
        return bool(doi and doi.startswith("10.") and "/" in doi)


if __name__ == "__main__":  # pragma: no cover
    from src.agents.handler_agent import HandlerAgent
    from src.agents.agent_factory import AgentFactory
    from src.agents.collaborative.shared_memory import SharedMemory

    memory = SharedMemory()
    factory = AgentFactory()

    print("\n=== Knowledge Monitor ===\n")
    agent = HandlerAgent(
        shared_memory=memory,
        agent_factory=factory,
    )
    agent.shared_memory = {}
    monitor = KnowledgeMonitor(agent=agent, autostart=False)
    monitor.run_monitoring_cycle()
    print("\n=== Successfully Monitored the Knowledge system ===\n")
