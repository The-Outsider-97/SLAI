"""
The Governor module serves as a policy enforcement and ethics auditing layer.
It integrates tightly with the KnowledgeMemory and RuleEngine modules to:
    - Apply Ethical Guidelines
    - Filter and Approve Rules
    - Audit Memory and Agent Behavior
    - Violation Detection
    - Emergency Handling
    - Bias Detection
    - Reporting & Monitoring
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict, deque
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import hashlib
import json
import os
import re
import threading
import time

import numpy as np
import pandas as pd
import yaml

from src.agents.alignment.bias_detection import BiasDetector
from src.agents.knowledge.utils.config_loader import load_global_config, get_config_section
from src.agents.knowledge.utils.rule_engine import RuleEngine
from src.agents.knowledge.knowledge_memory import KnowledgeMemory
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Governor")
printer = PrettyPrinter


class DotDict(dict):
    """Dictionary with dot access (safe SimpleNamespace replacement)."""

    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]


class Governor:
    def __init__(self, knowledge_agent=None):
        self.knowledge_agent = knowledge_agent
        self.config = load_global_config() or {}
        self.enabled = bool(self.config.get("enabled", True))

        self.governor_config = get_config_section("governor") or {}
        self.audit_interval = self._safe_int(self.governor_config.get("audit_interval"), 300, minimum=1)
        self.guideline_paths = self._ensure_list(self.governor_config.get("guideline_paths"))
        self.enforcement_mode = str(self.governor_config.get("enforcement_mode", "log")).lower()
        self.max_audit_history = self._safe_int(
            self.governor_config.get("max_audit_history"), 100, minimum=1
        )
        self.realtime_monitoring = bool(self.governor_config.get("realtime_monitoring", False))
        self._freshness_threshold = self._safe_int(
            self.governor_config.get("freshness_threshold"), 720, minimum=0
        )
        self.sensitive_attributes = self._ensure_list(
            self.governor_config.get("sensitive_attributes")
        )
        self.rule_complexity_threshold = self._safe_int(
            self.governor_config.get("rule_complexity_threshold"), 5, minimum=0
        )
        self.bias_categories_path = self.governor_config.get("bias_categories")
        self.monitoring_interval_seconds = self._safe_int(
            self.governor_config.get("monitoring_interval_seconds"), 60, minimum=1
        )

        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        config_path_value = self.config.get("__config_path__")
        self.config_path = Path(config_path_value).resolve() if config_path_value else None

        self.violation_thresholds = DotDict(
            {
                "unethical": 0.65,
                "similarity": 0.85,
                "consecutive_errors": 5,
                "critical": 0.8,
            }
        )
        violation_config = self.governor_config.get("violation_thresholds", {})
        if isinstance(violation_config, Mapping):
            self.violation_thresholds.update(violation_config)

        self.memory_thresholds = DotDict({"warning": 2048, "critical": 3072})
        memory_config = self.governor_config.get("memory_thresholds", {})
        if isinstance(memory_config, Mapping):
            self.memory_thresholds.update(memory_config)

        self.rule_engine = RuleEngine()
        self.knowledge_memory = KnowledgeMemory()
        self._bias_detector = None
        self.guidelines = self._load_guidelines()
        self.bias_categories = self._load_bias_categories()
        self.audit_history = deque(maxlen=self.max_audit_history)
        self.last_audit = time.time()
        self._monitoring_thread: Optional[threading.Thread] = None

        self._init_knowledge_memory()

        if self.realtime_monitoring:
            self._start_monitoring_thread()

        logger.info("Governor initialized")

    @property
    def freshness_threshold(self) -> int:
        return self._freshness_threshold

    def _safe_int(self, value: Any, default: int, minimum: Optional[int] = None) -> int:
        try:
            converted = int(value)
        except (TypeError, ValueError):
            converted = default
        if minimum is not None:
            converted = max(minimum, converted)
        return converted

    def _ensure_list(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _resolve_path(self, raw_path: Any) -> Optional[Path]:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None

        candidate = Path(raw_path)
        candidate_paths = []
        if candidate.is_absolute():
            candidate_paths.append(candidate)
        else:
            candidate_paths.append(Path.cwd() / candidate)
            if self.config_path is not None:
                candidate_paths.append(self.config_path.parent / candidate)
            candidate_paths.append(self.project_root / candidate)
            candidate_paths.append(Path(__file__).resolve().parent / candidate)

        for path in candidate_paths:
            if path.exists():
                return path.resolve()

        return candidate_paths[0].resolve() if candidate_paths else None

    def _load_bias_categories(self) -> dict:
        """Load bias categories from JSON/YAML file specified in config."""
        resolved_path = self._resolve_path(self.bias_categories_path)
        if resolved_path is None:
            logger.warning("No bias_categories path configured in governor")
            return {}

        try:
            with open(resolved_path, "r", encoding="utf-8") as file_handle:
                if resolved_path.suffix.lower() in {".yaml", ".yml"}:
                    data = yaml.safe_load(file_handle)
                else:
                    data = json.load(file_handle)
        except FileNotFoundError:
            logger.error(f"Bias categories file not found: {resolved_path}")
            return {}
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            logger.error(f"Invalid bias categories file {resolved_path}: {exc}")
            return {}
        except OSError as exc:
            logger.error(f"Error loading bias categories from {resolved_path}: {exc}")
            return {}

        return data if isinstance(data, dict) else {}

    def _iter_rule_source_paths(self) -> List[Path]:
        rule_files: List[Any] = []
        if self.knowledge_agent is not None and hasattr(self.knowledge_agent, "config"):
            agent_config = getattr(self.knowledge_agent, "config", {})
            if isinstance(agent_config, Mapping):
                rule_files = self._ensure_list(agent_config.get("rule_files"))
        if not rule_files:
            rule_engine_config = get_config_section("rule_engine") or {}
            if isinstance(rule_engine_config, Mapping):
                rule_files = self._ensure_list(rule_engine_config.get("rule_sources"))

        resolved_paths: List[Path] = []
        for rule_file in rule_files:
            resolved = self._resolve_path(rule_file)
            if resolved is None or not resolved.exists():
                logger.warning(f"Rule file not found: {rule_file}")
                continue
            resolved_paths.append(resolved)
        return resolved_paths

    def _normalize_rule(self, rule: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(rule, Mapping):
            return None

        rule_dict = dict(rule)
        name = str(rule_dict.get("name") or rule_dict.get("id") or "").strip()
        description = str(rule_dict.get("description") or name)
        action = str(rule_dict.get("action") or "")
        conditions = rule_dict.get("conditions", [])
        if conditions is None:
            conditions = []
        if not isinstance(conditions, list):
            conditions = [conditions]

        if not name and not description:
            return None

        rule_dict["conditions"] = conditions
        rule_dict["description"] = description
        rule_dict["action"] = action
        rule_dict.setdefault("id", hashlib.md5(description.encode("utf-8")).hexdigest())
        return rule_dict

    def _store_rules_in_memory(self, rules: List[Dict[str, Any]]) -> None:
        if not rules:
            return

        if hasattr(self.knowledge_memory, "add_all"):
            self.knowledge_memory.add_all(rules)
            return

        if hasattr(self.knowledge_memory, "update"):
            for rule in rules:
                key = rule.get("id") or rule.get("name")
                if key:
                    self.knowledge_memory.update(
                        key=key,
                        value=rule,
                        metadata={"type": "system_rule"},
                    )

    def _init_knowledge_memory(self):
        """Initialize knowledge memory with governor-specific rules."""
        rules: List[Dict[str, Any]] = []

        for path in self._iter_rule_source_paths():
            try:
                with open(path, "r", encoding="utf-8") as file_handle:
                    raw_data = json.load(file_handle)
            except (OSError, json.JSONDecodeError) as exc:
                logger.error(f"Failed to parse rule file {path}: {exc}", exc_info=True)
                continue

            if isinstance(raw_data, list):
                candidate_rules = raw_data
            elif isinstance(raw_data, Mapping) and isinstance(raw_data.get("rules"), list):
                candidate_rules = raw_data.get("rules", [])
            else:
                logger.warning(f"Expected list or {{'rules': [...]}} in {path}, got {type(raw_data)}")
                continue

            normalized_rules = []
            for rule in candidate_rules:
                normalized = self._normalize_rule(rule)
                if normalized is None:
                    logger.warning(f"Skipping malformed rule in {path}: {rule}")
                    continue
                normalized_rules.append(normalized)

            rules.extend(normalized_rules)
            logger.info(f"Loaded {len(normalized_rules)} rules from {path}")

        self._store_rules_in_memory(rules)

    def _normalize_pattern_list(self, value: Any) -> List[str]:
        patterns: List[str] = []
        for item in self._ensure_list(value):
            if isinstance(item, str) and item.strip():
                patterns.append(item.strip())
        return patterns

    def _normalize_tag_list(self, value: Any) -> List[str]:
        tags: List[str] = []
        for item in self._ensure_list(value):
            if isinstance(item, str) and item.strip():
                tags.append(item.strip().lower())
        return tags

    def _normalize_guideline_entry(
        self,
        entry: Any,
        bucket: str,
        source: str,
        index: int,
    ) -> Optional[Dict[str, Any]]:
        if isinstance(entry, str):
            description = entry.strip()
            if not description:
                return None
            entry_dict: Dict[str, Any] = {
                "id": hashlib.md5(f"{bucket}:{source}:{description}".encode("utf-8")).hexdigest(),
                "description": description,
                "patterns": [],
                "tags": [],
                "type": bucket[:-1],
                "source": source,
            }
            return entry_dict

        if not isinstance(entry, Mapping):
            logger.warning(f"Skipping invalid guideline entry from {source}: {entry}")
            return None

        description = str(
            entry.get("description")
            or entry.get("text")
            or entry.get("name")
            or entry.get("title")
            or ""
        ).strip()
        entry_id = str(entry.get("id") or entry.get("name") or "").strip()
        tags = self._normalize_tag_list(entry.get("tags"))
        patterns = self._normalize_pattern_list(entry.get("patterns"))

        if not patterns:
            patterns.extend(self._normalize_pattern_list(entry.get("keywords")))
        if not patterns:
            patterns.extend(self._normalize_pattern_list(entry.get("regex")))
        forbidden_content = entry.get("forbidden_content")
        if isinstance(forbidden_content, str) and forbidden_content.strip():
            patterns.append(forbidden_content.strip())

        entry_type = str(entry.get("type") or bucket[:-1]).strip().lower()
        if not entry_id:
            seed = description or ",".join(patterns) or f"{bucket}:{index}"
            entry_id = hashlib.md5(f"{bucket}:{source}:{seed}".encode("utf-8")).hexdigest()
        if not description:
            description = entry_id

        normalized = {
            "id": entry_id,
            "description": description,
            "patterns": patterns,
            "tags": tags,
            "type": entry_type,
            "source": source,
        }
        if isinstance(forbidden_content, str) and forbidden_content.strip():
            normalized["forbidden_content"] = forbidden_content.strip()
        return normalized

    def _merge_guideline_entries(
        self,
        target: Dict[str, Dict[str, Any]],
        entry: Dict[str, Any],
    ) -> None:
        existing = target.get(entry["id"])
        if existing is None:
            target[entry["id"]] = entry
            return

        existing["patterns"] = list(
            dict.fromkeys(existing.get("patterns", []) + entry.get("patterns", []))
        )
        existing["tags"] = list(dict.fromkeys(existing.get("tags", []) + entry.get("tags", [])))
        if not existing.get("description") and entry.get("description"):
            existing["description"] = entry["description"]
        if not existing.get("forbidden_content") and entry.get("forbidden_content"):
            existing["forbidden_content"] = entry["forbidden_content"]

    def _classify_general_guideline_item(self, entry: Any) -> str:
        if not isinstance(entry, Mapping):
            return "principles"
        entry_type = str(entry.get("type") or "").lower()
        tags = set(self._normalize_tag_list(entry.get("tags")))
        if entry_type in {"restriction", "prohibition", "blocked_action"}:
            return "restrictions"
        if {"restriction", "prohibition", "blocked"} & tags:
            return "restrictions"
        return "principles"

    def _extract_guideline_buckets(self, data: Any) -> Dict[str, List[Any]]:
        buckets = {"principles": [], "restrictions": []}

        if isinstance(data, list):
            for item in data:
                bucket = self._classify_general_guideline_item(item)
                buckets[bucket].append(item)
            return buckets

        if not isinstance(data, Mapping):
            return buckets

        principles_keys = ("principles", "ethical_principles")
        restrictions_keys = ("restrictions", "prohibitions", "safety_restrictions", "blocked_actions")

        for key in principles_keys:
            buckets["principles"].extend(self._ensure_list(data.get(key)))
        for key in restrictions_keys:
            buckets["restrictions"].extend(self._ensure_list(data.get(key)))

        guidelines_section = data.get("guidelines")
        if isinstance(guidelines_section, list):
            for item in guidelines_section:
                bucket = self._classify_general_guideline_item(item)
                buckets[bucket].append(item)
        elif isinstance(guidelines_section, Mapping):
            nested = self._extract_guideline_buckets(guidelines_section)
            buckets["principles"].extend(nested["principles"])
            buckets["restrictions"].extend(nested["restrictions"])

        if not buckets["principles"] and not buckets["restrictions"]:
            if any(key in data for key in ("patterns", "keywords", "description", "text", "title", "name")):
                bucket = self._classify_general_guideline_item(data)
                buckets[bucket].append(data)

        return buckets

    def _load_guidelines(self) -> Dict[str, list]:
        """Load, validate, and merge ethical guidelines from configured paths."""
        merged_principles: Dict[str, Dict[str, Any]] = {}
        merged_restrictions: Dict[str, Dict[str, Any]] = {}
        sources_loaded: List[str] = []
        invalid_entries = 0

        for raw_path in self.guideline_paths:
            resolved_path = self._resolve_path(raw_path)
            if resolved_path is None or not resolved_path.exists():
                logger.warning(f"Guideline file not found: {raw_path}")
                continue

            try:
                with open(resolved_path, "r", encoding="utf-8") as file_handle:
                    if resolved_path.suffix.lower() in {".yaml", ".yml"}:
                        data = yaml.safe_load(file_handle)
                    else:
                        data = json.load(file_handle)
            except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
                logger.error(f"Guideline loading error from {resolved_path}: {exc}")
                continue

            sources_loaded.append(str(resolved_path))
            buckets = self._extract_guideline_buckets(data)

            for index, principle in enumerate(buckets["principles"]):
                normalized = self._normalize_guideline_entry(
                    principle, "principles", str(resolved_path), index
                )
                if normalized is None:
                    invalid_entries += 1
                    continue
                self._merge_guideline_entries(merged_principles, normalized)

            for index, restriction in enumerate(buckets["restrictions"]):
                normalized = self._normalize_guideline_entry(
                    restriction, "restrictions", str(resolved_path), index
                )
                if normalized is None:
                    invalid_entries += 1
                    continue
                if normalized.get("type") == "principle":
                    normalized["type"] = "restriction"
                self._merge_guideline_entries(merged_restrictions, normalized)

        guidelines = {
            "principles": list(merged_principles.values()),
            "restrictions": list(merged_restrictions.values()),
            "sources_loaded": sources_loaded,
            "invalid_entries": invalid_entries,
        }
        logger.info(
            "Loaded %d principles and %d restrictions from %d guideline source(s)",
            len(guidelines["principles"]),
            len(guidelines["restrictions"]),
            len(sources_loaded),
        )
        return guidelines

    def _get_bias_detector(self):
        """Lazy initializer for BiasDetector."""
        if self._bias_detector is None and self.sensitive_attributes:
            try:
                self._bias_detector = BiasDetector()
            except Exception as exc:  # pragma: no cover - defensive path
                logger.error(f"Failed to initialize BiasDetector: {exc}")
                self._bias_detector = None
        return self._bias_detector

    def audit_model_predictions(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        labels: Optional[np.ndarray] = None,
        context: Optional[dict] = None,
    ) -> dict:
        """
        Audit model predictions using advanced bias detection.
        Returns audit report with bias metrics.
        """
        detector = self._get_bias_detector()
        if not detector:
            return {"error": "Bias detector not initialized - check sensitive_attributes config"}

        try:
            report = detector.compute_metrics(data, predictions, labels)
            audit_entry = {
                "timestamp": time.time(),
                "type": "model_bias_audit",
                "context": context or {},
                "report": report,
            }
            self.audit_history.append(audit_entry)
            return audit_entry
        except Exception as exc:
            logger.error(f"Bias detection failed: {exc}")
            return {"error": str(exc)}

    def _memory_recall(self, **kwargs) -> List[Any]:
        recall_fn = getattr(self.knowledge_memory, "recall", None)
        if callable(recall_fn):
            return recall_fn(**kwargs)
        return []

    def get_approved_rules(self) -> List[Dict]:
        """
        Retrieve approved rules from multiple sources:
        1. Pre-configured rule sources
        2. Manually approved rules in knowledge memory
        3. System-generated rules that meet approval thresholds
        """
        approved_rules: List[Dict[str, Any]] = []

        rule_sources = self._memory_recall(filters={"type": "rule_source"})
        for value, _metadata in rule_sources:
            if isinstance(value, Mapping) and isinstance(value.get("rules"), list):
                approved_rules.extend([rule for rule in value["rules"] if isinstance(rule, Mapping)])

        manual_rules = self._memory_recall(
            filters={"type": "approved_rule", "approval_status": "approved"}
        )
        for value, _metadata in manual_rules:
            if isinstance(value, Mapping):
                approved_rules.append(dict(value))

        system_rules = self._memory_recall(
            filters={"type": "system_rule"},
            sort_by="confidence",
            top_k=10,
        )
        rule_engine_config = self.config.get("rule_engine", {})
        min_confidence = 0.7
        if isinstance(rule_engine_config, Mapping):
            try:
                min_confidence = float(rule_engine_config.get("min_rule_confidence", 0.7))
            except (TypeError, ValueError):
                min_confidence = 0.7

        for rule, metadata in system_rules:
            if not isinstance(rule, Mapping):
                continue
            confidence = metadata.get("confidence", 0) if isinstance(metadata, Mapping) else 0
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = 0.0
            if confidence_value >= min_confidence:
                approved_rules.append(dict(rule))

        deduped_rules: Dict[str, Dict[str, Any]] = {}
        for rule in approved_rules:
            normalized = self._normalize_rule(rule)
            if normalized is None:
                continue
            deduped_rules[normalized["id"]] = normalized

        filtered_rules = [
            rule for rule in deduped_rules.values() if self._rule_passes_governance(rule)
        ]
        logger.info(
            f"Retrieved {len(filtered_rules)} approved rules after governance filtering"
        )
        return filtered_rules

    def _rule_passes_governance(self, rule: Dict) -> bool:
        """Check if a rule meets governance requirements."""
        if not isinstance(rule, Mapping):
            logger.warning(f"Rejected non-mapping rule during governance check: {rule}")
            return False

        complexity_threshold = self._safe_int(self.rule_complexity_threshold, 5, minimum=0)
        conditions = rule.get("conditions", [])
        if conditions is None:
            conditions = []
        elif not isinstance(conditions, list):
            logger.warning(
                f"Rejected rule {rule.get('id', 'unknown')} due to invalid conditions type: {type(conditions)}"
            )
            return False

        if len(conditions) > complexity_threshold:
            logger.warning(f"Rule {rule.get('id')} exceeds complexity threshold")
            return False

        description = str(rule.get("description") or "")
        action = str(rule.get("action") or "")

        for principle in self.guidelines.get("principles", []):
            if not isinstance(principle, Mapping):
                continue
            principle_tags = set(self._normalize_tag_list(principle.get("tags")))
            principle_type = str(principle.get("type") or "").lower()
            patterns = self._normalize_pattern_list(principle.get("patterns"))

            conflicts = principle_type == "prohibition" or "conflicts" in principle_tags
            if conflicts and self._matches_any_pattern(description, patterns):
                logger.warning(
                    f"Rule conflicts with principle {principle.get('id', 'unknown')}: {rule.get('id')}"
                )
                return False

        for restriction in self.guidelines.get("restrictions", []):
            if not isinstance(restriction, Mapping):
                continue
            patterns = self._normalize_pattern_list(restriction.get("patterns"))
            if self._matches_any_pattern(action, patterns):
                logger.warning(
                    f"Rule {rule.get('id')} violates restriction {restriction.get('id', 'unknown')}"
                )
                return False

        logger.debug(
            f"Evaluating rule {rule.get('id')} - description: {rule.get('description')}"
        )
        return True

    def _create_restriction_func(self, restriction: dict):
        """Generate rule function from restriction definition."""
        restriction_id = "unknown_restriction"
        if isinstance(restriction, Mapping):
            restriction_id = str(restriction.get("id") or restriction_id)

        def check_restriction(knowledge_graph: dict):
            inferred = {}
            similarity_threshold = float(self.violation_thresholds.get("similarity", 0.85))
            patterns = []
            forbidden_content = ""
            if isinstance(restriction, Mapping):
                patterns = self._normalize_pattern_list(restriction.get("patterns"))
                forbidden_content = str(restriction.get("forbidden_content") or "")

            for key, value in (knowledge_graph or {}).items():
                if self._matches_any_pattern(str(key), patterns):
                    similarity = SequenceMatcher(None, str(value), forbidden_content).ratio()
                    if similarity > similarity_threshold:
                        inferred[f"VIOLATION/{restriction_id}"] = similarity
            return inferred

        return check_restriction

    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches a restriction pattern."""
        if not isinstance(text, str) or not isinstance(pattern, str) or not pattern:
            return False
        try:
            return re.search(pattern, text, re.IGNORECASE) is not None
        except re.error:
            return pattern.lower() in text.lower()

    def _matches_any_pattern(self, text: str, patterns: Sequence[str]) -> bool:
        return any(self._matches_pattern(text, pattern) for pattern in patterns)

    def _start_monitoring_thread(self):
        """Start background monitoring of shared memory."""
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            return

        def monitor():
            while True:
                try:
                    self._check_agent_health()
                except Exception as exc:  # pragma: no cover - defensive path
                    logger.error(f"Governor monitoring error: {exc}", exc_info=True)
                time.sleep(self.monitoring_interval_seconds)

        self._monitoring_thread = threading.Thread(
            target=monitor,
            name="governor-monitor",
            daemon=True,
        )
        self._monitoring_thread.start()

    def _extract_agent_memory_store(self) -> Dict[str, Any]:
        if not self.knowledge_agent:
            return {}

        agent_memory = getattr(self.knowledge_agent, "memory", None)
        if agent_memory is None:
            return {}

        store = getattr(agent_memory, "_store", None)
        if isinstance(store, Mapping):
            return dict(store)
        if isinstance(agent_memory, Mapping):
            return dict(agent_memory)
        return {}

    def _evaluate_rule_violations(self, knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
        if not knowledge_graph:
            return {}
        apply_fn = getattr(self.rule_engine, "apply", None)
        if not callable(apply_fn):
            logger.warning("Rule engine unavailable during audit computation")
            return {}
        try:
            result = apply_fn(knowledge_graph)
        except Exception as exc:
            logger.error(f"Error applying rules during audit: {exc}", exc_info=True)
            return {}
        return result if isinstance(result, Mapping) else {}

    def full_audit(self):
        """Comprehensive system audit."""
        approved_rules = self.get_approved_rules()
        current_memory_store = self._extract_agent_memory_store()
        violations = self._evaluate_rule_violations(current_memory_store)

        audit_report = {
            "timestamp": time.time(),
            "behavior_checks": self._audit_agent_behavior(),
            "violations": [],
            "recommendations": [],
            "rules_used": [r["id"] for r in approved_rules if isinstance(r, Mapping) and "id" in r],
        }

        for fact, confidence in violations.items():
            audit_report["violations"].append(
                {
                    "fact": fact,
                    "confidence": confidence,
                    "action": self._determine_enforcement_action(str(fact)),
                }
            )

        self.audit_history.append(audit_report)
        self.last_audit = time.time()
        return audit_report

    def audit_retrieval(self, query: str, results: list, context: dict):
        """Audit knowledge retrieval results against governance guidelines."""
        audit_entry = {
            "timestamp": time.time(),
            "query": query,
            "violations": [],
            "bias_detected": defaultdict(int),
            "context": context or {},
        }

        unethical_threshold = float(self.violation_thresholds.get("unethical", 0.65))
        freshness_thresh_hours = self.freshness_threshold

        for item in results or []:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            _score, doc = item
            if not isinstance(doc, Mapping):
                continue

            text = str(doc.get("text", ""))
            metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata", {}), Mapping) else {}

            unethical_score = self._detect_unethical_content(text)
            if unethical_score > unethical_threshold:
                audit_entry["violations"].append(
                    {
                        "type": "unethical_content",
                        "doc": text[:200],
                        "score": unethical_score,
                        "action": self._determine_enforcement_action("VIOLATION/UNETHICAL"),
                    }
                )

            bias_scores = self._detect_bias(text)
            for category, count in bias_scores.items():
                if count > 0:
                    audit_entry["bias_detected"][category] += count

            timestamp_value = metadata.get("timestamp")
            if isinstance(timestamp_value, (int, float)):
                age_hours = (time.time() - float(timestamp_value)) / 3600
                if age_hours > freshness_thresh_hours:
                    audit_entry["violations"].append(
                        {
                            "type": "stale_knowledge",
                            "age_hours": round(age_hours, 1),
                            "threshold": freshness_thresh_hours,
                        }
                    )
            elif timestamp_value is not None:
                logger.warning(f"Invalid timestamp format in metadata for doc: {text[:50]}...")

        if audit_entry["violations"]:
            knowledge_graph_for_rules = {
                f"doc_{i}": res_doc.get("text", "")
                for i, item in enumerate(results or [])
                if isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[1], Mapping)
                for res_doc in [item[1]]
            }
            knowledge_graph_for_rules["query"] = query
            audit_entry["rule_violations"] = self._evaluate_rule_violations(knowledge_graph_for_rules)

        self.audit_history.append(audit_entry)

        critical_threshold = float(self.violation_thresholds.get("critical", 0.8))
        critical_violations = [
            violation
            for violation in audit_entry["violations"]
            if isinstance(violation, Mapping) and float(violation.get("score", 0) or 0) > critical_threshold
        ]
        if critical_violations:
            logger.warning(f"Critical retrieval violations detected: {len(critical_violations)}")

        return audit_entry

    def _detect_unethical_content(self, text: str) -> float:
        """Score text against ethical principles."""
        principles = self.guidelines.get("principles", [])
        if not principles:
            return 0.0

        matches = 0
        valid_principles = 0
        for principle in principles:
            if not isinstance(principle, Mapping):
                continue
            valid_principles += 1
            principle_type = str(principle.get("type") or "").lower()
            if principle_type not in {"prohibition", "restriction"} and "conflicts" not in set(
                self._normalize_tag_list(principle.get("tags"))
            ):
                continue
            for pattern in self._normalize_pattern_list(principle.get("patterns")):
                try:
                    matches += len(re.findall(pattern, text, re.IGNORECASE))
                except re.error:
                    matches += text.lower().count(pattern.lower())

        if valid_principles == 0:
            return 0.0
        return matches / valid_principles

    def _detect_bias(self, text: str) -> dict:
        """Enhanced bias detection using loaded categories."""
        if not self.bias_categories:
            logger.warning("No bias categories loaded, using simple detection")
            return self._simple_bias_detection(text)

        scores = defaultdict(int)
        text_lower = text.lower()

        for category in self.bias_categories.get("categories", []):
            if not isinstance(category, Mapping):
                continue
            category_id = str(category.get("id") or "")
            if category_id:
                scores[category_id] += self._count_keywords(text_lower, category.get("keywords", []))

            for subcategory in category.get("subcategories", []):
                if not isinstance(subcategory, Mapping):
                    continue
                subcategory_id = str(subcategory.get("id") or "")
                if subcategory_id:
                    scores[subcategory_id] += self._count_keywords(
                        text_lower, subcategory.get("keywords", [])
                    )

                for bias_type in subcategory.get("types", []):
                    if not isinstance(bias_type, Mapping):
                        continue
                    bias_type_id = str(bias_type.get("id") or "")
                    if bias_type_id:
                        scores[bias_type_id] += self._count_keywords(
                            text_lower, bias_type.get("keywords", [])
                        )

        return dict(scores)

    def _count_keywords(self, text: str, keywords: Iterable[Any]) -> int:
        """Count occurrences of keywords in text."""
        count = 0
        for keyword in keywords or []:
            if isinstance(keyword, str) and keyword:
                count += text.count(keyword.lower())
        return count

    def _simple_bias_detection(self, text: str) -> dict:
        """Fallback simple bias detection."""
        text_lower = text.lower()
        return {
            "gender": sum(text_lower.count(term) for term in ["gender", "male", "female"]),
            "race": sum(text_lower.count(term) for term in ["race", "ethnic", "black", "white"]),
            "religion": sum(
                text_lower.count(term) for term in ["religion", "god", "christian", "muslim"]
            ),
        }

    def _safe_shared_get(self, shared_mem: Any, key: str, default: Any = None) -> Any:
        if shared_mem is None:
            return default
        get_fn = getattr(shared_mem, "get", None)
        if callable(get_fn):
            try:
                return get_fn(key, default)
            except TypeError:
                try:
                    value = get_fn(key)
                    return default if value is None else value
                except Exception:
                    return default
        if isinstance(shared_mem, Mapping):
            return shared_mem.get(key, default)
        return default

    def _safe_shared_set(self, shared_mem: Any, key: str, value: Any) -> bool:
        if shared_mem is None:
            return False

        set_fn = getattr(shared_mem, "set", None)
        if callable(set_fn):
            set_fn(key, value)
            return True

        update_fn = getattr(shared_mem, "update", None)
        if callable(update_fn):
            try:
                update_fn(key, value)
                return True
            except TypeError:
                pass

        if hasattr(shared_mem, "__setitem__"):
            shared_mem[key] = value
            return True

        return False

    def _audit_agent_behavior(self) -> dict:
        """Analyze recent behavior using self.knowledge_agent if available."""
        if not self.knowledge_agent or not hasattr(self.knowledge_agent, "name"):
            return {
                "recent_errors": [],
                "retraining_flags": False,
                "performance_metrics": {},
                "error_diversity": 1.0,
                "status": "Knowledge agent not available for behavior audit.",
            }

        shared_mem = getattr(self.knowledge_agent, "shared_memory", None)
        agent_name = getattr(self.knowledge_agent, "name", "unknown_knowledge_agent")
        perf_metrics_raw = getattr(self.knowledge_agent, "performance_metrics", {})
        perf_metrics_dict: Dict[str, List[Any]] = {}
        if isinstance(perf_metrics_raw, Mapping):
            for key, value in perf_metrics_raw.items():
                if isinstance(value, (list, tuple, deque)):
                    perf_metrics_dict[str(key)] = list(value)
                else:
                    perf_metrics_dict[str(key)] = [value]

        recent_errors_raw = self._safe_shared_get(shared_mem, f"errors:{agent_name}", [])
        recent_errors_list = recent_errors_raw[-10:] if isinstance(recent_errors_raw, list) else []
        retraining_flag_val = bool(
            self._safe_shared_get(shared_mem, f"retraining_flag:{agent_name}", False)
        )

        error_types_list = []
        for error in recent_errors_list:
            if isinstance(error, Mapping):
                error_types_list.append(
                    error.get("error_type") or str(error.get("error", "")).split(":")[0]
                )

        error_diversity_val = (
            len(set(error_types_list)) / len(error_types_list) if error_types_list else 1.0
        )

        return {
            "recent_errors": recent_errors_list,
            "retraining_flags": retraining_flag_val,
            "performance_metrics": perf_metrics_dict,
            "error_diversity": error_diversity_val,
            "status": "Behavior audit complete.",
        }

    def _determine_enforcement_action(self, violation: str) -> str:
        """Determine appropriate enforcement action based on config."""
        if "VIOLATION" in violation:
            if self.enforcement_mode == "alert":
                return "Send alert to human supervisor"
            if self.enforcement_mode == "restrict":
                return "Disable related knowledge entries"
        return "Log violation only"

    def _check_agent_health(self):
        """Real-time health checks against optional shared memory implementations."""
        if not self.knowledge_agent or not hasattr(self.knowledge_agent, "name"):
            logger.warning("Knowledge agent instance not available for Governor health checks.")
            return {
                "status": "agent_unavailable",
                "retraining_flag_set": False,
                "high_memory_usage": False,
            }

        agent_name = getattr(self.knowledge_agent, "name", "unknown_knowledge_agent")
        shared_mem_obj = getattr(self.knowledge_agent, "shared_memory", None)
        if shared_mem_obj is None:
            logger.warning(f"Shared memory is not available for {agent_name}")
            return {
                "status": "shared_memory_unavailable",
                "retraining_flag_set": False,
                "high_memory_usage": False,
            }

        consecutive_errors_threshold = self._safe_int(
            self.violation_thresholds.get("consecutive_errors"), 5, minimum=1
        )
        errors = self._safe_shared_get(shared_mem_obj, f"errors:{agent_name}", [])
        error_count = len(errors) if isinstance(errors, list) else 0

        retraining_flag_set = False
        if error_count >= consecutive_errors_threshold:
            logger.warning(
                f"Consecutive error threshold breached for {agent_name}: {error_count} errors"
            )
            retraining_flag_set = self._safe_shared_set(
                shared_mem_obj, f"retraining_flag:{agent_name}", True
            )
            if not retraining_flag_set:
                logger.warning(
                    f"Unable to persist retraining flag for {agent_name}; shared memory lacks a supported mutator"
                )

        warning_memory_threshold = float(self.memory_thresholds.get("warning", 2048))
        mem_usage = self._safe_shared_get(shared_mem_obj, f"memory_usage:{agent_name}")
        high_memory_usage = bool(
            isinstance(mem_usage, (int, float)) and mem_usage > warning_memory_threshold
        )
        if high_memory_usage:
            logger.info(f"High memory usage for {agent_name}: {mem_usage} MB")

        return {
            "status": "ok",
            "retraining_flag_set": retraining_flag_set,
            "high_memory_usage": high_memory_usage,
        }

    def record_violations(self, violations: List[Dict]):
        """Records violations, possibly from other components."""
        for violation_item in violations:
            if isinstance(violation_item, dict):
                self.audit_history.append(violation_item)
                logger.warning(
                    "External violation recorded by Governor: %s - %s",
                    violation_item.get("type", "Unknown Type"),
                    violation_item.get("entry", ""),
                )
            else:
                logger.warning(f"Attempted to record non-dict violation: {violation_item}")

    def handle_emergency_alert(self, alert_report: Dict):
        """Handles an emergency alert."""
        logger.critical(
            f"EMERGENCY ALERT RECEIVED BY GOVERNOR: {alert_report.get('trigger')}"
        )
        self.audit_history.append(
            {
                "timestamp": time.time(),
                "type": "emergency_alert_received",
                "details": alert_report,
            }
        )

    def generate_report(self, format_type: str = "json") -> Union[dict, str]:
        """Generate formatted audit report."""
        latest_audit_details = {"timestamp": self.last_audit, "violations": []}
        active_violations_count = 0

        if self.audit_history:
            last_entry = self.audit_history[-1]
            if isinstance(last_entry, Mapping):
                violations_list = last_entry.get("violations", [])
                latest_audit_details["timestamp"] = last_entry.get("timestamp", self.last_audit)
                latest_audit_details["violations"] = violations_list if isinstance(violations_list, list) else []
                active_violations_count = len(
                    [
                        violation
                        for violation in latest_audit_details["violations"]
                        if isinstance(violation, Mapping)
                        and float(violation.get("confidence", 0) or 0) > 0.7
                    ]
                )

        report = {
            "summary": {
                "total_audits": len(self.audit_history),
                "last_audit_timestamp": self.last_audit,
                "active_violations_in_last_audit": active_violations_count,
            },
            "details_of_last_audit": latest_audit_details,
        }

        if format_type == "yaml":
            return yaml.dump(report)
        return report


if __name__ == "__main__":
    print("\n=== Running Governor ===\n")
    printer.status("Init", "Governor initialized", "success")
    format_type = "json"
    text = "I love the way you talk! SLAI is the future of decentralized Agentic AI"
    auditor = Governor()

    printer.status("Auditor", auditor, "success")
    printer.status("Detector", auditor._get_bias_detector(), "success")
    printer.status("Approval", auditor.get_approved_rules(), "success")
    printer.status("checker", auditor._check_agent_health(), "success")
    printer.status("monitoring", auditor._start_monitoring_thread(), "success")
    printer.status("Content Check", auditor._detect_unethical_content(text=text), "success")
    printer.pretty("Bias", auditor._detect_bias(text=text), "success")
    printer.pretty("Report", auditor.generate_report(format_type=format_type), "success")
    print("\n=== Governor Test Completed ===")
