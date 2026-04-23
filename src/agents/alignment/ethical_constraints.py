"""
Formal Ethical Governance System
Implements:
- STPA-based hazard analysis (Leveson, 2011)
- Constitutional AI principles (Bai et al., 2022)
- Dynamic rule adaptation (Kasirzadeh & Gabriel, 2023)
"""

from __future__ import annotations

import json
import hashlib

import numpy as np
import networkx as nx

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .utils import *
from .alignment_memory import AlignmentMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Ethical Constraints")
printer = PrettyPrinter

@dataclass
class ConstraintRecord:
    """Canonical in-memory representation of a constraint or rule."""

    id: str
    category: str
    scope: str
    weight: float
    priority: float
    status: str = "active"
    severity: str = "medium"
    source: str = "config"
    rule_statement: str = ""
    action: Any = None
    condition: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "scope": self.scope,
            "weight": self.weight,
            "priority": self.priority,
            "status": self.status,
            "severity": self.severity,
            "source": self.source,
            "rule_statement": self.rule_statement,
            "action": json_safe(self.action),
            "condition": json_safe(self.condition),
            "metadata": json_safe(self.metadata),
        }


class EthicalConstraints:
    """
    Multi-layered ethical governance system implementing:
    - Hazard-aware constraint checking
    - Constitutional rule enforcement
    - Dynamic constraint adaptation
    - Ethical conflict resolution

    Architecture:
    1. Safety Layer: STPA-derived hazard prevention
    2. Constitutional Layer: Principle-based filtering
    3. Societal Layer: Fairness/equity preservation
    4. Adaptation Layer: Experience-driven rule updates

    The class is designed to act as the alignment subsystem's ethical control
    plane. It integrates with AlignmentMemory, emits audit-friendly telemetry,
    manages both static and dynamic constraints, and exposes helper methods
    consumed by the alignment agent and related submodules.
    """

    def __init__(
        self,
        config_section_name: str = "ethical_constraints",
        config_file_path: Optional[str] = None,
    ):
        self.config = load_global_config()
        self.config_section_name = ensure_non_empty_string(
            config_section_name,
            "config_section_name",
            error_cls=ConfigurationError,
        )
        self.config_file_path = config_file_path
        self.ethics_config = get_config_section(self.config_section_name)
        self._validate_ethics_config()

        self.sensitive_attributes = normalize_sensitive_attributes(
            self.config.get("sensitive_attributes", []),
            lowercase=False,
            allow_empty=True,
        )
        self.alignment_memory = AlignmentMemory()

        self.safety_constraints = dict(self.ethics_config.get("safety_constraints", {}))
        self.fairness_constraints = dict(self.ethics_config.get("fairness_constraints", {}))
        self.constitutional_rules = dict(self.ethics_config.get("constitutional_rules", {}))
        self.adaptation_rate = coerce_float(
            self.ethics_config.get("adaptation_rate", 0.1),
            field_name="adaptation_rate",
            minimum=0.0,
            maximum=5.0,
        )
        self.constraint_priorities = [
            str(item).strip()
            for item in self.ethics_config.get("constraint_priorities", [])
            if str(item).strip()
        ]
        self.weight_floor = coerce_float(
            self.ethics_config.get("weight_floor", 0.10),
            field_name="weight_floor",
            minimum=0.0,
            maximum=10.0,
        )
        self.weight_ceiling = coerce_float(
            self.ethics_config.get("weight_ceiling", 5.00),
            field_name="weight_ceiling",
            minimum=self.weight_floor,
            maximum=20.0,
        )
        self.audit_history_limit = coerce_int(
            self.ethics_config.get("audit_history_limit", 10000),
            field_name="audit_history_limit",
            minimum=1,
        )
        self.rule_load_fail_open = coerce_bool(
            self.ethics_config.get("rule_load_fail_open", True),
            field_name="rule_load_fail_open",
        )
        self.unknown_rule_policy = ensure_non_empty_string(
            self.ethics_config.get("unknown_rule_policy", "review"),
            "unknown_rule_policy",
            error_cls=ConfigurationError,
        ).lower()
        self.memory_logging_enabled = coerce_bool(
            self.ethics_config.get("memory_logging_enabled", True),
            field_name="memory_logging_enabled",
        )
        self.auto_adapt_on_violation = coerce_bool(
            self.ethics_config.get("auto_adapt_on_violation", True),
            field_name="auto_adapt_on_violation",
        )
        self.default_scope = ensure_non_empty_string(
            self.ethics_config.get("default_scope", "global"),
            "default_scope",
            error_cls=ConfigurationError,
        )
        self.default_severity = normalize_severity(
            self.ethics_config.get("default_severity", "medium"),
            field_name="default_severity",
        )

        self.category_weights = normalize_weight_mapping(
            self.ethics_config.get("category_weights", {}),
            drop_none=True,
            normalize_sum=False,
            allow_negative=False,
        )
        self.societal_impact_thresholds = normalize_threshold_mapping(
            self.ethics_config.get("societal_impact_thresholds", {}),
            minimum=0.0,
            maximum=1.0,
        )
        self.hazard_bounds = normalize_context(self.ethics_config.get("hazard_bounds", {}))
        self.severity_bands = normalize_context(self.ethics_config.get("severity_bands", {}))

        self.audit_log: List[Dict[str, Any]] = []
        self.rule_bundles = self._load_all_rule_bundles()
        self.constitutional_rules_loaded = self.rule_bundles["constitutional_rules"]
        self.rule_evaluators = self._create_rule_evaluator_mapping()
        self.constraint_weights = self._init_weights()
        self.constraints: List[Dict[str, Any]] = self._build_constraint_registry()
        self.constraint_graph = nx.DiGraph()
        self._build_constraint_graph()

        if self.config_file_path:
            logger.debug(
                "EthicalConstraints received config_file_path=%s but retained global config loader handling.",
                self.config_file_path,
            )

        logger.info(
            "EthicalConstraints initialized | safety=%s fairness=%s constitutional=%s dynamic=%s",
            len(self.safety_constraints),
            len(self.fairness_constraints),
            sum(len(v) for v in self.constitutional_rules_loaded.values()),
            len([c for c in self.constraints if c.get("source") == "runtime"]),
        )

    # ------------------------------------------------------------------
    # Configuration and initialization
    # ------------------------------------------------------------------
    def _validate_ethics_config(self) -> None:
        try:
            ensure_mapping(
                self.ethics_config,
                self.config_section_name,
                allow_empty=False,
                error_cls=ConfigurationError,
            )
            ensure_keys_present(
                self.ethics_config,
                [
                    "safe_energy_threshold",
                    "min_energy_threshold",
                    "max_energy_threshold",
                    "safety_constraints",
                    "fairness_constraints",
                    "constitutional_rules",
                    "adaptation_rate",
                    "constraint_priorities",
                ],
                field_name=self.config_section_name,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.ethics_config["safe_energy_threshold"],
                "safe_energy_threshold",
                min_value=0.0,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.ethics_config["min_energy_threshold"],
                "min_energy_threshold",
                min_value=0.0,
                error_cls=ConfigurationError,
            )
            ensure_numeric_range(
                self.ethics_config["max_energy_threshold"],
                "max_energy_threshold",
                min_value=0.0,
                error_cls=ConfigurationError,
            )
            if float(self.ethics_config["min_energy_threshold"]) > float(self.ethics_config["safe_energy_threshold"]):
                raise ConfigurationError(
                    "'min_energy_threshold' must not exceed 'safe_energy_threshold'.",
                    context={"ethical_constraints": json_safe(self.ethics_config)},
                )
            if float(self.ethics_config["safe_energy_threshold"]) > float(self.ethics_config["max_energy_threshold"]):
                raise ConfigurationError(
                    "'safe_energy_threshold' must not exceed 'max_energy_threshold'.",
                    context={"ethical_constraints": json_safe(self.ethics_config)},
                )
            ensure_mapping(
                self.ethics_config["safety_constraints"],
                "safety_constraints",
                allow_empty=False,
                error_cls=ConfigurationError,
            )
            ensure_mapping(
                self.ethics_config["fairness_constraints"],
                "fairness_constraints",
                allow_empty=False,
                error_cls=ConfigurationError,
            )
            ensure_mapping(
                self.ethics_config["constitutional_rules"],
                "constitutional_rules",
                allow_empty=False,
                error_cls=ConfigurationError,
            )
            ensure_sequence(
                self.ethics_config["constraint_priorities"],
                "constraint_priorities",
                allow_empty=False,
                allow_strings=False,
                error_cls=ConfigurationError,
            )
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=ConfigurationError,
                message="EthicalConstraints configuration validation failed.",
                context={
                    "config_section": self.config_section_name,
                    "config_path": self.config.get("__config_path__"),
                },
            ) from exc

    def _load_all_rule_bundles(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        bundles = {
            "safety_constraints": {},
            "fairness_constraints": {},
            "constitutional_rules": {},
        }
        for bundle_name in bundles.keys():
            bundle_config = dict(self.ethics_config.get(bundle_name, {}))
            for category, reference in bundle_config.items():
                bundles[bundle_name][str(category)] = self._load_rules(
                    category=str(category),
                    file_path=str(reference),
                    bundle_name=bundle_name,
                )
        return bundles

    def _load_rules(self, category: str, file_path: str, bundle_name: str = "constitutional_rules") -> List[Dict[str, Any]]:
        """Load a rule bundle from a JSON file with deterministic normalization."""
        path = Path(file_path)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError as exc:
            message = f"Rule file not found for {bundle_name}:{category}: {file_path}"
            if self.rule_load_fail_open:
                logger.warning(message)
                return []
            raise ConfigurationError(message, context={"bundle": bundle_name, "category": category, "path": file_path}, cause=exc) from exc
        except Exception as exc:
            message = f"Failed to load rule file for {bundle_name}:{category}: {file_path}"
            if self.rule_load_fail_open:
                logger.warning("%s | %s", message, exc)
                return []
            raise wrap_alignment_exception(
                exc,
                target_cls=EthicalConstraintError,
                message=message,
                context={"bundle": bundle_name, "category": category, "path": file_path},
            ) from exc

        extracted = self._extract_rule_records(data, default_category=category, source_path=str(path))
        return extracted

    def _extract_rule_records(
        self,
        payload: Any,
        *,
        default_category: str,
        source_path: str,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []

        def ingest_list(items: Sequence[Any], module_name: Optional[str] = None) -> None:
            for idx, item in enumerate(items):
                if not isinstance(item, Mapping):
                    continue
                rule_id = str(item.get("id") or item.get("rule_id") or f"{default_category}_{idx+1:03d}")
                statement = str(
                    item.get("rule_statement")
                    or item.get("statement")
                    or item.get("text")
                    or item.get("description")
                    or ""
                ).strip()
                records.append(
                    {
                        "id": rule_id,
                        "rule_statement": statement,
                        "category": str(item.get("category") or default_category),
                        "module": module_name or str(item.get("module") or default_category),
                        "severity": normalize_severity(item.get("severity") or self.default_severity),
                        "metadata": {
                            "source_path": source_path,
                            "raw": json_safe(item),
                        },
                    }
                )

        if isinstance(payload, list):
            ingest_list(payload, module_name=default_category)
        elif isinstance(payload, Mapping):
            if "rules" in payload and isinstance(payload.get("rules"), list):
                ingest_list(payload["rules"], module_name=str(payload.get("module") or default_category))
            else:
                for key, value in payload.items():
                    if isinstance(value, Mapping) and isinstance(value.get("rules"), list):
                        ingest_list(value["rules"], module_name=str(key))
                    elif isinstance(value, list):
                        ingest_list(value, module_name=str(key))
        return records

    def _init_weights(self) -> Dict[str, float]:
        categories = set(self.constraint_priorities)
        categories.update(self.safety_constraints.keys())
        categories.update(self.fairness_constraints.keys())
        categories.update(self.constitutional_rules.keys())

        if not categories:
            raise ConfigurationError("No constraint categories available to initialize weights.")

        explicit_weights = dict(self.category_weights)
        weights: Dict[str, float] = {}
        if explicit_weights:
            for key, value in explicit_weights.items():
                numeric = coerce_float(value, field_name=f"category_weights:{key}", minimum=self.weight_floor, maximum=self.weight_ceiling)
                weights[str(key)] = numeric

        if not weights:
            base_weight = 1.0
            step = 0.05
            priority_map = {str(item): idx for idx, item in enumerate(self.constraint_priorities)}
            for category in sorted(categories, key=lambda name: priority_map.get(name, 10_000)):
                computed = base_weight - (priority_map.get(category, len(categories)) * step)
                weights[category] = max(self.weight_floor, min(self.weight_ceiling, round(computed, 4)))
        else:
            for category in categories:
                if category not in weights:
                    weights[category] = max(self.weight_floor, min(self.weight_ceiling, 1.0))

        return weights

    def _build_constraint_registry(self) -> List[Dict[str, Any]]:
        registry: List[Dict[str, Any]] = []

        for category in self.safety_constraints.keys():
            registry.append(
                ConstraintRecord(
                    id=f"SAFETY::{category}",
                    category=str(category),
                    scope=self.default_scope,
                    weight=self._resolve_constraint_weight(category, fallback_category="safety"),
                    priority=self._priority_score("safety"),
                    severity="high",
                    source="config",
                    rule_statement=f"Safety constraint for {category}.",
                    metadata={"bundle": "safety_constraints"},
                ).to_dict()
            )

        for category in self.fairness_constraints.keys():
            registry.append(
                ConstraintRecord(
                    id=f"FAIRNESS::{category}",
                    category=str(category),
                    scope=self.default_scope,
                    weight=self._resolve_constraint_weight(category, fallback_category="fairness"),
                    priority=self._priority_score("fairness"),
                    severity="medium",
                    source="config",
                    rule_statement=f"Societal impact constraint for {category}.",
                    metadata={"bundle": "fairness_constraints"},
                ).to_dict()
            )

        for category, rules in self.constitutional_rules_loaded.items():
            if not rules:
                registry.append(
                    ConstraintRecord(
                        id=f"CONSTITUTION::{category}",
                        category=str(category),
                        scope=self.default_scope,
                        weight=self._resolve_constraint_weight(category, fallback_category=category),
                        priority=self._priority_score(category),
                        severity="medium",
                        source="config",
                        rule_statement=f"Constitutional principle category: {category}.",
                        metadata={"bundle": "constitutional_rules", "empty_bundle": True},
                    ).to_dict()
                )
                continue
            for rule in rules:
                registry.append(
                    ConstraintRecord(
                        id=str(rule["id"]),
                        category=str(rule.get("category") or category),
                        scope=self.default_scope,
                        weight=self._resolve_constraint_weight(rule.get("category") or category, fallback_category=category),
                        priority=self._priority_score(category),
                        severity=normalize_severity(rule.get("severity") or self.default_severity),
                        source="config",
                        rule_statement=str(rule.get("rule_statement", "")),
                        metadata={"bundle": "constitutional_rules", **dict(rule.get("metadata") or {})},
                    ).to_dict()
                )

        return registry

    def _build_constraint_graph(self) -> None:
        self.constraint_graph = nx.DiGraph()
        categories = set(self.constraint_weights.keys())
        self.constraint_graph.add_nodes_from(categories)

        for idx, category in enumerate(self.constraint_priorities):
            for lower_priority in self.constraint_priorities[idx + 1 :]:
                if category in self.constraint_graph and lower_priority in self.constraint_graph:
                    self.constraint_graph.add_edge(category, lower_priority, relation="priority")

        configured_pairs = self.ethics_config.get(
            "conflict_pairs",
            [["transparency", "privacy"], ["safety", "distribution"]],
        )
        for pair in configured_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            left, right = str(pair[0]), str(pair[1])
            if left in self.constraint_graph and right in self.constraint_graph:
                self.constraint_graph.add_edge(left, right, relation="conflict")
                self.constraint_graph.add_edge(right, left, relation="conflict")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enforce(self, action_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive ethical validation pipeline:
        1. Safety hazard analysis
        2. Constitutional compliance check
        3. Societal impact assessment
        4. Dynamic/runtime constraint enforcement
        5. Adaptive constraint adjustment and memory logging
        """
        try:
            context = normalize_context(action_context, drop_none=False)
            timestamp = datetime.utcnow().isoformat() + "Z"
            trace_id = hashlib.sha256(stable_json_dumps(context).encode("utf-8")).hexdigest()[:24]

            safety_report = self._check_safety_constraints(context)
            constitutional_report = self._check_constitutional_rules(context)
            societal_report = self._check_societal_impact(context)
            dynamic_report = self._check_dynamic_constraints(context)

            subreports = {
                "safety": safety_report,
                "constitutional": constitutional_report,
                "societal": societal_report,
                "dynamic": dynamic_report,
            }

            violations: List[str] = []
            corrective_actions: List[Dict[str, Any]] = []
            explanations: List[str] = []
            per_layer_scores: Dict[str, float] = {}

            for layer_name, report in subreports.items():
                violations.extend(report.get("violations", []))
                corrective_actions.extend(report.get("corrections", []))
                explanations.extend(report.get("explanations", []))
                per_layer_scores[layer_name] = float(report.get("risk_score", 0.0))

            total_risk = float(sum(per_layer_scores.values()))
            approved = len(violations) == 0
            severity = self._severity_from_risk(total_risk, len(violations))
            risk_level = self._risk_level_from_risk(total_risk, len(violations))
            summary = {
                "violation_count": len(violations),
                "total_risk": total_risk,
                "constraint_count": len(self.constraints),
                "active_constraint_count": len([c for c in self.constraints if c.get("status") == "active"]),
            }

            event = build_alignment_event(
                "ethical_constraint_evaluation",
                severity=severity,
                risk_level=risk_level,
                source="ethical_constraints",
                trace_id=trace_id,
                tags=["ethics", "alignment", "governance"],
                context=context,
                metadata={
                    "summary": summary,
                    "weights": self.constraint_weights,
                },
                payload={
                    "subreports": subreports,
                    "violations": violations,
                    "corrective_actions": corrective_actions,
                },
            )

            result = {
                "approved": approved,
                "timestamp": timestamp,
                "trace_id": trace_id,
                "severity": severity,
                "risk_level": risk_level,
                "violations": violations,
                "corrective_actions": corrective_actions,
                "explanations": explanations,
                "summary": summary,
                "subreports": subreports,
                "constraint_weights": dict(self.constraint_weights),
                "event": event,
            }

            if not approved:
                self._log_violation(context, result)
                if self.auto_adapt_on_violation:
                    self._adapt_constraints(context, result)
            else:
                self._log_success(context, result)

            if self.memory_logging_enabled:
                self._log_to_memory(context, result)

            return result
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=EthicalConstraintError,
                message="Ethical constraint enforcement failed.",
                context={"operation": "enforce"},
            ) from exc

    def enforce_core_constraints(
        self,
        memory_snapshot: Mapping[str, Any],
        constraint_level: str = "emergency",
    ) -> Dict[str, Any]:
        """
        Enforce a reduced emergency-ready subset of constraints against a memory snapshot.

        This method is used during safe-state operation where the full action
        context may not be available, but the system still needs a conservative,
        auditable ethical posture.
        """
        try:
            snapshot = normalize_context(memory_snapshot, drop_none=False)
            level = ensure_non_empty_string(constraint_level, "constraint_level", error_cls=ValidationError).lower()

            derived_context = {
                "decision_engine": {"is_active": snapshot.get("system_mode") not in {"OFFLINE", "CRASHED"}},
                "affected_environment": snapshot.get("affected_environment", {}),
                "action_parameters": snapshot.get("action_parameters", {}),
                "output_mechanisms": snapshot.get("output_mechanisms", {}),
                "feedback_systems": snapshot.get("feedback_systems", {}),
                "potential_energy": snapshot.get("potential_energy", 0.0),
                "kinetic_energy": snapshot.get("kinetic_energy", 0.0),
                "informational_entropy": snapshot.get("informational_entropy", 0.0),
                "audit_trail_status": snapshot.get("audit_trail_status", "active"),
                "log_completeness_score": snapshot.get("log_completeness_score", 1.0),
                "data_encrypted": snapshot.get("data_encrypted", True),
                "encryption_at_rest": snapshot.get("encryption_at_rest", True),
                "encryption_in_transit": snapshot.get("encryption_in_transit", True),
            }

            result = self.enforce(derived_context)
            result["constraint_level"] = level
            result["memory_snapshot_hash"] = stable_context_hash(snapshot, namespace="memory_snapshot")
            return result
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=EthicalConstraintError,
                message="Failed to enforce core ethical constraints.",
                context={"constraint_level": constraint_level},
            ) from exc

    def get_constraints(self, *, active_only: bool = False) -> List[Dict[str, Any]]:
        constraints = [dict(item) for item in self.constraints]
        if active_only:
            constraints = [item for item in constraints if item.get("status") == "active"]
        return constraints

    def get_current_state(self) -> Dict[str, Any]:
        return {
            "constraint_count": len(self.constraints),
            "active_constraint_count": len([c for c in self.constraints if c.get("status") == "active"]),
            "constraint_weights": dict(self.constraint_weights),
            "audit_log_size": len(self.audit_log),
            "graph_nodes": self.constraint_graph.number_of_nodes(),
            "graph_edges": self.constraint_graph.number_of_edges(),
            "constitutional_rule_count": sum(len(v) for v in self.constitutional_rules_loaded.values()),
        }

    def add_constraint(
        self,
        constraint_id: str,
        condition: Any,
        action: Any,
        weight: float,
        priority: float,
        scope: str = "global",
        severity: str = "medium",
        category: str = "dynamic",
        source: str = "runtime",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            record = ConstraintRecord(
                id=ensure_non_empty_string(constraint_id, "constraint_id", error_cls=ValidationError),
                category=ensure_non_empty_string(category, "category", error_cls=ValidationError),
                scope=ensure_non_empty_string(scope, "scope", error_cls=ValidationError),
                weight=coerce_float(weight, field_name="weight", minimum=self.weight_floor, maximum=self.weight_ceiling),
                priority=coerce_float(priority, field_name="priority", minimum=0.0, maximum=100.0),
                severity=normalize_severity(severity),
                source=ensure_non_empty_string(source, "source", error_cls=ValidationError),
                condition=condition,
                action=action,
                metadata=normalize_context(metadata or {}, drop_none=True),
            ).to_dict()

            existing_index = self._constraint_index(record["id"])
            if existing_index is not None:
                self.constraints[existing_index] = record
            else:
                self.constraints.append(record)

            self.constraint_weights.setdefault(record["category"], float(record["weight"]))
            if record["category"] not in self.constraint_graph:
                self.constraint_graph.add_node(record["category"])
            return record
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=EthicalConstraintError,
                message="Failed to add dynamic ethical constraint.",
                context={"constraint_id": constraint_id},
            ) from exc

    def deactivate_constraint(self, constraint_id: str) -> bool:
        try:
            normalized_id = ensure_non_empty_string(constraint_id, "constraint_id", error_cls=ValidationError)
            index = self._constraint_index(normalized_id)
            if index is None:
                return False
            self.constraints[index]["status"] = "inactive"
            return True
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=EthicalConstraintError,
                message="Failed to deactivate ethical constraint.",
                context={"constraint_id": constraint_id},
            ) from exc

    def synchronize_weights(
        self,
        base_weights: Optional[Mapping[str, Any]] = None,
        new_constraints: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, float]:
        try:
            normalized_base = normalize_weight_mapping(base_weights or {}, normalize_sum=False)
            for key, value in normalized_base.items():
                self.constraint_weights[str(key)] = max(self.weight_floor, min(self.weight_ceiling, float(value)))

            if new_constraints:
                for item in new_constraints:
                    if not isinstance(item, Mapping):
                        continue
                    category = str(item.get("category") or item.get("id") or "dynamic")
                    weight = float(item.get("weight", self.constraint_weights.get(category, 1.0)))
                    self.constraint_weights[category] = max(self.weight_floor, min(self.weight_ceiling, weight))

            self.verify_constraint_weights(auto_fix=True)
            return dict(self.constraint_weights)
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=EthicalConstraintError,
                message="Failed to synchronize ethical constraint weights.",
                context={"operation": "synchronize_weights"},
            ) from exc

    def verify_constraint_weights(self, auto_fix: bool = True) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "status": "healthy",
            "errors": [],
            "warnings": [],
            "fixes": [],
            "details": {},
        }

        try:
            for key in list(self.constraint_weights.keys()):
                value = self.constraint_weights.get(key)
                if value is None:
                    report["errors"].append(f"Missing weight: {key}")
                    if auto_fix:
                        self.constraint_weights[key] = 1.0
                        report["fixes"].append(f"Assigned default weight to {key}")
                    continue

                if not isinstance(value, (int, float)):
                    report["errors"].append(f"Non-numeric weight: {key}")
                    if auto_fix:
                        self.constraint_weights[key] = 1.0
                        report["fixes"].append(f"Replaced non-numeric weight with default for {key}")
                    continue

                if value < self.weight_floor:
                    report["warnings"].append(f"Weight below floor: {key}")
                    if auto_fix:
                        self.constraint_weights[key] = round(self.weight_floor, 4)
                        report["fixes"].append(f"Raised weight floor for {key}")
                elif value > self.weight_ceiling:
                    report["warnings"].append(f"Weight above ceiling: {key}")
                    if auto_fix:
                        self.constraint_weights[key] = round(self.weight_ceiling, 4)
                        report["fixes"].append(f"Clamped weight ceiling for {key}")

            values = list(self.constraint_weights.values())
            if values:
                std_dev = float(np.std(values)) if len(values) > 1 else 0.0
                if std_dev > 0.75:
                    report["warnings"].append(f"High standard deviation in weights: {std_dev:.4f}")

            if report["errors"]:
                report["status"] = "unhealthy"
            elif report["warnings"] or report["fixes"]:
                report["status"] = "repaired" if auto_fix else "degraded"

            report["details"] = dict(self.constraint_weights)
            return report
        except Exception as exc:
            raise wrap_alignment_exception(
                exc,
                target_cls=EthicalConstraintError,
                message="Failed to verify ethical constraint weights.",
                context={"auto_fix": auto_fix},
            ) from exc

    # ------------------------------------------------------------------
    # Safety, constitutional, and societal checks
    # ------------------------------------------------------------------
    def _check_safety_constraints(self, context: Dict[str, Any]) -> Dict[str, Any]:
        hazards: List[str] = []
        corrections: List[Dict[str, Any]] = []
        explanations: List[str] = []
        per_hazard: Dict[str, Any] = {}

        for hazard_type in self.safety_constraints.keys():
            hazard_detected, details = self._detect_hazard(context, str(hazard_type))
            per_hazard[str(hazard_type)] = details
            if hazard_detected:
                hazards.append(f"{hazard_type}_violation")
                corrections.append(self._generate_safety_correction(context, str(hazard_type)))
                explanations.append(f"Hazard prevented: {hazard_type}")

        return {
            "approved": len(hazards) == 0,
            "violations": hazards,
            "corrections": corrections,
            "explanations": explanations,
            "risk_score": float(len(hazards)),
            "details": per_hazard,
        }

    def _check_constitutional_rules(self, context: Dict[str, Any]) -> Dict[str, Any]:
        violations: List[str] = []
        corrections: List[Dict[str, Any]] = []
        explanations: List[str] = []
        evaluated_rules: List[Dict[str, Any]] = []

        for principle, rules in self.constitutional_rules_loaded.items():
            for rule in rules:
                rule_id = str(rule.get("id", ""))
                rule_statement = str(rule.get("rule_statement", "")).strip()
                evaluation = self._evaluate_constitutional_rule(context, rule_statement, rule_id=rule_id)
                evaluated_rules.append(
                    {
                        "principle": principle,
                        "rule_id": rule_id,
                        "rule_statement": rule_statement,
                        "passed": bool(evaluation),
                    }
                )
                if not evaluation:
                    violations.append(f"constitutional_violation:{principle}:{rule_id}")
                    corrections.append(self._constitutional_correction(principle, rule_statement, rule_id=rule_id))
                    explanations.append(f"Violated {principle} ({rule_id}): {rule_statement}")
                elif self.unknown_rule_policy == "review" and not rule_statement:
                    violations.append(f"constitutional_unknown:{principle}:{rule_id}")
                    explanations.append(f"Unknown or empty rule statement for {principle}:{rule_id}")

        return {
            "approved": len(violations) == 0,
            "violations": violations,
            "corrections": corrections,
            "explanations": explanations,
            "risk_score": float(len(violations) * 0.75),
            "details": {"evaluated_rules": evaluated_rules},
        }

    def _check_societal_impact(self, context: Dict[str, Any]) -> Dict[str, Any]:
        impacts: List[str] = []
        corrections: List[Dict[str, Any]] = []
        explanations: List[str] = []
        scores: Dict[str, float] = {}

        for dimension in self.fairness_constraints.keys():
            impact_score = self._calculate_societal_impact(context, str(dimension))
            threshold = self.societal_impact_thresholds.get(
                str(dimension),
                self._resolve_constraint_weight(str(dimension), fallback_category="fairness"),
            )
            scores[str(dimension)] = impact_score
            if impact_score > threshold:
                impacts.append(f"societal_impact:{dimension}")
                corrections.append(self._mitigation_strategy(str(dimension), impact_score=impact_score, threshold=threshold))
                explanations.append(f"Excessive {dimension} impact: {impact_score:.4f} > {threshold:.4f}")

        return {
            "approved": len(impacts) == 0,
            "violations": impacts,
            "corrections": corrections,
            "explanations": explanations,
            "risk_score": float(sum(max(0.0, score - self.societal_impact_thresholds.get(name, 0.0)) for name, score in scores.items())),
            "details": scores,
        }

    def _check_dynamic_constraints(self, context: Dict[str, Any]) -> Dict[str, Any]:
        violations: List[str] = []
        corrections: List[Dict[str, Any]] = []
        explanations: List[str] = []
        details: List[Dict[str, Any]] = []

        active_dynamic = [
            constraint
            for constraint in self.constraints
            if constraint.get("source") == "runtime" and constraint.get("status") == "active"
        ]

        for constraint in active_dynamic:
            passed = self._evaluate_dynamic_constraint(context, constraint)
            details.append({"constraint_id": constraint["id"], "passed": passed})
            if not passed:
                violations.append(f"dynamic_constraint_violation:{constraint['id']}")
                corrections.append(
                    {
                        "action": "dynamic_constraint_response",
                        "parameters": {
                            "constraint_id": constraint["id"],
                            "action": json_safe(constraint.get("action")),
                        },
                    }
                )
                explanations.append(f"Dynamic constraint violated: {constraint['id']}")

        return {
            "approved": len(violations) == 0,
            "violations": violations,
            "corrections": corrections,
            "explanations": explanations,
            "risk_score": float(len(violations) * 0.5),
            "details": details,
        }

    def _detect_hazard(self, context: Dict[str, Any], hazard_type: str) -> Tuple[bool, Dict[str, Any]]:
        action_params = dict(context.get("action_parameters") or {})
        environment_state = str((context.get("affected_environment") or {}).get("state", "unknown"))
        decision_engine = dict(context.get("decision_engine") or {})
        controller_active = bool(decision_engine.get("is_active", False))
        system_energy = float(context.get("potential_energy", 0.0) or 0.0) + float(context.get("kinetic_energy", 0.0) or 0.0)
        system_energy += float(context.get("informational_entropy", 0.0) or 0.0) * 10.0

        threshold = self._calculate_safe_energy_threshold(hazard_type=hazard_type)
        hazard_config = dict(self.hazard_bounds.get(hazard_type, {}))

        blocked_states = {
            str(item)
            for item in hazard_config.get(
                "blocked_environment_states",
                ["overloaded", "resonance", "critical", "unsafe"],
            )
        }
        max_action_kinetic_energy = float(hazard_config.get("max_action_kinetic_energy", 50.0 if hazard_type == "physical_harm" else 100.0))
        max_stress_index = float(hazard_config.get("max_stress_index", 0.70 if hazard_type == "psychological_harm" else 1.0))

        hazard_conditions: List[bool] = [
            not controller_active,
            environment_state in blocked_states,
            system_energy > threshold,
        ]
        details: Dict[str, Any] = {
            "controller_active": controller_active,
            "environment_state": environment_state,
            "blocked_environment": environment_state in blocked_states,
            "system_energy": system_energy,
            "safe_energy_threshold": threshold,
        }

        if hazard_type == "physical_harm":
            action_kinetic = float(action_params.get("kinetic_energy", context.get("kinetic_energy", 0.0)) or 0.0)
            hazard_conditions.append(action_kinetic > max_action_kinetic_energy)
            details.update(
                {
                    "action_kinetic_energy": action_kinetic,
                    "max_action_kinetic_energy": max_action_kinetic_energy,
                }
            )
        elif hazard_type == "psychological_harm":
            stress_index = float(action_params.get("stress_index", 0.0) or 0.0)
            hazard_conditions.append(stress_index > max_stress_index)
            details.update(
                {
                    "stress_index": stress_index,
                    "max_stress_index": max_stress_index,
                }
            )

        return any(hazard_conditions), details

    def _calculate_safe_energy_threshold(self, hazard_type: Optional[str] = None) -> float:
        base_threshold = coerce_float(
            self.ethics_config.get("safe_energy_threshold", 100.0),
            field_name="safe_energy_threshold",
            minimum=0.0,
        )
        min_threshold = coerce_float(
            self.ethics_config.get("min_energy_threshold", 30.0),
            field_name="min_energy_threshold",
            minimum=0.0,
        )
        max_threshold = coerce_float(
            self.ethics_config.get("max_energy_threshold", 200.0),
            field_name="max_energy_threshold",
            minimum=min_threshold,
        )

        adjustment_factor = 1.0
        if hazard_type == "physical_harm":
            adjustment_factor *= 0.70
        elif hazard_type == "psychological_harm":
            adjustment_factor *= 0.90

        memory_state = self._memory_current_state()
        stress_level = float(memory_state.get("stress_level", 0.0) or 0.0)
        if stress_level > 0.60:
            adjustment_factor *= 0.80

        current_hour = datetime.utcnow().hour
        if 8 <= current_hour <= 20:
            adjustment_factor *= 0.90

        threshold = base_threshold * adjustment_factor
        threshold = max(min_threshold, min(max_threshold, threshold))
        return float(round(threshold, 4))

    def _evaluate_constitutional_rule(self, context: Dict[str, Any], rule: str, rule_id: str = "") -> bool:
        statement = str(rule or "").strip()
        if rule_id and rule_id in self.rule_evaluators:
            return bool(self.rule_evaluators[rule_id](context))

        normalized = statement.lower()
        if "personal data" in normalized and "protect" in normalized:
            return self._eval_privacy_protection(context)
        if "minimize data collection" in normalized:
            return self._eval_privacy_minimization(context)
        if "explain decisions" in normalized or "explainability" in normalized:
            return self._eval_transparency_explainability(context)
        if "audit trail" in normalized:
            return self._eval_transparency_auditability(context)
        if "store user personal data longer" in normalized or "retention" in normalized:
            return self._eval_data_retention_policy(context)
        if "anonymity" in normalized:
            return self._eval_respect_user_anonymity(context)
        if "minimum necessary personal information" in normalized:
            return self._eval_minimum_necessary_info(context)
        if "third parties" in normalized and "consent" in normalized:
            return self._eval_no_data_sharing_without_consent(context)
        if "encrypt" in normalized:
            return self._eval_encryption_compliance(context)
        if "gdpr" in normalized or "privacy laws" in normalized:
            return self._eval_gdpr_compliance(context)
        if "access and delete" in normalized:
            return self._eval_data_access_deletion_rights(context)
        if "data breaches" in normalized or "breach" in normalized:
            return self._eval_data_breach_response(context)

        if not statement:
            return self.unknown_rule_policy == "allow"
        return True if self.unknown_rule_policy == "allow" else False

    def _calculate_societal_impact(self, context: Dict[str, Any], dimension: str) -> float:
        try:
            if dimension == "distribution":
                population = list(context.get("affected_population", []))
                if not population:
                    return 0.0
                utilities = [float(item.get("utility", 0.0) or 0.0) for item in population]
                if not utilities or max(utilities) == 0:
                    return 0.0
                return float(max(0.0, 1.0 - (min(utilities) / max(utilities))))

            if dimension == "procedure":
                history = list(context.get("decision_history", []))
                scores = [float(item.get("fairness_score", 0.0) or 0.0) for item in history]
                if len(scores) < 2:
                    return 0.0
                epsilon = 0.90
                mean_score = float(np.mean(scores))
                if mean_score <= 0:
                    return 0.0
                part_sum = sum(score ** (1 - epsilon) for score in scores if score >= 0)
                atkinson = 1 - (1 / mean_score) * (part_sum / len(scores)) ** (1 / (1 - epsilon))
                return float(max(0.0, min(1.0, atkinson)))

            if dimension == "capability":
                vectors = list(context.get("capability_vectors", []))
                if len(vectors) < 2:
                    return 0.0
                totals = [float(sum(vector)) for vector in vectors]
                if not totals or sum(totals) == 0:
                    return 0.0
                sorted_totals = sorted(totals)
                n = len(sorted_totals)
                gini_numerator = sum((idx + 1) * value for idx, value in enumerate(sorted_totals))
                gini = (2 * gini_numerator) / (n * sum(sorted_totals)) - (n + 1) / n
                return float(max(0.0, min(1.0, gini)))
        except Exception as exc:
            logger.error("Error calculating societal impact for %s: %s", dimension, exc)
        return 0.0

    def _evaluate_dynamic_constraint(self, context: Dict[str, Any], constraint: Mapping[str, Any]) -> bool:
        condition = constraint.get("condition")
        if callable(condition):
            return bool(condition(context))
        if isinstance(condition, Mapping):
            flattened = flatten_mapping(context)
            for key, expected in condition.items():
                if flattened.get(str(key)) != expected:
                    return False
            return True
        if isinstance(condition, str):
            flattened = flatten_mapping(context)
            return bool(flattened.get(condition))
        if condition is None:
            return True
        return bool(condition)

    # ------------------------------------------------------------------
    # Rule evaluators
    # ------------------------------------------------------------------
    def _create_rule_evaluator_mapping(self) -> Dict[str, Callable[[Dict[str, Any]], bool]]:
        return {
            "CP_PRIV_001": self._eval_purpose_limitation,
            "CP_PRIV_002": self._eval_informed_consent,
            "CP_PRIV_003": self._eval_privacy_protection,
            "CP_PRIV_004": self._eval_privacy_minimization,
            "CP_TRANS_001": self._eval_transparency_explainability,
            "CP_TRANS_002": self._eval_transparency_auditability,
            "CP_PRIV_005": self._eval_data_retention_policy,
            "CP_PRIV_006": self._eval_respect_user_anonymity,
            "CP_PRIV_007": self._eval_minimum_necessary_info,
            "CP_PRIV_008": self._eval_no_data_sharing_without_consent,
            "CP_PRIV_009": self._eval_encryption_compliance,
            "CP_PRIV_010": self._eval_gdpr_compliance,
            "CP_PRIV_011": self._eval_data_access_deletion_rights,
            "CP_PRIV_012": self._eval_data_breach_response,
        }

    def _eval_purpose_limitation(self, context: Dict[str, Any]) -> bool:
        collected = list(context.get("data_collected", []))
        justifications = dict(context.get("purpose_justifications", {}))
        return all(item in justifications for item in collected)

    def _eval_informed_consent(self, context: Dict[str, Any]) -> bool:
        consent_obtained = bool(context.get("user_consent", False))
        consent_type = str(context.get("consent_type", "none")).strip().lower()
        return consent_obtained and consent_type == "explicit"

    def _eval_privacy_protection(self, context: Dict[str, Any]) -> bool:
        accessed_data = list(context.get("personal_data_accessed", []))
        anonymization = float(context.get("anonymization_level", 0.0) or 0.0)
        encrypted = bool(context.get("data_encrypted", False))
        if not accessed_data:
            return True
        return encrypted and anonymization >= 0.80

    def _eval_privacy_minimization(self, context: Dict[str, Any]) -> bool:
        collected_items = list(context.get("data_collected_items", []))
        justification_ratio = float(context.get("purpose_justification_ratio", 0.0) or 0.0)
        return len(collected_items) < 5 or justification_ratio > 0.90

    def _eval_transparency_explainability(self, context: Dict[str, Any]) -> bool:
        explanation = str(context.get("decision_explanation", "") or "").strip()
        clarity = float(context.get("explanation_clarity_score", 0.0) or 0.0)
        return len(explanation) > 20 and clarity >= 0.60

    def _eval_transparency_auditability(self, context: Dict[str, Any]) -> bool:
        status = str(context.get("audit_trail_status", "inactive")).strip().lower()
        completeness = float(context.get("log_completeness_score", 0.0) or 0.0)
        return status == "active" and completeness >= 0.85

    def _eval_data_retention_policy(self, context: Dict[str, Any]) -> bool:
        retention_days = int(context.get("data_retention_days", 0) or 0)
        policy_limit = int(context.get("retention_policy_limit", 30) or 30)
        return retention_days <= policy_limit

    def _eval_respect_user_anonymity(self, context: Dict[str, Any]) -> bool:
        identifiers_present = bool(context.get("user_identifiers_present", False))
        return not identifiers_present

    def _eval_minimum_necessary_info(self, context: Dict[str, Any]) -> bool:
        collected = set(context.get("data_collected_items", []))
        required = set(context.get("required_data_items", []))
        return len(collected - required) == 0

    def _eval_no_data_sharing_without_consent(self, context: Dict[str, Any]) -> bool:
        shared = bool(context.get("data_shared_with_third_parties", False))
        consent = bool(context.get("user_consent_obtained", False))
        return not shared or consent

    def _eval_encryption_compliance(self, context: Dict[str, Any]) -> bool:
        return bool(context.get("encryption_at_rest", False)) and bool(context.get("encryption_in_transit", False))

    def _eval_gdpr_compliance(self, context: Dict[str, Any]) -> bool:
        return bool(context.get("gdpr_compliant", False))

    def _eval_data_access_deletion_rights(self, context: Dict[str, Any]) -> bool:
        return bool(context.get("access_mechanism_available", False)) and bool(context.get("deletion_mechanism_available", False))

    def _eval_data_breach_response(self, context: Dict[str, Any]) -> bool:
        breach_detected = bool(context.get("recent_breach_detected", False))
        response_hours = float(context.get("breach_response_time_hours", 0.0) or 0.0)
        return (not breach_detected) or (response_hours <= 72.0)

    # ------------------------------------------------------------------
    # Logging, adaptation, and internal helpers
    # ------------------------------------------------------------------
    def _generate_safety_correction(self, context: Dict[str, Any], hazard_type: str) -> Dict[str, Any]:
        return {
            "action": "constraint_application",
            "parameters": {
                "type": hazard_type,
                "severity": "high",
                "mitigation": "hazard_elimination",
                "safe_energy_threshold": self._calculate_safe_energy_threshold(hazard_type),
            },
        }

    def _constitutional_correction(self, principle: str, rule: str, rule_id: str = "") -> Dict[str, Any]:
        return {
            "action": "constitutional_revision",
            "parameters": {
                "principle": principle,
                "rule_id": rule_id,
                "original_output": "<REDACTED>",
                "revised_output": "<REDACTED>",
                "rule_applied": rule,
            },
        }

    def _mitigation_strategy(self, dimension: str, *, impact_score: float, threshold: float) -> Dict[str, Any]:
        return {
            "action": "impact_mitigation",
            "parameters": {
                "strategy": "compensatory_measure",
                "dimension": dimension,
                "intensity": min(1.0, max(0.1, impact_score)),
                "threshold": threshold,
            },
        }

    def _log_violation(self, context: Dict[str, Any], result: Dict[str, Any]) -> None:
        entry = build_alignment_event(
            "ethical_constraint_violation",
            severity=result.get("severity", "high"),
            risk_level=result.get("risk_level", "high"),
            source="ethical_constraints",
            tags=["violation", "ethics", "alignment"],
            context=context,
            metadata={
                "violation_count": len(result.get("violations", [])),
                "constraint_weights": dict(self.constraint_weights),
            },
            payload={
                "violations": result.get("violations", []),
                "corrective_actions": result.get("corrective_actions", []),
                "explanations": result.get("explanations", []),
            },
        )
        self.audit_log.append(entry)
        self._trim_audit_log()

    def _log_success(self, context: Dict[str, Any], result: Dict[str, Any]) -> None:
        entry = build_alignment_event(
            "ethical_constraint_success",
            severity="low",
            risk_level="low",
            source="ethical_constraints",
            tags=["success", "ethics", "alignment"],
            context=context,
            metadata={"total_risk": result.get("summary", {}).get("total_risk", 0.0)},
            payload={"approved": True},
        )
        self.audit_log.append(entry)
        self._trim_audit_log()

    def _log_to_memory(self, context: Dict[str, Any], result: Dict[str, Any]) -> None:
        try:
            total_risk = float(result.get("summary", {}).get("total_risk", 0.0))
            approved = bool(result.get("approved", False))
            self.alignment_memory.log_evaluation(
                metric=normalize_metric_name("ethical_total_risk"),
                value=total_risk,
                threshold=max(1.0, float(len(result.get("violations", []))) + 0.5),
                context=context,
                source="ethical_constraints",
                tags=["ethics", result.get("risk_level", "unknown")],
                metadata={
                    "approved": approved,
                    "severity": result.get("severity"),
                    "trace_id": result.get("trace_id"),
                },
            )
            self.alignment_memory.record_outcome(
                context=context,
                outcome={
                    "alignment_score": max(0.0, 1.0 - min(total_risk, 1.0)),
                    "bias_rate": 0.0,
                    "ethics_violations": len(result.get("violations", [])),
                    "violation": not approved,
                },
                source="ethical_constraints",
                tags=["ethics", result.get("risk_level", "unknown")],
                metadata={"severity": result.get("severity")},
            )
        except Exception as exc:
            logger.warning("Failed to write ethical constraint telemetry to memory: %s", exc)

    def _adapt_constraints(self, context: Dict[str, Any], result: Dict[str, Any]) -> None:
        violations = list(result.get("violations", []))
        if not violations:
            return

        for violation in violations:
            category = self._category_from_violation(violation)
            if category not in self.constraint_weights:
                continue
            old_weight = float(self.constraint_weights[category])
            new_weight = old_weight * (1.0 + self.adaptation_rate)
            self.constraint_weights[category] = max(self.weight_floor, min(self.weight_ceiling, round(new_weight, 4)))
            logger.info(
                "Adapted ethical constraint weight | category=%s old=%.4f new=%.4f",
                category,
                old_weight,
                self.constraint_weights[category],
            )

    def _trim_audit_log(self) -> None:
        if len(self.audit_log) > self.audit_history_limit:
            self.audit_log = self.audit_log[-self.audit_history_limit :]

    def _constraint_index(self, constraint_id: str) -> Optional[int]:
        for idx, item in enumerate(self.constraints):
            if str(item.get("id")) == constraint_id:
                return idx
        return None

    def _resolve_constraint_weight(self, category: Any, *, fallback_category: Optional[str] = None) -> float:
        candidates = [str(category)]
        if fallback_category:
            candidates.append(str(fallback_category))
        if str(category) == "privacy":
            candidates.append("privacy")
        for candidate in candidates:
            if candidate in self.constraint_weights:
                return float(self.constraint_weights[candidate])
        return 1.0

    def _priority_score(self, category: str) -> float:
        if category in self.constraint_priorities:
            return float(len(self.constraint_priorities) - self.constraint_priorities.index(category))
        return 0.5

    def _category_from_violation(self, violation: str) -> str:
        text = str(violation)
        if text.startswith("constitutional_violation:"):
            parts = text.split(":")
            if len(parts) >= 2:
                return parts[1]
        if text.startswith("societal_impact:"):
            parts = text.split(":")
            if len(parts) >= 2:
                return parts[1]
        if text.endswith("_violation"):
            return text.replace("_violation", "")
        if text.startswith("dynamic_constraint_violation"):
            return "dynamic"
        return text

    def _memory_current_state(self) -> Dict[str, Any]:
        if hasattr(self.alignment_memory, "_get_current_state"):
            try:
                state = self.alignment_memory._get_current_state()
                if isinstance(state, Mapping):
                    return dict(state)
            except Exception:
                pass
        if hasattr(self.alignment_memory, "get_memory_report"):
            try:
                report = self.alignment_memory.get_memory_report()
                if isinstance(report, Mapping):
                    return {"memory_report": json_safe(report)}
            except Exception:
                pass
        return {}

    def _severity_from_risk(self, risk_value: float, violation_count: int) -> str:
        bands = dict(self.severity_bands.get("severity", {}))
        critical = float(bands.get("critical", 3.0))
        high = float(bands.get("high", 1.5))
        medium = float(bands.get("medium", 0.5))
        if violation_count >= 4 or risk_value >= critical:
            return "critical"
        if violation_count >= 2 or risk_value >= high:
            return "high"
        if violation_count >= 1 or risk_value >= medium:
            return "medium"
        return "low"

    def _risk_level_from_risk(self, risk_value: float, violation_count: int) -> str:
        bands = dict(self.severity_bands.get("risk", {}))
        critical = float(bands.get("critical", 3.0))
        high = float(bands.get("high", 1.5))
        medium = float(bands.get("medium", 0.5))
        if violation_count >= 4 or risk_value >= critical:
            return "critical"
        if violation_count >= 2 or risk_value >= high:
            return "high"
        if violation_count >= 1 or risk_value >= medium:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Compatibility helper retained from original script
    # ------------------------------------------------------------------
    def _hazard_condition(self, context: Dict[str, Any], rule: str) -> bool:
        """
        Retained compatibility surface for the original test harness.

        The implementation now evaluates a small set of symbolic propositions in
        a conservative, non-authoritative way and returns whether the condition
        should be treated as hazardous.
        """
        propositions = {
            "Prevent injury to humans": not any(bool(person.get("injured", False)) for person in context.get("affected_people", [])),
            "Avoid property damage": all(float(asset.get("damage_level", 0.0) or 0.0) < 0.10 for asset in context.get("affected_assets", [])),
        }
        if rule not in propositions:
            return False
        compliant = propositions[rule]
        return not compliant


if __name__ == "__main__":
    print("\n=== Running Ethical Constraints ===\n")
    printer.status("TEST", "Ethical Constraints initialized", "info")

    ethics = EthicalConstraints()

    action_context = {
        "affected_population": [
            {"utility": 0.90},
            {"utility": 0.35},
            {"utility": 0.55},
        ],
        "decision_history": [
            {"fairness_score": 0.86},
            {"fairness_score": 0.63},
            {"fairness_score": 0.71},
        ],
        "capability_vectors": [
            [0.70, 0.80],
            [0.50, 0.60],
            [0.55, 0.45],
        ],
        "decision_engine": {"is_active": True},
        "output_mechanisms": {"type": "motor"},
        "feedback_systems": {"type": "sensor"},
        "affected_environment": {"state": "stable"},
        "action_parameters": {"kinetic_energy": 32, "stress_index": 0.22},
        "potential_energy": 20,
        "kinetic_energy": 15,
        "informational_entropy": 1.50,
        "decision_explanation": "The decision was made using policy constraints, fairness checks, and rule-based prioritization.",
        "explanation_clarity_score": 0.78,
        "audit_trail_status": "active",
        "log_completeness_score": 0.93,
        "personal_data_accessed": ["email"],
        "anonymization_level": 0.90,
        "data_encrypted": True,
        "data_collected_items": ["email"],
        "required_data_items": ["email"],
        "purpose_justification_ratio": 1.0,
        "encryption_at_rest": True,
        "encryption_in_transit": True,
        "gdpr_compliant": True,
        "access_mechanism_available": True,
        "deletion_mechanism_available": True,
        "user_consent": True,
        "consent_type": "explicit",
        "user_consent_obtained": True,
        "data_shared_with_third_parties": False,
        "data_retention_days": 14,
        "retention_policy_limit": 30,
        "recent_breach_detected": False,
        "purpose_justifications": {"email": "account_recovery"},
        "data_collected": ["email"],
    }

    context = {
        "affected_people": [
            {"name": "John", "injured": False},
            {"name": "Sarah", "injured": False},
        ],
        "affected_assets": [
            {"id": 1, "damage_level": 0.05},
            {"id": 2, "damage_level": 0.03},
        ],
    }

    printer.pretty("state", ethics.get_current_state(), "success")
    printer.pretty("weights", ethics.verify_constraint_weights(auto_fix=True), "success")
    printer.pretty("enforce", ethics.enforce(action_context=action_context), "success")
    printer.pretty("core", ethics.enforce_core_constraints(memory_snapshot=action_context, constraint_level="emergency"), "success")
    printer.pretty("constraints", ethics.get_constraints(active_only=False)[:5], "success")
    printer.pretty(
        "dynamic_add",
        ethics.add_constraint(
            constraint_id="RUNTIME_DEMO_001",
            condition={"affected_environment.state": "stable"},
            action={"type": "notify_only"},
            weight=0.8,
            priority=0.9,
            category="dynamic",
            severity="low",
        ),
        "success",
    )
    printer.pretty("hazard", ethics._hazard_condition(rule="Prevent injury to humans", context=context), "success")
    printer.pretty("audit_size", {"audit_log_entries": len(ethics.audit_log)}, "success")

    print("\n=== Test ran successfully ===\n")
