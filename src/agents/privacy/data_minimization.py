"""
- Purpose-based field suppression.
- Tokenization/masking/redaction before storage or tool calls.
- Least-data-required policy checks.
"""

from __future__ import annotations

import hashlib
import json

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .utils import (get_config_section, load_global_config,
                    # error
                    ToolPayloadSanitizationError, PrivacyConfigurationError, PrivacyError,
                    PolicyEvaluationError, PrivacyDecision, RedactionError,
                    normalize_privacy_exception, sanitize_privacy_context)
from .privacy_memory import PrivacyMemory
from logs.logger import PrettyPrinter, get_logger

logger = get_logger("Data Minimization and Redaction")
printer = PrettyPrinter


class DataMinimization:
    """Runtime data minimization and redaction engine.

    The minimization layer enforces least-data-required rules before payloads are
    persisted or transferred to downstream tools. It combines allowlist-based
    suppression with field-level redaction and records the resulting privacy
    decision in ``privacy_memory`` for traceability.
    """

    SUPPORTED_STRATEGIES = {
        "keep",
        "drop",
        "mask",
        "partial_mask",
        "last4",
        "tokenize",
        "hash",
    }

    def __init__(self) -> None:
        self.config = load_global_config()
        self.min_config = get_config_section("data_minimization")
        self.memory = PrivacyMemory()

        self.enabled = bool(self.min_config.get("enabled", True))
        self.strict_mode = bool(self.min_config.get("strict_mode", True))
        self.sanitize_freeform_context = bool(self.min_config.get("sanitize_freeform_context", True))
        self.record_decisions_in_memory = bool(self.min_config.get("record_decisions_in_memory", True))
        self.record_lineage_events = bool(self.min_config.get("record_lineage_events", True))
        self.write_shared_contract = bool(self.min_config.get("write_shared_contract", True))
        self.store_redaction_metadata = bool(self.min_config.get("store_redaction_metadata", True))
        self.enforce_allowlist = bool(self.min_config.get("enforce_allowlist", True))
        self.drop_unlisted_fields = bool(self.min_config.get("drop_unlisted_fields", True))
        self.fail_on_unmapped_sensitive_fields = bool(
            self.min_config.get("fail_on_unmapped_sensitive_fields", False)
        )
        self.allow_empty_payload_after_minimization = bool(
            self.min_config.get("allow_empty_payload_after_minimization", False)
        )

        self.default_policy_version = str(self.min_config.get("default_policy_version", "v1"))
        self.default_decision_stage = str(
            self.min_config.get("default_decision_stage", "minimization.runtime_gate")
        )
        self.default_redaction_reason = str(
            self.min_config.get("default_redaction_reason", "least-data-required enforcement")
        )
        self.default_tool_context = str(self.min_config.get("default_tool_context", "tool_call"))
        self.default_storage_context = str(self.min_config.get("default_storage_context", "storage"))
        self.default_sensitive_strategy = str(
            self.min_config.get("default_sensitive_strategy", "tokenize")
        ).lower()
        self.secret_field_strategy = str(self.min_config.get("secret_field_strategy", "drop")).lower()
        self.identifier_field_strategy = str(
            self.min_config.get("identifier_field_strategy", "partial_mask")
        ).lower()

        self.partial_mask_visible_prefix = int(self.min_config.get("partial_mask_visible_prefix", 2))
        self.partial_mask_visible_suffix = int(self.min_config.get("partial_mask_visible_suffix", 2))
        self.max_depth = int(self.min_config.get("max_depth", 8))
        self.max_items_per_container = int(self.min_config.get("max_items_per_container", 200))
        self.max_allowed_fields = int(self.min_config.get("max_allowed_fields", 200))
        self.max_sensitive_fields = int(self.min_config.get("max_sensitive_fields", 200))
        self.max_field_strategies = int(self.min_config.get("max_field_strategies", 200))

        self.sensitive_field_names = self._normalize_name_list(
            self.min_config.get(
                "sensitive_field_names",
                [
                    "email",
                    "phone",
                    "full_name",
                    "name",
                    "address",
                    "ssn",
                    "social_security_number",
                    "passport",
                    "national_id",
                    "account_number",
                    "patient_id",
                    "diagnosis",
                    "medical_record",
                    "dob",
                    "date_of_birth",
                    "ip_address",
                    "device_id",
                    "user_id",
                ],
            )
        )
        self.secret_field_names = self._normalize_name_list(
            self.min_config.get(
                "secret_field_names",
                [
                    "password",
                    "secret",
                    "token",
                    "api_key",
                    "authorization",
                    "cookie",
                    "refresh_token",
                    "access_token",
                    "session",
                    "cvv",
                    "credit_card",
                ],
            )
        )
        self.always_drop_fields = self._normalize_name_list(
            self.min_config.get("always_drop_fields", [])
        )
        self.always_keep_fields = self._normalize_name_list(
            self.min_config.get("always_keep_fields", [])
        )
        self.default_field_strategies = self._normalize_strategy_map(
            self.min_config.get("default_field_strategies", {})
        )

        self._validate_config()
        logger.info("DataMinimization initialized with production-ready controls.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        integer_fields = {
            "partial_mask_visible_prefix": self.partial_mask_visible_prefix,
            "partial_mask_visible_suffix": self.partial_mask_visible_suffix,
            "max_depth": self.max_depth,
            "max_items_per_container": self.max_items_per_container,
            "max_allowed_fields": self.max_allowed_fields,
            "max_sensitive_fields": self.max_sensitive_fields,
            "max_field_strategies": self.max_field_strategies,
        }
        for field_name, value in integer_fields.items():
            if value < 0:
                raise PrivacyConfigurationError(
                    section="data_minimization",
                    details=f"'{field_name}' must be >= 0, received {value!r}",
                )

        if self.default_sensitive_strategy not in self.SUPPORTED_STRATEGIES:
            raise PrivacyConfigurationError(
                section="data_minimization",
                details=(
                    "'default_sensitive_strategy' must be one of "
                    f"{sorted(self.SUPPORTED_STRATEGIES)}, received {self.default_sensitive_strategy!r}"
                ),
            )
        if self.secret_field_strategy not in self.SUPPORTED_STRATEGIES:
            raise PrivacyConfigurationError(
                section="data_minimization",
                details=(
                    "'secret_field_strategy' must be one of "
                    f"{sorted(self.SUPPORTED_STRATEGIES)}, received {self.secret_field_strategy!r}"
                ),
            )
        if self.identifier_field_strategy not in self.SUPPORTED_STRATEGIES:
            raise PrivacyConfigurationError(
                section="data_minimization",
                details=(
                    "'identifier_field_strategy' must be one of "
                    f"{sorted(self.SUPPORTED_STRATEGIES)}, received {self.identifier_field_strategy!r}"
                ),
            )

    def _require_enabled(self, operation: str) -> None:
        if not self.enabled:
            raise PolicyEvaluationError(
                stage=operation,
                details="data_minimization is disabled by configuration",
                context={"config_section": "data_minimization"},
            )

    @staticmethod
    def _normalize_identity(value: Any, field_name: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"'{field_name}' must be a non-empty string")
        return normalized

    @staticmethod
    def _normalize_name(value: Any) -> str:
        return str(value).strip().lower().replace("-", "_").replace(" ", "_")

    def _normalize_name_list(self, values: Optional[Sequence[Any]]) -> List[str]:
        if not values:
            return []
        normalized: List[str] = []
        seen = set()
        for value in values:
            item = self._normalize_name(value)
            if not item or item in seen:
                continue
            normalized.append(item)
            seen.add(item)
        return normalized

    def _normalize_field_patterns(self, values: Optional[Sequence[Any]], *, max_items: int) -> List[str]:
        normalized = self._normalize_name_list(values)
        if len(normalized) > max_items:
            raise ValueError(f"Too many field patterns supplied; maximum is {max_items}")
        return normalized

    def _normalize_strategy_map(self, value: Optional[Mapping[str, Any]]) -> Dict[str, str]:
        if not value:
            return {}
        normalized: Dict[str, str] = {}
        if len(value) > self.max_field_strategies:
            raise ValueError(
                f"Too many field strategies supplied; maximum is {self.max_field_strategies}"
            )
        for key, strategy in value.items():
            normalized_key = self._normalize_name(key)
            normalized_strategy = self._normalize_name(strategy)
            if normalized_strategy not in self.SUPPORTED_STRATEGIES:
                raise ValueError(
                    f"Unsupported redaction strategy {strategy!r} for field {key!r}"
                )
            normalized[normalized_key] = normalized_strategy
        return normalized

    def _normalize_context(self, payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not payload:
            return {}
        if self.sanitize_freeform_context:
            return sanitize_privacy_context(payload)
        return dict(payload)

    @staticmethod
    def _strip_indices(path: str) -> str:
        result: List[str] = []
        inside = False
        for char in path:
            if char == "[":
                inside = True
                continue
            if char == "]":
                inside = False
                continue
            if not inside:
                result.append(char)
        return "".join(result)

    def _path_variants(self, path: str) -> Tuple[str, str, str]:
        normalized = self._normalize_name(path)
        without_indices = self._normalize_name(self._strip_indices(path))
        leaf = without_indices.split(".")[-1] if without_indices else without_indices
        return normalized, without_indices, leaf

    def _pattern_matches_path(self, pattern: str, path: str) -> bool:
        normalized_pattern = self._normalize_name(pattern)
        normalized_path, without_indices, leaf = self._path_variants(path)
        if normalized_pattern in {"*", "**"}:
            return True
        if normalized_pattern.endswith(".*"):
            prefix = normalized_pattern[:-2]
            return without_indices == prefix or without_indices.startswith(prefix + ".")
        if normalized_pattern == normalized_path or normalized_pattern == without_indices:
            return True
        if "." not in normalized_pattern and normalized_pattern == leaf:
            return True
        return without_indices.startswith(normalized_pattern + ".")

    def _path_allowed(self, path: str, patterns: Sequence[str]) -> bool:
        if not patterns:
            return True
        return any(self._pattern_matches_path(pattern, path) for pattern in patterns)

    def _path_has_descendant(self, path: str, patterns: Sequence[str]) -> bool:
        if not patterns:
            return True
        _, without_indices, _ = self._path_variants(path)
        prefix = without_indices + "." if without_indices else ""
        for pattern in patterns:
            normalized_pattern = self._normalize_name(pattern)
            if normalized_pattern in {"*", "**"}:
                return True
            if normalized_pattern.endswith(".*"):
                normalized_pattern = normalized_pattern[:-2]
            if normalized_pattern == without_indices:
                return True
            if prefix and normalized_pattern.startswith(prefix):
                return True
            if "." not in normalized_pattern and normalized_pattern == without_indices.split(".")[-1]:
                return True
        return False

    def _is_sensitive_field(self, path: str, sensitive_fields: Sequence[str]) -> Tuple[bool, str]:
        _, without_indices, leaf = self._path_variants(path)
        if self.always_drop_fields and self._path_allowed(path, self.always_drop_fields):
            return True, "always_drop"
        if sensitive_fields and self._path_allowed(path, sensitive_fields):
            return True, "runtime_sensitive"
        if self.secret_field_names and self._path_allowed(path, self.secret_field_names):
            return True, "secret"
        if self.sensitive_field_names and self._path_allowed(path, self.sensitive_field_names):
            return True, "sensitive"
        if leaf in self.secret_field_names:
            return True, "secret"
        if leaf in self.sensitive_field_names:
            return True, "sensitive"
        if without_indices in self.secret_field_names:
            return True, "secret"
        if without_indices in self.sensitive_field_names:
            return True, "sensitive"
        return False, ""

    @staticmethod
    def _stable_token(value: Any, *, prefix: str = "tok") -> str:
        raw = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()[:16]
        return f"{prefix}_{digest}"

    @staticmethod
    def _short_hash(value: Any) -> str:
        raw = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def _mask_email(self, value: str) -> str:
        if "@" not in value:
            return self._partial_mask(value)
        local, domain = value.split("@", 1)
        masked_local = self._partial_mask(local)
        return f"{masked_local}@{domain}"

    def _partial_mask(self, value: str) -> str:
        if not value:
            return value
        if len(value) <= self.partial_mask_visible_prefix + self.partial_mask_visible_suffix:
            return "*" * len(value)
        prefix = value[: self.partial_mask_visible_prefix]
        suffix = value[-self.partial_mask_visible_suffix :] if self.partial_mask_visible_suffix else ""
        middle = "*" * max(1, len(value) - len(prefix) - len(suffix))
        return f"{prefix}{middle}{suffix}"

    def _apply_scalar_strategy(self, value: Any, strategy: str) -> Any:
        if strategy == "keep":
            return value
        if strategy == "drop":
            return None
        if strategy == "tokenize":
            return self._stable_token(value)
        if strategy == "hash":
            return self._short_hash(value)
        if isinstance(value, (int, float, bool)):
            if strategy in {"mask", "partial_mask", "last4"}:
                return self._stable_token(value, prefix="num")
        if isinstance(value, (dict, list, tuple, set)):
            if strategy == "mask":
                return {"__redacted__": True, "type": type(value).__name__}
            return self._stable_token(value)

        text = str(value)
        if strategy == "mask":
            return "[REDACTED]"
        if strategy == "partial_mask":
            if "@" in text:
                return self._mask_email(text)
            return self._partial_mask(text)
        if strategy == "last4":
            if len(text) <= 4:
                return "*" * len(text)
            return f"{'*' * max(1, len(text) - 4)}{text[-4:]}"
        return "[REDACTED]"

    @staticmethod
    def _count_leaf_fields(value: Any) -> int:
        if isinstance(value, Mapping):
            if not value:
                return 1
            return sum(DataMinimization._count_leaf_fields(item) for item in value.values())
        if isinstance(value, list):
            if not value:
                return 1
            return sum(DataMinimization._count_leaf_fields(item) for item in value)
        return 1

    @staticmethod
    def _collect_leaf_paths(value: Any, path: str = "") -> List[str]:
        paths: List[str] = []
        if isinstance(value, Mapping):
            for key, item in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                paths.extend(DataMinimization._collect_leaf_paths(item, child_path))
            return paths
        if isinstance(value, list):
            for index, item in enumerate(value):
                child_path = f"{path}[{index}]"
                paths.extend(DataMinimization._collect_leaf_paths(item, child_path))
            return paths
        return [path]

    def _strategy_for_path(self, path: str, *,
        field_strategies: Mapping[str, str],
        sensitive_fields: Sequence[str],
    ) -> Optional[str]:
        for pattern, strategy in field_strategies.items():
            if self._pattern_matches_path(pattern, path):
                return strategy

        sensitive, category = self._is_sensitive_field(path, sensitive_fields)
        if not sensitive:
            return None
        if category in {"always_drop", "secret"}:
            return self.secret_field_strategy
        if category == "sensitive":
            return self.identifier_field_strategy
        return self.default_sensitive_strategy

    def _build_allowlist(self, *,
        allowed_fields: Sequence[str],
        required_fields: Sequence[str],
    ) -> List[str]:
        if not self.enforce_allowlist:
            return []
        merged: List[str] = []
        seen = set()
        for field in [*required_fields, *allowed_fields, *self.always_keep_fields]:
            normalized = self._normalize_name(field)
            if normalized and normalized not in seen:
                merged.append(normalized)
                seen.add(normalized)
        return merged

    def _minimize_node(self, value: Any, *, path: str,
        allowlist: Sequence[str],
        field_strategies: Mapping[str, str],
        sensitive_fields: Sequence[str],
        removed_fields: List[str],
        masked_fields: List[str],
        redaction_actions: List[str],
        detected_entities: List[str],
        depth: int,
    ) -> Tuple[bool, Any]:
        if depth > self.max_depth:
            raise PolicyEvaluationError(
                stage="data_minimization.recursive_walk",
                details=f"Maximum payload depth exceeded at path '{path}'",
                context={"max_depth": self.max_depth, "path": path},
            )

        if isinstance(value, Mapping):
            result: Dict[str, Any] = {}
            items = list(value.items())
            for index, (key, item) in enumerate(items):
                if index >= self.max_items_per_container:
                    raise PolicyEvaluationError(
                        stage="data_minimization.recursive_walk",
                        details=(
                            f"Container at path '{path or '<root>'}' exceeds the configured item limit"
                        ),
                        context={"max_items_per_container": self.max_items_per_container, "path": path},
                    )
                child_path = f"{path}.{key}" if path else str(key)
                is_leaf = not isinstance(item, (Mapping, list))

                if is_leaf and allowlist and not self._path_allowed(child_path, allowlist):
                    if self.drop_unlisted_fields:
                        removed_fields.append(child_path)
                        redaction_actions.append(f"drop:{child_path}")
                        continue
                    raise PolicyEvaluationError(
                        stage="data_minimization.allowlist",
                        details=f"Field '{child_path}' is not allowed for the approved purpose.",
                        context={"path": child_path, "allowlist": list(allowlist)},
                    )

                keep_child, minimized_child = self._minimize_node(
                    item,
                    path=child_path,
                    allowlist=allowlist,
                    field_strategies=field_strategies,
                    sensitive_fields=sensitive_fields,
                    removed_fields=removed_fields,
                    masked_fields=masked_fields,
                    redaction_actions=redaction_actions,
                    detected_entities=detected_entities,
                    depth=depth + 1,
                )
                if keep_child:
                    result[key] = minimized_child
                elif isinstance(item, (Mapping, list)) and self._path_has_descendant(child_path, allowlist):
                    continue
            return bool(result), result

        if isinstance(value, list):
            result_list: List[Any] = []
            for index, item in enumerate(value[: self.max_items_per_container]):
                child_path = f"{path}[{index}]"
                keep_child, minimized_child = self._minimize_node(
                    item,
                    path=child_path,
                    allowlist=allowlist,
                    field_strategies=field_strategies,
                    sensitive_fields=sensitive_fields,
                    removed_fields=removed_fields,
                    masked_fields=masked_fields,
                    redaction_actions=redaction_actions,
                    detected_entities=detected_entities,
                    depth=depth + 1,
                )
                if keep_child:
                    result_list.append(minimized_child)
            return bool(result_list), result_list

        strategy = self._strategy_for_path(
            path,
            field_strategies=field_strategies,
            sensitive_fields=sensitive_fields,
        )

        sensitive, category = self._is_sensitive_field(path, sensitive_fields)
        if sensitive:
            detected_entities.append(self._path_variants(path)[2])

        if sensitive and strategy is None and self.fail_on_unmapped_sensitive_fields:
            raise RedactionError(
                operation="unmapped_sensitive_field",
                details=f"Sensitive field '{path}' has no redaction strategy.",
                context={"path": path, "category": category},
            )

        if strategy is None or strategy == "keep":
            return True, value

        if strategy == "drop":
            removed_fields.append(path)
            redaction_actions.append(f"drop:{path}")
            return False, None

        redacted_value = self._apply_scalar_strategy(value, strategy)
        masked_fields.append(path)
        redaction_actions.append(f"{strategy}:{path}")
        return True, redacted_value

    def _deduplicate(self, values: Sequence[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for value in values:
            if value not in seen:
                result.append(value)
                seen.add(value)
        return result

    def _compute_sensitivity_score(self, *,
        original_field_count: int,
        masked_fields: Sequence[str],
        removed_fields: Sequence[str],
        detected_entities: Sequence[str],
    ) -> float:
        if original_field_count <= 0:
            return 0.0
        weighted = (len(masked_fields) * 0.75) + (len(removed_fields) * 1.0) + (len(set(detected_entities)) * 0.25)
        return min(1.0, round(weighted / float(original_field_count), 4))

    def _record_memory_artifacts(self, *,
        request_id: str,
        stage: str,
        decision: PrivacyDecision,
        rationale: str,
        sanitized_payload: Any,
        masked_fields: Sequence[str],
        removed_fields: Sequence[str],
        redaction_actions: Sequence[str],
        detected_entities: Sequence[str],
        sensitivity_score: float,
        subject_id: Optional[str],
        record_id: Optional[str],
        purpose: str,
        policy_id: Optional[str],
        policy_version: Optional[str],
        audit_trail_ref: Optional[str],
        source_context: Optional[str],
        destination_context: Optional[str],
        reason: str,
        original_field_count: int,
        retained_field_count: int,
        context: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        memory_updates: Dict[str, Any] = {}
        contract_payload = {
            "request_id": request_id,
            "sensitivity_score": sensitivity_score,
            "detected_entities": self._deduplicate(detected_entities),
            "redaction_actions": list(redaction_actions),
            "audit_trail_ref": audit_trail_ref,
        }

        if self.write_shared_contract:
            memory_updates["shared_contract"] = self.memory.write_request_contract(**contract_payload)

        transformation_ref = None
        if self.store_redaction_metadata and (masked_fields or removed_fields):
            transformation = self.memory.store_redaction_transformation(
                request_id=request_id,
                record_id=record_id or f"request::{request_id}",
                masked_fields=self._deduplicate([*masked_fields, *removed_fields]),
                strategy="|".join(sorted({action.split(":", 1)[0] for action in redaction_actions})) or "keep",
                reason=reason,
                original_field_count=original_field_count,
                retained_field_count=retained_field_count,
                context={
                    "destination_context": destination_context,
                    "source_context": source_context,
                    "sanitized_payload_preview": sanitize_privacy_context(
                        {"sanitized_payload": sanitized_payload}
                    ).get("sanitized_payload"),
                },
                audit_trail_ref=audit_trail_ref,
            )
            transformation_ref = transformation.get("latest_transformation_ref")
            memory_updates["transformation"] = transformation

        if self.record_decisions_in_memory:
            memory_updates["decision"] = self.memory.record_privacy_decision(
                request_id=request_id,
                stage=stage,
                decision=decision,
                rationale=rationale,
                subject_id=subject_id,
                record_id=record_id,
                purpose=purpose,
                sensitivity_score=sensitivity_score,
                detected_entities=self._deduplicate(detected_entities),
                redaction_actions=list(redaction_actions),
                policy_id=policy_id,
                policy_version=policy_version or self.default_policy_version,
                audit_trail_ref=audit_trail_ref,
                context={
                    "destination_context": destination_context,
                    "source_context": source_context,
                    "masked_fields": list(masked_fields),
                    "removed_fields": list(removed_fields),
                    **self._normalize_context(context),
                },
            )

        if self.record_lineage_events and (source_context or destination_context):
            memory_updates["lineage"] = self.memory.record_lineage_event(
                request_id=request_id,
                operation="payload_minimization",
                stage=stage,
                record_id=record_id,
                source_context=source_context,
                destination_context=destination_context,
                subject_id=subject_id,
                purpose=purpose,
                context={
                    "decision": decision.value,
                    "transformation_ref": transformation_ref,
                    "redaction_actions": list(redaction_actions),
                },
                audit_trail_ref=audit_trail_ref,
            )
        return memory_updates

    def _handle_exception(self, exc: Exception, *, stage: str,
                          context: Optional[Mapping[str, Any]] = None) -> PrivacyError:
        if isinstance(exc, PrivacyError):
            return exc
        return normalize_privacy_exception(exc, stage=stage, context=dict(context or {}))

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    def evaluate_least_data_required(
        self,
        payload: Mapping[str, Any],
        *,
        purpose: str,
        allowed_fields: Optional[Sequence[str]] = None,
        required_fields: Optional[Sequence[str]] = None,
        sensitive_fields: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        operation = "evaluate_least_data_required"
        try:
            self._require_enabled(operation)
            self._normalize_identity(purpose, "purpose")
            normalized_allowed = self._normalize_field_patterns(
                allowed_fields,
                max_items=self.max_allowed_fields,
            )
            normalized_required = self._normalize_field_patterns(
                required_fields,
                max_items=self.max_allowed_fields,
            )
            normalized_sensitive = self._normalize_field_patterns(
                sensitive_fields,
                max_items=self.max_sensitive_fields,
            )
            allowlist = self._build_allowlist(
                allowed_fields=normalized_allowed,
                required_fields=normalized_required,
            )

            leaf_paths = self._collect_leaf_paths(payload)
            allowed_paths = [path for path in leaf_paths if self._path_allowed(path, allowlist)] if allowlist else leaf_paths
            dropped_paths = [path for path in leaf_paths if allowlist and not self._path_allowed(path, allowlist)]
            sensitive_detected = [
                self._path_variants(path)[2]
                for path in leaf_paths
                if self._is_sensitive_field(path, normalized_sensitive)[0]
            ]

            return {
                "purpose": purpose,
                "enforce_allowlist": self.enforce_allowlist,
                "allowed_fields": normalized_allowed,
                "required_fields": normalized_required,
                "sensitive_fields": normalized_sensitive,
                "allowed_paths": allowed_paths,
                "dropped_paths": dropped_paths,
                "detected_sensitive_entities": self._deduplicate(sensitive_detected),
                "original_field_count": len(leaf_paths),
                "projected_retained_field_count": len(allowed_paths),
            }
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="data_minimization.evaluate_least_data_required",
                context={"purpose": purpose},
            ) from exc

    def minimize_payload(self, payload: Mapping[str, Any], *, purpose: str, request_id: str,
        stage: Optional[str] = None,
        subject_id: Optional[str] = None,
        record_id: Optional[str] = None,
        allowed_fields: Optional[Sequence[str]] = None,
        required_fields: Optional[Sequence[str]] = None,
        sensitive_fields: Optional[Sequence[str]] = None,
        field_strategies: Optional[Mapping[str, Any]] = None,
        reason: Optional[str] = None,
        source_context: Optional[str] = None,
        destination_context: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        operation = "minimize_payload"
        try:
            self._require_enabled(operation)
            normalized_request_id = self._normalize_identity(request_id, "request_id")
            normalized_purpose = self._normalize_identity(purpose, "purpose")
            normalized_stage = str(stage or self.default_decision_stage)
            normalized_reason = str(reason or self.default_redaction_reason)
            normalized_allowed = self._normalize_field_patterns(
                allowed_fields,
                max_items=self.max_allowed_fields,
            )
            normalized_required = self._normalize_field_patterns(
                required_fields,
                max_items=self.max_allowed_fields,
            )
            normalized_sensitive = self._normalize_field_patterns(
                sensitive_fields,
                max_items=self.max_sensitive_fields,
            )
            normalized_field_strategies = {
                **self.default_field_strategies,
                **self._normalize_strategy_map(field_strategies),
            }

            allowlist = self._build_allowlist(
                allowed_fields=normalized_allowed,
                required_fields=normalized_required,
            )
            original_field_count = self._count_leaf_fields(payload)
            removed_fields: List[str] = []
            masked_fields: List[str] = []
            redaction_actions: List[str] = []
            detected_entities: List[str] = []

            keep_root, sanitized_payload = self._minimize_node(
                payload,
                path="",
                allowlist=allowlist,
                field_strategies=normalized_field_strategies,
                sensitive_fields=normalized_sensitive,
                removed_fields=removed_fields,
                masked_fields=masked_fields,
                redaction_actions=redaction_actions,
                detected_entities=detected_entities,
                depth=0,
            )

            if not keep_root:
                sanitized_payload = {}

            retained_field_paths = self._collect_leaf_paths(sanitized_payload)
            retained_field_count = len(retained_field_paths)
            missing_required_fields = [
                field
                for field in normalized_required
                if not any(
                    self._pattern_matches_path(field, retained_path)
                    or self._pattern_matches_path(retained_path, field)
                    for retained_path in retained_field_paths
                )
            ]

            if missing_required_fields and self.strict_mode:
                raise PolicyEvaluationError(
                    stage="data_minimization.required_fields",
                    details="Required fields are missing after minimization.",
                    context={"missing_required_fields": missing_required_fields},
                )

            empty_after_minimization = retained_field_count == 0
            if empty_after_minimization and not self.allow_empty_payload_after_minimization:
                raise ToolPayloadSanitizationError(
                    tool_name=destination_context or source_context or "payload_destination",
                    details="Payload became empty after minimization and empty payloads are disallowed.",
                    context={
                        "request_id": normalized_request_id,
                        "purpose": normalized_purpose,
                        "stage": normalized_stage,
                    },
                )

            masked_fields = self._deduplicate(masked_fields)
            removed_fields = self._deduplicate(removed_fields)
            redaction_actions = self._deduplicate(redaction_actions)
            detected_entities = self._deduplicate(detected_entities)
            sensitivity_score = self._compute_sensitivity_score(
                original_field_count=original_field_count,
                masked_fields=masked_fields,
                removed_fields=removed_fields,
                detected_entities=detected_entities,
            )

            decision = (
                PrivacyDecision.MODIFY if (masked_fields or removed_fields) else PrivacyDecision.ALLOW
            )
            rationale = (
                "Payload was minimized to satisfy least-data-required constraints."
                if decision == PrivacyDecision.MODIFY
                else "Payload satisfied least-data-required constraints without modification."
            )

            memory_updates = self._record_memory_artifacts(
                request_id=normalized_request_id,
                stage=normalized_stage,
                decision=decision,
                rationale=rationale,
                sanitized_payload=sanitized_payload,
                masked_fields=masked_fields,
                removed_fields=removed_fields,
                redaction_actions=redaction_actions,
                detected_entities=detected_entities,
                sensitivity_score=sensitivity_score,
                subject_id=subject_id,
                record_id=record_id,
                purpose=normalized_purpose,
                policy_id=policy_id,
                policy_version=policy_version,
                audit_trail_ref=audit_trail_ref,
                source_context=source_context,
                destination_context=destination_context,
                reason=normalized_reason,
                original_field_count=original_field_count,
                retained_field_count=retained_field_count,
                context=context,
            )

            return {
                "request_id": normalized_request_id,
                "purpose": normalized_purpose,
                "stage": normalized_stage,
                "decision": decision.value,
                "sanitized_payload": sanitized_payload,
                "masked_fields": masked_fields,
                "removed_fields": removed_fields,
                "retained_fields": retained_field_paths,
                "missing_required_fields": missing_required_fields,
                "detected_entities": detected_entities,
                "redaction_actions": redaction_actions,
                "source_context": source_context,
                "destination_context": destination_context,
                "original_field_count": original_field_count,
                "retained_field_count": retained_field_count,
                "removed_field_count": len(removed_fields),
                "masked_field_count": len(masked_fields),
                "empty_after_minimization": empty_after_minimization,
                "sensitivity_score": sensitivity_score,
                "policy_id": policy_id,
                "policy_version": policy_version or self.default_policy_version,
                "audit_trail_ref": audit_trail_ref,
                "memory_updates": memory_updates,
            }
        except Exception as exc:
            raise self._handle_exception(
                exc,
                stage="data_minimization.minimize_payload",
                context={
                    "request_id": request_id,
                    "purpose": purpose,
                    "stage": stage or self.default_decision_stage,
                    "destination_context": destination_context,
                    "source_context": source_context,
                },
            ) from exc

    def minimize_for_tool_call(self, tool_name: str, payload: Mapping[str, Any], *,
        purpose: str,
        request_id: str,
        subject_id: Optional[str] = None,
        record_id: Optional[str] = None,
        allowed_fields: Optional[Sequence[str]] = None,
        required_fields: Optional[Sequence[str]] = None,
        sensitive_fields: Optional[Sequence[str]] = None,
        field_strategies: Optional[Mapping[str, Any]] = None,
        source_context: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_tool_name = self._normalize_identity(tool_name, "tool_name")
        return self.minimize_payload(
            payload,
            purpose=purpose,
            request_id=request_id,
            stage=f"{self.default_tool_context}.{normalized_tool_name}",
            subject_id=subject_id,
            record_id=record_id,
            allowed_fields=allowed_fields,
            required_fields=required_fields,
            sensitive_fields=sensitive_fields,
            field_strategies=field_strategies,
            reason=f"Least-data-required minimization before tool call '{normalized_tool_name}'.",
            source_context=source_context or "runtime",
            destination_context=normalized_tool_name,
            policy_id=policy_id,
            policy_version=policy_version,
            audit_trail_ref=audit_trail_ref,
            context={"tool_name": normalized_tool_name, **self._normalize_context(context)},
        )

    def minimize_for_storage(self, storage_target: str, payload: Mapping[str, Any], *,
        purpose: str,
        request_id: str,
        subject_id: Optional[str] = None,
        record_id: Optional[str] = None,
        allowed_fields: Optional[Sequence[str]] = None,
        required_fields: Optional[Sequence[str]] = None,
        sensitive_fields: Optional[Sequence[str]] = None,
        field_strategies: Optional[Mapping[str, Any]] = None,
        source_context: Optional[str] = None,
        policy_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        audit_trail_ref: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_storage_target = self._normalize_identity(storage_target, "storage_target")
        return self.minimize_payload(
            payload,
            purpose=purpose,
            request_id=request_id,
            stage=f"{self.default_storage_context}.{normalized_storage_target}",
            subject_id=subject_id,
            record_id=record_id,
            allowed_fields=allowed_fields,
            required_fields=required_fields,
            sensitive_fields=sensitive_fields,
            field_strategies=field_strategies,
            reason=f"Least-data-required minimization before storage in '{normalized_storage_target}'.",
            source_context=source_context or "runtime",
            destination_context=normalized_storage_target,
            policy_id=policy_id,
            policy_version=policy_version,
            audit_trail_ref=audit_trail_ref,
            context={"storage_target": normalized_storage_target, **self._normalize_context(context)},
        )


if __name__ == "__main__":
    print("\n=== Running data Minimization===\n")
    printer.status("TEST", "data Minimization initialized", "info")

    minimizer = DataMinimization()

    sample_payload = {
        "ticket_id": "T-001",
        "email": "user@example.com",
        "phone": "+1-202-555-0141",
        "full_name": "Jane Customer",
        "issue_summary": "Cannot log in after password reset",
        "password": "super-secret-password",
        "session_token": "abc123-session-token",
        "metadata": {
            "ip_address": "203.0.113.5",
            "browser": "Firefox",
            "notes": "Customer provided billing zip and DOB in prior thread.",
        },
        "attachments": [
            {"file_name": "invoice.pdf", "content": "<binary-like-payload>", "content_type": "application/pdf"},
            {"file_name": "photo.png", "content": "<binary-like-payload>", "content_type": "image/png"},
        ],
    }

    evaluation = minimizer.evaluate_least_data_required(
        sample_payload,
        purpose="support_resolution",
        allowed_fields=[
            "ticket_id",
            "issue_summary",
            "email",
            "phone",
            "metadata.browser",
            "attachments.file_name",
            "attachments.content_type",
        ],
        required_fields=["ticket_id", "issue_summary"],
        sensitive_fields=["metadata.ip_address", "attachments.content"],
    )
    printer.status("TEST", "Least-data-required evaluation completed", "info")

    tool_result = minimizer.minimize_for_tool_call(
        "ticketing_connector",
        sample_payload,
        purpose="support_resolution",
        request_id="request-min-001",
        subject_id="subject-001",
        record_id="record-001",
        allowed_fields=[
            "ticket_id",
            "issue_summary",
            "email",
            "phone",
            "metadata.browser",
            "attachments.file_name",
            "attachments.content_type",
        ],
        required_fields=["ticket_id", "issue_summary"],
        sensitive_fields=["metadata.ip_address", "attachments.content"],
        field_strategies={
            "email": "partial_mask",
            "phone": "last4",
            "attachments.content": "drop",
        },
        source_context="chat_runtime",
        policy_id="minimization-policy-001",
        policy_version="2026.04",
        context={"tool_name": "ticketing_connector", "payload": sample_payload},
    )
    printer.status("TEST", "Tool-call payload minimization completed", "info")

    storage_result = minimizer.minimize_for_storage(
        "privacy_safe_archive",
        sample_payload,
        purpose="support_resolution",
        request_id="request-min-002",
        subject_id="subject-001",
        record_id="record-002",
        allowed_fields=["ticket_id", "issue_summary", "email", "metadata.browser"],
        required_fields=["ticket_id", "issue_summary"],
        field_strategies={"email": "tokenize", "full_name": "drop", "password": "drop"},
        source_context="chat_runtime",
        policy_id="minimization-policy-001",
        policy_version="2026.04",
        context={"storage_target": "privacy_safe_archive", "payload": sample_payload},
    )
    printer.status("TEST", "Storage payload minimization completed", "info")

    trace_tool = minimizer.memory.privacy_decision_trace("request-min-001")
    trace_storage = minimizer.memory.privacy_decision_trace("request-min-002")

    print(
        json.dumps(
            {
                "evaluation": evaluation,
                "tool_result": tool_result,
                "storage_result": storage_result,
                "trace_tool": trace_tool,
                "trace_storage": trace_storage,
                "memory_stats": minimizer.memory.stats(),
            },
            indent=2,
            default=str,
        )
    )

    print("\n=== Test ran successfully ===\n")