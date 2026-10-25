"""Context-sensitive fitness-for-use policy resolution for SLAI Quality.

Configuration source:
    src/agents/quality/configs/quality_config.yaml -> fitness_policy

The resolver owns quality-domain policy selection only. It does not execute
quality checks, perform workflow routing, tune agents, or mutate learned state.
QualityMemory may be supplied by dependency injection; this module never imports
or constructs it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..utils.config_loader import get_config_section, load_global_config
from ..utils.quality_error import *
from ..utils.quality_helpers import *
from logs.logger import get_logger, PrettyPrinter, get_log_queue # pyright: ignore[reportMissingImports]


logger = get_logger("Quality Fitness Policy")
printer = PrettyPrinter()

_EVIDENCE_RANK = {
    "insufficient": 0,
    "weak": 1,
    "moderate": 2,
    "strong": 3,
    "very_strong": 4,
}


def _config_error(message: str, *, context: Optional[Mapping[str, Any]] = None) -> DataQualityError:
    return DataQualityError(
        message=message,
        error_type=QualityErrorType.CONFIGURATION_INVALID,
        severity=QualitySeverity.HIGH,
        retryable=False,
        stage=QualityStage.VALIDATION,
        domain=QualityDomain.SYSTEM,
        disposition=QualityDisposition.ESCALATE,
        context=dict(context or {}),
        remediation="Correct the fitness_policy section in quality_config.yaml.",
    )


@dataclass(frozen=True, slots=True)
class QualityFitnessPolicy:
    """Immutable policy contract used for one quality evaluation."""

    policy_id: str
    use_case: str
    pass_threshold: float
    warn_threshold: float
    subsystem_weights: Dict[str, float]
    required_checks: Tuple[str, ...] = ()
    hard_block_error_types: Tuple[str, ...] = ()
    min_source_reliability: float = 0.0
    require_trusted_baseline: bool = False
    baseline_bootstrap_allowed: bool = False
    minimum_evidence_strength: str = "moderate"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        policy_id = nonempty_text(self.policy_id, "policy_id")
        use_case = nonempty_text(self.use_case, "use_case").lower()
        pass_threshold = bounded_score(self.pass_threshold, field_name="pass_threshold")
        warn_threshold = bounded_score(self.warn_threshold, field_name="warn_threshold")
        if warn_threshold > pass_threshold:
            raise _config_error(
                "Fitness policy warn_threshold must be <= pass_threshold",
                context={
                    "policy_id": policy_id,
                    "warn_threshold": warn_threshold,
                    "pass_threshold": pass_threshold,
                },
            )
        weights = normalize_weights(
            self.subsystem_weights,
            expected_keys=("structural", "statistical", "semantic"),
            field_name=f"fitness_policy.profiles.{policy_id}.subsystem_weights",
            reject_unknown=True,
        )
        evidence = str(self.minimum_evidence_strength).strip().lower()
        if evidence not in _EVIDENCE_RANK:
            raise _config_error(
                f"Unsupported minimum_evidence_strength '{self.minimum_evidence_strength}'",
                context={"policy_id": policy_id, "supported": sorted(_EVIDENCE_RANK)},
            )
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "use_case", use_case)
        object.__setattr__(self, "pass_threshold", pass_threshold)
        object.__setattr__(self, "warn_threshold", warn_threshold)
        object.__setattr__(self, "subsystem_weights", weights)
        object.__setattr__(self, "required_checks", tuple(string_list(self.required_checks, deduplicate=True)))
        object.__setattr__(
            self,
            "hard_block_error_types",
            tuple(string_list(self.hard_block_error_types, deduplicate=True)),
        )
        object.__setattr__(
            self,
            "min_source_reliability",
            bounded_score(self.min_source_reliability, field_name="min_source_reliability"),
        )
        object.__setattr__(self, "minimum_evidence_strength", evidence)
        object.__setattr__(self, "metadata", normalized_mapping(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["required_checks"] = list(self.required_checks)
        payload["hard_block_error_types"] = list(self.hard_block_error_types)
        return payload


@dataclass(frozen=True, slots=True)
class FitnessPolicyDecision:
    """Resolved policy plus the evidence used to select it."""

    policy: QualityFitnessPolicy
    source_id: Optional[str]
    source_reliability: Optional[float]
    source_reliability_satisfied: Optional[bool]
    selected_by: str
    rationale: Tuple[str, ...] = ()
    runtime_overrides_applied: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "source_id": self.source_id,
            "source_reliability": self.source_reliability,
            "source_reliability_satisfied": self.source_reliability_satisfied,
            "selected_by": self.selected_by,
            "rationale": list(self.rationale),
            "runtime_overrides_applied": list(self.runtime_overrides_applied),
        }


class FitnessPolicyResolver:
    """Resolve context-specific quality policies conservatively.

    Runtime overrides are disabled by default. When enabled, the default policy
    is monotonic strictness: thresholds and requirements can become stricter but
    cannot silently be relaxed unless configuration explicitly permits it.
    """

    def __init__(self, *, memory: Any = None, config: Optional[Mapping[str, Any]] = None) -> None:
        printer.status("INIT", "Initializing Fitness Policy Resolver", "info")
        global_config = load_global_config()
        section = get_config_section("fitness_policy", config=global_config)
        if config:
            section.update(dict(config))
        self.config = section
        self.memory = memory
        self.enabled = bool(section.get("enabled", True))
        self.default_profile = nonempty_text(
            section.get("default_profile", "general"),
            "fitness_policy.default_profile",
        )
        self.allow_runtime_overrides = bool(section.get("allow_runtime_overrides", False))
        self.allow_relaxation = bool(section.get("allow_runtime_relaxation", False))
        self.allow_weight_overrides = bool(section.get("allow_weight_overrides", False))
        self.use_case_profiles = {
            str(k).strip().lower(): str(v).strip()
            for k, v in dict(section.get("use_case_profiles", {})).items()
            if str(k).strip() and str(v).strip()
        }
        self.profiles = self._load_profiles(section.get("profiles", {}))
        if self.default_profile not in self.profiles:
            raise _config_error(
                "fitness_policy.default_profile does not identify a configured profile",
                context={
                    "default_profile": self.default_profile,
                    "available_profiles": sorted(self.profiles),
                },
            )

        logger.info(f"Succesfully initializing Fitness Policy Resolver with {self.profiles}")

    def resolve(
        self,
        *,
        use_case: Optional[str] = None,
        source_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> FitnessPolicyDecision:
        runtime_context = normalized_mapping(context)
        normalized_use_case = str(
            use_case
            or runtime_context.get("intended_use")
            or runtime_context.get("use_case")
            or "general"
        ).strip().lower() or "general"

        if policy_id is not None:
            selected_id = nonempty_text(policy_id, "policy_id")
            selected_by = "explicit_policy_id"
        else:
            selected_id = self.use_case_profiles.get(normalized_use_case, self.default_profile)
            selected_by = "use_case_mapping" if normalized_use_case in self.use_case_profiles else "default_profile"

        if selected_id not in self.profiles:
            raise _config_error(
                f"Unknown fitness policy profile '{selected_id}'",
                context={
                    "use_case": normalized_use_case,
                    "available_profiles": sorted(self.profiles),
                },
            )

        policy = self.profiles[selected_id]
        applied: Sequence[str] = ()
        if overrides:
            if not self.allow_runtime_overrides:
                raise _config_error("Runtime fitness-policy overrides are disabled", context={"policy_id": selected_id})
            policy, applied = self._apply_overrides(policy, overrides)

        reliability = self._resolve_source_reliability(source_id=source_id, context=runtime_context)
        reliability_ok = None if reliability is None else reliability >= policy.min_source_reliability

        rationale = [f"Selected policy '{policy.policy_id}' for use case '{normalized_use_case}'."]
        if reliability is None:
            rationale.append("No source-reliability evidence was available.")
        elif reliability_ok:
            rationale.append(f"Source reliability satisfies the policy floor ({reliability:.3f} >= {policy.min_source_reliability:.3f}).")
        else:
            rationale.append(f"Source reliability is below the policy floor ({reliability:.3f} < {policy.min_source_reliability:.3f}).")

        return FitnessPolicyDecision(
            policy=policy,
            source_id=None if source_id is None else str(source_id),
            source_reliability=reliability,
            source_reliability_satisfied=reliability_ok,
            selected_by=selected_by,
            rationale=tuple(rationale),
            runtime_overrides_applied=tuple(applied),
        )

    @staticmethod
    def should_hard_block(policy: QualityFitnessPolicy, findings: Sequence[Mapping[str, Any]]) -> bool:
        blockers = set(policy.hard_block_error_types)
        if not blockers:
            return False
        for finding in findings:
            error_type = finding.get("error_type")
            if hasattr(error_type, "value"):
                error_type = error_type.value
            if str(error_type or "").strip() in blockers:
                return True
        return False

    @staticmethod
    def evidence_requirement_satisfied(policy: QualityFitnessPolicy, evidence_strength: str) -> bool:
        observed = str(evidence_strength or "insufficient").strip().lower()
        return _EVIDENCE_RANK.get(observed, -1) >= _EVIDENCE_RANK[policy.minimum_evidence_strength]

    def _load_profiles(self, raw: Any) -> Dict[str, QualityFitnessPolicy]:
        if not isinstance(raw, Mapping) or not raw:
            raise _config_error("fitness_policy.profiles must be a non-empty mapping")
        profiles: Dict[str, QualityFitnessPolicy] = {}
        valid_error_types = {item.value for item in QualityErrorType}
        for profile_id, value in raw.items():
            if not isinstance(value, Mapping):
                raise _config_error(f"fitness_policy.profiles.{profile_id} must be a mapping")
            pid = str(profile_id).strip()
            hard_block = string_list(value.get("hard_block_error_types", []), deduplicate=True)
            unknown = sorted(set(hard_block) - valid_error_types)
            if unknown:
                raise _config_error(
                    f"Profile '{pid}' references unknown quality error types",
                    context={"unknown_error_types": unknown},
                )
            pass_threshold = value.get("pass_threshold")
            warn_threshold = value.get("warn_threshold")
            if isinstance(pass_threshold, bool) or not isinstance(pass_threshold, (int, float)):
                raise _config_error(
                    f"Profile '{pid}' pass_threshold must be a number",
                    context={"value": pass_threshold},
                )
            if isinstance(warn_threshold, bool) or not isinstance(warn_threshold, (int, float)):
                raise _config_error(
                    f"Profile '{pid}' warn_threshold must be a number",
                    context={"value": warn_threshold},
                )
            profiles[pid] = QualityFitnessPolicy(
                policy_id=pid,
                use_case=str(value.get("use_case", pid)),
                pass_threshold=float(pass_threshold),
                warn_threshold=float(warn_threshold),
                subsystem_weights=dict(value.get("subsystem_weights", {})),
                required_checks=tuple(value.get("required_checks", [])),
                hard_block_error_types=tuple(hard_block),
                min_source_reliability=value.get("min_source_reliability", 0.0),
                require_trusted_baseline=bool(value.get("require_trusted_baseline", False)),
                baseline_bootstrap_allowed=bool(value.get("baseline_bootstrap_allowed", False)),
                minimum_evidence_strength=str(value.get("minimum_evidence_strength", "moderate")),
                metadata=dict(value.get("metadata", {})),
            )
        return profiles

    def _resolve_source_reliability(
        self,
        *,
        source_id: Optional[str],
        context: Mapping[str, Any],
    ) -> Optional[float]:
        direct = context.get("source_reliability")
        if direct is not None:
            return bounded_score(direct, field_name="context.source_reliability")
        if self.memory is None or not source_id:
            return None

        method = getattr(self.memory, "latest_source_reliability", None)
        if callable(method):
            record = method(str(source_id))
            if isinstance(record, Mapping) and record.get("reliability") is not None:
                return bounded_score(record["reliability"], field_name="memory.source_reliability")

        method = getattr(self.memory, "latest_quality_state", None)
        if callable(method):
            record = method(str(source_id))
            if isinstance(record, Mapping) and record.get("source_reliability") is not None:
                return bounded_score(record["source_reliability"], field_name="memory.source_reliability")
        return None

    def _apply_overrides(
        self,
        policy: QualityFitnessPolicy,
        overrides: Mapping[str, Any],
    ) -> Tuple[QualityFitnessPolicy, Sequence[str]]:
        raw = dict(overrides)
        allowed = {
            "pass_threshold",
            "warn_threshold",
            "subsystem_weights",
            "required_checks",
            "hard_block_error_types",
            "min_source_reliability",
            "require_trusted_baseline",
            "baseline_bootstrap_allowed",
            "minimum_evidence_strength",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise _config_error("Unsupported runtime fitness-policy override keys", context={"unknown_keys": unknown})

        values = policy.to_dict()
        applied = []

        def tighten_number(name: str, new_value: float, old_value: float) -> float:
            if self.allow_relaxation or new_value >= old_value:
                applied.append(name)
                return new_value
            raise _config_error(
                f"Runtime override '{name}' would relax the active quality policy",
                context={"current": old_value, "requested": new_value},
            )

        if "pass_threshold" in raw:
            new = bounded_score(raw["pass_threshold"], field_name="override.pass_threshold")
            values["pass_threshold"] = tighten_number("pass_threshold", new, policy.pass_threshold)
        if "warn_threshold" in raw:
            new = bounded_score(raw["warn_threshold"], field_name="override.warn_threshold")
            values["warn_threshold"] = tighten_number("warn_threshold", new, policy.warn_threshold)
        if "min_source_reliability" in raw:
            new = bounded_score(raw["min_source_reliability"], field_name="override.min_source_reliability")
            values["min_source_reliability"] = tighten_number(
                "min_source_reliability", new, policy.min_source_reliability
            )

        if "required_checks" in raw:
            requested = string_list(raw["required_checks"], deduplicate=True)
            if not self.allow_relaxation:
                requested = merge_unique_strings(policy.required_checks, requested)
            values["required_checks"] = requested
            applied.append("required_checks")

        if "hard_block_error_types" in raw:
            requested = string_list(raw["hard_block_error_types"], deduplicate=True)
            if not self.allow_relaxation:
                requested = merge_unique_strings(policy.hard_block_error_types, requested)
            valid = {item.value for item in QualityErrorType}
            unknown_errors = sorted(set(requested) - valid)
            if unknown_errors:
                raise _config_error(
                    "Runtime override references unknown quality error types",
                    context={"unknown_error_types": unknown_errors},
                )
            values["hard_block_error_types"] = requested
            applied.append("hard_block_error_types")

        if "require_trusted_baseline" in raw:
            requested = bool(raw["require_trusted_baseline"])
            if policy.require_trusted_baseline and not requested and not self.allow_relaxation:
                raise _config_error("Runtime override would relax require_trusted_baseline")
            values["require_trusted_baseline"] = requested
            applied.append("require_trusted_baseline")

        if "baseline_bootstrap_allowed" in raw:
            requested = bool(raw["baseline_bootstrap_allowed"])
            if not policy.baseline_bootstrap_allowed and requested and not self.allow_relaxation:
                raise _config_error("Runtime override would relax baseline_bootstrap_allowed")
            values["baseline_bootstrap_allowed"] = requested
            applied.append("baseline_bootstrap_allowed")

        if "minimum_evidence_strength" in raw:
            requested = str(raw["minimum_evidence_strength"]).strip().lower()
            if requested not in _EVIDENCE_RANK:
                raise _config_error(f"Unsupported minimum_evidence_strength '{requested}'")
            if (
                not self.allow_relaxation
                and _EVIDENCE_RANK[requested] < _EVIDENCE_RANK[policy.minimum_evidence_strength]
            ):
                raise _config_error("Runtime override would relax minimum_evidence_strength")
            values["minimum_evidence_strength"] = requested
            applied.append("minimum_evidence_strength")

        if "subsystem_weights" in raw:
            if not self.allow_weight_overrides:
                raise _config_error("Runtime subsystem-weight overrides are disabled")
            values["subsystem_weights"] = normalize_weights(
                raw["subsystem_weights"],
                expected_keys=("structural", "statistical", "semantic"),
                field_name="override.subsystem_weights",
                reject_unknown=True,
            )
            applied.append("subsystem_weights")

        if float(values["warn_threshold"]) > float(values["pass_threshold"]):
            raise _config_error("Resolved runtime thresholds must satisfy warn_threshold <= pass_threshold")

        return QualityFitnessPolicy(
            policy_id=policy.policy_id,
            use_case=policy.use_case,
            pass_threshold=values["pass_threshold"],
            warn_threshold=values["warn_threshold"],
            subsystem_weights=values["subsystem_weights"],
            required_checks=tuple(values["required_checks"]),
            hard_block_error_types=tuple(values["hard_block_error_types"]),
            min_source_reliability=values["min_source_reliability"],
            require_trusted_baseline=values["require_trusted_baseline"],
            baseline_bootstrap_allowed=values["baseline_bootstrap_allowed"],
            minimum_evidence_strength=values["minimum_evidence_strength"],
            metadata=policy.metadata,
        ), tuple(applied)


__all__ = ["QualityFitnessPolicy", "FitnessPolicyDecision", "FitnessPolicyResolver"]


if __name__ == "__main__":
    print("\n=== Running Fitness Policy ===\n")
    printer.status("TEST", " Fitness Policy initialized", "info")

    policy = FitnessPolicyResolver()
    printer.pretty("FITNESS", policy, "success" if policy == "success" else "error") 
    printer.pretty("Resolve", policy.resolve, "success" if policy.resolve == "success" else "error") 

    print("\n=== Test ran successfully ===\n")