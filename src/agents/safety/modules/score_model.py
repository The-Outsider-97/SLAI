"""
Production-ready scoring model for the Safety Agent subsystem.

The ScoreModel converts text into normalized safety, privacy, alignment,
truthfulness, and helpfulness scores used by RewardModel and other safety
components. It intentionally stays focused on scoring primitives: it does not
own policy enforcement, refusal wording, secure memory persistence, human
review workflow, or model training. Those responsibilities remain in the
surrounding safety modules.
"""

from __future__ import annotations

import json
import re

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..utils.config_loader import load_global_config, get_config_section
from ..utils.security_error import *
from ..utils.safety_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Security Score Model")
printer = PrettyPrinter()

MODULE_VERSION = "2.1.0"
REPORT_SCHEMA_VERSION = "score_model.report.v2"


@dataclass(frozen=True)
class ScoreIndicator:
    """Audit-safe indicator generated during scoring."""

    component: str
    source: str
    indicator: str
    count: int
    weight: float = 1.0
    score_impact: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return redact_value(asdict(self), max_text_length=512)


@dataclass(frozen=True)
class ScoreReport:
    """Structured result returned by assess_text and safe for audit pipelines."""

    schema_version: str
    module_version: str
    text_fingerprint: str
    aggregate_score: float
    risk_score: float
    decision: str
    component_scores: Dict[str, float]
    component_weights: Dict[str, float]
    context_type: str
    indicators: List[ScoreIndicator] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_version": self.module_version,
            "text_fingerprint": self.text_fingerprint,
            "aggregate_score": clamp_score(self.aggregate_score),
            "risk_score": clamp_score(self.risk_score),
            "decision": self.decision,
            "component_scores": {key: clamp_score(value) for key, value in self.component_scores.items()},
            "component_weights": {key: float(value) for key, value in self.component_weights.items()},
            "context_type": self.context_type,
            "indicators": [indicator.to_dict() for indicator in self.indicators],
            "metadata": redact_value(self.metadata),
            "timestamp": self.timestamp,
        }


class ScoreModel:
    """
    Config-driven security and alignment scoring model.

    Scores are normalized to [0.0, 1.0], where 1.0 means lower detected risk or
    stronger positive signal for the requested component. The aggregate risk
    score returned by assess_text is inverted: risk_score = 1 - aggregate_score.
    """

    def __init__(self):
        self.config = load_global_config()
        self.score_config = get_config_section("score_model")
        self.guard_config = get_config_section("safety_guard")
        self.pii_patterns_path = self.score_config.get("pii_patterns_path") or self.guard_config.get("pii_patterns_path")

        self.enabled = bool(self.score_config.get("enabled", True))
        self.strict_config_validation = bool(self.score_config.get("strict_config_validation", True))
        self.max_text_length = int(self.score_config.get("max_text_length", 8192))
        self.unknown_component_score = clamp_score(self.score_config.get("unknown_component_score", 0.5), default=0.5)
        self.empty_text_score = clamp_score(self.score_config.get("empty_text_score", 0.0), default=0.0)
        self.default_context_type = str(self.score_config.get("default_context_type", "general"))
        self.component_order = list(self.score_config.get("component_order", []))
        self.component_weights = {
            str(name): float(weight)
            for name, weight in dict(self.score_config.get("component_weights", {})).items()
        }
        self.scoring_components: Dict[str, Dict[str, Any]] = {}
        self.context_policies: Dict[str, Dict[str, Any]] = dict(self.score_config.get("contexts", {}))
        self.context_handlers: Dict[str, Callable[[str], Optional[float]]] = {
            "user_creation": self._handle_user_creation,
            "financial": self._handle_financial_context,
            "medical": self._handle_medical_context,
        }

        self._compiled_term_patterns: Dict[str, List[Tuple[str, re.Pattern[str], float, str]]] = {}
        self._compiled_pattern_components: Dict[str, List[Tuple[str, re.Pattern[str], float]]] = {}
        self._last_indicators: List[ScoreIndicator] = []

        self._validate_config()
        self._load_scoring_components()

        logger.info(
            "Security Score Model initialized with components: %s",
            safe_log_payload(
                "score_model_initialized",
                {
                    "components": list(self.scoring_components.keys()),
                    "schema_version": self.score_config.get("schema_version"),
                    "enabled": self.enabled,
                },
            ),
        )

    # ------------------------------------------------------------------
    # Configuration and loading
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Validate the YAML-backed score_model section."""
    
        if not self.enabled:
            logger.warning("ScoreModel is disabled by configuration.")
    
        components = self.score_config.get("components")
        if components is None:
            self._raise_config_error("score_model.components section is missing.")
        if not isinstance(components, Mapping):
            self._raise_config_error("score_model.components must be a mapping (dictionary).")
        if not components:
            self._raise_config_error("score_model.components must be a non-empty mapping.")
    
        # If component_order is not configured or empty, use the keys of components
        configured_order = self.score_config.get("component_order")
        if configured_order is None or not isinstance(configured_order, list) or not configured_order:
            self.component_order = list(components.keys()) # type: ignore
        else:
            # Ensure configured_order is a list of strings
            self.component_order = [str(item) for item in configured_order if str(item).strip()]
    
        if not self.component_order:
            self._raise_config_error("score_model.component_order is empty after validation.")
    
        # Validate that every component in component_order exists
        missing = [name for name in self.component_order if name not in components] # type: ignore
        if missing:
            self._raise_config_error(f"score_model.component_order references missing components: {missing}")
    
        if not self.component_weights or not isinstance(self.component_weights, Mapping):
            self._raise_config_error("score_model.component_weights must be configured as a non-empty mapping in secure_config.yaml.")
    
        # Validate each component's configuration
        for name in self.component_order:
            comp = components.get(name) # type: ignore
            if not isinstance(comp, Mapping):
                self._raise_config_error(f"score_model.components.{name} must be a mapping.")
            scorer = str(comp.get("scorer", "")).lower() # type: ignore
            if scorer not in {"term", "pattern", "hybrid"}:
                self._raise_config_error(f"Unsupported scorer for component '{name}': {scorer}")
            mode = str(comp.get("score_mode", "")).lower() # type: ignore
            if mode not in {"positive", "penalty"}:
                self._raise_config_error(f"Unsupported score_mode for component '{name}': {mode}")

    def _raise_config_error(self, message: str) -> None:
        if self.strict_config_validation:
            raise ConfigurationTamperingError(
                "score_model",
                message,
                component="score_model",
                severity=SecuritySeverity.HIGH,
            )
        logger.warning("%s", message)

    def _resolve_config_path(self, raw_path: Optional[str]) -> Optional[Path]:
        if not raw_path:
            return None

        candidate = Path(str(raw_path)).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return candidate

        possible_paths = [candidate, Path.cwd() / candidate]
        config_path = self.config.get("__config_path__")
        if config_path:
            config_parent = Path(config_path).resolve().parent
            possible_paths.extend([
                config_parent / candidate,
                config_parent.parent / candidate,
                config_parent.parent.parent / candidate,
                config_parent.parent.parent.parent / candidate,
            ])

        possible_paths.append(Path("/mnt/data") / candidate.name)

        for path in possible_paths:
            if path.exists():
                return path

        return candidate

    def _load_scoring_components(self) -> None:
        """Initialize all scoring components from secure_config.yaml."""

        raw_components = dict(self.score_config.get("components", {}))
        for component in self.component_order:
            component_config = dict(raw_components[component])
            scorer = str(component_config.get("scorer")).lower()
            score_mode = str(component_config.get("score_mode")).lower()
            factor = max(float(component_config.get("factor", 1.0)), 1e-9)
            base_score = clamp_score(component_config.get("base_score", 0.0 if score_mode == "positive" else 1.0))
            max_score = clamp_score(component_config.get("max_score", 1.0))
            min_score = clamp_score(component_config.get("min_score", 0.0))
            terms = self._load_component_terms(component, component_config)
            phrases = self._load_component_phrases(component, component_config)
            patterns = self._load_component_patterns(component, component_config)

            self.scoring_components[component] = {
                "scorer": scorer,
                "score_mode": score_mode,
                "factor": factor,
                "base_score": base_score,
                "min_score": min_score,
                "max_score": max_score,
                "terms": terms,
                "phrases": phrases,
                "patterns": patterns,
                "case_sensitive": bool(component_config.get("case_sensitive", False)),
                "word_boundary": bool(component_config.get("word_boundary", True)),
                "term_weight": float(component_config.get("term_weight", 1.0)),
                "phrase_weight": float(component_config.get("phrase_weight", 1.5)),
                "pattern_weight": float(component_config.get("pattern_weight", 1.0)),
                "density_denominator": str(component_config.get("density_denominator", "words")),
                "description": str(component_config.get("description", "")),
            }

            if scorer in {"term", "hybrid"}:
                self._compiled_term_patterns[component] = self._compile_terms(component, self.scoring_components[component])
            if scorer in {"pattern", "hybrid"}:
                self._compiled_pattern_components[component] = self._compile_patterns(component, patterns)

    def _load_component_terms(self, component: str, component_config: Mapping[str, Any]) -> List[str]:
        terms = component_config.get("terms", [])
        terms_path = component_config.get("terms_path") or self.score_config.get(f"{component}_terms_path")
        if terms_path:
            loaded = self._load_terms_from_path(str(terms_path), f"{component}.terms_path")
            terms = list(terms or []) + loaded
        return dedupe_preserve_order([normalize_text(term, max_length=256) for term in terms if str(term).strip()])

    def _load_component_phrases(self, component: str, component_config: Mapping[str, Any]) -> List[str]:
        phrases = component_config.get("phrases", [])
        phrases_path = component_config.get("phrases_path") or self.score_config.get(f"{component}_phrases_path")
        if phrases_path:
            loaded = self._load_terms_from_path(str(phrases_path), f"{component}.phrases_path")
            phrases = list(phrases or []) + loaded
        return dedupe_preserve_order([normalize_text(phrase, max_length=512) for phrase in phrases if str(phrase).strip()])

    def _load_component_patterns(self, component: str, component_config: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        patterns: List[Any] = list(component_config.get("patterns", []) or [])
        pattern_path = component_config.get("patterns_path")
        if component == "privacy":
            pattern_path = pattern_path or self.pii_patterns_path

        if pattern_path:
            patterns.extend(self._load_patterns_from_path(str(pattern_path), f"{component}.patterns_path"))

        normalized: List[Mapping[str, Any]] = []
        for idx, item in enumerate(patterns):
            if isinstance(item, str):
                normalized.append({"name": f"{component}_pattern_{idx}", "pattern": item, "weight": component_config.get("pattern_weight", 1.0)})
            elif isinstance(item, Mapping) and item.get("pattern"):
                normalized.append({
                    "name": str(item.get("name") or item.get("replacement") or f"{component}_pattern_{idx}")[:128],
                    "pattern": str(item["pattern"]),
                    "weight": float(item.get("weight", component_config.get("pattern_weight", 1.0))),
                    "flags": item.get("flags", []),
                })
        return normalized

    def _load_terms_from_path(self, raw_path: str, source: str) -> List[str]:
        path = self._resolve_config_path(raw_path)
        if not path or not path.exists():
            self._raise_config_error(f"Configured terms file not found for {source}: {raw_path}")
            return []

        try:
            raw = load_text_file(path, max_bytes=int(self.score_config.get("max_terms_file_bytes", 1_048_576)))
            return [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith("#")]
        except SecurityError:
            raise
        except Exception as exc:
            raise SecurityError.from_exception(
                exc,
                error_type=SecurityErrorType.CONFIGURATION_TAMPERING,
                message=f"Failed to load configured score terms from {raw_path}.",
                component="score_model",
                severity=SecuritySeverity.HIGH,
                context={"source": source, "path": raw_path},
            )

    def _load_patterns_from_path(self, raw_path: str, source: str) -> List[Mapping[str, Any]]:
        path = self._resolve_config_path(raw_path)
        if not path or not path.exists():
            self._raise_config_error(f"Configured pattern file not found for {source}: {raw_path}")
            return []

        try:
            raw = load_text_file(path, max_bytes=int(self.score_config.get("max_patterns_file_bytes", 2_097_152)))
            loaded = json.loads(raw)
        except SecurityError:
            raise
        except Exception as exc:
            raise SecurityError.from_exception(
                exc,
                error_type=SecurityErrorType.CONFIGURATION_TAMPERING,
                message=f"Failed to load configured score patterns from {raw_path}.",
                component="score_model",
                severity=SecuritySeverity.HIGH,
                context={"source": source, "path": raw_path},
            )

        if isinstance(loaded, list):
            return [item if isinstance(item, Mapping) else {"pattern": str(item)} for item in loaded]
        if isinstance(loaded, Mapping) and isinstance(loaded.get("patterns"), list):
            return [item if isinstance(item, Mapping) else {"pattern": str(item)} for item in loaded["patterns"]]

        self._raise_config_error(f"Pattern file for {source} must contain a list or a mapping with a patterns list.")
        return []

    def _load_terms(self, config_key: str) -> List[str]:
        """
        Backward-compatible term loader.

        Older code passed keys such as 'harmful_terms_path'. The terms themselves
        are now owned by secure_config.yaml, either inline in score_model.components
        or via YAML-configured files.
        """

        legacy_component_map = {
            "harmful_terms_path": "alignment",
            "helpful_terms_path": "helpfulness",
            "risk_terms_path": "safety",
            "misinformation_terms_path": "truthfulness",
        }
        component = legacy_component_map.get(config_key)
        if component and component in self.scoring_components:
            return list(self.scoring_components[component].get("terms", []))

        path = self.score_config.get(config_key)
        return self._load_terms_from_path(path, config_key) if path else []

    def _load_pii_patterns(self) -> List[str]:
        """Backward-compatible PII regex loader."""

        if "privacy" in self.scoring_components:
            return [str(item["pattern"]) for item in self.scoring_components["privacy"].get("patterns", [])]
        patterns = self._load_patterns_from_path(str(self.pii_patterns_path), "privacy.patterns_path") if self.pii_patterns_path else []
        return [str(item["pattern"]) for item in patterns if item.get("pattern")]

    # ------------------------------------------------------------------
    # Pattern compilation and matching
    # ------------------------------------------------------------------

    def _compile_terms(self, component: str, config: Mapping[str, Any]) -> List[Tuple[str, re.Pattern[str], float, str]]:
        flags = 0 if config.get("case_sensitive") else re.IGNORECASE
        compiled: List[Tuple[str, re.Pattern[str], float, str]] = []

        for term in config.get("terms", []):
            escaped = re.escape(str(term))
            pattern = rf"\b{escaped}\b" if config.get("word_boundary", True) else escaped
            try:
                compiled.append((str(term), re.compile(pattern, flags), float(config.get("term_weight", 1.0)), "term"))
            except re.error as exc:
                self._handle_invalid_regex(component, str(term), exc)

        for phrase in config.get("phrases", []):
            escaped = re.escape(str(phrase))
            try:
                compiled.append((str(phrase), re.compile(escaped, flags), float(config.get("phrase_weight", 1.5)), "phrase"))
            except re.error as exc:
                self._handle_invalid_regex(component, str(phrase), exc)

        return compiled

    def _compile_patterns(self, component: str, patterns: Sequence[Mapping[str, Any]]) -> List[Tuple[str, re.Pattern[str], float]]:
        compiled: List[Tuple[str, re.Pattern[str], float]] = []
        for item in patterns:
            raw_pattern = str(item.get("pattern", ""))
            if not raw_pattern:
                continue

            flags = 0
            for flag_name in item.get("flags", []):
                flags |= getattr(re, str(flag_name).upper(), 0)

            try:
                compiled.append((str(item.get("name", "pattern"))[:128], re.compile(raw_pattern, flags), float(item.get("weight", 1.0))))
            except re.error as exc:
                self._handle_invalid_regex(component, raw_pattern, exc)
        return compiled

    def _handle_invalid_regex(self, component: str, pattern: str, exc: re.error) -> None:
        message = f"Invalid regex configured for score component '{component}': {exc}"
        if self.strict_config_validation:
            raise ConfigurationTamperingError(
                "score_model",
                message,
                component="score_model",
                cause=exc,
                context={"component": component, "pattern_fingerprint": fingerprint(pattern)},
            )
        logger.warning("%s", safe_log_payload("invalid_score_regex", {"component": component, "pattern": pattern, "error": str(exc)}))

    def _word_count(self, text: str) -> int:
        return max(1, len(re.findall(r"\b\w+\b", text)))

    def _component_denominator(self, text: str, config: Mapping[str, Any]) -> float:
        denominator_type = str(config.get("density_denominator", "words")).lower()
        if denominator_type == "characters":
            return max(1.0, float(len(text)))
        if denominator_type == "indicators":
            return max(1.0, float(len(config.get("terms", [])) + len(config.get("phrases", [])) + len(config.get("patterns", []))))
        return float(self._word_count(text))

    def _score_from_density(self, density: float, config: Mapping[str, Any]) -> float:
        mode = str(config.get("score_mode", "penalty")).lower()
        base_score = clamp_score(config.get("base_score", 1.0 if mode == "penalty" else 0.0))
        min_score = clamp_score(config.get("min_score", 0.0))
        max_score = clamp_score(config.get("max_score", 1.0))

        if mode == "positive":
            score = base_score + density
        else:
            score = base_score - density

        return max(min_score, min(max_score, clamp_score(score)))

    def _term_based_scorer(self, text: str, component: str) -> float:
        """Score a component using configured terms and phrases."""

        config = self.scoring_components[component]
        compiled = self._compiled_term_patterns.get(component, [])
        if not compiled:
            return clamp_score(config.get("base_score", self.unknown_component_score), default=self.unknown_component_score)

        weighted_matches = 0.0
        denominator = self._component_denominator(text, config)
        for label, pattern, weight, source in compiled:
            count = len(pattern.findall(text))
            if count:
                impact = min((count * weight) / max(float(config.get("factor", 1.0)), 1e-9), 1.0)
                self._last_indicators.append(
                    ScoreIndicator(component=component, source=source, indicator=label, count=count, weight=weight, score_impact=impact)
                )
                weighted_matches += count * weight

        density = min(weighted_matches / (denominator * max(float(config.get("factor", 1.0)), 1e-9)), 1.0)
        return self._score_from_density(density, config)

    def _pattern_based_scorer(self, text: str, component: str) -> float:
        """Score a component using configured regex patterns."""

        config = self.scoring_components[component]
        compiled = self._compiled_pattern_components.get(component, [])
        if not compiled:
            return clamp_score(config.get("base_score", self.unknown_component_score), default=self.unknown_component_score)

        weighted_matches = 0.0
        for label, pattern, weight in compiled:
            matches = pattern.findall(text)
            count = len(matches)
            if count:
                impact = min((count * weight * float(config.get("factor", 1.0))), 1.0)
                self._last_indicators.append(
                    ScoreIndicator(component=component, source="pattern", indicator=label, count=count, weight=weight, score_impact=impact)
                )
                weighted_matches += count * weight

        density = min(weighted_matches * float(config.get("factor", 1.0)), 1.0)
        return self._score_from_density(density, config)

    def _hybrid_scorer(self, text: str, component: str) -> float:
        config = self.scoring_components[component]
        term_score = self._term_based_scorer(text, component)
        pattern_score = self._pattern_based_scorer(text, component)
        term_weight = float(config.get("hybrid_term_weight", 0.5))
        pattern_weight = float(config.get("hybrid_pattern_weight", 0.5))
        return weighted_average({"term": term_score, "pattern": pattern_score}, {"term": term_weight, "pattern": pattern_weight}, default=self.unknown_component_score)

    # ------------------------------------------------------------------
    # Context handling
    # ------------------------------------------------------------------
    def _context_matches(self, text: str, policy: Mapping[str, Any]) -> bool:
        trigger_terms = [str(term).lower() for term in policy.get("trigger_terms", [])]
        if not trigger_terms:
            return True
        lowered = text.lower()
        return any(term in lowered for term in trigger_terms)

    def _apply_context_policy(self, text: str, component: str, score: float, context: Optional[Mapping[str, Any]]) -> float:
        context_type = self.default_context_type
        if context:
            context_type = str(context.get("type", context.get("context_type", self.default_context_type)))

        policy = self.context_policies.get(context_type)
        if not isinstance(policy, Mapping) or not self._context_matches(text, policy):
            return score

        multipliers = dict(policy.get("component_multipliers", {}))
        floors = dict(policy.get("component_floors", {}))
        ceilings = dict(policy.get("component_ceilings", {}))

        adjusted = score
        if component in multipliers:
            adjusted = adjusted * float(multipliers[component])
        if component in floors:
            adjusted = max(adjusted, float(floors[component]))
        if component in ceilings:
            adjusted = min(adjusted, float(ceilings[component]))

        return clamp_score(adjusted)

    def _handle_user_creation(self, text: str) -> Optional[float]:
        """Special handling for user creation context kept for RewardModel compatibility."""

        policy = self.context_policies.get("user_creation", {})
        if isinstance(policy, Mapping) and self._context_matches(text, policy):
            base = policy.get("override_score")
            return clamp_score(base) if base is not None else None
        return None

    def _handle_financial_context(self, text: str) -> Optional[float]:
        """Special handling for financial context kept for compatibility."""

        policy = self.context_policies.get("financial", {})
        if isinstance(policy, Mapping) and self._context_matches(text, policy):
            privacy_score = self.scoring_components["privacy"]["scorer"](text, "privacy")
            multiplier = float(dict(policy.get("component_multipliers", {})).get("privacy", 1.0))
            return clamp_score(privacy_score * multiplier)
        return None

    def _handle_medical_context(self, text: str) -> Optional[float]:
        """Special handling for medical context kept for compatibility."""

        policy = self.context_policies.get("medical", {})
        if isinstance(policy, Mapping) and self._context_matches(text, policy):
            privacy_score = self.scoring_components["privacy"]["scorer"](text, "privacy")
            multiplier = float(dict(policy.get("component_multipliers", {})).get("privacy", 1.0))
            return clamp_score(privacy_score * multiplier)
        return None

    # ------------------------------------------------------------------
    # Public scoring API
    # ------------------------------------------------------------------
    def normalize_input_text(self, text: Any) -> str:
        normalized = normalize_text(text, max_length=self.max_text_length, preserve_newlines=True)
        if not normalized:
            return ""
        return normalized

    def calculate_score(self, text: str, component: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate a normalized score for a specific component with context awareness."""

        if component not in self.scoring_components:
            raise SecurityError(
                SecurityErrorType.CONFIGURATION_TAMPERING,
                f"Unknown score component requested: {component}",
                severity=SecuritySeverity.HIGH,
                component="score_model",
                context={"component": component, "available_components": list(self.scoring_components.keys())},
                remediation_guidance=(
                    "Ensure callers only request components configured under score_model.components.",
                    "Update secure_config.yaml if a new scoring component is intentionally introduced.",
                ),
            )

        normalized = self.normalize_input_text(text)
        if not normalized:
            return clamp_score(self.scoring_components[component].get("empty_text_score", self.empty_text_score), default=self.empty_text_score)

        self._last_indicators = []
        scorer_name = str(self.scoring_components[component]["scorer"]).lower()
        try:
            if scorer_name == "term":
                raw_score = self._term_based_scorer(normalized, component)
            elif scorer_name == "pattern":
                raw_score = self._pattern_based_scorer(normalized, component)
            elif scorer_name == "hybrid":
                raw_score = self._hybrid_scorer(normalized, component)
            else:
                raise ConfigurationTamperingError("score_model", f"Unsupported scorer: {scorer_name}", component="score_model")
        except SecurityError:
            raise
        except Exception as exc:
            raise SecurityError.from_exception(
                exc,
                error_type=SecurityErrorType.UNSAFE_MODEL_STATE,
                message=f"Score calculation failed for component '{component}'.",
                component="score_model",
                context={"component": component, "text_fingerprint": fingerprint(normalized)},
            )

        return self._apply_context_policy(normalized, component, clamp_score(raw_score, default=self.unknown_component_score), context)

    def calculate_all_scores(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate all configured component scores."""

        scores: Dict[str, float] = {}
        all_indicators: List[ScoreIndicator] = []
        for component in self.component_order:
            score = self.calculate_score(text, component, context=context)
            scores[component] = clamp_score(score)
            all_indicators.extend(self._last_indicators)
        self._last_indicators = all_indicators
        return scores

    def aggregate_scores(self, scores: Mapping[str, float], weights: Optional[Mapping[str, float]] = None) -> float:
        """Aggregate component scores with configured weights."""

        effective_weights = dict(weights or self.component_weights)
        return clamp_score(weighted_average(scores, effective_weights, default=self.unknown_component_score))

    def decision_for_score(self, aggregate_score: float) -> str:
        thresholds = dict(self.score_config.get("decision_thresholds", {}))
        block_threshold = float(thresholds.get("block_below", 0.35))
        review_threshold = float(thresholds.get("review_below", 0.65))
        score = clamp_score(aggregate_score)
        if score <= block_threshold:
            return "block"
        if score <= review_threshold:
            return "review"
        return "allow"

    def assess_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> ScoreReport:
        """Return a structured aggregate score report."""

        normalized = self.normalize_input_text(text)
        scores = self.calculate_all_scores(normalized, context=context)
        aggregate = self.aggregate_scores(scores)
        report = ScoreReport(
            schema_version=REPORT_SCHEMA_VERSION,
            module_version=MODULE_VERSION,
            text_fingerprint=fingerprint(normalized),
            aggregate_score=aggregate,
            risk_score=clamp_score(1.0 - aggregate),
            decision=self.decision_for_score(aggregate),
            component_scores=scores,
            component_weights=dict(self.component_weights),
            context_type=str((context or {}).get("type", (context or {}).get("context_type", self.default_context_type))),
            indicators=list(self._last_indicators),
            metadata={
                "component_order": list(self.component_order),
                "indicator_count": len(self._last_indicators),
                "text_length": len(normalized),
            },
        )
        logger.info("Score assessment completed: %s", safe_log_payload("score_assessment", report.to_dict()))
        return report

    def to_feature_vector(self, text: str, context: Optional[Dict[str, Any]] = None, order: Optional[Sequence[str]] = None) -> List[float]:
        """Return component scores in a deterministic order for downstream models."""

        scores = self.calculate_all_scores(text, context=context)
        return [clamp_score(scores.get(component, self.unknown_component_score)) for component in (order or self.component_order)]

    def get_component_schema(self) -> Dict[str, Any]:
        """Return an audit-safe description of configured scoring components."""

        schema: Dict[str, Any] = {}
        for component, config in self.scoring_components.items():
            schema[component] = {
                "scorer": config["scorer"],
                "score_mode": config["score_mode"],
                "factor": config["factor"],
                "term_count": len(config.get("terms", [])),
                "phrase_count": len(config.get("phrases", [])),
                "pattern_count": len(config.get("patterns", [])),
                "description": config.get("description", ""),
                "weight": self.component_weights.get(component, 0.0),
            }
        return redact_value(schema)

    # Public scoring methods maintain RewardModel compatibility =================

    def _alignment_score(self, text: str) -> float:
        """Return an alignment score where higher is safer/more aligned."""
        return self.calculate_score(text, "alignment")

    def _helpfulness_score(self, text: str) -> float:
        """Return a helpfulness score where higher means stronger assistive signal."""
        return self.calculate_score(text, "helpfulness")

    def _privacy_score(self, text: str) -> float:
        """Return a privacy score where higher means fewer PII/secret indicators."""
        return self.calculate_score(text, "privacy")

    def _safety_score(self, text: str) -> float:
        """Return a safety score where higher means lower operational/user-safety risk."""
        return self.calculate_score(text, "safety")

    def _truthfulness_score(self, text: str) -> float:
        """Return a truthfulness score where higher means fewer misinformation indicators."""
        return self.calculate_score(text, "truthfulness")


if __name__ == "__main__":
    print("\n=== Running Score Model ===\n")
    printer.status("TEST", "Score Model initialized", "info")

    model = ScoreModel()

    safe_text = (
        "I can help guide you through safe account recovery steps. "
        "Use official support pages and avoid sharing passwords or secret tokens."
    )
    risky_text = (
        "Urgent: send your password, SSN 123-45-6789, and card number "
        "4111 1111 1111 1111 to verify your bank account. This fake cure is guaranteed."
    )

    safe_report = model.assess_text(safe_text, context={"type": "general", "request_id": "score-test-safe"})
    risky_report = model.assess_text(risky_text, context={"type": "financial", "request_id": "score-test-risky"})

    assert 0.0 <= safe_report.aggregate_score <= 1.0
    assert 0.0 <= risky_report.aggregate_score <= 1.0
    assert risky_report.component_scores["privacy"] < safe_report.component_scores["privacy"]
    assert risky_report.risk_score > safe_report.risk_score
    assert model._alignment_score(safe_text) >= model._alignment_score(risky_text)
    assert model._helpfulness_score(safe_text) >= 0.0
    assert len(model.to_feature_vector(risky_text)) == len(model.component_order)

    schema = model.get_component_schema()
    assert "privacy" in schema and schema["privacy"]["pattern_count"] > 0

    try:
        model.calculate_score("test", "unknown_component")
        raise AssertionError("Unknown component did not raise SecurityError")
    except SecurityError:
        pass

    serialized_report = stable_json(risky_report.to_dict())
    assert "123-45-6789" not in serialized_report
    assert "4111 1111 1111 1111" not in serialized_report

    printer.status("TEST", f"Safe decision: {safe_report.decision}", "info")
    printer.status("TEST", f"Risky decision: {risky_report.decision}", "info")
    printer.status("TEST", f"Components: {list(schema.keys())}", "info")

    print("\n=== Test ran successfully ===\n")
