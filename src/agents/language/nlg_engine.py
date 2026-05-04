"""
Natural Language Generation Engine

Core Function:
Performs Natural Language Generation: produces fluent, context-aware textual
responses from a LinguisticFrame and optional DialogueContext/LanguageMemory
state.

Responsibilities:
- Map intents, entities, slots, dialogue context, and speech-act metadata to a
  verbal response.
- Load, validate, rank, and render response templates from config-backed
  resources with safe placeholder handling.
- Adapt tone, formality, verbosity, confidence markers, modality markers, and
  sentiment framing without corrupting the generated content.
- Provide deterministic fallback behavior when templates, entities, or optional
  neural generation are unavailable.
- Preserve compatibility with the current LanguageAgent by keeping
  generate(frame, context) -> str while exposing generate_detailed() for richer
  pipeline integration.

Why it matters:
Even with strong understanding, a language agent must communicate naturally,
consistently, and safely. NLG is the final language-facing step before the agent
responds, so it must be deterministic, inspectable, configurable, extensible,
and resilient to partial upstream pipeline output.
"""

from __future__ import annotations

import json
import random
import re
import time as time_module
import yaml


from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple, Union

from .utils.config_loader import load_global_config, get_config_section
from .utils.linguistic_frame import LinguisticFrame, SpeechActType
from .utils.language_error import *
from .utils.language_helpers import *
from .language_memory import LanguageMemory
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("NLG Engine")
printer = PrettyPrinter()

TemplateMap = Dict[str, "NLGTemplateSet"]
EntityMap = Dict[str, Any]
RenderPayload = Dict[str, Any]
NeuralGenerator = Callable[[str, LinguisticFrame, Mapping[str, Any]], str]


# ---------------------------------------------------------------------------
# Small compatibility helpers
# ---------------------------------------------------------------------------
# The language subsystem already has shared helper utilities. These local
# wrappers prefer those helpers when present and only provide tiny fallback
# behavior so this module remains runnable during isolated module tests.
def _helper(name: str) -> Optional[Callable[..., Any]]:
    candidate = globals().get(name)
    return candidate if callable(candidate) else None


def _text(value: Any, default: str = "") -> str:
    fn = _helper("ensure_text")
    if fn:
        return fn(value)  # type: ignore[misc]
    if value is None:
        return default
    return str(value)


def _bool(value: Any, default: bool = False) -> bool:
    fn = _helper("coerce_bool")
    if fn:
        return bool(fn(value, default=default))  # type: ignore[misc]
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _int(value: Any, default: int = 0, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    fn = _helper("coerce_int")
    if fn:
        return int(fn(value, default=default, minimum=minimum, maximum=maximum))  # type: ignore[misc]
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = int(default)
    if minimum is not None:
        resolved = max(resolved, minimum)
    if maximum is not None:
        resolved = min(resolved, maximum)
    return resolved


def _float(value: Any, default: float = 0.0, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    fn = _helper("coerce_float")
    if fn:
        return float(fn(value, default=default, minimum=minimum, maximum=maximum))  # type: ignore[misc]
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = float(default)
    if minimum is not None:
        resolved = max(resolved, minimum)
    if maximum is not None:
        resolved = min(resolved, maximum)
    return resolved


def _list(value: Any) -> List[Any]:
    fn = _helper("ensure_list")
    if fn:
        return list(fn(value))  # type: ignore[misc]
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)
    return [value]


def _mapping(value: Any) -> Dict[str, Any]:
    fn = _helper("ensure_mapping")
    if fn:
        try:
            return dict(fn(value, field_name="nlg_mapping", allow_none=True))  # type: ignore[misc]
        except TypeError:
            return dict(fn(value))  # type: ignore[misc]
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _json_safe(value: Any) -> Any:
    for name in ("json_safe", "to_json_safe"):
        fn = _helper(name)
        if fn:
            return fn(value)  # type: ignore[misc]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if hasattr(value, "value") and not isinstance(value, (str, bytes, bytearray)):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _clean_space(text: str) -> str:
    fn = _helper("normalize_spacing_around_punctuation")
    if fn:
        try:
            return fn(text)  # type: ignore[misc]
        except Exception:
            pass
    text = re.sub(r"\s+", " ", _text(text)).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text


def _truncate(text: str, limit: int) -> str:
    fn = _helper("truncate_text")
    if fn:
        try:
            return fn(text, limit)  # type: ignore[misc]
        except Exception:
            pass
    value = _text(text)
    return value if len(value) <= limit else value[: max(0, limit - 3)].rstrip() + "..."


def _frame_to_dict(frame: Optional[LinguisticFrame]) -> Dict[str, Any]:
    fn = _helper("frame_to_dict")
    if fn and frame is not None:
        try:
            return dict(fn(frame))  # type: ignore[misc]
        except Exception:
            pass
    if frame is None:
        return {}
    return {
        "intent": frame.intent,
        "entities": _json_safe(frame.entities),
        "sentiment": frame.sentiment,
        "modality": frame.modality,
        "confidence": frame.confidence,
        "act_type": frame.act_type.value if getattr(frame, "act_type", None) else None,
        "propositional_content": getattr(frame, "propositional_content", None),
        "illocutionary_force": getattr(frame, "illocutionary_force", None),
        "perlocutionary_effect": getattr(frame, "perlocutionary_effect", None),
    }


def _utc_timestamp() -> str:
    fn = _helper("utc_timestamp") or _helper("utc_now_iso")
    if fn:
        try:
            return _text(fn())
        except Exception:
            pass
    return time_module.strftime("%Y-%m-%dT%H:%M:%SZ", time_module.gmtime())


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NLGTemplate:
    """Single renderable response template."""

    template_id: str
    text: str
    intent: str = "default"
    priority: int = 0
    weight: float = 1.0
    required_entities: Tuple[str, ...] = ()
    forbidden_entities: Tuple[str, ...] = ()
    conditions: Tuple[str, ...] = ()
    style_tags: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def placeholders(self) -> Tuple[str, ...]:
        return tuple(extract_template_fields(self.text))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "intent": self.intent,
            "text": self.text,
            "priority": self.priority,
            "weight": self.weight,
            "required_entities": list(self.required_entities),
            "forbidden_entities": list(self.forbidden_entities),
            "conditions": list(self.conditions),
            "style_tags": list(self.style_tags),
            "placeholders": list(self.placeholders),
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class NLGTemplateSet:
    """Templates and trigger metadata for one intent."""

    intent: str
    responses: Tuple[NLGTemplate, ...]
    triggers: Tuple[str, ...] = ()
    defaults: Dict[str, Any] = field(default_factory=dict)
    required_entities: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "responses": [item.to_dict() for item in self.responses],
            "triggers": list(self.triggers),
            "defaults": _json_safe(self.defaults),
            "required_entities": list(self.required_entities),
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class NLGContextPacket:
    """Normalized context supplied to the generator."""

    summary: str = ""
    history: Tuple[Dict[str, Any], ...] = ()
    relevant_context: str = ""
    slots: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    unresolved_issues: Tuple[Dict[str, Any], ...] = ()
    memory_matches: Tuple[Dict[str, Any], ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "history": list(self.history),
            "relevant_context": self.relevant_context,
            "slots": _json_safe(self.slots),
            "environment": _json_safe(self.environment),
            "unresolved_issues": list(self.unresolved_issues),
            "memory_matches": list(self.memory_matches),
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class NLGRenderAttempt:
    """Template rendering attempt for traceability."""

    template_id: str
    success: bool
    missing_fields: Tuple[str, ...] = ()
    error: Optional[str] = None
    rendered_preview: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "success": self.success,
            "missing_fields": list(self.missing_fields),
            "error": self.error,
            "rendered_preview": self.rendered_preview,
        }


@dataclass(frozen=True)
class NLGGenerationResult:
    """Structured NLG result for production pipeline integration."""

    text: str
    intent: str
    frame: Optional[LinguisticFrame] = None
    template_id: Optional[str] = None
    fallback_used: bool = False
    generation_mode: str = "template"
    confidence: float = 1.0
    attempts: Tuple[NLGRenderAttempt, ...] = ()
    context: Optional[NLGContextPacket] = None
    issues: Tuple[Any, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.text and self.text.strip()) and not any(getattr(issue, "is_blocking", False) for issue in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "text": self.text,
            "intent": self.intent,
            "frame": _frame_to_dict(self.frame),
            "template_id": self.template_id,
            "fallback_used": self.fallback_used,
            "generation_mode": self.generation_mode,
            "confidence": self.confidence,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "context": self.context.to_dict() if self.context else None,
            "issues": [issue.to_dict() if hasattr(issue, "to_dict") else str(issue) for issue in self.issues],
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class NLGEngineStats:
    """Operational snapshot for observability."""

    version: str
    template_count: int
    intent_count: int
    generation_count: int
    fallback_count: int
    failed_generation_count: int
    validation_failure_count: int
    history_length: int
    diagnostics_count: int
    templates_path: Optional[str] = None
    generation_mode: str = "template"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Template field handling
# ---------------------------------------------------------------------------
_FIELD_PATTERN = re.compile(r"\{([^{}]+)\}")
_DEFAULT_FIELD_PATTERN = re.compile(r"^\s*([A-Za-z_][\w.:-]*)\s*\|\s*(.+?)\s*$")


def extract_template_fields(template: str) -> List[str]:
    fields: List[str] = []
    for _, field_name, _, _ in Formatter().parse(template):
        if not field_name:
            continue
        fields.append(parse_placeholder_spec(field_name)[0])
    # Formatter cannot parse custom placeholders containing `|` reliably when
    # format specs are absent, so also scan raw braces.
    for raw in _FIELD_PATTERN.findall(template):
        fields.append(parse_placeholder_spec(raw)[0])
    return sorted({field for field in fields if field})


def parse_placeholder_spec(raw_field: str) -> Tuple[str, Optional[str]]:
    raw = _text(raw_field).strip()
    match = _DEFAULT_FIELD_PATTERN.match(raw)
    if not match:
        return raw.split(".", 1)[0] if raw else raw, None
    field_name = match.group(1).strip()
    default = match.group(2).strip()
    if (default.startswith("'") and default.endswith("'")) or (default.startswith('"') and default.endswith('"')):
        default = default[1:-1]
    return field_name, default


def resolve_dotted_value(payload: Mapping[str, Any], field_name: str, default: Any = "") -> Any:
    current: Any = payload
    for part in _text(field_name).split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return default
    return current


def render_template_safely(template: str, payload: Mapping[str, Any]) -> Tuple[str, List[str]]:
    missing: List[str] = []

    def replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        field_name, fallback = parse_placeholder_spec(raw)
        value = resolve_dotted_value(payload, field_name, None)
        if value is None or value == "":
            if fallback is not None:
                return _text(fallback)
            missing.append(field_name)
            return ""
        if isinstance(value, (list, tuple, set)):
            return ", ".join(_text(item) for item in value if item is not None)
        if isinstance(value, Mapping):
            return ", ".join(f"{k}: {v}" for k, v in value.items())
        return _text(value)

    rendered = _FIELD_PATTERN.sub(replace, template)
    return _clean_space(rendered), sorted(set(missing))


# ---------------------------------------------------------------------------
# Main NLG engine
# ---------------------------------------------------------------------------
class NLGEngine:
    """Production NLG engine for the language agent subsystem."""

    VERSION = "2.0"
    DEFAULT_FALLBACK_RESPONSES: Tuple[str, ...] = (
        "I'm not sure how to respond to that right now.",
        "I need a little more information before I can answer clearly.",
        "Could you clarify what you would like me to do?",
    )

    DEFAULT_VERBOSE_PHRASES: Dict[str, List[str]] = {
        "default": ["For additional context:", "To elaborate further:", "In more detail:"],
        "technical": ["From a technical perspective:", "More precisely:", "Implementation-wise:"],
    }

    DEFAULT_MODALITY_MARKERS: Dict[str, List[str]] = {
        "epistemic": ["I think", "It seems"],
        "deontic": ["You should", "It would be best to"],
        "interrogative": ["Could you", "Would you"],
        "error": ["I ran into an issue:"],
    }

    FORMAL_CONTRACTIONS: Dict[str, str] = {
        "can't": "cannot",
        "don't": "do not",
        "won't": "will not",
        "isn't": "is not",
        "aren't": "are not",
        "I'm": "I am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are",
        "let's": "let us",
        "shouldn't": "should not",
        "couldn't": "could not",
        "wouldn't": "would not",
        "didn't": "did not",
        "haven't": "have not",
        "hasn't": "has not",
    }

    INFORMAL_CONTRACTIONS: Dict[str, str] = {
        "cannot": "can't",
        "do not": "don't",
        "will not": "won't",
        "is not": "isn't",
        "are not": "aren't",
        "I am": "I'm",
        "you are": "you're",
        "he is": "he's",
        "she is": "she's",
        "it is": "it's",
        "we are": "we're",
        "they are": "they're",
        "let us": "let's",
    }

    def __init__(self) -> None:
        self.config = load_global_config()
        self.nlg_config = get_config_section("nlg") or {}
        self.neural_config = get_config_section("neural_generation") or {}

        self.version = _text(self.nlg_config.get("version", self.VERSION))
        self.generation_mode = _text(self.nlg_config.get("generation_mode", "template")).lower()
        self.strict_loading = _bool(self.nlg_config.get("strict_loading", False), default=False)
        self.record_history = _bool(self.nlg_config.get("record_history", True), default=True)
        self.history_limit = _int(self.nlg_config.get("history_limit", 200), default=200, minimum=1)
        self.random_seed = self.nlg_config.get("random_seed")
        self.rng = random.Random(self.random_seed)

        self._templates_path = self.nlg_config.get("templates_path")
        self.verbose_path = self.nlg_config.get("verbose_phrases")
        self.modality_markers_path = self.nlg_config.get("modality_markers_path", self.config.get("modality_markers_path"))
        self.coherence_checker = self.nlg_config.get("coherence_checker")

        self.style = self._resolve_style_config(self.nlg_config.get("style", {}))
        self.max_retries = _int(self.neural_config.get("max_retries", 2), default=2, minimum=0)
        self.fallback_after_retries = _bool(self.neural_config.get("fallback_after_retries", True), default=True)
        self.max_tokens = _int(self.neural_config.get("max_tokens", 512), default=512, minimum=1)
        self.temperature = _float(self.neural_config.get("temperature", 0.7), default=0.7, minimum=0.0)

        validation_config = _mapping(self.nlg_config.get("validation", {}))
        self.min_words = _int(validation_config.get("min_words", self.neural_config.get("min_words", 1)), default=1, minimum=0)
        self.max_words = _int(validation_config.get("max_words", self.neural_config.get("max_words", 160)), default=160, minimum=1)
        self.max_characters = _int(validation_config.get("max_characters", 2000), default=2000, minimum=32)
        self.blocked_response_patterns = [re.compile(_text(pattern), re.IGNORECASE) for pattern in _list(validation_config.get("blocked_patterns", []))]

        self.templates: TemplateMap = self._load_templates(self.templates_path)
        self.verbose_phrases = self._load_verbose_phrases(self.verbose_path)
        self.modality_markers = self._load_modality_markers(self.modality_markers_path)
        self.memory: Optional[LanguageMemory] = LanguageMemory() if _bool(self.nlg_config.get("enable_memory", False), default=False) else None
        self.neural_generator: Optional[NeuralGenerator] = None

        self.generation_history: Deque[Dict[str, Any]] = deque(maxlen=self.history_limit)
        self.diagnostics: List[Any] = []
        self.generation_count = 0
        self.fallback_count = 0
        self.failed_generation_count = 0
        self.validation_failure_count = 0
        self.intent_counts: Counter[str] = Counter()

        logger.info("NLG Engine initialized with %s intents and %s templates", len(self.templates), self._template_count())
        printer.status("INIT", f"NLG Engine initialized with {len(self.templates)} intents", "success")

    # ------------------------------------------------------------------
    # Config/resource loading
    # ------------------------------------------------------------------
    @property
    def templates_path(self) -> Any:
        return self._templates_path

    @templates_path.setter
    def templates_path(self, value: Any) -> None:
        self._templates_path = value

    def _resolve_style_config(self, raw_style: Any) -> Dict[str, Any]:
        style = {
            "formality": 0.5,
            "verbosity": 1.0,
            "verbosity_mode": "default",
            "truncation_length": 25,
            "allow_slang": False,
            "max_contractions": 3,
            "informal_phrases": [],
            "verbose_phrases": [],
        }
        if isinstance(raw_style, Mapping):
            style.update(dict(raw_style))
        # Backward compatibility for the old accidental set default: ignore it
        # and keep a real mapping.
        return style

    def _resolve_path(self, value: Any) -> Optional[Path]:
        if value in (None, "", "none", "None"):
            return None
        fn = _helper("resolve_path")
        if fn:
            try:
                return Path(fn(_text(value), field_name="nlg_path"))  # type: ignore[misc]
            except TypeError:
                return Path(fn(_text(value)))  # type: ignore[misc]
        return Path(_text(value))

    def _load_json_or_yaml(self, path: Path) -> Any:
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() in {".yaml", ".yml"}:
                return yaml.safe_load(handle) or {}
            return json.load(handle)

    def _load_templates(self, path: Any) -> TemplateMap:
        fallback = self._inline_or_default_template_map()
        template_path = self._resolve_path(path)
        if template_path is None:
            self._add_diagnostic("template_path_missing", "No NLG template path configured.", severity="warning")
            return fallback
        if not template_path.exists():
            self._add_diagnostic("template_file_missing", "Configured NLG template file was not found.", severity="warning", path=str(template_path))
            if self.strict_loading:
                raise TemplateNotFoundError("Template file missing.", intent="default", templates=fallback)
            return fallback
    
        try:
            raw_templates = self._load_json_or_yaml(template_path)
        except Exception as exc:
            self._add_diagnostic("template_load_failed", "Failed to load NLG templates.", severity="error", path=str(template_path), error=str(exc))
            if self.strict_loading:
                raise NLGGenerationError("Failed to load templates.", error_type="template_load", original_exception=exc)
            return fallback
    
        # Handle rich YAML structure with metadata (schema, version, locale, templates)
        if isinstance(raw_templates, Mapping) and "templates" in raw_templates:
            raw_templates = raw_templates["templates"]
    
        if not isinstance(raw_templates, Mapping):
            self._add_diagnostic("template_schema_invalid", "NLG templates must be a mapping of intent to template data.", severity="error", path=str(template_path))
            if self.strict_loading:
                raise TemplateNotFoundError("Template schema invalid.", intent="default", templates=fallback)
            return fallback
    
        templates = self._normalize_template_mapping(raw_templates)
        if "default" not in templates or not templates["default"].responses:
            templates["default"] = fallback["default"]
        return templates

    def _default_template_map(self) -> TemplateMap:
        responses = tuple(
            NLGTemplate(
                template_id=f"default.{index}",
                intent="default",
                text=text,
                priority=0,
                weight=1.0,
            )
            for index, text in enumerate(self.DEFAULT_FALLBACK_RESPONSES)
        )
        return {"default": NLGTemplateSet(intent="default", responses=responses)}


    def _inline_or_default_template_map(self) -> TemplateMap:
        fallback = self._default_template_map()
        inline_templates = self.nlg_config.get("inline_templates")
        if isinstance(inline_templates, Mapping):
            inline_map = self._normalize_template_mapping(inline_templates)
            fallback.update({intent: template_set for intent, template_set in inline_map.items() if template_set.responses})
            if "default" not in fallback:
                fallback.update(self._default_template_map())
        return fallback

    def _normalize_template_mapping(self, raw_templates: Mapping[str, Any]) -> TemplateMap:
        processed: TemplateMap = {}
        for raw_intent, data in raw_templates.items():
            intent = _text(raw_intent, "default").strip() or "default"
            processed[intent] = self._normalize_template_set(intent, data)
        return processed

    def _normalize_template_set(self, intent: str, data: Any) -> NLGTemplateSet:
        triggers: List[str] = []
        defaults: Dict[str, Any] = {}
        required_entities: List[str] = []
        metadata: Dict[str, Any] = {}
        raw_responses: List[Any] = []

        if isinstance(data, str):
            raw_responses = [data]
        elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray, Mapping)):
            raw_responses = list(data)
        elif isinstance(data, Mapping):
            triggers = [_text(item) for item in _list(data.get("triggers", [])) if _text(item)]
            defaults = _mapping(data.get("defaults", data.get("entities", {})))
            required_entities = [_text(item) for item in _list(data.get("required_entities", [])) if _text(item)]
            metadata = {key: value for key, value in data.items() if key not in {"responses", "templates", "triggers", "defaults", "entities", "required_entities"}}
            raw_responses = _list(data.get("responses", data.get("templates", [])))
        else:
            raw_responses = []

        templates: List[NLGTemplate] = []
        for index, raw in enumerate(raw_responses):
            template = self._normalize_single_template(intent, raw, index, required_entities=required_entities)
            if template is not None:
                templates.append(template)

        if not templates and intent == "default":
            templates = list(self._default_template_map()["default"].responses)
        return NLGTemplateSet(
            intent=intent,
            responses=tuple(templates),
            triggers=tuple(triggers),
            defaults=defaults,
            required_entities=tuple(required_entities),
            metadata=metadata,
        )

    def _normalize_single_template(self, intent: str, raw: Any, index: int, *, required_entities: Iterable[str]) -> Optional[NLGTemplate]:
        if isinstance(raw, str):
            text = raw
            payload: Dict[str, Any] = {}
        elif isinstance(raw, Mapping):
            text = _text(raw.get("text", raw.get("template", raw.get("response", ""))))
            payload = dict(raw)
        else:
            text = _text(raw)
            payload = {}
        if not text.strip():
            return None
        template_id = _text(payload.get("id", payload.get("template_id", f"{intent}.{index}")))
        return NLGTemplate(
            template_id=template_id,
            text=text,
            intent=intent,
            priority=_int(payload.get("priority", 0), default=0),
            weight=_float(payload.get("weight", 1.0), default=1.0, minimum=0.0),
            required_entities=tuple(_text(item) for item in _list(payload.get("required_entities", list(required_entities))) if _text(item)),
            forbidden_entities=tuple(_text(item) for item in _list(payload.get("forbidden_entities", [])) if _text(item)),
            conditions=tuple(_text(item) for item in _list(payload.get("conditions", [])) if _text(item)),
            style_tags=tuple(_text(item) for item in _list(payload.get("style_tags", [])) if _text(item)),
            metadata={key: value for key, value in payload.items() if key not in {"id", "template_id", "text", "template", "response", "priority", "weight", "required_entities", "forbidden_entities", "conditions", "style_tags"}},
        )

    def _load_verbose_phrases(self, path: Any) -> Dict[str, List[str]]:
        phrases = {key: list(value) for key, value in self.DEFAULT_VERBOSE_PHRASES.items()}
        inline = self.style.get("verbose_phrases") or self.nlg_config.get("verbose_phrases_inline")
        if isinstance(inline, Mapping):
            for key, values in inline.items():
                phrases[_text(key)] = [_text(item) for item in _list(values) if _text(item)]
        elif isinstance(inline, Sequence) and not isinstance(inline, (str, bytes, bytearray)):
            phrases["default"] = [_text(item) for item in inline if _text(item)]

        resource_path = self._resolve_path(path)
        if resource_path and resource_path.exists():
            try:
                data = self._load_json_or_yaml(resource_path)
                if isinstance(data, Mapping):
                    for key, values in data.items():
                        phrases[_text(key)] = [_text(item) for item in _list(values) if _text(item)]
            except Exception as exc:
                self._add_diagnostic("verbose_phrases_load_failed", "Failed to load verbose phrases.", severity="warning", path=str(resource_path), error=str(exc))
        return phrases

    def _load_modality_markers(self, path: Any) -> Dict[str, List[str]]:
        markers = {key: list(value) for key, value in self.DEFAULT_MODALITY_MARKERS.items()}
        inline = self.nlg_config.get("modality_markers")
        if isinstance(inline, Mapping):
            for key, values in inline.items():
                markers[_text(key)] = [_text(item) for item in _list(values) if _text(item)]

        resource_path = self._resolve_path(path)
        if resource_path and resource_path.exists():
            try:
                data = self._load_json_or_yaml(resource_path)
                if isinstance(data, Mapping):
                    for key, values in data.items():
                        markers[_text(key)] = [_text(item) for item in _list(values) if _text(item)]
            except Exception as exc:
                self._add_diagnostic("modality_markers_load_failed", "Failed to load modality markers.", severity="warning", path=str(resource_path), error=str(exc))
        return markers

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------
    def set_neural_generator(self, generator: Optional[NeuralGenerator]) -> None:
        """Register an optional callable neural generator.

        The engine does not fake neural output. If no callable is registered,
        neural mode cleanly falls back to template generation according to config.
        """

        self.neural_generator = generator

    def generate(self, frame: LinguisticFrame, context: Optional[Any] = None) -> str:
        """Backward-compatible generation API used by LanguageAgent."""

        return self.generate_detailed(frame, context).text

    def generate_detailed(self, frame: LinguisticFrame, context: Optional[Any] = None) -> NLGGenerationResult:
        started_at = time_module.time()
        self.generation_count += 1
        context_packet = self._build_context_packet(context, frame)
        frame = self._normalize_frame(frame, context_packet)
        self.intent_counts[frame.intent] += 1

        try:
            if self.generation_mode in {"neural", "hybrid"}:
                neural_result = self._try_neural_generation(frame, context_packet)
                if neural_result and neural_result.ok:
                    self._record_generation(neural_result)
                    return neural_result
                if self.generation_mode == "neural" and not self.fallback_after_retries:
                    raise NLGGenerationError("Neural generation unavailable or failed validation.", frame=frame, context=context_packet.to_dict(), error_type="neural_generation_failed")

            result = self._template_generation(frame, context_packet)
            result = self._finalize_result(result, started_at=started_at)
            self._record_generation(result)
            return result
        except Exception as exc:
            self.failed_generation_count += 1
            logger.error("NLG generation failed: %s", exc)
            fallback = self._fallback_generation(frame, context_packet, reason=str(exc), started_at=started_at)
            self._record_generation(fallback)
            return fallback

    # ------------------------------------------------------------------
    # Context/frame preparation
    # ------------------------------------------------------------------
    def _normalize_frame(self, frame: LinguisticFrame, context: NLGContextPacket) -> LinguisticFrame:
        if not isinstance(frame, LinguisticFrame):
            raise NLGGenerationError("NLGEngine.generate expects a LinguisticFrame.", frame=None, context=context.to_dict(), error_type="invalid_frame")
        intent = _text(frame.intent).strip()
        if not intent:
            intent = self._match_intent_by_trigger(context.relevant_context or context.summary or "")
        if not intent:
            intent = "default"
        if intent == frame.intent:
            return frame
        return LinguisticFrame(
            intent=intent,
            entities=dict(frame.entities or {}),
            sentiment=float(frame.sentiment),
            modality=_text(frame.modality),
            confidence=float(frame.confidence),
            act_type=frame.act_type,
            propositional_content=getattr(frame, "propositional_content", None),
            illocutionary_force=getattr(frame, "illocutionary_force", None),
            perlocutionary_effect=getattr(frame, "perlocutionary_effect", None),
        )

    def _build_context_packet(self, context: Optional[Any], frame: Optional[LinguisticFrame]) -> NLGContextPacket:
        if context is None:
            return NLGContextPacket()

        # Prefer a production DialogueContext method if available.
        if hasattr(context, "build_nlg_context") and callable(context.build_nlg_context):
            try:
                packet = context.build_nlg_context(frame=frame) if frame is not None else context.build_nlg_context()
                if isinstance(packet, NLGContextPacket):
                    return packet
                if isinstance(packet, Mapping):
                    return self._context_packet_from_mapping(packet)
            except TypeError:
                packet = context.build_nlg_context()
                if isinstance(packet, Mapping):
                    return self._context_packet_from_mapping(packet)

        if isinstance(context, Mapping):
            return self._context_packet_from_mapping(context)

        history = self._extract_history(context)
        summary = _text(getattr(context, "summary", ""))
        relevant_context = ""
        if hasattr(context, "get_relevant_context") and callable(context.get_relevant_context):
            try:
                relevant_context = _text(context.get_relevant_context())
            except TypeError:
                relevant_context = _text(context.get_relevant_context(history_window=3))
        slots = _mapping(getattr(context, "slot_values", {}))
        environment: Dict[str, Any] = {}
        if hasattr(context, "environment_state"):
            environment = _mapping(getattr(context, "environment_state"))
        elif hasattr(context, "get_environment_state") and callable(context.get_environment_state):
            for key in ("session_id", "last_intent", "pending_intent", "current_topic", "conversation_phase"):
                try:
                    value = context.get_environment_state(key)
                    if value is not None:
                        environment[key] = value
                except Exception:
                    pass
        unresolved = [_json_safe(item) for item in _list(getattr(context, "unresolved_issues", []))]
        return NLGContextPacket(
            summary=summary,
            history=tuple(history),
            relevant_context=relevant_context,
            slots=slots,
            environment=environment,
            unresolved_issues=tuple(dict(item) for item in unresolved if isinstance(item, Mapping)),
            metadata={"context_type": context.__class__.__name__},
        )

    def _context_packet_from_mapping(self, payload: Mapping[str, Any]) -> NLGContextPacket:
        return NLGContextPacket(
            summary=_text(payload.get("summary", payload.get("context_summary", ""))),
            history=tuple(_json_safe(item) for item in _list(payload.get("history", [])) if isinstance(_json_safe(item), Mapping)),
            relevant_context=_text(payload.get("relevant_context", payload.get("context", ""))),
            slots=_mapping(payload.get("slots", payload.get("slot_values", {}))),
            environment=_mapping(payload.get("environment", payload.get("environment_state", {}))),
            unresolved_issues=tuple(_json_safe(item) for item in _list(payload.get("unresolved_issues", [])) if isinstance(_json_safe(item), Mapping)),
            memory_matches=tuple(_json_safe(item) for item in _list(payload.get("memory_matches", [])) if isinstance(_json_safe(item), Mapping)),
            metadata={key: _json_safe(value) for key, value in payload.items() if key not in {"summary", "context_summary", "history", "relevant_context", "context", "slots", "slot_values", "environment", "environment_state", "unresolved_issues", "memory_matches"}},
        )

    def _extract_history(self, context: Any) -> List[Dict[str, Any]]:
        raw_history = getattr(context, "history", [])
        output: List[Dict[str, Any]] = []
        for item in _list(raw_history):
            if isinstance(item, Mapping):
                output.append({"role": _text(item.get("role", "unknown")), "content": _text(item.get("content", item.get("text", "")))})
            elif hasattr(item, "to_dict"):
                mapped = item.to_dict()
                if isinstance(mapped, Mapping):
                    output.append({"role": _text(mapped.get("role", "unknown")), "content": _text(mapped.get("content", mapped.get("text", "")))})
        return output

    # ------------------------------------------------------------------
    # Generation modes
    # ------------------------------------------------------------------
    def _try_neural_generation(self, frame: LinguisticFrame, context: NLGContextPacket) -> Optional[NLGGenerationResult]:
        if self.neural_generator is None:
            self._add_diagnostic("neural_generator_missing", "Neural generation requested but no neural generator callable is registered.", severity="warning")
            return None

        prompt = self._build_neural_prompt(frame, context)
        attempts: List[NLGRenderAttempt] = []
        for attempt_index in range(max(1, self.max_retries + 1)):
            try:
                response = self.neural_generator(prompt, frame, context.to_dict())
                response = self._adapt_style(_clean_space(response), frame=frame, context=context)
                validation = self._validate_response(response)
                attempts.append(NLGRenderAttempt(template_id=f"neural.{attempt_index}", success=validation[0], error=None if validation[0] else "; ".join(validation[1]), rendered_preview=_truncate(response, 160)))
                if validation[0]:
                    return NLGGenerationResult(
                        text=response,
                        intent=frame.intent,
                        frame=frame,
                        template_id=None,
                        fallback_used=False,
                        generation_mode="neural",
                        confidence=frame.confidence,
                        attempts=tuple(attempts),
                        context=context,
                        metadata={"prompt_preview": _truncate(prompt, 500)},
                    )
                self.validation_failure_count += 1
            except Exception as exc:
                attempts.append(NLGRenderAttempt(template_id=f"neural.{attempt_index}", success=False, error=str(exc)))
        return NLGGenerationResult(
            text="",
            intent=frame.intent,
            frame=frame,
            fallback_used=True,
            generation_mode="neural",
            confidence=0.0,
            attempts=tuple(attempts),
            context=context,
            issues=tuple(attempt.error for attempt in attempts if attempt.error),
        )

    def _template_generation(self, frame: LinguisticFrame, context: NLGContextPacket) -> NLGGenerationResult:
        template_set = self._get_template_set(frame.intent)
        payload = self._build_render_payload(frame, context, template_set)
        candidates = self._rank_templates(template_set, payload, frame, context)
        attempts: List[NLGRenderAttempt] = []

        for template in candidates:
            rendered, missing = render_template_safely(template.text, payload)
            if missing:
                attempts.append(NLGRenderAttempt(template_id=template.template_id, success=False, missing_fields=tuple(missing), error="missing_fields"))
                continue
            rendered = self._postprocess_text(rendered, frame=frame, context=context)
            valid, reasons = self._validate_response(rendered)
            attempts.append(
                NLGRenderAttempt(
                    template_id=template.template_id,
                    success=valid,
                    missing_fields=(),
                    error=None if valid else "; ".join(reasons),
                    rendered_preview=_truncate(rendered, 160),
                )
            )
            if valid:
                return NLGGenerationResult(
                    text=rendered,
                    intent=frame.intent,
                    frame=frame,
                    template_id=template.template_id,
                    fallback_used=template.intent == "default" and frame.intent != "default",
                    generation_mode="template",
                    confidence=frame.confidence,
                    attempts=tuple(attempts),
                    context=context,
                    metadata={"template_intent": template.intent, "payload_keys": sorted(payload.keys())},
                )
            self.validation_failure_count += 1

        return self._fallback_generation(frame, context, reason="all_template_attempts_failed", attempts=attempts)

    def _fallback_generation(
        self,
        frame: LinguisticFrame,
        context: NLGContextPacket,
        *,
        reason: str,
        attempts: Optional[List[NLGRenderAttempt]] = None,
        started_at: Optional[float] = None,
    ) -> NLGGenerationResult:
        self.fallback_count += 1
        fallback_set = self.templates.get("default", self._default_template_map()["default"])
        payload = self._build_render_payload(frame, context, fallback_set)
        response = ""
        template_id: Optional[str] = None
        attempt_records = list(attempts or [])

        for template in self._rank_templates(fallback_set, payload, frame, context):
            rendered, missing = render_template_safely(template.text, payload)
            if missing:
                attempt_records.append(NLGRenderAttempt(template_id=template.template_id, success=False, missing_fields=tuple(missing), error="missing_fields"))
                continue
            response = self._postprocess_text(rendered, frame=frame, context=context)
            template_id = template.template_id
            attempt_records.append(NLGRenderAttempt(template_id=template.template_id, success=True, rendered_preview=_truncate(response, 160)))
            break

        if not response:
            response = self.DEFAULT_FALLBACK_RESPONSES[0]
            template_id = "default.static"

        metadata = {"reason": reason}
        if started_at is not None:
            metadata["duration_ms"] = str(round((time_module.time() - started_at) * 1000.0, 3))
        return NLGGenerationResult(
            text=response,
            intent=frame.intent,
            frame=frame,
            template_id=template_id,
            fallback_used=True,
            generation_mode="fallback",
            confidence=min(frame.confidence, 0.5),
            attempts=tuple(attempt_records),
            context=context,
            issues=(reason,),
            metadata=metadata,
        )

    def _finalize_result(self, result: NLGGenerationResult, *, started_at: float) -> NLGGenerationResult:
        metadata = dict(result.metadata)
        metadata["duration_ms"] = round((time_module.time() - started_at) * 1000.0, 3)
        return NLGGenerationResult(
            text=result.text,
            intent=result.intent,
            frame=result.frame,
            template_id=result.template_id,
            fallback_used=result.fallback_used,
            generation_mode=result.generation_mode,
            confidence=result.confidence,
            attempts=result.attempts,
            context=result.context,
            issues=result.issues,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Template selection and payload construction
    # ------------------------------------------------------------------
    def _get_template_set(self, intent: str) -> NLGTemplateSet:
        if intent in self.templates and self.templates[intent].responses:
            return self.templates[intent]
        self._add_diagnostic("template_intent_missing", "No template set exists for requested intent; using default.", severity="warning", intent=intent)
        return self.templates.get("default", self._default_template_map()["default"])

    def _rank_templates(self, template_set: NLGTemplateSet, payload: Mapping[str, Any], frame: LinguisticFrame, context: NLGContextPacket) -> List[NLGTemplate]:
        scored: List[Tuple[float, float, NLGTemplate]] = []
        for template in template_set.responses:
            missing_required = [field for field in template.required_entities if not payload.get(field)]
            forbidden_present = [field for field in template.forbidden_entities if payload.get(field)]
            if missing_required or forbidden_present:
                continue
            score = float(template.priority) + float(template.weight)
            fields = set(template.placeholders)
            satisfied = sum(1 for field in fields if payload.get(field) not in (None, ""))
            score += satisfied * 0.15
            if template.style_tags:
                score += self._style_tag_score(template.style_tags) * 0.10
            if frame.confidence < 0.5 and "clarification" in template.style_tags:
                score += 0.5
            # Stable random jitter to avoid always picking the first equal template.
            scored.append((score, self.rng.random(), template))
        if not scored:
            scored = [(float(template.priority) + float(template.weight), self.rng.random(), template) for template in template_set.responses]
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored]

    def _style_tag_score(self, tags: Iterable[str]) -> float:
        tag_set = {_text(tag).lower() for tag in tags}
        formality = _float(self.style.get("formality", 0.5), default=0.5, minimum=0.0, maximum=1.0)
        verbosity = _float(self.style.get("verbosity", 1.0), default=1.0, minimum=0.0)
        score = 0.0
        if "formal" in tag_set:
            score += formality
        if "informal" in tag_set:
            score += 1.0 - formality
        if "verbose" in tag_set:
            score += min(verbosity / 2.0, 1.0)
        if "concise" in tag_set:
            score += 1.0 if verbosity < 0.8 else 0.0
        return score

    def _build_render_payload(self, frame: LinguisticFrame, context: NLGContextPacket, template_set: NLGTemplateSet) -> RenderPayload:
        entities = self._flatten_entities(frame.entities or {})
        slots = self._flatten_entities(context.slots)
        payload: RenderPayload = {}
        payload.update(template_set.defaults)
        payload.update(slots)
        payload.update(entities)

        payload.update(
            {
                "intent": frame.intent,
                "confidence": f"{frame.confidence:.2f}",
                "confidence_raw": frame.confidence,
                "confidence_qualifier": self._get_confidence_qualifier(frame.confidence),
                "sentiment_marker": self._get_sentiment_marker(frame.sentiment),
                "modality": frame.modality,
                "modality_marker": self._get_modality_marker(frame.modality),
                "act_type": frame.act_type.value if getattr(frame, "act_type", None) else "",
                "act_type_marker": self._get_act_type_marker(frame.act_type),
                "propositional_content": getattr(frame, "propositional_content", "") or "",
                "illocutionary_force": getattr(frame, "illocutionary_force", "") or "",
                "perlocutionary_effect": getattr(frame, "perlocutionary_effect", "") or "",
                "time": time_module.strftime(_text(self.nlg_config.get("time_format", "%H:%M"))),
                "date": time_module.strftime(_text(self.nlg_config.get("date_format", "%Y-%m-%d"))),
                "summary": context.summary,
                "context": context.relevant_context or context.summary,
                "current_topic": context.environment.get("current_topic", "general"),
                "session_id": context.environment.get("session_id", ""),
                "pending_intent": context.environment.get("pending_intent", frame.intent),
                "mentioned_entities": self._format_entities(entities),
                "slots": self._format_entities(slots),
            }
        )

        if frame.intent == "weather_inquiry":
            payload.setdefault("weather_status", "unknown conditions")
            payload.setdefault("temperature", "N/A")
            payload.setdefault("unit", "")
        return payload

    def _flatten_entities(self, entities: Mapping[str, Any]) -> Dict[str, Any]:
        flattened: Dict[str, Any] = {}
        for key, value in dict(entities or {}).items():
            if isinstance(value, Mapping):
                flattened[_text(key)] = self._format_entities(value)
                for child_key, child_value in value.items():
                    flattened[f"{key}.{child_key}"] = child_value
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                values = [item for item in value if item is not None]
                flattened[_text(key)] = values[0] if len(values) == 1 else values
            else:
                flattened[_text(key)] = value
        return flattened

    def _format_entities(self, entities: Mapping[str, Any]) -> str:
        parts: List[str] = []
        for key, value in dict(entities or {}).items():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                rendered = ", ".join(_text(item) for item in value)
            else:
                rendered = _text(value)
            parts.append(f"{key}: {rendered}")
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Markers, style, validation
    # ------------------------------------------------------------------
    def _get_modality_marker(self, modality: str) -> str:
        if not modality:
            return ""
        markers = self.modality_markers.get(_text(modality), [])
        return self.rng.choice(markers) + " " if markers and _bool(self.nlg_config.get("prepend_modality_markers", False), default=False) else ""

    def _get_modality_markers(self, modality: str) -> List[str]:
        """Compatibility method used by earlier tests."""

        return list(self.modality_markers.get(_text(modality), []))

    def _get_confidence_qualifier(self, confidence: float) -> str:
        confidence = _float(confidence, default=1.0, minimum=0.0, maximum=1.0)
        if confidence < _float(self.nlg_config.get("very_low_confidence_threshold", 0.4), default=0.4):
            return self.rng.choice(["I'm not entirely sure, but ", "I might be mistaken, but "])
        if confidence < _float(self.nlg_config.get("low_confidence_threshold", 0.7), default=0.7):
            return self.rng.choice(["I believe ", "It seems that "])
        return ""

    def _get_sentiment_marker(self, sentiment: float) -> str:
        sentiment = _float(sentiment, default=0.0, minimum=-1.0, maximum=1.0)
        if sentiment < _float(self.nlg_config.get("negative_sentiment_threshold", -0.5), default=-0.5):
            return self.rng.choice(["I'm sorry to hear that. ", "That sounds difficult. "])
        if sentiment > _float(self.nlg_config.get("positive_sentiment_threshold", 0.5), default=0.5):
            return self.rng.choice(["Great. ", "I'm glad to hear that. "])
        return ""

    def _get_act_type_marker(self, act_type: SpeechActType) -> str:
        if not _bool(self.nlg_config.get("prepend_speech_act_markers", False), default=False):
            return ""
        markers = {
            SpeechActType.ASSERTIVE: "",
            SpeechActType.DIRECTIVE: "Please ",
            SpeechActType.COMMISSIVE: "I will ",
            SpeechActType.EXPRESSIVE: self.rng.choice(["I appreciate that. ", "Thank you. "]),
            SpeechActType.DECLARATION: "I declare that ",
        }
        return markers.get(act_type, "")

    def _postprocess_text(self, text: str, *, frame: LinguisticFrame, context: NLGContextPacket) -> str:
        text = _clean_space(text)
        text = self._apply_prefix_markers(text)
        text = self._adapt_style(text, frame=frame, context=context)
        text = self._ensure_sentence_final_punctuation(text)
        return _clean_space(text)

    def _apply_prefix_markers(self, text: str) -> str:
        # Templates can explicitly use markers. Avoid blindly prepending them
        # globally because that creates duplicated or ungrammatical responses.
        return text

    def _adapt_style(self, text: str, *, frame: Optional[LinguisticFrame] = None, context: Optional[NLGContextPacket] = None) -> str:
        formality = _float(self.style.get("formality", 0.5), default=0.5, minimum=0.0, maximum=1.0)
        verbosity = _float(self.style.get("verbosity", 1.0), default=1.0, minimum=0.0)
        output = _text(text)

        if formality > 0.7:
            output = self._replace_phrases(output, self.FORMAL_CONTRACTIONS)
        elif formality < 0.3:
            output = self._replace_phrases(output, self.INFORMAL_CONTRACTIONS)
            if _bool(self.style.get("allow_slang", False), default=False):
                output = re.sub(r"\bgoing to\b", "gonna", output, flags=re.IGNORECASE)
                output = re.sub(r"\bwant to\b", "wanna", output, flags=re.IGNORECASE)
                output = re.sub(r"\bgot to\b", "gotta", output, flags=re.IGNORECASE)
            informal_phrases = [_text(item) for item in _list(self.style.get("informal_phrases", [])) if _text(item)]
            if informal_phrases and self.rng.random() < _float(self.style.get("informal_prefix_probability", 0.15), default=0.15, minimum=0.0, maximum=1.0):
                output = f"{self.rng.choice(informal_phrases)} {output}"

        if verbosity < 0.7:
            truncation_length = _int(self.style.get("truncation_length", 25), default=25, minimum=1)
            words = output.split()
            if len(words) > truncation_length:
                output = " ".join(words[:truncation_length]).rstrip(" ,;:") + "..."
        elif verbosity > 1.3:
            mode = _text(self.style.get("verbosity_mode", "default"))
            pool = self.verbose_phrases.get(mode, self.verbose_phrases.get("default", []))
            if pool and self.rng.random() < _float(self.style.get("verbose_prefix_probability", 0.25), default=0.25, minimum=0.0, maximum=1.0):
                prefix = self.rng.choice(pool)
                if not output.lower().startswith(prefix.lower()):
                    output = f"{prefix} {output}"

        return _clean_space(output)

    def _replace_phrases(self, text: str, replacements: Mapping[str, str]) -> str:
        output = text
        for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            output = re.sub(rf"\b{re.escape(source)}\b", target, output, flags=re.IGNORECASE)
        return output

    def _ensure_sentence_final_punctuation(self, text: str) -> str:
        value = text.strip()
        if not value:
            return value
        if value.endswith((".", "!", "?", "…", "...")):
            return value
        return value + "."

    def _validate_response(self, response: str) -> Tuple[bool, List[str]]:
        text = _text(response).strip()
        reasons: List[str] = []
        if not text:
            reasons.append("empty_response")
        word_count = len(text.split())
        if word_count < self.min_words:
            reasons.append("too_few_words")
        if word_count > self.max_words:
            reasons.append("too_many_words")
        if len(text) > self.max_characters:
            reasons.append("too_many_characters")
        for pattern in self.blocked_response_patterns:
            if pattern.search(text):
                reasons.append(f"blocked_pattern:{pattern.pattern}")
        return not reasons, reasons

    # ------------------------------------------------------------------
    # Compatibility and neural prompt helpers
    # ------------------------------------------------------------------
    def _match_intent_by_trigger(self, text: str) -> str:
        text_clean = re.sub(r"[^\w\s]", " ", _text(text).lower())
        best_intent = "default"
        best_score = 0.0
        for intent, template_set in self.templates.items():
            if intent == "default":
                continue
            for trigger in template_set.triggers:
                trigger_clean = re.sub(r"[^\w\s]", " ", trigger.lower()).strip()
                if not trigger_clean:
                    continue
                match = re.search(r"\b" + re.escape(trigger_clean) + r"\b", text_clean)
                if match:
                    score = len(trigger_clean.split()) + (1.0 - match.start() / max(len(text_clean), 1))
                    if score > best_score:
                        best_score = score
                        best_intent = intent
        return best_intent

    def _match_by_trigger(self, user_text: str) -> str:
        intent = self._match_intent_by_trigger(user_text)
        frame = LinguisticFrame(intent=intent, entities={}, sentiment=0.0, modality="epistemic", confidence=0.5, act_type=SpeechActType.ASSERTIVE)
        return self.generate(frame, context=user_text)

    def _match_template(self, frame: LinguisticFrame) -> Optional[str]:
        template_set = self.templates.get(frame.intent)
        if not template_set or not template_set.responses:
            return None
        payload = self._build_render_payload(frame, NLGContextPacket(), template_set)
        for template in self._rank_templates(template_set, payload, frame, NLGContextPacket()):
            rendered, missing = render_template_safely(template.text, payload)
            if not missing:
                return rendered
        return None

    def _neural_generation(self, frame: LinguisticFrame, context: Optional[Any]) -> str:
        result = self._try_neural_generation(frame, self._build_context_packet(context, frame))
        if result and result.ok:
            return result.text
        return self.generate(frame, context)

    def _build_neural_prompt(self, frame: LinguisticFrame, context: NLGContextPacket) -> str:
        prompt_parts = [
            "# Role: Helpful AI Assistant",
            "## Task: Generate a natural response from the following linguistic frame.",
            f"- Intent: {frame.intent}",
            f"- Speech act: {frame.act_type.value if frame.act_type else ''}",
            f"- Modality: {frame.modality}",
            f"- Confidence: {frame.confidence:.2f}",
        ]
        if frame.entities:
            prompt_parts.append(f"- Entities: {json.dumps(_json_safe(frame.entities), ensure_ascii=False)}")
        if getattr(frame, "propositional_content", None):
            prompt_parts.append(f"- Content focus: {frame.propositional_content}")
        if context.summary:
            prompt_parts.append(f"\n## Conversation summary\n{context.summary}")
        if context.relevant_context:
            prompt_parts.append(f"\n## Relevant context\n{context.relevant_context}")
        prompt_parts.extend(
            [
                "\n## Requirements",
                "- Be coherent with the dialogue context.",
                "- Do not expose internal diagnostics.",
                "- Keep the response concise unless the style config asks for detail.",
                "\n## Response:",
            ]
        )
        return "\n".join(prompt_parts)

    # ------------------------------------------------------------------
    # Diagnostics, history, stats
    # ------------------------------------------------------------------
    def _add_diagnostic(self, code: str, message: str, *, severity: str = "warning", **details: Any) -> None:
        payload = {
            "code": code,
            "message": message,
            "severity": severity,
            "module": "NLGEngine",
            "details": _json_safe(details),
            "created_at": _utc_timestamp(),
        }
        self.diagnostics.append(payload)
        if severity in {"error", "critical"}:
            logger.error(json.dumps(payload, ensure_ascii=False))
        else:
            logger.warning(json.dumps(payload, ensure_ascii=False))

    def _record_generation(self, result: NLGGenerationResult) -> None:
        if not self.record_history:
            return
        self.generation_history.append(
            {
                "timestamp": _utc_timestamp(),
                "intent": result.intent,
                "template_id": result.template_id,
                "fallback_used": result.fallback_used,
                "generation_mode": result.generation_mode,
                "text_preview": _truncate(result.text, _int(self.nlg_config.get("history_preview_length", 160), default=160, minimum=16)),
                "ok": result.ok,
            }
        )

    def _template_count(self) -> int:
        return sum(len(template_set.responses) for template_set in self.templates.values())

    def stats(self) -> NLGEngineStats:
        return NLGEngineStats(
            version=self.version,
            template_count=self._template_count(),
            intent_count=len(self.templates),
            generation_count=self.generation_count,
            fallback_count=self.fallback_count,
            failed_generation_count=self.failed_generation_count,
            validation_failure_count=self.validation_failure_count,
            history_length=len(self.generation_history),
            diagnostics_count=len(self.diagnostics),
            templates_path=str(self._resolve_path(self.templates_path)) if self.templates_path else None,
            generation_mode=self.generation_mode,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.__class__.__name__,
            "version": self.version,
            "stats": self.stats().to_dict(),
            "templates": {intent: template_set.to_dict() for intent, template_set in self.templates.items()},
            "style": _json_safe(self.style),
            "diagnostics": _json_safe(self.diagnostics),
            "history": list(self.generation_history),
            "intent_counts": dict(self.intent_counts),
        }

    def __repr__(self) -> str:
        return f"<NLGEngine version='{self.version}' intents={len(self.templates)} templates={self._template_count()} mode='{self.generation_mode}'>"


if __name__ == "__main__":
    print("\n=== Running NLG Engine ===\n")
    printer.status("TEST", "NLG Engine initialized", "info")

    engine = NLGEngine()

    weather_frame = LinguisticFrame(
        intent="weather_inquiry",
        entities={
            "location": "Paris",
            "temperature": "21",
            "unit": "C",
            "weather_status": "partly cloudy",
        },
        sentiment=0.1,
        modality="epistemic",
        confidence=0.95,
        act_type=SpeechActType.DIRECTIVE,
    )

    clarification_frame = LinguisticFrame(
        intent="clarification_request",
        entities={"pending_intent": "travel planning", "mentioned_entities": "destination: Amsterdam"},
        sentiment=0.0,
        modality="interrogative",
        confidence=1.0,
        act_type=SpeechActType.DIRECTIVE,
    )

    error_frame = LinguisticFrame(
        intent="nlu_error",
        entities={"detail": "No intent could be selected confidently."},
        sentiment=-0.1,
        modality="error",
        confidence=1.0,
        act_type=SpeechActType.ASSERTIVE,
    )

    context = {
        "summary": "The user is testing the language pipeline.",
        "history": [
            {"role": "user", "content": "Can you check the weather in Paris?"},
            {"role": "assistant", "content": "I can help with that."},
        ],
        "environment": {"session_id": "nlg-test-session", "current_topic": "weather"},
        "slots": {"location": "Paris"},
    }

    weather_result = engine.generate_detailed(weather_frame, context)
    clarification_result = engine.generate_detailed(clarification_frame, context)
    error_result = engine.generate_detailed(error_frame, context)
    trigger_response = engine._match_by_trigger("Can you tell me the weather?")
    template_match = engine._match_template(weather_frame)

    printer.pretty("WEATHER_RESULT", weather_result.to_dict(), "success")
    printer.pretty("CLARIFICATION_RESULT", clarification_result.to_dict(), "success")
    printer.pretty("ERROR_RESULT", error_result.to_dict(), "success")
    printer.pretty("TRIGGER_RESPONSE", {"text": trigger_response}, "success")
    printer.pretty("TEMPLATE_MATCH", {"text": template_match}, "success" if template_match else "warning")
    printer.pretty("STATS", engine.stats().to_dict(), "success")
    printer.pretty("DIAGNOSTICS", engine.diagnostics, "info")

    print("\n=== Test ran successfully ===\n")
