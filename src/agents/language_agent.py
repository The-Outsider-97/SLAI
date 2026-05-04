from __future__ import annotations

__version__ = "2.1.0"

"""
Production-ready Language Agent.

Core Function:
Coordinates the language subsystem pipeline and returns the final agent response.

Pipeline:
    User Input
    -> SafetyGuard input sanitization
    -> OrthographyProcessor: spellcheck + normalize
    -> NLPEngine: tokenization, lemmatization, POS, dependencies
    -> GrammarProcessor: grammar checks
    -> NLUEngine: intent/entities/sentiment/modality -> LinguisticFrame
    -> DialogueContext: context assembly, slots, unresolved issues, topic state
    -> NLGEngine: response generation
    -> SafetyGuard output sanitization
    -> DialogueContext + SharedMemory: final turn + trace logging
    -> Agent Output

Responsibilities:
- Preserve the public agent API: pipeline(...), process(...), predict(...), act(...), and BaseAgent execution.
- Keep agent-level runtime policy in agents_config.yaml under the language_agent section.
- Initialize language components exactly once and inject shared dependencies where appropriate.
- Pass precomputed NLP artifacts into GrammarProcessor and NLUEngine instead of rerunning NLP.
- Keep the agent language-only: no external-retrieval delegation belongs here.
- Produce structured pipeline traces, component health reports, diagnostics, and deterministic recovery responses.

Integration notes:
The agent coordinates modules; it does not own their internal configuration.
Language module/resource configuration remains in language_config.yaml. Agent
runtime behavior, stage policy, shared-memory audit, and recovery policy live in
agents_config.yaml under the language_agent section.
"""

import time as time_module
import uuid

from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from .base.utils.main_config_loader import load_global_config, get_config_section
from .base_agent import BaseAgent
from .safety.safety_guard import SafetyGuard
from .language.orthography_processor import OrthographyProcessor
from .language.dialogue_context import DialogueContext
from .language.grammar_processor import GrammarProcessor, GrammarAnalysisResult
from .language.nlg_engine import NLGEngine
from .language.nlu_engine import NLUEngine, Wordlist
from .language.nlp_engine import NLPEngine
from .language.utils.linguistic_frame import LinguisticFrame, SpeechActType
from .language.utils.language_error import *
from .language.utils.language_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Language Agent")
printer = PrettyPrinter()


_STAGE_ALIASES = {
    "dialogue_context_pre_nlg": "dialogue_pre_nlg",
    "dialogue_context_post_nlg": "dialogue_post_nlg",
}

class PipelineStatus(str, Enum):
    """Terminal or intermediate status for a pipeline execution."""

    SUCCESS = "success"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"


class StageName(str, Enum):
    """Canonical language-agent pipeline stage names."""

    SAFETY_INPUT = "safety_input"
    ORTHOGRAPHY = "orthography"
    NLP = "nlp"
    GRAMMAR = "grammar"
    NLU = "nlu"
    DIALOGUE_PRE_NLG = "dialogue_pre_nlg"
    NLG = "nlg"
    SAFETY_OUTPUT = "safety_output"
    DIALOGUE_POST_NLG = "dialogue_post_nlg"
    SHARED_MEMORY = "shared_memory"


class StagePolicy(str, Enum):
    """Configurable policy for stage failures."""

    CONTINUE = "continue"
    FAIL = "fail"
    BLOCK = "block"


@dataclass(frozen=True)
class StageRecord:
    """Audit record for one pipeline stage."""

    name: str
    ok: bool
    started_at: float
    finished_at: float
    duration_ms: int
    policy: str = StagePolicy.CONTINUE.value
    message: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "policy": self.policy,
            "message": self.message,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "metadata": json_safe(self.metadata),
        }


@dataclass
class PipelineTrace:
    """Complete trace for one agent pipeline execution."""

    trace_id: str
    session_id: Optional[str]
    started_at: float
    input_preview: str
    status: PipelineStatus = PipelineStatus.SUCCESS
    records: List[StageRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    finished_at: Optional[float] = None

    def add(self, record: StageRecord) -> None:
        self.records.append(record)
        if not record.ok and self.status == PipelineStatus.SUCCESS:
            self.status = PipelineStatus.PARTIAL

    def warn(self, message: str) -> None:
        if message and message not in self.warnings:
            self.warnings.append(message)
        if self.status == PipelineStatus.SUCCESS:
            self.status = PipelineStatus.PARTIAL

    def finish(self, status: Optional[PipelineStatus] = None) -> None:
        self.finished_at = time_module.time()
        if status is not None:
            self.status = status

    @property
    def duration_ms(self) -> int:
        return int(((self.finished_at or time_module.time()) - self.started_at) * 1000)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "input_preview": self.input_preview,
            "warnings": list(self.warnings),
            "records": [record.to_dict() for record in self.records],
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class PipelineArtifacts:
    """Intermediate artifacts created by the pipeline."""

    original_text: str
    sanitized_text: str = ""
    orthography_text: str = ""
    nlp_tokens: Tuple[Any, ...] = ()
    dependencies: Tuple[Any, ...] = ()
    grammar_result: Optional[GrammarAnalysisResult] = None
    frame: Optional[LinguisticFrame] = None

    def to_dict(self, *, preview_chars: int = 240) -> Dict[str, Any]:
        return {
            "original_text": compact(self.original_text, preview_chars),
            "sanitized_text": compact(self.sanitized_text, preview_chars),
            "orthography_text": compact(self.orthography_text, preview_chars),
            "token_count": len(self.nlp_tokens),
            "dependency_count": len(self.dependencies),
            "grammar": grammar_summary(self.grammar_result),
            "frame": frame_to_dict(self.frame),
        }


@dataclass(frozen=True)
class LanguageAgentResponse:
    """Structured agent response returned by process()/predict()/perform_task()."""

    response: str
    confidence: float
    intent: str
    status: str
    trace_id: str
    session_id: Optional[str] = None
    frame: Optional[LinguisticFrame] = None
    grammar_ok: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.response,
            "confidence": self.confidence,
            "intent": self.intent,
            "status": self.status,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "frame": frame_to_dict(self.frame),
            "grammar_ok": self.grammar_ok,
            "metadata": json_safe(self.metadata),
        }


class LanguageAgentRuntimeError(RuntimeError):
    """Recoverable language-agent runtime failure."""


class LanguageAgentConfigurationError(ValueError):
    """Language-agent configuration failure."""


class LanguageAgent(BaseAgent):
    """Production coordinator for the language subsystem."""

    DEFAULT_POLICY: Dict[str, Any] = {
        "low_confidence_threshold": 0.45,
        "reprompt_limit": 3,
        "clarification_triggers": ["unknown", "unknown_intent", "nlu_error", "low_confidence_intent"],
        "fallback_responses": [
            "Could you please rephrase that?",
            "I'm not quite sure I understand. Can you say that another way?",
            "Could you provide a little more detail?",
        ],
        "error_responses": {
            "safety_blocked": "I'm sorry, your input triggered a safety concern. Please rephrase or try something else.",
            "nlp_error": "I had trouble understanding the structure of your message. Could you try again?",
            "nlu_error": "I'm having difficulty grasping the meaning. Please rephrase.",
            "nlg_error": "I apologize, I couldn't formulate a proper response.",
            "default_error": "Sorry, something went wrong on my end.",
        },
    }

    def __init__(self, shared_memory: Any, agent_factory: Any, config: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory, config=config)
        self.config = load_global_config()
        self.language_config: Dict[str, Any] = dict(get_config_section("language_agent") or {})
        if config:
            self.language_config.update(dict(config))

        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.pipeline_history: Deque[Dict[str, Any]] = deque(
            maxlen=coerce_int(self.language_config.get("history_limit"), default=200, minimum=1)
        )
        self.stage_failures: Counter[str] = Counter()
        self.stage_successes: Counter[str] = Counter()
        self.component_status: Dict[str, Dict[str, Any]] = {}
        self.started_at = time_module.time()

        self._load_config()
        self._validate_config()
        self._initialize_components()
        self._publish_event("initialized", self.health_check())

        logger.info("Language Agent initialized")
        printer.status("INIT", "Language Agent initialized", "success")

    # ------------------------------------------------------------------
    # Configuration and initialization
    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        cfg = self.language_config
        self.enabled = coerce_bool(cfg.get("enabled"), default=True)
        self.max_input_chars = coerce_int(cfg.get("max_input_chars"), default=20000, minimum=1)
        self.history_limit = coerce_int(cfg.get("history_limit"), default=200, minimum=1)
        self.trace_preview_chars = coerce_int(cfg.get("trace_payload_preview_chars"), default=500, minimum=80)
        self.response_preview_chars = coerce_int(cfg.get("response_preview_chars"), default=240, minimum=40)
        self.return_structured_from_perform_task = coerce_bool(cfg.get("return_structured_from_perform_task"), default=True)
        self.include_trace_in_predict = coerce_bool(cfg.get("include_trace_in_predict"), default=False)

        observability = ensure_mapping(cfg.get("observability"), field_name="language_agent.observability", allow_none=True)
        self.publish_lifecycle_events = coerce_bool(observability.get("publish_lifecycle_events"), default=True)
        self.log_stage_timings = coerce_bool(observability.get("log_stage_timings"), default=True)
        self.record_component_health = coerce_bool(observability.get("record_component_health"), default=True)

        self.input_safety_depth = str(cfg.get("input_safety_depth", "balanced"))
        self.output_safety_depth = str(cfg.get("output_safety_depth", "balanced"))
        self.fail_closed_on_input_safety = coerce_bool(cfg.get("fail_closed_on_input_safety"), default=True)
        self.fail_closed_on_output_safety = coerce_bool(cfg.get("fail_closed_on_output_safety"), default=False)
        self.pass_precomputed_nlp_to_nlu = coerce_bool(cfg.get("pass_precomputed_nlp_to_nlu"), default=True)
        self.use_structured_orthography = coerce_bool(cfg.get("use_structured_orthography"), default=True)
        self.use_structured_nlp = coerce_bool(cfg.get("use_structured_nlp"), default=True)
        self.use_structured_nlg = coerce_bool(cfg.get("use_structured_nlg"), default=True)
        self.session_timeout_seconds = coerce_float(cfg.get("session_timeout_seconds"), default=1800.0, minimum=0.0)

        default_order = [stage.value for stage in StageName if stage != StageName.SHARED_MEMORY]
        self.pipeline_order = tuple(self._normalize_stage_name(stage)
            for stage in (cfg.get("pipeline_order") or default_order))
        self.component_init_order = tuple(
            str(item)
            for item in (
                cfg.get("component_init_order")
                or ["wordlist", "orthography", "grammar", "dialogue_context", "nlp", "nlu", "nlg", "safety"]
            )
        )

        self.stage_policy: Dict[str, StagePolicy] = {}
        for stage, raw_policy in ensure_mapping(cfg.get("stage_failure_policy"),
            field_name="language_agent.stage_failure_policy", allow_none=True).items():
            self.stage_policy[self._normalize_stage_name(stage)] = normalize_stage_policy(raw_policy)
        
        self.stage_enabled: Dict[str, bool] = {
            self._normalize_stage_name(stage): coerce_bool(value, default=True)
            for stage, value in ensure_mapping(cfg.get("stage_enabled"),
                field_name="language_agent.stage_enabled", allow_none=True).items()}

        # Backward-compatible booleans still work if provided.
        if "continue_on_orthography_error" in cfg:
            self.stage_policy[StageName.ORTHOGRAPHY.value] = (
                StagePolicy.CONTINUE if coerce_bool(cfg.get("continue_on_orthography_error"), default=True) else StagePolicy.FAIL
            )
        if "continue_on_grammar_error" in cfg:
            self.stage_policy[StageName.GRAMMAR.value] = (
                StagePolicy.CONTINUE if coerce_bool(cfg.get("continue_on_grammar_error"), default=True) else StagePolicy.FAIL
            )

        self.dialogue_policy = deep_merge(self.DEFAULT_POLICY, ensure_mapping(cfg.get("dialogue_policy"), field_name="language_agent.dialogue_policy", allow_none=True))
        self.component_policy = ensure_mapping(cfg.get("component_policy"), field_name="language_agent.component_policy", allow_none=True)
        self.inject_shared_nlp_into_nlu = coerce_bool(self.component_policy.get("inject_shared_nlp_into_nlu"), default=True)
        self.fail_on_missing_required_component = coerce_bool(self.component_policy.get("fail_on_missing_required_component"), default=True)

        self.shared_config = ensure_mapping(cfg.get("shared_memory"), field_name="language_agent.shared_memory", allow_none=True)
        self.record_shared_trace = coerce_bool(self.shared_config.get("record_trace"), default=True)
        self.trace_key_prefix = str(self.shared_config.get("trace_key_prefix", "language_agent:trace"))
        self.last_response_key = str(self.shared_config.get("last_response_key", "language_agent:last_response"))
        self.event_channel = str(self.shared_config.get("event_channel", "language_agent.events"))
        self.shared_ttl = none_or_float(self.shared_config.get("ttl_seconds", 86400))

    def _validate_config(self) -> None:
        valid = {stage.value for stage in StageName}
        unknown = [stage for stage in self.pipeline_order if stage not in valid]
        if unknown:
            raise LanguageAgentConfigurationError(f"Unsupported pipeline stage(s): {unknown}")
        for stage in self.stage_policy:
            if stage not in valid:
                raise LanguageAgentConfigurationError(f"Unsupported stage_failure_policy key: {stage}")
        if self.max_input_chars < 1:
            raise LanguageAgentConfigurationError("language_agent.max_input_chars must be positive.")
        if self.session_timeout_seconds < 0:
            raise LanguageAgentConfigurationError("language_agent.session_timeout_seconds cannot be negative.")

    def _initialize_components(self) -> None:
        initializers = {
            "wordlist": self._init_wordlist,
            "orthography": self._init_orthography,
            "grammar": self._init_grammar,
            "dialogue_context": self._init_dialogue_context,
            "nlp": self._init_nlp,
            "nlu": self._init_nlu,
            "nlg": self._init_nlg,
            "safety": self._init_safety,
        }
        for name in self.component_init_order:
            initializer = initializers.get(name)
            if initializer is None:
                self._record_component_status(name, False, "Unknown component initializer.")
                if self.fail_on_missing_required_component:
                    raise LanguageAgentConfigurationError(f"Unknown language component initializer: {name}")
                continue
            started = time_module.time()
            try:
                initializer()
                self._record_component_status(name, True, "initialized", started_at=started)
            except Exception as exc:
                self._record_component_status(name, False, str(exc), started_at=started, error=exc)
                logger.error("Language component initialization failed: %s: %s", name, exc, exc_info=True)
                if self.fail_on_missing_required_component:
                    raise

        if not hasattr(self, "safety_guard"):
            self.safety_guard = SafetyGuard()
        if not hasattr(self, "dialogue_context"):
            self.dialogue_context = DialogueContext()

    def _init_wordlist(self) -> None:
        self.wordlist = Wordlist()

    def _init_orthography(self) -> None:
        self.orthography_processor = OrthographyProcessor()

    def _init_grammar(self) -> None:
        self.grammar_processor = GrammarProcessor()

    def _init_dialogue_context(self) -> None:
        self.dialogue_context = DialogueContext()

    def _init_nlp(self) -> None:
        self.nlp_engine = NLPEngine()

    def _init_nlu(self) -> None:
        kwargs: Dict[str, Any] = {"wordlist_instance": getattr(self, "wordlist", None)}
        if self.inject_shared_nlp_into_nlu and hasattr(self, "nlp_engine"):
            kwargs["nlp_engine"] = self.nlp_engine
        self.nlu_engine = NLUEngine(**kwargs)
        if not getattr(self.nlu_engine, "intent_patterns", None):
            logger.warning("NLU Engine loaded no intent patterns.")
        else:
            logger.info("NLU Engine loaded %s intent patterns", len(getattr(self.nlu_engine, "intent_patterns", {})))

    def _init_nlg(self) -> None:
        self.nlg_engine = NLGEngine()

    def _init_safety(self) -> None:
        self.safety_guard = SafetyGuard()

    def _record_component_status(
        self,
        name: str,
        ok: bool,
        message: str,
        *,
        started_at: Optional[float] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        if not self.record_component_health:
            return
        now = time_module.time()
        self.component_status[name] = {
            "ok": ok,
            "message": message,
            "initialized_at": now,
            "duration_ms": int((now - started_at) * 1000) if started_at else 0,
            "error_type": type(error).__name__ if error else None,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def pipeline(self, user_input_text: str, session_id: Optional[str] = None) -> str:
        return self.process(user_input_text, session_id=session_id).response

    def process(self, user_input_text: str, session_id: Optional[str] = None, **metadata: Any) -> LanguageAgentResponse:
        if not self.enabled:
            raise LanguageAgentRuntimeError("LanguageAgent is disabled by configuration.")
        original_text = ensure_text(user_input_text)
        if len(original_text) > self.max_input_chars:
            raise LanguageAgentRuntimeError(
                f"Input exceeds max_input_chars ({len(original_text)} > {self.max_input_chars})."
            )

        trace = PipelineTrace(
            trace_id=f"lang-{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            started_at=time_module.time(),
            input_preview=compact(original_text, self.trace_preview_chars),
            metadata={"metadata": json_safe(metadata)},
        )
        artifacts = PipelineArtifacts(original_text=original_text)
        response_text = ""

        try:
            self._set_session(session_id)
            self._reset_session_if_stale(trace)

            sanitized = self._stage_safety_input(original_text, trace)
            artifacts = replace_artifacts(artifacts, sanitized_text=sanitized)
            if trace.status == PipelineStatus.BLOCKED:
                response_text = self._policy_error_response("safety_blocked")
                return self._finalize_response(response_text, artifacts, trace, PipelineStatus.BLOCKED)

            ortho_text = self._stage_orthography(sanitized, trace)
            artifacts = replace_artifacts(artifacts, orthography_text=ortho_text)

            nlp_tokens, dependencies = self._stage_nlp(ortho_text, trace)
            artifacts = replace_artifacts(artifacts, nlp_tokens=tuple(nlp_tokens), dependencies=tuple(dependencies))
            if not nlp_tokens:
                frame = self._error_frame("nlp_error", {"reason": "No tokens produced"})
                artifacts = replace_artifacts(artifacts, frame=frame)
                response_text = self._generate_response(frame, trace)
                return self._finalize_response(response_text, artifacts, trace, PipelineStatus.PARTIAL)

            grammar_result = self._stage_grammar(ortho_text, nlp_tokens, dependencies, trace)
            artifacts = replace_artifacts(artifacts, grammar_result=grammar_result)

            frame = self._stage_nlu(ortho_text, nlp_tokens, dependencies, grammar_result, trace)
            artifacts = replace_artifacts(artifacts, frame=frame)
            self._apply_dialogue_policy(frame, trace)

            self._stage_dialogue_pre_nlg(original_text, ortho_text, frame, grammar_result, trace)
            response_text = self._generate_response(frame, trace)
            response_text = self._stage_safety_output(response_text, trace)
            self._stage_dialogue_post_nlg(original_text, response_text, frame, grammar_result, trace)

            return self._finalize_response(response_text, artifacts, trace, trace.status)
        except Exception as exc:
            logger.error("LanguageAgent pipeline failed: %s", exc, exc_info=True)
            trace.warn(f"Pipeline failure: {type(exc).__name__}: {exc}")
            frame = artifacts.frame or self._error_frame("internal_error", {"detail": str(exc), "error_type": type(exc).__name__})
            artifacts = replace_artifacts(artifacts, frame=frame)
            response_text = response_text or self._policy_error_response("default_error")
            return self._finalize_response(response_text, artifacts, trace, PipelineStatus.FAILED)

    def predict(self, input_data: Any) -> Dict[str, Any]:
        text, session_id, metadata = self._extract_task_payload(input_data)
        result = self.process(text, session_id=session_id, **metadata)
        payload = result.to_dict()
        if not self.include_trace_in_predict:
            payload.get("metadata", {}).pop("trace", None)
        return payload

    def act(self, input_data: Any) -> str:
        text, session_id, metadata = self._extract_task_payload(input_data)
        return self.process(text, session_id=session_id, **metadata).response

    def perform_task(self, input_data: Any) -> Any: # type: ignore
        text, session_id, metadata = self._extract_task_payload(input_data)
        result = self.process(text, session_id=session_id, **metadata)
        return result.to_dict() if self.return_structured_from_perform_task else result.response

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------
    def _stage_safety_input(self, text: str, trace: PipelineTrace) -> str:
        stage = StageName.SAFETY_INPUT
        started = time_module.time()
        try:
            sanitized = self.safety_guard.sanitize(text, depth=self.input_safety_depth)
            self._add_stage_record(trace, stage, True, started, "Input safety sanitization complete.")
            return sanitized if sanitized is not None else text
        except Exception as exc:
            self._add_stage_record(trace, stage, False, started, "Input safety sanitization failed.", error=exc)
            if self.fail_closed_on_input_safety or self._stage_policy(stage) == StagePolicy.BLOCK:
                trace.finish(PipelineStatus.BLOCKED)
                return ""
            trace.warn("Input safety failed open by configuration.")
            return text

    def _stage_orthography(self, text: str, trace: PipelineTrace) -> str:
        stage = StageName.ORTHOGRAPHY
        if not self._stage_is_enabled(stage):
            self._add_stage_record(trace, stage, True, time_module.time(), "Orthography stage disabled.")
            return text
        started = time_module.time()
        try:
            if self.use_structured_orthography and callable(getattr(self.orthography_processor, "process_text", None)):
                result = self.orthography_processor.process_text(text)
                corrected = getattr(result, "corrected_text", None) or getattr(result, "normalized_text", None) or text
                self._add_stage_record(
                    trace,
                    stage,
                    True,
                    started,
                    "Orthography structured processing complete.",
                    metadata={"changed": bool(getattr(result, "changed", corrected != text)), "edits": len(getattr(result, "edits", ()) or ())},
                )
                return ensure_text(corrected) or text
            corrected = self.orthography_processor.batch_process(text)
            self._add_stage_record(trace, stage, True, started, "Orthography string processing complete.")
            return corrected if corrected and corrected.strip() else text
        except Exception as exc:
            self._add_stage_record(trace, stage, False, started, "Orthography failed; using sanitized text.", error=exc)
            self._handle_stage_failure(stage, exc)
            return text

    def _stage_nlp(self, text: str, trace: PipelineTrace) -> Tuple[List[Any], List[Any]]:
        stage = StageName.NLP
        started = time_module.time()
        try:
            tokens: List[Any] = []
            dependencies: List[Any] = []
            if self.use_structured_nlp and callable(getattr(self.nlp_engine, "analyze_text", None)):
                result = self.nlp_engine.analyze_text(text)
                tokens = list(getattr(result, "tokens", ()) or [])
                dependencies = list(getattr(result, "dependencies", ()) or [])
            if not tokens:
                tokens = list(self.nlp_engine.process_text(text) or [])
            if not dependencies:
                dependencies = list(self.nlp_engine.apply_dependency_rules(tokens) or [])
            self._add_stage_record(
                trace,
                stage,
                bool(tokens),
                started,
                "NLP processing complete." if tokens else "NLP produced no tokens.",
                metadata={"token_count": len(tokens), "dependency_count": len(dependencies)},
            )
            return tokens, dependencies
        except Exception as exc:
            self._add_stage_record(trace, stage, False, started, "NLP failed.", error=exc)
            self._handle_stage_failure(stage, exc)
            return [], []

    def _stage_grammar(self, text: str, tokens: Sequence[Any], dependencies: Sequence[Any], trace: PipelineTrace) -> Optional[GrammarAnalysisResult]:
        stage = StageName.GRAMMAR
        if not self._stage_is_enabled(stage):
            self._add_stage_record(trace, stage, True, time_module.time(), "Grammar stage disabled.")
            return None
        started = time_module.time()
        try:
            result: Optional[GrammarAnalysisResult]
            if callable(getattr(self.grammar_processor, "analyze_tokens", None)):
                result = self.grammar_processor.analyze_tokens(tokens, dependencies=dependencies, full_text_snippet=text)
            elif callable(getattr(self.grammar_processor, "build_input_tokens", None)):
                grammar_tokens = self.grammar_processor.build_input_tokens(tokens, dependencies=dependencies, full_text=text)
                result = self.grammar_processor.analyze_text([grammar_tokens], full_text_snippet=text)
            else:
                grammar_tokens = build_grammar_input_tokens(tokens, dependencies, text)
                result = self.grammar_processor.analyze_text([grammar_tokens], full_text_snippet=text)
            issue_count = grammar_issue_count(result)
            self._add_stage_record(
                trace,
                stage,
                True,
                started,
                "Grammar analysis complete.",
                metadata={"is_grammatical": getattr(result, "is_grammatical", None), "issue_count": issue_count},
            )
            return result
        except Exception as exc:
            self._add_stage_record(trace, stage, False, started, "Grammar failed; continuing without grammar result.", error=exc)
            self._handle_stage_failure(stage, exc)
            return None

    def _stage_nlu(
        self,
        text: str,
        tokens: Sequence[Any],
        dependencies: Sequence[Any],
        grammar_result: Optional[GrammarAnalysisResult],
        trace: PipelineTrace,
    ) -> LinguisticFrame:
        stage = StageName.NLU
        started = time_module.time()
        try:
            if self.pass_precomputed_nlp_to_nlu:
                frame = self.nlu_engine.parse(
                    text,
                    nlp_tokens=list(tokens),
                    dependencies=list(dependencies),
                    grammar_result=grammar_result,
                )
            else:
                frame = self.nlu_engine.parse(text)
            if not isinstance(frame, LinguisticFrame):
                frame = coerce_frame(frame)
            self._normalize_frame(frame)
            self._add_stage_record(
                trace,
                stage,
                True,
                started,
                "NLU parse complete.",
                metadata={"intent": frame.intent, "confidence": frame.confidence, "entity_count": len(frame.entities or {})},
            )
            return frame
        except Exception as exc:
            self._add_stage_record(trace, stage, False, started, "NLU failed; using nlu_error frame.", error=exc)
            self._handle_stage_failure(stage, exc)
            return self._error_frame("nlu_error", {"error_module": "nlu_engine", "detail": str(exc)})

    def _stage_dialogue_pre_nlg(
        self,
        original_text: str,
        processed_text: str,
        frame: LinguisticFrame,
        grammar_result: Optional[GrammarAnalysisResult],
        trace: PipelineTrace,
    ) -> None:
        stage = StageName.DIALOGUE_PRE_NLG
        started = time_module.time()
        try:
            if callable(getattr(self.dialogue_context, "record_user_turn", None)):
                self.dialogue_context.record_user_turn(
                    original_text,
                    frame=frame,
                    grammar_result=grammar_result,
                    nlu_result=None,
                    metadata={"processed_text": processed_text, "trace_id": trace.trace_id},
                )
            else:
                self._context_message("user", original_text)
                self.dialogue_context.register_intent(intent=frame.intent, confidence=frame.confidence)
                self._update_slots(frame.entities or {})
            if callable(getattr(self.dialogue_context, "prepare_for_nlg", None)):
                self.dialogue_context.prepare_for_nlg(frame=frame, grammar_result=grammar_result)
            self._add_stage_record(trace, stage, True, started, "Dialogue context prepared for NLG.")
        except Exception as exc:
            self._add_stage_record(trace, stage, False, started, "Dialogue pre-NLG update failed.", error=exc)
            self._handle_stage_failure(stage, exc)

    def _generate_response(self, frame: LinguisticFrame, trace: PipelineTrace) -> str:
        stage = StageName.NLG
        started = time_module.time()
        try:
            if self._needs_clarification(frame):
                frame = self._clarification_frame(frame)
            if self.use_structured_nlg and callable(getattr(self.nlg_engine, "generate_detailed", None)):
                result = self.nlg_engine.generate_detailed(frame=frame, context=self.dialogue_context)
                text = ensure_text(getattr(result, "text", "") or getattr(result, "response", ""))
                metadata = result.to_dict() if hasattr(result, "to_dict") else json_safe(result)
                self._add_stage_record(trace, stage, bool(text), started, "NLG detailed generation complete.", metadata={"result": compact(metadata, self.trace_preview_chars)})
                return text or self._policy_error_response("nlg_error")
            text = ensure_text(self.nlg_engine.generate(frame=frame, context=self.dialogue_context))
            self._add_stage_record(trace, stage, bool(text), started, "NLG generation complete.")
            return text or self._policy_error_response("nlg_error")
        except Exception as exc:
            self._add_stage_record(trace, stage, False, started, "NLG failed.", error=exc)
            self._handle_stage_failure(stage, exc)
            return self._policy_error_response("nlg_error")

    def _stage_safety_output(self, text: str, trace: PipelineTrace) -> str:
        stage = StageName.SAFETY_OUTPUT
        started = time_module.time()
        try:
            sanitized = self.safety_guard.sanitize(text, depth=self.output_safety_depth)
            self._add_stage_record(trace, stage, True, started, "Output safety sanitization complete.")
            return sanitized if sanitized and sanitized.strip() else text
        except Exception as exc:
            self._add_stage_record(trace, stage, False, started, "Output safety sanitization failed.", error=exc)
            if self.fail_closed_on_output_safety or self._stage_policy(stage) == StagePolicy.BLOCK:
                return self._policy_error_response("safety_blocked")
            trace.warn("Output safety failed open by configuration.")
            return text

    def _stage_dialogue_post_nlg(
        self,
        user_text: str,
        response_text: str,
        frame: LinguisticFrame,
        grammar_result: Optional[GrammarAnalysisResult],
        trace: PipelineTrace,
    ) -> None:
        stage = StageName.DIALOGUE_POST_NLG
        started = time_module.time()
        try:
            if callable(getattr(self.dialogue_context, "record_agent_turn", None)):
                self.dialogue_context.record_agent_turn(
                    response_text,
                    frame=frame,
                    metadata={"trace_id": trace.trace_id},
                )
            elif callable(getattr(self.dialogue_context, "record_pipeline_turn", None)):
                self.dialogue_context.record_pipeline_turn(
                    user_text=user_text,
                    agent_text=response_text,
                    frame=frame,
                    grammar_result=grammar_result,
                    trace_id=trace.trace_id,
                )
            else:
                self._context_message("agent", response_text)
            self._add_stage_record(trace, stage, True, started, "Dialogue context logged agent turn.")
        except Exception as exc:
            self._add_stage_record(trace, stage, False, started, "Dialogue post-NLG update failed.", error=exc)
            self._handle_stage_failure(stage, exc)

    def _stage_shared_memory(self, response: LanguageAgentResponse, trace: PipelineTrace) -> None:
        stage = StageName.SHARED_MEMORY
        if not self.record_shared_trace or self.shared_memory is None:
            return
        started = time_module.time()
        try:
            payload = response.to_dict()
            trace_payload = trace.to_dict()
            setter = getattr(self.shared_memory, "set", None) or getattr(self.shared_memory, "put", None)
            if callable(setter):
                ttl_kwargs = {"ttl": self.shared_ttl} if self.shared_ttl is not None else {}
                setter(f"{self.trace_key_prefix}:{trace.trace_id}", trace_payload, **ttl_kwargs)
                setter(self.last_response_key, payload, **ttl_kwargs)
            publisher = getattr(self.shared_memory, "publish", None)
            if callable(publisher) and self.publish_lifecycle_events:
                publisher(self.event_channel, {"event": "pipeline_completed", "trace_id": trace.trace_id, "status": response.status})
            self._add_stage_record(trace, stage, True, started, "Shared memory trace written.")
        except Exception as exc:
            self._add_stage_record(trace, stage, False, started, "Shared memory trace write failed.", error=exc)
            logger.debug("Shared-memory trace write failed: %s", exc)

    # ------------------------------------------------------------------
    # Dialogue, response, and compatibility helpers
    # ------------------------------------------------------------------
    def _apply_dialogue_policy(self, frame: LinguisticFrame, trace: PipelineTrace) -> bool:
        threshold = float(self.dialogue_policy.get("low_confidence_threshold", 0.45))
        if frame.confidence >= threshold:
            self._clear_low_confidence()
            return False
        trace.warn(f"Low confidence intent: {frame.intent} ({frame.confidence:.2f})")
        if not any(issue_description(issue) == "low_confidence_intent" for issue in getattr(self.dialogue_context, "unresolved_issues", [])):
            self._add_unresolved("low_confidence_intent")
        self._env_set("pending_intent", frame.intent)
        self._env_set("pending_entities", frame.entities or {})
        return True

    def _needs_clarification(self, frame: LinguisticFrame) -> bool:
        if frame.intent in set(self.dialogue_policy.get("clarification_triggers", [])):
            return True
        return any(issue_description(issue) == "low_confidence_intent" for issue in getattr(self.dialogue_context, "unresolved_issues", []))

    def _clarification_frame(self, frame: LinguisticFrame) -> LinguisticFrame:
        pending_intent = self._env_get("pending_intent") or frame.intent or "your request"
        pending_entities = self._env_get("pending_entities") or frame.entities or {}
        mentioned = ", ".join(f"{key}: {format_entity_value(value)}" for key, value in ensure_mapping(pending_entities, field_name="pending_entities", allow_none=True).items())
        return LinguisticFrame(
            intent="clarification_request",
            entities={"pending_intent": str(pending_intent), "mentioned_entities": mentioned or "the details provided"},
            sentiment=0.0,
            modality="interrogative",
            confidence=1.0,
            act_type=SpeechActType.DIRECTIVE,
        )

    def _finalize_response(
        self,
        text: str,
        artifacts: PipelineArtifacts,
        trace: PipelineTrace,
        status: PipelineStatus,
    ) -> LanguageAgentResponse:
        trace.finish(status if status != PipelineStatus.SUCCESS or trace.status == PipelineStatus.SUCCESS else trace.status)
        frame = artifacts.frame or self._error_frame("internal_error", {})
        grammar_ok = getattr(artifacts.grammar_result, "is_grammatical", None) if artifacts.grammar_result is not None else None
        response = LanguageAgentResponse(
            response=ensure_text(text) or self._policy_error_response("default_error"),
            confidence=float(getattr(frame, "confidence", 0.0) or 0.0),
            intent=str(getattr(frame, "intent", "unknown") or "unknown"),
            status=trace.status.value,
            trace_id=trace.trace_id,
            session_id=trace.session_id,
            frame=frame,
            grammar_ok=grammar_ok,
            metadata={"trace": trace.to_dict(), "artifacts": artifacts.to_dict(preview_chars=self.response_preview_chars)},
        )
        self.pipeline_history.append(response.to_dict())
        self._stage_shared_memory(response, trace)
        return response

    def load_dialogue_policy(self) -> Dict[str, Any]:
        return dict(self.dialogue_policy)

    def response(self) -> Dict[str, Any]:
        return self.health_check()

    def health_check(self) -> Dict[str, Any]:
        components = {
            "wordlist": hasattr(self, "wordlist"),
            "orthography_processor": hasattr(self, "orthography_processor"),
            "nlp_engine": hasattr(self, "nlp_engine"),
            "grammar_processor": hasattr(self, "grammar_processor"),
            "nlu_engine": hasattr(self, "nlu_engine"),
            "dialogue_context": hasattr(self, "dialogue_context"),
            "nlg_engine": hasattr(self, "nlg_engine"),
            "safety_guard": hasattr(self, "safety_guard"),
        }
        return {
            "ok": all(components.values()),
            "version": __version__,
            "enabled": self.enabled,
            "uptime_seconds": round(time_module.time() - self.started_at, 3),
            "history_size": len(self.pipeline_history),
            "stage_successes": dict(self.stage_successes),
            "stage_failures": dict(self.stage_failures),
            "component_status": json_safe(self.component_status),
            "components": components,
        }

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "health": self.health_check(),
            "last_trace": self.pipeline_history[-1].get("metadata", {}).get("trace") if self.pipeline_history else None,
            "history_count": len(self.pipeline_history),
            "policy": {
                "stage_failure_policy": {key: value.value for key, value in self.stage_policy.items()},
                "stage_enabled": dict(self.stage_enabled),
                "pass_precomputed_nlp_to_nlu": self.pass_precomputed_nlp_to_nlu,
            },
        }

    def clear_context(self) -> bool:
        if callable(getattr(self.dialogue_context, "clear", None)):
            self.dialogue_context.clear()
            return True
        return False

    def _set_session(self, session_id: Optional[str]) -> None:
        if session_id and self._env_get("session_id") != session_id:
            self._env_set("session_id", session_id)

    def _context_message(self, role: str, content: str) -> None:
        try:
            self.dialogue_context.add_message(role=role, content=content)
        except Exception as exc:
            logger.debug("DialogueContext.add_message failed: %s", exc)

    def _update_slots(self, entities: Mapping[str, Any]) -> None:
        for key, value in entities.items():
            item = value[0] if isinstance(value, list) and value else value
            if item is None or (isinstance(item, str) and not item.strip()):
                continue
            try:
                self.dialogue_context.update_slot(key, item)
            except Exception as exc:
                logger.debug("DialogueContext.update_slot failed for %s: %s", key, exc)

    def _env_get(self, key: str, default: Any = None) -> Any:
        getter = getattr(self.dialogue_context, "get_environment_state", None)
        if callable(getter):
            try:
                return getter(key)
            except TypeError:
                state = getter()
                return state.get(key, default) if isinstance(state, Mapping) else default
        return default

    def _env_set(self, key: str, value: Any) -> None:
        setter = getattr(self.dialogue_context, "update_environment_state", None)
        if callable(setter):
            try:
                setter(key, value)
            except Exception as exc:
                logger.debug("DialogueContext.update_environment_state failed for %s: %s", key, exc)

    def _add_unresolved(self, issue: str, slot: Optional[str] = None) -> None:
        method = getattr(self.dialogue_context, "add_unresolved", None)
        if callable(method):
            try:
                method(issue=issue, slot=slot)
            except TypeError:
                method(issue, slot)

    def _clear_low_confidence(self) -> None:
        issues = getattr(self.dialogue_context, "unresolved_issues", None)
        if isinstance(issues, list):
            self.dialogue_context.unresolved_issues = [issue for issue in issues if issue_description(issue) != "low_confidence_intent"]
        self._env_set("pending_intent", None)
        self._env_set("pending_entities", {})

    def _reset_session_if_stale(self, trace: PipelineTrace) -> None:
        if self.session_timeout_seconds <= 0:
            return
        getter = getattr(self.dialogue_context, "get_time_since_last_interaction", None)
        if callable(getter):
            try:
                if getter() > self.session_timeout_seconds and callable(getattr(self.dialogue_context, "clear", None)):
                    self.dialogue_context.clear()
                    trace.warn("Dialogue context was cleared after session timeout.")
            except Exception as exc:
                logger.debug("Session staleness check failed: %s", exc)

    def _normalize_frame(self, frame: LinguisticFrame) -> None:
        if frame.intent == "get_time":
            frame.intent = "time_request"
        if frame.intent == "get_date":
            frame.intent = "date_request"
        if frame.intent == "ask_definition":
            frame.intent = "definition_request"
        if frame.intent == "greetings":
            frame.intent = "greeting"
        if frame.intent == "affirm":
            frame.intent = "affirmation"
        if frame.intent == "deny":
            frame.intent = "denial"
        if frame.intent == "time_request" and "time" not in (frame.entities or {}):
            frame.entities["time"] = "current system time"

    def _error_frame(self, intent: str, entities: Mapping[str, Any]) -> LinguisticFrame:
        return LinguisticFrame(
            intent=intent,
            entities=dict(entities),
            sentiment=0.0,
            modality="error",
            confidence=1.0,
            act_type=SpeechActType.ASSERTIVE,
        )

    def _policy_error_response(self, key: str) -> str:
        errors = ensure_mapping(self.dialogue_policy.get("error_responses"), field_name="dialogue_policy.error_responses", allow_none=True)
        return str(errors.get(key) or errors.get("default_error") or self.DEFAULT_POLICY["error_responses"]["default_error"])

    def _stage_is_enabled(self, stage: StageName) -> bool:
        return coerce_bool(self.stage_enabled.get(stage.value, True), default=True)

    def _stage_policy(self, stage: StageName) -> StagePolicy:
        return self.stage_policy.get(stage.value, StagePolicy.CONTINUE)

    def _handle_stage_failure(self, stage: StageName, exc: BaseException) -> None:
        policy = self._stage_policy(stage)
        if policy in {StagePolicy.FAIL, StagePolicy.BLOCK}:
            raise exc

    def _add_stage_record(
        self,
        trace: PipelineTrace,
        stage: StageName,
        ok: bool,
        started: float,
        message: Optional[str] = None,
        *,
        error: Optional[BaseException] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        finished = time_module.time()
        if ok:
            self.stage_successes[stage.value] += 1
        else:
            self.stage_failures[stage.value] += 1
        record = StageRecord(
            name=stage.value,
            ok=ok,
            started_at=started,
            finished_at=finished,
            duration_ms=int((finished - started) * 1000),
            policy=self._stage_policy(stage).value,
            message=message,
            error_type=type(error).__name__ if error else None,
            error_message=str(error) if error else None,
            metadata=dict(metadata or {}),
        )
        trace.add(record)
        if self.log_stage_timings:
            level = logger.info if ok else logger.warning
            level("LanguageAgent stage=%s ok=%s duration_ms=%s message=%s", stage.value, ok, record.duration_ms, message)

    def _publish_event(self, event: str, payload: Mapping[str, Any]) -> None:
        if not self.publish_lifecycle_events or self.shared_memory is None:
            return
        publisher = getattr(self.shared_memory, "publish", None)
        if callable(publisher):
            try:
                publisher(self.event_channel, {"event": event, "payload": json_safe(payload), "timestamp": time_module.time()})
            except Exception as exc:
                logger.debug("LanguageAgent lifecycle publish failed: %s", exc)

    def _extract_task_payload(self, input_data: Any) -> Tuple[str, Optional[str], Dict[str, Any]]:
        if isinstance(input_data, str):
            return input_data, None, {}
        if isinstance(input_data, Mapping):
            for key in ("text", "query", "input", "message", "content", "payload"):
                value = input_data.get(key)
                if isinstance(value, str):
                    metadata = {str(k): v for k, v in input_data.items() if k not in {key, "session_id"}}
                    return value, input_data.get("session_id"), metadata
            return str(dict(input_data)), input_data.get("session_id"), {}
        return str(input_data), None, {}
    
    def _normalize_stage_name(self, stage: Any) -> str:
        value = str(stage or "").strip()
        return _STAGE_ALIASES.get(value, value)


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------
def compact(value: Any, max_length: int = 240) -> str:
    try:
        return compact_text(value, max_length=max_length)  # type: ignore[name-defined]
    except Exception:
        text = ensure_text(value)
        return text if len(text) <= max_length else text[: max_length - 3] + "..."


def grammar_summary(result: Optional[GrammarAnalysisResult]) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    return {
        "is_grammatical": getattr(result, "is_grammatical", None),
        "issue_count": grammar_issue_count(result),
        "sentence_count": len(getattr(result, "sentence_analyses", ()) or ()),
    }


def grammar_issue_count(result: Optional[GrammarAnalysisResult]) -> int:
    if result is None:
        return 0
    issues = getattr(result, "issues", None)
    if issues is not None:
        return len(issues)
    count = 0
    for sentence in getattr(result, "sentence_analyses", ()) or ():
        if isinstance(sentence, Mapping):
            count += len(sentence.get("issues", ()) or ())
    return count


def frame_to_dict(frame: Optional[LinguisticFrame]) -> Optional[Dict[str, Any]]:
    if frame is None:
        return None
    if hasattr(frame, "to_dict") and callable(getattr(frame, "to_dict")):
        return frame.to_dict()
    return {
        "intent": getattr(frame, "intent", None),
        "entities": json_safe(getattr(frame, "entities", {})),
        "sentiment": getattr(frame, "sentiment", None),
        "modality": getattr(frame, "modality", None),
        "confidence": getattr(frame, "confidence", None),
        "act_type": getattr(getattr(frame, "act_type", None), "value", getattr(frame, "act_type", None)),
    }


def coerce_frame(value: Any) -> LinguisticFrame:
    if isinstance(value, LinguisticFrame):
        return value
    if isinstance(value, Mapping):
        return LinguisticFrame(
            intent=str(value.get("intent", "unknown")),
            entities=dict(value.get("entities", {}) or {}),
            sentiment=float(value.get("sentiment", 0.0) or 0.0),
            modality=str(value.get("modality", "unknown")),
            confidence=float(value.get("confidence", 0.0) or 0.0),
            act_type=value.get("act_type", SpeechActType.ASSERTIVE),
        )
    return LinguisticFrame("unknown", {}, 0.0, "unknown", 0.0, SpeechActType.ASSERTIVE)


def build_grammar_input_tokens(tokens: Sequence[Any], dependencies: Sequence[Any], text: str) -> List[Any]:
    from .language.grammar_processor import InputToken as GrammarInputToken

    dep_by_child: Dict[int, Any] = {}
    for relation in dependencies:
        child = getattr(relation, "dependent_index", None)
        if child is not None and int(child) not in dep_by_child:
            dep_by_child[int(child)] = relation

    output: List[Any] = []
    cursor = 0
    for position, token in enumerate(tokens):
        token_text = ensure_text(getattr(token, "text", token))
        token_index = int(getattr(token, "index", position))
        relation = dep_by_child.get(token_index)
        head = token_index
        dep = "root"
        if relation is not None:
            dep = ensure_text(getattr(relation, "relation", "dep"))
            raw_head = getattr(relation, "head_index", token_index)
            raw_head_text = getattr(relation, "head", "")
            head = token_index if raw_head_text == "ROOT" and dep == "root" else int(raw_head)
        start = getattr(token, "start_char", getattr(token, "start_char_abs", None))
        end = getattr(token, "end_char", getattr(token, "end_char_abs", None))
        if start is None or end is None:
            found = text.find(token_text, cursor)
            start = found if found >= 0 else cursor
            end = int(start) + len(token_text) - 1
            cursor = int(end) + 1
        output.append(
            GrammarInputToken(
                text=token_text,
                lemma=ensure_text(getattr(token, "lemma", token_text)).lower(),
                pos=ensure_text(getattr(token, "pos", getattr(token, "upos", "X"))).upper(),
                index=token_index,
                head=head,
                dep=dep,
                start_char_abs=int(start),
                end_char_abs=int(end),
                upos=getattr(token, "upos", None),
                xpos=getattr(token, "xpos", None),
                morphology=dict(getattr(token, "morphology", getattr(token, "feats", {})) or {}),
            )
        )
    return output


def issue_description(issue: Any) -> str:
    if isinstance(issue, Mapping):
        return str(issue.get("description") or issue.get("issue") or issue.get("code") or "")
    return str(getattr(issue, "description", getattr(issue, "issue", getattr(issue, "code", ""))))


def format_entity_value(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return ", ".join(str(item) for item in value if item is not None)
    return str(value)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base or {})
    for key, value in dict(override or {}).items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def none_or_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def normalize_stage_policy(value: Any) -> StagePolicy:
    normalized = str(value or StagePolicy.CONTINUE.value).strip().lower()
    if normalized == StagePolicy.FAIL.value:
        return StagePolicy.FAIL
    if normalized == StagePolicy.BLOCK.value:
        return StagePolicy.BLOCK
    return StagePolicy.CONTINUE


def replace_artifacts(artifacts: PipelineArtifacts, **changes: Any) -> PipelineArtifacts:
    data = {
        "original_text": artifacts.original_text,
        "sanitized_text": artifacts.sanitized_text,
        "orthography_text": artifacts.orthography_text,
        "nlp_tokens": artifacts.nlp_tokens,
        "dependencies": artifacts.dependencies,
        "grammar_result": artifacts.grammar_result,
        "frame": artifacts.frame,
    }
    data.update(changes)
    return PipelineArtifacts(**data)



if __name__ == "__main__":
    print("\n=== Running Language Agent ===\n")
    from .collaborative.shared_memory import SharedMemory
    from .agent_factory import AgentFactory

    shared_memory = SharedMemory()
    agent_factory = AgentFactory()

    try:
        language_agent = LanguageAgent(
            shared_memory=shared_memory,
            agent_factory=agent_factory,
            config=None
        )
        print("Language Agent initialized. Type your message below.")
        print("Type 'exit' or 'quit' to end the session.\n")

        session_id = f"interactive-session-{time_module.time()}"

        while True:
            try:
                user_input = input("User: ").strip()
                if user_input.lower() in {"exit", "quit"}:
                    print("Exiting Language Agent. Goodbye!")
                    break
                if not user_input:
                    print("Agent: Please say something.\n")
                    continue

                # Using perform_task as the entry point
                response = language_agent.pipeline(user_input_text=user_input, session_id=session_id)
                # Or directly call pipeline:
                # response = language_agent.pipeline(user_input, session_id=session_id)

                print(f"Agent: {response}\n")

            except KeyboardInterrupt:
                print("\n[Interrupted] Exiting Language Agent.")
                break
            except Exception as e:
                logger.error(f"Error in interactive loop: {e}", exc_info=True)
                print(f"[Error] Something went wrong: {e}")
                # break # Optional: break on error or allow continuation

    except Exception as e:
        logger.error(f"Failed to initialize LanguageAgent: {e}", exc_info=True)
        print(f"[Fatal Error] Could not start Language Agent: {e}")


#if __name__ == "__main__":
#    print("\n=== Running  Language agent ===\n")
#    printer.status("TEST", " Language agent initialized", "info")
#    from .collaborative.shared_memory import SharedMemory

#    def agent_factory_stub(name: str, cfg: Optional[Mapping[str, Any]] = None) -> Any:
#        return None

#    shared_memory_instance = SharedMemory()
#    language_agent = LanguageAgent(shared_memory=shared_memory_instance, agent_factory=agent_factory_stub)

#    session = f"language-agent-test-{int(time_module.time())}"
#    samples = ["hello there", "what time is it", "define recursion"]
#    for sample in samples:
#        result = language_agent.process(sample, session_id=session)
#        print(f"User: {sample}")
#        print(f"Agent: {result.response}")
#        print(f"Intent: {result.intent} | Confidence: {result.confidence:.2f} | Trace: {result.trace_id}\n")

#    health = language_agent.health_check()
#    assert health["ok"] is True
#    assert len(language_agent.pipeline_history) >= 1

#    print("\n=== Test ran successfully ===\n")
