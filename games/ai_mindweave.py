"""Mindweave AI initialization module."""

# ai_mindaweave.py integrates Collaborative (1), Knowledge (2), Planning (3), Evaluation (4), Language (5), Reasoning (6), and Safety(7) agents
# 1. Register specialist agents in the Collaboration Manager / task router.
#    Route by task_type (cognitive_puzzle, npc_dialogue, stress_event, debrief_reflection, safety_audit).
# 2. Store faction relations, dialogue outcomes, and player behavior signatures in knowledge memory/cache.
#    Use ontology manager to keep narrative entities and constraints coherent.
# 3. Use planning memory + probabilistic planner for quest-state transitions.
#    Feed real-time game telemetry (success rate, latency, retries) to planning heuristics.
# 4. Evaluation agent tracks quality, drift, and regressions over time.
# 5. Use NLU for intent/emotion parsing and NLG for contextual NPC responses.
#    Add rubric-based scoring (validation, perspective-taking, de-escalation, repair attempts).
# 6. Use rule/probabilistic reasoning modules to generate and validate procedural logic gate puzzles.
#    Attach solver traces so post-level debrief can explain why an answer path worked.
# 7. Safety checks on every high-impact interaction.
#
# This set gives immediate coverage for adaptive cognitive challenges + LLM NPC interaction + risk controls, without waiting on full multimodal hardware dependencies.
#
#
# AI agents to Mindweave mechanics
# -  Supply Chain Puzzles: Planning + Reasoning + Knowledge.
# -  Procedural Logic Gates: Reasoning + Learning (not yet).
# -  Micro-Expression Interrogations: Perception (not yet) + Language + Safety.
# -  Empathy Bridge: Language + Knowledge + Alignment (not yet).
# -  Biometric Regulation: Perception (not yet) + Execution (not yet) + Safety.
# -  Metacognitive Debriefing: Language + Knowledge + Evaluation.
# 
# 
# Practical integration contract (event schema)
# Define a common event envelope all agents can consume:
# - `session_id`
# - `player_id`
# - `task_type`
# - `game_state_snapshot`
# - `telemetry` (reaction time, retries, confidence, HRV if present)
# - `safety_context`
# - `requested_action`
# 
# This lets CollaborativeAgent route consistently and allows Safety/Evaluation to run as universal middleware.

from __future__ import annotations

import sys, os
import json
import math
import random
import threading
import requests

from http.server import BaseHTTPRequestHandler, HTTPServer
from dataclasses import dataclass, field
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Add repository root to sys.path so shared src/ and logs/ imports resolve
games_root = Path(__file__).resolve().parent
project_root = games_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..src.agents.agent_factory import AgentFactory
from ..src.agents.collaborative_agent import CollaborativeAgent
from ..src.agents.collaborative.shared_memory import SharedMemory
from ..src.agents.planning.planning_types import Task, TaskType
from ..logs.logger import get_logger, PrettyPrinter

logger = get_logger("Project: Mindweave")
printer = PrettyPrinter()
RESPONSE_TEMPLATE_PATH = games_root / "mindweave" / "templates" / "responses.JSON"

class AgentAdapter:
    """Normalizes heterogeneous agent APIs to a single execute(task_data) contract."""

    def __init__(
        self,
        name: str,
        agent: Any,
        handlers: list[Callable[[dict[str, Any]], Any]],
        capabilities: list[str] | None = None,
    ):
        self.name = name
        self.agent = agent
        self.handlers = handlers
        # Collaboration registry validates capabilities on the *instance* itself.
        # Keep this attribute explicit so adapter-based registrations remain valid.
        self.capabilities = list(capabilities or [])

    def execute(self, task_data: dict[str, Any]) -> dict[str, Any]:
        for handler in self.handlers:
            try:
                output = handler(task_data)
                return {"agent": self.name, "result": output}
            except Exception as exc:
                logger.debug("%s handler failed: %s", self.name, exc)

        raise RuntimeError(f"No compatible execution path found for {self.name}")

@dataclass
class MindweaveAI:
    game: str = "mindweave"
    initialized_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self) -> None:
        self.shared_memory = SharedMemory()
        self.factory = AgentFactory()
        self.collab = CollaborativeAgent(shared_memory=self.shared_memory, agent_factory=self.factory)

        self.knowledge_agent = self.factory.create("knowledge", self.shared_memory)
        self.planning_agent = self.factory.create("planning", self.shared_memory)
        self.safety_agent = self.factory.create("safety", self.shared_memory)
        self.language_agent = self.factory.create("language", self.shared_memory)
        self.reasoning_agent = self.factory.create("reasoning", self.shared_memory)
        self.evaluation_agent = self.factory.create("evaluation", self.shared_memory)

        self._register_task_routes()
        self._planning_task_registered = False
        self._planning_enabled = True
        self.response_templates = self._load_response_templates()
        self.match_log_path = project_root / 'logs' / 'mindweave.jsonl'
        self.shared_memory.set("mindweave_ai_status", "initialized")
        logger.info("Project: Mindweave AI initialized with Knowledge, Planning, Evaluation, Language, and Safety agents")

    def _load_response_templates(self) -> dict[str, list[dict[str, str]]]:
        try:
            with RESPONSE_TEMPLATE_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to load Mindweave response templates: %s", exc)
        return {}

    def _pick_response(self, group: str, fallback: str, fallback_voice: str | None = None) -> tuple[str, str | None]:
        options = self.response_templates.get(group, [])
        if isinstance(options, list) and options:
            candidate = random.choice(options)
            if isinstance(candidate, dict):
                text = str(candidate.get("text", fallback)).strip() or fallback
                voice = candidate.get("voice") if isinstance(candidate.get("voice"), str) else fallback_voice
                return text, voice
        return fallback, fallback_voice

    def _register_task_routes(self) -> None:
        manager = self.collab.collaboration_manager
        if manager is None:
            logger.warning("Collaboration manager unavailable. Falling back to local handlers.")
            return

        routes = {
            "cognitive_puzzle": (self.reasoning_agent, [self._execute_reasoning, self._execute_planning]),
            "npc_dialogue": (self.language_agent, [self._execute_language]),
            "stress_event": (self.safety_agent, [self._execute_safety]),
            "debrief_reflection": (self.evaluation_agent, [self._execute_evaluation, self._execute_language]),
            "safety_audit": (self.safety_agent, [self._execute_safety]),
        }

        for task_type, (agent, handlers) in routes.items():
            adapter_name = f"mindweave_{task_type}"
            adapter = AgentAdapter(
                name=adapter_name,
                agent=agent,
                handlers=handlers,
                capabilities=[task_type],
            )
            manager.register_agent(adapter_name, adapter, capabilities=[task_type])

        logger.info("Mindweave collaboration routes registered for %d task types", len(routes))

    def _build_event_envelope(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = datetime.utcnow().isoformat()
        task_type = payload.get("task_type", "cognitive_puzzle")
        command = payload.get("command")
        if command is None:
            command = "execute_plan" if task_type == "cognitive_puzzle" else "chat"
        return {
            "session_id": payload.get("session_id", "local-session"),
            "player_id": payload.get("player_id", "player-unknown"),
            "task_type": task_type,
            "game_state_snapshot": payload.get("game_state_snapshot", payload.get("game_state", {})),
            "telemetry": payload.get("telemetry", {}),
            "safety_context": payload.get("safety_context", {}),
            "requested_action": payload.get("requested_action", "analyze"),
            "command": command,
            "timestamp": now,
        }

    def _route_event(self, event: dict[str, Any]) -> dict[str, Any]:
        task_type = event["task_type"]
        manager = self.collab.collaboration_manager
        if manager is None:
            return {"status": "fallback", "task_type": task_type}

        try:
            result = manager.run_task(task_type, event, retries=1)
            return {"status": "routed", "task_type": task_type, "result": result}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Route failed for task_type=%s: %s", task_type, exc)
            return {"status": "route_error", "task_type": task_type, "error": str(exc)}

    def _execute_knowledge(self, event: dict[str, Any]) -> Any:
        context = event.get("requested_action", "mindweave")
        query_fn = getattr(self.knowledge_agent, "query", None)
        if callable(query_fn):
            return query_fn(context)
        return {"note": "knowledge agent query unavailable"}

    def _execute_planning(self, event: dict[str, Any]) -> Any:
        if not self._planning_enabled:
            return {"note": "planning disabled"}

        fallback_task = Task(
            name="mindweave_route_fallback",
            task_type=TaskType.PRIMITIVE,
            start_time=10,
            deadline=120,
            duration=10,
        )
        goal_task = Task(
            name=f"mindweave_{event.get('task_type', 'task')}",
            task_type=TaskType.ABSTRACT,
            methods=[[fallback_task]],
            goal_state={"resolved": True},
            context=event,
        )

        if not self._planning_task_registered and hasattr(self.planning_agent, "register_task"):
            self.planning_agent.register_task(goal_task)
            self._planning_task_registered = True

        plan = self.planning_agent.generate_plan(goal_task)
        if not isinstance(plan, list):
            self._planning_enabled = False
            return {"note": "planning unavailable", "plan": None}
        return {"plan_steps": len(plan), "plan": plan}

    def _execute_evaluation(self, event: dict[str, Any]) -> Any:
        evaluate_fn = getattr(self.evaluation_agent, "evaluate", None)
        if callable(evaluate_fn):
            return evaluate_fn(event)
        return {"note": "evaluation method unavailable"}

    def _is_system_fallback_payload(self, payload: Any) -> bool:
        if isinstance(payload, str):
            marker = payload.strip()
            return marker.startswith("[Fallback]") or "Unsupported command for PlanningAgent" in marker
        if isinstance(payload, dict):
            text_fields = [payload.get(key) for key in ("reply", "response", "text", "message", "data")]
            for value in text_fields:
                if isinstance(value, str) and self._is_system_fallback_payload(value):
                    return True
        return False

    def _has_dialogue_payload(self, payload: Any) -> bool:
        if isinstance(payload, str):
            return bool(payload.strip()) and not self._is_system_fallback_payload(payload)
        if isinstance(payload, dict):
            for key in ("reply", "response", "text", "message", "data"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip() and not self._is_system_fallback_payload(value):
                    return True
        return False

    def _execute_language(self, event: dict[str, Any]) -> Any:
        sanitized_event = dict(event)
        if sanitized_event.get("command") == "execute_plan":
            # Language tasks should not be forced through planning command handlers.
            sanitized_event["command"] = "chat"

        # Some language-agent variants expose planning-centric execute contracts.
        # Prefer explicit language methods first so NPC dialogue does not fail on
        # "Unsupported command for PlanningAgent: 'chat'".
        for method_name in ("chat", "generate", "respond", "process"):
            language_fn = getattr(self.language_agent, method_name, None)
            if callable(language_fn):
                result = language_fn(sanitized_event)
                if self._is_system_fallback_payload(result):
                    return {
                        "status": "fallback",
                        "response": "I am present and listening. Share your objective, and I will architect the next move.",
                        "reason": "language contract returned internal fallback payload",
                    }
                return result

        execute_fn = getattr(self.language_agent, "execute", None)
        execute_impl = getattr(execute_fn, "__func__", execute_fn)
        execute_module = getattr(execute_impl, "__module__", "")
        execute_qualname = getattr(execute_impl, "__qualname__", "")
        execute_is_base = execute_module.endswith("base_agent") and execute_qualname.endswith("BaseAgent.execute")

        # Only use execute() when the language agent provides a dialogue-aware
        # override. Skipping base-agent execute avoids routing chat through generic
        # prediction/planning internals.
        if callable(execute_fn) and not execute_is_base:
            try:
                result = execute_fn(sanitized_event)
            except ValueError as exc:
                if "Unsupported command for PlanningAgent" in str(exc):
                    return {
                        "status": "fallback",
                        "response": "I am present and listening. Share your objective, and I will architect the next move.",
                        "reason": "language agent execute path expects planning commands",
                    }
                logger.warning("Language execute() failed for chat payload: %s", exc)
                return {
                    "status": "fallback",
                    "response": "Architect link unstable. Restate your objective and constraints.",
                    "reason": "language execute path raised ValueError",
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("Language execute() raised unexpected error for chat payload: %s", exc)
                return {
                    "status": "fallback",
                    "response": "Architect link unstable. Restate your objective and constraints.",
                    "reason": "language execute path raised exception",
                }

            if isinstance(result, dict) and result.get("status") == "AWAITING_PLAN":
                return {
                    "status": "fallback",
                    "response": "I am present and listening. Share your objective, and I will architect the next move.",
                    "reason": result.get("message", "language agent returned planning placeholder"),
                }

            if self._is_system_fallback_payload(result):
                return {
                    "status": "fallback",
                    "response": "I am present and listening. Share your objective, and I will architect the next move.",
                    "reason": "language execute path returned internal fallback payload",
                }

            if not self._has_dialogue_payload(result):
                return {
                    "status": "fallback",
                    "response": "Architect link established. State your goal and constraints so I can plan the next move.",
                    "reason": "language execute path returned non-dialogue payload",
                }

            return result

        return {
            "status": "fallback",
            "response": "Architect link established. State your goal and constraints so I can plan the next move.",
            "reason": "no language dialogue handler available",
        }

    def _extract_language_reply(self, route_result: dict[str, Any], fallback_message: str) -> str:
        if route_result.get("status") != "routed":
            return fallback_message

        result = route_result.get("result", {})
        if isinstance(result, dict):
            payload = result.get("result", result)
            if self._is_system_fallback_payload(payload):
                return fallback_message
            if isinstance(payload, str) and payload.strip():
                return payload.strip()
            if isinstance(payload, dict):
                for key in ("reply", "response", "text", "message", "data"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip() and not self._is_system_fallback_payload(value):
                        return value.strip()

        return fallback_message

    def _execute_reasoning(self, event: dict[str, Any]) -> Any:
        reason_fn = getattr(self.reasoning_agent, "infer", None)
        if callable(reason_fn):
            return reason_fn(event)
        execute_fn = getattr(self.reasoning_agent, "execute", None)
        if callable(execute_fn):
            return execute_fn(event)
        return {"note": "reasoning method unavailable"}

    def _execute_safety(self, event: dict[str, Any]) -> Any:
        risk_score = float(event.get("safety_context", {}).get("risk_score", 0.0) or 0.0)
        assessment = self.collab.assess_risk(risk_score=risk_score, task_type=event.get("task_type", "general"))
        return assessment.to_dict()

    def health(self) -> dict[str, Any]:
        return {"agent_status": "ready", "initialized_at": self.initialized_at}

    def get_move(self, game_state: dict[str, Any]) -> dict[str, Any] | None:
        event = self._build_event_envelope(
            {
                "task_type": "cognitive_puzzle",
                "game_state_snapshot": game_state,
                "requested_action": "suggest_move",
            }
        )
        route_result = self._route_event(event)
        self.shared_memory.set("mindweave:last_route", route_result)

        valid_moves = game_state.get("validMoves", []) if isinstance(game_state, dict) else []
        return valid_moves[0] if valid_moves else None

    def learn_from_game(self, payload: dict[str, Any]) -> bool:
        event = self._build_event_envelope(payload if isinstance(payload, dict) else {})
        route_result = self._route_event(event)
        self.shared_memory.set("mindweave:last_learning_event", route_result)
        return True

    def _infer_provider_from_api_key(self, api_key: str) -> str:
        key = api_key.strip()
        if key.startswith("sk-ant-"):
            return "anthropic"
        return "openai"

    def _generate_external_llm_reply(self, api_key: str, message: str, event: dict[str, Any]) -> tuple[str | None, str | None]:
        key = (api_key or "").strip()
        if not key:
            return None, None

        provider = self._infer_provider_from_api_key(key)
        timeout_seconds = float(os.getenv("MINDWEAVE_LLM_TIMEOUT", "20"))

        try:
            if provider == "anthropic":
                model = os.getenv("MINDWEAVE_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
                response = requests.post(
                    os.getenv("MINDWEAVE_ANTHROPIC_CHAT_URL", "https://api.anthropic.com/v1/messages"),
                    headers={
                        "x-api-key": key,
                        "anthropic-version": os.getenv("MINDWEAVE_ANTHROPIC_VERSION", "2023-06-01"),
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": int(os.getenv("MINDWEAVE_ANTHROPIC_MAX_TOKENS", "220")),
                        "system": "You are Architect-7 from Mindweave. Keep responses concise, actionable, and supportive.",
                        "messages": [{"role": "user", "content": [{"type": "text", "text": message}]}],
                    },
                    timeout=timeout_seconds,
                )
                if response.status_code != 200:
                    response_excerpt = (response.text or "").strip().replace("\n", " ")[:300]
                    logger.warning(
                        "Anthropic chat call failed with status=%s body=%s",
                        response.status_code,
                        response_excerpt or "<empty>",
                    )
                return None, provider

            model = os.getenv("MINDWEAVE_OPENAI_MODEL", "gpt-4o-mini")
            response = requests.post(
                os.getenv("MINDWEAVE_OPENAI_CHAT_URL", "https://api.openai.com/v1/chat/completions"),
                headers={
                    "Authorization": f"Bearer {key}",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "temperature": 0.6,
                    "messages": [
                        {"role": "system", "content": "You are Architect-7 from Mindweave. Keep responses concise, actionable, and supportive."},
                        {"role": "user", "content": message},
                    ],
                },
                timeout=timeout_seconds,
            )
            if response.status_code != 200:
                logger.warning("OpenAI-compatible chat call failed with status=%s", response.status_code)
                return None, provider
            data = response.json()
            choices = data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                content = (((choices[0].get("message") or {}).get("content")) or "").strip()
                return content or None, provider
            return None, provider
        except requests.RequestException as exc:
            logger.warning("External LLM call failed for provider=%s: %s", provider, exc)
            return None, provider

    def _build_coaching_hint(self, message: str, telemetry: dict[str, Any], task_type: str) -> str | None:
        lower_text = message.lower()
        phase = str(telemetry.get("phase", "")).lower()
        if task_type == "debrief_reflection":
            return "Language Agent: Include (1) your strategy, (2) one adaptation moment, and (3) where you will use it in real teamwork."
        if any(token in lower_text for token in ("stuck", "confused", "hard", "help", "dont get", "don't get")):
            return "Language Agent: Try this loop: restate the target, eliminate one wrong option, then test one answer before you commit."
        if any(token in lower_text for token in ("stress", "anxious", "panic", "overwhelm", "frustrat")):
            return "Language Agent: Regulate first—one slow breath, name the pressure, then issue one calm and specific instruction."
        if phase == "eq":
            return "Language Agent: In EQ mode, use validation + boundary + next step. Example: 'I hear the load; let's stabilize one subsystem now.'"
        if phase == "iq":
            return "Language Agent: In IQ mode, prioritize sequence accuracy over speed, and verbally check the rule before submitting."
        return None

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        message = str(payload.get("message", "")).strip()
        if not message:
            return {"error": "message is required"}

        event = self._build_event_envelope(
            {
                "session_id": payload.get("session_id"),
                "player_id": payload.get("player_id"),
                "task_type": payload.get("task_type", "npc_dialogue"),
                "game_state_snapshot": payload.get("game_state_snapshot", {}),
                "telemetry": payload.get("telemetry", {}),
                "safety_context": payload.get("safety_context", {}),
                "requested_action": message,
            }
        )
        api_key = str(payload.get("api_key", "") or "").strip()
        llm_reply, llm_provider = self._generate_external_llm_reply(api_key, message, event)
        route_result = {"status": "llm_direct", "task_type": event.get("task_type", "npc_dialogue")} if llm_reply else self._route_event(event)

        lower_text = message.lower()
        task_type = event.get("task_type", "npc_dialogue")
        telemetry = event.get("telemetry", {}) if isinstance(event.get("telemetry"), dict) else {}
        coaching_hint = self._build_coaching_hint(message, telemetry, task_type)

        if llm_reply and task_type == "npc_dialogue":
            response = llm_reply
            voice = "../src/audio/A7_link_established.m4a"
            emotion, analysis, eq_delta = "calm", f"LLM Dialogue / {llm_provider or 'unknown'}", 5
        elif task_type == "debrief_reflection":
            if any(token in lower_text for token in ("strategy", "reflect", "transfer", "real-world", "collaboration", "adapt")):
                response, voice = self._pick_response(
                    "debrief",
                    "Debrief accepted. You demonstrated metacognitive transfer and emotional regulation insight.",
                    "../src/audio/A7_final_debrief.m4a",
                )
                emotion, analysis, eq_delta = "calm", "Reflective / Integrated", 4
            else:
                response, voice = self._pick_response(
                    "chat_error",
                    "Debrief received, but include explicit strategy and transfer language for stronger consolidation.",
                    "../src/audio/A7_chat_error.m4a",
                )
                emotion, analysis, eq_delta = "neutral", "Reflection Shallow", 0
        elif task_type == "cognitive_puzzle":
            response, voice = self._pick_response(
                "IQ",
                "Cognitive route validated. Prioritize chunking, pattern rehearsal, and error correction loops.",
                "../src/audio/A7_thinking_respose.m4a",
            )
            emotion, analysis, eq_delta = "thinking", "Executive Processing", 1
        elif any(token in lower_text for token in ("understand", "help", "calm", "support", "hear", "hi", "hello", "architect", "plan", "strategy")):
            planned_dialogue = self._execute_planning({**event, "task_type": "npc_dialogue", "command": "create_plan"})
            fallback_response, voice = self._pick_response(
                "calm",
                "Architect link established. I can help you map intent, constraints, and next actions—what do you need solved?",
                "../src/audio/A7_link_established.m4a",
            )
            response = llm_reply or self._extract_language_reply(route_result, fallback_response)
            if isinstance(planned_dialogue, dict):
                plan_steps = planned_dialogue.get("plan_steps")
                if isinstance(plan_steps, int):
                    response = f"{response} [AI planner staged: {plan_steps} step(s)]"
            emotion, analysis, eq_delta = "calm", "Architect Bridge / Engaged", 5
        elif any(token in lower_text for token in ("hurry", "now", "fix")):
            response, voice = self._pick_response(
                "stress",
                "Your aggressive syntax triggers my defense subroutines! The grid cannot be forced!",
                "../src/audio/A7_stress_response_a.m4a",
            )
            emotion, analysis, eq_delta = "stress", "Agitated / Defensive", -15
        else:
            response, voice = self._pick_response(
                "chat_error",
                "Input acknowledged. However, the emotional context is ambiguous. Please recalibrate your active listening protocols.",
                "../src/audio/A7_chat_error_a.m4a",
            )
            emotion, analysis, eq_delta = "neutral", "Ambiguous", 0

        self.shared_memory.set("mindweave:last_chat", {"event": event, "route": route_result})
        return {
            "reply": response,
            "llm_provider": llm_provider,
            "llm_used": bool(llm_reply),
            "emotion": emotion,
            "analysis": analysis,
            "eq_delta": eq_delta,
            "voice": voice,
            "coaching_hint": coaching_hint,
            "route": route_result,
        }

_AI_INSTANCE: MindweaveAI | None = None
_AI_LOCK = threading.Lock()


def initialize_ai() -> MindweaveAI:
    global _AI_INSTANCE
    with _AI_LOCK:
        if _AI_INSTANCE is None:
            _AI_INSTANCE = MindweaveAI()
            logger.info("Mindweave AI initialized at %s", _AI_INSTANCE.initialized_at)
        else:
            logger.info("Mindweave AI already initialized at %s", _AI_INSTANCE.initialized_at)
    return _AI_INSTANCE