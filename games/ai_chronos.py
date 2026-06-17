"""Chronos AI runtime with full SLAI multi-agent integration.

This module provides production-grade Chronos move selection by combining:
- Knowledge retrieval for strategic guidance.
- Planning-agent task synthesis for objective alignment.
- Execution-agent arbitration across candidate actions.
- Persistent learning updates from completed matches.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, cast

games_root = Path(__file__).resolve().parent
project_root = games_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.agent_factory import AgentFactory  # type: ignore
from src.agents.collaborative.shared_memory import SharedMemory  # type: ignore
from src.agents.planning.planning_types import Task, TaskType  # type: ignore
from logs.logger import PrettyPrinter, get_logger  # type: ignore


logger = get_logger("Chronos")
printer = PrettyPrinter()

# ---------------------------------------------------------------------------
# Constants and DTOs
# ---------------------------------------------------------------------------
AI_OWNER = 1
HUMAN_OWNER = 0
KNOWN_ACTION_TYPES = {"move", "attack", "claim", "pass", "end", "continue"}
PIECE_VALUES = {
    "Strategos": 7.0,
    "Commander": 7.0,
    "Warden": 3.5,
    "Guardian": 3.5,
    "Scout": 1.8,
    "Token": 1.0,
}
ACTION_BASE_WEIGHTS = {
    "attack": 52.0,
    "claim": 45.0,
    "move": 26.0,
    "pass": -38.0,
    "end": -4.0,
    "continue": 8.0,
}
PHASE_ORDER = {"opening": 0, "planning": 1, "midgame": 2, "late": 3, "endgame": 4}


@dataclass(slots=True)
class AgentHandle:
    name: str
    instance: Any
    available: bool
    reason: str | None = None


@dataclass(slots=True)
class SafetyResult:
    valid: bool
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    penalty: float = 0.0


@dataclass(slots=True)
class CandidateMove:
    move: dict[str, Any]
    canonical: str
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    reason: list[str] = field(default_factory=list)
    safety: SafetyResult = field(default_factory=lambda: SafetyResult(valid=True))


@dataclass(slots=True)
class TacticalAnalysis:
    board_size: int
    phase: str
    round_index: int
    score_diff: float
    material_diff: float
    core_balance: float
    token_balance: float
    initiative: float
    enemy_pressure: float
    attack_opportunities: int
    claim_opportunities: int
    move_opportunities: int
    threatened_ai_units: int
    threatened_human_units: int
    ai_core_units: int
    human_core_units: int
    legal_action_count: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "board_size": self.board_size,
            "phase": self.phase,
            "round_index": self.round_index,
            "score_diff": round(self.score_diff, 4),
            "material_diff": round(self.material_diff, 4),
            "core_balance": round(self.core_balance, 4),
            "token_balance": round(self.token_balance, 4),
            "initiative": round(self.initiative, 4),
            "enemy_pressure": round(self.enemy_pressure, 4),
            "attack_opportunities": self.attack_opportunities,
            "claim_opportunities": self.claim_opportunities,
            "move_opportunities": self.move_opportunities,
            "threatened_ai_units": self.threatened_ai_units,
            "threatened_human_units": self.threatened_human_units,
            "ai_core_units": self.ai_core_units,
            "human_core_units": self.human_core_units,
            "legal_action_count": self.legal_action_count,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class AdaptiveProfile:
    stance: str
    confidence: float
    exploration: float
    risk_tolerance: float
    token_conservation: float
    action_bias: dict[str, float]
    zone_bias: dict[str, float]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stance": self.stance,
            "confidence": round(self.confidence, 4),
            "exploration": round(self.exploration, 4),
            "risk_tolerance": round(self.risk_tolerance, 4),
            "token_conservation": round(self.token_conservation, 4),
            "action_bias": dict(self.action_bias),
            "zone_bias": dict(self.zone_bias),
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class StrategicPlan:
    objective: str
    phase: str
    priorities: list[str]
    risk_posture: str
    target_zones: list[str]
    preferred_actions: list[str]
    agent_steps: list[str] = field(default_factory=list)
    source: str = "local_planner"

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "phase": self.phase,
            "priorities": list(self.priorities),
            "risk_posture": self.risk_posture,
            "target_zones": list(self.target_zones),
            "preferred_actions": list(self.preferred_actions),
            "agent_steps": list(self.agent_steps),
            "source": self.source,
        }


class NullAgent:
    """Safe no-op substitute for unavailable SLAI agents."""

    def __init__(self, name: str, reason: str = "not_initialized") -> None:
        self.name = name
        self.reason = reason

    def perform_task(self, task_data: Any) -> dict[str, Any]:
        return {"status": "skipped", "agent": self.name, "reason": self.reason, "input_type": type(task_data).__name__}

    def query(self, query: str) -> str:
        return ""

    def observe(self, signal: dict[str, Any]) -> None:
        return None

    def learn(self, payload: dict[str, Any]) -> None:
        return None

    def learn_from_feedback(self, feedback: dict[str, Any]) -> None:
        return None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or isinstance(value, bool):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _round_score_map(values: Mapping[str, float]) -> dict[str, float]:
    return {key: round(float(value), 4) for key, value in values.items()}


# ---------------------------------------------------------------------------
# Chronos runtime
# ---------------------------------------------------------------------------
@dataclass
class ChronosAI:
    game: str = "chronos"
    initialized_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.shared_memory = SharedMemory()
        self.factory = None
        self.agent_handles: dict[str, AgentHandle] = {}
        self.random = random.Random()

        self.logs_dir = project_root / "logs"
        self.match_log_path = self.logs_dir / "chronos_matches.jsonl"
        self.learning_store_path = self.logs_dir / "chronos_learning_state.json"
        self.decision_log_path = self.logs_dir / "chronos_decisions.jsonl"
        self.training_checkpoint_path = self.logs_dir / "chronos_training_checkpoint.json"
        self.training_summary_path = self.logs_dir / "chronos_training_summary.json"

        self.learning_state_error: str | None = None
        self._learning_state_loaded = False
        self._learning_state_mtime = 0.0
        self.current_match_signatures: list[str] = []
        self.last_trace: dict[str, Any] | None = None

        self.stats: dict[str, Any] = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "average_reward": 0.0,
            "average_final_score": 0.0,
            "last_result": None,
            "last_updated": None,
        }
        self.action_weights: dict[str, float] = {}
        self.phase_action_weights: dict[str, dict[str, float]] = {}
        self.zone_weights: dict[str, float] = {"core": 0.0, "near_core": 0.0, "perimeter": 0.0}
        self.move_signature_weights: dict[str, float] = {}
        self.opening_signature_stats: dict[str, dict[str, int]] = {}
        self.opponent_patterns: dict[str, Any] = {
            "action_counts": {},
            "core_pressure_events": 0,
            "attack_events": 0,
            "last_seen": None,
        }
        self.mistake_tracker: dict[str, int] = {}
        self.adaptive_state: dict[str, Any] = {
            "confidence": 0.5,
            "last_stance": "balanced",
            "risk_appetite": 0.0,
            "opponent_aggression": 0.0,
            "opponent_claim_pressure": 0.0,
            "opponent_focus_core": 0.0,
            "history": [],
        }

        self._initialize_agents()
        self._load_learning_state()
        self._shared_set("chronos_ai_status", "initialized")
        self._shared_set("chronos_stats", dict(self.stats))
        logger.info("Chronos AI initialized with defensive SLAI multi-agent runtime")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Return runtime health and integration diagnostics."""

        agent_status = {
            name: {"available": handle.available, "reason": handle.reason}
            for name, handle in sorted(self.agent_handles.items())
        }
        ready_agents = [name for name, handle in self.agent_handles.items() if handle.available]
        degraded_agents = [name for name, handle in self.agent_handles.items() if not handle.available]
        status = "ready" if ready_agents else "degraded"
        return {
            "status": status,
            "agent_status": status,
            "game": self.game,
            "initialized_at": self.initialized_at,
            "agents": agent_status,
            "ready_agents": sorted(ready_agents),
            "degraded_agents": sorted(degraded_agents),
            "stats": dict(self.stats),
            "learning_state_loaded": self._learning_state_loaded,
            "learning_state_error": self.learning_state_error,
            "learning_state_path": str(self.learning_store_path),
            "match_log_path": str(self.match_log_path),
            "decision_log_path": str(self.decision_log_path),
            "last_trace_available": isinstance(self.last_trace, dict),
            "last_confidence": None
            if not isinstance(self.last_trace, dict)
            else self.last_trace.get("confidence"),
            "training_status": self.training_status(),
        }

    def get_move(self, game_state: dict[str, Any]) -> dict[str, Any] | None:
        """Select a legal Chronos move.

        By default this returns a raw move object to remain compatible with the
        existing frontend and shared backend. If ``game_state`` includes one of
        ``debug``, ``include_metadata``, ``return_metadata`` or
        ``_debug_response`` as a truthy value, the method returns the rich
        response envelope instead.
        """

        wants_debug = isinstance(game_state, dict) and self._wants_debug_response(game_state)
        if not isinstance(game_state, dict):
            response = self._fallback_response("game_state must be a JSON object", None)
            return response if wants_debug else None

        self._sync_learning_state_if_changed()

        try:
            normalized = self._normalize_state(game_state)
            if normalized["phase"] == "strategos_decision":
                choice = self._choose_mutual_strategos_choice(normalized)
                trace = self._build_trace(
                    normalized=normalized,
                    tactical=None,
                    knowledge_context="mutual Strategos decision",
                    plan=None,
                    candidates=[],
                    selected_move={"choice": choice},
                    confidence=0.72,
                    fallback=False,
                    fallback_reason=None,
                    safety=SafetyResult(valid=True),
                    agent_notes={"execution": "selected continue/end by material and score parity"},
                )
                self._record_trace(trace)
                return self._debug_response({"choice": choice}, trace) if wants_debug else {"choice": choice}

            legal_actions = normalized["legal_actions"]
            if not legal_actions:
                response = self._fallback_response("no legal actions supplied", normalized)
                return response if wants_debug else None

            tactical = self._analyze_tactics(normalized)
            knowledge_context = self._retrieve_knowledge_context(normalized, tactical)
            adaptive_profile = self._build_adaptive_profile(normalized, tactical)
            plan = self._generate_strategic_plan(normalized, tactical, knowledge_context, adaptive_profile)
            candidates = self._generate_candidates(normalized, tactical, knowledge_context, plan, adaptive_profile)

            selected, arbitration_notes = self._arbitrate_move(normalized, candidates, plan, adaptive_profile)
            if selected is None:
                response = self._fallback_response("no candidate passed safety validation", normalized)
                return response if wants_debug else None

            final_safety = self._validate_candidate(selected, normalized)
            if not final_safety.valid:
                next_valid = self._next_best_valid_candidate(candidates, normalized, rejected={selected.canonical})
                if next_valid is None:
                    response = self._fallback_response(f"selected candidate rejected: {final_safety.reason}", normalized)
                    return response if wants_debug else None
                selected = next_valid
                final_safety = selected.safety
                arbitration_notes.append("execution candidate rejected; next-best legal candidate selected")

            confidence = self._confidence_from_candidate(selected, candidates, tactical, adaptive_profile)
            move_signature = self._move_signature(selected.move, normalized["board_size"])
            if move_signature:
                self.current_match_signatures.append(move_signature)
                self.current_match_signatures = self.current_match_signatures[-512:]

            trace = self._build_trace(
                normalized=normalized,
                tactical=tactical,
                knowledge_context=knowledge_context,
                plan=plan,
                candidates=candidates,
                selected_move=selected.move,
                confidence=confidence,
                fallback=False,
                fallback_reason=None,
                safety=final_safety,
                agent_notes={"execution": "; ".join(arbitration_notes) or "highest safe candidate selected"},
            )
            self._record_trace(trace)
            self._append_jsonl(self.decision_log_path, trace)

            return self._debug_response(selected.move, trace) if wants_debug else selected.move
        except Exception as error:  # noqa: BLE001
            logger.exception("Chronos get_move failed")
            response = self._fallback_response(f"move pipeline failed: {type(error).__name__}: {error}", None)
            return response if wants_debug else None

    def learn_from_game(self, payload: dict[str, Any]) -> bool:
        """Update local learning state from a completed game payload."""

        if not isinstance(payload, dict):
            return False

        try:
            self._sync_learning_state_if_changed()
            enriched = {
                **payload,
                "game": self.game,
                "logged_at": _utc_now(),
                "schema": "chronos.learning.v2",
            }
            self._append_jsonl(self.match_log_path, enriched)
            self._update_stats(payload)
            self._update_learning_weights(payload)
            self._update_opponent_patterns(payload)
            self._notify_learning_agents(enriched)
            self._save_learning_state()
            self._shared_set("chronos_last_game", enriched)
            self._shared_set("chronos_stats", dict(self.stats))
            return True
        except Exception as error:  # noqa: BLE001
            logger.warning("Chronos learning update failed: %s", error)
            return False

    def training_status(self) -> dict[str, Any]:
        checkpoint = self._read_json_file(self.training_checkpoint_path)
        summary = self._read_json_file(self.training_summary_path)
        completed = _safe_int(checkpoint.get("completed_episodes", summary.get("completed_episodes", 0)))
        meta = checkpoint.get("meta", {}) if isinstance(checkpoint.get("meta"), dict) else {}
        total = _safe_int(meta.get("episodes", summary.get("episodes", 0)))
        progress = round((completed / total) * 100.0, 2) if total > 0 else 0.0
        checkpoint_mtime = self.training_checkpoint_path.stat().st_mtime if self.training_checkpoint_path.exists() else 0.0
        seconds_since_update = max(0.0, time.time() - checkpoint_mtime) if checkpoint_mtime else None
        active = seconds_since_update is not None and seconds_since_update <= 15.0 and completed < total
        return {
            "available": bool(checkpoint or summary),
            "active": bool(active),
            "completed_episodes": completed,
            "total_episodes": total,
            "progress_percent": progress,
            "depth": _safe_int(meta.get("depth", summary.get("depth", 0))),
            "counter_playouts": _safe_int(meta.get("playouts", summary.get("counter_playouts", 0))),
            "explored_states": _safe_int(summary.get("explored_states", 0)),
            "counter_evals": _safe_int(summary.get("counter_evals", summary.get("counter_evaluations", 0))),
            "wins": summary.get("wins", {}),
            "losses": summary.get("losses", {}),
            "draw_like": summary.get("draw_like", {}),
            "seconds_since_update": None if seconds_since_update is None else round(seconds_since_update, 2),
            "checkpoint_path": str(self.training_checkpoint_path),
            "summary_path": str(self.training_summary_path),
        }

    # ------------------------------------------------------------------
    # Agent integration
    # ------------------------------------------------------------------

    def _initialize_agents(self) -> None:
        agent_names = [
            "knowledge",
            "planning",
            "execution",
            "learning",
            "adaptive",
            "safety",
            "evaluation",
            "observability",
        ]
        if AgentFactory is None:
            for name in agent_names:
                self.agent_handles[name] = AgentHandle(name, NullAgent(name, "AgentFactory import failed"), False, "AgentFactory import failed")
            return

        try:
            self.factory = AgentFactory()  # type: ignore[misc,operator]
        except Exception as error:  # noqa: BLE001
            reason = f"AgentFactory initialization failed: {type(error).__name__}: {error}"
            logger.warning("%s", reason)
            for name in agent_names:
                self.agent_handles[name] = AgentHandle(name, NullAgent(name, reason), False, reason)
            return

        for name in agent_names:
            try:
                instance = cast(Any, self.factory).create(name, self.shared_memory)
                self.agent_handles[name] = AgentHandle(name, instance, True, None)
            except Exception as error:  # noqa: BLE001
                reason = f"{type(error).__name__}: {error}"
                logger.warning("Chronos agent '%s' unavailable: %s", name, reason)
                self.agent_handles[name] = AgentHandle(name, NullAgent(name, reason), False, reason)

    def _agent(self, name: str) -> Any:
        handle = self.agent_handles.get(name)
        if handle is None:
            return NullAgent(name, "not registered")
        return handle.instance

    def _shared_set(self, key: str, value: Any) -> None:
        try:
            self.shared_memory.set(key, value)
        except Exception:  # noqa: BLE001
            pass

    def _shared_get(self, key: str, default: Any = None) -> Any:
        try:
            return self.shared_memory.get(key, default)
        except Exception:  # noqa: BLE001
            return default

    # ------------------------------------------------------------------
    # State normalization and tactical analysis
    # ------------------------------------------------------------------

    def _normalize_state(self, game_state: dict[str, Any]) -> dict[str, Any]:
        board_size = self._board_size(game_state)
        legal_actions_raw = (
            game_state.get("validMoves")
            or game_state.get("legalMoves")
            or game_state.get("legal_actions")
            or game_state.get("actions")
            or []
        )
        legal_actions = [action for action in legal_actions_raw if isinstance(action, dict)] if isinstance(legal_actions_raw, list) else []
        units = self._extract_units(game_state)
        players = game_state.get("players", []) if isinstance(game_state.get("players"), list) else []
        phase = str(game_state.get("phase") or game_state.get("stage") or "planning").lower()
        round_index = _safe_int(game_state.get("round", game_state.get("roundIndex", game_state.get("turnNumber", 0))))
        if phase in {"planning", "active", "action"}:
            if round_index <= 2:
                strategic_phase = "opening"
            elif round_index >= 9:
                strategic_phase = "late"
            else:
                strategic_phase = "midgame"
        else:
            strategic_phase = phase

        legal_canon = {_canonical_json(action) for action in legal_actions}
        return {
            "raw": game_state,
            "board_size": board_size,
            "legal_actions": legal_actions,
            "legal_action_canon": legal_canon,
            "units": units,
            "players": players,
            "phase": phase,
            "strategic_phase": strategic_phase,
            "round_index": round_index,
            "current_player": game_state.get("currentPlayer", game_state.get("turn", game_state.get("activePlayer", AI_OWNER))),
            "timestamp": time.time(),
        }

    def _analyze_tactics(self, normalized: dict[str, Any]) -> TacticalAnalysis:
        units = normalized["units"]
        legal_actions = normalized["legal_actions"]
        board_size = normalized["board_size"]
        ai_units = [u for u in units if self._owner_of(u) == AI_OWNER and _safe_int(u.get("hp", u.get("health", 1)), 1) > 0]
        human_units = [u for u in units if self._owner_of(u) == HUMAN_OWNER and _safe_int(u.get("hp", u.get("health", 1)), 1) > 0]
        material_ai = sum(self._piece_value(u) for u in ai_units)
        material_human = sum(self._piece_value(u) for u in human_units)
        ai_core = sum(1 for unit in ai_units if self._is_unit_in_core(unit, board_size))
        human_core = sum(1 for unit in human_units if self._is_unit_in_core(unit, board_size))

        player_scores, token_counts = self._extract_player_metrics(normalized)
        attack_count = sum(1 for action in legal_actions if action.get("type") == "attack")
        claim_count = sum(1 for action in legal_actions if action.get("type") == "claim")
        move_count = sum(1 for action in legal_actions if action.get("type") == "move")
        total = max(1, len(legal_actions))
        initiative = ((attack_count * 1.25) + (claim_count * 1.15) + (move_count * 0.65)) / total

        threatened_ai = self._count_threatened_units(ai_units, human_units)
        threatened_human = self._count_threatened_units(human_units, ai_units)
        score_diff = player_scores.get(AI_OWNER, 0.0) - player_scores.get(HUMAN_OWNER, 0.0)
        material_diff = material_ai - material_human
        core_balance = float(ai_core - human_core)
        token_balance = float(token_counts.get(AI_OWNER, 0) - token_counts.get(HUMAN_OWNER, 0))
        enemy_pressure = _clamp(
            0.22
            + max(0.0, -score_diff) * 0.035
            + max(0.0, -material_diff) * 0.045
            + max(0.0, -core_balance) * 0.09
            + threatened_ai * 0.065,
            0.0,
            1.0,
        )

        notes: list[str] = []
        if threatened_ai:
            notes.append("own_units_under_threat")
        if attack_count:
            notes.append("attack_window_available")
        if claim_count:
            notes.append("core_claim_available")
        if core_balance < 0:
            notes.append("opponent_core_pressure")
        if material_diff < 0:
            notes.append("material_disadvantage")

        return TacticalAnalysis(
            board_size=board_size,
            phase=normalized["strategic_phase"],
            round_index=normalized["round_index"],
            score_diff=score_diff,
            material_diff=material_diff,
            core_balance=core_balance,
            token_balance=token_balance,
            initiative=initiative,
            enemy_pressure=enemy_pressure,
            attack_opportunities=attack_count,
            claim_opportunities=claim_count,
            move_opportunities=move_count,
            threatened_ai_units=threatened_ai,
            threatened_human_units=threatened_human,
            ai_core_units=ai_core,
            human_core_units=human_core,
            legal_action_count=len(legal_actions),
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Knowledge, planning, adaptation
    # ------------------------------------------------------------------

    def _retrieve_knowledge_context(self, normalized: dict[str, Any], tactical: TacticalAnalysis) -> str:
        baseline = (
            "Chronos strategy: control core tempo, avoid exposing Strategos, convert attacks only when safety is favorable, "
            "use claims to force opponent displacement, and preserve token flexibility in late-game conversion."
        )
        query = (
            f"Chronos {tactical.phase} board-{tactical.board_size} strategy; "
            f"score_diff={tactical.score_diff}; material_diff={tactical.material_diff}; "
            f"core_balance={tactical.core_balance}; enemy_pressure={tactical.enemy_pressure}; "
            f"notes={','.join(tactical.notes)}"
        )
        agent = self._agent("knowledge")
        try:
            if hasattr(agent, "query"):
                response = agent.query(query)
                if response:
                    context = str(response)
                    self._shared_set("chronos_strategy_context", context)
                    return context
            if hasattr(agent, "perform_task"):
                response = agent.perform_task({"type": "chronos_strategy_retrieval", "query": query})
                if isinstance(response, dict):
                    for key in ("context", "answer", "result", "text"):
                        value = response.get(key)
                        if value:
                            context = str(value)
                            self._shared_set("chronos_strategy_context", context)
                            return context
        except Exception as error:  # noqa: BLE001
            logger.warning("Chronos knowledge retrieval failed: %s", error)
        return baseline

    def _generate_strategic_plan(
        self,
        normalized: dict[str, Any],
        tactical: TacticalAnalysis,
        knowledge_context: str,
        adaptive: AdaptiveProfile,
    ) -> StrategicPlan:
        objective = "contest_core_and_preserve_strategos"
        priorities = ["maintain_legal_action_safety", "improve_core_control", "preserve_strategos"]
        preferred_actions = ["claim", "attack", "move"]
        target_zones = ["core", "near_core"]
        risk_posture = "balanced"

        if tactical.enemy_pressure > 0.62 or tactical.threatened_ai_units >= 2:
            objective = "stabilize_defense_before_conversion"
            priorities = ["reduce_immediate_threat", "avoid_strategos_exposure", "recover_core_access"]
            preferred_actions = ["attack", "move", "claim"]
            risk_posture = "defensive"
        elif tactical.attack_opportunities and tactical.threatened_human_units >= tactical.threatened_ai_units:
            objective = "convert_tactical_attack_window"
            priorities = ["remove_high_value_target", "retain_tempo", "secure_next_claim"]
            preferred_actions = ["attack", "claim", "move"]
            risk_posture = "assertive"
        elif tactical.core_balance < 0 or tactical.claim_opportunities:
            objective = "retake_or_secure_core"
            priorities = ["claim_core", "occupy_near_core", "deny_opponent_claims"]
            preferred_actions = ["claim", "move", "attack"]
            risk_posture = "territorial"
        elif tactical.phase in {"late", "endgame"} and tactical.score_diff >= 0:
            objective = "convert_advantage_safely"
            priorities = ["avoid_unforced_loss", "force_safe_claims", "trade_only_when_profitable"]
            preferred_actions = ["claim", "move", "attack"]
            risk_posture = "conservative"

        agent_steps: list[str] = []
        source = "local_planner"
        agent = self._agent("planning")
        try:
            now = time.time()
            primitive = Task(
                name=f"chronos_execute_{objective}",
                task_type=TaskType.PRIMITIVE,
                start_time=now,
                deadline=now + 60,
                duration=1,
                context={"game": self.game, "objective": objective},
            )
            goal = Task(
                name=f"chronos_plan_{objective}_{int(now)}",
                task_type=TaskType.ABSTRACT,
                start_time=now,
                deadline=now + 90,
                duration=2,
                methods=[[primitive]],
                goal_state={"safe_legal_move_selected": True},
                context={
                    "game": self.game,
                    "tactical": tactical.to_dict(),
                    "knowledge_context": knowledge_context,
                    "adaptive": adaptive.to_dict(),
                },
            )
            if hasattr(agent, "register_task"):
                agent.register_task(goal)
            if hasattr(agent, "generate_plan"):
                plan_result = agent.generate_plan(goal)
                if isinstance(plan_result, list):
                    agent_steps = [str(getattr(step, "name", step)) for step in plan_result]
                    source = "planning_agent"
            elif hasattr(agent, "perform_task"):
                response = agent.perform_task({"type": "chronos_plan", "goal": goal, "tactical": tactical.to_dict()})
                if isinstance(response, dict):
                    steps = response.get("steps") or response.get("plan")
                    if isinstance(steps, list):
                        agent_steps = [str(step) for step in steps]
                        source = "planning_agent"
        except Exception as error:  # noqa: BLE001
            logger.warning("Chronos planning-agent call skipped: %s", error)

        if not agent_steps:
            agent_steps = priorities

        plan = StrategicPlan(
            objective=objective,
            phase=tactical.phase,
            priorities=priorities,
            risk_posture=risk_posture,
            target_zones=target_zones,
            preferred_actions=preferred_actions,
            agent_steps=agent_steps,
            source=source,
        )
        self._shared_set("chronos_last_plan", plan.to_dict())
        return plan

    def _build_adaptive_profile(self, normalized: dict[str, Any], tactical: TacticalAnalysis) -> AdaptiveProfile:
        base_confidence = _clamp(
            0.48
            + tactical.score_diff * 0.035
            + tactical.material_diff * 0.035
            + tactical.core_balance * 0.05
            - tactical.enemy_pressure * 0.18
            + (_safe_float(self.adaptive_state.get("confidence"), 0.5) - 0.5) * 0.25,
            0.1,
            0.95,
        )
        stance = "balanced"
        notes = ["balanced_progression"]
        if base_confidence < 0.34 or tactical.material_diff < -4:
            stance = "comeback"
            notes = ["recover_initiative", "increase_practical_pressure"]
        elif tactical.enemy_pressure > 0.68 or tactical.threatened_ai_units >= 2:
            stance = "defensive"
            notes = ["reduce_enemy_pressure", "protect_critical_units"]
        elif tactical.core_balance < 0 or tactical.claim_opportunities > 2:
            stance = "core_contest"
            notes = ["prioritize_core_control"]
        elif base_confidence > 0.72 and tactical.score_diff >= 0:
            stance = "conversion"
            notes = ["secure_existing_advantage"]

        profiles = {
            "comeback": ({"attack": 8.0, "claim": 5.0, "move": 3.5, "pass": -8.0}, {"core": 8.5, "near_core": 5.0, "perimeter": -2.0}, 0.78, 0.7, 1.15),
            "defensive": ({"attack": 4.5, "claim": 2.5, "move": 6.5, "pass": -4.0}, {"core": 4.0, "near_core": 5.5, "perimeter": 1.0}, 0.28, 1.8, 0.35),
            "core_contest": ({"attack": 4.0, "claim": 8.0, "move": 4.0, "pass": -7.0}, {"core": 10.0, "near_core": 5.5, "perimeter": -2.0}, 0.52, 1.2, 0.65),
            "conversion": ({"attack": 2.5, "claim": 7.0, "move": 3.0, "pass": -3.0}, {"core": 8.0, "near_core": 3.5, "perimeter": -2.5}, 0.33, 1.7, 0.4),
            "balanced": ({"attack": 4.0, "claim": 5.5, "move": 3.0, "pass": -5.0}, {"core": 7.0, "near_core": 4.0, "perimeter": 0.0}, 0.45, 1.35, 0.55),
        }
        action_bias, zone_bias, risk_tolerance, token_conservation, exploration = profiles[stance]

        agent = self._agent("adaptive")
        try:
            if hasattr(agent, "perform_task"):
                response = agent.perform_task(
                    {
                        "type": "chronos_adaptation",
                        "tactical": tactical.to_dict(),
                        "local_profile": {"stance": stance, "confidence": base_confidence},
                    }
                )
                if isinstance(response, dict):
                    candidate_confidence = response.get("confidence")
                    if candidate_confidence is None and isinstance(response.get("policy_metrics"), dict):
                        candidate_confidence = response["policy_metrics"].get("recent_success_rate")
                    if candidate_confidence is not None:
                        base_confidence = _clamp(_safe_float(candidate_confidence, base_confidence), 0.1, 0.95)
        except Exception as error:  # noqa: BLE001
            logger.warning("Adaptive agent task skipped: %s", error)

        profile = AdaptiveProfile(
            stance=stance,
            confidence=base_confidence,
            exploration=exploration,
            risk_tolerance=risk_tolerance,
            token_conservation=token_conservation,
            action_bias=dict(action_bias),
            zone_bias=dict(zone_bias),
            notes=notes,
        )
        self._update_adaptive_state(profile, tactical)
        return profile

    # ------------------------------------------------------------------
    # Candidate generation, scoring, safety, execution
    # ------------------------------------------------------------------

    def _generate_candidates(
        self,
        normalized: dict[str, Any],
        tactical: TacticalAnalysis,
        knowledge_context: str,
        plan: StrategicPlan,
        adaptive: AdaptiveProfile,
    ) -> list[CandidateMove]:
        candidates: list[CandidateMove] = []
        for action in normalized["legal_actions"]:
            candidate = CandidateMove(move=action, canonical=_canonical_json(action))
            candidate.safety = self._validate_candidate(candidate, normalized)
            if candidate.safety.valid:
                self._score_candidate(candidate, normalized, tactical, knowledge_context, plan, adaptive)
            else:
                candidate.score = -10_000.0 + candidate.safety.penalty
                candidate.score_breakdown["safety_penalty"] = candidate.safety.penalty
                candidate.reason.append(candidate.safety.reason or "safety_rejected")
            candidates.append(candidate)

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def _score_candidate(
        self,
        candidate: CandidateMove,
        normalized: dict[str, Any],
        tactical: TacticalAnalysis,
        knowledge_context: str,
        plan: StrategicPlan,
        adaptive: AdaptiveProfile,
    ) -> None:
        move = candidate.move
        move_type = str(move.get("type", "unknown")).lower()
        unit_map = self._build_unit_map(normalized["units"])
        acting = unit_map.get(str(move.get("unitId", "")))
        target = self._target_of(move)
        board_size = normalized["board_size"]
        breakdown: dict[str, float] = {}
        reason: list[str] = []

        base = ACTION_BASE_WEIGHTS.get(move_type, -18.0)
        breakdown["action_type_value"] = base
        reason.append(f"type={move_type}")

        if move_type not in KNOWN_ACTION_TYPES:
            breakdown["unknown_action_penalty"] = -28.0
            reason.append("unknown_action_type")

        phase_bonus = self._phase_relevance_bonus(move_type, tactical, plan)
        breakdown["phase_relevance"] = phase_bonus

        tactical_bonus = 0.0
        attack_value = 0.0
        defense_value = 0.0
        disruption = 0.0
        if move_type == "attack":
            target_value = PIECE_VALUES.get(str(target.get("type")), 1.2)
            attack_value += 18.0 + (target_value * 16.0)
            if str(target.get("type")) in {"Strategos", "Commander"}:
                attack_value += 5000.0
                reason.append("strategos_capture_opportunity")
            if target.get("owner") == HUMAN_OWNER:
                disruption += 8.0
            defense_value += 5.0 if tactical.enemy_pressure > 0.55 else 0.0
        elif move_type == "claim":
            tactical_bonus += 22.0
            if tactical.core_balance <= 0:
                tactical_bonus += 18.0
                reason.append("core_recovery_claim")
        elif move_type == "move":
            tactical_bonus += 5.0

        breakdown["tactical_value"] = tactical_bonus
        breakdown["attack_value"] = attack_value
        breakdown["defense_value"] = defense_value
        breakdown["opponent_disruption"] = disruption

        zone = self._zone_for_action(move, acting, board_size)
        zone_bonus = {"core": 31.0, "near_core": 14.0, "perimeter": 2.0}.get(zone, 0.0)
        zone_bonus += self.zone_weights.get(zone, 0.0)
        zone_bonus += adaptive.zone_bias.get(zone, 0.0)
        breakdown["zone_control"] = zone_bonus
        if zone == "core":
            breakdown["core_control"] = 18.0
            reason.append("core_target")
        else:
            breakdown["core_control"] = 0.0

        resource_efficiency = self._resource_efficiency(move, tactical, adaptive)
        breakdown["resource_efficiency"] = resource_efficiency

        tempo = 0.0
        if move_type in {"attack", "claim"}:
            tempo += 9.0
        if move_type == "pass":
            tempo -= 14.0
        if tactical.phase in {"late", "endgame"} and move_type == "claim":
            tempo += 6.0
        breakdown["tempo"] = tempo

        threat_reduction = 0.0
        risk_penalty = 0.0
        if acting:
            destination = self._infer_destination(move, acting)
            enemy_threat = self._estimate_enemy_threat(destination, acting, unit_map)
            risk_penalty -= enemy_threat * (26.0 - adaptive.risk_tolerance * 10.0)
            if str(acting.get("type")) in {"Strategos", "Commander"} and enemy_threat:
                risk_penalty -= 40.0
                reason.append("strategos_exposure_risk")
            if move_type == "attack" and tactical.threatened_ai_units:
                threat_reduction += 6.0
        breakdown["threat_reduction"] = threat_reduction
        breakdown["risk_penalty"] = risk_penalty

        plan_alignment = self._plan_alignment_score(move_type, zone, plan)
        breakdown["plan_alignment"] = plan_alignment

        knowledge_signal = self._knowledge_score(move_type, zone, knowledge_context)
        breakdown["knowledge_agent_signal"] = knowledge_signal

        adaptive_signal = adaptive.action_bias.get(move_type, 0.0)
        breakdown["adaptive_agent_signal"] = adaptive_signal

        learned_prior = self.action_weights.get(move_type, 0.0)
        learned_prior += self.phase_action_weights.get(tactical.phase, {}).get(move_type, 0.0)
        signature = self._move_signature(move, board_size)
        learned_prior += self.move_signature_weights.get(signature, 0.0)
        if tactical.round_index <= 2 and signature:
            opening_stats = self.opening_signature_stats.get(signature, {})
            games = _safe_int(opening_stats.get("games"), 0)
            wins = _safe_int(opening_stats.get("wins"), 0)
            losses = _safe_int(opening_stats.get("losses"), 0)
            if games >= 3:
                win_rate = wins / max(1, games)
                learned_prior += (win_rate - 0.5) * 10.0
                if losses / max(1, games) > 0.65:
                    learned_prior -= 5.0
        breakdown["learned_prior"] = learned_prior

        safety_penalty = candidate.safety.penalty
        if candidate.safety.warnings:
            safety_penalty -= 2.0 * len(candidate.safety.warnings)
        breakdown["safety_penalty"] = safety_penalty

        fallback_priority = 0.0
        if move_type != "pass":
            fallback_priority += 3.0
        if candidate.safety.valid:
            fallback_priority += 2.0
        breakdown["fallback_priority"] = fallback_priority

        exploration = 0.0
        experience = _safe_int(self.stats.get("games_played"), 0)
        if experience < 1200:
            exploration = self.random.uniform(0.0, adaptive.exploration * max(0.25, 1.0 - experience / 2000.0))
        breakdown["controlled_exploration"] = exploration

        candidate.score_breakdown = breakdown
        candidate.score = sum(breakdown.values())
        candidate.reason = reason + [f"zone={zone}", f"plan={plan.objective}", f"stance={adaptive.stance}"]

    def _validate_candidate(self, candidate: CandidateMove, normalized: dict[str, Any]) -> SafetyResult:
        move = candidate.move
        if not isinstance(move, dict):
            return SafetyResult(False, "move is not an object", penalty=-500.0)
        if candidate.canonical not in normalized["legal_action_canon"]:
            return SafetyResult(False, "move is not present in legal action list", penalty=-500.0)
        return self._validate_action_schema(move, normalized)

    def _validate_action_schema(self, move: dict[str, Any], normalized: dict[str, Any]) -> SafetyResult:
        warnings: list[str] = []
        move_type_raw = move.get("type")
        if not isinstance(move_type_raw, str) or not move_type_raw.strip():
            return SafetyResult(False, "missing action type", penalty=-300.0)
        move_type = move_type_raw.lower()
        if move_type not in KNOWN_ACTION_TYPES:
            warnings.append("unknown action type accepted only because the game engine listed it as legal")

        if move_type == "pass":
            return SafetyResult(True, warnings=warnings, penalty=0.0)

        unit_id = move.get("unitId", move.get("unit_id"))
        if move_type in {"move", "attack", "claim"} and not isinstance(unit_id, str):
            return SafetyResult(False, "missing unitId for unit action", penalty=-250.0)

        unit_map = self._build_unit_map(normalized["units"])
        acting = unit_map.get(str(unit_id)) if unit_id is not None else None
        if move_type in {"move", "attack", "claim"} and acting is None:
            return SafetyResult(False, "acting unit not found", penalty=-240.0)
        if acting and _safe_int(acting.get("hp", 1), 1) <= 0:
            return SafetyResult(False, "acting unit is not alive", penalty=-240.0)

        target = self._target_of(move)
        if move_type in {"move", "claim", "attack"} and not isinstance(target, dict):
            return SafetyResult(False, "target must be an object", penalty=-220.0)

        board_size = normalized["board_size"]
        r, c = self._coords_of(target)
        if move_type in {"move", "claim"}:
            if not isinstance(r, int) or not isinstance(c, int):
                return SafetyResult(False, "move/claim target requires integer coordinates", penalty=-210.0)
            if not (0 <= r < board_size and 0 <= c < board_size):
                return SafetyResult(False, "target coordinates outside board", penalty=-210.0)
        if move_type == "attack":
            has_target_id = isinstance(target.get("id"), str) and bool(target.get("id"))
            has_coords = isinstance(r, int) and isinstance(c, int) and 0 <= r < board_size and 0 <= c < board_size
            if not has_target_id and not has_coords:
                return SafetyResult(False, "attack target needs id or valid coordinates", penalty=-210.0)
            target_owner = target.get("owner", target.get("playerId"))
            if acting is not None and target_owner == acting.get("owner"):
                return SafetyResult(False, "attack target appears friendly", penalty=-260.0)

        try:
            agent = self._agent("safety")
            if hasattr(agent, "perform_task"):
                response = agent.perform_task({"type": "chronos_action_validation", "action": move, "board_size": board_size})
                if isinstance(response, dict):
                    explicit_valid = response.get("valid")
                    explicit_status = str(response.get("status", "")).lower()
                    if explicit_valid is False or explicit_status in {"blocked", "rejected", "invalid"}:
                        return SafetyResult(False, str(response.get("reason") or "safety agent rejected action"), penalty=-280.0)
        except Exception as error:  # noqa: BLE001
            warnings.append(f"safety agent unavailable: {type(error).__name__}")

        return SafetyResult(True, warnings=warnings, penalty=0.0)

    def _arbitrate_move(
        self,
        normalized: dict[str, Any],
        candidates: list[CandidateMove],
        plan: StrategicPlan,
        adaptive: AdaptiveProfile,
    ) -> tuple[CandidateMove | None, list[str]]:
        notes: list[str] = []
        valid_candidates = [candidate for candidate in candidates if candidate.safety.valid]
        if not valid_candidates:
            return None, ["no valid candidates"]

        selected = valid_candidates[0]
        agent = self._agent("execution")
        action_candidates = [
            {
                "name": f"candidate_{index}",
                "index": index,
                "priority": int(candidate.score + 10_000),
                "score": candidate.score,
                "move": candidate.move,
                "reason": "; ".join(candidate.reason),
            }
            for index, candidate in enumerate(valid_candidates[:12])
        ]
        context = {
            "game": self.game,
            "objective": plan.objective,
            "risk_posture": plan.risk_posture,
            "adaptive": adaptive.to_dict(),
            "top_score": valid_candidates[0].score,
        }
        try:
            chosen: Any = None
            if hasattr(agent, "action_selector") and hasattr(agent.action_selector, "select"):
                chosen = agent.action_selector.select(action_candidates, context)
                notes.append("execution.action_selector used")
            elif hasattr(agent, "perform_task"):
                chosen = agent.perform_task({"type": "chronos_arbitration", "candidates": action_candidates, "context": context})
                notes.append("execution.perform_task used")

            chosen_index = self._extract_candidate_index(chosen)
            if chosen_index is not None and 0 <= chosen_index < len(valid_candidates):
                proposed = valid_candidates[chosen_index]
                if proposed.safety.valid and proposed.canonical in normalized["legal_action_canon"]:
                    selected = proposed
                    notes.append(f"execution selected candidate_{chosen_index}")
                else:
                    notes.append("execution-selected candidate failed local safety; ignored")
        except Exception as error:  # noqa: BLE001
            notes.append(f"execution agent skipped: {type(error).__name__}: {error}")

        return selected, notes

    # ------------------------------------------------------------------
    # Trace, fallback, and response shaping
    # ------------------------------------------------------------------

    def _build_trace(
        self,
        *,
        normalized: dict[str, Any],
        tactical: TacticalAnalysis | None,
        knowledge_context: str,
        plan: StrategicPlan | None,
        candidates: list[CandidateMove],
        selected_move: dict[str, Any] | None,
        confidence: float,
        fallback: bool,
        fallback_reason: str | None,
        safety: SafetyResult,
        agent_notes: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        top_candidates = [
            {
                "move": candidate.move,
                "score": round(candidate.score, 4),
                "breakdown": _round_score_map(candidate.score_breakdown),
                "reason": "; ".join(candidate.reason),
                "safety": {
                    "valid": candidate.safety.valid,
                    "reason": candidate.safety.reason,
                    "warnings": list(candidate.safety.warnings),
                },
            }
            for candidate in candidates[:8]
        ]
        plan_dict = plan.to_dict() if isinstance(plan, StrategicPlan) else None
        tactical_dict = tactical.to_dict() if isinstance(tactical, TacticalAnalysis) else None
        agent_notes = dict(agent_notes or {})
        trace = {
            "schema": "chronos.agent_trace.v2",
            "timestamp": _utc_now(),
            "interpreted_game_state": {
                "phase": normalized.get("phase"),
                "strategic_phase": normalized.get("strategic_phase"),
                "round_index": normalized.get("round_index"),
                "board_size": normalized.get("board_size"),
                "unit_count": len(normalized.get("units", [])),
            },
            "legal_actions_count": len(normalized.get("legal_actions", [])),
            "legal_actions_sample": normalized.get("legal_actions", [])[:10],
            "candidate_actions_count": len(candidates),
            "candidate_scores": top_candidates,
            "tactical_risks_detected": tactical_dict,
            "strategic_objective": None if plan_dict is None else plan_dict.get("objective"),
            "plan": plan_dict,
            "agent_contributions": {
                "knowledge": knowledge_context[:800] if isinstance(knowledge_context, str) else "",
                "planning": "none" if plan_dict is None else json.dumps(plan_dict, ensure_ascii=False)[:1000],
                "execution": agent_notes.get("execution", "local score ordering"),
                "learning": f"games={self.stats.get('games_played', 0)} action_priors={self.action_weights}",
                "adaptive": json.dumps(self.adaptive_state, ensure_ascii=False)[:1000],
                "safety": "valid" if safety.valid else f"invalid: {safety.reason}",
                "evaluation": agent_notes.get("evaluation", "confidence computed locally"),
                "observability": agent_notes.get("observability", "trace stored in shared memory and decision log"),
            },
            "safety_validation_result": {
                "valid": safety.valid,
                "reason": safety.reason,
                "warnings": list(safety.warnings),
                "penalty": safety.penalty,
            },
            "final_move_selection": selected_move,
            "confidence": round(confidence, 4),
            "fallback": fallback,
            "fallback_reason": fallback_reason,
        }
        return trace

    def _record_trace(self, trace: dict[str, Any]) -> None:
        self.last_trace = trace
        self._shared_set("chronos_last_decision", trace)
        try:
            agent = self._agent("observability")
            if hasattr(agent, "perform_task"):
                agent.perform_task({"type": "chronos_trace", "trace": trace})
        except Exception:  # noqa: BLE001
            pass

    def _fallback_response(self, reason: str, normalized: dict[str, Any] | None) -> dict[str, Any]:
        normalized = normalized or {
            "phase": None,
            "strategic_phase": None,
            "round_index": None,
            "board_size": 0,
            "units": [],
            "legal_actions": [],
        }
        legal_actions = normalized.get("legal_actions", []) if isinstance(normalized, dict) else []
        fallback_move = None
        if isinstance(legal_actions, list):
            pass_moves = [action for action in legal_actions if isinstance(action, dict) and action.get("type") == "pass"]
            if pass_moves:
                fallback_move = pass_moves[0]
        trace = self._build_trace(
            normalized=normalized,
            tactical=None,
            knowledge_context="fallback path",
            plan=None,
            candidates=[],
            selected_move=fallback_move,
            confidence=0.0 if fallback_move is None else 0.2,
            fallback=True,
            fallback_reason=reason,
            safety=SafetyResult(valid=fallback_move is not None, reason=None if fallback_move else reason),
            agent_notes={"execution": "controlled fallback"},
        )
        self._record_trace(trace)
        if fallback_move is not None:
            return self._debug_response(fallback_move, trace)
        return self._debug_response(None, trace)

    @staticmethod
    def _debug_response(move: dict[str, Any] | None, trace: dict[str, Any]) -> dict[str, Any]:
        plan = trace.get("plan") if isinstance(trace.get("plan"), dict) else {}
        agent_contrib = trace.get("agent_contributions", {}) if isinstance(trace.get("agent_contributions"), dict) else {}
        objective = plan.get("objective") if isinstance(plan, dict) else ""
        return {
            "move": move,
            "confidence": trace.get("confidence", 0.0),
            "strategy": objective or "fallback",
            "reasoning": trace.get("fallback_reason") or f"Selected by strategy: {objective}",
            "agent_trace": {
                "knowledge": agent_contrib.get("knowledge", ""),
                "planning": agent_contrib.get("planning", ""),
                "execution": agent_contrib.get("execution", ""),
                "learning": agent_contrib.get("learning", ""),
                "adaptive": agent_contrib.get("adaptive", ""),
                "safety": agent_contrib.get("safety", ""),
            },
            "fallback": bool(trace.get("fallback")),
            "fallback_reason": trace.get("fallback_reason"),
            "debug_trace": trace,
        }

    @staticmethod
    def _wants_debug_response(game_state: dict[str, Any]) -> bool:
        return any(bool(game_state.get(key)) for key in ("debug", "include_metadata", "return_metadata", "_debug_response"))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _update_stats(self, payload: dict[str, Any]) -> None:
        outcome = str(payload.get("outcome", payload.get("result", "draw"))).lower()
        if outcome not in {"win", "loss", "draw"}:
            winner = payload.get("winner")
            if winner == AI_OWNER:
                outcome = "win"
            elif winner == HUMAN_OWNER:
                outcome = "loss"
            else:
                outcome = "draw"
        reward = _safe_float(payload.get("reward"), 0.0)
        final_score = _safe_float(payload.get("final_score", payload.get("score_delta")), 0.0)
        self.stats["games_played"] = _safe_int(self.stats.get("games_played"), 0) + 1
        if outcome == "win":
            self.stats["wins"] = _safe_int(self.stats.get("wins"), 0) + 1
        elif outcome == "loss":
            self.stats["losses"] = _safe_int(self.stats.get("losses"), 0) + 1
        else:
            self.stats["draws"] = _safe_int(self.stats.get("draws"), 0) + 1
        games = max(1, _safe_int(self.stats["games_played"], 1))
        self.stats["average_reward"] = round(_safe_float(self.stats.get("average_reward")) + ((reward - _safe_float(self.stats.get("average_reward"))) / games), 4)
        self.stats["average_final_score"] = round(_safe_float(self.stats.get("average_final_score")) + ((final_score - _safe_float(self.stats.get("average_final_score"))) / games), 4)
        self.stats["last_result"] = outcome
        self.stats["last_updated"] = _utc_now()

    def _update_learning_weights(self, payload: dict[str, Any]) -> None:
        outcome = str(payload.get("outcome", payload.get("result", "draw"))).lower()
        reward = _safe_float(payload.get("reward"), 0.0)
        delta = 0.15
        if outcome == "win":
            delta = 0.75
        elif outcome == "loss":
            delta = -0.65
        elif reward < 0:
            delta = -0.2
        elif reward > 0:
            delta = 0.25

        signatures = self._payload_signatures(payload)
        if not signatures:
            signatures = list(self.current_match_signatures)

        last_trace = self.last_trace or self._shared_get("chronos_last_decision", {}) or {}
        selected_move = last_trace.get("final_move_selection") if isinstance(last_trace, dict) else None
        if isinstance(selected_move, dict):
            action_type = str(selected_move.get("type", "unknown")).lower()
            self.action_weights[action_type] = _clamp(self.action_weights.get(action_type, 0.0) + delta, -10.0, 14.0)
            phase = str((last_trace.get("interpreted_game_state") or {}).get("strategic_phase", "unknown")) if isinstance(last_trace, dict) else "unknown"
            self.phase_action_weights.setdefault(phase, {})
            self.phase_action_weights[phase][action_type] = _clamp(self.phase_action_weights[phase].get(action_type, 0.0) + (delta * 0.45), -8.0, 10.0)
            zone = self._zone_for_action(selected_move, None, _safe_int(payload.get("board_size"), 9))
            self.zone_weights[zone] = _clamp(self.zone_weights.get(zone, 0.0) + delta * 0.35, -8.0, 10.0)

        scale = _clamp(max(0.2, abs(reward) / 80.0), 0.2, 1.5)
        for index, signature in enumerate(signatures):
            recency = 0.5 + ((index + 1) / max(1, len(signatures)))
            change = delta * 0.18 * recency * scale
            self.move_signature_weights[signature] = _clamp(self.move_signature_weights.get(signature, 0.0) + change, -12.0, 12.0)

        if signatures:
            opening_signature = signatures[0]
            opening_stats = self.opening_signature_stats.get(opening_signature, {"games": 0, "wins": 0, "losses": 0})
            opening_stats["games"] = _safe_int(opening_stats.get("games"), 0) + 1
            if outcome == "win":
                opening_stats["wins"] = _safe_int(opening_stats.get("wins"), 0) + 1
            elif outcome == "loss":
                opening_stats["losses"] = _safe_int(opening_stats.get("losses"), 0) + 1
            self.opening_signature_stats[opening_signature] = opening_stats

        mistakes = payload.get("mistakes") or payload.get("mistake_signatures") or []
        if isinstance(mistakes, list):
            for mistake in mistakes:
                if isinstance(mistake, str):
                    self.mistake_tracker[mistake] = _safe_int(self.mistake_tracker.get(mistake), 0) + 1
                    self.move_signature_weights[mistake] = _clamp(self.move_signature_weights.get(mistake, 0.0) - 0.8, -12.0, 12.0)

        self._prune_learning_maps()
        self.current_match_signatures = []

    def _payload_signatures(self, payload: dict[str, Any]) -> list[str]:
        raw = payload.get("ai_move_signatures") or payload.get("move_signatures") or []
        signatures: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, str):
                    signatures.append(item)
                elif isinstance(item, dict):
                    signatures.append(self._move_signature(item, _safe_int(payload.get("board_size"), 9)))
        move_history = payload.get("move_history") or payload.get("moves")
        if isinstance(move_history, list):
            for item in move_history:
                if isinstance(item, dict):
                    signatures.append(self._move_signature(item, _safe_int(payload.get("board_size"), 9)))
        return [sig for sig in signatures if sig]

    def _update_opponent_patterns(self, payload: dict[str, Any]) -> None:
        actions = payload.get("opponent_actions") or payload.get("human_moves") or []
        if not isinstance(actions, list):
            return
        counts = self.opponent_patterns.setdefault("action_counts", {})
        for action in actions:
            if not isinstance(action, dict):
                continue
            action_type = str(action.get("type", "unknown")).lower()
            counts[action_type] = _safe_int(counts.get(action_type), 0) + 1
            if action_type == "attack":
                self.opponent_patterns["attack_events"] = _safe_int(self.opponent_patterns.get("attack_events"), 0) + 1
            target = self._target_of(action)
            r, c = self._coords_of(target)
            if isinstance(r, int) and isinstance(c, int) and self._is_core_cell(r, c, _safe_int(payload.get("board_size"), 9)):
                self.opponent_patterns["core_pressure_events"] = _safe_int(self.opponent_patterns.get("core_pressure_events"), 0) + 1
        self.opponent_patterns["last_seen"] = _utc_now()

    def _notify_learning_agents(self, enriched: dict[str, Any]) -> None:
        for name, method_name in (("learning", "learn"), ("learning", "observe"), ("adaptive", "learn_from_feedback")):
            agent = self._agent(name)
            try:
                method = getattr(agent, method_name, None)
                if callable(method):
                    if method_name == "learn_from_feedback":
                        method({
                            "reward": _safe_float(enriched.get("reward"), 0.0),
                            "success": str(enriched.get("outcome", "")).lower() == "win",
                            "meta_context": {"game": self.game, "stats": self.stats, "adaptive": self.adaptive_state},
                        })
                    else:
                        method(enriched)
            except Exception as error:  # noqa: BLE001
                logger.warning("Chronos %s.%s update skipped: %s", name, method_name, error)

    def _save_learning_state(self) -> None:
        payload = {
            "schema": "chronos.learning_state.v2",
            "updated_at": _utc_now(),
            "stats": self.stats,
            "action_weights": self.action_weights,
            "phase_action_weights": self.phase_action_weights,
            "zone_weights": self.zone_weights,
            "move_signature_weights": self.move_signature_weights,
            "opening_signature_stats": self.opening_signature_stats,
            "opponent_patterns": self.opponent_patterns,
            "mistake_tracker": self.mistake_tracker,
            "adaptive_state": self.adaptive_state,
        }
        self.learning_store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.learning_store_path.with_suffix(self.learning_store_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.learning_store_path)
        self._learning_state_loaded = True
        self._learning_state_mtime = self.learning_store_path.stat().st_mtime
        self.learning_state_error = None

    def _load_learning_state(self) -> None:
        if not self.learning_store_path.exists():
            self._learning_state_loaded = False
            self._learning_state_mtime = 0.0
            return
        try:
            payload = json.loads(self.learning_store_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("learning state root must be an object")
            self.stats.update(self._sanitize_stats(payload.get("stats")))
            self.action_weights = self._sanitize_numeric_mapping(payload.get("action_weights"), -10.0, 14.0)
            self.phase_action_weights = self._sanitize_nested_numeric_mapping(payload.get("phase_action_weights"), -8.0, 10.0)
            loaded_zone = self._sanitize_numeric_mapping(payload.get("zone_weights"), -8.0, 10.0)
            for zone in ("core", "near_core", "perimeter"):
                self.zone_weights[zone] = loaded_zone.get(zone, self.zone_weights.get(zone, 0.0))
            self.move_signature_weights = self._sanitize_numeric_mapping(payload.get("move_signature_weights"), -12.0, 12.0)
            self.opening_signature_stats = self._sanitize_opening_stats(payload.get("opening_signature_stats"))
            if isinstance(payload.get("opponent_patterns"), dict):
                self.opponent_patterns.update(payload["opponent_patterns"])
            self.mistake_tracker = {
                str(key): max(0, _safe_int(value, 0))
                for key, value in (payload.get("mistake_tracker") or {}).items()
            } if isinstance(payload.get("mistake_tracker"), dict) else {}
            if isinstance(payload.get("adaptive_state"), dict):
                self.adaptive_state.update(payload["adaptive_state"])
                self.adaptive_state["confidence"] = _clamp(_safe_float(self.adaptive_state.get("confidence"), 0.5), 0.1, 0.95)
            self._learning_state_loaded = True
            self._learning_state_mtime = self.learning_store_path.stat().st_mtime
            self.learning_state_error = None
        except Exception as error:  # noqa: BLE001
            self.learning_state_error = f"{type(error).__name__}: {error}"
            self._learning_state_loaded = False
            logger.warning("Failed to load Chronos learning state: %s", self.learning_state_error)

    def _sync_learning_state_if_changed(self) -> None:
        if not self.learning_store_path.exists():
            return
        try:
            mtime = self.learning_store_path.stat().st_mtime
        except OSError:
            return
        if not self._learning_state_loaded or mtime > self._learning_state_mtime:
            self._load_learning_state()

    # ------------------------------------------------------------------
    # Scoring support helpers
    # ------------------------------------------------------------------

    def _phase_relevance_bonus(self, move_type: str, tactical: TacticalAnalysis, plan: StrategicPlan) -> float:
        bonus = 0.0
        if tactical.phase == "opening":
            bonus += {"move": 6.0, "claim": 4.0, "attack": 2.0, "pass": -6.0}.get(move_type, 0.0)
        elif tactical.phase == "midgame":
            bonus += {"attack": 7.0, "claim": 7.0, "move": 3.0, "pass": -8.0}.get(move_type, 0.0)
        elif tactical.phase in {"late", "endgame"}:
            bonus += {"claim": 9.0, "attack": 5.0, "move": 2.0, "pass": -4.0}.get(move_type, 0.0)
        if move_type in plan.preferred_actions:
            bonus += 6.0 - plan.preferred_actions.index(move_type)
        return bonus

    def _resource_efficiency(self, move: dict[str, Any], tactical: TacticalAnalysis, adaptive: AdaptiveProfile) -> float:
        token_id = move.get("tokenId", move.get("token_id"))
        token_cost = move.get("tokenCost", move.get("cost"))
        score = 0.0
        if isinstance(token_id, int):
            score += max(0, 8 - token_id) * (0.8 + adaptive.token_conservation * 0.4)
        if token_cost is not None:
            score -= _safe_float(token_cost, 0.0) * (1.2 + adaptive.token_conservation * 0.25)
        if tactical.phase in {"late", "endgame"} and str(move.get("type")) == "pass":
            score -= 6.0
        return score

    @staticmethod
    def _plan_alignment_score(move_type: str, zone: str, plan: StrategicPlan) -> float:
        score = 0.0
        if move_type in plan.preferred_actions:
            score += 11.0 - (plan.preferred_actions.index(move_type) * 2.5)
        if zone in plan.target_zones:
            score += 7.0
        plan_text = " ".join(plan.priorities + plan.agent_steps).lower()
        if "attack" in plan_text and move_type == "attack":
            score += 4.0
        if "claim" in plan_text and move_type == "claim":
            score += 4.0
        if "protect" in plan_text and move_type == "move":
            score += 2.0
        return score

    @staticmethod
    def _knowledge_score(move_type: str, zone: str, knowledge_context: str) -> float:
        text = knowledge_context.lower() if isinstance(knowledge_context, str) else ""
        score = 0.0
        if "core" in text and zone == "core":
            score += 7.0
        if any(word in text for word in ("attack", "aggressive", "capture")) and move_type == "attack":
            score += 5.5
        if any(word in text for word in ("protect", "safe", "preserve")) and move_type == "move":
            score += 3.0
        if "claim" in text and move_type == "claim":
            score += 5.0
        return score

    def _confidence_from_candidate(
        self,
        selected: CandidateMove,
        candidates: list[CandidateMove],
        tactical: TacticalAnalysis,
        adaptive: AdaptiveProfile,
    ) -> float:
        valid_scores = [candidate.score for candidate in candidates if candidate.safety.valid]
        if not valid_scores:
            return 0.0
        sorted_scores = sorted(valid_scores, reverse=True)
        margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0] - 10.0)
        score_component = _clamp((selected.score + 50.0) / 180.0, 0.05, 0.95)
        margin_component = _clamp(margin / 28.0, 0.0, 0.25)
        pressure_penalty = tactical.enemy_pressure * 0.13
        confidence = (adaptive.confidence * 0.48) + (score_component * 0.42) + margin_component - pressure_penalty
        return _clamp(confidence, 0.05, 0.97)

    def _next_best_valid_candidate(self, candidates: list[CandidateMove], normalized: dict[str, Any], rejected: set[str]) -> CandidateMove | None:
        for candidate in candidates:
            if candidate.canonical in rejected:
                continue
            safety = self._validate_candidate(candidate, normalized)
            if safety.valid:
                candidate.safety = safety
                return candidate
        return None

    @staticmethod
    def _extract_candidate_index(chosen: Any) -> int | None:
        if isinstance(chosen, dict):
            for key in ("index", "candidate_index"):
                if isinstance(chosen.get(key), int):
                    return int(chosen[key])
            name = chosen.get("name") or chosen.get("candidate") or chosen.get("selected")
            if isinstance(name, str) and name.startswith("candidate_"):
                try:
                    return int(name.split("_", 1)[1])
                except ValueError:
                    return None
        if isinstance(chosen, int):
            return chosen
        return None

    # ------------------------------------------------------------------
    # Board, unit, action helpers
    # ------------------------------------------------------------------

    def _choose_mutual_strategos_choice(self, normalized: dict[str, Any]) -> str:
        scores, _tokens = self._extract_player_metrics(normalized)
        units = normalized.get("units", [])
        ai_alive = [unit for unit in units if self._owner_of(unit) == AI_OWNER and _safe_int(unit.get("hp", unit.get("health", 1)), 1) > 0]
        human_alive = [unit for unit in units if self._owner_of(unit) == HUMAN_OWNER and _safe_int(unit.get("hp", unit.get("health", 1)), 1) > 0]
        ai_score = scores.get(AI_OWNER, 0.0) + sum(self._piece_value(unit) for unit in ai_alive)
        human_score = scores.get(HUMAN_OWNER, 0.0) + sum(self._piece_value(unit) for unit in human_alive)
        return "continue" if ai_score >= human_score else "end"

    def _board_size(self, game_state: dict[str, Any]) -> int:
        for key in ("boardSize", "board_size", "gridSize", "size"):
            if key in game_state:
                value = _safe_int(game_state.get(key), 0)
                if value > 0:
                    return value
        board = game_state.get("board")
        if isinstance(board, dict):
            for key in ("size", "boardSize", "rows", "height"):
                value = _safe_int(board.get(key), 0)
                if value > 0:
                    return value
        if isinstance(board, list) and board:
            return len(board)
        units = self._extract_units(game_state)
        max_axis = 8
        for unit in units:
            r, c = self._coords_of(unit)
            if isinstance(r, int):
                max_axis = max(max_axis, r)
            if isinstance(c, int):
                max_axis = max(max_axis, c)
        return max_axis + 1

    def _extract_units(self, game_state: dict[str, Any]) -> list[dict[str, Any]]:
        raw_units = game_state.get("units") or game_state.get("pieces") or game_state.get("tokens")
        if isinstance(raw_units, list):
            return [dict(unit) for unit in raw_units if isinstance(unit, dict)]
        board = game_state.get("board")
        units: list[dict[str, Any]] = []
        if isinstance(board, list):
            for r, row in enumerate(board):
                if not isinstance(row, list):
                    continue
                for c, cell in enumerate(row):
                    if isinstance(cell, dict) and ("unit" in cell or "piece" in cell):
                        unit = cell.get("unit") or cell.get("piece")
                        if isinstance(unit, dict):
                            unit = dict(unit)
                            unit.setdefault("r", r)
                            unit.setdefault("c", c)
                            units.append(unit)
                    elif isinstance(cell, dict) and "owner" in cell and "type" in cell:
                        unit = dict(cell)
                        unit.setdefault("r", r)
                        unit.setdefault("c", c)
                        units.append(unit)
        return units

    def _build_unit_map(self, units: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        unit_map: dict[str, dict[str, Any]] = {}
        for unit in units:
            unit_id = unit.get("id", unit.get("unitId", unit.get("uid")))
            if unit_id is None:
                continue
            normalized = dict(unit)
            normalized["id"] = str(unit_id)
            normalized["owner"] = self._owner_of(unit)
            r, c = self._coords_of(unit)
            normalized["r"] = r
            normalized["c"] = c
            normalized["hp"] = _safe_int(unit.get("hp", unit.get("health", 1)), 1)
            unit_map[str(unit_id)] = normalized
        return unit_map

    @staticmethod
    def _owner_of(unit: Mapping[str, Any]) -> Any:
        return unit.get("owner", unit.get("playerId", unit.get("player", unit.get("side"))))

    @staticmethod
    def _coords_of(payload: Mapping[str, Any]) -> tuple[int | None, int | None]:
        r = payload.get("r", payload.get("row", payload.get("y")))
        c = payload.get("c", payload.get("col", payload.get("x")))
        return (r if isinstance(r, int) else None, c if isinstance(c, int) else None)

    @staticmethod
    def _target_of(move: Mapping[str, Any]) -> dict[str, Any]:
        target = move.get("target")
        if isinstance(target, dict):
            return dict(target)
        params = move.get("params")
        if isinstance(params, dict) and isinstance(params.get("target"), dict):
            return dict(params["target"])
        return {}

    def _piece_value(self, unit: Mapping[str, Any]) -> float:
        return PIECE_VALUES.get(str(unit.get("type", unit.get("piece", "Scout"))), 1.0)

    def _extract_player_metrics(self, normalized: dict[str, Any]) -> tuple[dict[Any, float], dict[Any, int]]:
        scores: dict[Any, float] = {AI_OWNER: 0.0, HUMAN_OWNER: 0.0}
        token_counts: dict[Any, int] = {AI_OWNER: 0, HUMAN_OWNER: 0}
        for player in normalized.get("players", []):
            if not isinstance(player, dict):
                continue
            pid = player.get("id", player.get("playerId", player.get("owner")))
            scores[pid] = _safe_float(player.get("score"), 0.0)
            tokens = player.get("tokens", player.get("hand", []))
            token_counts[pid] = len(tokens) if isinstance(tokens, list) else _safe_int(tokens, 0)
        return scores, token_counts

    def _count_threatened_units(self, own_units: list[dict[str, Any]], enemy_units: list[dict[str, Any]]) -> int:
        threat_count = 0
        for own in own_units:
            own_pos = self._coords_of(own)
            if own_pos[0] is None or own_pos[1] is None:
                continue
            for enemy in enemy_units:
                enemy_pos = self._coords_of(enemy)
                if enemy_pos[0] is None or enemy_pos[1] is None:
                    continue
                distance = max(abs(own_pos[0] - enemy_pos[0]), abs(own_pos[1] - enemy_pos[1]))
                enemy_range = 2 if str(enemy.get("type")) == "Scout" else 1
                if distance <= enemy_range:
                    threat_count += 1
                    break
        return threat_count

    def _infer_destination(self, move: Mapping[str, Any], acting_unit: Mapping[str, Any]) -> tuple[int, int] | None:
        if str(move.get("type", "")).lower() in {"move", "claim"}:
            target = self._target_of(move)
            r, c = self._coords_of(target)
            if isinstance(r, int) and isinstance(c, int):
                return r, c
        r, c = self._coords_of(acting_unit)
        if isinstance(r, int) and isinstance(c, int):
            return r, c
        return None

    def _estimate_enemy_threat(
        self,
        destination: tuple[int, int] | None,
        acting_unit: Mapping[str, Any],
        unit_map: Mapping[str, dict[str, Any]],
    ) -> int:
        if destination is None:
            return 0
        dr, dc = destination
        owner = acting_unit.get("owner")
        threat = 0
        for enemy in unit_map.values():
            if enemy.get("owner") == owner:
                continue
            if _safe_int(enemy.get("hp", 1), 1) <= 0:
                continue
            er, ec = self._coords_of(enemy)
            if not isinstance(er, int) or not isinstance(ec, int):
                continue
            distance = max(abs(dr - er), abs(dc - ec))
            enemy_range = 2 if str(enemy.get("type")) == "Scout" else 1
            if distance <= enemy_range:
                threat += 1
        return threat

    def _zone_for_action(self, move: Mapping[str, Any], acting_unit: Mapping[str, Any] | None, board_size: int) -> str:
        target = self._target_of(move)
        r, c = self._coords_of(target)
        if (not isinstance(r, int) or not isinstance(c, int)) and acting_unit:
            r, c = self._coords_of(acting_unit)
        if isinstance(r, int) and isinstance(c, int):
            return self._classify_zone(r, c, board_size)
        return "perimeter"

    def _move_signature(self, move: Mapping[str, Any] | None, board_size: int) -> str:
        if not isinstance(move, Mapping):
            return ""
        move_type = str(move.get("type", "unknown")).lower()
        target = self._target_of(move)
        r, c = self._coords_of(target)
        if isinstance(r, int) and isinstance(c, int):
            return f"{move_type}:{self._classify_zone(r, c, board_size)}:{r}:{c}"
        target_id = target.get("id")
        if isinstance(target_id, str):
            return f"{move_type}:target:{target_id}"
        return move_type

    def _is_unit_in_core(self, unit: Mapping[str, Any], board_size: int) -> bool:
        r, c = self._coords_of(unit)
        return isinstance(r, int) and isinstance(c, int) and self._is_core_cell(r, c, board_size)

    @staticmethod
    def _distance_to_center(r: int, c: int, board_size: int) -> int:
        center = board_size // 2
        return max(abs(r - center), abs(c - center))

    @staticmethod
    def _is_core_cell(r: int, c: int, board_size: int) -> bool:
        center = board_size // 2
        return (center - 1) <= r <= (center + 1) and (center - 1) <= c <= (center + 1)

    def _classify_zone(self, r: int, c: int, board_size: int) -> str:
        if self._is_core_cell(r, c, board_size):
            return "core"
        center = board_size // 2
        if max(abs(r - center), abs(c - center)) <= 2:
            return "near_core"
        return "perimeter"

    def _update_adaptive_state(self, profile: AdaptiveProfile, tactical: TacticalAnalysis) -> None:
        history = self.adaptive_state.get("history") if isinstance(self.adaptive_state.get("history"), list) else []
        history.append({"timestamp": time.time(), "stance": profile.stance, "confidence": profile.confidence, "enemy_pressure": tactical.enemy_pressure, "core_balance": tactical.core_balance})
        self.adaptive_state.update(
            {
                "confidence": profile.confidence,
                "last_stance": profile.stance,
                "risk_appetite": (profile.risk_tolerance * 2.0) - 1.0,
                "opponent_aggression": tactical.enemy_pressure,
                "opponent_claim_pressure": max(0.0, -tactical.core_balance),
                "opponent_focus_core": max(0.0, -tactical.core_balance) / 3.0,
                "history": history[-100:],
            }
        )

    # ------------------------------------------------------------------
    # File and sanitization helpers
    # ------------------------------------------------------------------

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _sanitize_stats(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        return {
            "games_played": max(0, _safe_int(payload.get("games_played"), 0)),
            "wins": max(0, _safe_int(payload.get("wins"), 0)),
            "losses": max(0, _safe_int(payload.get("losses"), 0)),
            "draws": max(0, _safe_int(payload.get("draws"), 0)),
            "average_reward": _safe_float(payload.get("average_reward"), 0.0),
            "average_final_score": _safe_float(payload.get("average_final_score"), 0.0),
            "last_result": payload.get("last_result"),
            "last_updated": payload.get("last_updated"),
        }

    @staticmethod
    def _sanitize_numeric_mapping(payload: Any, lower: float, upper: float) -> dict[str, float]:
        if not isinstance(payload, dict):
            return {}
        clean: dict[str, float] = {}
        for key, value in payload.items():
            if not isinstance(key, str):
                continue
            clean[key] = _clamp(_safe_float(value, 0.0), lower, upper)
        return clean

    def _sanitize_nested_numeric_mapping(self, payload: Any, lower: float, upper: float) -> dict[str, dict[str, float]]:
        if not isinstance(payload, dict):
            return {}
        clean: dict[str, dict[str, float]] = {}
        for key, value in payload.items():
            if isinstance(key, str):
                clean[key] = self._sanitize_numeric_mapping(value, lower, upper)
        return clean

    @staticmethod
    def _sanitize_opening_stats(payload: Any) -> dict[str, dict[str, int]]:
        if not isinstance(payload, dict):
            return {}
        clean: dict[str, dict[str, int]] = {}
        for signature, stats in payload.items():
            if not isinstance(signature, str) or not isinstance(stats, dict):
                continue
            clean[signature] = {
                "games": max(0, _safe_int(stats.get("games"), 0)),
                "wins": max(0, _safe_int(stats.get("wins"), 0)),
                "losses": max(0, _safe_int(stats.get("losses"), 0)),
            }
        return clean

    def _prune_learning_maps(self) -> None:
        if len(self.move_signature_weights) > 2000:
            ranked = sorted(self.move_signature_weights.items(), key=lambda item: abs(item[1]), reverse=True)
            self.move_signature_weights = dict(ranked[:2000])
        if len(self.opening_signature_stats) > 350:
            ranked = sorted(self.opening_signature_stats.items(), key=lambda item: _safe_int(item[1].get("games"), 0), reverse=True)
            self.opening_signature_stats = dict(ranked[:350])
        if len(self.mistake_tracker) > 500:
            ranked = sorted(self.mistake_tracker.items(), key=lambda item: item[1], reverse=True)
            self.mistake_tracker = dict(ranked[:500])


_ai_player: ChronosAI | None = None


def initialize_ai() -> ChronosAI:
    """Initialize and cache the Chronos AI runtime."""

    global _ai_player
    if _ai_player is None:
        _ai_player = ChronosAI()
    return _ai_player
