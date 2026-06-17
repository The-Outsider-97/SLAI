"""Aether Shift SLAI multi-agent AI runtime.

This module upgrades Aether Shift from a frontend heuristic fallback into a
traceable, defensive, SLAI-powered strategic game-AI runtime while preserving
existing public integration points:

    initialize_ai()
    AetherShiftAI.health()
    AetherShiftAI.get_move(game_state)
    AetherShiftAI.learn_from_game(payload)

The runtime intentionally keeps the returned move compatible with the existing
frontend (`{cardId, actionIndex, target}`), and stores the richer response shape
in shared memory / diagnostics / local JSONL logs.
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
from typing import Any, Callable, Iterable, Optional, cast

games_root = Path(__file__).resolve().parent
project_root = games_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.agent_factory import AgentFactory # type: ignore
from src.agents.collaborative.shared_memory import SharedMemory # type: ignore
from src.agents.planning.planning_types import Task, TaskType # type: ignore
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]


logger = get_logger("Aether Shift")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Aether constants inferred from repository frontend/game logic
# ---------------------------------------------------------------------------
BOARD_SIZE = 5
POWER_WELLS = (
    {"row": 0, "col": 0},
    {"row": 0, "col": 4},
    {"row": 4, "col": 0},
    {"row": 4, "col": 4},
)
POWER_WELL_KEYS = {f"{well['row']},{well['col']}" for well in POWER_WELLS}
TILE_CONNECTIONS = {
    "STRAIGHT": [True, False, True, False],
    "CURVE": [True, True, False, False],
    "T_JUNCTION": [False, True, True, True],
    "CROSS": [True, True, True, True],
}
PLACE_TYPES = ("STRAIGHT", "CURVE", "T_JUNCTION")
ACTION_TYPES = ("PLACE", "SHIFT", "ROTATE", "ADVANCE", "ATTUNE")
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_BASE_WEIGHTS = {
    "ATTUNE": 38.0,
    "ADVANCE": 30.0,
    "PLACE": 18.0,
    "ROTATE": 14.0,
    "SHIFT": 13.0,
}
PHASES = ("opening", "midgame", "conversion")


@dataclass
class AgentSlot:
    name: str
    agent: Any
    available: bool
    reason: str | None = None


@dataclass
class AetherStateView:
    raw: dict[str, Any]
    board: list[list[Any]]
    players: dict[Any, Any]
    active_player: int
    opponent_id: int
    turn: int
    actions_remaining: int
    captured_wells: dict[str, int]
    valid_moves: list[dict[str, Any]]
    phase: str


@dataclass
class StrategicAnalysis:
    active_path: float
    opponent_path: float
    active_wells: int
    opponent_wells: int
    active_resonators: int
    opponent_resonators: int
    active_on_well: bool
    opponent_on_well: bool
    initiative: float
    pressure: float
    urgent_defense: bool
    conversion_ready: bool
    opponent_threat: float
    notes: list[str] = field(default_factory=list)


@dataclass
class AdaptiveProfile:
    stance: str
    confidence_bias: float
    risk_tolerance: float
    exploration: float
    action_bias: dict[str, float]
    notes: list[str] = field(default_factory=list)


@dataclass
class CandidateScore:
    move: dict[str, Any]
    action: str
    score: float
    confidence: float
    reason: str
    valid: bool
    safety: dict[str, Any]
    simulated: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Main runtime
# ---------------------------------------------------------------------------
@dataclass
class AetherShiftAI:
    game: str = "aether_shift"
    initialized_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self) -> None:
        self.shared_memory = SharedMemory()
        self.factory = AgentFactory()

        self.match_log_path = project_root / "logs" / "aether_matches.jsonl"
        self.decision_log_path = project_root / "logs" / "aether_decisions.jsonl"
        self.learning_store_path = project_root / "logs" / "aether_learning_state.json"
        self._learning_state_loaded = False
        self._learning_state_mtime = 0.0

        self.agent_slots: dict[str, AgentSlot] = {}
        for agent_type in (
            "knowledge",
            "planning",
            "execution",
            "learning",
            "adaptive",
            "safety",
            "evaluation",
            "quality",
            "observability",
        ):
            self.agent_slots[agent_type] = self._create_agent_slot(agent_type)

        self.knowledge_agent = self.agent_slots["knowledge"].agent
        self.planning_agent = self.agent_slots["planning"].agent
        self.execution_agent = self.agent_slots["execution"].agent
        self.learning_agent = self.agent_slots["learning"].agent
        self.adaptive_agent = self.agent_slots["adaptive"].agent
        self.safety_agent = self.agent_slots["safety"].agent
        self.evaluation_agent = self.agent_slots["evaluation"].agent
        self.quality_agent = self.agent_slots["quality"].agent
        self.observability_agent = self.agent_slots["observability"].agent

        self._planning_enabled = True
        self.last_decision_response: dict[str, Any] | None = None
        self.last_fallback_reason: str | None = None
        self.current_match_signatures: list[str] = []

        self.stats: dict[str, Any] = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "average_ai_score": 0.0,
            "average_opponent_score": 0.0,
            "average_reward": 0.0,
            "last_result": None,
            "last_updated": None,
        }
        self.action_weights: dict[str, float] = {}
        self.phase_action_weights: dict[str, dict[str, float]] = {phase: {} for phase in PHASES}
        self.move_signature_weights: dict[str, float] = {}
        self.opponent_profile: dict[str, Any] = {
            "path_pressure": 0.0,
            "well_pressure": 0.0,
            "shift_frequency": 0.0,
            "attune_frequency": 0.0,
            "last_seen_actions": [],
        }
        self.mistakes: list[dict[str, Any]] = []

        self._load_learning_state()
        self.shared_memory.set("aether_ai_status", "initialized")
        self.shared_memory.set("aether_stats", self.stats)
        self.shared_memory.set("aether_agent_availability", self.agent_availability())
        logger.info("Aether Shift AI initialized with SLAI multi-agent orchestration")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "agent_status": "ready",
            "initialized_at": self.initialized_at,
            "game": self.game,
            "agents": self.agent_availability(),
            "stats": self.stats,
            "action_weights": self.action_weights,
            "phase_action_weights": self.phase_action_weights,
            "move_signature_count": len(self.move_signature_weights),
            "opponent_profile": self.opponent_profile,
            "last_fallback_reason": self.last_fallback_reason,
            "last_decision_summary": self._last_decision_summary(),
            "logs": {
                "matches": str(self.match_log_path),
                "decisions": str(self.decision_log_path),
                "learning_state": str(self.learning_store_path),
            },
        }

    def get_move(self, game_state: dict[str, Any]) -> dict[str, Any] | None:
        """Return the raw move object expected by existing Aether clients."""

        response = self.get_move_response(game_state)
        move = response.get("move")
        return move if isinstance(move, dict) else None

    def get_move_response(self, game_state: dict[str, Any]) -> dict[str, Any]:
        """Return a full explainable move response for diagnostics/bridges."""

        started = time.time()
        trace: dict[str, Any] = {
            "normalization": "not_started",
            "tactical": "not_started",
            "knowledge": "not_started",
            "planning": "not_started",
            "candidate_generation": "not_started",
            "scoring": "not_started",
            "safety": "not_started",
            "execution": "not_started",
            "learning": "loaded" if self._learning_state_loaded else "cold_start",
            "adaptive": "not_started",
            "observability": "not_started",
        }

        if not isinstance(game_state, dict):
            return self._controlled_fallback_response(
                reason="game_state_must_be_object",
                trace=trace,
                started=started,
            )

        self._sync_learning_state_if_changed()
        self.shared_memory.set("aether_last_raw_state", game_state)

        state_view = self._normalize_state(game_state, trace)
        if state_view is None:
            return self._controlled_fallback_response(
                reason="state_normalization_failed",
                trace=trace,
                started=started,
            )

        analysis = self._analyze_state(state_view)
        trace["tactical"] = {
            "phase": state_view.phase,
            "active_path": round(analysis.active_path, 2),
            "opponent_path": round(analysis.opponent_path, 2),
            "active_wells": analysis.active_wells,
            "opponent_wells": analysis.opponent_wells,
            "urgent_defense": analysis.urgent_defense,
            "conversion_ready": analysis.conversion_ready,
            "notes": analysis.notes,
        }

        knowledge_context = self._get_strategy_context(state_view, analysis, trace)
        adaptive_profile = self._build_adaptive_profile(state_view, analysis, trace)
        plan = self._generate_plan(state_view, analysis, knowledge_context, adaptive_profile, trace)

        candidates = self._generate_candidates(state_view, trace)
        if not candidates:
            return self._controlled_fallback_response(
                reason="no_legal_actions_available",
                trace=trace,
                started=started,
            )

        scored = self._score_candidates(
            candidates=candidates,
            state_view=state_view,
            analysis=analysis,
            knowledge_context=knowledge_context,
            plan=plan,
            adaptive_profile=adaptive_profile,
            trace=trace,
        )
        valid_scored = [candidate for candidate in scored if candidate.valid]
        if not valid_scored:
            return self._controlled_fallback_response(
                reason="no_candidate_passed_safety_validation",
                trace=trace,
                started=started,
                top_candidates=scored[:5],
            )

        selected = self._arbitrate_execution(valid_scored, state_view, analysis, adaptive_profile, trace)
        if selected is None:
            selected = valid_scored[0]
            trace["execution"] = "execution_agent_unavailable_best_score_used"

        final_safety = self._validate_move(selected.move, state_view)
        if not final_safety.get("valid"):
            trace["safety"] = {"final_validation_failed": final_safety}
            selected = next((candidate for candidate in valid_scored if self._validate_move(candidate.move, state_view).get("valid")), None)
            if selected is None:
                return self._controlled_fallback_response(
                    reason="final_safety_rejected_all_candidates",
                    trace=trace,
                    started=started,
                    top_candidates=valid_scored[:5],
                )

        move_signature = self._move_signature(selected.move, state_view)
        if move_signature:
            self.current_match_signatures.append(move_signature)
            self.current_match_signatures = self.current_match_signatures[-200:]

        confidence = self._confidence_from_score(selected, valid_scored, adaptive_profile)
        top_candidates = [self._candidate_to_debug(candidate) for candidate in valid_scored[:6]]
        response = {
            "move": selected.move,
            "confidence": confidence,
            "strategy": self._strategy_label(state_view, analysis, adaptive_profile),
            "reasoning": selected.reason,
            "agent_trace": {
                "knowledge": trace.get("knowledge"),
                "planning": trace.get("planning"),
                "execution": trace.get("execution"),
                "learning": trace.get("learning"),
                "adaptive": trace.get("adaptive"),
                "safety": final_safety,
                "evaluation": trace.get("evaluation"),
                "quality": trace.get("quality"),
                "observability": trace.get("observability"),
            },
            "fallback": False,
            "fallback_reason": None,
            "debug": {
                "elapsed_ms": round((time.time() - started) * 1000.0, 3),
                "phase": state_view.phase,
                "valid_move_count": len(state_view.valid_moves),
                "candidate_count": len(scored),
                "top_candidates": top_candidates,
                "analysis": trace.get("tactical"),
                "move_signature": move_signature,
            },
        }

        self.last_decision_response = response
        self.last_fallback_reason = None
        self.shared_memory.set("aether_last_decision", response)
        self._append_jsonl(self.decision_log_path, self._redact_for_log(response))
        self._emit_observability_event("aether.move.selected", response)
        return response

    def learn_from_game(self, payload: dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False

        try:
            self._sync_learning_state_if_changed()
            enriched_payload = {
                **payload,
                "game": self.game,
                "logged_at": datetime.utcnow().isoformat(),
            }
            self._append_jsonl(self.match_log_path, enriched_payload)

            self._call_agent("learning", "learn", enriched_payload)
            self._update_score_stats(payload)
            self._update_action_preferences(payload)
            self._update_signature_preferences(payload)
            self._update_opponent_profile(payload)
            self._save_learning_state()

            self.shared_memory.set("aether_last_game", enriched_payload)
            self.shared_memory.set("aether_stats", self.stats)
            return True
        except Exception as error:  # noqa: BLE001
            logger.warning("Aether Shift learning update failed: %s", error)
            return False

    def training_status(self) -> dict[str, Any]:
        return {
            "available": True,
            "active": False,
            "mode": "local_outcome_learning",
            "note": "Aether has local learning persistence but no dedicated repository self-play trainer in this package.",
            "stats": self.stats,
            "action_weights": self.action_weights,
            "phase_action_weights": self.phase_action_weights,
            "move_signature_count": len(self.move_signature_weights),
            "match_log_path": str(self.match_log_path),
            "learning_state_path": str(self.learning_store_path),
        }

    def agent_availability(self) -> dict[str, dict[str, Any]]:
        return {
            name: {"available": slot.available, "reason": slot.reason}
            for name, slot in self.agent_slots.items()
        }

    # ------------------------------------------------------------------
    # Agent orchestration
    # ------------------------------------------------------------------

    def _create_agent_slot(self, agent_type: str) -> AgentSlot:
        try:
            agent = self.factory.create(agent_type, self.shared_memory)
            return AgentSlot(name=agent_type, agent=agent, available=True)
        except Exception as error:  # noqa: BLE001
            logger.warning("Aether %s agent unavailable: %s", agent_type, error)
            return AgentSlot(
                name=agent_type,
                agent=None,
                available=False,
                reason=str(error),
            )

    def _call_agent(self, agent_type: str, method_name: str, *args: Any, **kwargs: Any) -> Any:
        slot = self.agent_slots.get(agent_type)
        if not slot or not slot.available:
            return None
        method = getattr(slot.agent, method_name, None)
        if not callable(method):
            return None
        try:
            return method(*args, **kwargs)
        except Exception as error:  # noqa: BLE001
            logger.warning("Aether %s.%s failed: %s", agent_type, method_name, error)
            slot.available = False
            slot.reason = f"{method_name} failed: {error}"
            return None

    def _get_strategy_context(
        self,
        state_view: AetherStateView,
        analysis: StrategicAnalysis,
        trace: dict[str, Any],
    ) -> str:
        baseline = (
            "Aether Shift strategy: maximize connected path progress, deny opponent edge connection, "
            "use shifts and rotations to break opponent tempo, and conserve resonators for Power Wells or forced locks."
        )
        query = (
            f"Aether Shift {state_view.phase} strategy: active_path={analysis.active_path:.1f}, "
            f"opponent_path={analysis.opponent_path:.1f}, active_wells={analysis.active_wells}, "
            f"opponent_wells={analysis.opponent_wells}, actions_remaining={state_view.actions_remaining}. "
            "Retrieve useful patterns, bad move patterns, well-control tactics, shift/rotation traps, and conversion plans."
        )
        result = self._call_agent("knowledge", "query", query)
        if result:
            context = str(result)
            trace["knowledge"] = {
                "status": "used",
                "query": query[:180],
                "context_preview": context[:240],
            }
            self.shared_memory.set("aether_strategy_context", context)
            return context
        trace["knowledge"] = {"status": "fallback", "context_preview": baseline}
        return baseline

    def _generate_plan(
        self,
        state_view: AetherStateView,
        analysis: StrategicAnalysis,
        strategy_context: str,
        adaptive_profile: AdaptiveProfile,
        trace: dict[str, Any],
    ) -> list[Any] | None:
        if not self._planning_enabled:
            trace["planning"] = "disabled_after_previous_failure"
            return None

        now = time.time()
        fallback_task = Task(
            name=f"aether_select_tactical_move_fallback_{int(now)}",
            task_type=TaskType.PRIMITIVE,
            start_time=now + 1,
            deadline=now + 120,
            duration=10,
            context={"game": self.game, "phase": state_view.phase},
        )
        goal_task = Task(
            name=f"aether_select_best_{state_view.phase}_move_{int(now)}",
            task_type=TaskType.ABSTRACT,
            start_time=now,
            deadline=now + 180,
            duration=20,
            methods=[[fallback_task]],
            goal_state={"move_selected": True, "legal": True},
            context={
                "game": self.game,
                "phase": state_view.phase,
                "turn": state_view.turn,
                "actions_remaining": state_view.actions_remaining,
                "analysis": {
                    "active_path": analysis.active_path,
                    "opponent_path": analysis.opponent_path,
                    "active_wells": analysis.active_wells,
                    "opponent_wells": analysis.opponent_wells,
                    "urgent_defense": analysis.urgent_defense,
                    "conversion_ready": analysis.conversion_ready,
                },
                "strategy_context": strategy_context,
                "adaptive_profile": adaptive_profile.__dict__,
            },
        )

        try:
            self._call_agent("planning", "register_task", goal_task)
            plan = self._call_agent("planning", "generate_plan", goal_task)
            if isinstance(plan, list):
                trace["planning"] = {
                    "status": "used",
                    "steps": [getattr(step, "name", str(step)) for step in plan[:8]],
                    "step_count": len(plan),
                }
                self.shared_memory.set("aether_last_plan", trace["planning"])
                return plan
        except Exception as error:  # noqa: BLE001
            logger.warning("Aether planning failed: %s", error)
            self._planning_enabled = False
        trace["planning"] = {
            "status": "fallback",
            "objective": self._strategy_label(state_view, analysis, adaptive_profile),
        }
        return [fallback_task]

    def _build_adaptive_profile(
        self,
        state_view: AetherStateView,
        analysis: StrategicAnalysis,
        trace: dict[str, Any],
    ) -> AdaptiveProfile:
        if analysis.urgent_defense:
            stance = "defensive_disruption"
            risk_tolerance = 0.28
            action_bias = {"SHIFT": 7.5, "ROTATE": 6.0, "ADVANCE": 1.5, "ATTUNE": 1.0, "PLACE": 2.0}
            notes = ["opponent_conversion_threat"]
        elif analysis.conversion_ready:
            stance = "conversion"
            risk_tolerance = 0.36
            action_bias = {"ATTUNE": 8.5, "ADVANCE": 6.0, "PLACE": 2.0, "ROTATE": 1.5, "SHIFT": 0.5}
            notes = ["secure_well_or_finish_path"]
        elif analysis.active_path < analysis.opponent_path - 20:
            stance = "comeback"
            risk_tolerance = 0.72
            action_bias = {"ADVANCE": 5.5, "PLACE": 5.0, "SHIFT": 4.0, "ROTATE": 3.0, "ATTUNE": 2.0}
            notes = ["recover_path_tempo"]
        else:
            stance = "balanced_control"
            risk_tolerance = 0.48
            action_bias = {"ADVANCE": 4.0, "ATTUNE": 4.5, "PLACE": 3.2, "ROTATE": 2.5, "SHIFT": 2.2}
            notes = ["develop_path_and_wells"]

        confidence_bias = max(
            -0.2,
            min(0.25, (analysis.active_path - analysis.opponent_path) / 350.0 + (analysis.active_wells - analysis.opponent_wells) * 0.04),
        )
        exploration = max(0.15, min(1.3, 0.75 - confidence_bias + max(0.0, analysis.pressure) * 0.25))

        response = self._call_agent(
            "adaptive",
            "perform_task",
            {
                "type": "aether_adaptation",
                "goal": "adjust_move_priorities",
                "context": {
                    "stance": stance,
                    "phase": state_view.phase,
                    "pressure": analysis.pressure,
                    "opponent_threat": analysis.opponent_threat,
                    "initiative": analysis.initiative,
                    "opponent_profile": self.opponent_profile,
                },
            },
        )
        if isinstance(response, dict):
            maybe_conf = response.get("confidence")
            if maybe_conf is None and isinstance(response.get("policy_metrics"), dict):
                maybe_conf = response["policy_metrics"].get("recent_success_rate")
            if maybe_conf is not None:
                try:
                    confidence_bias += (float(maybe_conf) - 0.5) * 0.18
                except (TypeError, ValueError):
                    pass
            notes.append("adaptive_agent_signal")
            trace["adaptive"] = {"status": "used", "response_preview": str(response)[:220]}
        else:
            trace["adaptive"] = {"status": "fallback", "stance": stance}

        return AdaptiveProfile(
            stance=stance,
            confidence_bias=max(-0.3, min(0.3, confidence_bias)),
            risk_tolerance=risk_tolerance,
            exploration=exploration,
            action_bias=action_bias,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Normalization, legal moves, tactical analysis
    # ------------------------------------------------------------------

    def _normalize_state(self, game_state: dict[str, Any], trace: dict[str, Any]) -> AetherStateView | None:
        try:
            board = game_state.get("board")
            if not self._valid_board(board):
                trace["normalization"] = "invalid_board"
                return None

            players = game_state.get("players")
            if not isinstance(players, dict):
                trace["normalization"] = "invalid_players"
                return None

            active_player = self._coerce_int(game_state.get("activePlayer", 2), default=2)
            if active_player not in (1, 2):
                active_player = 2
            opponent_id = 1 if active_player == 2 else 2
            turn = max(1, self._coerce_int(game_state.get("turn", 1), default=1))
            actions_remaining = max(0, min(2, self._coerce_int(game_state.get("actionsRemaining", 2), default=2)))

            captured_wells = self._sanitize_captured_wells(game_state.get("capturedWells"))
            valid_moves = game_state.get("validMoves") if isinstance(game_state.get("validMoves"), list) else []
            valid_moves = self._dedupe_moves([move for move in valid_moves if isinstance(move, dict)])
            if not valid_moves:
                valid_moves = self._infer_valid_moves(game_state)

            legal_moves = []
            temp_state = {**game_state, "validMoves": valid_moves}
            temp_view = AetherStateView(
                raw=temp_state,
                board=board,
                players=players,
                active_player=active_player,
                opponent_id=opponent_id,
                turn=turn,
                actions_remaining=actions_remaining,
                captured_wells=captured_wells,
                valid_moves=valid_moves,
                phase="opening",
            )
            for move in valid_moves:
                safety = self._validate_move(move, temp_view, require_membership=False)
                if safety.get("valid"):
                    legal_moves.append(move)

            phase = self._classify_phase(turn, captured_wells, game_state, active_player, opponent_id)
            view = AetherStateView(
                raw={**game_state, "validMoves": legal_moves, "capturedWells": captured_wells},
                board=board,
                players=players,
                active_player=active_player,
                opponent_id=opponent_id,
                turn=turn,
                actions_remaining=actions_remaining,
                captured_wells=captured_wells,
                valid_moves=legal_moves,
                phase=phase,
            )
            trace["normalization"] = {
                "status": "ok",
                "active_player": active_player,
                "turn": turn,
                "actions_remaining": actions_remaining,
                "legal_moves": len(legal_moves),
                "phase": phase,
            }
            return view
        except Exception as error:  # noqa: BLE001
            logger.warning("Aether normalization failed: %s", error)
            trace["normalization"] = f"error:{error}"
            return None

    def _analyze_state(self, state_view: AetherStateView) -> StrategicAnalysis:
        active = state_view.active_player
        opponent = state_view.opponent_id
        active_path = self._path_progress(state_view.raw, active)
        opponent_path = self._path_progress(state_view.raw, opponent)
        active_wells = self._captured_count(state_view.raw, active)
        opponent_wells = self._captured_count(state_view.raw, opponent)
        active_player = self._get_player(state_view.raw, active) or {}
        opponent_player = self._get_player(state_view.raw, opponent) or {}
        active_resonators = self._coerce_int(active_player.get("resonators"), 0)
        opponent_resonators = self._coerce_int(opponent_player.get("resonators"), 0)
        active_on_well = self._is_on_power_well(state_view.raw, active)
        opponent_on_well = self._is_on_power_well(state_view.raw, opponent)
        initiative = (active_path - opponent_path) / 100.0 + (active_wells - opponent_wells) * 0.55
        opponent_threat = (opponent_path / 100.0) + opponent_wells * 0.4 + (0.18 if opponent_on_well else 0.0)
        pressure = max(0.0, opponent_threat - ((active_path / 120.0) + active_wells * 0.28))
        urgent_defense = opponent_path >= 75 or opponent_wells >= 2 or opponent_threat >= 1.2
        conversion_ready = active_path >= 75 or active_wells >= 2 or (active_on_well and active_resonators > 0)
        notes: list[str] = []
        if urgent_defense:
            notes.append("opponent_near_conversion")
        if conversion_ready:
            notes.append("active_conversion_window")
        if active_resonators <= 1:
            notes.append("low_resonator_supply")
        return StrategicAnalysis(
            active_path=active_path,
            opponent_path=opponent_path,
            active_wells=active_wells,
            opponent_wells=opponent_wells,
            active_resonators=active_resonators,
            opponent_resonators=opponent_resonators,
            active_on_well=active_on_well,
            opponent_on_well=opponent_on_well,
            initiative=initiative,
            pressure=pressure,
            urgent_defense=urgent_defense,
            conversion_ready=conversion_ready,
            opponent_threat=opponent_threat,
            notes=notes,
        )

    def _generate_candidates(self, state_view: AetherStateView, trace: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = []
        for move in state_view.valid_moves:
            safety = self._validate_move(move, state_view)
            if safety.get("valid"):
                candidates.append(move)
        trace["candidate_generation"] = {
            "input_valid_moves": len(state_view.valid_moves),
            "candidate_count": len(candidates),
        }
        return candidates

    def _infer_valid_moves(self, game_state: dict[str, Any]) -> list[dict[str, Any]]:
        moves: list[dict[str, Any]] = []
        face_up_cards = game_state.get("faceUpCards", [])
        if not isinstance(face_up_cards, list):
            return []
        for card in face_up_cards:
            if not isinstance(card, dict):
                continue
            actions = card.get("actions", [])
            if not isinstance(actions, list):
                continue
            for action_index, action in enumerate(actions):
                for target in self._valid_targets_for_action(game_state, str(action).upper()):
                    moves.append({"cardId": card.get("id"), "actionIndex": action_index, "target": target})
        return self._dedupe_moves(moves)

    def _valid_targets_for_action(self, game_state: dict[str, Any], action: str) -> list[dict[str, int]]:
        board = game_state.get("board")
        active_player = self._coerce_int(game_state.get("activePlayer", 2), default=2)
        player = self._get_player(game_state, active_player)
        if not self._valid_board(board) or not isinstance(player, dict):
            return []
        board = cast(list[list[Any]], board)
        targets: list[dict[str, int]] = []

        if action == "PLACE":
            position = player.get("position")
            coords = self._extract_position_coords(position)
            if coords is None:
                return []
            row0, col0 = coords
            for d_row, d_col in DIRECTIONS:
                row = row0 + d_row
                col = col0 + d_col
                if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row][col] is None:
                    targets.append({"row": row, "col": col})
            return targets

        if action == "ROTATE":
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    tile = board[row][col]
                    if tile is None:
                        continue
                    if isinstance(tile, dict) and tile.get("hasResonator") and self._coerce_int(tile.get("resonatorOwner"), 0) != active_player:
                        continue
                    targets.append({"row": row, "col": col})
            return targets

        if action == "ADVANCE":
            position = player.get("position")
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    if self._is_valid_move(board, position, {"row": row, "col": col}):
                        targets.append({"row": row, "col": col})
            return targets

        if action == "ATTUNE":
            coords = self._extract_position_coords(player.get("position"))
            if coords is None:
                return []
            row, col = coords
            tile = board[row][col]
            if isinstance(tile, dict) and not tile.get("hasResonator") and self._coerce_int(player.get("resonators"), 0) > 0:
                # Match frontend rules: grey/neutral Power Wells cannot be captured by attune.
                if self._is_power_well(row, col) and tile.get("color") == "neutral":
                    return []
                targets.append({"row": row, "col": col})
            return targets

        if action == "SHIFT":
            for col in range(BOARD_SIZE):
                targets.append({"row": 0, "col": col})
                targets.append({"row": BOARD_SIZE - 1, "col": col})
            for row in range(1, BOARD_SIZE - 1):
                targets.append({"row": row, "col": 0})
                targets.append({"row": row, "col": BOARD_SIZE - 1})
            return targets

        return []

    # ------------------------------------------------------------------
    # Scoring and arbitration
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        *,
        candidates: list[dict[str, Any]],
        state_view: AetherStateView,
        analysis: StrategicAnalysis,
        knowledge_context: str,
        plan: list[Any] | None,
        adaptive_profile: AdaptiveProfile,
        trace: dict[str, Any],
    ) -> list[CandidateScore]:
        scored: list[CandidateScore] = []
        for move in candidates:
            scored.append(
                self._score_move(
                    move=move,
                    state_view=state_view,
                    analysis=analysis,
                    knowledge_context=knowledge_context,
                    plan=plan,
                    adaptive_profile=adaptive_profile,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        trace["scoring"] = {
            "scored_count": len(scored),
            "best_score": round(scored[0].score, 3) if scored else None,
            "best_action": scored[0].action if scored else None,
        }
        trace["evaluation"] = self._agent_quality_evaluation(scored[:5], state_view, analysis)
        return scored

    def _score_move(
        self,
        *,
        move: dict[str, Any],
        state_view: AetherStateView,
        analysis: StrategicAnalysis,
        knowledge_context: str,
        plan: list[Any] | None,
        adaptive_profile: AdaptiveProfile,
    ) -> CandidateScore:
        safety = self._validate_move(move, state_view)
        action = self._resolve_action(state_view.raw, move)
        if not safety.get("valid") or not action:
            return CandidateScore(
                move=move,
                action=action or "UNKNOWN",
                score=-100000.0,
                confidence=0.0,
                reason=f"safety_rejected:{safety.get('reason')}",
                valid=False,
                safety=safety,
            )

        simulated = self._simulate_move(state_view.raw, move, action)
        if simulated is None:
            return CandidateScore(
                move=move,
                action=action,
                score=-90000.0,
                confidence=0.0,
                reason="simulation_failed",
                valid=False,
                safety={"valid": False, "reason": "simulation_failed"},
            )

        active = state_view.active_player
        opponent = state_view.opponent_id
        before_active_eval = self._evaluate_state(state_view.raw, active)
        before_opp_eval = self._evaluate_state(state_view.raw, opponent)
        after_active_eval = self._evaluate_state(simulated, active)
        after_opp_eval = self._evaluate_state(simulated, opponent)

        score = ACTION_BASE_WEIGHTS.get(action, -15.0)
        score += adaptive_profile.action_bias.get(action, 0.0)
        score += self.action_weights.get(action, 0.0)
        score += self.phase_action_weights.get(state_view.phase, {}).get(action, 0.0)
        score += self.move_signature_weights.get(self._move_signature(move, state_view), 0.0)
        reasons = [f"action={action}", f"phase={state_view.phase}", f"stance={adaptive_profile.stance}"]

        winner = simulated.get("winner")
        if winner == active:
            return CandidateScore(
                move=move,
                action=action,
                score=750000.0,
                confidence=0.99,
                reason="immediate_win",
                valid=True,
                safety=safety,
                simulated=simulated,
            )
        if winner == opponent:
            score -= 750000.0
            reasons.append("self_loss_penalty")

        active_delta = after_active_eval - before_active_eval
        opp_delta = after_opp_eval - before_opp_eval
        score += active_delta * 0.18
        score -= opp_delta * 0.22
        reasons.append(f"active_delta={active_delta:.1f}")
        reasons.append(f"opp_delta={opp_delta:.1f}")

        progress_before = analysis.active_path
        progress_after = self._path_progress(simulated, active)
        opp_progress_after = self._path_progress(simulated, opponent)
        progress_gain = progress_after - progress_before
        opponent_progress_reduction = max(0.0, analysis.opponent_path - opp_progress_after)
        well_before = analysis.active_wells
        well_after = self._captured_count(simulated, active)
        opp_well_after = self._captured_count(simulated, opponent)
        well_gain = well_after - well_before

        if progress_gain > 0:
            score += progress_gain * (3.6 if action == "ADVANCE" else 2.0)
            reasons.append(f"path_gain={progress_gain:.1f}")
        if opponent_progress_reduction > 0:
            score += opponent_progress_reduction * (3.0 if action in {"SHIFT", "ROTATE"} else 1.2)
            reasons.append(f"disrupts_opponent_path={opponent_progress_reduction:.1f}")
        if well_gain > 0:
            score += well_gain * 13000.0
            reasons.append(f"well_gain={well_gain}")
        if opp_well_after < analysis.opponent_wells:
            score += (analysis.opponent_wells - opp_well_after) * 2500.0
            reasons.append("breaks_opponent_well_control")

        target = move.get("target") if isinstance(move.get("target"), dict) else {}
        target_row, target_col = self._target_coords(target)
        if target_row is not None and target_col is not None:
            zone_value = self._target_zone_value(target_row, target_col, active)
            score += zone_value
            reasons.append(f"target_value={zone_value:.1f}")
            if self._is_power_well(target_row, target_col):
                score += 500.0 if action in {"ATTUNE", "ADVANCE", "PLACE"} else 120.0
                reasons.append("power_well_target")

        if action == "ATTUNE":
            if analysis.active_resonators <= 1 and well_gain <= 0 and progress_after < 80:
                score -= 180.0
                reasons.append("conserves_last_resonator_penalty")
            if well_after >= 2:
                score += 850.0
                reasons.append("conversion_pressure")
            if well_after >= 3:
                score += 50000.0
                reasons.append("three_well_win_pressure")

        if action == "PLACE":
            connectivity = self._local_connectivity_potential(simulated, target_row, target_col) if target_row is not None else 0
            score += connectivity * 35.0
            reasons.append(f"place_connectivity={connectivity}")

        if action == "ROTATE":
            if progress_gain <= 0 and opponent_progress_reduction <= 0:
                score -= 20.0
                reasons.append("low_rotation_impact")

        if action == "SHIFT":
            if opponent_progress_reduction <= 0 and progress_gain <= 0 and not analysis.urgent_defense:
                score -= 18.0
                reasons.append("low_shift_impact")
            if self._player_displaced(simulated, state_view.raw, opponent):
                score += 70.0
                reasons.append("opponent_displaced")

        # Two-action tempo: inspect best follow-up when AI keeps the turn.
        if self._coerce_int(simulated.get("actionsRemaining"), 0) > 0 and self._coerce_int(simulated.get("activePlayer"), 0) == active:
            followup = self._best_follow_up_score(simulated, active)
            score += followup * 0.045
            reasons.append(f"followup={followup:.1f}")

        if self._coerce_int(simulated.get("activePlayer"), 0) == opponent:
            reply = self._best_follow_up_score(simulated, opponent, max_moves=40)
            risk_multiplier = 0.07 + (0.08 * (1.0 - adaptive_profile.risk_tolerance))
            score -= reply * risk_multiplier
            reasons.append(f"reply_risk={reply:.1f}")

        lowered_context = knowledge_context.lower()
        if "well" in lowered_context and action == "ATTUNE":
            score += 18.0
            reasons.append("knowledge_well_signal")
        if "shift" in lowered_context and action == "SHIFT":
            score += 12.0
            reasons.append("knowledge_shift_signal")
        if "path" in lowered_context and action in {"PLACE", "ADVANCE", "ROTATE"}:
            score += 10.0
            reasons.append("knowledge_path_signal")

        if plan:
            plan_text = " ".join(str(getattr(step, "name", step)).lower() for step in plan)
            if action.lower() in plan_text or state_view.phase in plan_text:
                score += min(40.0, 8.0 + len(plan) * 2.0)
                reasons.append("plan_aligned")

        # Controlled non-determinism only inside close bands, decayed by experience.
        games_played = max(0, self._coerce_int(self.stats.get("games_played"), 0))
        exploration_scale = max(0.15, 1.0 - min(0.75, games_played / 600.0))
        score += random.uniform(0.0, adaptive_profile.exploration * 3.5 * exploration_scale)

        if safety.get("warnings"):
            score -= len(safety["warnings"]) * 8.0
            reasons.append(f"safety_warnings={len(safety['warnings'])}")

        confidence = max(0.02, min(0.97, 0.48 + adaptive_profile.confidence_bias + min(0.28, score / 200000.0)))
        return CandidateScore(
            move=move,
            action=action,
            score=score,
            confidence=confidence,
            reason="; ".join(reasons),
            valid=True,
            safety=safety,
            simulated=simulated,
        )

    def _arbitrate_execution(
        self,
        candidates: list[CandidateScore],
        state_view: AetherStateView,
        analysis: StrategicAnalysis,
        adaptive_profile: AdaptiveProfile,
        trace: dict[str, Any],
    ) -> CandidateScore | None:
        candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
        if not candidates:
            return None

        best_score = candidates[0].score
        top_band = [candidate for candidate in candidates if candidate.score >= best_score - max(4.0, abs(best_score) * 0.01)]
        selected = top_band[0]

        action_candidates = [
            {
                "name": f"candidate_{index}",
                "priority": max(1, int(candidate.score + 100000)),
                "move": candidate.move,
                "action": candidate.action,
                "score": candidate.score,
                "reason": candidate.reason,
            }
            for index, candidate in enumerate(candidates[:10])
        ]
        execution_context = {
            "game": self.game,
            "phase": state_view.phase,
            "turn": state_view.turn,
            "actions_remaining": state_view.actions_remaining,
            "analysis": analysis.__dict__,
            "adaptive": adaptive_profile.__dict__,
        }

        agent = self.agent_slots.get("execution", AgentSlot("execution", None, False)).agent
        try:
            selector = getattr(agent, "action_selector", None)
            if selector is not None and hasattr(selector, "select"):
                result = selector.select(action_candidates, execution_context)
                selected_name = result.get("name") if isinstance(result, dict) else None
                if isinstance(selected_name, str) and selected_name.startswith("candidate_"):
                    idx = int(selected_name.split("_", 1)[1])
                    if 0 <= idx < len(candidates):
                        candidate = candidates[idx]
                        if self._validate_move(candidate.move, state_view).get("valid"):
                            selected = candidate
                            trace["execution"] = {"status": "agent_selector_used", "selected": selected_name}
                            return selected
        except Exception as error:  # noqa: BLE001
            logger.warning("Aether execution arbitration skipped: %s", error)

        prediction = self._call_agent("execution", "predict", execution_context)
        if isinstance(prediction, dict):
            trace["execution"] = {"status": "predict_signal", "preview": str(prediction)[:220], "selected": "best_scored"}
        else:
            trace["execution"] = {"status": "fallback_best_scored"}
        return selected

    # ------------------------------------------------------------------
    # Safety validation and simulation
    # ------------------------------------------------------------------

    def _validate_move(self, move: dict[str, Any], state_view: AetherStateView, *, require_membership: bool = True) -> dict[str, Any]:
        warnings: list[str] = []
        if not isinstance(move, dict):
            return {"valid": False, "reason": "move_not_object"}
        if "cardId" not in move:
            return {"valid": False, "reason": "missing_cardId"}
        if "actionIndex" not in move:
            return {"valid": False, "reason": "missing_actionIndex"}
        if not isinstance(move.get("actionIndex"), int):
            return {"valid": False, "reason": "actionIndex_must_be_int"}
        target = move.get("target")
        if not isinstance(target, dict):
            return {"valid": False, "reason": "missing_target"}
        row, col = self._target_coords(target)
        if row is None or col is None:
            return {"valid": False, "reason": "target_row_col_must_be_int"}
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return {"valid": False, "reason": "target_out_of_bounds"}

        action = self._resolve_action(state_view.raw, move)
        if action not in ACTION_TYPES:
            return {"valid": False, "reason": "unknown_or_unavailable_action", "action": action}

        if require_membership and not self._move_in_legal_set(move, state_view.valid_moves):
            return {"valid": False, "reason": "move_not_in_legal_moves", "action": action}

        if action == "ATTUNE":
            tile = state_view.board[row][col]
            if isinstance(tile, dict) and self._is_power_well(row, col) and tile.get("color") == "neutral":
                return {"valid": False, "reason": "cannot_capture_neutral_power_well", "action": action}

        simulated = self._simulate_move(state_view.raw, move, action)
        if simulated is None:
            return {"valid": False, "reason": "simulation_rejected_move", "action": action}

        quality_result = self._call_agent(
            "safety",
            "perform_task",
            {
                "type": "aether_move_validation",
                "move": move,
                "action": action,
                "phase": state_view.phase,
                "strict": True,
            },
        )
        if isinstance(quality_result, dict) and quality_result.get("reject"):
            return {"valid": False, "reason": "safety_agent_rejected", "action": action, "agent": quality_result}
        if isinstance(quality_result, dict) and quality_result.get("warning"):
            warnings.append(str(quality_result.get("warning")))

        return {"valid": True, "reason": "ok", "action": action, "warnings": warnings}

    def _simulate_move(self, game_state: dict[str, Any], move: dict[str, Any], action: str) -> dict[str, Any] | None:
        try:
            state = json.loads(json.dumps(game_state))
        except Exception:
            return None

        target = move.get("target") if isinstance(move.get("target"), dict) else None
        row, col = self._target_coords(target)
        if row is None or col is None or not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return None

        board = state.get("board")
        active_player = self._coerce_int(state.get("activePlayer", 2), default=2)
        player = self._get_player(state, active_player)
        if not self._valid_board(board) or not isinstance(player, dict):
            return None
        board = cast(list[list[Any]], board)

        if action == "PLACE":
            if board[row][col] is not None:
                return None
            if not self._is_adjacent(player.get("position"), {"row": row, "col": col}):
                return None
            tile_type = self._best_place_type(state, active_player, row, col)
            board[row][col] = {
                "id": f"ai-{time.time_ns()}",
                "type": tile_type,
                "rotation": 0,
                "color": player.get("color", "neutral"),
                "hasResonator": False,
                "playersPresent": [],
            }

        elif action == "ROTATE":
            tile = board[row][col]
            if not isinstance(tile, dict):
                return None
            if tile.get("hasResonator") and self._coerce_int(tile.get("resonatorOwner"), 0) != active_player:
                return None
            tile["rotation"] = (self._coerce_int(tile.get("rotation"), 0) + 90) % 360

        elif action == "ADVANCE":
            current_pos = player.get("position")
            if not self._is_valid_move(board, current_pos, {"row": row, "col": col}):
                return None
            current_coords = self._extract_position_coords(current_pos)
            if current_coords is None:
                return None
            old_row, old_col = current_coords
            old_tile = board[old_row][old_col]
            if isinstance(old_tile, dict) and isinstance(old_tile.get("playersPresent"), list):
                old_tile["playersPresent"] = [pid for pid in old_tile["playersPresent"] if self._coerce_int(pid, -1) != active_player]
            player["position"] = {"row": row, "col": col}
            new_tile = board[row][col]
            if isinstance(new_tile, dict):
                present = new_tile.setdefault("playersPresent", [])
                if active_player not in present:
                    present.append(active_player)

        elif action == "ATTUNE":
            tile = board[row][col]
            if not isinstance(tile, dict):
                return None
            pos = player.get("position")
            if not isinstance(pos, dict) or pos.get("row") != row or pos.get("col") != col:
                return None
            if tile.get("hasResonator"):
                return None
            if self._coerce_int(player.get("resonators"), 0) <= 0:
                return None
            if self._is_power_well(row, col) and tile.get("color") == "neutral":
                return None
            tile["hasResonator"] = True
            tile["resonatorOwner"] = active_player
            player["resonators"] = self._coerce_int(player.get("resonators"), 0) - 1

        elif action == "SHIFT":
            if row not in (0, BOARD_SIZE - 1) and col not in (0, BOARD_SIZE - 1):
                return None
            self._apply_shift(state, row, col)

        else:
            return None

        self._finalize_state_after_action(state)
        state["validMoves"] = self._infer_valid_moves(state)
        return state

    def _finalize_state_after_action(self, state: dict[str, Any]) -> None:
        active_player = self._coerce_int(state.get("activePlayer", 2), 2)
        state["actionsRemaining"] = self._coerce_int(state.get("actionsRemaining", 2), 2) - 1
        state["capturedWells"] = self._refresh_captured_wells(state)

        if self._path_progress(state, active_player) >= 100.0:
            state["winner"] = active_player
            state["winReason"] = "Path Completed!"
            return

        if self._coerce_int(state.get("actionsRemaining"), 0) <= 0:
            if self._captured_count(state, active_player) >= 3:
                state["winner"] = active_player
                state["winReason"] = "3 Power Wells Captured!"
                return
            state["activePlayer"] = 1 if active_player == 2 else 2
            state["actionsRemaining"] = 2
            state["turn"] = self._coerce_int(state.get("turn", 1), 1) + 1
        state["selectedCardId"] = None
        state["selectedActionIndex"] = None

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _update_score_stats(self, payload: dict[str, Any]) -> None:
        ai_player_id = self._coerce_int(payload.get("aiPlayerId", payload.get("ai_player", 2)), default=2)
        winner = payload.get("winner")
        match_record = payload.get("matchRecord") if isinstance(payload.get("matchRecord"), dict) else {}

        ai_score = self._coerce_number(payload.get("aiScore", payload.get("ai_score", 0)))
        opponent_score = self._coerce_number(payload.get("opponentScore", payload.get("opponent_score", 0)))
        if ai_score == 0.0 and isinstance(match_record, dict):
            points = self._coerce_number(match_record.get("points"), 0.0)
            ai_score = -points if ai_player_id == 2 else points
            opponent_score = -ai_score

        games_played = self._coerce_int(self.stats.get("games_played", 0), 0) + 1
        self.stats["games_played"] = games_played
        self.stats["average_ai_score"] = self._running_average(self.stats.get("average_ai_score"), ai_score, games_played)
        self.stats["average_opponent_score"] = self._running_average(self.stats.get("average_opponent_score"), opponent_score, games_played)

        reward = self._coerce_number(payload.get("reward"), ai_score - opponent_score)
        self.stats["average_reward"] = self._running_average(self.stats.get("average_reward"), reward, games_played)

        if winner in {"draw", None, 0, "0", "none"}:
            result = "draw"
            self.stats["draws"] = self._coerce_int(self.stats.get("draws", 0), 0) + 1
        elif self._coerce_int(winner, default=-1) == ai_player_id:
            result = "win"
            self.stats["wins"] = self._coerce_int(self.stats.get("wins", 0), 0) + 1
        else:
            result = "loss"
            self.stats["losses"] = self._coerce_int(self.stats.get("losses", 0), 0) + 1
        self.stats["last_result"] = result
        self.stats["last_updated"] = datetime.utcnow().isoformat()

    def _update_action_preferences(self, payload: dict[str, Any]) -> None:
        result = str(self.stats.get("last_result", "draw"))
        reward = {"win": 0.42, "loss": -0.36, "draw": 0.08}.get(result, 0.0)
        ai_actions = payload.get("aiActions") or payload.get("ai_actions") or []
        if not isinstance(ai_actions, list):
            ai_actions = []
        for entry in ai_actions:
            action = self._extract_action_from_log_entry(entry)
            if action not in ACTION_BASE_WEIGHTS:
                continue
            old = self.action_weights.get(action, 0.0)
            self.action_weights[action] = max(-10.0, min(12.0, old + reward))
            phase = str(entry.get("phase", "midgame")) if isinstance(entry, dict) else "midgame"
            if phase not in PHASES:
                phase = "midgame"
            phase_bucket = self.phase_action_weights.setdefault(phase, {})
            phase_bucket[action] = max(-8.0, min(10.0, phase_bucket.get(action, 0.0) + reward * 0.65))

        # Learn from current match signatures even when frontend action log is sparse.
        for signature in self.current_match_signatures:
            old = self.move_signature_weights.get(signature, 0.0)
            self.move_signature_weights[signature] = max(-10.0, min(10.0, old + reward * 0.35))
        self.current_match_signatures = []

    def _update_signature_preferences(self, payload: dict[str, Any]) -> None:
        result = str(self.stats.get("last_result", "draw"))
        delta = {"win": 0.18, "loss": -0.16, "draw": 0.04}.get(result, 0.0)
        signatures = payload.get("aiMoveSignatures") or payload.get("ai_move_signatures") or []
        if not isinstance(signatures, list):
            return
        for signature in signatures:
            if not isinstance(signature, str) or not signature:
                continue
            self.move_signature_weights[signature] = max(-10.0, min(10.0, self.move_signature_weights.get(signature, 0.0) + delta))
        if len(self.move_signature_weights) > 1500:
            ranked = sorted(self.move_signature_weights.items(), key=lambda item: abs(item[1]), reverse=True)
            self.move_signature_weights = dict(ranked[:1500])

    def _update_opponent_profile(self, payload: dict[str, Any]) -> None:
        actions = payload.get("opponentActions") or payload.get("opponent_actions") or []
        if not isinstance(actions, list):
            actions = []
        action_names = [self._extract_action_from_log_entry(entry) for entry in actions]
        action_names = [action for action in action_names if action]
        if action_names:
            shift_rate = action_names.count("SHIFT") / len(action_names)
            attune_rate = action_names.count("ATTUNE") / len(action_names)
            self.opponent_profile["shift_frequency"] = self._ewma(self.opponent_profile.get("shift_frequency", 0.0), shift_rate)
            self.opponent_profile["attune_frequency"] = self._ewma(self.opponent_profile.get("attune_frequency", 0.0), attune_rate)
            self.opponent_profile["last_seen_actions"] = action_names[-20:]
        captured = payload.get("capturedWells")
        if isinstance(captured, dict):
            ai_id = self._coerce_int(payload.get("aiPlayerId", 2), 2)
            opp_id = 1 if ai_id == 2 else 2
            opp_wells = sum(1 for owner in captured.values() if self._coerce_int(owner, 0) == opp_id)
            self.opponent_profile["well_pressure"] = self._ewma(self.opponent_profile.get("well_pressure", 0.0), opp_wells / 3.0)

    def _load_learning_state(self) -> None:
        if not self.learning_store_path.exists():
            return
        try:
            payload = json.loads(self.learning_store_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            loaded_stats = payload.get("stats")
            if isinstance(loaded_stats, dict):
                for key in self.stats:
                    if key in loaded_stats:
                        self.stats[key] = loaded_stats[key]
            self.action_weights = self._sanitize_numeric_mapping(payload.get("action_weights"), -10.0, 12.0)
            phase_payload = payload.get("phase_action_weights")
            if isinstance(phase_payload, dict):
                for phase in PHASES:
                    self.phase_action_weights[phase] = self._sanitize_numeric_mapping(phase_payload.get(phase), -8.0, 10.0)
            self.move_signature_weights = self._sanitize_numeric_mapping(payload.get("move_signature_weights"), -10.0, 10.0)
            if isinstance(payload.get("opponent_profile"), dict):
                self.opponent_profile.update(payload["opponent_profile"])
            if isinstance(payload.get("mistakes"), list):
                self.mistakes = [entry for entry in payload["mistakes"] if isinstance(entry, dict)][-200:]
            self._learning_state_loaded = True
            self._learning_state_mtime = self.learning_store_path.stat().st_mtime
        except Exception as error:  # noqa: BLE001
            logger.warning("Failed to load Aether learning state: %s", error)
            self._learning_state_loaded = False

    def _save_learning_state(self) -> None:
        payload = {
            "updated_at": datetime.utcnow().isoformat(),
            "stats": self.stats,
            "action_weights": self.action_weights,
            "phase_action_weights": self.phase_action_weights,
            "move_signature_weights": self.move_signature_weights,
            "opponent_profile": self.opponent_profile,
            "mistakes": self.mistakes[-200:],
        }
        self.learning_store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.learning_store_path.with_suffix(f"{self.learning_store_path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.learning_store_path)
        self._learning_state_loaded = True
        self._learning_state_mtime = self.learning_store_path.stat().st_mtime

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
    # Board helpers
    # ------------------------------------------------------------------

    def _evaluate_state(self, state: dict[str, Any], player_id: int) -> float:
        if state.get("winner") == player_id:
            return 250000.0
        opponent = 1 if player_id == 2 else 2
        if state.get("winner") == opponent:
            return -250000.0
        player = self._get_player(state, player_id)
        if not isinstance(player, dict):
            return -5000.0
        my_wells = self._captured_count(state, player_id)
        opp_wells = self._captured_count(state, opponent)
        my_progress = self._path_progress(state, player_id)
        opp_progress = self._path_progress(state, opponent)
        score = my_wells * 8500.0 - opp_wells * 9500.0
        score += my_progress * 145.0 - opp_progress * 168.0
        resonators = self._coerce_int(player.get("resonators"), 0)
        score += resonators * 80.0
        pos = player.get("position")
        coords = self._extract_position_coords(pos)
        if coords is not None:
            row, col = coords
            goal_row = self._coerce_int(player.get("goalRow"), 0)
            score += max(0, BOARD_SIZE - abs(goal_row - row)) * 225.0
            if self._is_power_well(row, col):
                score += 1100.0
            score += self._local_connectivity_potential(state, row, col) * 45.0
        return score

    def _best_follow_up_score(self, state: dict[str, Any], player_id: int, max_moves: int = 50) -> float:
        moves = state.get("validMoves") if isinstance(state.get("validMoves"), list) else []
        if not moves:
            moves = self._infer_valid_moves(state)
        best = -math.inf
        for move in moves[:max_moves]:
            action = self._resolve_action(state, move)
            if not action:
                continue
            simulated = self._simulate_move(state, move, action)
            if simulated is None:
                continue
            value = self._evaluate_state(simulated, player_id) - self._evaluate_state(state, player_id)
            if simulated.get("winner") == player_id:
                value += 100000.0
            best = max(best, value)
        return best if best > -math.inf else -120.0

    def _path_progress(self, state: dict[str, Any], player_id: int) -> float:
        player = self._get_player(state, player_id)
        board = state.get("board")
        if not isinstance(player, dict) or not self._valid_board(board):
            return 0.0
        board = cast(list[list[Any]], board)
        start_row = self._coerce_int(player.get("homeRow"), 0)
        goal_row = self._coerce_int(player.get("goalRow"), 0)
        queue: list[tuple[int, int]] = []
        visited: set[tuple[int, int]] = set()
        max_progress = 0
        if not (0 <= start_row < BOARD_SIZE and 0 <= goal_row < BOARD_SIZE):
            return 0.0
        for col in range(BOARD_SIZE):
            if board[start_row][col] is not None:
                queue.append((start_row, col))
                visited.add((start_row, col))
        head = 0
        while head < len(queue):
            row, col = queue[head]
            head += 1
            max_progress = max(max_progress, abs(row - start_row))
            if row == goal_row:
                return 100.0
            for d_row, d_col in DIRECTIONS:
                nr, nc = row + d_row, col + d_col
                if (nr, nc) in visited:
                    continue
                if self._is_valid_move(board, {"row": row, "col": col}, {"row": nr, "col": nc}):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return min(99.0, (max_progress / (BOARD_SIZE - 1)) * 90.0)

    def _is_valid_move(self, board: list[list[Any]], from_pos: Any, to_pos: Any) -> bool:
        from_coords = self._extract_position_coords(from_pos)
        to_coords = self._extract_position_coords(to_pos)
        if from_coords is None or to_coords is None:
            return False
        fr, fc = from_coords
        tr, tc = to_coords
        if not (0 <= fr < BOARD_SIZE and 0 <= fc < BOARD_SIZE and 0 <= tr < BOARD_SIZE and 0 <= tc < BOARD_SIZE):
            return False
        from_tile = board[fr][fc]
        to_tile = board[tr][tc]
        if not isinstance(from_tile, dict) or not isinstance(to_tile, dict):
            return False
        from_conns = self._rotated_connections(str(from_tile.get("type")), self._coerce_int(from_tile.get("rotation"), 0))
        to_conns = self._rotated_connections(str(to_tile.get("type")), self._coerce_int(to_tile.get("rotation"), 0))
        d_row = tr - fr
        d_col = tc - fc
        if d_row == -1 and d_col == 0:
            return bool(from_conns[0] and to_conns[2])
        if d_row == 0 and d_col == 1:
            return bool(from_conns[1] and to_conns[3])
        if d_row == 1 and d_col == 0:
            return bool(from_conns[2] and to_conns[0])
        if d_row == 0 and d_col == -1:
            return bool(from_conns[3] and to_conns[1])
        return False

    def _rotated_connections(self, tile_type: str, rotation: int) -> list[bool]:
        base = list(TILE_CONNECTIONS.get(tile_type, [False, False, False, False]))
        shifts = (rotation // 90) % 4
        for _ in range(shifts):
            base.insert(0, base.pop())
        return base

    def _best_place_type(self, state: dict[str, Any], player_id: int, row: int, col: int) -> str:
        best_type = PLACE_TYPES[0]
        best_score = -math.inf
        player = self._get_player(state, player_id) or {}
        color = player.get("color", "neutral") if isinstance(player, dict) else "neutral"
        for tile_type in PLACE_TYPES:
            temp = json.loads(json.dumps(state))
            temp["board"][row][col] = {
                "id": f"probe-{tile_type}",
                "type": tile_type,
                "rotation": 0,
                "color": color,
                "hasResonator": False,
                "playersPresent": [],
            }
            score = self._path_progress(temp, player_id) + self._local_connectivity_potential(temp, row, col) * 8.0
            if score > best_score:
                best_score = score
                best_type = tile_type
        return best_type

    def _apply_shift(self, state: dict[str, Any], row: int, col: int) -> None:
        board = state["board"]
        players = state.get("players", {})
        axis = ""
        direction = 0
        if row == 0:
            axis = "col"
            direction = 1
        elif row == BOARD_SIZE - 1:
            axis = "col"
            direction = -1
        elif col == 0:
            axis = "row"
            direction = 1
        elif col == BOARD_SIZE - 1:
            axis = "row"
            direction = -1

        if axis == "row":
            row_tiles = board[row]
            if direction == 1:
                row_tiles.insert(0, row_tiles.pop())
            else:
                row_tiles.append(row_tiles.pop(0))
            for player in players.values() if isinstance(players, dict) else []:
                pos = player.get("position") if isinstance(player, dict) else None
                if isinstance(pos, dict) and pos.get("row") == row:
                    pos["col"] = (self._coerce_int(pos.get("col"), 0) + direction) % BOARD_SIZE

        if axis == "col":
            column = [board[r][col] for r in range(BOARD_SIZE)]
            if direction == 1:
                column.insert(0, column.pop())
            else:
                column.append(column.pop(0))
            for r in range(BOARD_SIZE):
                board[r][col] = column[r]
            for player in players.values() if isinstance(players, dict) else []:
                pos = player.get("position") if isinstance(player, dict) else None
                if isinstance(pos, dict) and pos.get("col") == col:
                    pos["row"] = (self._coerce_int(pos.get("row"), 0) + direction) % BOARD_SIZE

    def _refresh_captured_wells(self, state: dict[str, Any]) -> dict[str, int]:
        captured: dict[str, int] = {}
        board = state.get("board")
        if not self._valid_board(board):
            return captured
        for well in POWER_WELLS:
            tile = board[well["row"]][well["col"]]
            if isinstance(tile, dict) and tile.get("hasResonator") and tile.get("resonatorOwner"):
                captured[f"{well['row']},{well['col']}"] = self._coerce_int(tile.get("resonatorOwner"), 0)
        return captured

    # ------------------------------------------------------------------
    # Small utility methods
    # ------------------------------------------------------------------

    def _resolve_action(self, game_state: dict[str, Any], move: dict[str, Any]) -> str:
        cards = game_state.get("faceUpCards", [])
        if not isinstance(cards, list):
            return ""
        card_id = move.get("cardId")
        action_index = self._coerce_int(move.get("actionIndex"), default=-1)
        card = next((item for item in cards if isinstance(item, dict) and item.get("id") == card_id), None)
        actions = card.get("actions", []) if isinstance(card, dict) else []
        if isinstance(actions, list) and 0 <= action_index < len(actions):
            return str(actions[action_index]).upper()
        return ""

    def _extract_action_from_log_entry(self, entry: Any) -> str:
        if not isinstance(entry, dict):
            return ""
        action = entry.get("action") or entry.get("type") or entry.get("selectedAction")
        if action == "SELECTED_ACTION":
            action = entry.get("action")
        return str(action or "").upper()

    def _valid_board(self, board: Any) -> bool:
        return isinstance(board, list) and len(board) == BOARD_SIZE and all(isinstance(row, list) and len(row) == BOARD_SIZE for row in board)

    def _sanitize_captured_wells(self, payload: Any) -> dict[str, int]:
        if not isinstance(payload, dict):
            return {}
        result: dict[str, int] = {}
        for key, owner in payload.items():
            if str(key) in POWER_WELL_KEYS:
                owner_int = self._coerce_int(owner, 0)
                if owner_int in (1, 2):
                    result[str(key)] = owner_int
        return result

    def _dedupe_moves(self, moves: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for move in moves:
            if not isinstance(move, dict):
                continue
            signature = self._raw_move_signature(move)
            if signature:
                seen[signature] = move
        return list(seen.values())

    def _raw_move_signature(self, move: dict[str, Any]) -> str:
        target = move.get("target") if isinstance(move.get("target"), dict) else {}
        row, col = self._target_coords(target)
        if row is None or col is None:
            return ""
        return f"{move.get('cardId')}:{move.get('actionIndex')}:{row}:{col}"

    def _move_signature(self, move: dict[str, Any], state_view: AetherStateView) -> str:
        action = self._resolve_action(state_view.raw, move) or "UNKNOWN"
        target = move.get("target") if isinstance(move.get("target"), dict) else {}
        row, col = self._target_coords(target)
        if row is None or col is None:
            return action
        return f"{state_view.phase}:{action}:{self._target_zone(row, col)}:{row}:{col}"

    def _move_in_legal_set(self, move: dict[str, Any], legal_moves: list[dict[str, Any]]) -> bool:
        sig = self._raw_move_signature(move)
        return bool(sig) and any(self._raw_move_signature(other) == sig for other in legal_moves)

    def _target_coords(self, target: Any) -> tuple[int | None, int | None]:
        if not isinstance(target, dict):
            return None, None
        row = target.get("row")
        col = target.get("col")
        return (row if isinstance(row, int) else None, col if isinstance(col, int) else None)

    def _classify_phase(self, turn: int, captured_wells: dict[str, int], game_state: dict[str, Any], active: int, opponent: int) -> str:
        active_path = self._path_progress(game_state, active)
        opponent_path = self._path_progress(game_state, opponent)
        max_wells = max(
            sum(1 for owner in captured_wells.values() if owner == active),
            sum(1 for owner in captured_wells.values() if owner == opponent),
        )
        if turn <= 3 and active_path < 50 and opponent_path < 50 and max_wells == 0:
            return "opening"
        if active_path >= 75 or opponent_path >= 75 or max_wells >= 2:
            return "conversion"
        return "midgame"

    def _strategy_label(self, state_view: AetherStateView, analysis: StrategicAnalysis, adaptive_profile: AdaptiveProfile) -> str:
        if analysis.urgent_defense:
            return "disrupt opponent conversion while preserving legal counterplay"
        if analysis.conversion_ready:
            return "convert current advantage through path completion or Power Well lock"
        if state_view.phase == "opening":
            return "build connected lane and preserve resonator economy"
        return f"{adaptive_profile.stance}: improve path tempo and contest Power Wells"

    def _controlled_fallback_response(
        self,
        *,
        reason: str,
        trace: dict[str, Any],
        started: float,
        top_candidates: list[CandidateScore] | None = None,
    ) -> dict[str, Any]:
        self.last_fallback_reason = reason
        response = {
            "move": None,
            "confidence": 0.0,
            "strategy": "controlled fallback",
            "reasoning": f"No safe legal Aether move returned: {reason}",
            "agent_trace": trace,
            "fallback": True,
            "fallback_reason": reason,
            "debug": {
                "elapsed_ms": round((time.time() - started) * 1000.0, 3),
                "top_candidates": [self._candidate_to_debug(candidate) for candidate in (top_candidates or [])[:5]],
            },
        }
        self.last_decision_response = response
        self.shared_memory.set("aether_last_decision", response)
        self._append_jsonl(self.decision_log_path, self._redact_for_log(response))
        return response

    def _last_decision_summary(self) -> dict[str, Any] | None:
        if not isinstance(self.last_decision_response, dict):
            return None
        return {
            "move": self.last_decision_response.get("move"),
            "confidence": self.last_decision_response.get("confidence"),
            "strategy": self.last_decision_response.get("strategy"),
            "fallback": self.last_decision_response.get("fallback"),
            "fallback_reason": self.last_decision_response.get("fallback_reason"),
        }

    def _candidate_to_debug(self, candidate: CandidateScore) -> dict[str, Any]:
        return {
            "move": candidate.move,
            "action": candidate.action,
            "score": round(candidate.score, 3),
            "confidence": round(candidate.confidence, 3),
            "reason": candidate.reason[:500],
            "valid": candidate.valid,
            "safety": candidate.safety,
        }

    def _confidence_from_score(self, selected: CandidateScore, candidates: list[CandidateScore], adaptive: AdaptiveProfile) -> float:
        if not candidates:
            return 0.0
        if len(candidates) == 1:
            return max(0.1, min(0.96, selected.confidence))
        second = candidates[1].score
        margin = selected.score - second
        margin_bonus = max(0.0, min(0.18, margin / max(1000.0, abs(selected.score))))
        return round(max(0.05, min(0.99, selected.confidence + margin_bonus + adaptive.confidence_bias * 0.25)), 3)

    def _agent_quality_evaluation(self, candidates: list[CandidateScore], state_view: AetherStateView, analysis: StrategicAnalysis) -> dict[str, Any]:
        payload = {
            "type": "aether_candidate_quality",
            "phase": state_view.phase,
            "candidate_count": len(candidates),
            "best": self._candidate_to_debug(candidates[0]) if candidates else None,
            "analysis": analysis.__dict__,
        }
        for agent_type in ("evaluation", "quality"):
            response = self._call_agent(agent_type, "perform_task", payload)
            if isinstance(response, dict):
                return {"status": f"{agent_type}_used", "response_preview": str(response)[:240]}
        return {"status": "local_quality_check", "candidate_count": len(candidates)}

    def _emit_observability_event(self, event_name: str, response: dict[str, Any]) -> None:
        payload = {"event": event_name, "game": self.game, "timestamp": time.time(), "summary": self._last_decision_summary()}
        self._call_agent("observability", "perform_task", payload)
        self.shared_memory.set("aether_last_observability_event", payload)
        if isinstance(response.get("agent_trace"), dict):
            response["agent_trace"]["observability"] = "event_recorded"

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except Exception as error:  # noqa: BLE001
            logger.warning("Failed to append %s: %s", path, error)

    def _redact_for_log(self, payload: dict[str, Any]) -> dict[str, Any]:
        safe = json.loads(json.dumps(payload, default=str))
        debug = safe.get("debug") if isinstance(safe, dict) else None
        if isinstance(debug, dict) and isinstance(debug.get("top_candidates"), list):
            debug["top_candidates"] = debug["top_candidates"][:5]
        return safe

    def _local_connectivity_potential(self, state: dict[str, Any], row: int | None, col: int | None) -> int:
        if row is None or col is None:
            return 0
        board = state.get("board")
        if not self._valid_board(board) or not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
            return 0
        board = cast(list[list[Any]], board)
        tile = board[row][col]
        if not isinstance(tile, dict):
            return 0
        count = 0
        for d_row, d_col in DIRECTIONS:
            nr, nc = row + d_row, col + d_col
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] is not None:
                if self._is_valid_move(board, {"row": row, "col": col}, {"row": nr, "col": nc}) or self._is_valid_move(board, {"row": nr, "col": nc}, {"row": row, "col": col}):
                    count += 1
        return count

    def _target_zone(self, row: int, col: int) -> str:
        if self._is_power_well(row, col):
            return "power_well"
        if row in (0, BOARD_SIZE - 1) or col in (0, BOARD_SIZE - 1):
            return "edge"
        if row == BOARD_SIZE // 2 and col == BOARD_SIZE // 2:
            return "center"
        return "inner"

    def _target_zone_value(self, row: int, col: int, active_player: int) -> float:
        zone = self._target_zone(row, col)
        if zone == "power_well":
            return 180.0
        if zone == "center":
            return 90.0
        if zone == "inner":
            return 55.0
        player = active_player
        goal_row = 0 if player == 2 else BOARD_SIZE - 1
        return 35.0 + max(0.0, (BOARD_SIZE - abs(goal_row - row)) * 12.0)

    def _player_displaced(self, after: dict[str, Any], before: dict[str, Any], player_id: int) -> bool:
        before_player = self._get_player(before, player_id) or {}
        after_player = self._get_player(after, player_id) or {}
        return before_player.get("position") != after_player.get("position")

    def _captured_count(self, state: dict[str, Any], player_id: int) -> int:
        captured = state.get("capturedWells", {})
        if not isinstance(captured, dict):
            return 0
        return sum(1 for owner in captured.values() if self._coerce_int(owner, 0) == player_id)

    def _get_player(self, state: dict[str, Any], player_id: int) -> dict[str, Any] | None:
        players = state.get("players", {})
        if not isinstance(players, dict):
            return None
        player = players.get(player_id)
        if player is None:
            player = players.get(str(player_id))
        return player if isinstance(player, dict) else None

    def _is_power_well(self, row: int, col: int) -> bool:
        return f"{row},{col}" in POWER_WELL_KEYS

    def _is_on_power_well(self, state: dict[str, Any], player_id: int) -> bool:
        player = self._get_player(state, player_id)
        if not isinstance(player, dict):
            return False
        coords = self._extract_position_coords(player.get("position"))
        return False if coords is None else self._is_power_well(*coords)

    def _is_adjacent(self, from_pos: Any, to_pos: dict[str, int]) -> bool:
        coords = self._extract_position_coords(from_pos)
        if coords is None:
            return False
        row, col = coords
        return abs(row - to_pos["row"]) + abs(col - to_pos["col"]) == 1

    @staticmethod
    def _extract_position_coords(position: Any) -> tuple[int, int] | None:
        if not isinstance(position, dict):
            return None
        row = position.get("row")
        col = position.get("col")
        if isinstance(row, int) and isinstance(col, int):
            return row, col
        return None

    def _sanitize_numeric_mapping(self, payload: Any, lower: float, upper: float) -> dict[str, float]:
        if not isinstance(payload, dict):
            return {}
        result: dict[str, float] = {}
        for key, value in payload.items():
            if not isinstance(key, str):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            result[key] = max(lower, min(upper, numeric))
        return result

    @staticmethod
    def _running_average(previous: Any, new_value: float, n: int) -> float:
        prev = 0.0
        try:
            prev = float(previous)
        except (TypeError, ValueError):
            prev = 0.0
        return round(prev + ((new_value - prev) / max(1, n)), 4)

    @staticmethod
    def _ewma(previous: Any, sample: float, alpha: float = 0.25) -> float:
        try:
            prev = float(previous)
        except (TypeError, ValueError):
            prev = 0.0
        return round((alpha * sample) + ((1.0 - alpha) * prev), 4)

    @staticmethod
    def _coerce_number(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


_ai_player: AetherShiftAI | None = None


def initialize_ai() -> AetherShiftAI:
    """Initialize once and reuse across shared backend calls."""

    global _ai_player
    if _ai_player is None:
        _ai_player = AetherShiftAI()
        logger.info("Aether Shift AI initialized at %s", _ai_player.initialized_at)
    return _ai_player
