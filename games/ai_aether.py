"""Aether Shift AI initialization + move selection runtime."""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add repository root to sys.path so shared src/ and logs/ imports resolve
games_root = Path(__file__).resolve().parent
project_root = games_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ..src.agents.agent_factory import AgentFactory
from ..src.agents.collaborative.shared_memory import SharedMemory
from ..src.agents.planning.planning_types import Task, TaskType
from ..logs.logger import get_logger, PrettyPrinter

logger = get_logger("Aether Shift")
printer = PrettyPrinter()

ACTION_BY_CARD_ID = {
    "card-place": "PLACE",
    "card-shift": "SHIFT",
    "card-rotate": "ROTATE",
    "card-advance": "ADVANCE",
    "card-attune": "ATTUNE",
}

@dataclass
class AetherShiftAI:
    game: str = "aether_shift"
    initialized_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self) -> None:
        self.shared_memory = SharedMemory()
        self.factory = AgentFactory()

        self.knowledge_agent = self.factory.create("knowledge", self.shared_memory)
        self.planning_agent = self.factory.create("planning", self.shared_memory)
        self.execution_agent = self.factory.create("execution", self.shared_memory)
        self.learning_agent = self.factory.create("learning", self.shared_memory)

        self._planning_task_registered = False
        self._planning_enabled = True
        self.match_log_path = project_root / 'logs' / 'aether_matches.jsonl'
        self.shared_memory.set("aether_ai_status", "initialized")
        logger.info("Aether Shift AI initialized with Knowledge, Planning, Execution, and Learning agents")

    def health(self) -> dict[str, Any]:
        return {
            "agent_status": "ready",
            "initialized_at": self.initialized_at,
            "agents": ["knowledge", "planning", "execution", "learning"],
        }

    def get_move(self, game_state: dict[str, Any]) -> dict[str, Any] | None:
        valid_moves = game_state.get("validMoves", []) if isinstance(game_state, dict) else []
        if not valid_moves:
            return None

        self.shared_memory.set("aether_last_state", game_state)
        strategy_context = self._get_strategy_context()
        plan = self._generate_plan(game_state, strategy_context)

        best_move, best_score = self._select_move_via_execution(
            valid_moves=valid_moves,
            game_state=game_state,
            strategy_context=strategy_context,
            plan=plan,
        )

        if best_move is None:
            best_move = random.choice(valid_moves)

        self.shared_memory.set(
            "aether_last_decision",
            {
                "move": best_move,
                "score": best_score,
                "plan_steps": len(plan) if isinstance(plan, list) else 0,
                "used_execution_agent": bool(best_move),
                "timestamp": time.time(),
            },
        )
        return best_move

    def learn_from_game(self, payload: dict[str, Any]) -> bool:
        try:
            if hasattr(self.learning_agent, "learn"):
                self.learning_agent.learn(payload)

            enriched_payload = {
                **payload,
                "logged_at": datetime.utcnow().isoformat(),
            }
            self.shared_memory.set("aether_last_game", enriched_payload)

            self.match_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.match_log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{json.dumps(enriched_payload, ensure_ascii=False)}\n")

            if isinstance(payload.get("aiActions"), list):
                self.shared_memory.set("aether_last_ai_actions", payload["aiActions"])

            return True
        except Exception as error:  # noqa: BLE001
            logger.warning("Aether Shift learning update failed: %s", error)
            return False

    def _get_strategy_context(self) -> str:
        try:
            if hasattr(self.knowledge_agent, "query"):
                result = self.knowledge_agent.query("Aether Shift strategy for board control and wells")
                self.shared_memory.set("aether_strategy_context", str(result))
                return str(result)
            return ""
        except Exception as error:  # noqa: BLE001
            logger.warning("Knowledge agent query failed: %s", error)
            return ""

    def _generate_plan(self, game_state: dict[str, Any], strategy_context: str) -> list[Task] | None:
        if not self._planning_enabled:
            return None

        try:
            now = time.time()
            fallback_task = Task(
                name="select_best_aether_move_fallback",
                task_type=TaskType.PRIMITIVE,
                start_time=now + 10,
                deadline=now + 300,
                duration=120,
                context={"game": self.game},
            )
            goal_task = Task(
                name="select_best_aether_move",
                task_type=TaskType.ABSTRACT,
                start_time=now + 5,
                deadline=now + 600,
                methods=[[fallback_task]],
                goal_state={"move_selected": True},
                context={"game_state": game_state, "strategy": strategy_context},
            )

            if not self._planning_task_registered and hasattr(self.planning_agent, "register_task"):
                self.planning_agent.register_task(goal_task)
                self._planning_task_registered = True

            if hasattr(self.planning_agent, "generate_plan"):
                plan = self.planning_agent.generate_plan(goal_task)
                if isinstance(plan, list):
                    self.shared_memory.set("aether_last_plan", [task.name for task in plan])
                    return plan
        except Exception as error:  # noqa: BLE001
            logger.warning("Planning agent failed: %s", error)
            self._planning_enabled = False
        return None

    def _select_move_via_execution(
        self,
        valid_moves: list[dict[str, Any]],
        game_state: dict[str, Any],
        strategy_context: str,
        plan: list[Task] | None,
    ) -> tuple[dict[str, Any] | None, float]:
        execution_signal = {}
        try:
            if hasattr(self.execution_agent, "predict"):
                execution_signal = self.execution_agent.predict({
                    "game": self.game,
                    "strategy": strategy_context,
                    "plan_size": len(plan) if plan else 0,
                })
        except Exception as error:  # noqa: BLE001
            logger.warning("Execution agent prediction failed: %s", error)

        bonus = 0.0
        if isinstance(execution_signal, dict) and execution_signal.get("selected_action") != "idle":
            bonus = float(execution_signal.get("confidence", 0.0)) * 3.0

        scored_moves: list[tuple[dict[str, Any], float]] = []
        for move in valid_moves:
            score = self._score_move(move, game_state, strategy_context, plan) + bonus
            scored_moves.append((move, score))

        if not scored_moves:
            return None, float("-inf")

        best_score = max(score for _, score in scored_moves)
        top_band = [move for move, score in scored_moves if score >= best_score - 2.0]
        best_move = random.choice(top_band) if top_band else None

        return best_move, best_score

    def _score_move(
        self,
        move: dict[str, Any],
        game_state: dict[str, Any],
        strategy_context: str,
        plan: list[Task] | None,
    ) -> float:
        score = 0.0

        active_player = game_state.get("activePlayer", 2)
        players = game_state.get("players", {})
        current_player = players.get(str(active_player), players.get(active_player, {}))
        goal_row = current_player.get("goalRow", 0)

        target = move.get("target", {}) if isinstance(move, dict) else {}
        row = target.get("row")
        col = target.get("col")

        card_id = move.get("cardId", "")
        action_type = ACTION_BY_CARD_ID.get(card_id, "")

        if action_type == "ATTUNE":
            score += 40
        elif action_type == "ADVANCE":
            score += 34
        elif action_type == "PLACE":
            score += 16
        elif action_type == "ROTATE":
            score += 14
        elif action_type == "SHIFT":
            score += 10

        if action_type in {"ADVANCE", "PLACE"} and isinstance(row, int):
            score += max(0, 4 - abs(goal_row - row)) * 6

        power_wells = game_state.get("powerWells", [])
        is_power_well_target = isinstance(row, int) and isinstance(col, int) and any(
            w.get("row") == row and w.get("col") == col for w in power_wells if isinstance(w, dict)
        )
        if is_power_well_target:
            score += 40

        captured_wells = game_state.get("capturedWells", {})
        well_key = f"{row},{col}" if isinstance(row, int) and isinstance(col, int) else None

        if action_type == "ATTUNE" and is_power_well_target and well_key and well_key not in captured_wells:
            my_wells = sum(1 for owner in captured_wells.values() if owner == active_player)
            score += 60
            if my_wells >= 2:
                score += 500

        opponent_id = 1 if active_player == 2 else 2
        opponent = players.get(str(opponent_id), players.get(opponent_id, {}))
        opponent_pos = opponent.get("position", {}) if isinstance(opponent, dict) else {}
        opponent_goal = opponent.get("goalRow") if isinstance(opponent, dict) else None
        if isinstance(opponent_goal, int) and isinstance(opponent_pos.get("row"), int):
            opp_distance_to_goal = abs(opponent_goal - opponent_pos["row"])
            if opp_distance_to_goal <= 1 and action_type in {"SHIFT", "ROTATE", "PLACE"}:
                score += 24

        if action_type == "SHIFT":
            # Keep SHIFT useful but avoid deterministic edge-locking bias.
            score += random.uniform(-1.5, 1.5)

        if strategy_context:
            score += 2
        if plan:
            score += min(len(plan), 4)

        return score

def initialize_ai() -> AetherShiftAI:
    ai = AetherShiftAI()
    logger.info("Aether Shift AI initialized at %s", ai.initialized_at)
    return ai