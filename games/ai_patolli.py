"""Patolli AI initialization + move selection runtime.

Patolli is an ancient Mesoamerican race-and-wager board game associated with the
Aztecs. The strategy model in this module prioritizes forward progress, safety,
and race completion while integrating the shared R-Games multi-agent stack.
"""

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
from ..src.agents.collaborative_agent import CollaborativeAgent
from ..src.agents.planning.planning_types import Task, TaskType
from ..logs.logger import get_logger, PrettyPrinter

logger = get_logger("Patolli")
printer = PrettyPrinter()

@dataclass
class PulucAI:
    game: str = "puluc"
    initialized_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self) -> None:
        self.shared_memory = SharedMemory()
        self.factory = AgentFactory()
        self.collab = CollaborativeAgent(shared_memory=self.shared_memory, agent_factory=self.factory)

        self.knowledge_agent = self.factory.create("knowledge", self.shared_memory)
        self.planning_agent = self.factory.create("planning", self.shared_memory)
        self.execution_agent = self.factory.create("execution", self.shared_memory)
        self.learning_agent = self.factory.create("learning", self.shared_memory)

        self._planning_task_registered = False
        self._planning_enabled = True

        self.match_log_path = project_root / "logs" / "puluc_matches.jsonl"
        self.learning_store_path = project_root / "logs" / "puluc_learning_state.json"

        self.stats: dict[str, Any] = {
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "average_ai_score": 0.0,
            "average_opponent_score": 0.0,
            "last_result": None,
            "last_updated": None,
        }
        self.capture_weights: dict[str, float] = {}
        self.position_weights: dict[str, float] = {}
        self.capture_weights: dict[str, float] = {}
        self._load_learning_state()

        self.shared_memory.set("puluc_ai_status", "initialized")
        self.shared_memory.set("puluc_stats", self.stats)
        logger.info("Puluc AI initialized with Knowledge, Planning, Execution, and Learning agents")

    def health(self) -> dict[str, Any]:
        return {
            "agent_status": "ready",
            "initialized_at": self.initialized_at,
            "agents": ["knowledge", "planning", "execution", "learning", "collaborative"],
            "game": self.game,
            "stats": self.stats,
        }

    def get_move(self, game_state: dict[str, Any]) -> dict[str, Any] | None:
        valid_moves = game_state.get("validMoves", []) if isinstance(game_state, dict) else []
        if not valid_moves:
            return None

        self.shared_memory.set("puluc_last_state", game_state)
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
            "puluc_last_decision",
            {
                "move": best_move,
                "score": best_score,
                "plan_steps": len(plan) if isinstance(plan, list) else 0,
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
            self.shared_memory.set("puluc_last_game", enriched_payload)

            self.match_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.match_log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{json.dumps(enriched_payload, ensure_ascii=False)}\n")

            self._update_score_stats(payload)
            self._update_capture_preferences(payload)
            self._update_position_preferences(payload)
            self._save_learning_state()
            return True
        except Exception as error:  # noqa: BLE001
            logger.warning("Puluc learning update failed: %s", error)
            return False

    def _get_strategy_context(self) -> str:
        baseline = (
            "Puluc strategy: build safe forward progress, maximize capture chains, "
            "and avoid exposing long token stacks to counter-capture."
        )
        try:
            if hasattr(self.knowledge_agent, "query"):
                result = self.knowledge_agent.query("Puluc strategy for race tempo and capture risk management")
                context = str(result) if result else baseline
                self.shared_memory.set("puluc_strategy_context", context)
                return context
            return baseline
        except Exception as error:  # noqa: BLE001
            logger.warning("Knowledge agent query failed: %s", error)
            return baseline

    def _generate_plan(self, game_state: dict[str, Any], strategy_context: str) -> list[Task] | None:
        if not self._planning_enabled:
            return None

        try:
            now = time.time()
            fallback_task = Task(
                name="select_best_puluc_move_fallback",
                task_type=TaskType.PRIMITIVE,
                start_time=now + 3,
                deadline=now + 120,
                duration=20,
                context={"game": self.game},
            )
            goal_task = Task(
                name="select_best_puluc_move",
                task_type=TaskType.ABSTRACT,
                start_time=now + 1,
                deadline=now + 240,
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
                    self.shared_memory.set("puluc_last_plan", [getattr(step, "name", str(step)) for step in plan])
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
        execution_bonus = 0.0
        try:
            if hasattr(self.execution_agent, "predict"):
                execution_signal = self.execution_agent.predict(
                    {
                        "game": self.game,
                        "strategy": strategy_context,
                        "plan_size": len(plan) if plan else 0,
                    }
                )
                if isinstance(execution_signal, dict) and execution_signal.get("selected_action") != "idle":
                    execution_bonus = float(execution_signal.get("confidence", 0.0)) * 2.5
        except Exception as error:  # noqa: BLE001
            logger.warning("Execution agent prediction failed: %s", error)

        scored: list[tuple[dict[str, Any], float]] = []
        for move in valid_moves:
            score = self._score_move(move, game_state, strategy_context, plan) + execution_bonus
            scored.append((move, score))

        if not scored:
            return None, float("-inf")

        best_score = max(score for _, score in scored)
        near_best = [move for move, score in scored if score >= best_score - 1.5]
        return random.choice(near_best), best_score

    def _score_move(
        self,
        move: dict[str, Any],
        game_state: dict[str, Any],
        strategy_context: str,
        plan: list[Task] | None,
    ) -> float:
        score = 0.0

        if not isinstance(move, dict):
            return -1000.0

        move_type = str(move.get("type", "")).lower()
        if move_type in {"move", "advance", "step"}:
            score += 12
        elif move_type in {"capture", "take", "stack_capture"}:
            score += 24
        elif move_type in {"enter_home", "home", "finish"}:
            score += 30

        destination = move.get("to") or move.get("target") or move.get("position")
        if isinstance(destination, (int, float)):
            destination_index = int(destination)
            score += destination_index * 0.7
            score += self.position_weights.get(str(destination_index), 0.0)
        elif isinstance(destination, dict) and isinstance(destination.get("index"), int):
            destination_index = destination["index"]
            score += destination_index * 0.7
            score += self.position_weights.get(str(destination_index), 0.0)

        captured = move.get("captured") or move.get("captures") or move.get("capturedCount")
        if isinstance(captured, bool) and captured:
            score += 16
        elif isinstance(captured, (int, float)):
            score += min(float(captured), 6.0) * 8.0
            score += self.capture_weights.get(str(int(captured)), 0.0)

        stack_size = move.get("stackSize") or move.get("tokenStack")
        if isinstance(stack_size, (int, float)):
            score += min(float(stack_size), 5.0) * 2.0

        risk = move.get("risk")
        if isinstance(risk, (int, float)):
            score -= float(risk) * 8.0

        if strategy_context:
            score += 2.0
        if plan:
            score += min(len(plan), 5)

        score += random.uniform(-1.0, 1.0)
        return score

    def _update_score_stats(self, payload: dict[str, Any]) -> None:
        self.stats["games_played"] += 1
        ai_score, opponent_score = self._extract_scores(payload)

        n = max(1, self.stats["games_played"])
        self.stats["average_ai_score"] = ((self.stats["average_ai_score"] * (n - 1)) + ai_score) / n
        self.stats["average_opponent_score"] = ((self.stats["average_opponent_score"] * (n - 1)) + opponent_score) / n

        result = payload.get("result")
        if isinstance(result, str):
            normalized = result.strip().lower()
            if normalized in {"win", "won", "victory", "ai_win"}:
                self.stats["wins"] += 1
                self.stats["last_result"] = "win"
            elif normalized in {"loss", "lose", "lost", "defeat", "ai_loss"}:
                self.stats["losses"] += 1
                self.stats["last_result"] = "loss"
            else:
                self.stats["draws"] += 1
                self.stats["last_result"] = "draw"
        else:
            if ai_score > opponent_score:
                self.stats["wins"] += 1
                self.stats["last_result"] = "win"
            elif ai_score < opponent_score:
                self.stats["losses"] += 1
                self.stats["last_result"] = "loss"
            else:
                self.stats["draws"] += 1
                self.stats["last_result"] = "draw"

        self.stats["last_updated"] = datetime.utcnow().isoformat()
        self.shared_memory.set("puluc_stats", self.stats)

    def _extract_scores(self, payload: dict[str, Any]) -> tuple[float, float]:
        ai_score = payload.get("aiScore")
        opponent_score = payload.get("opponentScore")
        if isinstance(ai_score, (int, float)) and isinstance(opponent_score, (int, float)):
            return float(ai_score), float(opponent_score)

        players = payload.get("players")
        if isinstance(players, list):
            ai_value = 0.0
            opp_value = 0.0
            for player in players:
                if not isinstance(player, dict):
                    continue
                score = float(player.get("score", 0.0) or 0.0)
                marker = str(player.get("role") or player.get("type") or "").lower()
                pid = player.get("id")
                if marker in {"ai", "bot", "computer"} or pid in {1, "1", "ai"}:
                    ai_value = score
                else:
                    opp_value = max(opp_value, score)
            return ai_value, opp_value

        return 0.0, 0.0

    def _update_capture_preferences(self, payload: dict[str, Any]) -> None:
        actions = payload.get("aiActions")
        if not isinstance(actions, list):
            return

        reward = 1.0 if self.stats.get("last_result") == "win" else -0.3
        for action in actions:
            if not isinstance(action, dict):
                continue
            captured = action.get("captured") or action.get("captures") or action.get("capturedCount")
            if isinstance(captured, bool):
                captured = 1 if captured else 0
            if not isinstance(captured, (int, float)):
                continue
            key = str(int(captured))
            current = self.capture_weights.get(key, 0.0)
            self.capture_weights[key] = round((current * 0.85) + reward, 4)

        self.shared_memory.set("puluc_capture_weights", self.capture_weights)

    def _update_position_preferences(self, payload: dict[str, Any]) -> None:
        actions = payload.get("aiActions")
        if not isinstance(actions, list):
            return

        reward = 0.8 if self.stats.get("last_result") == "win" else -0.2
        for action in actions:
            if not isinstance(action, dict):
                continue

            destination = action.get("to") or action.get("target") or action.get("position")
            destination_index: int | None = None

            if isinstance(destination, (int, float)):
                destination_index = int(destination)
            elif isinstance(destination, dict) and isinstance(destination.get("index"), int):
                destination_index = destination["index"]

            if destination_index is None:
                continue

            key = str(destination_index)
            current = self.position_weights.get(key, 0.0)
            self.position_weights[key] = round((current * 0.88) + reward, 4)

        self.shared_memory.set("puluc_position_weights", self.position_weights)

    def _load_learning_state(self) -> None:
        if not self.learning_store_path.exists():
            return
        try:
            with self.learning_store_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data.get("stats"), dict):
                self.stats.update(data["stats"])
            if isinstance(data.get("capture_weights"), dict):
                self.capture_weights = {
                    str(key): float(value)
                    for key, value in data["capture_weights"].items()
                    if isinstance(value, (int, float))
                }
            if isinstance(data.get("position_weights"), dict):
                self.position_weights = {
                    str(key): float(value)
                    for key, value in data["position_weights"].items()
                    if isinstance(value, (int, float))
                }
        except Exception as error:  # noqa: BLE001
            logger.warning("Unable to load Puluc learning state: %s", error)

    def _save_learning_state(self) -> None:
        try:
            self.learning_store_path.parent.mkdir(parents=True, exist_ok=True)
            with self.learning_store_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "stats": self.stats,
                        "capture_weights": self.capture_weights,
                        "position_weights": self.position_weights,
                        "updated_at": datetime.utcnow().isoformat(),
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as error:  # noqa: BLE001
            logger.warning("Unable to save Puluc learning state: %s", error)


def initialize_ai() -> PulucAI:
    ai = PulucAI()
    logger.info("Puluc AI initialized at %s", ai.initialized_at)
    return ai