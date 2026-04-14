"""Chronos AI runtime with full SLAI multi-agent integration.

This module provides production-grade Chronos move selection by combining:
- Knowledge retrieval for strategic guidance.
- Planning-agent task synthesis for objective alignment.
- Execution-agent arbitration across candidate actions.
- Persistent learning updates from completed matches.
"""

from __future__ import annotations

import json
import random
import sys
import time

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

# Add repository root to sys.path so shared src/ and logs/ imports resolve.
games_root = Path(__file__).resolve().parent
project_root = games_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.agent_factory import AgentFactory
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.planning.planning_types import Task, TaskType
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Chronos")
printer = PrettyPrinter()

PIECE_VALUES = {"Strategos": 3.0, "Warden": 2.0, "Scout": 1.0}
ACTION_BASE_WEIGHTS = {
    "attack": 42.0,
    "claim": 36.0,
    "move": 24.0,
    "pass": -22.0,
}


@dataclass
class ChronosAI:
    game: str = "chronos"
    initialized_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self) -> None:
        self.shared_memory = SharedMemory()
        self.factory = AgentFactory()

        self.knowledge_agent = self.factory.create("knowledge", self.shared_memory)
        self.planning_agent = self.factory.create("planning", self.shared_memory)
        self.execution_agent = self.factory.create("execution", self.shared_memory)
        self.learning_agent = self.factory.create("learning", self.shared_memory)
        self.adaptive_agent = self.factory.create("adaptive", self.shared_memory)

        self._planning_enabled = True

        self.match_log_path = project_root / "logs" / "chronos_matches.jsonl"
        self.learning_store_path = project_root / "logs" / "chronos_learning_state.json"

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
        self.zone_weights: dict[str, float] = {"core": 0.0, "near_core": 0.0, "perimeter": 0.0}

        self._load_learning_state()
        self.shared_memory.set("chronos_ai_status", "initialized")
        self.shared_memory.set("chronos_stats", self.stats)
        logger.info("Chronos AI initialized with SLAI knowledge/planning/execution/learning agents")

    # ------------------------------ public API ------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "agent_status": "ready",
            "initialized_at": self.initialized_at,
            "agents": ["knowledge", "planning", "execution", "learning"],
            "game": self.game,
            "stats": self.stats,
            "action_weights": self.action_weights,
            "zone_weights": self.zone_weights,
        }

    def get_move(self, game_state: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(game_state, dict):
            return None

        phase = game_state.get("phase")
        if phase == "strategos_decision":
            choice = self._choose_mutual_strategos_choice(game_state)
            self.shared_memory.set("chronos_last_decision", {"phase": phase, "choice": choice, "time": time.time()})
            return {"choice": choice}

        valid_moves = game_state.get("validMoves", [])
        if not isinstance(valid_moves, list) or not valid_moves:
            return None

        strategy_context = self._get_strategy_context(game_state)
        plan = self._generate_plan(game_state, strategy_context)

        best_move, best_score, top_candidates = self._select_move_via_execution(
            valid_moves=valid_moves,
            game_state=game_state,
            strategy_context=strategy_context,
            plan=plan,
        )
        if best_move is None:
            best_move = random.choice(valid_moves)

        self.shared_memory.set(
            "chronos_last_decision",
            {
                "move": best_move,
                "score": best_score,
                "top_candidates": top_candidates,
                "plan_steps": len(plan) if isinstance(plan, list) else 0,
                "timestamp": time.time(),
            },
        )
        return best_move

    def learn_from_game(self, payload: dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False

        try:
            enriched_payload = {
                **payload,
                "logged_at": datetime.utcnow().isoformat(),
                "game": self.game,
            }
            self.match_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.match_log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{json.dumps(enriched_payload, ensure_ascii=False)}\\n")

            if hasattr(self.learning_agent, "learn"):
                cast(Any, self.learning_agent).learn(enriched_payload)

            self._update_stats(payload)
            self._update_preferences(payload)
            self._save_learning_state()

            self.shared_memory.set("chronos_last_game", enriched_payload)
            self.shared_memory.set("chronos_stats", self.stats)
            return True
        except Exception as error:  # noqa: BLE001
            logger.warning("Chronos learning update failed: %s", error)
            return False

    # --------------------------- strategos decision -------------------------

    def _choose_mutual_strategos_choice(self, game_state: dict[str, Any]) -> str:
        players = game_state.get("players", [])
        ai_score = 0
        human_score = 0
        if isinstance(players, list):
            for player in players:
                if not isinstance(player, dict):
                    continue
                if player.get("id") == 1:
                    ai_score = int(player.get("score", 0) or 0)
                elif player.get("id") == 0:
                    human_score = int(player.get("score", 0) or 0)

        units = self._extract_units(game_state)
        ai_units = sum(1 for unit in units if unit.get("owner") == 1 and int(unit.get("hp", 0) or 0) > 0)
        human_units = sum(1 for unit in units if unit.get("owner") == 0 and int(unit.get("hp", 0) or 0) > 0)

        # Continue if we have parity or initiative edge; end if materially behind.
        if ai_score + ai_units >= human_score + human_units:
            return "continue"
        return "end"

    # --------------------------- multi-agent pipeline -----------------------

    def _get_strategy_context(self, game_state: dict[str, Any]) -> str:
        baseline = (
            "Chronos strategy: contest central core tempo, coordinate claim pressure with safe token usage, "
            "and prioritize forcing attacks that remove Warden screens before Strategos exposure."
        )
        board_size = self._board_size(game_state)
        query = f"Chronos board-size-{board_size} tactical strategy for core control, claim timing, and commander safety"
        try:
            if hasattr(self.knowledge_agent, "query"):
                response = cast(Any, self.knowledge_agent).query(query)
                if response:
                    context = str(response)
                    self.shared_memory.set("chronos_strategy_context", context)
                    return context
        except Exception as error:  # noqa: BLE001
            logger.warning("Chronos knowledge query failed: %s", error)
        return baseline

    def _generate_plan(self, game_state: dict[str, Any], strategy_context: str) -> list[Task] | None:
        if not self._planning_enabled:
            return None

        try:
            now = time.time()
            plan_uid = f"{int(now)}-{int(game_state.get('round', 0) or 0)}"
            fallback = Task(
                name=f"chronos_select_move_fallback_{plan_uid}",
                task_type=TaskType.PRIMITIVE,
                start_time=now + 30,
                deadline=now + 330,
                duration=20,
                context={"game": self.game},
            )
            goal = Task(
                name=f"chronos_select_best_move_{plan_uid}",
                task_type=TaskType.ABSTRACT,
                start_time=now + 20,
                deadline=now + 360,
                methods=[[fallback]],
                goal_state={"move_selected": True},
                context={
                    "game_state": {
                        "round": game_state.get("round"),
                        "phase": game_state.get("phase"),
                        "valid_move_count": len(game_state.get("validMoves", [])),
                    },
                    "strategy": strategy_context,
                },
            )

            if hasattr(self.planning_agent, "register_task"):
                try:
                    cast(Any, self.planning_agent).register_task(goal)
                except Exception as reg_error:  # noqa: BLE001
                    logger.warning("Chronos planner task registration skipped: %s", reg_error)

            if hasattr(self.planning_agent, "generate_plan"):
                plan = cast(Any, self.planning_agent).generate_plan(goal)
                if isinstance(plan, list):
                    self.shared_memory.set(
                        "chronos_last_plan",
                        [getattr(step, "name", str(step)) for step in plan],
                    )
                    return plan
        except Exception as error:  # noqa: BLE001
            logger.warning("Chronos planning failed: %s", error)
            self._planning_enabled = False
        return None

    def _select_move_via_execution(
        self,
        valid_moves: list[dict[str, Any]],
        game_state: dict[str, Any],
        strategy_context: str,
        plan: list[Task] | None,
    ) -> tuple[dict[str, Any] | None, float, list[dict[str, Any]]]:
        scored_moves: list[tuple[dict[str, Any], float, str]] = []
        for move in valid_moves:
            score, reason = self._score_move(move, game_state, strategy_context, plan)
            scored_moves.append((move, score, reason))

        if not scored_moves:
            return None, float("-inf"), []

        scored_moves.sort(key=lambda entry: entry[1], reverse=True)
        best_score = scored_moves[0][1]
        top_band = [entry for entry in scored_moves if entry[1] >= best_score - 1.0]

        selected_move = scored_moves[0][0]
        if len(top_band) > 1:
            selected_move = random.choice(top_band)[0]

        # Execution agent arbitration: if available, let it select among top candidates.
        try:
            if hasattr(self.execution_agent, "action_selector"):
                action_candidates = [
                    {
                        "name": f"candidate_{index}",
                        "priority": max(1, int(score + 100)),
                        "move": move,
                        "index": index,
                    }
                    for index, (move, score, _reason) in enumerate(scored_moves[:12])
                ]
                execution_context = {
                    "board_size": self._board_size(game_state),
                    "round": game_state.get("round", 0),
                    "phase": game_state.get("phase", "planning"),
                    "knowledge_signals": self._extract_knowledge_signals(strategy_context),
                    "plan_signals": self._extract_plan_signals(plan),
                    "top_candidate_score": best_score,
                }
                selected = cast(Any, self.execution_agent).action_selector.select(action_candidates, execution_context)
                selected_name = selected.get("name") if isinstance(selected, dict) else None
                if isinstance(selected_name, str) and selected_name.startswith("candidate_"):
                    index = int(selected_name.split("_", 1)[1])
                    if 0 <= index < len(action_candidates):
                        selected_move = action_candidates[index]["move"]
        except Exception as error:  # noqa: BLE001
            logger.warning("Chronos execution arbitration skipped: %s", error)

        top_candidates = [
            {
                "move": move,
                "score": round(score, 3),
                "reason": reason,
            }
            for move, score, reason in scored_moves[:5]
        ]
        return selected_move, best_score, top_candidates

    # ------------------------------ scoring model ---------------------------

    def _score_move(
        self,
        move: dict[str, Any],
        game_state: dict[str, Any],
        strategy_context: str,
        plan: list[Task] | None,
    ) -> tuple[float, str]:
        move_type = str(move.get("type", "pass"))
        params = move.get("params") if isinstance(move.get("params"), dict) else {}
        target = move.get("target") if isinstance(move.get("target"), dict) else params.get("target", {})
        target = target if isinstance(target, dict) else {}

        unit_map = self._build_unit_map(game_state)
        acting_unit = unit_map.get(move.get("unitId"))
        if move_type != "pass" and not acting_unit:
            return -10000.0, "missing acting unit"

        board_size = self._board_size(game_state)
        tr = target.get("r") if isinstance(target.get("r"), int) else target.get("row")
        tc = target.get("c") if isinstance(target.get("c"), int) else target.get("col")

        score = ACTION_BASE_WEIGHTS.get(move_type, -8.0)
        reasons: list[str] = [f"base={round(score, 2)}", f"type={move_type}"]

        if isinstance(acting_unit, dict):
            unit_type = str(acting_unit.get("type", "Scout"))
            score += PIECE_VALUES.get(unit_type, 1.0) * 3.2
            reasons.append(f"unit={unit_type}")

        # Core pressure and board geometry.
        if isinstance(tr, int) and isinstance(tc, int):
            zone = self._classify_zone(tr, tc, board_size)
            score += {"core": 42.0, "near_core": 19.0, "perimeter": 4.0}.get(zone, 0.0)
            score += self.zone_weights.get(zone, 0.0)
            reasons.append(f"zone={zone}")

            distance_penalty = self._distance_to_center(tr, tc, board_size) * 1.2
            score -= distance_penalty

        # Tactical gains.
        if move_type == "attack" and isinstance(target, dict):
            target_type = target.get("type")
            target_value = PIECE_VALUES.get(str(target_type), 1.0)
            score += target_value * 30.0
            if target_type == "Strategos":
                score += 5000.0
                reasons.append("capture_strategos")

        if move_type == "claim":
            score += 35.0
            if isinstance(acting_unit, dict) and acting_unit.get("type") == "Strategos":
                score += 12.0

        # Safety: avoid high-threat landing cells.
        if isinstance(acting_unit, dict):
            destination = self._infer_destination(move, acting_unit)
            threat = self._estimate_enemy_threat(destination, acting_unit, unit_map)
            score -= threat * 20.0
            if threat > 0:
                reasons.append(f"threat={threat}")

        # Token efficiency (lower token preserves flexibility).
        token_id = move.get("tokenId")
        if isinstance(token_id, int):
            score += max(0, 6 - token_id) * 1.6

        # Plan alignment bonus.
        if plan:
            plan_names = [getattr(step, "name", "") for step in plan]
            if any(move_type in str(name).lower() for name in plan_names):
                score += 11.0
                reasons.append("plan_aligned")

        # Strategy-context lexical signals.
        lowered = strategy_context.lower() if isinstance(strategy_context, str) else ""
        if "core" in lowered and isinstance(tr, int) and isinstance(tc, int) and self._is_core_cell(tr, tc, board_size):
            score += 9.0
        if "protect" in lowered and isinstance(acting_unit, dict) and acting_unit.get("type") == "Strategos":
            score += 8.0
        if "aggressive" in lowered and move_type == "attack":
            score += 7.5

        # Learned action priors + controlled exploration.
        score += self.action_weights.get(move_type, 0.0)
        score += random.uniform(0.0, 2.5)

        return score, "; ".join(reasons)

    # ------------------------------- learning -------------------------------

    def _update_stats(self, payload: dict[str, Any]) -> None:
        outcome = str(payload.get("outcome", "draw")).lower()
        reward = float(payload.get("reward", 0.0) or 0.0)
        final_score = float(payload.get("final_score", 0.0) or 0.0)

        self.stats["games_played"] += 1
        if outcome == "win":
            self.stats["wins"] += 1
        elif outcome == "loss":
            self.stats["losses"] += 1
        else:
            self.stats["draws"] += 1

        games_played = max(1, int(self.stats["games_played"]))
        prev_reward_avg = float(self.stats["average_reward"])
        prev_score_avg = float(self.stats["average_final_score"])

        self.stats["average_reward"] = round(
            prev_reward_avg + ((reward - prev_reward_avg) / games_played),
            4,
        )
        self.stats["average_final_score"] = round(
            prev_score_avg + ((final_score - prev_score_avg) / games_played),
            3,
        )
        self.stats["last_result"] = outcome
        self.stats["last_updated"] = datetime.utcnow().isoformat()

    def _update_preferences(self, payload: dict[str, Any]) -> None:
        outcome = str(payload.get("outcome", "draw")).lower()
        delta = 0.0
        if outcome == "win":
            delta = 0.6
        elif outcome == "loss":
            delta = -0.5
        else:
            delta = 0.1

        last_decision = self.shared_memory.get("chronos_last_decision") or {}
        move = last_decision.get("move") if isinstance(last_decision, dict) else None
        if isinstance(move, dict):
            action = str(move.get("type", "pass"))
            self.action_weights[action] = max(-8.0, min(12.0, self.action_weights.get(action, 0.0) + delta))

            target = move.get("target") if isinstance(move.get("target"), dict) else {}
            board_size = int(payload.get("board_size", 9) or 9)
            r = target.get("r") if isinstance(target.get("r"), int) else target.get("row")
            c = target.get("c") if isinstance(target.get("c"), int) else target.get("col")
            if isinstance(r, int) and isinstance(c, int):
                zone = self._classify_zone(r, c, board_size)
                self.zone_weights[zone] = max(-6.0, min(10.0, self.zone_weights.get(zone, 0.0) + (delta * 0.5)))

        if hasattr(self.learning_agent, "observe"):
            try:
                signal = {
                    "reward": float(payload.get("reward", 0.0) or 0.0),
                    "outcome": outcome,
                    "board_size": int(payload.get("board_size", 9) or 9),
                    "weights": {
                        "actions": self.action_weights,
                        "zones": self.zone_weights,
                    },
                }
                cast(Any, self.learning_agent).observe(signal)
            except Exception as error:  # noqa: BLE001
                logger.warning("Learning-agent observe skipped: %s", error)

    def _load_learning_state(self) -> None:
        if not self.learning_store_path.exists():
            return
        try:
            payload = json.loads(self.learning_store_path.read_text(encoding="utf-8"))
            self.stats.update(payload.get("stats", {}))
            self.action_weights.update(payload.get("action_weights", {}))
            self.zone_weights.update(payload.get("zone_weights", {}))
        except Exception as error:  # noqa: BLE001
            logger.warning("Failed to load Chronos learning state: %s", error)

    def _save_learning_state(self) -> None:
        try:
            self.learning_store_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "stats": self.stats,
                "action_weights": self.action_weights,
                "zone_weights": self.zone_weights,
                "updated_at": datetime.utcnow().isoformat(),
            }
            self.learning_store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as error:  # noqa: BLE001
            logger.warning("Failed to persist Chronos learning state: %s", error)

    # ------------------------------ state helpers ---------------------------

    def _extract_units(self, game_state: dict[str, Any]) -> list[dict[str, Any]]:
        units = game_state.get("units", [])
        if isinstance(units, list):
            return [unit for unit in units if isinstance(unit, dict)]
        return []

    def _build_unit_map(self, game_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        unit_map: dict[str, dict[str, Any]] = {}
        for unit in self._extract_units(game_state):
            unit_id = unit.get("id")
            if not isinstance(unit_id, str):
                continue
            unit_map[unit_id] = {
                "id": unit_id,
                "type": unit.get("type"),
                "owner": unit.get("owner", unit.get("playerId")),
                "r": unit.get("r", unit.get("row")),
                "c": unit.get("c", unit.get("col")),
                "hp": unit.get("hp", unit.get("health", 1)),
            }
        return unit_map

    def _infer_destination(self, move: dict[str, Any], acting_unit: dict[str, Any]) -> tuple[int, int] | None:
        move_type = move.get("type")
        if move_type == "move":
            target = move.get("target") if isinstance(move.get("target"), dict) else {}
            r = target.get("r") if isinstance(target.get("r"), int) else target.get("row")
            c = target.get("c") if isinstance(target.get("c"), int) else target.get("col")
            if isinstance(r, int) and isinstance(c, int):
                return r, c

        r = acting_unit.get("r")
        c = acting_unit.get("c")
        if isinstance(r, int) and isinstance(c, int):
            return r, c
        return None

    def _estimate_enemy_threat(
        self,
        destination: tuple[int, int] | None,
        acting_unit: dict[str, Any],
        unit_map: dict[str, dict[str, Any]],
    ) -> int:
        if destination is None:
            return 0
        dr, dc = destination
        owner = acting_unit.get("owner")
        threat = 0
        for enemy in unit_map.values():
            if enemy.get("owner") == owner:
                continue
            if int(enemy.get("hp", 0) or 0) <= 0:
                continue
            er, ec = enemy.get("r"), enemy.get("c")
            if not isinstance(er, int) or not isinstance(ec, int):
                continue
            distance = max(abs(dr - er), abs(dc - ec))
            enemy_range = 2 if enemy.get("type") == "Scout" else 1
            if distance <= enemy_range:
                threat += 1
        return threat

    def _extract_plan_signals(self, plan: list[Task] | None) -> dict[str, Any]:
        if not isinstance(plan, list):
            return {"step_count": 0, "action_bias": {}}
        action_bias: dict[str, int] = {}
        for step in plan:
            name = getattr(step, "name", None)
            if isinstance(name, str):
                action_bias[name] = action_bias.get(name, 0) + 1
        return {"step_count": len(plan), "action_bias": action_bias}

    def _extract_knowledge_signals(self, strategy_context: str) -> dict[str, bool]:
        text = strategy_context.lower() if isinstance(strategy_context, str) else ""
        return {
            "aggressive": "aggressive" in text,
            "core_control": "core" in text,
            "protect": "protect" in text or "safety" in text,
        }

    def _board_size(self, game_state: dict[str, Any]) -> int:
        board = game_state.get("board")
        if isinstance(board, list) and board:
            return len(board)

        units = self._extract_units(game_state)
        max_axis = 8
        for unit in units:
            r = unit.get("r") if isinstance(unit.get("r"), int) else unit.get("row")
            c = unit.get("c") if isinstance(unit.get("c"), int) else unit.get("col")
            if isinstance(r, int):
                max_axis = max(max_axis, r)
            if isinstance(c, int):
                max_axis = max(max_axis, c)
        return max_axis + 1

    def _distance_to_center(self, r: int, c: int, board_size: int) -> int:
        center = board_size // 2
        return max(abs(r - center), abs(c - center))

    def _is_core_cell(self, r: int, c: int, board_size: int) -> bool:
        center = board_size // 2
        return (center - 1) <= r <= (center + 1) and (center - 1) <= c <= (center + 1)

    def _classify_zone(self, r: int, c: int, board_size: int) -> str:
        if self._is_core_cell(r, c, board_size):
            return "core"

        center = board_size // 2
        if max(abs(r - center), abs(c - center)) <= 2:
            return "near_core"
        return "perimeter"


_ai_player: ChronosAI | None = None


def initialize_ai() -> ChronosAI:
    global _ai_player
    if _ai_player is None:
        _ai_player = ChronosAI()
    return _ai_player
