"""Aether Shift AI initialization + move selection runtime.

This module provides a production-grade integration for Aether Shift with
knowledge/planning/execution/learning agents, plus a deterministic tactical
search layer that understands Aether's board mechanics.
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

# Add repository root to sys.path so shared src/ and logs/ imports resolve
games_root = Path(__file__).resolve().parent
project_root = games_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agents.agent_factory import AgentFactory
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.planning.planning_types import Task, TaskType
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Aether Shift")
printer = PrettyPrinter()

BOARD_SIZE = 5
POWER_WELLS = ({"row": 0, "col": 0}, {"row": 0, "col": 4}, {"row": 4, "col": 0}, {"row": 4, "col": 4})
TILE_CONNECTIONS = {
    "STRAIGHT": [True, False, True, False],
    "CURVE": [True, True, False, False],
    "T_JUNCTION": [False, True, True, True],
    "CROSS": [True, True, True, True],
}
PLACE_TYPES = ("STRAIGHT", "CURVE", "T_JUNCTION")
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_BASE_WEIGHTS = {
    "ATTUNE": 32.0,
    "ADVANCE": 24.0,
    "PLACE": 14.0,
    "ROTATE": 12.0,
    "SHIFT": 11.0,
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

        self.match_log_path = project_root / "logs" / "aether_matches.jsonl"
        self.learning_store_path = project_root / "logs" / "aether_learning_state.json"

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
        self.action_weights: dict[str, float] = {}
        self._load_learning_state()

        self.shared_memory.set("aether_ai_status", "initialized")
        self.shared_memory.set("aether_stats", self.stats)
        self.shared_memory.set("aether_action_weights", self.action_weights)
        logger.info("Aether Shift AI initialized with Knowledge, Planning, Execution, and Learning agents")

    def health(self) -> dict[str, Any]:
        return {
            "agent_status": "ready",
            "initialized_at": self.initialized_at,
            "agents": ["knowledge", "planning", "execution", "learning"],
            "game": self.game,
            "stats": self.stats,
            "action_weights": self.action_weights,
        }

    def get_move(self, game_state: dict[str, Any]) -> dict[str, Any] | None:
        valid_moves = game_state.get("validMoves", []) if isinstance(game_state, dict) else []
        if not valid_moves:
            return None

        self.shared_memory.set("aether_last_state", game_state)
        strategy_context = self._get_strategy_context()
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
            "aether_last_decision",
            {
                "move": best_move,
                "score": best_score,
                "plan_steps": len(plan) if isinstance(plan, list) else 0,
                "top_candidates": top_candidates,
                "timestamp": time.time(),
            },
        )
        return best_move

    def learn_from_game(self, payload: dict[str, Any]) -> bool:
        try:
            if hasattr(self.learning_agent, "learn"):
                cast(Any, self.learning_agent).learn(payload)

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

            self._update_score_stats(payload)
            self._update_action_preferences(payload)
            self._save_learning_state()
            return True
        except Exception as error:  # noqa: BLE001
            logger.warning("Aether Shift learning update failed: %s", error)
            return False

    def _get_strategy_context(self) -> str:
        baseline = (
            "Aether Shift strategy: secure path tempo, deny enemy path completion with shifts/rotations, "
            "and prioritize power-well locks when they create immediate or forced wins."
        )
        try:
            if hasattr(self.knowledge_agent, "query"):
                result = cast(Any, self.knowledge_agent).query("Aether Shift strategy for board control and wells")
                context = str(result) if result else baseline
                self.shared_memory.set("aether_strategy_context", context)
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
                name="select_best_aether_move_fallback",
                task_type=TaskType.PRIMITIVE,
                start_time=now + 3,
                deadline=now + 180,
                duration=30,
                context={"game": self.game},
            )
            goal_task = Task(
                name="select_best_aether_move",
                task_type=TaskType.ABSTRACT,
                start_time=now + 1,
                deadline=now + 240,
                methods=[[fallback_task]],
                goal_state={"move_selected": True},
                context={"game_state": game_state, "strategy": strategy_context},
            )

            if not self._planning_task_registered and hasattr(self.planning_agent, "register_task"):
                cast(Any, self.planning_agent).register_task(goal_task)
                self._planning_task_registered = True

            if hasattr(self.planning_agent, "generate_plan"):
                plan = cast(Any, self.planning_agent).generate_plan(goal_task)
                if isinstance(plan, list):
                    self.shared_memory.set("aether_last_plan", [getattr(step, "name", str(step)) for step in plan])
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
    ) -> tuple[dict[str, Any] | None, float, list[dict[str, Any]]]:
        execution_bonus = 0.0
        try:
            if hasattr(self.execution_agent, "predict"):
                execution_signal = cast(Any, self.execution_agent).predict(
                    {
                        "game": self.game,
                        "strategy": strategy_context,
                        "plan_size": len(plan) if plan else 0,
                        "turn": game_state.get("turn"),
                        "actions_remaining": game_state.get("actionsRemaining"),
                    }
                )
                if isinstance(execution_signal, dict) and execution_signal.get("selected_action") != "idle":
                    execution_bonus = float(execution_signal.get("confidence", 0.0)) * 2.0
        except Exception as error:  # noqa: BLE001
            logger.warning("Execution agent prediction failed: %s", error)

        scored_moves: list[tuple[dict[str, Any], float, str]] = []
        for move in valid_moves:
            score, reason = self._score_move(move, game_state, strategy_context, plan)
            scored_moves.append((move, score + execution_bonus, reason))

        if not scored_moves:
            return None, float("-inf"), []

        scored_moves.sort(key=lambda item: item[1], reverse=True)
        best_score = scored_moves[0][1]
        top_band = [entry for entry in scored_moves if entry[1] >= best_score - 1.0]
        best_move = random.choice(top_band)[0] if top_band else scored_moves[0][0]

        top_candidates = [
            {
                "move": move,
                "score": round(score, 3),
                "reason": reason,
            }
            for move, score, reason in scored_moves[:5]
        ]
        return best_move, best_score, top_candidates

    def _score_move(
        self,
        move: dict[str, Any],
        game_state: dict[str, Any],
        strategy_context: str,
        plan: list[Task] | None,
    ) -> tuple[float, str]:
        action = self._resolve_action(game_state, move)
        if not action:
            return -10000.0, "Unresolvable action"

        active_player = int(game_state.get("activePlayer", 2))
        opponent_id = 1 if active_player == 2 else 2
        action_weight = self.action_weights.get(action, 0.0)

        score = ACTION_BASE_WEIGHTS.get(action, 0.0) + action_weight
        reasons: list[str] = [f"action={action}"]

        simulated = self._simulate_move(game_state, move, action)
        if simulated is None:
            return -9000.0, "Simulation failed"

        if simulated.get("winner") == active_player:
            return 500000.0, "Immediate winning move"
        if simulated.get("winner") == opponent_id:
            return -500000.0, "Self-losing move"

        score += self._evaluate_state(simulated, active_player) * 1.0
        score -= self._evaluate_state(simulated, opponent_id) * 0.55

        if action in {"ADVANCE", "ATTUNE", "ROTATE"} and self._is_on_power_well(simulated, active_player):
            score += 18
            reasons.append("occupy/capture well lane")

        if action == "ATTUNE":
            captured_before = self._captured_count(game_state, active_player)
            captured_after = self._captured_count(simulated, active_player)
            if captured_after > captured_before:
                gained = captured_after - captured_before
                score += 60 * gained
                reasons.append(f"captured_wells+{gained}")
                if captured_after >= 3:
                    score += 12000
                    reasons.append("threatens/achieves well victory")

        if action == "SHIFT":
            opp_progress_before = self._path_progress(game_state, opponent_id)
            opp_progress_after = self._path_progress(simulated, opponent_id)
            if opp_progress_after < opp_progress_before:
                delta = opp_progress_before - opp_progress_after
                score += delta * 0.9
                reasons.append(f"opponent_path_disrupted={delta:.1f}")

        # Two-action lookahead: if we still move this turn, inspect strongest continuation.
        if int(game_state.get("actionsRemaining", 0)) > 1 and simulated.get("activePlayer") == active_player:
            continuation = self._best_follow_up_score(simulated, active_player)
            score += continuation * 0.35
            reasons.append(f"followup={continuation:.1f}")

        # Opponent reply risk after our move.
        if simulated.get("activePlayer") == opponent_id:
            opp_reply = self._best_follow_up_score(simulated, opponent_id, max_moves=28)
            score -= opp_reply * 0.25
            reasons.append(f"opp_reply_risk={opp_reply:.1f}")

        if strategy_context:
            score += 2.5
        if plan:
            score += min(len(plan), 5)

        return score, "; ".join(reasons)

    def _evaluate_state(self, state: dict[str, Any], player_id: int) -> float:
        if state.get("winner") == player_id:
            return 100000.0

        opponent_id = 1 if player_id == 2 else 2
        if state.get("winner") == opponent_id:
            return -100000.0

        player = self._get_player(state, player_id)
        if not player:
            return -5000.0

        my_wells = self._captured_count(state, player_id)
        opp_wells = self._captured_count(state, opponent_id)

        score = my_wells * 5200 - opp_wells * 6000
        my_progress = self._path_progress(state, player_id)
        opp_progress = self._path_progress(state, opponent_id)
        score += my_progress * 90 - opp_progress * 115

        if isinstance(player.get("position"), dict):
            row = player["position"].get("row")
            goal_row = player.get("goalRow")
            if isinstance(row, int) and isinstance(goal_row, int):
                score += max(0, BOARD_SIZE - abs(goal_row - row)) * 150
            if self._is_on_power_well(state, player_id):
                score += 850

        # Piece economy: more resonators retained is useful unless near win.
        resonators = player.get("resonators")
        if isinstance(resonators, int):
            score += resonators * 45

        return score

    def _best_follow_up_score(self, state: dict[str, Any], player_id: int, max_moves: int = 36) -> float:
        moves = state.get("validMoves") if isinstance(state.get("validMoves"), list) else []
        if not moves:
            moves = self._infer_valid_moves(state)
        if not moves:
            return -50.0

        best = -100000.0
        for move in moves[:max_moves]:
            action = self._resolve_action(state, move)
            if not action:
                continue
            next_state = self._simulate_move(state, move, action)
            if next_state is None:
                continue
            val = self._evaluate_state(next_state, player_id)
            if next_state.get("winner") == player_id:
                val += 10000
            best = max(best, val)
        return best if best > -100000.0 else -100.0

    def _simulate_move(self, game_state: dict[str, Any], move: dict[str, Any], action: str) -> dict[str, Any] | None:
        state = json.loads(json.dumps(game_state))
        target = move.get("target") if isinstance(move.get("target"), dict) else None
        if not target:
            return None

        row = target.get("row")
        col = target.get("col")
        if not isinstance(row, int) or not isinstance(col, int):
            return None
        if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
            return None

        board = state.get("board")
        active_player = int(state.get("activePlayer", 1))
        player = self._get_player(state, active_player)
        if player is None or not isinstance(board, list):
            return None

        if action == "PLACE":
            if board[row][col] is not None:
                return None
            if not self._is_adjacent(player.get("position"), {"row": row, "col": col}):
                return None
            tile_type = self._best_place_type(state, active_player, row, col)
            board[row][col] = {
                "id": f"sim-{time.time_ns()}",
                "type": tile_type,
                "rotation": 0,
                "color": player.get("color", "neutral"),
                "hasResonator": False,
                "playersPresent": [],
            }

        elif action == "ROTATE":
            tile = board[row][col]
            if tile is None:
                return None
            if tile.get("hasResonator") and int(tile.get("resonatorOwner", 0)) != active_player:
                return None
            tile["rotation"] = (int(tile.get("rotation", 0)) + 90) % 360

        elif action == "ADVANCE":
            current_pos = player.get("position")
            if not self._is_valid_move(board, current_pos, {"row": row, "col": col}):
                return None
            current_coords = self._extract_position_coords(current_pos)
            if current_coords is None:
                return None
            current_row, current_col = current_coords
            old_tile = board[current_row][current_col]
            if old_tile and isinstance(old_tile.get("playersPresent"), list):
                old_tile["playersPresent"] = [pid for pid in old_tile["playersPresent"] if int(pid) != active_player]
            player["position"] = {"row": row, "col": col}
            new_tile = board[row][col]
            if new_tile is not None:
                present = new_tile.setdefault("playersPresent", [])
                if active_player not in present:
                    present.append(active_player)

        elif action == "ATTUNE":
            tile = board[row][col]
            if tile is None:
                return None
            pos = player.get("position")
            if not isinstance(pos, dict) or pos.get("row") != row or pos.get("col") != col:
                return None
            if tile.get("hasResonator"):
                return None
            if int(player.get("resonators", 0)) <= 0:
                return None
            if self._is_power_well(row, col) and tile.get("color") == "neutral":
                return None

            tile["hasResonator"] = True
            tile["resonatorOwner"] = active_player
            player["resonators"] = int(player.get("resonators", 0)) - 1

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
        active_player = int(state.get("activePlayer", 1))

        state["actionsRemaining"] = int(state.get("actionsRemaining", 2)) - 1
        state["capturedWells"] = self._refresh_captured_wells(state)

        if self._path_progress(state, active_player) >= 100:
            state["winner"] = active_player
            state["winReason"] = "Path Completed!"
            return

        if int(state.get("actionsRemaining", 0)) <= 0:
            if self._captured_count(state, active_player) >= 3:
                state["winner"] = active_player
                state["winReason"] = "3 Power Wells Captured!"
                return

            state["activePlayer"] = 1 if active_player == 2 else 2
            state["actionsRemaining"] = 2
            state["turn"] = int(state.get("turn", 0)) + 1

    def _infer_valid_moves(self, game_state: dict[str, Any]) -> list[dict[str, Any]]:
        moves: list[dict[str, Any]] = []
        face_up_cards = game_state.get("faceUpCards", [])
        for card in face_up_cards:
            actions = card.get("actions", []) if isinstance(card, dict) else []
            for action_index, action in enumerate(actions):
                for target in self._valid_targets_for_action(game_state, str(action)):
                    moves.append(
                        {
                            "cardId": card.get("id"),
                            "actionIndex": action_index,
                            "target": target,
                        }
                    )
        return moves

    def _valid_targets_for_action(self, game_state: dict[str, Any], action: str) -> list[dict[str, int]]:
        board = game_state.get("board", [])
        active_player = int(game_state.get("activePlayer", 1))
        player = self._get_player(game_state, active_player)
        if not isinstance(board, list) or player is None:
            return []

        targets: list[dict[str, int]] = []

        if action == "PLACE":
            position = player.get("position")
            if not isinstance(position, dict):
                return []
            for d_row, d_col in DIRECTIONS:
                row = position.get("row", -10) + d_row
                col = position.get("col", -10) + d_col
                if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row][col] is None:
                    targets.append({"row": row, "col": col})
            return targets

        if action == "ROTATE":
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    tile = board[row][col]
                    if tile is None:
                        continue
                    if tile.get("hasResonator") and int(tile.get("resonatorOwner", 0)) != active_player:
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
            position = player.get("position")
            if not isinstance(position, dict):
                return []
            row, col = position.get("row"), position.get("col")
            if isinstance(row, int) and isinstance(col, int):
                tile = board[row][col]
                if tile and not tile.get("hasResonator") and int(player.get("resonators", 0)) > 0:
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

    def _resolve_action(self, game_state: dict[str, Any], move: dict[str, Any]) -> str:
        cards = game_state.get("faceUpCards", [])
        card_id = move.get("cardId")
        action_index = int(move.get("actionIndex", 0))
        card = next((item for item in cards if isinstance(item, dict) and item.get("id") == card_id), None)
        if not card:
            return ""
        actions = card.get("actions", [])
        if 0 <= action_index < len(actions):
            action = actions[action_index]
            return str(action).upper()
        return ""

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
                popped = row_tiles.pop()
                row_tiles.insert(0, popped)
            else:
                popped = row_tiles.pop(0)
                row_tiles.append(popped)

            for player in players.values():
                pos = player.get("position") if isinstance(player, dict) else None
                if isinstance(pos, dict) and pos.get("row") == row:
                    col_value = pos.get("col")
                    if not isinstance(col_value, int):
                        continue
                    new_col = col_value + direction
                    if new_col >= BOARD_SIZE:
                        new_col = 0
                    if new_col < 0:
                        new_col = BOARD_SIZE - 1
                    pos["col"] = new_col

        if axis == "col":
            column = [board[r][col] for r in range(BOARD_SIZE)]
            if direction == 1:
                popped = column.pop()
                column.insert(0, popped)
            else:
                popped = column.pop(0)
                column.append(popped)
            for r in range(BOARD_SIZE):
                board[r][col] = column[r]

            for player in players.values():
                pos = player.get("position") if isinstance(player, dict) else None
                if isinstance(pos, dict) and pos.get("col") == col:
                    row_value = pos.get("row")
                    if not isinstance(row_value, int):
                        continue
                    new_row = row_value + direction
                    if new_row >= BOARD_SIZE:
                        new_row = 0
                    if new_row < 0:
                        new_row = BOARD_SIZE - 1
                    pos["row"] = new_row

    def _captured_count(self, state: dict[str, Any], player_id: int) -> int:
        captured = state.get("capturedWells", {})
        if not isinstance(captured, dict):
            return 0
        return sum(1 for owner in captured.values() if int(owner) == player_id)

    def _refresh_captured_wells(self, state: dict[str, Any]) -> dict[str, int]:
        captured: dict[str, int] = {}
        board = state.get("board", [])
        for well in POWER_WELLS:
            tile = board[well["row"]][well["col"]]
            if tile and tile.get("hasResonator") and tile.get("resonatorOwner"):
                captured[f"{well['row']},{well['col']}"] = int(tile["resonatorOwner"])
        return captured

    def _path_progress(self, state: dict[str, Any], player_id: int) -> float:
        player = self._get_player(state, player_id)
        board = state.get("board")
        if not player or not isinstance(board, list):
            return 0.0

        start_row = int(player.get("homeRow", 0))
        goal_row = int(player.get("goalRow", 0))

        queue: list[tuple[int, int]] = []
        visited: set[tuple[int, int]] = set()
        max_progress = 0

        for col in range(BOARD_SIZE):
            if board[start_row][col] is not None:
                queue.append((start_row, col))
                visited.add((start_row, col))

        head = 0
        while head < len(queue):
            curr_row, curr_col = queue[head]
            head += 1

            max_progress = max(max_progress, abs(curr_row - start_row))
            if curr_row == goal_row:
                return 100.0

            for d_row, d_col in DIRECTIONS:
                next_row = curr_row + d_row
                next_col = curr_col + d_col
                if (next_row, next_col) in visited:
                    continue
                if self._is_valid_move(
                    board,
                    {"row": curr_row, "col": curr_col},
                    {"row": next_row, "col": next_col},
                ):
                    visited.add((next_row, next_col))
                    queue.append((next_row, next_col))

        return min(99.0, (max_progress / (BOARD_SIZE - 1)) * 90.0)

    def _is_valid_move(self, board: list[list[Any]], from_pos: Any, to_pos: Any) -> bool:
        from_coords = self._extract_position_coords(from_pos)
        to_coords = self._extract_position_coords(to_pos)
        if from_coords is None or to_coords is None:
            return False

        from_row, from_col = from_coords
        to_row, to_col = to_coords
        if not (0 <= to_row < BOARD_SIZE and 0 <= to_col < BOARD_SIZE):
            return False

        from_tile = board[from_row][from_col]
        to_tile = board[to_row][to_col]
        if from_tile is None or to_tile is None:
            return False

        from_conns = self._rotated_connections(str(from_tile.get("type")), int(from_tile.get("rotation", 0)))
        to_conns = self._rotated_connections(str(to_tile.get("type")), int(to_tile.get("rotation", 0)))

        d_row = to_row - from_row
        d_col = to_col - from_col
        if d_row == -1 and d_col == 0:
            return from_conns[0] and to_conns[2]
        if d_row == 0 and d_col == 1:
            return from_conns[1] and to_conns[3]
        if d_row == 1 and d_col == 0:
            return from_conns[2] and to_conns[0]
        if d_row == 0 and d_col == -1:
            return from_conns[3] and to_conns[1]
        return False

    def _rotated_connections(self, tile_type: str, rotation: int) -> list[bool]:
        base = TILE_CONNECTIONS.get(tile_type, [False, False, False, False])
        shifts = (rotation // 90) % 4
        rotated = list(base)
        for _ in range(shifts):
            rotated.insert(0, rotated.pop())
        return rotated

    def _best_place_type(self, state: dict[str, Any], player_id: int, row: int, col: int) -> str:
        # Choose the tile that maximizes local connectivity + path reach.
        best_type = PLACE_TYPES[0]
        best_score = float("-inf")

        for tile_type in PLACE_TYPES:
            temp = json.loads(json.dumps(state))
            temp_player = self._get_player(temp, player_id)
            player_color = temp_player.get("color", "neutral") if isinstance(temp_player, dict) else "neutral"
            temp["board"][row][col] = {
                "type": tile_type,
                "rotation": 0,
                "color": player_color,
                "hasResonator": False,
                "playersPresent": [],
            }
            local = self._evaluate_state(temp, player_id)
            if local > best_score:
                best_score = local
                best_type = tile_type

        return best_type

    def _is_adjacent(self, from_pos: Any, to_pos: dict[str, int]) -> bool:
        coords = self._extract_position_coords(from_pos)
        if coords is None:
            return False
        from_row, from_col = coords
        d_row = abs(from_row - to_pos["row"])
        d_col = abs(from_col - to_pos["col"])
        return (d_row + d_col) == 1

    @staticmethod
    def _extract_position_coords(position: Any) -> tuple[int, int] | None:
        if not isinstance(position, dict):
            return None
        row_value = position.get("row")
        col_value = position.get("col")
        if not isinstance(row_value, int) or not isinstance(col_value, int):
            return None
        return cast(int, row_value), cast(int, col_value)

    def _is_power_well(self, row: int, col: int) -> bool:
        return any(well["row"] == row and well["col"] == col for well in POWER_WELLS)

    def _is_on_power_well(self, state: dict[str, Any], player_id: int) -> bool:
        player = self._get_player(state, player_id)
        if not player:
            return False
        position = player.get("position")
        if not isinstance(position, dict):
            return False
        row, col = position.get("row"), position.get("col")
        if not isinstance(row, int) or not isinstance(col, int):
            return False
        return self._is_power_well(row, col)

    def _get_player(self, state: dict[str, Any], player_id: int) -> dict[str, Any] | None:
        players = state.get("players", {})
        if not isinstance(players, dict):
            return None
        return players.get(str(player_id), players.get(player_id))

    def _update_score_stats(self, payload: dict[str, Any]) -> None:
        ai_player_id = int(payload.get("aiPlayerId", payload.get("ai_player", 2)))
        winner = payload.get("winner")

        ai_score = self._coerce_number(payload.get("aiScore", payload.get("ai_score", 0)))
        opponent_score = self._coerce_number(payload.get("opponentScore", payload.get("opponent_score", 0)))

        games_played = int(self.stats.get("games_played", 0)) + 1
        self.stats["games_played"] = games_played

        prev_ai_avg = self._coerce_number(self.stats.get("average_ai_score", 0.0))
        prev_opp_avg = self._coerce_number(self.stats.get("average_opponent_score", 0.0))

        self.stats["average_ai_score"] = ((prev_ai_avg * (games_played - 1)) + ai_score) / games_played
        self.stats["average_opponent_score"] = ((prev_opp_avg * (games_played - 1)) + opponent_score) / games_played

        if winner in {"draw", None, 0}:
            self.stats["draws"] = int(self.stats.get("draws", 0)) + 1
            self.stats["last_result"] = "draw"
        elif self._coerce_int(winner, default=-1) == ai_player_id:
            self.stats["wins"] = int(self.stats.get("wins", 0)) + 1
            self.stats["last_result"] = "win"
        else:
            self.stats["losses"] = int(self.stats.get("losses", 0)) + 1
            self.stats["last_result"] = "loss"

        self.stats["last_updated"] = datetime.utcnow().isoformat()
        self.shared_memory.set("aether_stats", self.stats)

    def _update_action_preferences(self, payload: dict[str, Any]) -> None:
        ai_actions = payload.get("aiActions") or payload.get("ai_actions") or []
        if not isinstance(ai_actions, list):
            return

        result = str(self.stats.get("last_result", "draw"))
        learning_rate = 0.3 if result == "win" else (-0.25 if result == "loss" else 0.08)

        for move in ai_actions:
            if not isinstance(move, dict):
                continue
            action = str(move.get("action") or move.get("type") or "").upper()
            if action not in ACTION_BASE_WEIGHTS:
                continue
            previous = float(self.action_weights.get(action, 0.0))
            self.action_weights[action] = max(-8.0, min(8.0, previous + learning_rate))

        self.shared_memory.set("aether_action_weights", self.action_weights)

    def _load_learning_state(self) -> None:
        if not self.learning_store_path.exists():
            return

        try:
            payload = json.loads(self.learning_store_path.read_text(encoding="utf-8"))
            if isinstance(payload.get("stats"), dict):
                self.stats.update(payload["stats"])
            if isinstance(payload.get("action_weights"), dict):
                self.action_weights = {str(k): float(v) for k, v in payload["action_weights"].items()}
        except Exception as error:  # noqa: BLE001
            logger.warning("Failed to load Aether learning state: %s", error)

    def _save_learning_state(self) -> None:
        self.learning_store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_at": datetime.utcnow().isoformat(),
            "stats": self.stats,
            "action_weights": self.action_weights,
        }
        self.learning_store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _coerce_number(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _coerce_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

def initialize_ai() -> AetherShiftAI:
    ai = AetherShiftAI()
    logger.info("Aether Shift AI initialized at %s", ai.initialized_at)
    return ai
