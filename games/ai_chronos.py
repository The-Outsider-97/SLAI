import sys
import os
import json
import random
from datetime import datetime
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Add AI folder to sys.path so 'src' and 'logs' can be imported as top-level packages
ai_root = project_root / "AI"
sys.path.insert(0, str(ai_root))

try:
    from src.agents.agent_factory import AgentFactory
    from src.agents.collaborative.shared_memory import SharedMemory
    from src.agents.planning.planning_types import Task, TaskType
    from logs.logger import get_logger
except ImportError as e:
    print(f"Error importing modules: {e}")
    raise

logger = get_logger("AI_Main")

class AIPlayer:
    def __init__(self):
        logger.info("Initializing AI Player...")
        self.shared_memory = SharedMemory()
        self.factory = AgentFactory()
        
        # Create agents
        try:
            self.knowledge_agent = self.factory.create("knowledge", self.shared_memory)
            # Load R-Games strategy into knowledge agent
            try:
                self.knowledge_agent.load_from_directory()
                logger.info("Knowledge Agent loaded documents from templates.")
            except Exception as ke:
                logger.warning(f"Knowledge Agent failed to load documents: {ke}")

            self.planning_agent = self.factory.create("planning", self.shared_memory)
            self.execution_agent = self.factory.create("execution", self.shared_memory)
            self.learning_agent = self.factory.create("learning", self.shared_memory)
            self.planning_enabled = True
            self._planning_task_registered = False
            self._style_weights = {
                "aggression": 1.0,
                "core_control": 1.0,
                "safety": 1.0,
            }
            self._result_window = []
            self._board_preferences = {}
            self._rl_alpha = 0.2
            self._rl_gamma = 0.92
            self._rl_epsilon = 0.12
            self._rl_scale = 18.0
            self._rl_q_table = {}
            self._episode_trace = []
            self._performance_log = []
            self._current_game_id = None
            self._learning_store_path = project_root / "AI" / "logs" / "chronos_learning_state.json"
            self._load_learning_state()
            
            logger.info("AI Player initialized with Knowledge, Planning, Execution, and Learning agents.")
        except Exception as e:
            logger.error(f"Failed to initialize AI Player: {e}", exc_info=True)
            self.planning_enabled = False
            self._planning_task_registered = False
            # We don't raise here to allow the server to start even if agents fail
            pass

    def get_move(self, game_state):
        try:
            phase = game_state.get('phase')
            if phase == 'strategos_decision':
                choice = self._choose_mutual_strategos_choice(game_state)
                logger.info(f"Strategos decision phase: AI selected '{choice}'.")
                return {'choice': choice}

            valid_moves = game_state.get('validMoves', [])
            if not valid_moves:
                logger.warning("No valid moves provided in game state.")
                return None
            
            # 1. Knowledge Retrieval: Get strategy context
            strategy_context = ""
            try:
                # Query for R-Games-specific strategies using the best available API
                strategy_context = self._get_strategy_context()
                logger.info(f"Knowledge Agent retrieved strategy context: {strategy_context[:100]}...")
            except Exception as e:
                logger.warning(f"Knowledge Agent query failed: {e}")

            # 2. Planning: Generate a high-level plan for the current state
            plan = None
            if self.planning_enabled:
                try:
                    # Normalize optional planner registry payloads used by scheduler internals
                    agent_registry = self.shared_memory.get('agent_registry')
                    if not isinstance(agent_registry, dict):
                        self.shared_memory.set('agent_registry', {})

                    # Create a formal Task for the planning agent
                    fallback_task = Task(
                        name="select_best_move_fallback",
                        task_type=TaskType.PRIMITIVE,
                        start_time=10,
                        deadline=3600,
                        duration=300,
                    )
                    goal_task = Task(
                        name="select_best_move",
                        task_type=TaskType.ABSTRACT,
                        methods=[[fallback_task]],
                        goal_state={"move_selected": True},
                        context={
                            "game_state": game_state,
                            "strategy": strategy_context
                        }
                    )

                    # Register only once to avoid noisy duplicate-registration warnings.
                    if (
                        not self._planning_task_registered
                        and hasattr(self.planning_agent, 'register_task')
                    ):
                        self.planning_agent.register_task(goal_task)
                        self._planning_task_registered = True

                    plan = self.planning_agent.generate_plan(goal_task)
                    if not isinstance(plan, list):
                        plan = None
                        logger.warning("Planning Agent returned no usable plan. Disabling planner for this session.")
                        self.planning_enabled = False
                    logger.info(f"Planning Agent generated plan with {len(plan) if plan else 0} steps.")
                except Exception as e:
                    logger.warning(f"Planning Agent failed to generate plan: {e}")
                    # Prevent repeated scheduler/validation error spam for subsequent turns.
                    self.planning_enabled = False

            # 3. Execution: bridge high-level plan to concrete action selection.
            board_size = self._board_size(game_state)
            self._prepare_episode_tracking(game_state)
            self._adapt_style_for_board(board_size)
            self._apply_learning_strategy(game_state)
            opening_move = self._select_opening_pattern_move(valid_moves, game_state)
            if opening_move:
                logger.info(f"AI selected opening pattern move: {opening_move}")
                self._record_episode_step(game_state, opening_move)
                return opening_move

            best_move, best_score = self._select_move_via_execution_agent(
                valid_moves=valid_moves,
                game_state=game_state,
                strategy_context=strategy_context,
                plan=plan,
            )
            logger.info(f"AI selected move with score {best_score}: {best_move}")
            self._record_episode_step(game_state, best_move)
            return best_move

        except Exception as e:
            logger.error(f"Error in get_move: {e}", exc_info=True)
            # Fallback to random move if everything fails
            import random
            return random.choice(game_state.get('validMoves', [])) if game_state.get('validMoves') else None

    def _choose_mutual_strategos_choice(self, game_state):
        """Choose continue/end when both Strategos units are eliminated."""
        ai_player_id = 1
        human_player_id = 0

        players = game_state.get('players', []) if isinstance(game_state, dict) else []
        ai_score = 0
        human_score = 0
        if isinstance(players, list):
            for player in players:
                if not isinstance(player, dict):
                    continue
                if player.get('id') == ai_player_id:
                    ai_score = player.get('score', 0) or 0
                elif player.get('id') == human_player_id:
                    human_score = player.get('score', 0) or 0

        ai_units = 0
        human_units = 0
        for unit in self._extract_units(game_state):
            if not isinstance(unit, dict) or (unit.get('hp', 0) or 0) <= 0:
                continue
            owner = unit.get('owner')
            if owner == ai_player_id:
                ai_units += 1
            elif owner == human_player_id:
                human_units += 1

        # Keep playing when we're ahead or equal; end if materially behind.
        if ai_score < human_score and ai_units < human_units:
            return 'end'
        return 'continue'

    def _score_move(self, move, game_state, strategy, plan):
        """
        Heuristic scoring function that incorporates agent 'intelligence'.
        """
        score = 0.0
        
        # Piece values from rulebook
        PIECE_VALUES = {
            'Strategos': 3,
            'Warden': 2,
            'Scout': 1
        }
        
        move_type = move.get('type')
        params = move.get('params', {}) if isinstance(move.get('params', {}), dict) else {}
        target = params.get('target') or move.get('target') or {}
        if not isinstance(target, dict):
            target = {}
        tr, tc = target.get('r'), target.get('c')
        unit_id = move.get('unitId')
        
        unit_map, board_map = self._build_unit_and_board_maps(game_state)
        acting_unit = unit_map.get(unit_id)
        
        if not acting_unit:
            return -1000

        # 1. Core Control (High Priority)
        # Multipliers: Center = 2x, Adjacent = 1x
        # Use dynamic center instead of hardcoded 4
        board_size = self._board_size(game_state)
        center = board_size // 2
        if tr == center and tc == center:
            score += 100 * self._style_weights["core_control"]
        elif self._is_core_cell(tr, tc, board_size):
            score += 40 * self._style_weights["core_control"]
            
        # 2. Attack (High Priority)
        if move_type == 'attack':
            # Attacking is good, especially high-value targets
            target_unit_type = target.get('type')
            target_value = PIECE_VALUES.get(target_unit_type, 1)
            score += 50 * target_value * self._style_weights["aggression"]
            
            # If we can eliminate the Strategos, it's a winning move
            if target_unit_type == 'Strategos':
                score += 1000
            
        # 3. Piece Protection & Value
        # Moving high value pieces to safety or better positions
        destination = self._infer_destination(move, acting_unit)
        if destination is not None:
            destination_threat = self._estimate_enemy_threat(destination, acting_unit, unit_map, board_map)
            score -= destination_threat * 28 * self._style_weights["safety"]

            if acting_unit.get("type") == "Strategos":
                score -= destination_threat * 20
            elif acting_unit.get("type") == "Scout" and move_type == "move":
                # Scouts are more expendable and can be used for tempo.
                score += 3

        # 4. Mobility / initiative
        projected_mobility = self._estimate_mobility_after_move(move, acting_unit, unit_map, board_map)
        score += projected_mobility * 4

        # 5. Planning Alignment
        if plan and isinstance(plan, list):
            for step in plan:
                # If the plan suggests a specific unit or action type
                if hasattr(step, 'name') and step.name == move_type:
                    score += 25
                if hasattr(step, 'context') and step.context.get('unit_id') == unit_id:
                    score += 20

        # 6. Strategic Context (from Knowledge Agent)
        strategy_lower = strategy.lower()
        if "core control" in strategy_lower and self._is_core_cell(tr, tc, self._board_size(game_state)):
            score += 15
        if "aggressive" in strategy_lower and move_type == 'attack':
            score += 15
        if "protect" in strategy_lower and acting_unit.get("type") == "Strategos":
            score += 12

        # 7. Tactical priorities
        score += self._score_tactical_objectives(move, acting_unit, unit_map, board_map)

        # 8. RL value bonus + small stochasticity for exploration.
        state_key = self._encode_state(game_state)
        action_key = self._encode_action(move, acting_unit, game_state)
        score += self._rl_scale * self._q_value(state_key, action_key)

        # 9. Random factor for variety
        import random
        score += random.uniform(0, 5)
        
        return score

    def _select_move_via_execution_agent(self, valid_moves, game_state, strategy_context, plan):
        """
        Use the execution agent as the low-level selector between candidate moves.
        Falls back to direct heuristic scoring if execution subsystems are unavailable.
        """
        if not isinstance(valid_moves, list) or not valid_moves:
            return None, -float('inf')

        move_scores = []
        for index, move in enumerate(valid_moves):
            score = self._score_move(move, game_state, strategy_context, plan)
            move_scores.append((index, move, score))

        # Persist scored candidates for transparency and downstream learning.
        self.shared_memory.set(
            "execution_move_candidates",
            [
                {
                    "index": idx,
                    "move": candidate,
                    "score": score,
                }
                for idx, candidate, score in move_scores
            ],
        )

        # If execution agent is missing, deterministic fallback to max-score move.
        if not getattr(self, 'execution_agent', None):
            best_index, best_move, best_score = max(move_scores, key=lambda item: item[2])
            return best_move, best_score

        execution_context = self._build_execution_context(game_state, strategy_context, plan, move_scores)
        action_candidates = [
            {
                "name": f"move_{idx}",
                "priority": max(1, int(score)),
                "preconditions": [],
                "move_index": idx,
            }
            for idx, _move, score in move_scores
        ]

        try:
            selected = self.execution_agent.action_selector.select(action_candidates, execution_context)
            selected_index = selected.get("move_index")
            if selected_index is None and isinstance(selected.get("name"), str) and selected["name"].startswith("move_"):
                selected_index = int(selected["name"].split("_", 1)[1])

            if isinstance(selected_index, int):
                for idx, candidate_move, score in move_scores:
                    if idx == selected_index:
                        self.shared_memory.set("execution_last_selection", {"index": idx, "score": score})
                        return candidate_move, score
        except Exception as exec_error:
            logger.warning(f"Execution Agent action selection failed, falling back to heuristic ranking: {exec_error}")

        best_index, best_move, best_score = max(move_scores, key=lambda item: item[2])
        self.shared_memory.set("execution_last_selection", {"index": best_index, "score": best_score, "fallback": True})
        return best_move, best_score

    def _build_execution_context(self, game_state, strategy_context, plan, move_scores):
        """Compose a compact context payload used by the execution action selector."""
        plan_signals = self._extract_plan_signals(plan)
        knowledge_signals = self._extract_knowledge_signals(strategy_context)

        top_score = max((score for _idx, _move, score in move_scores), default=0.0)
        avg_score = (sum(score for _idx, _move, score in move_scores) / len(move_scores)) if move_scores else 0.0

        return {
            "energy": 10.0,
            "max_energy": 10.0,
            "has_destination": True,
            "object_detected": False,
            "hand_empty": True,
            "holding_object": False,
            "time_critical": False,
            "round": game_state.get("round", 0),
            "board_size": self._board_size(game_state),
            "top_candidate_score": top_score,
            "average_candidate_score": avg_score,
            "knowledge_signals": knowledge_signals,
            "plan_signals": plan_signals,
        }

    def _extract_plan_signals(self, plan):
        if not isinstance(plan, list):
            return {"step_count": 0, "action_bias": {}}

        action_bias = {}
        for step in plan:
            step_name = getattr(step, "name", None)
            if isinstance(step_name, str):
                action_bias[step_name] = action_bias.get(step_name, 0) + 1
        return {"step_count": len(plan), "action_bias": action_bias}

    def _extract_knowledge_signals(self, strategy_context):
        if not isinstance(strategy_context, str):
            return {"aggressive": False, "core_control": False, "protect": False}

        text = strategy_context.lower()
        return {
            "aggressive": "aggressive" in text,
            "core_control": "core control" in text,
            "protect": "protect" in text,
        }

    def _build_unit_and_board_maps(self, game_state):
        unit_map = {}
        board_map = {}

        board = game_state.get('board', [])
        if isinstance(board, list):
            for r, row in enumerate(board):
                if not isinstance(row, list):
                    continue
                for c, cell in enumerate(row):
                    if not isinstance(cell, dict):
                        continue
                    unit = cell.get('unit')
                    if not isinstance(unit, dict):
                        continue
                    normalized = {
                        'id': unit.get('id'),
                        'type': unit.get('type'),
                        'owner': unit.get('owner'),
                        'hp': unit.get('hp', 1),
                        'r': r,
                        'c': c,
                    }
                    unit_map[normalized['id']] = normalized
                    board_map[(r, c)] = normalized

        for unit in self._extract_units(game_state):
            unit_id = unit.get('id')
            if unit_id in unit_map:
                continue
            normalized = {
                'id': unit_id,
                'type': unit.get('type'),
                'owner': unit.get('owner'),
                'hp': unit.get('hp', 1),
                'r': unit.get('r'),
                'c': unit.get('c'),
            }
            unit_map[unit_id] = normalized
            if normalized['r'] is not None and normalized['c'] is not None:
                board_map[(normalized['r'], normalized['c'])] = normalized

        return unit_map, board_map

    def _infer_destination(self, move, acting_unit):
        move_type = move.get("type")
        if move_type == "move":
            target = move.get("target") or move.get("params", {}).get("target") or {}
            if isinstance(target, dict) and target.get("r") is not None and target.get("c") is not None:
                return (target.get("r"), target.get("c"))
        if acting_unit.get("r") is not None and acting_unit.get("c") is not None:
            return (acting_unit.get("r"), acting_unit.get("c"))
        return None

    def _estimate_enemy_threat(self, destination, acting_unit, unit_map, board_map):
        if destination is None:
            return 0
        dr, dc = destination
        threat = 0
        owner = acting_unit.get("owner")
        for enemy in unit_map.values():
            if enemy.get("owner") == owner or enemy.get("hp", 0) <= 0:
                continue
            er, ec = enemy.get("r"), enemy.get("c")
            if er is None or ec is None:
                continue
            distance = max(abs(dr - er), abs(dc - ec))
            attack_range = 2 if enemy.get("type") == "Scout" else 1
            if distance <= attack_range and self._line_clear((er, ec), (dr, dc), board_map, owner):
                threat += 1
        return threat

    def _line_clear(self, start, end, board_map, acting_owner):
        sr, sc = start
        er, ec = end
        dr = er - sr
        dc = ec - sc
        steps = max(abs(dr), abs(dc))
        if steps <= 1:
            return True
        step_r = 0 if dr == 0 else int(dr / abs(dr))
        step_c = 0 if dc == 0 else int(dc / abs(dc))
        r, c = sr, sc
        for _ in range(1, steps):
            r += step_r
            c += step_c
            occupant = board_map.get((r, c))
            if occupant and occupant.get("owner") != acting_owner:
                return False
        return True

    def _estimate_mobility_after_move(self, move, acting_unit, unit_map, board_map):
        destination = self._infer_destination(move, acting_unit)
        if destination is None:
            return 0
        r, c = destination
        mobility = 0
        board_size = self._max_board_index(game_state=None, unit_map=unit_map, board_map=board_map) + 1
        for nr in range(max(0, r - 2), min(board_size - 1, r + 2) + 1):
            for nc in range(max(0, c - 2), min(board_size - 1, c + 2) + 1):
                if (nr, nc) == (r, c):
                    continue
                if (nr, nc) not in board_map:
                    mobility += 1
        return mobility

    def _score_tactical_objectives(self, move, acting_unit, unit_map, board_map):
        score = 0
        move_type = move.get("type")
        owner = acting_unit.get("owner")

        if move_type == "attack":
            target = move.get("target") or move.get("params", {}).get("target") or {}
            if isinstance(target, dict):
                tr, tc = target.get("r"), target.get("c")
                victim = board_map.get((tr, tc)) if tr is not None and tc is not None else None
                if victim and victim.get("type") == "Warden":
                    # Remove tanky front-line units to open center access.
                    score += 18

        # Encourage moves that support allies near the core.
        destination = self._infer_destination(move, acting_unit)
        if destination is not None:
            dr, dc = destination
            for ally in unit_map.values():
                if ally.get("owner") != owner or ally.get("id") == acting_unit.get("id"):
                    continue
                ar, ac = ally.get("r"), ally.get("c")
                if ar is None or ac is None:
                    continue
                if max(abs(dr - ar), abs(dc - ac)) <= 1 and self._is_core_cell(ar, ac, self._max_board_index(game_state=None, unit_map=unit_map, board_map=board_map) + 1):
                    score += 8
        return score

    def _is_core_cell(self, r, c, board_size=9):
        if r is None or c is None:
            return False
        center = board_size // 2
        return (center - 1) <= r <= (center + 1) and (center - 1) <= c <= (center + 1)

    def _board_size(self, game_state):
        board = game_state.get('board', []) if isinstance(game_state, dict) else []
        if isinstance(board, list) and board:
            return len(board)
        return 9

    def _max_board_index(self, game_state=None, unit_map=None, board_map=None):
        if game_state is not None:
            return self._board_size(game_state) - 1
        coords = []
        if isinstance(unit_map, dict):
            coords.extend([(u.get('r'), u.get('c')) for u in unit_map.values()])
        if isinstance(board_map, dict):
            coords.extend(list(board_map.keys()))
        max_coord = 8
        for r, c in coords:
            if isinstance(r, int):
                max_coord = max(max_coord, r)
            if isinstance(c, int):
                max_coord = max(max_coord, c)
        return max_coord

    def _adapt_style_for_board(self, board_size):
        board_bias = max(0, board_size - 9)
        self._style_weights['core_control'] = min(1.6, 1.0 + board_bias * 0.05)
        self._style_weights['aggression'] = max(0.85, 1.05 - board_bias * 0.02)

    def _select_opening_pattern_move(self, valid_moves, game_state):
        if not isinstance(valid_moves, list) or not valid_moves:
            return None

        round_idx = game_state.get('round', 0)
        if round_idx > 1:
            return None

        board_size = self._board_size(game_state)
        center = board_size // 2
        preferred_targets = [
            (center - 1, center),
            (center, center),
            (center + 1, center),
        ]

        for target_r, target_c in preferred_targets:
            for move in valid_moves:
                if move.get('type') != 'move':
                    continue
                target = move.get('target') or move.get('params', {}).get('target') or {}
                if isinstance(target, dict) and target.get('r') == target_r and target.get('c') == target_c:
                    return move
        return None

    def _extract_units(self, game_state):
        """Return a normalized flat list of unit dicts from varying payload shapes."""
        units = []

        payload_units = game_state.get('units', [])
        if isinstance(payload_units, list):
            for unit in payload_units:
                if isinstance(unit, dict):
                    units.append(unit)

        board = game_state.get('board', [])
        if isinstance(board, list):
            for row in board:
                if not isinstance(row, list):
                    continue
                for cell in row:
                    if not isinstance(cell, dict):
                        continue
                    unit = cell.get('unit')
                    if isinstance(unit, dict):
                        units.append(unit)

        return units

    def _get_strategy_context(self):
        """Query the knowledge agent via whichever retrieval API is available."""
        query_text = "R-Games game strategy and piece value"

        # Prefer explicit query API if present.
        if hasattr(self.knowledge_agent, 'query'):
            response = self.knowledge_agent.query(query_text)
            if isinstance(response, dict) and isinstance(response.get('results'), list):
                return " ".join(
                    r.get('content', '')
                    for r in response['results']
                    if isinstance(r, dict)
                ).strip()

        # Fallback to contextual search API.
        if hasattr(self.knowledge_agent, 'contextual_search'):
            results = self.knowledge_agent.contextual_search(query_text)
            return self._stringify_knowledge_results(results)

        # Fallback to basic retrieve API.
        if hasattr(self.knowledge_agent, 'retrieve'):
            results = self.knowledge_agent.retrieve(query_text)
            return self._stringify_knowledge_results(results)

        return ""

    def _stringify_knowledge_results(self, results):
        if not isinstance(results, list):
            return ""

        chunks = []
        for item in results:
            if isinstance(item, tuple) and len(item) >= 2:
                candidate = item[1]
                if isinstance(candidate, dict):
                    text = candidate.get('content') or candidate.get('text')
                    if isinstance(text, str):
                        chunks.append(text)
                elif isinstance(candidate, str):
                    chunks.append(candidate)
            elif isinstance(item, dict):
                text = item.get('content') or item.get('text')
                if isinstance(text, str):
                    chunks.append(text)
            elif isinstance(item, str):
                chunks.append(item)

        return " ".join(chunks).strip()

    def learn_from_game(self, result):
        try:
            logger.info(f"Learning from game result: {result.get('outcome')}")
            outcome = (result.get('outcome') or '').lower()
            board_size = int(result.get('board_size', 9)) if result.get('board_size') else 9

            # Reward is tied to final match score (AI controls player 2).
            reward = result.get('reward')
            if reward is None:
                final_score = result.get('final_score')
                try:
                    reward = -float(final_score) / 100.0 if final_score is not None else None
                except (TypeError, ValueError):
                    reward = None

            if reward is not None:
                try:
                    reward = max(-1.0, min(1.0, float(reward)))
                except (TypeError, ValueError):
                    reward = 0.0

            # Monte-Carlo RL update: propagate terminal reward through the full game trajectory.
            if self._episode_trace:
                discounted_return = float(reward)
                for state_key, action_key in reversed(self._episode_trace):
                    old_q = self._q_value(state_key, action_key)
                    updated_q = old_q + self._rl_alpha * (discounted_return - old_q)
                    self._set_q_value(state_key, action_key, updated_q)
                    discounted_return *= self._rl_gamma

            self._log_performance(result, reward)
            self._episode_trace = []

            self._style_weights["aggression"] = max(0.7, min(1.6, self._style_weights["aggression"] + (0.08 * reward)))
            self._style_weights["core_control"] = max(0.7, min(1.6, self._style_weights["core_control"] + (0.05 * reward)))
            self._style_weights["safety"] = max(0.7, min(1.7, self._style_weights["safety"] - (0.06 * reward)))

            if outcome == 'win':
                self._style_weights["aggression"] = min(1.6, self._style_weights["aggression"] + 0.02)
            elif outcome == 'loss':
                self._style_weights["safety"] = min(1.7, self._style_weights["safety"] + 0.03)

            self._result_window.append(outcome)
            self._result_window = self._result_window[-10:]

            losses = len([item for item in self._result_window if item == 'loss'])
            if losses >= 3:
                self._style_weights['safety'] = min(1.6, self._style_weights['safety'] + 0.03)

            board_pref = self._board_preferences.setdefault(board_size, {'wins': 0, 'losses': 0})
            if outcome == 'win':
                board_pref['wins'] += 1
            elif outcome == 'loss':
                board_pref['losses'] += 1

            self.shared_memory.set('ai_style_weights', dict(self._style_weights))
            self.shared_memory.set('ai_board_preferences', dict(self._board_preferences))

            # Integrate with learning agent using real reward-derived supervision.
            if self.learning_agent:
                task_embedding = self._build_learning_embedding(result)
                best_strategy = 'planning' if reward >= 0.2 else 'rl' if reward <= -0.2 else 'dqn'
                self.learning_agent.observe(
                    task_embedding=task_embedding,
                    best_agent_strategy_name=best_strategy
                )
                self.learning_agent.train_from_embeddings()

            self._save_learning_state()
            return True
        except Exception as e:
            logger.error(f"Error in learn_from_game: {e}", exc_info=True)
            return False

    def _prepare_episode_tracking(self, game_state):
        game_id = game_state.get('gameId') if isinstance(game_state, dict) else None
        round_idx = game_state.get('round', 0) if isinstance(game_state, dict) else 0
        if game_id and game_id != self._current_game_id:
            self._episode_trace = []
            self._current_game_id = game_id
        elif round_idx in (0, 1) and len(self._episode_trace) > 24:
            self._episode_trace = []

    def _record_episode_step(self, game_state, move):
        unit_map, _ = self._build_unit_and_board_maps(game_state)
        acting_unit = unit_map.get(move.get('unitId')) if isinstance(move, dict) else None
        self._episode_trace.append((self._encode_state(game_state), self._encode_action(move, acting_unit, game_state)))

    def _encode_state(self, game_state):
        board_size = self._board_size(game_state)
        round_bucket = int(game_state.get('round', 0) // 2)
        players = game_state.get('players', []) if isinstance(game_state, dict) else []
        ai_score = next((p.get('score', 0) for p in players if isinstance(p, dict) and p.get('id') == 1), 0)
        human_score = next((p.get('score', 0) for p in players if isinstance(p, dict) and p.get('id') == 0), 0)
        score_bucket = int((ai_score - human_score) // 2)
        return f"bs{board_size}|r{round_bucket}|sd{score_bucket}"

    def _encode_action(self, move, acting_unit, game_state=None):
        move_type = move.get('type', 'unknown') if isinstance(move, dict) else 'unknown'
        unit_type = acting_unit.get('type', 'unknown') if isinstance(acting_unit, dict) else 'unknown'
        target = (move.get('target') or move.get('params', {}).get('target') or {}) if isinstance(move, dict) else {}
        board_size = self._board_size(game_state) if isinstance(game_state, dict) else 9
        zone = 'core' if isinstance(target, dict) and self._is_core_cell(target.get('r'), target.get('c'), board_size) else 'edge'
        return f"{move_type}:{unit_type}:{zone}"

    def _q_value(self, state_key, action_key):
        return float(self._rl_q_table.get(state_key, {}).get(action_key, 0.0))

    def _set_q_value(self, state_key, action_key, value):
        self._rl_q_table.setdefault(state_key, {})[action_key] = float(value)

    def _build_learning_embedding(self, result):
        outcome = (result.get('outcome') or '').lower()
        final_score = float(result.get('final_score', 0) or 0)
        board_size = float(result.get('board_size', 9) or 9)
        outcome_value = 1.0 if outcome == 'win' else -1.0 if outcome == 'loss' else 0.0
        embedding = np.zeros(256, dtype=np.float32)
        embedding[:6] = [
            final_score / 100.0,
            float(result.get('reward', 0) or 0),
            outcome_value,
            board_size / 16.0,
            self._style_weights['aggression'],
            self._style_weights['safety'],
        ]
        return embedding

    def _apply_learning_strategy(self, game_state):
        if not self.learning_agent:
            return
        try:
            state_key = self._encode_state(game_state)
            hash_value = abs(hash(state_key)) % 1000 / 1000.0
            embedding = np.zeros(256, dtype=np.float32)
            embedding[:4] = [hash_value, self._style_weights['aggression'], self._style_weights['core_control'], self._style_weights['safety']]
            if torch is not None:
                strategy_input = torch.tensor(embedding, dtype=torch.float32)
            else:
                strategy_input = embedding
            selected = self.learning_agent.select_agent_strategy(strategy_input)
            if selected == 'planning':
                self._style_weights['core_control'] = min(1.7, self._style_weights['core_control'] + 0.02)
            elif selected == 'rl':
                self._style_weights['aggression'] = min(1.7, self._style_weights['aggression'] + 0.02)
            elif selected == 'dqn':
                self._style_weights['safety'] = min(1.8, self._style_weights['safety'] + 0.02)
        except Exception as strategy_error:
            logger.warning(f"Learning strategy selection skipped: {strategy_error}")

    def _log_performance(self, result, reward):
        entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'outcome': (result.get('outcome') or '').lower(),
            'final_score': result.get('final_score'),
            'reward': float(reward),
            'board_size': result.get('board_size', 9),
            'round': result.get('round'),
            'trajectory_length': len(self._episode_trace),
        }
        self._performance_log.append(entry)
        self._performance_log = self._performance_log[-500:]
        self.shared_memory.set('ai_recent_performance', list(self._performance_log[-30:]))

    def _load_learning_state(self):
        if not self._learning_store_path.exists():
            return
        try:
            payload = json.loads(self._learning_store_path.read_text())
            self._style_weights.update(payload.get('style_weights', {}))
            self._board_preferences = payload.get('board_preferences', {}) or {}
            self._rl_q_table = payload.get('rl_q_table', {}) or {}
            self._performance_log = payload.get('performance_log', []) or []
        except Exception as load_error:
            logger.warning(f"Failed to load learning state: {load_error}")

    def _save_learning_state(self):
        try:
            self._learning_store_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'style_weights': self._style_weights,
                'board_preferences': self._board_preferences,
                'rl_q_table': self._rl_q_table,
                'performance_log': self._performance_log[-500:],
            }
            self._learning_store_path.write_text(json.dumps(payload, indent=2))
        except Exception as save_error:
            logger.warning(f"Failed to persist learning state: {save_error}")

# Initialize AI Player instance
ai_player = None

def initialize_ai():
    global ai_player
    if ai_player is None:
        try:
            ai_player = AIPlayer()
        except Exception as e:
            logger.critical(f"Failed to initialize AI Player on startup: {e}")
    return ai_player

class AIRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # API Routes
        if path in ('/health', '/api/ai/health'):
            self._set_headers()
            status = "ready" if ai_player else "initializing"
            self.wfile.write(json.dumps({"status": "healthy", "agent_status": status}).encode('utf-8'))
            return

        # Static File Serving
        if path == '/':
            path = '/index.html'
        
        # Remove leading slash to get relative path
        relative_path = path.lstrip('/')
        file_path = project_root / relative_path
        
        if file_path.exists() and file_path.is_file():
            # Security check: ensure file is within project root
            try:
                file_path.resolve().relative_to(project_root.resolve())
            except ValueError:
                self._set_headers(403)
                self.wfile.write(json.dumps({"error": "Forbidden"}).encode('utf-8'))
                return

            import mimetypes
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                mime_type = 'application/octet-stream'
            
            self.send_response(200)
            self.send_header('Content-type', mime_type)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode('utf-8'))

    def do_POST(self):
        parsed_path = urlparse(self.path)
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode('utf-8'))
        except json.JSONDecodeError:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
            return

        if parsed_path.path in ('/move', '/api/ai/move'):
            if not ai_player:
                self._set_headers(503)
                self.wfile.write(json.dumps({"error": "AI Player not initialized"}).encode('utf-8'))
                return

            move = ai_player.get_move(data)
            if move:
                self._set_headers()
                if isinstance(move, dict) and 'choice' in move:
                    self.wfile.write(json.dumps({"choice": move['choice']}).encode('utf-8'))
                else:
                    self.wfile.write(json.dumps({"move": move}).encode('utf-8'))
            else:
                self._set_headers(404)
                self.wfile.write(json.dumps({"error": "No valid move found"}).encode('utf-8'))

        elif parsed_path.path in ('/learn', '/api/ai/learn'):
            if not ai_player:
                self._set_headers(503)
                self.wfile.write(json.dumps({"error": "AI Player not initialized"}).encode('utf-8'))
                return

            success = ai_player.learn_from_game(data)
            if success:
                self._set_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Learning updated"}).encode('utf-8'))
            else:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": "Learning update failed"}).encode('utf-8'))
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=AIRequestHandler):
    initialize_ai()
    # Use PYTHON_PORT environment variable if available, otherwise default to 5000
    port = int(os.environ.get('PYTHON_PORT', 5000))
    server_address = ('0.0.0.0', port)
    httpd = server_class(server_address, handler_class)
    logger.info(f"Starting AI Server on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logger.info("AI Server stopped.")

if __name__ == "__main__":
    run()
