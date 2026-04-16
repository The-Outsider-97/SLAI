"""Agent-factory-driven Chronos training script.

This script trains the Chronos AI runtime (`games/ai_chronos.py`) by running
adversarial self-play over supported board sizes (9x9, 11x11, 13x13).

Design goals:
- Use only the relevant SLAI agents exposed through `AgentFactory` indirectly
  via `ChronosAI` (knowledge/planning/execution/learning/adaptive).
- Enumerate broad move and counter-move spaces each turn.
- Emphasize purposeful, win-focused play through minimax counter-analysis and
  reward shaping based on victory, core control, and material preservation.

Usage:
    python -m games.train_chronos_agents --episodes 45 --depth 2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Ensure repository imports resolve when run as script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .ai_chronos import ChronosAI

PIECE_VALUE = {"Strategos": 6.0, "Warden": 3.0, "Scout": 1.5}
NEIGHBOR_DELTAS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


@dataclass
class UnitState:
    id: str
    owner: int
    type: str
    r: int
    c: int
    hp: int = 1


@dataclass
class MiniChronosState:
    board_size: int
    turn: int
    round_index: int
    units: list[UnitState]
    scores: list[int]
    winner: int | None = None


class ChronosTrainingSimulator:
    """Deterministic-enough self-play simulator for Chronos-oriented training."""

    def __init__(self, board_size: int, seed: int = 0):
        self.board_size = board_size
        self.random = random.Random(seed)
        self.max_rounds = board_size * 3

    def initialize(self) -> MiniChronosState:
        units: list[UnitState] = []
        center = self.board_size // 2
        for owner, row in ((0, 0), (1, self.board_size - 1)):
            for c in range(self.board_size):
                if c == center:
                    piece = "Strategos"
                elif c in {center - 1, center + 1}:
                    piece = "Warden"
                else:
                    piece = "Scout"
                units.append(UnitState(id=f"P{owner}_{piece}_{c}", owner=owner, type=piece, r=row, c=c, hp=1))
        return MiniChronosState(
            board_size=self.board_size,
            turn=1,
            round_index=0,
            units=units,
            scores=[0, 0],
            winner=None,
        )

    def rollout(self, ai: ChronosAI, search_depth: int, playouts: int) -> dict[str, Any]:
        state = self.initialize()
        ai_move_signatures: list[dict[str, Any]] = []
        explored_states = 0
        counter_evals = 0

        while state.winner is None and state.round_index < self.max_rounds:
            legal_moves = self._legal_moves(state, owner=state.turn)
            if not legal_moves:
                state.winner = 1 - state.turn
                break

            if state.turn == 1:
                # Build game_state payload expected by ChronosAI.
                game_state = self._build_game_state_payload(state, legal_moves)
                chosen_move = ai.get_move(game_state)
                if chosen_move not in legal_moves:
                    chosen_move = self._choose_best_counter_aware_move(state, legal_moves, owner=1, depth=search_depth)
                ai_move_signatures.append(self._compact_move(chosen_move))
            else:
                chosen_move = self._choose_best_counter_aware_move(state, legal_moves, owner=0, depth=search_depth)

            state = self._apply_move(state, chosen_move)
            explored_states += len(legal_moves)

            # Counter-analysis playouts for robustness across possible strategies.
            counter_evals += self._run_counter_playouts(state, playouts=playouts, depth=search_depth)

            if state.turn == 1:
                state.round_index += 1

            self._update_scoring(state)
            state.winner = self._winner_if_any(state)

        if state.winner is None:
            state.winner = self._winner_by_score_tiebreak(state)

        reward = self._reward_signal(state)
        return {
            "winner": state.winner,
            "reward": reward,
            "final_score": state.scores[1] - state.scores[0],
            "board_size": self.board_size,
            "rounds": state.round_index,
            "ai_moves": ai_move_signatures,
            "explored_states": explored_states,
            "counter_evals": counter_evals,
        }

    def _build_game_state_payload(self, state: MiniChronosState, valid_moves: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "phase": "planning",
            "round": state.round_index,
            "board": {"size": state.board_size},
            "players": [
                {"id": 0, "score": state.scores[0]},
                {"id": 1, "score": state.scores[1]},
            ],
            "units": [
                {
                    "id": u.id,
                    "owner": u.owner,
                    "type": u.type,
                    "r": u.r,
                    "c": u.c,
                    "hp": u.hp,
                }
                for u in state.units
                if u.hp > 0
            ],
            "validMoves": valid_moves,
        }

    def _legal_moves(self, state: MiniChronosState, owner: int) -> list[dict[str, Any]]:
        board = {(u.r, u.c): u for u in state.units if u.hp > 0}
        moves: list[dict[str, Any]] = []

        for unit in state.units:
            if unit.owner != owner or unit.hp <= 0:
                continue

            for dr, dc in NEIGHBOR_DELTAS:
                nr, nc = unit.r + dr, unit.c + dc
                if not (0 <= nr < state.board_size and 0 <= nc < state.board_size):
                    continue

                target = board.get((nr, nc))
                if target is None:
                    moves.append({
                        "type": "move",
                        "unitId": unit.id,
                        "target": {"r": nr, "c": nc},
                    })
                    if self._is_core_cell(nr, nc, state.board_size):
                        moves.append({
                            "type": "claim",
                            "unitId": unit.id,
                            "target": {"r": nr, "c": nc},
                        })
                elif target.owner != owner:
                    moves.append({
                        "type": "attack",
                        "unitId": unit.id,
                        "target": {
                            "id": target.id,
                            "owner": target.owner,
                            "type": target.type,
                            "r": target.r,
                            "c": target.c,
                        },
                    })

        moves.append({"type": "pass", "unitId": "none", "target": {"r": -1, "c": -1}})
        # Deduplicate by JSON signature to keep search efficient.
        dedup: dict[str, dict[str, Any]] = {}
        for mv in moves:
            dedup[json.dumps(mv, sort_keys=True)] = mv
        return list(dedup.values())

    def _apply_move(self, state: MiniChronosState, move: dict[str, Any]) -> MiniChronosState:
        units = [UnitState(**u.__dict__) for u in state.units]
        unit_by_id = {u.id: u for u in units}
        actor = unit_by_id.get(str(move.get("unitId", "")))
        mtype = str(move.get("type", "pass")).lower()

        if actor and actor.hp > 0:
            if mtype in {"move", "claim"}:
                tr = int(move.get("target", {}).get("r", actor.r))
                tc = int(move.get("target", {}).get("c", actor.c))
                occupant = next((u for u in units if u.hp > 0 and u.r == tr and u.c == tc and u.id != actor.id), None)
                if occupant is None:
                    actor.r, actor.c = tr, tc
            elif mtype == "attack":
                target_id = move.get("target", {}).get("id")
                target = unit_by_id.get(str(target_id)) if target_id else None
                if target and target.owner != actor.owner and target.hp > 0:
                    target.hp -= 1

        next_state = MiniChronosState(
            board_size=state.board_size,
            turn=1 - state.turn,
            round_index=state.round_index,
            units=units,
            scores=list(state.scores),
            winner=state.winner,
        )
        return next_state

    def _run_counter_playouts(self, state: MiniChronosState, playouts: int, depth: int) -> int:
        """Explore plausible opponent counters to widen strategic coverage."""
        evals = 0
        probe_owner = state.turn
        legal = self._legal_moves(state, owner=probe_owner)
        if not legal:
            return 0

        sample_n = min(playouts, len(legal))
        sampled = self.random.sample(legal, k=sample_n)
        for mv in sampled:
            _ = self._minimax(self._apply_move(state, mv), root_owner=probe_owner, to_play=1 - probe_owner, depth=depth)
            evals += 1
        return evals

    def _choose_best_counter_aware_move(self, state: MiniChronosState, legal_moves: list[dict[str, Any]], owner: int, depth: int) -> dict[str, Any]:
        best_value = -math.inf
        best_moves: list[dict[str, Any]] = []
        for mv in legal_moves:
            next_state = self._apply_move(state, mv)
            value = self._minimax(next_state, root_owner=owner, to_play=1 - owner, depth=depth)
            if value > best_value:
                best_value = value
                best_moves = [mv]
            elif abs(value - best_value) < 1e-9:
                best_moves.append(mv)
        return self.random.choice(best_moves) if best_moves else self.random.choice(legal_moves)

    def _minimax(self, state: MiniChronosState, root_owner: int, to_play: int, depth: int) -> float:
        winner = self._winner_if_any(state)
        if winner is not None:
            return 10_000.0 if winner == root_owner else -10_000.0
        if depth <= 0:
            return self._evaluate_state(state, root_owner)

        moves = self._legal_moves(state, owner=to_play)
        if not moves:
            return 10_000.0 if to_play != root_owner else -10_000.0

        if to_play == root_owner:
            return max(
                self._minimax(self._apply_move(state, mv), root_owner, 1 - to_play, depth - 1)
                for mv in moves
            )
        return min(
            self._minimax(self._apply_move(state, mv), root_owner, 1 - to_play, depth - 1)
            for mv in moves
        )

    def _evaluate_state(self, state: MiniChronosState, owner: int) -> float:
        own_units = [u for u in state.units if u.owner == owner and u.hp > 0]
        opp_units = [u for u in state.units if u.owner != owner and u.hp > 0]
        own_material = sum(PIECE_VALUE.get(u.type, 1.0) for u in own_units)
        opp_material = sum(PIECE_VALUE.get(u.type, 1.0) for u in opp_units)
        own_core = sum(1.0 for u in own_units if self._is_core_cell(u.r, u.c, state.board_size))
        opp_core = sum(1.0 for u in opp_units if self._is_core_cell(u.r, u.c, state.board_size))
        score_delta = float(state.scores[owner] - state.scores[1 - owner])
        return (own_material - opp_material) * 5.0 + (own_core - opp_core) * 8.0 + score_delta * 2.0

    def _winner_if_any(self, state: MiniChronosState) -> int | None:
        alive_strategos = {0: False, 1: False}
        for unit in state.units:
            if unit.type == "Strategos" and unit.hp > 0:
                alive_strategos[unit.owner] = True

        if alive_strategos[0] and alive_strategos[1]:
            target_score = max(18, state.board_size + 9)
            if state.scores[0] >= target_score and state.scores[0] > state.scores[1]:
                return 0
            if state.scores[1] >= target_score and state.scores[1] > state.scores[0]:
                return 1
            return None
        if alive_strategos[1] and not alive_strategos[0]:
            return 1
        if alive_strategos[0] and not alive_strategos[1]:
            return 0
        return self._winner_by_score_tiebreak(state)

    def _winner_by_score_tiebreak(self, state: MiniChronosState) -> int:
        if state.scores[1] > state.scores[0]:
            return 1
        if state.scores[0] > state.scores[1]:
            return 0
        return 1

    def _reward_signal(self, state: MiniChronosState) -> float:
        winner_bonus = 120.0 if state.winner == 1 else -120.0
        score_term = (state.scores[1] - state.scores[0]) * 3.0
        own_material = sum(PIECE_VALUE.get(u.type, 1.0) for u in state.units if u.owner == 1 and u.hp > 0)
        opp_material = sum(PIECE_VALUE.get(u.type, 1.0) for u in state.units if u.owner == 0 and u.hp > 0)
        return winner_bonus + score_term + (own_material - opp_material) * 4.0

    def _update_scoring(self, state: MiniChronosState) -> None:
        for owner in (0, 1):
            core_control = 0.0
            for u in state.units:
                if u.owner != owner or u.hp <= 0 or not self._is_core_cell(u.r, u.c, state.board_size):
                    continue
                core_control += PIECE_VALUE.get(u.type, 1.0)
            state.scores[owner] += int(core_control)

    @staticmethod
    def _is_core_cell(r: int, c: int, board_size: int) -> bool:
        center = board_size // 2
        return abs(r - center) <= 1 and abs(c - center) <= 1

    @staticmethod
    def _compact_move(move: dict[str, Any]) -> dict[str, Any]:
        tgt = move.get("target", {}) if isinstance(move, dict) else {}
        return {
            "type": move.get("type"),
            "unitId": move.get("unitId"),
            "target": {
                "r": tgt.get("r"),
                "c": tgt.get("c"),
                "id": tgt.get("id"),
                "type": tgt.get("type"),
            },
        }


def train_chronos_model(
    *,
    episodes: int,
    depth: int,
    playouts: int,
    seed: int,
    log_every: int = 1,
    checkpoint_path: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    board_sizes = [9, 11, 13]
    summary: dict[str, Any] = {
        "episodes": episodes,
        "depth": depth,
        "counter_playouts": playouts,
        "board_sizes": board_sizes,
        "wins": {size: 0 for size in board_sizes},
        "losses": {size: 0 for size in board_sizes},
        "draw_like": {size: 0 for size in board_sizes},
        "explored_states": 0,
        "counter_evals": 0,
    }

    start_episode = 0
    if checkpoint_path and resume and checkpoint_path.exists():
        checkpoint = _load_checkpoint(checkpoint_path)
        meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
        if (
            isinstance(meta, dict)
            and int(meta.get("episodes", -1)) == int(episodes)
            and int(meta.get("depth", -1)) == int(depth)
            and int(meta.get("playouts", -1)) == int(playouts)
            and int(meta.get("seed", -1)) == int(seed)
        ):
            saved_summary = checkpoint.get("summary")
            if isinstance(saved_summary, dict):
                summary = _merge_summary(summary, saved_summary, board_sizes)
            start_episode = max(0, min(episodes, int(checkpoint.get("completed_episodes", 0) or 0)))
            print(
                f"[Chronos Training] Resuming from checkpoint: episode {start_episode + 1}/{episodes}",
                flush=True,
            )
        else:
            print(
                "[Chronos Training] Checkpoint ignored (configuration mismatch with current run).",
                flush=True,
            )

    rng = random.Random(seed)
    for _ in range(start_episode):
        rng.randint(1, 10_000_000)

    ai = ChronosAI()

    if start_episode >= episodes:
        summary["health"] = ai.health()
        return summary

    for episode in range(start_episode, episodes):
        episode_number = episode + 1
        board_size = board_sizes[episode % len(board_sizes)]
        sim_seed = rng.randint(1, 10_000_000)
        simulator = ChronosTrainingSimulator(board_size=board_size, seed=sim_seed)
        result = simulator.rollout(ai=ai, search_depth=depth, playouts=playouts)

        outcome = "win" if result["winner"] == 1 else "loss"
        if result["winner"] == 1:
            summary["wins"][board_size] += 1
        elif result["winner"] == 0:
            summary["losses"][board_size] += 1
        else:
            summary["draw_like"][board_size] += 1
            outcome = "draw"

        payload = {
            "outcome": outcome,
            "reward": float(result["reward"]),
            "final_score": int(result["final_score"]),
            "board_size": int(board_size),
            "rounds": int(result["rounds"]),
            "ai_move_signatures": result["ai_moves"],
            "explored_states": int(result["explored_states"]),
            "counter_evaluations": int(result["counter_evals"]),
            "training_mode": "agent_factory_counterfactual_selfplay",
        }
        ai.learn_from_game(payload)

        summary["explored_states"] += int(result["explored_states"])
        summary["counter_evals"] += int(result["counter_evals"])
        summary["completed_episodes"] = episode_number

        should_log = log_every > 0 and (
            episode_number == 1
            or episode_number == episodes
            or (episode_number % log_every == 0)
        )
        if should_log:
            print(
                (
                    f"[Chronos Training] Episode {episode_number}/{episodes} "
                    f"| board={board_size}x{board_size} "
                    f"| outcome={outcome} "
                    f"| reward={float(result['reward']):.2f} "
                    f"| explored_states={int(result['explored_states'])} "
                    f"| counter_evals={int(result['counter_evals'])}"
                ),
                flush=True,
            )
        if checkpoint_path:
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                summary=summary,
                completed_episodes=episode_number,
                meta={
                    "episodes": episodes,
                    "depth": depth,
                    "playouts": playouts,
                    "seed": seed,
                },
            )

    summary["health"] = ai.health()
    if checkpoint_path:
        _save_checkpoint(
            checkpoint_path=checkpoint_path,
            summary=summary,
            completed_episodes=episodes,
            meta={
                "episodes": episodes,
                "depth": depth,
                "playouts": playouts,
                "seed": seed,
            },
        )
    return summary


def _merge_summary(base: dict[str, Any], saved: dict[str, Any], board_sizes: list[int]) -> dict[str, Any]:
    merged = dict(base)
    merged["wins"] = {size: int(saved.get("wins", {}).get(str(size), saved.get("wins", {}).get(size, 0))) for size in board_sizes}
    merged["losses"] = {size: int(saved.get("losses", {}).get(str(size), saved.get("losses", {}).get(size, 0))) for size in board_sizes}
    merged["draw_like"] = {
        size: int(saved.get("draw_like", {}).get(str(size), saved.get("draw_like", {}).get(size, 0)))
        for size in board_sizes
    }
    merged["explored_states"] = int(saved.get("explored_states", merged["explored_states"]) or 0)
    merged["counter_evals"] = int(saved.get("counter_evals", merged["counter_evals"]) or 0)
    merged["completed_episodes"] = int(saved.get("completed_episodes", 0) or 0)
    return merged


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    try:
        return json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_checkpoint(
    *,
    checkpoint_path: Path,
    summary: dict[str, Any],
    completed_episodes: int,
    meta: dict[str, int],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_episodes": int(completed_episodes),
        "meta": meta,
        "summary": summary,
    }
    tmp_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(checkpoint_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Chronos AI via SLAI agent-factory orchestration.")
    parser.add_argument("--episodes", type=int, default=30, help="Total self-play episodes (distributed over 9x9/11x11/13x13).")
    parser.add_argument("--depth", type=int, default=2, help="Minimax counter-search depth for strategy/counter analysis.")
    parser.add_argument("--playouts", type=int, default=8, help="Counterfactual playout probes per turn.")
    parser.add_argument("--seed", type=int, default=7, help="Global random seed for reproducibility.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print training progress every N episodes (also logs first and last episode).",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=ROOT / "logs" / "chronos_training_summary.json",
        help="Where to write JSON summary output.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=ROOT / "logs" / "chronos_training_checkpoint.json",
        help="Checkpoint file used to resume interrupted training.",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Ignore existing checkpoint and start from episode 1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = train_chronos_model(
        episodes=max(1, args.episodes),
        depth=max(1, args.depth),
        playouts=max(1, args.playouts),
        seed=args.seed,
        log_every=max(1, args.log_every),
        checkpoint_path=args.checkpoint_path,
        resume=not args.fresh_start,
    )

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "episodes": summary["episodes"],
            "depth": summary["depth"],
            "counter_playouts": summary["counter_playouts"],
            "wins": summary["wins"],
            "losses": summary["losses"],
            "draw_like": summary["draw_like"],
            "explored_states": summary["explored_states"],
            "counter_evals": summary["counter_evals"],
            "summary_path": str(args.summary_path),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())