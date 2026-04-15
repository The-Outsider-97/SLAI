from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from .buffer_validation import Transition, TransitionValidator
from .utils.config_loader import get_config_section, load_global_config


@dataclass
class NStepConfig:
    """Configuration contract for n-step return transformation."""

    n_step: int = 3
    gamma: float = 0.99
    clear_on_terminal: bool = True

    @classmethod
    def from_config(cls, user_config: Optional[Dict[str, Any]] = None) -> "NStepConfig":
        load_global_config()
        config = dict(get_config_section("nstep") or {})
        if user_config:
            config.update(user_config.get("nstep", {}) if isinstance(user_config, dict) else {})

        n_step = int(config.get("n_step", 3))
        gamma = float(config.get("gamma", 0.99))
        if n_step <= 0:
            raise ValueError("n_step must be > 0")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")

        return cls(
            n_step=n_step,
            gamma=gamma,
            clear_on_terminal=bool(config.get("clear_on_terminal", True)),
        )


class NStepBuffer:
    """Transforms 1-step transitions into n-step transitions.

    Input transition format:
    (agent_id, state, action, reward, next_state, done)

    Output transition format (same schema):
    (agent_id_t, state_t, action_t, discounted_reward_{t:t+n-1}, next_state_{t+n-1}, done_{t+n-1})
    """

    def __init__(
        self,
        user_config: Optional[Dict[str, Any]] = None,
        validator: Optional[TransitionValidator] = None,
    ):
        self.config = NStepConfig.from_config(user_config=user_config)
        self.n_step = self.config.n_step
        self.gamma = self.config.gamma
        self.clear_on_terminal = self.config.clear_on_terminal

        self.validator = validator or TransitionValidator()
        self._queue: Deque[Transition] = deque()
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)

    def _build_nstep_transition(self, window: Sequence[Transition]) -> Transition:
        first = window[0]
        agent_id, state, action = first[0], first[1], first[2]

        discounted_reward = 0.0
        terminal = False
        final_next_state = window[-1][4]

        for idx, transition in enumerate(window):
            reward = float(transition[3])
            discounted_reward += (self.gamma ** idx) * reward
            final_next_state = transition[4]
            terminal = bool(transition[5])
            if terminal:
                break

        return (agent_id, state, action, discounted_reward, final_next_state, terminal)

    def _ready(self) -> bool:
        if not self._queue:
            return False
        if len(self._queue) >= self.n_step:
            return True
        return bool(self._queue[-1][5])  # terminal early flush for shorter tail

    def add(self, transition: Sequence[Any]) -> Optional[Transition]:
        """Append one transition; returns one n-step transition when available."""
        with self._lock:
            normalized = self.validator.sanitize_transition(tuple(transition))
            self._queue.append(normalized)

            if not self._ready():
                return None

            window_size = min(self.n_step, len(self._queue))
            window = [self._queue[i] for i in range(window_size)]
            output = self._build_nstep_transition(window)

            # slide by one after emitting
            self._queue.popleft()

            # optional explicit cleanup if terminal appeared at queue head progression
            if output[5] and self.clear_on_terminal:
                self._queue.clear()

            return output

    def add_components(self, agent_id: Any, state: Any, action: Any, reward: Any,
                       next_state: Any, done: Any) -> Optional[Transition]:
        return self.add((agent_id, state, action, reward, next_state, done))

    def flush(self) -> List[Transition]:
        """Flush remaining queue into truncated n-step transitions."""
        with self._lock:
            outputs: List[Transition] = []
            while self._queue:
                window_size = min(self.n_step, len(self._queue))
                window = [self._queue[i] for i in range(window_size)]
                outputs.append(self._build_nstep_transition(window))
                self._queue.popleft()
            return outputs

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "pending": len(self._queue),
                "n_step": self.n_step,
                "gamma": self.gamma,
                "clear_on_terminal": self.clear_on_terminal,
            }


__all__ = ["NStepConfig", "NStepBuffer"]
