"""Shared helpers for the learning tests: generic toy environments and a tiny 'adapter' loop.

Nothing here is specific to any application: the environments are textbook problems (contextual bandit, corridor walk) and the
loop is the smallest possible adapter that drives ``ExternalLearningService`` through its public protocol.
"""
from __future__ import annotations

import importlib
import sys
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.agents.learning.contracts import Observation
from src.agents.learning.external_learning import ExternalLearningConfig, ExternalLearningService
from src.agents.learning.softmax_policy import PolicyHyperparameters


def checkpoint_stack() -> Tuple[bool, str]:
    """(available, reason).  The checkpoint manager imports SLAI's logger, which needs a recent colorama."""
    try:
        importlib.import_module("checkpointing.checkpoint_manager")
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


class ContextualBandit:
    """K contexts (one-hot + bias feature), A actions; the best action for context c is (2c+1) % A.  Reward 1 if best else 0."""

    def __init__(self, contexts: int = 4, actions: int = 4, steps: int = 6) -> None:
        self.K, self.A, self.steps = contexts, actions, steps
        self.feature_dim = contexts + 1

    def best(self, c: int) -> int:
        return (2 * c + 1) % self.A

    def obs(self, c: int) -> Observation:
        f = np.zeros(self.feature_dim); f[c] = 1.0; f[-1] = 1.0
        return Observation.make(f, feature_dim=self.feature_dim, action_dim=self.A)


class Corridor:
    """Positions 0..n-1, start random in 1..n-2, actions {0: left, 1: right}; +1 only on reaching the right end (delayed reward)."""

    def __init__(self, n: int = 6, max_steps: int = 20) -> None:
        self.n, self.max_steps, self.A = n, max_steps, 2
        self.feature_dim = n

    def obs(self, pos: int) -> Observation:
        f = np.zeros(self.n); f[pos] = 1.0
        return Observation.make(f, feature_dim=self.n, action_dim=2)


def make_service(env, *, name: str = "toy", seed: int = 0, lr: float = 0.5, mode: str = "train", **kw) -> ExternalLearningService:
    hyper = kw.pop("hyper", None) or PolicyHyperparameters(learning_rate=lr, baseline_momentum=0.9)
    cfg = ExternalLearningConfig(name=name, feature_dim=env.feature_dim, action_dim=env.A, hyper=hyper, seed=seed, mode=mode, **kw)
    return ExternalLearningService(cfg)


def run_bandit_episode(svc: ExternalLearningService, env: ContextualBandit, ep_id: str, rng: np.random.Generator, *,
                       delayed: bool = False, mode: Optional[str] = None):
    """One adapter-driven episode.  Returns (correct, steps, update_result)."""
    svc.start_episode(ep_id, mode=mode)
    correct = 0
    for t in range(env.steps):
        c = int(rng.integers(env.K))
        d = svc.decide(ep_id, f"{ep_id}/{t}", env.obs(c), sequence=t)
        ok = d.action.index == env.best(c)
        correct += ok
        if not delayed:
            svc.feedback(ep_id, f"{ep_id}/{t}", 1.0 if ok else 0.0)
    terminal = 0.0
    if delayed:
        terminal = 1.0 if correct >= env.steps * 0.7 else -1.0
    res = svc.end_episode(ep_id, terminal_reward=terminal)
    return correct, env.steps, res


def eval_bandit(svc: ExternalLearningService, env: ContextualBandit, seeds: List[int], tag: str) -> float:
    """Greedy accuracy on fixed seeds in evaluate mode (must not change the policy)."""
    hits = total = 0
    for s in seeds:
        rng = np.random.default_rng(s)
        c, n, _ = run_bandit_episode(svc, env, f"{tag}-{s}", rng, mode="evaluate")
        hits += c; total += n
    return hits / total


# ---------------------------------------------------------------------------------------------------------------------
# importing LearningAgent where optional heavy dependencies (torch, gymnasium, ...) are not installed
# ---------------------------------------------------------------------------------------------------------------------
_OPTIONAL = ("torch", "gymnasium", "gym", "psutil", "sklearn", "scipy", "pandas", "cv2", "matplotlib", "transformers")


def import_learning_agent():
    """Import ``src.agents.learning_agent``.  Only third-party packages that are genuinely missing are replaced by inert
    stand-ins, which is enough for tests that drive ``LearningAgent`` methods with fake environments/learners.  Where the real
    packages exist (the project's own environment) nothing is stubbed.  Returns (module | None, reason)."""
    import importlib.abc, importlib.machinery, importlib.util
    from unittest import mock

    missing = {n for n in _OPTIONAL if importlib.util.find_spec(n) is None}
    if missing:
        class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in missing:
                    return importlib.machinery.ModuleSpec(name, self, is_package=True)

            def create_module(self, spec):
                m = mock.MagicMock(name=spec.name); m.__path__ = []; m.__spec__ = spec; m.__name__ = spec.name
                return m

            def exec_module(self, module):
                pass

        if not any(type(f).__name__ == "_Stub" for f in sys.meta_path):
            sys.meta_path.append(_Stub())
    try:
        import colorama  # SLAI's logger calls colorama.just_fix_windows_console (colorama >= 0.4.6)
        if not hasattr(colorama, "just_fix_windows_console"):
            colorama.just_fix_windows_console = lambda: None
    except ImportError:
        pass
    try:
        return importlib.import_module("src.agents.learning_agent"), ("stubbed: " + ",".join(sorted(missing))) if missing else "real dependencies"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
