"""Repro: does LearningAgent._run_episode(train=True) ever update the strategy's learner?
Runs the REAL _run_episode from src/agents/learning_agent.py. Only third-party packages that are not installed
(torch, gymnasium, ...) are stubbed so the module can be imported; the environment, the learner and the collaborators
are fakes that record calls.  No claim is made about DQN maths here - only about the call path."""
import sys, importlib.abc, importlib.machinery, os
from unittest import mock
import numpy as np

STUB = {'torch', 'gymnasium', 'gym', 'psutil', 'sklearn', 'scipy', 'pandas', 'cv2', 'matplotlib', 'transformers'}
class _F(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split('.')[0] in STUB:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = mock.MagicMock(name=spec.name); m.__path__ = []; m.__spec__ = spec; m.__name__ = spec.name; return m
    def exec_module(self, m): pass
sys.meta_path.append(_F())
sys.path.insert(0, os.environ.get('SLAI_ROOT', '.'))
from src.agents.learning_agent import LearningAgent

class Env:
    def __init__(self): self.t = 0
    def reset(self, seed=None): self.t = 0; return np.zeros(4, np.float32), {}
    def step(self, a):
        self.t += 1
        return np.full(4, self.t, np.float32), 1.0, self.t >= 5, {}

class Learner:
    def __init__(self): self.stored = 0; self.trained = 0
    def select_action(self, s, explore=True): return 0
    def store_transition(self, *t): self.stored += 1
    def train(self): self.trained += 1; return 0.5

def build():
    a = LearningAgent.__new__(LearningAgent)
    a.env = Env()
    a.state_processor = mock.Mock(); 
    class P:  # minimal tensor-like
        def __init__(s, v): s.v = v
        def numel(s): return len(s.v)
        def detach(s): return s
        def cpu(s): return s
        def numpy(s): return np.asarray(s.v)
    a.state_processor.process = lambda st: P(np.asarray(st, np.float32).reshape(-1))
    a.multi_task_learner = mock.MagicMock(); a.strategy_selector = mock.MagicMock()
    a.strategy_selector.train_from_embeddings.return_value = None
    a.learning_factory = mock.MagicMock(); a.learning_calculations = mock.MagicMock()
    a.learner_update_count = 0; a.learner_update_skips = 0  # counters added by the fix (absent on the pre-fix code; ignored there)
    a.observation_count = 0; a.performance_history = []; a.performance_metrics = {"strategy_loss": {}, "strategy_accuracy": {}}
    a._global_reward_stats = mock.MagicMock(); a._reward_stats = {}; a.error_history = []
    return a

a = build(); lr = Learner()
rep = a._run_episode("dqn", lr, max_steps=50, seed=1, train=True)
print("steps:", rep["steps"], "| learner.store_transition calls:", lr.stored, "| learner.train calls:", lr.trained,
      "| loss_samples:", rep["loss_samples"], "| avg_loss reported:", rep["avg_loss"])
ok = lr.trained > 0 and rep["loss_samples"] > 0
print("RESULT:", "learner updated during the train episode (fixed)" if ok else "DEFECT: train episode never updated the learner (avg_loss reported as 0.0)")
# Observed on the unfixed code (HEAD a854f17d, 2026-10-27): "steps: 5 | store_transition calls: 0 | train calls: 0 | loss_samples: 0 | avg_loss 0.0"
sys.exit(0 if ok else 1)
