import json
import os
import sys
import json
import uuid
import torch
import random
import logging
import itertools

from rnd_loop.evaluator import Evaluator
from modules.deployment.model_registry import register_model
from collaborative.shared_memory import SharedMemory
from modules.monitoring import Monitoring
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

logger = logging.getLogger("SLAI.HyperparamTuner")
logger.setLevel(logging.INFO)

class HyperparamTuner:
    def __init__(self, agent_class, search_space: dict, base_task: dict,
                 shared_memory=None, max_trials=10, mode="grid", checkpoint_file="logs/hparam_trials.jsonl"):
        self.agent_class = agent_class
        self.search_space = search_space
        self.base_task = base_task
        self.shared_memory = shared_memory or SharedMemory()
        self.monitoring = Monitoring(shared_memory=self.shared_memory)
        self.evaluator = Evaluator(shared_memory=self.shared_memory, monitoring=self.monitoring)
        self.max_trials = max_trials
        self.trials = []
        self.mode = mode
        self.checkpoint_file = checkpoint_file
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    def run(self):
        if self.mode == "grid":
            return self._run_grid_search()
        elif self.mode == "bayesian":
            return self._run_bayesian_search()
        else:
            raise ValueError(f"Unsupported tuning mode: {self.mode}")

    def _run_grid_search(self):
        param_grid = list(itertools.product(*self.search_space.values()))
        param_names = list(self.search_space.keys())
        logger.info(f"Running grid search with {len(param_grid)} configurations...")

        best_result = None
        best_score = float('inf')

        for i, values in enumerate(param_grid[:self.max_trials]):
            config = dict(zip(param_names, values))
            result = self._evaluate_trial(i, config)
            if result and result["score"] < best_score:
                best_score = result["score"]
                best_result = result

        if best_result:
            self._register_best(best_result)

        return best_result

    def _run_bayesian_search(self):
        dimensions = []
        param_names = []
        for key, values in self.search_space.items():
            if isinstance(values[0], float):
                dimensions.append(Real(min(values), max(values), name=key))
            elif isinstance(values[0], int):
                dimensions.append(Integer(min(values), max(values), name=key))
            else:
                dimensions.append(Categorical(values, name=key))
            param_names.append(key)

        def objective(params):
            config = dict(zip(param_names, params))
            result = self._evaluate_trial(len(self.trials), config)
            return result["score"] if result else float('inf')

        gp_minimize(objective, dimensions, n_calls=self.max_trials, random_state=42)
        best_result = min(self.trials, key=lambda x: x["score"])
        self._register_best(best_result)
        return best_result

    def _evaluate_trial(self, trial_index, config):
        try:
            agent = self.agent_class(shared_memory=self.shared_memory, **config)
            result = self.evaluator.evaluate_agent(agent, self.base_task, metadata={"trial": trial_index})
            metrics = result.get("result", {})
            score = metrics.get("risk_score", float('inf'))

            trial_record = {
                "trial": trial_index,
                "config": config,
                "metrics": metrics,
                "score": score
            }
            self.trials.append(trial_record)
            self._checkpoint_trial(trial_record)
            return trial_record

        except Exception as e:
            logger.warning(f"Trial {trial_index} failed: {e}")
            return None

    def _checkpoint_trial(self, trial_record):
        with open(self.checkpoint_file, "a") as f:
            f.write(json.dumps(trial_record) + "\n")

    def _register_best(self, best_result):
        register_model(
            model_name=f"{self.agent_class.__name__}_tuned",
            path="models/tuned_agent_config.json",
            metadata={
                "config": best_result["config"],
                "metrics": best_result["metrics"]
            }
        )
        if self.shared_memory:
            self.shared_memory.set("tuned_config", best_result["config"])
