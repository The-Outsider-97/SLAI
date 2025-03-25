import json
import os
import sys
import uuid
import torch
import logging
import itertools

from rnd_loop.evaluator import Evaluator
from modules.deployment.model_registry import register_model
from collaborative.shared_memory import SharedMemory
from modules.monitoring import Monitoring

logger = logging.getLogger("SLAI.HyperparamTuner")
logger.setLevel(logging.INFO)

class HyperparamTuner:
    def __init__(self, agent_class, search_space: dict, base_task: dict,
                 shared_memory=None, max_trials=10):
        self.agent_class = agent_class
        self.search_space = search_space
        self.base_task = base_task
        self.shared_memory = shared_memory or SharedMemory()
        self.monitoring = Monitoring(shared_memory=self.shared_memory)
        self.evaluator = Evaluator(shared_memory=self.shared_memory, monitoring=self.monitoring)
        self.max_trials = max_trials
        self.trials = []

    def run_grid_search(self):
        """
        Runs a simple grid search over the defined search space.
        """
        param_grid = list(itertools.product(*self.search_space.values()))
        param_names = list(self.search_space.keys())
        logger.info(f"Running grid search with {len(param_grid)} configurations...")

        best_result = None
        best_score = float('inf')  # Assume lower is better (e.g. risk_score)

        for i, values in enumerate(param_grid[:self.max_trials]):
            config = dict(zip(param_names, values))
            logger.info(f"[Trial {i+1}] Testing config: {config}")

            agent = self.agent_class(shared_memory=self.shared_memory, **config)
            result = self.evaluator.evaluate_agent(agent, task_data=self.base_task, metadata={"tuner_trial": i})

            metrics = result.get("result", {})
            score = metrics.get("risk_score", None)

            self.trials.append({
                "trial": i,
                "config": config,
                "metrics": metrics
            })

            if score is not None and score < best_score:
                best_score = score
                best_result = {
                    "trial": i,
                    "score": score,
                    "config": config,
                    "metrics": metrics
                }

        if best_result:
            logger.info(f"Best config found: {best_result['config']} with risk_score={best_result['score']}")
            self._register_best(best_result)
        else:
            logger.warning("No successful trials found.")

        return best_result

    def _register_best(self, best_result):
        """
        Push best config to model registry for downstream use.
        """
        register_model(
            model_name=f"{self.agent_class.__name__}_tuned",
            path="models/tuned_agent_config.json",  # Or link to actual .pkl after training
            metadata={
                "config": best_result["config"],
                "metrics": best_result["metrics"]
            }
        )

        if self.shared_memory:
            self.shared_memory.set("tuned_config", best_result["config"])
