import json
import os
import sys
import uuid
import torch
import logging

from rnd_loop.evaluator import Evaluator
from collaborative.shared_memory import SharedMemory
from modules.monitoring import Monitoring
from modules.deployment.model_registry import register_model

logger = logging.getLogger("SLAI.ExperimentManager")
logger.setLevel(logging.INFO)

class ExperimentManager:
    def __init__(self, shared_memory=None):
        self.shared_memory = shared_memory or SharedMemory()
        self.evaluator = Evaluator(shared_memory=self.shared_memory)
        self.results = []

    def run_experiments(self, agent_configs: list[dict], task_data: dict):
        """
        agent_configs: List of dicts like:
            {
                "agent_class": <AgentClass>,
                "init_args": {...},
                "name": "maml_v1"
            }
        task_data: Dict passed to agent.execute()
        """
        logger.info(f"Running {len(agent_configs)} experiments...")

        for config in agent_configs:
            agent_class = config["agent_class"]
            agent = agent_class(**config.get("init_args", {}))

            name = config.get("name", agent.__class__.__name__)
            logger.info(f"Evaluating agent: {name}")

            result = self.evaluator.evaluate_agent(
                agent=agent,
                task_data=task_data,
                metadata={"agent_name": name}
            )
            self.results.append(result)

        logger.info(f"All {len(self.results)} experiments completed.")
        return self.results

    def summarize_results(self, sort_key="risk_score", minimize=True):
        """
        Sorts and returns top agent based on evaluation metric (e.g., risk_score).
        """
        valid_results = [
            r for r in self.results
            if "result" in r and isinstance(r["result"], dict) and sort_key in r["result"]
        ]

        sorted_results = sorted(
            valid_results,
            key=lambda r: r["result"][sort_key],
            reverse=not minimize
        )

        logger.info(f"Top agent by '{sort_key}': {sorted_results[0]['agent']}")
        
        # Auto-register top model
        top = sorted_results[0]
        model_name = top["metadata"].get("agent_name", top["agent"])
        metrics = top["result"]
        register_model(model_name, path="models/best_model_placeholder.pkl", metadata=metrics)

        return sorted_results
