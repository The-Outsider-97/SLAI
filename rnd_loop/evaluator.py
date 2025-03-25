import json
import os
import sys
import uuid
import torch
import logging
from datetime import datetime

logger = logging.getLogger("SLAI.Evaluator")
logger.setLevel(logging.INFO)

class Evaluator:
    def __init__(self, shared_memory=None, log_path="logs/eval_runs.jsonl"):
        self.shared_memory = shared_memory
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def evaluate_agent(self, agent, task_data, metadata=None):
        """
        Evaluate any agent that implements `.execute(task_data)`.

        Returns a structured result dictionary.
        """
        run_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        try:
            result = agent.execute(task_data)
            logger.info(f"[Evaluator] Agent '{agent.__class__.__name__}' returned: {result}")
        except Exception as e:
            result = {
                "status": "failed",
                "error": str(e)
            }
            logger.error(f"[Evaluator] Agent '{agent.__class__.__name__}' failed: {e}")

        result_entry = {
            "run_id": run_id,
            "timestamp": timestamp,
            "agent": agent.__class__.__name__,
            "task_data": task_data,
            "result": result,
            "metadata": metadata or {}
        }

        self._log_run(result_entry)

        if self.shared_memory:
            self.shared_memory.set(f"eval_result_{run_id}", result_entry)

        return result_entry

    def _log_run(self, entry):
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(f"[Evaluator] Logged evaluation to {self.log_path}")
        except Exception as e:
            logger.error(f"Failed to write eval log: {e}")
