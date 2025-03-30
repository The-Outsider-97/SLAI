import json
import os, sys
import uuid
import torch
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime
from collaborative.shared_memory import SharedMemory
from logs.logger import get_logger
from traceback import format_exc

logger = logging.getLogger("SLAI.Evaluator")
logger.setLevel(logging.INFO)

class Evaluator:
    def __init__(self):
        self.shared_memory = SharedMemory()
        self.logger = get_logger("Evaluator")

    def evaluate_agent(self, agent, task_data, metadata=None):
        try:
            result = agent.execute(task_data)
            self.shared_memory.set("agent_eval", result)
            self.logger.info(f"Logged agent evaluation result: {result}")

            metric_name = metadata.get("metric") if metadata else None
            current_score = result.get("result", {}).get(metric_name) if metric_name else (
                result.get("result", {}).get("reward") or result.get("result", {}).get("accuracy")
            )

            if current_score is not None:
                history = self.shared_memory.get("eval_history") or []
                history.append({"timestamp": time.time(), "score": current_score})
                self.shared_memory.set("eval_history", history)
                self._generate_score_plot(history)

            if metadata and metadata.get("min_score") is not None:
                if current_score is not None and current_score < metadata["min_score"]:
                    self.logger.warning(f"[Evaluator] Score {current_score} fell below min threshold {metadata['min_score']}.")
                    return {
                        "status": "failed_minimum",
                        "score": current_score
                    }

            if metadata and "previous_score" in metadata:
                if current_score is not None and current_score < metadata["previous_score"]:
                    from recursive_improvement.rewriter import Rewriter
                    agent_file = metadata.get("agent_path")
                    if agent_file:
                        rewriter = Rewriter(agent_path=agent_file)
                        rewriter.rollback_model()

            if metadata and "previous_score" in metadata:
                history = self.shared_memory.get("eval_history") or []
                recent = history[-3:]
                if len(recent) == 3 and all(h["score"] < metadata["previous_score"] for h in recent):
                    try:
                        from agents.rsi_agent import RSI_Agent
                        RSI_Agent().trigger_on(agent_path=metadata.get("agent_path"))
                        self.logger.warning("[Evaluator] RSI agent triggered due to 3 consecutive poor evaluations.")
                    except Exception as rsi_error:
                        self.logger.error(f"Failed to trigger RSI Agent: {rsi_error}")

            if metadata and metadata.get("notify_collab"):
                try:
                    from collaboration_manager import CollaborationManager
                    CollaborationManager().notify_evaluation(
                        agent_id=agent.__class__.__name__, result=result
                    )
                except Exception as notify_err:
                    self.logger.warning(f"[Evaluator] Failed to notify collaboration manager: {notify_err}")

            return {
                "status": "success",
                "result": result
            }

        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}\n{format_exc()}")
            return {
                "status": "error",
                "message": str(e),
                "traceback": format_exc()
            }

    def _generate_score_plot(self, history):
        timestamps = [entry["timestamp"] for entry in history]
        scores = [entry["score"] for entry in history]
        if len(scores) < 2:
            return
        plt.figure(figsize=(8, 4))
        plt.plot(timestamps, scores, marker='o')
        plt.title("Evaluation Score Trend")
        plt.xlabel("Timestamp")
        plt.ylabel("Score")
        plt.grid(True)
        output_dir = os.path.join("logs", "visuals")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "score_trend.png"))
        plt.close()


"""
Usage Example:

from evaluator import Evaluator

metadata = {
    "previous_score": 0.92,
    "min_score": 0.5,
    "metric": "accuracy",
    "agent_path": "agents/dqn_agent.py",
    "notify_collab": True
}

result = evaluator.evaluate_agent(agent, task_data, metadata)
"""
