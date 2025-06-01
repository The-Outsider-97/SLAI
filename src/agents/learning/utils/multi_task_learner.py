

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

from src.agents.learning.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Multi Task Learner")
printer = PrettyPrinter

class MultiTaskLearner(nn.Module):
    """
    Coordinates learning across multiple tasks. Tracks loss, adjusts weights, and manages task priorities.
    Designed to be imported and used by LearningAgent.
    """

    def __init__(self, task_ids):
        super().__init__()
        """
        Args:
            task_ids (List[str]): Unique identifiers for tasks.
            initial_weights (dict or None): Optional dict mapping task_id -> float weight.
            rebalance_strategy (str): Strategy to adjust task weights ("uniform", "softmax", or "none").
            rebalance_temp (float): Temperature for softmax weighting (controls sharpness).
        """
        self.config = load_global_config()
        self.learner_config = get_config_section('multi_task_learner')
        self.task_ids = task_ids
        self.loss_history = defaultdict(list)
        self.task_weights = {tid: 1.0 for tid in task_ids}

        self.initial_weights = self.learner_config.get('initial_weights', True)
        self.rebalance_strategy = self.learner_config.get('rebalance_strategy', 'softmax')
        self.rebalance_temp = self.learner_config.get('rebalance_temp', 1.0)

        logger.info(f"MultiTaskLearner initialized with {len(task_ids)} tasks.")

    def update_loss(self, task_id, loss_value):
        """
        Add latest loss to history.
        """
        if task_id not in self.task_ids:
            logger.warning(f"Unknown task_id: {task_id}. Ignoring.")
            return
        self.loss_history[task_id].append(loss_value)

    def get_weighted_loss(self):
        """
        Computes the weighted loss across tasks.
        Returns:
            float: weighted loss value (scalar)
        """
        total_loss = 0.0
        for tid in self.task_ids:
            if not self.loss_history[tid]:
                continue
            latest_loss = self.loss_history[tid][-1]
            total_loss += self.task_weights[tid] * latest_loss
        return total_loss

    def rebalance(self):
        """
        Adjusts task weights based on recent performance.
        Strategies:
            - 'uniform': Equal weights
            - 'softmax': Weight harder tasks higher (e.g., high recent loss)
            - 'none': Keep weights as-is
        """
        if self.rebalance_strategy == "none":
            return

        if self.rebalance_strategy == "uniform":
            w = 1.0 / len(self.task_ids)
            for tid in self.task_ids:
                self.task_weights[tid] = w
            return

        # Softmax over average recent losses
        avg_losses = []
        for tid in self.task_ids:
            recent_losses = self.loss_history[tid][-5:]  # Last 5 steps
            avg_loss = np.mean(recent_losses) if recent_losses else 1.0
            avg_losses.append(avg_loss)

        # Convert to torch tensor for softmax
        loss_tensor = torch.tensor(avg_losses)
        inv_loss = -loss_tensor / self.rebalance_temp
        weights = torch.softmax(inv_loss, dim=0)

        for tid, w in zip(self.task_ids, weights):
            self.task_weights[tid] = float(w)

        logger.info(f"Rebalanced task weights: {self.task_weights}")

    def reset(self):
        """
        Clears history for all tasks.
        """
        self.loss_history = defaultdict(list)
        logger.info("Loss histories reset for all tasks.")

    def get_weights(self):
        return self.task_weights

    def get_recent_losses(self, window=5):
        """
        Returns recent average losses per task.
        """
        return {
            tid: np.mean(self.loss_history[tid][-window:]) if self.loss_history[tid] else 0.0
            for tid in self.task_ids
        }
