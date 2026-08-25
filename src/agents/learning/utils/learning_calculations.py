"""Centralized learning-related calculations for the learning subsystem.

This module provides deterministic, reusable mathematical utilities used by
agents, adaptation layers, policy/value training loops, and diagnostics.
It intentionally focuses on *learning* calculations rather than generic
scientific math.
"""

from __future__ import annotations

import math
import numpy as np # type: ignore
import torch # type: ignore

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any, Deque, Dict, List, Optional, Tuple

from .config_loader import get_config_section, load_global_config
from .learning_error import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Learning Calculations")
printer = PrettyPrinter()

EPSILON = 1e-12

@dataclass(frozen=True)
class DistributionDrift:
    """Structured drift result for two sample distributions."""

    kl_divergence: float
    js_divergence: float
    wasserstein_like_distance: float


class LearningCalculations:
    """Production-ready collection of learning-centric calculations."""

    def __init__(self) -> None:
        self.config = load_global_config()
        self.calc_config = get_config_section("learning_calculations")

        self.trend_window =int(self.calc_config.get("trend_window", 50))
        self.epsilon =float(self.calc_config.get("epsilon", 1e-12))
        self.k = int(self.calc_config.get("k", 5))

        self.max_trend_window = max(2, int(self.trend_window))
        self.performance_history: Deque[float] = deque(maxlen=max(100, self.trend_window * 4))

    # ------------------------------------------------------------------
    # Classification / prediction metrics
    # ------------------------------------------------------------------
    def calculate_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        if logits.ndim < 2:
            raise InvalidConfigError("logits must be at least 2D: [batch, classes]")
        if labels.ndim != 1:
            labels = labels.view(-1)
        preds = torch.argmax(logits, dim=-1).view(-1)
        if preds.numel() != labels.numel():
            raise InvalidConfigError("predictions and labels must have the same number of elements")
        return float((preds == labels).float().mean().item())

    def top_k_accuracy(self, logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
        if k <= 0:
            raise InvalidConfigError("k must be > 0")
        if logits.ndim != 2:
            raise InvalidConfigError("logits must be 2D: [batch, classes]")
        if labels.ndim != 1:
            labels = labels.view(-1)
        if logits.size(0) != labels.size(0):
            raise InvalidConfigError("batch dimension mismatch between logits and labels")

        k = min(k, logits.size(1))
        topk = torch.topk(logits, k=k, dim=1).indices
        correct = (topk == labels.unsqueeze(1)).any(dim=1)
        return float(correct.float().mean().item())

    # ------------------------------------------------------------------
    # Distribution and drift metrics
    # ------------------------------------------------------------------
    def calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """KL(P || Q) for discrete probability vectors."""
        p_arr = self._normalize_probability_vector(p)
        q_arr = self._normalize_probability_vector(q)
        return float(np.sum(p_arr * np.log((p_arr + self.epsilon) / (q_arr + self.epsilon))))

    def calculate_js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        p_arr = self._normalize_probability_vector(p)
        q_arr = self._normalize_probability_vector(q)
        m = 0.5 * (p_arr + q_arr)
        kl_pm = np.sum(p_arr * np.log((p_arr + self.epsilon) / (m + self.epsilon)))
        kl_qm = np.sum(q_arr * np.log((q_arr + self.epsilon) / (m + self.epsilon)))
        return float(0.5 * (kl_pm + kl_qm))

    def calculate_distribution_drift(self, p: np.ndarray, q: np.ndarray) -> DistributionDrift:
        p_arr = self._normalize_probability_vector(p)
        q_arr = self._normalize_probability_vector(q)
        cdf_p = np.cumsum(p_arr)
        cdf_q = np.cumsum(q_arr)
        wasserstein_like = float(np.mean(np.abs(cdf_p - cdf_q)))
        return DistributionDrift(
            kl_divergence=self.calculate_kl_divergence(p_arr, q_arr),
            js_divergence=self.calculate_js_divergence(p_arr, q_arr),
            wasserstein_like_distance=wasserstein_like,
        )

    # ------------------------------------------------------------------
    # Reinforcement-learning-centric calculations
    # ------------------------------------------------------------------
    def discounted_returns(self, rewards: Sequence[float], gamma: float = 0.99) -> List[float]:
        gamma = self._clamp(gamma, 0.0, 1.0)
        out = [0.0] * len(rewards)
        running = 0.0
        for idx in range(len(rewards) - 1, -1, -1):
            running = float(rewards[idx]) + gamma * running
            out[idx] = running
        return out

    def n_step_return(self, rewards: Sequence[float], gamma: float = 0.99, n: int = 3) -> float:
        if n <= 0:
            raise InvalidConfigError("n must be > 0")
        gamma = self._clamp(gamma, 0.0, 1.0)
        total = 0.0
        for i, reward in enumerate(rewards[:n]):
            total += (gamma ** i) * float(reward)
        return total

    def generalized_advantage_estimate(
        self,
        rewards: Sequence[float],
        values: Sequence[float],
        dones: Sequence[bool],
        *,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> List[float]:
        if not (len(rewards) == len(values) == len(dones)):
            raise InvalidConfigError("rewards, values, and dones must have the same length")
        gamma = self._clamp(gamma, 0.0, 1.0)
        lam = self._clamp(lam, 0.0, 1.0)

        gae = 0.0
        next_value = 0.0
        advantages = [0.0] * len(rewards)
        for idx in range(len(rewards) - 1, -1, -1):
            mask = 0.0 if dones[idx] else 1.0
            delta = float(rewards[idx]) + gamma * next_value * mask - float(values[idx])
            gae = delta + gamma * lam * mask * gae
            advantages[idx] = gae
            next_value = float(values[idx])
        return advantages

    # ------------------------------------------------------------------
    # Stability and gradient calculations
    # ------------------------------------------------------------------
    def gradient_global_norm(self, gradients: Iterable[torch.Tensor]) -> float:
        total_sq = 0.0
        for grad in gradients:
            if grad is None:
                continue
            total_sq += float(torch.sum(grad.detach().float() ** 2).item())
        return math.sqrt(total_sq)

    def clip_gradients_by_global_norm(self, gradients: Iterable[torch.Tensor], max_norm: float) -> float:
        if max_norm <= 0:
            raise InvalidConfigError("max_norm must be > 0")
        norm = self.gradient_global_norm(gradients)
        if norm <= max_norm or norm <= self.epsilon:
            return norm
        scale = max_norm / (norm + self.epsilon)
        for grad in gradients:
            if grad is not None:
                grad.mul_(scale)
        return norm

    # ------------------------------------------------------------------
    # Performance trend & aggregation
    # ------------------------------------------------------------------
    def update_performance(self, value: float) -> None:
        self.performance_history.append(float(value))

    def calculate_performance_trend(self, window: Optional[int] = None) -> float:
        if not self.performance_history:
            return 0.0
        window_size = max(2, int(window or self.trend_window))
        if len(self.performance_history) < window_size:
            return 0.0

        values = list(self.performance_history)
        recent = values[-window_size:]
        baseline = values[:window_size]
        recent_mean = float(np.mean(recent))
        baseline_mean = float(np.mean(baseline))
        return (recent_mean - baseline_mean) / (abs(baseline_mean) + self.epsilon)

    def summarize_rewards(self, rewards: Sequence[float]) -> Dict[str, float]:
        if not rewards:
            return {
                "count": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "sum": 0.0,
                "std": 0.0,
            }
        values = np.array([float(v) for v in rewards], dtype=np.float64)
        return {
            "count": float(values.size),
            "mean": float(np.mean(values)),
            "median": float(median(values.tolist())),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "sum": float(np.sum(values)),
            "std": float(np.std(values)),
        }

    def moving_average(self, values: Sequence[float], window_size: int = 10) -> List[float]:
        if window_size <= 0:
            raise InvalidConfigError("window_size must be > 0")
        buf: Deque[float] = deque(maxlen=window_size)
        out: List[float] = []
        for value in values:
            buf.append(float(value))
            out.append(sum(buf) / len(buf))
        return out

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _normalize_probability_vector(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise InvalidConfigError("probability vector cannot be empty")
        arr = np.clip(arr, 0.0, None)
        total = float(np.sum(arr))
        if total <= self.epsilon:
            return np.full_like(arr, 1.0 / arr.size)
        return arr / total

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        if minimum > maximum:
            raise InvalidConfigError("minimum cannot exceed maximum")
        return max(minimum, min(maximum, float(value)))


# Optional compatibility free-functions for existing imports.
def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return LearningCalculations().calculate_accuracy(logits, labels)


def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return LearningCalculations().calculate_kl_divergence(p, q)