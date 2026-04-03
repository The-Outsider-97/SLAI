
import math
import time
import numpy as np
import torch

from collections import defaultdict, deque

class LearningCalculations:
    def __init__(self):
        self.learning_calculations = []


    def _calculate_accuracy(self, logits, labels):
        """Calculate prediction accuracy"""
        preds = torch.argmax(logits, dim=1)
        return (preds == labels).float().mean().item()

    def _calculate_kl_divergence(self, p, q):
        """Compute KL(P||Q) between two multivariate Gaussian distributions"""
        # Epsilon to prevent singular matrices
        p += np.random.normal(0, self.epsilon, p.shape)
        q += np.random.normal(0, self.epsilon, q.shape)
        
        # Calculate means and covariance matrices
        mu_p = np.mean(p, axis=0)
        sigma_p = np.cov(p, rowvar=False) + np.eye(p.shape[1])*self.epsilon
        mu_q = np.mean(q, axis=0)
        sigma_q = np.cov(q, rowvar=False) + np.eye(q.shape[1])*self.epsilon
        
        # KL divergence formula for multivariate Gaussians
        diff = mu_q - mu_p
        sigma_q_inv = np.linalg.inv(sigma_q)
        n = mu_p.shape[0]
        
        trace_term = np.trace(sigma_q_inv @ sigma_p)
        quad_form = diff.T @ sigma_q_inv @ diff
        logdet_term = np.log(np.linalg.det(sigma_q)/np.linalg.det(sigma_p))
        
        return 0.5 * (trace_term + quad_form - n + logdet_term)

    def _calculate_performance_trend(self):
        """Calculate performance trend over last 100 episodes"""
        if len(self.performance_history) < 10:
            return 0.0
        
        recent = np.mean(self.performance_history[-10:])
        baseline = np.mean(self.performance_history[:10])
        return (recent - baseline) / (abs(baseline) + 1e-6)
    