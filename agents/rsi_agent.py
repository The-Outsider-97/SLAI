import random
import logging
import os
import sys
import torch
import numpy as np
from collections import deque
from collaborative.shared_memory import SharedMemory

class RSI_Agent:
    def __init__(self, state_size, action_size, shared_memory: SharedMemory, config: dict = None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.shared_memory = shared_memory
        self.config = config or {}
        self.model_id = "RSI_Agent"
        
    def execute(self, task_data):
        """
        Execute the RSI task using given data. Required for collaboration system.
        """
        print("[RSI_Agent] Executing task:", task_data)

        # Run training with dynamic self-tuning
        self.train()

        # Collect metrics
        evaluation = self.evaluate()

        # Optionally write to shared memory
        self.shared_memory.set("rsi_agent_last_eval", evaluation)

        return evaluation

    def act(self, state):
        # Simulate RSI rule-based action
        rsi_value = self.calculate_rsi(state)
        if rsi_value > 70:
            return 0  # sell
        elif rsi_value < 30:
            return 1  # buy
        return 2  # hold

    def calculate_rsi(self, prices, period: int = 14):
        prices = np.array(prices)
        if len(prices) < period + 1:
            return 50  # Neutral if not enough data

        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def reset(self):
        self.memory.clear()

    def sync_with_shared_memory(self):
        if self.shared_memory:
            self.shared_memory.update(self.model_id, {
                "epsilon": self.epsilon,
                "learning_rate": self.learning_rate,
                "memory_size": len(self.memory)
            })

    def load(self, filepath):
        pass  # Placeholder for model loading

    def save(self, filepath):
        pass  # Placeholder for model saving
