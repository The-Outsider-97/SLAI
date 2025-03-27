import os, sys
import yaml
import json
import logging
import numpy as np
import random
from collections import deque
from threading import Lock

logger = logging.getLogger("DistributedReplayBuffer")

class DistributedReplayBuffer:
    def __init__(self, capacity=100_000, seed=None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = Lock()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        logger.info(f"Initialized distributed replay buffer with capacity {capacity}")

    def push(self, agent_id, state, action, reward, next_state, done):
        """
        Add a transition with source agent_id (for traceability)
        """
        with self.lock:
            self.buffer.append((agent_id, state, action, reward, next_state, done))
            if len(self.buffer) == self.capacity:
                logger.debug("DistributedReplayBuffer is full. Oldest experiences are being discarded.")

    def sample(self, batch_size):
        with self.lock:
            if batch_size > len(self.buffer):
                raise ValueError("Not enough samples in buffer.")
            batch = random.sample(self.buffer, batch_size)

        agent_ids, states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return agent_ids, states, actions, rewards, next_states, dones

    def save(self, filepath):
        with self.lock:
            if not self.buffer:
                logger.warning("Tried to save an empty buffer.")
                return
            np.savez_compressed(filepath, buffer=np.array(self.buffer, dtype=object))
            logger.info(f"Replay buffer saved to {filepath}")

    def load(self, filepath):
        if not os.path.exists(filepath):
            logger.error(f"Replay buffer file not found: {filepath}")
            return
        data = np.load(filepath, allow_pickle=True)
        loaded = data['buffer']
        with self.lock:
            self.buffer.clear()
            self.buffer.extend(loaded.tolist())
        logger.info(f"Replay buffer loaded from {filepath} with {len(self.buffer)} items")

    def clear(self):
        with self.lock:
            self.buffer.clear()
            logger.info("Replay buffer cleared.")

    def __len__(self):
        return len(self.buffer)

    def get_all(self):
        with self.lock:
            if not self.buffer:
                return [], [], [], [], [], []
            agent_ids, states, actions, rewards, next_states, dones = map(np.array, zip(*self.buffer))
        return agent_ids, states, actions, rewards, next_states, dones
