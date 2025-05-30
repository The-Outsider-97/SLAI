import os, sys
import yaml
import json
import numpy as np
import random

from collections import deque, defaultdict
from datetime import datetime, timedelta

from logs.logger import get_logger

logger = get_logger("Replay Buffer")

CONFIG_PATH = "src/utils/buffer/configs/buffer_config.yaml"

class ReplayBuffer:
    """Experience replay buffer with uniform sampling"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        """Store transition (state, action, reward, next_state, done)"""
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        """Random batch of transitions"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

