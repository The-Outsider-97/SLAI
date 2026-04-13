import sys
import os
import json
import random
from datetime import datetime
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Add AI folder to sys.path so 'src' and 'logs' can be imported as top-level packages
ai_root = project_root / "AI"
sys.path.insert(0, str(ai_root))

from src.agents.agent_factory import AgentFactory
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.collaborative_agent import CollaborativeAgent
from src.agents.planning.planning_types import Task, TaskType
from logs.logger import get_logger

logger = get_logger("Puluc AI")

class AIPlayer:
    def __init__(self):
        logger.info("Initializing AI Player...")
        self.shared_memory = SharedMemory()
        self.factory = AgentFactory()
        self.collab = CollaborativeAgent(shared_memory=self.shared_memory, agent_factory=self.factory)

        # Create agents
        #
        # ===========
        #
        # ===========
        #

        self._register_task_routes()
        try:
            self._planning_task_registered = False
            self._planning_enabled = True
        
            logger.info("AI Player initialized with ... agents.")
        except Exception as e:
            logger.error(f"Failed to initialize AI Player: {e}", exc_info=True)
            self._planning_enabled = False
            self._planning_task_registered = True