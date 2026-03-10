import random
import time
import json
import hashlib
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any

from src.agents.collaborative.utils.config_loader import load_global_config, get_config_section
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Collaborative Error")
printer = PrettyPrinter

class OverloadError(Exception):
    def __init__(self,
                 error_type: ExecutionErrorType,
                 message: str,
                 severity: str = "medium",
                 context: Optional[Dict[str, Any]] = None,
                 collaborative_agent_state: Optional[Dict] = None,
                 remediation_guidance: Optional[str] = None):
        super().__init__(message)
        self.context = context or {}
        self.config = load_global_config()
        self.error_config = get_config_section('collaboration_error')
