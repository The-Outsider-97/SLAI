import logging
import time
import sys
import os
import torch
from datetime import datetime

logger = logging.getLogger('SafeAI.SecurityManager')
logger.setLevel(logging.INFO)

class SecurityManager:
    def __init__(self, shared_memory=None, policy_config=None):
        """
        policy_config: dictionary of agent permissions:
            {
                "agent_name": {
                    "can_access_data": True,
                    "can_modify_model": False,
                    "can_export": False
                }
            }
        """
        self.shared_memory = shared_memory
        self.policy_config = policy_config or {}
        self.violation_log = []

    def is_action_allowed(self, agent_name, action):
        """
        Checks whether the agent is allowed to perform the action.
        """
        permissions = self.policy_config.get(agent_name, {})
        allowed = permissions.get(action, False)

        if not allowed:
            self._log_violation(agent_name, action)

        return allowed

    def _log_violation(self, agent_name, action):
        timestamp = datetime.utcnow().isoformat()
        log_entry = {
            "time": timestamp,
            "agent": agent_name,
            "action": action,
            "status": "DENIED"
        }

        self.violation_log.append(log_entry)
        logger.warning(f"[SECURITY] Blocked {agent_name} from performing '{action}'")

        if self.shared_memory:
            self.shared_memory.set(f"security_violation_{agent_name}", log_entry)

    def get_violations(self):
        return self.violation_log

    def print_report(self):
        print("\n=== Security Audit Report ===")
        for entry in self.violation_log:
            print(f"[{entry['time']}] {entry['agent']} → {entry['action']} → ❌ DENIED")

    def update_policy(self, agent_name, action, allow=True):
        if agent_name not in self.policy_config:
            self.policy_config[agent_name] = {}
        self.policy_config[agent_name][action] = allow
        logger.info(f"[SECURITY] Updated {agent_name}: {action} set to {allow}")
