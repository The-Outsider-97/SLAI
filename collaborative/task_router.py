import logging
import random
import torch
import sys
import os
from registry import AgentRegistry

logger = logging.getLogger("SLAI.TaskRouter")
logger.setLevel(logging.INFO)

class TaskRouter:
    def __init__(self, registry: AgentRegistry, shared_memory=None):
        self.registry = registry
        self.shared_memory = shared_memory or {}

    def get_agents_by_task(self, task_type):
        """
        Return all agents that support a given task_type.
        """
        return {
            name: agent for name, agent in self._agents.items()
            if task_type in self._capabilities.get(name, [])
        }
    
    def route(self, task_type, task_data):
        eligible_agents = self.registry.get_agents_by_task(task_type)

        if not eligible_agents:
            raise Exception(f"No agents found for task type '{task_type}'")

        # Step 1: Rank agents by success history or priority
        sorted_agents = self._rank_agents(eligible_agents)

        # Step 2: Try each agent in order until success
        for agent_name, agent in sorted_agents:
            try:
                logger.info(f"Routing task '{task_type}' to agent: {agent_name}")
                result = agent.execute(task_data)

                # Step 3: Log success to shared memory
                self._record_success(agent_name)
                return result

            except Exception as e:
                logger.exception(f"Agent '{agent_name}' failed during task '{task_type}' execution.")
                self._record_failure(agent_name)

        # If all fail
        raise Exception(f"All agents failed for task type '{task_type}'")

    def _rank_agents(self, agent_dict):
        ranked = []
        for name, agent in agent_dict.items():
            meta = self.shared_memory.get("agent_stats", {}).get(name, {})
            success = meta.get("successes", 0)
            failures = meta.get("failures", 0)
            total = success + failures
            rate = success / total if total > 0 else 0.5  # default neutral score
            priority = meta.get("priority", 0)
            score = rate + (priority * 0.1)
            ranked.append((name, agent, score))

        ranked.sort(key=lambda tup: tup[2], reverse=True)
        return [(n, a) for n, a, _ in ranked]

    def _record_success(self, agent_name):
        stats = self.shared_memory.setdefault("agent_stats", {})
        entry = stats.setdefault(agent_name, {"successes": 0, "failures": 0, "priority": 0})
        entry["successes"] += 1

    def _record_failure(self, agent_name):
        stats = self.shared_memory.setdefault("agent_stats", {})
        entry = stats.setdefault(agent_name, {"successes": 0, "failures": 0, "priority": 0})
        entry["failures"] += 1
