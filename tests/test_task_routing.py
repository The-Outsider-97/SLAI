import os
import sys
import torch
import unittest
from collaborative.registry import AgentRegistry
from collaborative.task_router import TaskRouter

# Dummy Agent for testing
class DummyAgent:
    def __init__(self, name, should_fail=False):
        self.name = name
        self.should_fail = should_fail

    def execute(self, task_data):
        if self.should_fail:
            raise Exception(f"{self.name} failed intentionally.")
        return {"agent": self.name, "status": "success", "data": task_data}

class TestTaskRouter(unittest.TestCase):
    def setUp(self):
        self.registry = AgentRegistry()
        self.shared_memory = {}
        self.router = TaskRouter(self.registry, shared_memory=self.shared_memory)

    def test_successful_routing(self):
        agent = DummyAgent("Agent1")
        self.registry.register("Agent1", agent, capabilities=["test_task"])

        result = self.router.route("test_task", {"key": "value"})
        self.assertEqual(result["agent"], "Agent1")
        self.assertEqual(self.shared_memory["agent_stats"]["Agent1"]["successes"], 1)

    def test_fallback_routing(self):
        # First agent fails, second succeeds
        self.registry.register("FailAgent", DummyAgent("FailAgent", should_fail=True), capabilities=["test_task"])
        self.registry.register("GoodAgent", DummyAgent("GoodAgent"), capabilities=["test_task"])

        result = self.router.route("test_task", {"key": "value"})
        self.assertEqual(result["agent"], "GoodAgent")
        self.assertEqual(self.shared_memory["agent_stats"]["FailAgent"]["failures"], 1)
        self.assertEqual(self.shared_memory["agent_stats"]["GoodAgent"]["successes"], 1)

    def test_no_agents_available(self):
        with self.assertRaises(Exception) as context:
            self.router.route("nonexistent_task", {})
        self.assertIn("No agents found", str(context.exception))

    def test_all_agents_fail(self):
        self.registry.register("FailA", DummyAgent("FailA", should_fail=True), capabilities=["test_task"])
        self.registry.register("FailB", DummyAgent("FailB", should_fail=True), capabilities=["test_task"])

        with self.assertRaises(Exception) as context:
            self.router.route("test_task", {"key": "value"})
        self.assertIn("All agents failed", str(context.exception))

if __name__ == '__main__':
    unittest.main()
