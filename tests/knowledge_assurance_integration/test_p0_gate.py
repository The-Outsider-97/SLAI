"""Repository-integration security gate; run ONLY after applying P0 hardening.

Static guards supplement, but do not replace, integration and adversarial tests.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


class PhaseAIntegrationGate(unittest.TestCase):
    def test_legacy_code_executor_is_removed(self):
        s = read("src/agents/knowledge/modules/rule_engine.py")
        worker = s.split("def _run_rule_code(", 1)[1].split("class RuleEngine:", 1)[0]
        self.assertNotIn("exec(rule_code", worker)
        self.assertIn("interpret_legacy_rule_code", worker)

    def test_untrusted_document_action_path_disabled(self):
        s = read("src/agents/knowledge/perform_action.py")
        method = s.split("    def from_knowledge(", 1)[1].split("    def _process_document(", 1)[0]
        self.assertIn("knowledge_document_actions_disabled", method)
        self.assertNotIn("self._execute_action(", method)
        agent = read("src/agents/knowledge_agent.py")
        response = agent.split("    def respond_to_query(", 1)[1].split("    def _detect_action_trigger(", 1)[0]
        self.assertNotIn("orchestrator.execute_actions(", response)

    def test_public_only_legacy_retrieval(self):
        s = read("src/agents/knowledge_agent.py")
        ingestion = s.split("    def add_document(", 1)[1].split("    def _update_vocabulary(", 1)[0]
        search = s.split("    def retrieve(", 1)[1].split("    def _retrieve_tfidf(", 1)[0]
        self.assertIn('doc_metadata.get("visibility") != "public"', ingestion)
        self.assertIn('doc.get("metadata", {}).get("visibility") == "public"', search)

    def test_cache_hash_not_simhash_identity(self):
        s = read("src/agents/knowledge/knowledge_cache.py")
        method = s.split("    def hash_query(", 1)[1].split("    def _simhash(", 1)[0]
        self.assertIn("hashlib.sha256", method)
        self.assertNotIn("return self._simhash", method)

    def test_sync_preserves_conflicts(self):
        s = read("src/agents/knowledge/knowledge_sync.py")
        method = s.split("    def _sync_memory_with_external(", 1)[1].split("    def _sync_rule_engine(", 1)[0]
        self.assertIn("memory_conflicts_pending_review", method)
        self.assertNotIn("resolved = self.resolve_conflict", method)

    def test_live_health_probe_not_import(self):
        s = read("src/agents/knowledge/runtime/health.py")
        self.assertIn("self._bound_components", s)
        self.assertIn("No live instance bound", s)
        readiness = s.split("    def readiness(", 1)[1].split("    def shutdown(", 1)[0]
        self.assertIn("status != HealthStatus.HEALTHY", readiness)


if __name__ == "__main__":
    unittest.main()
