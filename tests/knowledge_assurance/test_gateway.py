"""Opt-in facade and dependency-injection regression cases."""
from pathlib import Path
import tempfile
import unittest

from assurance import AccessContext, KnowledgeAssuranceGateway, Lifecycle, RecordDraft, Scope


class MockProvenance:
    def register_source(self, source_id, info):
        return {"source_id": source_id, "locator": info.get("locator")}


class MockVerification:
    def verify(self, request):
        return {"status": "unverified_mock", "request": request}


class GatewayTests(unittest.TestCase):
    def test_single_repository_read(self):
        with tempfile.TemporaryDirectory() as t:
            gateway = KnowledgeAssuranceGateway(Path(t) / "knowledge.sqlite3")
            editor = AccessContext("editor", project_ids=frozenset({"p"}), can_write=True)
            gateway.put_knowledge(RecordDraft("project_knowledge", Scope.PROJECT, {"purpose": "A"}, project_id="p"), editor)
            self.assertEqual(1, len(gateway.find_knowledge(editor)))
            self.assertEqual(0, len(gateway.find_knowledge(None)))

    def test_peers_are_delegated_not_reimplemented(self):
        with tempfile.TemporaryDirectory() as t:
            gateway = KnowledgeAssuranceGateway(Path(t) / "knowledge.sqlite3",
                                                provenance_agent=MockProvenance(),
                                                verification_agent=MockVerification())
            self.assertEqual("prov:a", gateway.register_provenance_source("prov:a", {"locator": "file:///x"})["source_id"])
            self.assertEqual("unverified_mock", gateway.delegate_formal_verification({"op": "formal"})["status"])

    def test_unattached_peers_fail_closed(self):
        with tempfile.TemporaryDirectory() as t:
            gateway = KnowledgeAssuranceGateway(Path(t) / "knowledge.sqlite3")
            with self.assertRaises(RuntimeError):
                gateway.register_provenance_source("prov:a", {})
            with self.assertRaises(RuntimeError):
                gateway.delegate_formal_verification({})
