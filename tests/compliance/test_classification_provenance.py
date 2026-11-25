"""GOV-P0-01: classification assertions are evidence, not caller-controlled flags."""
from __future__ import annotations

import time
import unittest

from src.agents.governance.classification_evidence import (
    ClassificationAuthority, ClassificationError, ClassifierResult, ClassifierUnavailable, EvidenceOrigin, PrivacyClassifierAdapter)
from src.agents.governance.processing_context import DataClass, Decision, Operation, ProcessingContext
from tests.compliance._fixtures import GovernanceTestCase, make_activity, structural_test_classifier


class ClassificationProvenanceTests(GovernanceTestCase):
    def rule_reasons(self, decision):
        return {(r.rule_id, r.reason) for r in decision.results}

    def assertRefusedForClassification(self, decision):
        """Not executable, and refused (among other rules) by the classification rule: the data is not trusted non-personal."""
        self.assertFalse(decision.allowed, decision.to_dict())
        self.assertIn("GOV-CLS-02", {r.rule_id for r in decision.results}, decision.to_dict())

    # ---------------------------------------------------------------- the old bypass is closed
    def test_boolean_assertion_no_longer_grants_non_personal_status(self):
        c = ProcessingContext(Operation.PROCESS, "calc", data_class=DataClass.NON_PERSONAL, classification_verified=True)
        self.assertEqual(c.effective_data_class, DataClass.UNCLASSIFIED)
        d = self.core.authorize(c)
        self.assertRefusedForClassification(d)
        self.assertEqual(d.classification_status, "unverified_non_personal_claim")

    def test_real_evidence_for_structurally_non_personal_data_is_honoured(self):
        d = self.core.authorize(self.np_ctx(Operation.PROCESS, "calc"))
        self.assertEqual(d.decision, Decision.ALLOW, d.to_dict())
        self.assertEqual(d.classification_status, "evidence_ok:verified")

    # ---------------------------------------------------------------- forgery / tampering
    def test_forged_and_tampered_tokens_are_rejected(self):
        good = self.np_ctx(Operation.PROCESS, "calc")
        body, sig = good.classification_evidence.split(".", 1)
        for label, token in {
            "forged_signature": f"{body}.{'A' * len(sig)}",
            "garbage": "not-a-token",
            "other_body": f"{body[:-2]}xx.{sig}",
        }.items():
            with self.subTest(label):
                c = ProcessingContext(Operation.PROCESS, "calc", data_class=DataClass.NON_PERSONAL, classification_evidence=token,
                                      data_digest=good.data_digest)
                d = self.core.authorize(c)
                self.assertRefusedForClassification(d)
                self.assertTrue(d.classification_status.startswith("evidence_rejected:"), d.classification_status)

    def test_evidence_signed_by_another_deployment_key_is_rejected(self):
        other = ClassificationAuthority(b"another-deployment-key-0123456789-abcdef")
        other.attach_classifier(structural_test_classifier)
        token = other.classify(self.payload_np, purpose="calc", tenant_id=None)
        c = ProcessingContext(Operation.PROCESS, "calc", data_class=DataClass.NON_PERSONAL, classification_evidence=token,
                              data_digest=other.digest(self.payload_np))
        self.assertRefusedForClassification(self.core.authorize(c))

    # ---------------------------------------------------------------- binding to data / purpose / tenant / identity
    def test_evidence_cannot_be_moved_to_other_data(self):
        good = self.np_ctx(Operation.PROCESS, "calc")
        other_digest = self.core.classification.digest({"value": 43})
        c = ProcessingContext(Operation.PROCESS, "calc", data_class=DataClass.NON_PERSONAL,
                              classification_evidence=good.classification_evidence, data_digest=other_digest)
        d = self.core.authorize(c)
        self.assertRefusedForClassification(d)
        self.assertEqual(d.classification_status, "evidence_rejected:scope_mismatch")

    def test_evidence_without_data_digest_on_the_context_does_not_apply(self):
        good = self.np_ctx(Operation.PROCESS, "calc")
        c = ProcessingContext(Operation.PROCESS, "calc", data_class=DataClass.NON_PERSONAL, classification_evidence=good.classification_evidence)
        self.assertRefusedForClassification(self.core.authorize(c))

    def test_evidence_is_bound_to_purpose_tenant_and_principal(self):
        alice, mallory = self.ident("alice"), self.ident("mallory")
        good = self.np_ctx(Operation.PROCESS, "calc", identity=alice, tenant_id="tenant-a")
        common = dict(data_class=DataClass.NON_PERSONAL, classification_evidence=good.classification_evidence, data_digest=good.data_digest)
        cases = {
            "purpose": ProcessingContext(Operation.PROCESS, "marketing", identity=alice, tenant_id="tenant-a", **common),
            "tenant": ProcessingContext(Operation.PROCESS, "calc", identity=self.ident("alice", tenant="tenant-b"), tenant_id="tenant-b", **common),
            "principal": ProcessingContext(Operation.PROCESS, "calc", identity=mallory, tenant_id="tenant-a", **common),
        }
        for label, c in cases.items():
            with self.subTest(label):
                d = self.core.authorize(c)
                self.assertEqual(d.classification_status, "evidence_rejected:scope_mismatch")
                self.assertNotEqual(d.decision, Decision.ALLOW)
        self.assertEqual(self.core.authorize(ProcessingContext(Operation.PROCESS, "calc", identity=alice, tenant_id="tenant-a", **common)).decision, Decision.ALLOW)

    # ---------------------------------------------------------------- expiry, replay, reclassification
    def test_expired_evidence_is_rejected(self):
        authority = ClassificationAuthority(b"k" * 40, ttl=60.0, clock=lambda: 1000.0)
        authority.attach_classifier(structural_test_classifier)
        token = authority.classify(self.payload_np, purpose="calc", tenant_id=None)
        c = ProcessingContext(Operation.PROCESS, "calc", data_class=DataClass.NON_PERSONAL, classification_evidence=token,
                              data_digest=authority.digest(self.payload_np))
        self.assertEqual(authority.resolve(c, now=1030.0).status, "evidence_ok:verified")
        self.assertEqual(authority.resolve(c, now=1061.0).status, "evidence_rejected:expired")

    def test_single_use_evidence_cannot_be_replayed(self):
        token = self.core.classification.classify(self.payload_np, purpose="calc", tenant_id=None, single_use=True)
        digest = self.core.classification.digest(self.payload_np)
        def ctx():
            return ProcessingContext(Operation.PROCESS, "calc", data_class=DataClass.NON_PERSONAL, classification_evidence=token, data_digest=digest)
        self.assertEqual(self.core.authorize(ctx()).decision, Decision.ALLOW)
        replay = self.core.authorize(ctx())
        self.assertRefusedForClassification(replay)
        self.assertEqual(replay.classification_status, "evidence_rejected:consumed")

    def test_blocked_request_does_not_burn_single_use_evidence(self):
        token = self.core.classification.classify(self.payload_np, purpose="calc", tenant_id=None, single_use=True)
        digest = self.core.classification.digest(self.payload_np)
        blocked = ProcessingContext(Operation.EXECUTE, "calc", data_class=DataClass.NON_PERSONAL, classification_evidence=token,
                                    data_digest=digest, capability="nobody.registered", identity=self.ident())
        self.assertEqual(self.core.authorize(blocked).decision, Decision.BLOCK)
        ok = ProcessingContext(Operation.PROCESS, "calc", data_class=DataClass.NON_PERSONAL, classification_evidence=token, data_digest=digest)
        self.assertEqual(self.core.authorize(ok).decision, Decision.ALLOW)

    def test_reclassification_supersedes_earlier_evidence_for_the_resource(self):
        first = self.core.classification.classify({"value": 1}, purpose="calc", tenant_id=None, resource="doc-1")
        d1 = self.core.classification.digest({"value": 1})
        c1 = ProcessingContext(Operation.PROCESS, "calc", data_class=DataClass.NON_PERSONAL, classification_evidence=first, data_digest=d1, resource="doc-1")
        self.assertEqual(self.core.authorize(c1).decision, Decision.ALLOW)
        # the resource changes materially and is classified again
        self.core.classification.classify({"value": 2, "note": "name: Jan"}, purpose="calc", tenant_id=None, resource="doc-1")
        d = self.core.authorize(c1)
        self.assertEqual(d.classification_status, "evidence_rejected:superseded")
        self.assertRefusedForClassification(d)

    def test_invalidate_resource_on_material_change(self):
        token = self.core.classification.classify(self.payload_np, purpose="calc", tenant_id=None, resource="r1")
        self.assertEqual(self.core.classification.invalidate_resource(None, "r1", "recipient list changed"), 1)
        with self.assertRaises(ClassificationError) as cm:
            self.core.classification.verify(token)
        self.assertEqual(cm.exception.code, "superseded")

    # ---------------------------------------------------------------- origins: who can relax, who can only escalate
    def test_free_text_without_findings_can_never_become_non_personal(self):
        def free_text(payload, *, purpose):  # a classifier that found nothing in free text
            return ClassifierResult(DataClass.NON_PERSONAL, (DataClass.NON_PERSONAL,), 0.99, "t", "1", True, structural=False)
        self.core.classification.attach_classifier(free_text)
        payload = {"note": "Jan de Vries woont in Utrecht"}
        token = self.core.classification.classify(payload, purpose="calc", tenant_id=None)
        c = ProcessingContext(Operation.PROCESS, "calc", data_class=DataClass.NON_PERSONAL, classification_evidence=token,
                              data_digest=self.core.classification.digest(payload))
        d = self.core.authorize(c)
        self.assertRefusedForClassification(d)
        self.assertEqual(self.core.classification.verify(token).origin, EvidenceOrigin.INFERRED)

    def test_low_confidence_classification_is_inferred_and_cannot_relax(self):
        self.core.classification.attach_classifier(
            lambda payload, *, purpose: ClassifierResult(DataClass.NON_PERSONAL, (DataClass.NON_PERSONAL,), 0.5, "t", "1", True, True))
        token = self.core.classification.classify(self.payload_np, purpose="calc", tenant_id=None)
        self.assertEqual(self.core.classification.verify(token).origin, EvidenceOrigin.INFERRED)
        c = ProcessingContext(Operation.PROCESS, "calc", data_class=DataClass.NON_PERSONAL, classification_evidence=token,
                              data_digest=self.core.classification.digest(self.payload_np))
        self.assertRefusedForClassification(self.core.authorize(c))

    def test_evidence_can_escalate_an_understated_declaration(self):
        """A caller who under-declares special-category data as plain personal data gets the stricter treatment."""
        self.core.classification.attach_classifier(
            lambda payload, *, purpose: ClassifierResult(DataClass.SPECIAL_CATEGORY, (DataClass.SPECIAL_CATEGORY, DataClass.PERSONAL), 0.97, "t", "1", True))
        self.register()  # activity covers PERSONAL only and has no Art. 9 condition
        payload = {"note": "diagnosis: ..."}
        token = self.core.classification.classify(payload, purpose="document_editing", tenant_id="tenant-a", identity=self.ident())
        c = self.ctx(data_class=DataClass.PERSONAL, classification_evidence=token, data_digest=self.core.classification.digest(payload))
        d = self.core.authorize(c)
        self.assertEqual(d.effective_data_class, DataClass.SPECIAL_CATEGORY.value)
        self.assertIn("GOV-SPC-06", self.rule_ids(d))
        self.assertIn("GOV-PUR-05", self.rule_ids(d))

    def test_verified_evidence_never_relaxes_below_the_callers_own_stricter_declaration(self):
        token = self.core.classification.classify(self.payload_np, purpose="calc", tenant_id=None)
        c = ProcessingContext(Operation.PROCESS, "calc", data_class=DataClass.PERSONAL, classification_evidence=token,
                              data_digest=self.core.classification.digest(self.payload_np))
        self.assertEqual(self.core.authorize(c).effective_data_class, DataClass.PERSONAL.value)

    def test_attestation_requires_a_verified_role_holder_and_a_data_binding(self):
        digest = self.core.classification.digest({"anonymised": "x"})
        with self.assertRaises(ClassificationError) as cm:
            self.core.classification.attest(self.ident("intern", roles=("viewer",)), data_class=DataClass.ANONYMISED, data_digest=digest,
                                            purpose="research", tenant_id=None, rationale="irreversibly aggregated dataset")
        self.assertEqual(cm.exception.code, "not_authorised")
        dpo = self.human("dpo.jansen", roles=("dpo",))
        token = self.core.classification.attest(dpo, data_class=DataClass.ANONYMISED, data_digest=digest, purpose="research",
                                                tenant_id=None, rationale="irreversibly aggregated dataset, k>=20")
        c = ProcessingContext(Operation.PROCESS, "research", data_class=DataClass.ANONYMISED, classification_evidence=token, data_digest=digest)
        d = self.core.authorize(c)
        self.assertEqual(d.decision, Decision.ALLOW, d.to_dict())
        self.assertEqual(d.classification_status, "evidence_ok:authoritative")

    # ---------------------------------------------------------------- availability
    def test_unavailable_classifier_yields_no_evidence_and_fails_closed(self):
        self.core.classification.attach_classifier(None)
        with self.assertRaises(ClassifierUnavailable):
            self.core.classification.classify(self.payload_np, purpose="calc", tenant_id=None)
        def crashing(payload, *, purpose):
            raise RuntimeError("model file missing")
        self.core.classification.attach_classifier(crashing)
        with self.assertRaises(ClassifierUnavailable):
            self.core.classification.classify(self.payload_np, purpose="calc", tenant_id=None)
        c = ProcessingContext(Operation.PROCESS, "calc", data_class=DataClass.NON_PERSONAL)
        self.assertRefusedForClassification(self.core.authorize(c))

    def test_without_signing_key_nothing_verifies(self):
        import tempfile
        from src.agents.governance.core import GovernanceCore
        with tempfile.TemporaryDirectory() as tmp:
            core = GovernanceCore(tmp, master_key=None)
            try:
                self.assertFalse(core.classification.available)
                self.assertIn("classification_authority", core.health()["problems"])
            finally:
                core.close()

    # ---------------------------------------------------------------- adapter: the Privacy Agent stays the only detector
    def test_privacy_adapter_maps_detector_findings_without_adding_detection(self):
        class FakeDataID:
            def __init__(self, result): self.result, self.calls = result, []
            def identify_entities(self, payload, **kw):
                self.calls.append(kw)
                return self.result
        health = FakeDataID({"detected_entities": [{"category": "health_attribute", "confidence": 0.95}],
                             "sensitive_attributes": [{"attribute": "health", "confidence": 0.9}], "entity_counts": {"diagnosis": 1}})
        out = PrivacyClassifierAdapter(health)({"x": "y"}, purpose="p")
        self.assertEqual((out.data_class, out.confidence), (DataClass.SPECIAL_CATEGORY, 0.9))
        self.assertNotIn("request_id", health.calls[0])  # classification has no side effects on the detector's shared memory
        contact = PrivacyClassifierAdapter(FakeDataID({"detected_entities": [{"category": "direct_identifier", "confidence": 0.8}], "sensitive_attributes": []}))
        out = contact({"x": "y"}, purpose="p")
        self.assertEqual(out.data_class, DataClass.PERSONAL)
        self.assertLess(out.confidence, 0.85)  # a doubtful detection cannot become VERIFIED evidence
        nothing = PrivacyClassifierAdapter(FakeDataID({"detected_entities": [], "sensitive_attributes": []}))({"x": "free text"}, purpose="p")
        self.assertEqual(nothing.data_class, DataClass.UNCLASSIFIED)
        structural = PrivacyClassifierAdapter(FakeDataID({}), structural_check=lambda p: True)({"n": 1}, purpose="p")
        self.assertTrue(structural.structural)
        self.assertEqual(structural.data_class, DataClass.NON_PERSONAL)

        class Broken:
            def identify_entities(self, payload, **kw): raise RuntimeError("config invalid")
        with self.assertRaises(ClassifierUnavailable):
            PrivacyClassifierAdapter(Broken())({"x": "y"}, purpose="p")

    def test_real_privacy_data_id_is_the_detector_when_importable(self):
        try:
            from src.agents.privacy.data_id import DataID
            data_id = DataID()
        except Exception as exc:  # heavy optional runtime (logging/config/yaml) not available in this interpreter
            self.skipTest(f"DataID unavailable in this environment: {type(exc).__name__}")
        adapter = PrivacyClassifierAdapter(data_id)
        out = adapter({"customer": {"email": "jane.doe@example.com", "phone": "+1 202-555-0188"}}, purpose="support")
        self.assertIn(out.data_class, (DataClass.PERSONAL, DataClass.SPECIAL_CATEGORY))
        self.assertGreater(sum(out.findings.values()), 0)


if __name__ == "__main__":
    unittest.main()
