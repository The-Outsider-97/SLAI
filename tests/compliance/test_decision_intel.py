"""GOV-P2-01: review-queue triage and decision analytics (advisory only)."""

import unittest

from src.agents.governance.decision_intel import SOURCE_WEIGHTS, analytics, rank, triage_hints
from src.agents.governance.human_oversight import DecisionDossier
from src.agents.governance.processing_context import Decision, Operation
from tests.compliance._fixtures import GovernanceTestCase

D = DecisionDossier("Review this request carefully", ("ev-1",), "moderate uncertainty", ("do nothing",), ("harm",), ("limits",))


def item(aid, created, expires, fp=None, **triage):
    return {"approval_id": aid, "level": "approval", "created_at": created, "expires_at": expires, "context_fingerprint": fp or aid, "triage": triage}


class RankingTests(unittest.TestCase):
    NOW = 1000.0

    def test_reason_and_sensitivity_drive_priority(self):
        out = rank([
            item("adaptive", 990, 1890, source="GOV-ADP-01"),
            item("concern_sensitive", 990, 1890, source="GOV-COORD-02", sensitive=True, personal=True),
            item("oversight", 990, 1890, source="GOV-HO-14", personal=True, significant=True),
        ], now=self.NOW)
        self.assertEqual([r.approval_id for r in out], ["concern_sensitive", "oversight", "adaptive"])
        self.assertEqual([r.rank for r in out], [1, 2, 3])
        self.assertIn("sensitivity", out[0].components)

    def test_ranking_is_deterministic_and_order_independent(self):
        items = [item(f"a{i}", 990 + i % 3, 1890, source="GOV-HO-14") for i in range(9)]
        first = [r.approval_id for r in rank(items, now=self.NOW)]
        self.assertEqual(first, [r.approval_id for r in rank(list(reversed(items)), now=self.NOW)])
        self.assertEqual(first[:3], ["a0", "a3", "a6"])  # ties break on age, then id

    def test_nothing_starves_and_nearly_expired_items_are_seen_first(self):
        old_low = item("old_low", 0, 1000.5, source="GOV-ADP-01")  # advisory, but about to expire
        new_mid = item("new_mid", 990, 1890, source="GOV-HO-14")
        out = rank([old_low, new_mid], now=self.NOW)
        self.assertEqual(out[0].approval_id, "old_low")
        self.assertGreater(out[0].components["about_to_expire"], 0)

    def test_duplicate_requests_for_the_same_fingerprint_are_merged_in_the_view(self):
        out = rank([item("first", 900, 1800, fp="X", source="GOV-HO-14"), item("second", 950, 1850, fp="X", source="GOV-HO-14"),
                    item("other", 960, 1860, fp="Y", source="GOV-HO-14")], now=self.NOW)
        merged = [r for r in out if r.approval_id == "first"][0]
        self.assertEqual(merged.duplicates, ("second",))
        self.assertEqual(len(out), 2)

    def test_unknown_source_is_middling_not_ignored(self):
        out = rank([item("u", 990, 1890, source="MYSTERY"), item("c", 990, 1890, source="GOV-ADP-01"), item("h", 990, 1890, source="GOV-COORD-02")], now=self.NOW)
        self.assertEqual([r.approval_id for r in out], ["h", "u", "c"])

    def test_weights_are_explicit_and_cover_every_review_source(self):
        for rid in ("GOV-COORD-02", "GOV-HO-14", "GOV-TRF-09", "GOV-CND-02", "GOV-ADP-01"):
            self.assertIn(rid, SOURCE_WEIGHTS)


class IntegrationTests(GovernanceTestCase):
    def approval(self, ctx, *, now, triage=None):
        return self.core.oversight.request_approval(ctx, D, self.ident("svc-1"), now=now, ttl=900.0, triage=triage)

    def test_pending_lists_only_open_unexpired_reviews_with_their_hints(self):
        t0 = 1_000_000.0
        a = self.approval(self.ctx(resource="r1"), now=t0, triage={"source": "GOV-HO-14", "sensitive": True})
        b = self.approval(self.ctx(resource="r2"), now=t0 - 5000)  # expired long ago
        open_items = {p["approval_id"]: p for p in self.core.oversight.pending(now=t0 + 10)}
        self.assertIn(a, open_items)
        self.assertNotIn(b, open_items)
        self.assertEqual(open_items[a]["triage"]["source"], "GOV-HO-14")

    def test_triage_hints_do_not_influence_authorisation_binding(self):
        c = self.ctx(resource="r3")
        a1 = self.approval(c, now=1.0e6, triage={"source": "GOV-ADP-01"})
        a2 = self.approval(c, now=1.0e6, triage={"source": "GOV-COORD-02", "sensitive": True})
        self.assertEqual(self.core.oversight._load(a1)[1]["context_fingerprint"], self.core.oversight._load(a2)[1]["context_fingerprint"])
        self.assertEqual(self.core.oversight._load(a1)[1]["dossier_digest"], self.core.oversight._load(a2)[1]["dossier_digest"])

    def test_hints_carry_no_payload_or_identifiers(self):
        self.register()
        d = self.core.authorize(self.ctx(automated_decision=True, significant_effect=True, resource="secret-resource-name"))
        hints = triage_hints(d, self.ctx(automated_decision=True, significant_effect=True))
        self.assertEqual(set(hints), {"source", "data_class", "sensitive", "personal", "significant"})
        self.assertNotIn("secret-resource-name", repr(hints))

    def test_review_queue_and_analytics_over_real_decisions_and_approvals(self):
        self.register()
        reviewer = self.human("reviewer-1")
        t0 = 2_000_000.0
        ids = [self.approval(self.ctx(resource=f"q{i}"), now=t0 + i) for i in range(4)]
        for i, aid in enumerate(ids[:3]):
            self.core.oversight.mark_presented(aid, reviewer, now=t0 + 100 + i)
            self.core.oversight.decide(aid, reviewer, approve=(i != 2), rationale="reviewed against the dossier", reviewed_dossier_digest=D.digest(), now=t0 + 200 + i * 100)
        self.core.authorize(self.ctx(resource="d-ok"))
        self.core.authorize(self.ctx(resource="d-bad", activity_id="nope"))
        a = analytics(self.core)
        self.assertEqual(a["decisions"], 2)
        self.assertEqual(a["by_decision"].get("BLOCK"), 1)
        self.assertEqual(a["reviews"]["by_status"]["approved"], 2)
        self.assertEqual(a["reviews"]["by_status"]["rejected"], 1)
        self.assertAlmostEqual(a["reviews"]["rejection_rate"], 1 / 3, places=3)
        self.assertEqual(a["reviews"]["latency_seconds"]["n"], 3)
        self.assertGreater(a["reviews"]["latency_seconds"]["p95"], a["reviews"]["latency_seconds"]["p50"] - 1e-9)
        self.assertTrue(any(r == "GOV-LAW-04" for r, _ in a["top_restricting_rules"]))
        queue = self.core.review_queue(now=t0 + 400)
        self.assertEqual([q.approval_id for q in queue], [ids[3]])  # only the still-open one


if __name__ == "__main__":
    unittest.main()
