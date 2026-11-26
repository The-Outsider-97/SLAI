"""The tamper-evident audit ledger (src/agents/provenance/provenance_audit.py) and its use through the Provenance Agent.

Run: python -m unittest tests.agents.test_provenance_audit -v   (the agent tests need SLAI's agent dependencies, e.g. jsonschema >= 4.18, and are skipped with a
message when the agent cannot be imported; the ledger tests need nothing else)."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from src.agents.provenance.provenance_audit import GENESIS, AuditLedger
from src.agents.provenance.utils.provenance_errors import ProvenanceConflictError, ProvenanceStorageError, ProvenanceValidationError

HUMAN = {"id": "person-1", "type": "human"}


def _ledger(tmp: str, name: str = "ledger.jsonl", **kw) -> AuditLedger:
    return AuditLedger(Path(tmp) / name, **kw)


def _fill(ledger: AuditLedger, n: int = 5) -> None:
    for i in range(n):
        ledger.append(event_id=f"e{i}", kind="spadra.command", actor=HUMAN, subject={"elements": [f"el-{i}"]}, revision={"before": i, "after": i + 1}, payload={"op": "model.add_wall", "n": i})


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_chain_is_valid_and_ordered(self) -> None:
        led = _ledger(self.tmp); _fill(led)
        out = led.verify()
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["records"], 5)
        recs = led.records()
        self.assertEqual([r["seq"] for r in recs], [1, 2, 3, 4, 5])
        self.assertEqual(recs[0]["prev"], GENESIS)
        self.assertEqual(recs[1]["prev"], recs[0]["hash"])
        self.assertEqual(out["authenticity"], "unsigned")

    def test_reopen_continues_the_same_chain(self) -> None:
        led = _ledger(self.tmp); _fill(led, 3)
        again = _ledger(self.tmp)
        again.append(event_id="later", kind="spadra.command", actor=HUMAN)
        self.assertTrue(_ledger(self.tmp).verify()["ok"])
        self.assertEqual(again.head()["seq"], 4)

    def test_replayed_event_is_idempotent_and_a_conflicting_replay_is_refused(self) -> None:
        led = _ledger(self.tmp)
        first = led.append(event_id="x", kind="k", actor=HUMAN, payload={"a": 1})
        again = led.append(event_id="x", kind="k", actor=HUMAN, payload={"a": 1})
        self.assertEqual(first["hash"], again["hash"])
        self.assertEqual(len(led.records()), 1)
        with self.assertRaises(ProvenanceConflictError):
            led.append(event_id="x", kind="k", actor=HUMAN, payload={"a": 2})
        self.assertEqual(len(led.records()), 1)

    def test_changed_record_is_detected(self) -> None:
        led = _ledger(self.tmp); _fill(led)
        lines = led.path.read_text(encoding="utf-8").split("\n")
        rec = json.loads(lines[2]); rec["payload"]["n"] = 99
        lines[2] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
        led.path.write_text("\n".join(lines), encoding="utf-8")
        out = _ledger(self.tmp).verify()
        self.assertFalse(out["ok"])
        self.assertIn("record_changed", {p["problem"] for p in out["problems"]})

    def test_removed_and_reordered_records_are_detected(self) -> None:
        led = _ledger(self.tmp); _fill(led)
        lines = led.path.read_text(encoding="utf-8").split("\n")
        led.path.write_text("\n".join(lines[:1] + lines[2:]), encoding="utf-8")
        out = _ledger(self.tmp).verify()
        self.assertFalse(out["ok"]); self.assertIn("chain_broken", {p["problem"] for p in out["problems"]})
        led.path.write_text("\n".join([lines[1], lines[0]] + lines[2:]), encoding="utf-8")
        self.assertFalse(_ledger(self.tmp).verify()["ok"])

    def test_truncation_is_caught_by_an_anchor_kept_elsewhere(self) -> None:
        led = _ledger(self.tmp); _fill(led)
        anchor = led.head()
        lines = led.path.read_text(encoding="utf-8").split("\n")
        led.path.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")        # a clean cut: the chain itself is still consistent
        cut = _ledger(self.tmp)
        self.assertTrue(cut.verify()["ok"])
        self.assertFalse(cut.verify_anchor(anchor)["ok"])
        self.assertTrue(led.verify_anchor({"seq": 2, "hash": cut.records()[1]["hash"]})["ok"])

    def test_replaced_history_is_caught_by_an_anchor(self) -> None:
        led = _ledger(self.tmp); _fill(led, 3)
        anchor = led.head()
        other = _ledger(self.tmp, "other.jsonl")
        for i in range(3):
            other.append(event_id=f"f{i}", kind="spadra.command", actor=HUMAN, payload={"n": i})
        self.assertTrue(other.verify()["ok"])                                       # a perfectly consistent chain, but not the one that was anchored
        self.assertFalse(other.verify_anchor(anchor)["ok"])

    def test_torn_last_line_blocks_appends_until_recovered(self) -> None:
        led = _ledger(self.tmp); _fill(led, 3)
        with open(led.path, "ab") as handle:
            handle.write(b'{"schema":"slai.provenance.audit.v1","seq":4,"event_')   # a crash in the middle of a write
        torn = _ledger(self.tmp)
        self.assertFalse(torn.writable())
        self.assertIn("torn_tail", {p["problem"] for p in torn.verify()["problems"]})
        with self.assertRaises(ProvenanceStorageError):
            torn.append(event_id="n", kind="k", actor=HUMAN)
        self.assertGreater(torn.recover()["removed_bytes"], 0)
        self.assertTrue(torn.verify()["ok"]); self.assertEqual(torn.head()["seq"], 3)
        torn.append(event_id="n", kind="k", actor=HUMAN)
        self.assertTrue(_ledger(self.tmp).verify()["ok"])

    def test_secrets_reasoning_and_restricted_text_are_refused(self) -> None:
        led = _ledger(self.tmp)
        for key in ("token", "Password", "reasoning", "source_text"):
            with self.assertRaises(ProvenanceValidationError, msg=key):
                led.append(event_id=f"k-{key}", kind="k", actor=HUMAN, payload={"outer": {"inner": [{key: "x"}]}})
        with self.assertRaises(ProvenanceValidationError):
            led.append(event_id="big", kind="k", actor=HUMAN, payload={"blob": "x" * 70000})
        with self.assertRaises(ProvenanceValidationError):
            led.append(event_id="bad-actor", kind="k", actor={"id": "x", "type": "robot"})
        self.assertEqual(led.records(), [])

    def test_hmac_authenticity_is_claimed_only_with_the_key(self) -> None:
        key = b"k" * 32
        led = _ledger(self.tmp, hmac_key=key); _fill(led, 3)
        self.assertEqual(led.verify()["authenticity"], "hmac-verified")
        self.assertEqual(_ledger(self.tmp).verify()["authenticity"], "unsigned")      # without the key nothing is claimed
        self.assertEqual(_ledger(self.tmp, hmac_key=b"w" * 32).verify()["authenticity"], "hmac-failed")

    def test_concurrent_appends_keep_one_valid_chain(self) -> None:
        led = _ledger(self.tmp)
        def work(t: int) -> None:
            for i in range(25):
                led.append(event_id=f"t{t}-{i}", kind="k", actor=HUMAN, payload={"t": t, "i": i})
        threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        out = _ledger(self.tmp).verify()
        self.assertTrue(out["ok"], out["problems"][:3]); self.assertEqual(out["records"], 200)

    def test_export_redacts_without_touching_the_ledger(self) -> None:
        led = _ledger(self.tmp)
        led.append(event_id="a", kind="k", actor=HUMAN, payload={"user_note": "private", "value": 1})
        out = led.export(redact=["user_note"])
        self.assertEqual(out["records"][0]["payload"]["user_note"], "[redacted]")
        self.assertEqual(out["records"][0]["payload"]["value"], 1)
        self.assertTrue(led.verify()["ok"]); self.assertEqual(led.records()[0]["payload"]["user_note"], "private")


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from src.agents.collaborative.shared_memory import SharedMemory
            from src.agents.provenance_agent import ProvenanceAgent
        except Exception as exc:                                                    # noqa: BLE001 - an environment without the agent dependencies
            raise unittest.SkipTest(f"The Provenance Agent cannot be imported here: {type(exc).__name__}: {exc}") from exc
        cls._tmp = tempfile.TemporaryDirectory()
        from src.agents.provenance.provenance_store import ProvenanceStore
        cls.agent = ProvenanceAgent(SharedMemory(), None, {})
        cls.agent.provenance_store = ProvenanceStore(Path(cls._tmp.name) / "store.json")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_events_are_recorded_mirrored_and_verifiable(self) -> None:
        ledger = self.agent.audit_ledger(Path(self._tmp.name) / "agent-ledger.jsonl")
        a = self.agent.record_audit_event(ledger, event_id="finding-1", kind="regulation.evaluation", actor={"id": "spadra", "type": "system"}, payload={"rule": "R1"})
        b = self.agent.record_audit_event(ledger, event_id="proposal-1", kind="regulation.proposal", actor={"id": "spadra", "type": "system"}, causes=["finding-1"])
        self.assertTrue(a["mirrored"] and b["mirrored"], (a, b))
        self.assertIn("audit:finding-1", self.agent.get_ancestors("audit:proposal-1"))
        self.assertTrue(self.agent.verify_audit(ledger)["ok"])
        again = self.agent.record_audit_event(ledger, event_id="finding-1", kind="regulation.evaluation", actor={"id": "spadra", "type": "system"}, payload={"rule": "R1"})
        self.assertEqual(again["record"]["hash"], a["record"]["hash"])


if __name__ == "__main__":
    unittest.main()
