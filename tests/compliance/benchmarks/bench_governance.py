"""Measured micro-benchmarks for the governance layer. Run:  python -m tests.compliance.benchmarks.bench_governance

Reports p50/p95/p99 latency in microseconds. 'baseline' is the same call shape without any governance (a bare
function call), i.e. what the system did before the layer existed, so the figures show the added cost per decision.
Numbers depend entirely on the machine and on SQLite durability settings (synchronous=FULL); never quote them
without the environment block printed at the end.
"""
from __future__ import annotations

import json
import platform
import statistics
import sys
import tempfile
import time

from src.agents.governance.core import GovernanceCore
from src.agents.governance.delegation_guard import check_delegable
from src.agents.governance.processing_context import DataClass, LawfulBasis, Operation, ProcessingContext
from src.agents.governance.privacy_ledger import PrivacyLedger
from src.agents.governance.transparency import TransparencyMarker
from src.agents.governance.risk_classification import AISystemProfile
from src.agents.governance.utils.diagnostics import redact_text
from tests.compliance._fixtures import KEY, make_activity


def pct(samples, q):
    s = sorted(samples)
    return s[min(len(s) - 1, int(q * len(s)))]


def measure(fn, n):
    for _ in range(min(50, n)):
        fn()
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1e6)
    return {"n": n, "p50_us": round(pct(out, .50), 1), "p95_us": round(pct(out, .95), 1), "p99_us": round(pct(out, .99), 1),
            "mean_us": round(statistics.mean(out), 1), "ops_per_s": round(1e6 / statistics.mean(out))}


def main(n: int = 2000) -> dict:
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        core = GovernanceCore(tmp, master_key=KEY)
        core.activities.register(make_activity("act-1"), actor="dpo.jansen")
        tok = core.identity.issue("svc-writer", "tenant-a", actor_type="service", ttl_seconds=3600)
        ident = core.identity.verify(tok)
        allowed = ProcessingContext(operation=Operation.PROCESS, purpose="document_editing", data_class=DataClass.PERSONAL,
                                    classification_verified=True, activity_id="act-1", subject_ids=("s1",), tenant_id="tenant-a", identity=ident)
        blocked = ProcessingContext(operation=Operation.PROCESS, purpose="advertising", data_class=DataClass.PERSONAL,
                                    classification_verified=True, activity_id="act-1", subject_ids=("s1",), tenant_id="tenant-a", identity=ident)

        def work(ctx):  # stand-in for the guarded unit of work
            return {"ok": True}

        results["baseline_no_governance"] = measure(lambda: work(allowed), n)
        results["engine_evaluate_allow"] = measure(lambda: core.engine.evaluate(allowed), n)
        results["authorize_allow_recorded"] = measure(lambda: core.authorize(allowed), n)
        results["authorize_block_recorded"] = measure(lambda: core.authorize(blocked), n)

        ledger = PrivacyLedger(f"{tmp}/ledger.db", tombstone_salt=b"s" * 32)
        i = [0]

        def write():
            i[0] += 1
            ledger.record_consent("t1", f"s{i[0] % 50}", "p", explicit=False, notice_version="v1", evidence_ref="e")

        results["ledger_consent_write"] = measure(write, 500)
        results["ledger_consent_read"] = measure(lambda: ledger.consent_snapshot("t1", "s1", "p"), n)
        results["ledger_verify_chain_500_events"] = measure(ledger.verify_chain, 20)

        marker = TransparencyMarker(KEY)
        prof = AISystemProfile(system_id="s", name="n", intended_purpose="drafting", is_gpai_model=False, generates_synthetic_content=True)
        text = "x" * 4000
        results["transparency_mark_4kB_text"] = measure(lambda: marker.release(prof, text, model="m"), n)

        class T:
            def run(self):
                return 1

        t = T()
        results["delegation_guard_check"] = measure(lambda: check_delegable(t, target_name="t", action="run"), n)
        results["baseline_getattr"] = measure(lambda: getattr(t, "run"), n)
        results["log_sanitize_200B_line"] = measure(lambda: redact_text("user jan@example.nl called +31 6 12345678 from 10.1.2.3 " * 3), n)
        core.close()
        ledger.close()
    results["environment"] = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
                              "sqlite_durability": "synchronous=FULL, WAL", "note": "sandbox VM, not production hardware"}
    return results


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
