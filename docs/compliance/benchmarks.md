# Benchmarks (measured)

Run: `python -m tests.compliance.benchmarks.bench_governance`. Sandbox Linux VM, Python 3.10, SQLite `synchronous=FULL` (every recorded decision is fsynced).
**Not production hardware; re-run on the target.** "baseline" is the same call without governance (what the code did before).

| Operation | p50 | p95 | p99 | Note |
|---|---|---|---|---|
| baseline (no governance) | 0.1 us | 0.1 us | 0.2 us | bare function call |
| Policy engine evaluation only (ALLOW) | 79 us | 147 us | 263 us | pure computation |
| `authorize` incl. durable decision record (ALLOW) | 743 us | 1.7 ms | 4.2 ms | dominated by fsync |
| `authorize` incl. durable record (BLOCK) | 711 us | 1.1 ms | 2.5 ms | |
| Ledger consent write | 1.07 ms | 2.0 ms | 4.4 ms | hash chain + fsync |
| Ledger consent read | 8 us | 9 us | 22 us | |
| Ledger `verify_chain` (~550 events) | 0.86 ms | 1.2 ms | - | 20 samples |
| Art. 50 marking of 4 kB text | 10 us | 12 us | 23 us | |
| Delegation guard check | 1.4 us | 1.5 us | 2.1 us | vs 0.1 us bare getattr |
| Log sanitiser (~200 B line) | 19 us | 24 us | 35 us | per log record |

Reading: the deterministic checks are cheap; the durable audit record costs roughly 0.7 ms per decision. If that matters for a hot path,
batch decisions or relax durability knowingly - do not disable recording. No claim is made about end-to-end agent latency.
