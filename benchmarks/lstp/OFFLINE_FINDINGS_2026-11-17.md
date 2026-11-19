# LSTP v0.1 — verified offline findings (17 November 2026)

This is a **pre-API** verification record. No OpenAI, Gemini or Groq paid/hosted
model calls were executed, and no model-quality or token-saving result exists.

## Inspected code

- LSTP `main`: `8808b0980d478c732d964bde5d9af44e845c47e9`.
- SLAI `SLAI-v.2.4`: `a77f1ad22b72938a990acf6f7d1b977d780416b2`.
- Python 3.12.14 and Node.js 24.22.1 in the verification environment.

## Executed evidence

| Check | Observed outcome |
| --- | --- |
| `python tools/check_schema_contract.py` | PASS |
| `node interop/js/verify.mjs` | PASS — 2 positive JSON, 8 negative JSON, 1 positive Lattice, 1 negative Lattice fixtures |
| Four selected Python canonical/semantic/conformance test files | PASS — 53 tests |
| `python -m benchmarks.lstp.smoke_integration` | PASS — real SLAI adapter, 657 canonical bytes, zero trusted action authority |
| Proposed benchmark's own Python tests | PASS — 6 tests (including mocked 3-provider pipeline and resume) |
| Ruff on proposed benchmark code | PASS |
| `python tools/run_release_checks.py` from clean LSTP checkout | FAILED at Ruff — 131 pre-existing lint findings; later release phases were not run |
| Full LSTP `pytest -vv` attempt | INCOMPLETE — reached 55%; a 45-second diagnostic run timed out at `test_lattice_extended.py::test_parser_handles_scope_composition_and_zoom` |

The timed-out parser test is a suspected performance/hang issue requiring
targeted investigation; the observation alone does not prove the underlying
root cause. The existing engineering release gate is not green. The benchmark
runner records its result without silently changing project source or claiming
production readiness.

## Pilot coverage

- 100 deterministic synthetic cases, each represented as natural language,
  ordinary JSON and canonical LSTP JSON (300 representations).
- Three providers when locally authenticated: OpenAI, Groq and Gemini.
- 900 planned generation requests (100 cases × 3 formats × 3 models).
- Observed **median UTF-8 representation sizes** across the 100 paired cases:
  natural language 232.5 bytes, ordinary JSON 206 bytes, canonical LSTP
  610.5 bytes. These are **bytes, not model tokens**, and do not imply a
  speed/cost benefit or penalty without model-specific measurements.
- Separate paid-provider caps: OpenAI $5, Gemini $5; Groq only with explicit
  free-tier confirmation.
- Persistent per-request cost reservations, resumable JSONL results, pinned
  code revision, corpus digest, exact-match/field scoring and Markdown/JSON
  final report.
- Additional model-to-model handoff and real tool execution are outside this
  pilot's measured scope; they must not be inferred from its results.

## Next evidence required

1. Run `run_benchmark.ps1` on the user's Windows machine with actual API keys
   kept in local process environment.
2. Confirm current model availability, pricing and the Groq account's free tier.
3. Collect the 900 responses or documented budget/rate-limit stop conditions.
4. Review generated `final_report.md`, `final_report.json`, raw responses,
   paired comparisons, spending ledger and any missing data.

**Status: offline pilot implementation verified; live benchmark not executed.**
