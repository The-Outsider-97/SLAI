# SLAI / LSTP benchmark v0.2 — review-only replacement

This package replaces only `benchmarks/lstp/`. It neither edits core SLAI
agents nor LSTP's upstream repository. Do not extract over an unreviewed local
benchmark directory; back up your existing `benchmarks/lstp/` first.

## Fixed since pilot v0.1

- No unconditional `/v1/models` request. The original 403 occurred during
  optional model discovery, before any evaluation. v0.2 selects a
  documented, known-priced text model for each provider and lets the actual
  inference endpoint enforce account/model permissions. Inference 403s remain
  explicit blocking errors, never retried through a different route.
- Locally reads OpenAI, Groq and Gemini credentials from SLAI's `.env`, without
  printing or saving the secrets in reports. Process variables override `.env`.
- Fixes Windows PowerShell stderr handling for an existing non-zero Ruff gate.
- Corrects GPT-6 Luna's standard text pricing to $0.10 input / $0.50 output
  per million tokens (verified 2026-11-17).
- Adds 1,000-case default, configurable up to 5,400 distinct synthetic cases.
- Preserves each LLM/model/case's three formats as a paired randomized block.
- Reserves conservative budget for a whole paired block before starting it.
- Explicit `BUDGET_LIMITED` reports are completed without claiming 100% task
  coverage; provider errors produce actionable, partial reports.
- Adds tests for key behaviors including endpoint-denied 403, no model-listing
  dependency, deterministic 1,000-case generation and budget stops.

## Important research limitations

- This measures how faithfully a model extracts identical information from
  natural-language, ordinary-JSON and canonical LSTP inputs. It does **not**
  prove broad reasoning, multi-agent security, tool execution or downstream
  productivity improvements.
- The 5,400 possible synthetic cases are combinations of 20 resources, five
  actions, three permission levels, three output formats and six stages;
  sentences reuse four templates. They are not 5,400 independent real-world
  scenarios. Bootstrap intervals do not remedy that limitation.
- With 1,000 cases and three models, the complete plan is 9,000 model calls.
  The approved $5 OpenAI, $5 Gemini and free Groq budgets may stop the run
  before all are complete. A sample with missing conditions must remain partial.
- Local conservative token/cost estimates are not billing-provider hard caps.
  Actual billed amounts may differ. Confirm account-level budget controls and
  Groq free-tier status before any live requests.

## Review and run

```powershell
Set-Location 'G:\GAIA\AI\SLAI'
.\benchmarks\lstp\run_benchmark.ps1 -LSTPPath 'G:\GAIA\AI\LSTP' -SkipInstall
.\benchmarks\lstp\run_benchmark.ps1 -LSTPPath 'G:\GAIA\AI\LSTP' -Live -ConfirmPricing -ConfirmGroqFreeTier -SkipInstall
notepad '.\benchmarks\lstp\results\paired_1000_v02\final_report.md'
```

The first command makes zero provider API requests; the second makes paid
calls only after four critical offline gates pass and keys are available.
