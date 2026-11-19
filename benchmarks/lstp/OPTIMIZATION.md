# Running the optimization evaluation

This extends the existing SLAI benchmark; it does not overwrite its historical results. The model-facing API is opt-in through src/integrations/lstp_adapter.py. Existing agent inference calls are not automatically rewritten. For the exact seven-field structured extraction task, extract_action_fields(packet).as_mapping() already completes the task without an API request. prepare_action_extraction requires a real host token counter and retains the original source for independent host authorization.

From the SLAI root after applying the patches and installing the updated LSTP package:

~~~powershell
python -m pip install -e 'G:\GAIA\AI\LSTP[dev]'
python -m benchmarks.lstp.optimization --cases 1000 --output benchmarks/lstp/results/optimization_offline --baseline benchmarks/lstp/results/paired_1000_gemini_groq_v03
~~~

The offline report verifies canonical byte round trips, exact deterministic projections, fair JSON controls, CPU overhead, and UTF-8 sizes. Provider-token and API-cost fields remain null. Optional local BPE counts require tiktoken installed and an explicit encoding; they count instruction and payload content, not provider framing, generated output, caching or reasoning. Example additional argument:

~~~powershell
--tokenizer-command '["python","benchmarks/lstp/count_tokens.py","--encoding","o200k_base"]'
~~~

An encoding name is an explicit evaluation choice, not an assertion that a particular current provider uses it.

## Live run

Run the existing run_benchmark.ps1 without -Live to generate fresh schema, core-test, independent-JavaScript, adapter and engineering-release evidence. Then use that run's preflight.json for the optimization benchmark. All five checks must pass. The bundled standard text rates were checked on 18 November 2026; recheck the linked official sources before a later paid run. The OpenAI model page distinguishes standard rates from Batch/Flex, so the lower discounted table is not used. Keys must already be configured in the local process environment, as in the existing runner; do not paste them into commands or reports.

~~~powershell
.\benchmarks\lstp\run_benchmark.ps1 -LSTPPath 'G:\GAIA\AI\LSTP' -RunName optimization_preflight -SkipInstall
python -m benchmarks.lstp.optimization --cases 1000 --live --pricing benchmarks/lstp/pricing_2026-11-18.json --preflight benchmarks/lstp/results/optimization_preflight/preflight.json --budget-usd 0.50 --output benchmarks/lstp/results/optimization_live
~~~

The cap is per provider, using conservative local reservations. It is not a provider-side billing hard cap. Nine conditions per case across two providers plan 18,000 requests; the default conservative budget may stop the run before completion. Partial runs cannot pass the full-corpus targets. Resume uses the same command and pins source digests, corpus, models, prices, budget and output limit. Completed and failed requests are not automatically resubmitted. Unresolved crash reservations stop the affected provider and count as unknown billable usage. A concurrent live run in the same output directory is rejected by an exclusive lock; a stale lock after a process crash requires review before removal.

The comparison includes original ordinary JSON, original canonical LSTP, shorter JSON instructions, JSON with tuple output, canonical with tuple output, full L1 with its schema explanation, object/tuple projections, and an identical ordinary-JSON tuple control. Correctness and all recorded input/output usage are included. Estimated spend per successful task includes incorrect and failed/retried attempts whenever usage is known; missing usage or cached-input rates remain unknown. No taxes, provider invoices, explicit cache-object storage or undisclosed fees are inferred. These clients do not create explicit cache objects, request a discounted Batch/Flex service, or enable provider-specific structured-output modes. Their economic benefit must be evaluated with equivalent JSON controls before adoption.

The ordinary-JSON tuple control and LSTP projection are byte-identical requests. A 25% protocol-specific advantage cannot be inferred from identical requests. Pure deterministic extraction is likewise available to both representations.

## Verification commands

~~~powershell
python -m pytest -q benchmarks/lstp --import-mode=importlib
python -m ruff check --isolated --line-length 100 --select E,F,I,UP,B benchmarks/lstp src/integrations
~~~

The full LSTP engineering runner accepts --no-build-isolation when compatible local PEP 517 dependencies are already installed. --skip-build-smoke is an engineering-only option and does not certify final release readiness. The delivered evidence separately records a fresh wheel installation, CLI round trip, and all source-distribution tests running against the installed wheel with the existing developer-test dependencies supplied externally.

## Direct live launcher

Run `./benchmarks/lstp/run_live_optimization.ps1 -LSTPPath ../LSTP -Cases 1000 -BudgetPerProvider 5.00` from the SLAI checkout. The launcher locates its own repository and Python, loads only OpenAI/Gemini keys from that repository's `.env`, and reuses the recorded checks only when the Python sources match. Optional `-SLAIPath`, `-PythonPath`, `-EnvFile` and `-OutputDirectory` arguments override local paths. Fingerprints normalize CRLF to LF for portable Git checkouts; no test result is invented or skipped as a passing test. Existing verification is recorded with its engineering options.
