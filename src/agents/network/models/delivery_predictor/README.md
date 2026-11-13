# DeliveryPredictor

Predictive controller for the Network Agent (`src/agents/network/delivery_predictor.py`).

## What it does
For each delivery attempt it may do exactly two things, and only on top of what the deterministic stack already decided:

1. **Shorten** the transport timeout (never lengthen it, never below `min_timeout_ms`).
2. **Veto** a retry that policy and reliability already allowed.

Everything else (routing, policy, circuit breaker, idempotency, failover) stays deterministic and authoritative.
Any exception, malformed checkpoint or non-finite value degrades to the baseline; after
`max_consecutive_errors` errors the predictor sits out for `error_cooldown_s`.

## Modes (`delivery_predictor.mode` in `network_config.yaml`)
| mode | behaviour |
|---|---|
| `off` | not consulted |
| `shadow` (default) | predictions and "would have" decisions are recorded in the health snapshot; nothing is changed |
| `active` | decisions are applied. Requires a manifest whose `promotion.status` is `approved`, or `allow_experimental_active: true` |

## Current status: **experimental, shadow-only**
The shipped artifact (`models/delivery_predictor/manifest.json`) did **not** pass the fixed promotion gates on a
final evaluation with fresh seeds (45 sessions, seeds 200000+): the paired time-saving 95% CI includes zero
(mean +0.08 ms, CI -1.15..+1.46) and the delivered-rate delta CI low (-0.014) is below the -0.01 gate.
All data is synthetic fault injection against loopback servers, not production traffic. Re-run the pipeline on
real attempt logs before enabling `active`.

## Reproducing
```
python train_dp.py collect  --start 0  --sessions 40     # repeat with --start 40, 80, ...
python train_dp.py train
DP_EVAL_SEED_BASE=200000 python train_dp.py evaluate --start 0 --sessions 15 --run <dir>   # repeat 15, 30
python train_dp.py report  --run <dir>
python train_dp.py promote --run <dir>                   # applies the fixed gates, rewrites manifest promotion
```
Gates: >= 30 test sessions, delivered-rate delta CI low >= -0.01, net-time-saving CI low > 0,
duplicate ratio <= 1.10, mean overhead <= 2000 us, zero predictor errors.

## Checkpoint
Constrained JSON (no pickle). Success head: logistic or `gbt_logistic`; optional retry-specific logistic head;
latency head: ridge on log1p(ms) with empirical residual quantiles. Inference is pure Python. The feature list is
hashed (`feature_hash`); a mismatch refuses the checkpoint.
