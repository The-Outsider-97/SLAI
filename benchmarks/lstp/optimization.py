"""Paired optimization audit and resumable live verification, with fair JSON controls.

Offline measurements never stand in for provider usage or model correctness.
Live mode requires explicit model/pricing configuration and passing preflight.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import lstp
from lstp import (
    LSTPError,
    canonical_dumps,
    canonical_loads,
    compact_dumps,
    compact_loads,
    project_action,
)
from lstp.costs import TaskAttempt, TokenPrices, TokenUsage, account_attempts
from lstp.inference import (
    COMPACT_INSTRUCTION,
    OBJECT_INSTRUCTION,
    TUPLE_INPUT_INSTRUCTION,
    TUPLE_INSTRUCTION,
    decode_action_result,
)

from benchmarks.lstp.benchmark import (
    SYSTEM,
    BudgetLedger,
    _append_jsonl,
    _atomic_json,
    _jsonl_rows,
)
from benchmarks.lstp.cases import Case, corpus_sha256, generate_cases
from benchmarks.lstp.providers import ProviderError, query


def conditions(case: Case) -> dict[str, tuple[str, str, bool]]:
    packet = canonical_loads(case.canonical_lstp)
    projection = project_action(packet)
    ordinary = json.loads(case.ordinary_json)
    # This control uses ordinary JSON's fields, never the gold annotations.
    normalized = {
        "action": ordinary["action"],
        "resource": ordinary["target_resource"],
        "project": ordinary["project"],
        "stage": ordinary["stage"],
        "permission_mode": ordinary["requested_permission"],
        "output_format": ordinary["output_format"],
        "source": ordinary["source_identifier"],
    }
    json_tuple = json.dumps(list(normalized.values()), ensure_ascii=False, separators=(",", ":"))
    return {
        "ordinary_json": (SYSTEM, case.ordinary_json, False),
        "canonical_lstp": (SYSTEM, case.canonical_lstp, False),
        "json_minimal": (OBJECT_INSTRUCTION, case.ordinary_json, False),
        "json_tuple_output": (TUPLE_INSTRUCTION, case.ordinary_json, True),
        "canonical_minimal": (TUPLE_INSTRUCTION, case.canonical_lstp, True),
        "lossless_compact": (
            COMPACT_INSTRUCTION + TUPLE_INSTRUCTION,
            compact_dumps(packet).decode(),
            True,
        ),
        "projection_object": (
            OBJECT_INSTRUCTION,
            projection.dumps(tuples=False),
            False,
        ),
        "projection_tuple": (TUPLE_INPUT_INSTRUCTION, projection.dumps(), True),
        # Identical field tuples are available to ordinary JSON as well as LSTP.
        "json_tuple_control": (TUPLE_INPUT_INSTRUCTION, json_tuple, True),
    }


def _stats(values: list[int | float]) -> dict:
    ordered = sorted(values)
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": max(values),
    }


def offline(cases: list[Case], tokenizer_command: list[str] | None = None) -> dict:
    sizes, prompts, timings = defaultdict(list), [], defaultdict(list)
    restored, projected, fair_controls = 0, 0, 0
    for case in cases:
        packet = canonical_loads(case.canonical_lstp)
        start = time.perf_counter_ns()
        canonical = canonical_dumps(packet)
        timings["canonical_encode_us"].append((time.perf_counter_ns() - start) / 1000)
        start = time.perf_counter_ns()
        compact = compact_dumps(packet)
        timings["compact_encode_us"].append((time.perf_counter_ns() - start) / 1000)
        start = time.perf_counter_ns()
        decoded = compact_loads(compact)
        timings["compact_decode_us"].append((time.perf_counter_ns() - start) / 1000)
        restored += canonical_dumps(decoded) == canonical
        start = time.perf_counter_ns()
        projection = project_action(packet)
        timings["projection_us"].append((time.perf_counter_ns() - start) / 1000)
        projected += projection.as_mapping() == case.gold
        variants = conditions(case)
        fair_controls += variants["projection_tuple"] == variants["json_tuple_control"]
        for name, (system, user, _) in variants.items():
            sizes[name].append(len(user.encode("utf-8")))
            sizes[name + "/instruction_plus_payload"].append(len((system + user).encode("utf-8")))
            prompts.append({"condition": name, "system": system, "user": user})
        zipped = gzip.compress(canonical, mtime=0)
        sizes["gzip_network_bytes"].append(len(zipped))
        sizes["gzip_base64_bytes_not_model_tokens"].append(len(base64.b64encode(zipped)))
    token_stats = None
    if tokenizer_command:
        result = subprocess.run(
            tokenizer_command,
            input="".join(json.dumps(p) + "\n" for p in prompts),
            capture_output=True,
            text=True,
            check=True,
            timeout=180,
        )
        counts = [json.loads(line) for line in result.stdout.splitlines()]
        if len(counts) != len(prompts) or any(type(n) is not int or n < 0 for n in counts):
            raise ValueError("Tokenizer must return one non-negative integer per prompt")
        grouped = defaultdict(list)
        for prompt, count in zip(prompts, counts, strict=True):
            grouped[prompt["condition"]].append(count)
        token_stats = {
            name: {**_stats(values), "total": sum(values)} for name, values in grouped.items()
        }
    return {
        "kind": "OFFLINE_ONLY",
        "cases": len(cases),
        "lossless_canonical_byte_roundtrips": restored,
        "deterministic_projection_exact_matches": projected,
        "identical_json_tuple_controls": fair_controls,
        "byte_sizes": {name: _stats(values) for name, values in sizes.items()},
        "cpu_microseconds": {name: _stats(values) for name, values in timings.items()},
        "local_tokenizer_counts": token_stats,
        "tokenizer_command": tokenizer_command,
        "provider_token_savings": None,
        "provider_cost_savings": None,
        "model_correctness": None,
    }


def audit_baseline(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    mismatches = set()
    for row in _jsonl_rows(directory / "results.jsonl"):
        if row.get("status") != "ok":
            continue
        provider, model = row["provider"], row["model"]
        if manifest.get("models", {}).get(provider) != model:
            mismatches.add((provider, manifest.get("models", {}).get(provider), model))
        groups[(provider, model, row["condition"])].append(row)
    return {
        "manifest_model_mismatches": [list(item) for item in sorted(mismatches)],
        "source_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("manifest.json", "cases.jsonl", "results.jsonl")
        },
        "groups": [
            {
                "provider": provider,
                "model_in_raw_rows": model,
                "condition": condition,
                "responses": len(rows),
                "exact_matches": sum(r["score"]["exact_match"] for r in rows),
                "input_tokens": sum(r["input_tokens"] for r in rows),
                "output_tokens": sum(r["output_tokens"] for r in rows),
                "recorded_nominal_usd": str(
                    sum((Decimal(str(r["nominal_usd"])) for r in rows), Decimal(0))
                ),
            }
            for (provider, model, condition), rows in sorted(groups.items())
        ],
    }


def aggregate_live(rows: list[dict], config: dict, expected: int) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["provider"], row["model"], row["condition"])].append(row)
    metrics = {}
    for (provider, model, condition), attempts in groups.items():
        rates = config[provider]
        prices = TokenPrices(
            Decimal(rates["input_per_million"]),
            Decimal(rates["output_per_million"]),
            Decimal(rates["cached_input_per_million"])
            if rates.get("cached_input_per_million") is not None
            else None,
        )
        records = [
            TaskAttempt(
                row["case_id"],
                row.get("exact_match", False),
                TokenUsage(
                    row["input_tokens"],
                    row["output_tokens"],
                    row.get("cached_input_tokens", 0),
                )
                if "input_tokens" in row
                else None,
            )
            for row in attempts
        ]
        value = account_attempts(records, prices)
        complete = {row["case_id"] for row in attempts if row["status"] == "ok"}
        times = [row["latency_seconds"] for row in attempts if "latency_seconds" in row]
        value.update(
            {
                "completed_tasks": len(complete),
                "expected_tasks": expected,
                "all_tasks_responded": len(complete) == expected,
                "exact_match_rate": value["successful_tasks"] / expected,
                "latency_seconds": _stats(times) if times else None,
                "model": model,
            }
        )
        metrics[provider + "/" + condition] = value
    return metrics


def compare_targets(metrics: dict) -> dict:
    comparisons = {}
    for key, candidate in metrics.items():
        provider, condition = key.split("/", 1)
        if condition not in {
            "projection_tuple",
            "lossless_compact",
            "projection_object",
        }:
            continue
        for baseline in (
            "canonical_lstp",
            "ordinary_json",
            "json_tuple_output",
            "json_tuple_control",
        ):
            control = metrics.get(provider + "/" + baseline)
            comparison = {
                "status": "UNVERIFIED",
                "token_reduction": None,
                "cost_reduction": None,
            }
            if control and candidate["all_tasks_responded"] and control["all_tasks_responded"]:
                token_a, token_b = (
                    candidate["tokens_per_success"],
                    control["tokens_per_success"],
                )
                cost_a, cost_b = (
                    candidate["cost_per_success"],
                    control["cost_per_success"],
                )
                if token_a is not None and token_b and cost_a is not None and cost_b:
                    token_gain, cost_gain = 1 - token_a / token_b, 1 - cost_a / cost_b
                    quality_ok = candidate["exact_match_rate"] >= control["exact_match_rate"]
                    comparison.update(
                        {
                            "token_reduction": token_gain,
                            "cost_reduction": cost_gain,
                            "observed_quality_nonregression": quality_ok,
                            "status": "OBSERVED_PASS"
                            if min(token_gain, cost_gain) >= Decimal("0.25") and quality_ok
                            else "OBSERVED_FAIL",
                        }
                    )
            comparisons[key + "_vs_" + baseline] = comparison
    return comparisons


def run_live(
    cases: list[Case],
    output: Path,
    config: dict,
    budget: Decimal,
    max_output_tokens: int,
    seed: int,
) -> dict:
    ledger = BudgetLedger(output / "budget_ledger.json", caps={p: budget for p in config})
    results = output / "results.jsonl"
    prior = _jsonl_rows(results)
    completed = {r["job_id"] for r in prior}  # Failed attempts are never silently resubmitted.
    rng = random.Random(seed)
    jobs = [(case, provider) for case in cases for provider in config]
    rng.shuffle(jobs)
    stopped = {
        value["provider"]: "unresolved_crash_reservation"
        for value in ledger.state["pending_usd"].values()
    }
    for case, provider in jobs:
        if provider in stopped:
            continue
        rates = config[provider]
        prices = TokenPrices(
            Decimal(rates["input_per_million"]),
            Decimal(rates["output_per_million"]),
            Decimal(rates["cached_input_per_million"])
            if rates.get("cached_input_per_million") is not None
            else None,
        )
        variants = list(conditions(case).items())
        rng.shuffle(variants)
        for condition, (system, user, tuples) in variants:
            job_id = provider + ":" + case.case_id + ":" + condition
            if job_id in completed:
                continue
            if job_id in ledger.state["pending_usd"]:
                stopped[provider] = "unresolved_crash_reservation"
                break
            upper_input = 4 * len((system + user).encode("utf-8")) + 2048
            bound = (
                (
                    Decimal(upper_input)
                    * max(
                        prices.input_per_million,
                        prices.cached_input_per_million or Decimal(0),
                    )
                    + Decimal(max_output_tokens) * prices.output_per_million
                )
                / Decimal(1_000_000)
                * 3
            )
            if not ledger.reserve(job_id, provider, bound):
                stopped[provider] = "budget"
                break
            row = {
                "job_id": job_id,
                "case_id": case.case_id,
                "provider": provider,
                "model": rates["model"],
                "condition": condition,
            }
            try:
                response = query(
                    provider,
                    rates["model"],
                    system,
                    user,
                    max_output_tokens=max_output_tokens,
                )
                usage = TokenUsage(
                    int(response["input_tokens"]),
                    int(response["output_tokens"]),
                    int(response.get("cached_input_tokens", 0)),
                )
                estimate = prices.estimate(usage)
                ledger.finalize(job_id, estimate)
                try:
                    answer = decode_action_result(str(response["text"]), tuples=tuples)
                    exact = answer == case.gold
                except (ValueError, LSTPError):
                    exact = False
                row.update(response)
                row.update({"status": "ok", "exact_match": exact})
            except ProviderError as error:
                ledger.finalize(job_id, None)
                row.update({"status": "error", "exact_match": False, "error": str(error)})
                stopped[provider] = "provider_error"
            _append_jsonl(results, row)
            completed.add(job_id)
            print(f"{provider} {case.case_id} {condition}: {row['status']}", flush=True)
            if provider in stopped:
                break
    recorded = _jsonl_rows(results)
    recorded_ids = {row["job_id"] for row in recorded}
    for job_id in ledger.state["pending_usd"]:
        if job_id not in recorded_ids:
            provider, case_id, condition = job_id.split(":", 2)
            recorded.append(
                {
                    "job_id": job_id,
                    "provider": provider,
                    "case_id": case_id,
                    "condition": condition,
                    "model": config[provider]["model"],
                    "status": "unresolved",
                    "exact_match": False,
                }
            )
    return {"metrics": aggregate_live(recorded, config, len(cases)), "stops": stopped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--tokenizer-command", help="JSON array of executable and arguments")
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--pricing",
        type=Path,
        help="Explicit provider/model/rates/source/verified_at JSON",
    )
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--budget-usd", type=Decimal, default=Decimal("0.50"))
    parser.add_argument("--max-output-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    if (
        not args.budget_usd.is_finite()
        or args.budget_usd <= 0
        or not 64 <= args.max_output_tokens <= 2048
    ):
        parser.error("Require a positive finite budget and 64..2048 output tokens")
    cases = generate_cases(args.cases, args.seed)
    config = json.loads(args.pricing.read_text(encoding="utf-8")) if args.pricing else {}
    tokenizer = json.loads(args.tokenizer_command) if args.tokenizer_command else None
    report = {
        "offline": offline(cases, tokenizer),
        "historical_baseline": audit_baseline(args.baseline) if args.baseline else None,
        "live": None,
        "targets": "UNVERIFIED",
    }
    if report["offline"]["lossless_canonical_byte_roundtrips"] != len(cases) or report["offline"][
        "deterministic_projection_exact_matches"
    ] != len(cases):
        raise RuntimeError("Offline integrity gate failed")
    manifest = {
        "suite": "lstp-optimization-v1",
        "corpus_sha256": corpus_sha256(cases),
        "seed": args.seed,
        "cases": len(cases),
        "pricing": config,
        "max_output_tokens": args.max_output_tokens,
        "budget_usd": str(args.budget_usd),
        "source_sha256": {
            **{
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in Path(__file__).parent.glob("*.py")
            },
            **{
                "lstp/" + path.relative_to(Path(lstp.__file__).parent).as_posix(): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in Path(lstp.__file__).parent.rglob("*.py")
            },
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("Run identity changed; use a new output directory")
    _atomic_json(manifest_path, manifest)
    if args.live:
        if not args.preflight or not config:
            parser.error("Live mode requires pricing and passing preflight files")
        preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
        if not all(
            preflight.get(key) is True
            for key in (
                "schema",
                "pytest",
                "interop_js",
                "slai_adapter",
                "release_runner",
            )
        ):
            raise RuntimeError("All five preflight checks must pass before live calls")
        for provider, rates in config.items():
            if (
                provider not in {"openai", "gemini"}
                or not rates.get("source")
                or not rates.get("verified_at")
                or not rates.get("model")
            ):
                raise ValueError(
                    "Require supported provider, explicit model, price source and verification date"
                )
        lock = args.output / "live-run.lock"
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            report["live"] = run_live(
                cases,
                args.output,
                config,
                args.budget_usd,
                args.max_output_tokens,
                args.seed,
            )
        finally:
            os.close(descriptor)
            lock.unlink()
        report["targets"] = compare_targets(report["live"]["metrics"])
    # Decimal values are written as exact strings, never rounded into a false target pass.
    (args.output / "optimization_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(str(args.output / "optimization_report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
