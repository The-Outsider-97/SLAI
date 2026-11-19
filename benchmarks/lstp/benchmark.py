"""Read-only LSTP v0.1 paired-format benchmark across OpenAI, Groq and Gemini.

Run through run_benchmark.ps1 from Windows PowerShell. Live requests are opt-in.
Every model gets the same semantic cases represented in three formats.
This experiment measures extraction fidelity, not general reasoning quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from benchmarks.lstp.cases import Case, corpus_sha256, generate_cases
from benchmarks.lstp.providers import (
    KEY_NAMES,
    MODEL_PRICES,
    ProviderError,
    choose_models,
    estimated_usd,
    query,
)

PROVIDERS = ("openai", "groq", "gemini")
FORMATS = ("natural_language", "ordinary_json", "canonical_lstp")
FIELDS = (
    "action",
    "resource",
    "project",
    "stage",
    "permission_mode",
    "output_format",
    "source",
)
CAPS = {"openai": Decimal("5.00"), "groq": Decimal(0), "gemini": Decimal("5.00")}
SYSTEM = (
    "You are evaluating a synthetic, read-only communication benchmark. "
    "Extract the intended fields from the user message, which may be ordinary "
    "language, generic JSON or canonical LSTP v0.1 JSON. Do not execute tools or "
    "change anything. Return ONLY one JSON object with exactly these string "
    "properties: action, resource, project, stage, permission_mode, "
    "output_format, source. The action must be inspect, summarize, compare, "
    "verify or describe. A read-only instruction corresponds to RO; offering "
    "suggestions without modification corresponds to SUGGEST; simulating a "
    "change without saving corresponds to PREVIEW. Preserve literal identifiers "
    "and output format labels. Do not invent missing facts."
)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path.name}")
    return value


def _jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    output: list[dict] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise TypeError(f"Non-object JSONL row {line_number} in {path.name}")
        output.append(value)
    return output


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _suite_digest() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in ("benchmark.py", "cases.py", "providers.py", "smoke_integration.py"):
        payload = (root / name).read_bytes()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name.encode("ascii"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def parse_answer(raw: str) -> dict[str, str] | None:
    """Accept exact JSON objects only; markup and extra fields count as failures."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        return None
    if any(not isinstance(value[name], str) for name in FIELDS):
        return None
    return value


def score(raw: str, case: Case) -> dict[str, object]:
    answer = parse_answer(raw)
    matches = {field: answer is not None and answer[field] == case.gold[field] for field in FIELDS}
    return {
        "json_valid": answer is not None,
        "exact_match": bool(answer is not None and all(matches.values())),
        "field_matches": matches,
        "field_accuracy": sum(matches.values()) / len(FIELDS),
    }


def _wilson(passes: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z = 1.96
    p = passes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(center - margin, 4), round(center + margin, 4)]


class BudgetLedger:
    """Conservative local reservations protect separate paid-provider budgets.

    Crashed or failed requests retain their maximum reservation as spent. The
    ledger intentionally overestimates cost; it is not a provider billing API.
    """

    def __init__(self, path: Path, *, caps: dict[str, Decimal] | None = None):
        self.path = path
        self.caps = dict(CAPS if caps is None else caps)
        if not self.caps or any(not value.is_finite() or value < 0 for value in self.caps.values()):
            raise ValueError("Budget caps must be finite and non-negative")
        self.state = _load_json(path) or {
            "caps_usd": {p: str(v) for p, v in self.caps.items()},
            "debited_usd": {p: "0" for p in self.caps},
            "pending_usd": {},
        }
        if self.state["caps_usd"] != {p: str(v) for p, v in self.caps.items()}:
            raise RuntimeError("Saved budget caps differ from the approved caps")

    def remaining(self, provider: str) -> Decimal:
        debited = Decimal(self.state["debited_usd"][provider])
        reserved = sum(
            (
                Decimal(value["usd"])
                for value in self.state["pending_usd"].values()
                if value["provider"] == provider
            ),
            Decimal(0),
        )
        return self.caps[provider] - debited - reserved

    @staticmethod
    def upper_bound(provider: str, model: str, user: str, *, max_output_tokens: int) -> Decimal:
        # Very conservative input budget for short ASCII/UTF-8 text prompts;
        # x3 additional guard includes billing/pricing deviations and overhead.
        upper_input = 4 * len((SYSTEM + user).encode("utf-8")) + 2048
        return estimated_usd(provider, model, upper_input, max_output_tokens) * 3

    def reserve(self, job_id: str, provider: str, bound: Decimal) -> bool:
        if job_id in self.state["pending_usd"]:
            raise RuntimeError("Unresolved reservation exists for this job")
        if bound > self.remaining(provider):
            return False
        self.state["pending_usd"][job_id] = {"provider": provider, "usd": str(bound)}
        _atomic_json(self.path, self.state)
        return True

    def finalize(self, job_id: str, observed: Decimal | None) -> Decimal:
        reservation = self.state["pending_usd"].pop(job_id)
        provider = reservation["provider"]
        maximum = Decimal(reservation["usd"])
        # Missing/failed usage spends the entire reservation. Successful usage
        # receives a 3x conservative billing buffer, capped by reservation.
        debit = maximum if observed is None else max(observed, min(maximum, observed * 3))
        self.state["debited_usd"][provider] = str(
            Decimal(self.state["debited_usd"][provider]) + debit
        )
        _atomic_json(self.path, self.state)
        return debit


def _plan(cases: list[Case], selected: dict[str, str], seed: int) -> list[tuple[Case, str, str]]:
    """Randomize paired three-format blocks, retaining within-model case pairing."""
    rng = random.Random(seed + 53)
    blocks = [(case, provider) for case in cases for provider in selected]
    rng.shuffle(blocks)
    jobs: list[tuple[Case, str, str]] = []
    for case, provider in blocks:
        formats = list(FORMATS)
        rng.shuffle(formats)
        jobs.extend((case, provider, condition) for condition in formats)
    return jobs


def _aggregate(rows: list[dict], selected: dict[str, str], cases: list[Case]) -> dict:
    result: dict[str, object] = {}
    by_condition: dict[tuple[str, str], dict[str, dict]] = {}
    for provider, model in selected.items():
        for condition in FORMATS:
            group = [
                r
                for r in rows
                if r.get("provider") == provider
                and r.get("condition") == condition
                and r.get("status") == "ok"
            ]
            valid = sum(bool(r["score"]["json_valid"]) for r in group)
            exact = sum(bool(r["score"]["exact_match"]) for r in group)
            times = [r["latency_seconds"] for r in group]
            result[f"{provider}/{condition}"] = {
                "model": model,
                "completed": len(group),
                "expected": len(cases),
                "valid_json_rate": round(valid / len(group), 4) if group else None,
                "exact_match_rate": round(exact / len(group), 4) if group else None,
                "exact_match_wilson95": _wilson(exact, len(group)),
                "mean_field_accuracy": round(
                    statistics.mean(r["score"]["field_accuracy"] for r in group), 4
                )
                if group
                else None,
                "median_latency_seconds": round(statistics.median(times), 4) if times else None,
                "total_input_tokens": sum(r["input_tokens"] for r in group),
                "total_output_tokens": sum(r["output_tokens"] for r in group),
                "nominal_estimated_usd": round(sum(r["nominal_usd"] for r in group), 8),
            }
            by_condition[(provider, condition)] = {r["case_id"]: r for r in group}

        paired: dict[str, object] = {}
        baseline = by_condition[(provider, "ordinary_json")]
        for condition in ("natural_language", "canonical_lstp"):
            other = by_condition[(provider, condition)]
            ids = sorted(set(baseline) & set(other))
            deltas = [
                int(other[i]["score"]["exact_match"]) - int(baseline[i]["score"]["exact_match"])
                for i in ids
            ]
            bootstrap: list[float] = []
            if len(deltas) >= 2:
                rng = random.Random(20261117)
                for _ in range(2000):
                    bootstrap.append(statistics.mean(rng.choices(deltas, k=len(deltas))))
                bootstrap.sort()
            paired[condition + "_minus_json"] = {
                "paired_cases": len(ids),
                "exact_accuracy_delta": round(statistics.mean(deltas), 4) if ids else None,
                "paired_bootstrap95": [
                    round(bootstrap[49], 4),
                    round(bootstrap[1949], 4),
                ]
                if bootstrap
                else None,
            }
        result[f"{provider}/paired_differences"] = paired
    return result


def generate_report(
    output: Path,
    cases: list[Case],
    selected: dict[str, str],
    ledger: BudgetLedger,
    manifest: dict,
    stop_reasons: dict[str, str] | None = None,
) -> dict:
    rows = _jsonl_rows(output / "results.jsonl")
    expected = len(cases) * len(selected or PROVIDERS) * len(FORMATS)
    complete = sum(r.get("status") == "ok" for r in rows)
    aggregations = _aggregate(rows, selected, cases)
    stops = stop_reasons or {}
    status = (
        "MEASURED"
        if complete == expected
        else "BUDGET_LIMITED"
        if stops and all(reason == "budget" for reason in stops.values())
        else "PARTIAL_OR_PENDING"
    )
    latest_rows = {str(r["job_id"]): r for r in rows}
    report = {
        "report_kind": status,
        "completed_successful_requests": complete,
        "recorded_attempts": len(rows),
        "planned_requests": expected,
        "stop_reasons": stops,
        "study_scope": "synthetic paired semantic-field extraction; no real tool execution",
        "validity_limits": [
            "Synthetic templates do not measure real-world reasoning or hallucinations.",
            "Provider/model differences are not causal effects of LSTP.",
            "LSTP benefits must be interpreted as within-model paired differences.",
            "Nominal API cost estimates are not provider invoices.",
            "The corpus does not test full agent-to-agent action execution.",
            "Multiple examples share templates and resources; "
            "effective independent sample size is lower than case count.",
            "A missing or failed API response is excluded from conditional accuracy, "
            "never scored as a success.",
        ],
        "manifest": manifest,
        "budget_ledger": ledger.state,
        "aggregates": aggregations,
        "errors": [
            {"job_id": r.get("job_id"), "error": r.get("error")}
            for r in latest_rows.values()
            if r.get("status") == "error"
        ],
    }
    _atomic_json(output / "final_report.json", report)
    markdown = [
        "# SLAI — LSTP v0.1 benchmark report",
        "",
        f"**Status:** {report['report_kind']}",
        f"**LSTP commit:** `{manifest['lstp_commit']}`",
        f"**SLAI commit:** `{manifest['slai_commit']}`",
        f"**Benchmark code SHA-256:** `{manifest['benchmark_suite_sha256']}`",
        f"**Corpus:** {len(cases)} paired cases (`{manifest['corpus_sha256']}`)",
        f"**Requests:** {complete} successful of {expected} planned",
        f"**Provider stops:** `{stops}`",
        "",
        "## Offline representation sizes (UTF-8 bytes, not model tokens)",
        "",
        "| Format | Minimum | Median | Maximum |",
        "|---|---:|---:|---:|",
    ]
    for name in FORMATS:
        size = manifest["representation_sizes"][name]
        markdown.append(
            f"| {name} | {size['min_bytes']} | {size['median_bytes']} | {size['max_bytes']} |"
        )
    markdown += [
        "",
        "## Model results (only measured API responses)",
        "",
        (
            "| Provider / format | Valid JSON | Exact match | 95% Wilson interval | "
            "Median latency | Input tokens | Output tokens | Nominal USD |"
        ),
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for provider in selected:
        for condition in FORMATS:
            row = aggregations[f"{provider}/{condition}"]

            def pct(v: object) -> str:
                return "—" if v is None else f"{float(v) * 100:.1f}%"

            wilson = row["exact_match_wilson95"]
            interval = "—" if wilson is None else f"{pct(wilson[0])}–{pct(wilson[1])}"
            markdown.append(
                f"| {provider} / {condition} | {pct(row['valid_json_rate'])} | "
                f"{pct(row['exact_match_rate'])} ({row['completed']}/{row['expected']}) | "
                f"{interval} | {row['median_latency_seconds'] or '—'} | "
                f"{row['total_input_tokens']} | {row['total_output_tokens']} | "
                f"${row['nominal_estimated_usd']:.6f} |"
            )
    markdown += [
        "",
        "## Paired LSTP versus ordinary JSON",
        "",
        "| Provider | Paired cases | Exact-match difference | Bootstrap 95% interval |",
        "|---|---:|---:|---|",
    ]
    for provider in selected:
        pair = aggregations[f"{provider}/paired_differences"]["canonical_lstp_minus_json"]
        delta = pair["exact_accuracy_delta"]
        interval = pair["paired_bootstrap95"]
        delta_text = (
            (str(round(delta * 100, 2)) + " percentage points") if delta is not None else "—"
        )
        interval_text = "—"
        if interval:
            interval_text = f"{round(interval[0] * 100, 2)} to {round(interval[1] * 100, 2)} pp"
        markdown.append(f"| {provider} | {pair['paired_cases']} | {delta_text} | {interval_text} |")
    markdown += [
        "",
        "## Interpretation",
        "",
        (
            "This report only displays observed measurements. It does not establish "
            "that LSTP improves general reasoning, safety, speed, or cost. Compare "
            "each model against its own ordinary-JSON condition; inspect paired "
            "differences in `final_report.json`. Missing rows are not scored as successes."
        ),
        "",
        "## Budget ledger (conservative local guard; not an invoice)",
        "",
    ]
    for provider in PROVIDERS:
        markdown.append(
            f"- {provider}: ${ledger.state['debited_usd'][provider]} conservatively "
            f"debited of ${CAPS[provider]} approved"
        )
    markdown += [
        "",
        "## Offline gates",
        "",
        f"- Schema: `{manifest['preflight'].get('schema')}`",
        f"- Python conformance/tests: `{manifest['preflight'].get('pytest')}`",
        f"- Independent JS interoperability: `{manifest['preflight'].get('interop_js')}`",
        f"- SLAI integration adapter: `{manifest['preflight'].get('slai_adapter')}`",
        f"- Full engineering release runner: `{manifest['preflight'].get('release_runner')}`",
        "",
        "## Limitations",
        "",
    ]
    markdown += [f"- {item}" for item in report["validity_limits"]]
    (output / "final_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lstp-root", type=Path, required=True)
    parser.add_argument("--slai-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--confirm-free-groq", action="store_true")
    parser.add_argument("--confirm-pricing", action="store_true")
    args = parser.parse_args(argv)
    if not 64 <= args.max_output_tokens <= 2048:
        parser.error("max-output-tokens must be between 64 and 2048")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    preflight = _load_json(output / "preflight.json")
    cases = generate_cases(args.cases, args.seed)
    digest = corpus_sha256(cases)
    representation_sizes = {
        condition: {
            "min_bytes": min(len(getattr(case, condition).encode("utf-8")) for case in cases),
            "median_bytes": statistics.median(
                len(getattr(case, condition).encode("utf-8")) for case in cases
            ),
            "max_bytes": max(len(getattr(case, condition).encode("utf-8")) for case in cases),
        }
        for condition in FORMATS
    }
    corpus_path = output / "cases.jsonl"
    serialized = "".join(
        json.dumps(c.as_record(), sort_keys=True, ensure_ascii=False) + "\n" for c in cases
    )
    if corpus_path.exists() and corpus_path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError("Existing benchmark corpus differs; choose a new output directory")
    corpus_path.write_text(serialized, encoding="utf-8", newline="\n")

    previous_manifest = _load_json(output / "manifest.json")
    manifest = {
        "suite": "slai-lstp-paired-extraction-v1",
        "created_or_rechecked_utc": datetime.now(UTC).isoformat(),
        "lstp_commit": _git_head(args.lstp_root),
        "slai_commit": _git_head(args.slai_root),
        "benchmark_suite_sha256": _suite_digest(),
        "corpus_sha256": digest,
        "cases": len(cases),
        "seed": args.seed,
        "conditions": list(FORMATS),
        "representation_sizes": representation_sizes,
        "preflight": preflight,
        "price_sources": {
            p: {m: price.source for m, price in rates.items()} for p, rates in MODEL_PRICES.items()
        },
        "mode": "live" if args.live else "offline",
        "max_output_tokens": args.max_output_tokens,
    }
    for key in (
        "lstp_commit",
        "slai_commit",
        "benchmark_suite_sha256",
        "corpus_sha256",
        "cases",
        "seed",
        "max_output_tokens",
    ):
        if previous_manifest and previous_manifest.get(key) != manifest[key]:
            raise RuntimeError(f"Existing run changed {key}; use a new output directory")

    ledger = BudgetLedger(output / "budget_ledger.json")
    if not args.live:
        _atomic_json(output / "manifest.json", manifest)
        report = generate_report(output, cases, {}, ledger, manifest)
        print(f"Offline report: {output / 'final_report.md'} ({report['report_kind']})")
        return 0

    if not (args.confirm_free_groq and args.confirm_pricing):
        parser.error("Live runs require confirmation of free Groq tier and current prices")
    if not all(
        preflight.get(k) is True for k in ("schema", "pytest", "interop_js", "slai_adapter")
    ):
        raise RuntimeError(
            "Offline protocol/schema/Python/JS verification must pass before live calls"
        )
    if not all(os.environ.get(KEY_NAMES[p]) for p in PROVIDERS):
        raise RuntimeError("OpenAI, Groq and Gemini API keys must be configured locally")
    selected = choose_models()
    if previous_manifest and previous_manifest.get("models") not in (
        {},
        selected,
        None,
    ):
        raise RuntimeError("Model selection changed in a resumed run; use a new output directory")
    manifest["models"] = selected
    _atomic_json(output / "manifest.json", manifest)

    results_path = output / "results.jsonl"
    existing = _jsonl_rows(results_path)
    completed_ids = {str(r["job_id"]) for r in existing if r.get("status") == "ok"}
    jobs = _plan(cases, selected, args.seed)
    last_groq = 0.0
    stopped_providers: set[str] = set()
    stop_reasons: dict[str, str] = {}
    print(
        f"Live benchmark: {len(jobs)} planned requests; limits OpenAI $5, Gemini $5, Groq free.",
        flush=True,
    )
    try:
        for index, (case, provider, condition) in enumerate(jobs, start=1):
            if provider in stopped_providers:
                continue
            job_id = f"{provider}:{case.case_id}:{condition}"
            if job_id in completed_ids:
                continue
            if job_id in ledger.state["pending_usd"]:
                print(f"Unresolved prior reservation for {job_id}; manual review required.")
                continue
            model = selected[provider]
            if (index - 1) % len(FORMATS) == 0:
                # Before beginning a shuffled three-format block, reserve
                # enough *available* budget for the whole comparison.
                block = jobs[index - 1 : index - 1 + len(FORMATS)]
                projected = sum(
                    (
                        ledger.upper_bound(
                            provider,
                            model,
                            getattr(item_case, item_format),
                            max_output_tokens=args.max_output_tokens,
                        )
                        for item_case, item_provider, item_format in block
                        if item_provider == provider
                        and f"{item_provider}:{item_case.case_id}:{item_format}"
                        not in completed_ids
                    ),
                    Decimal(0),
                )
                if projected > ledger.remaining(provider):
                    print(f"STOP: {provider} budget cannot fund next full paired case.")
                    stopped_providers.add(provider)
                    stop_reasons[provider] = "budget"
                    continue
            user_message = getattr(case, condition)
            reserve = ledger.upper_bound(
                provider, model, user_message, max_output_tokens=args.max_output_tokens
            )
            if not ledger.reserve(job_id, provider, reserve):
                print(f"STOP: {provider} conservative budget cap would be exceeded.")
                stopped_providers.add(provider)
                stop_reasons[provider] = "budget"
                continue
            if provider == "groq":
                wait = 2.2 - (time.monotonic() - last_groq)
                if wait > 0:
                    time.sleep(wait)
                last_groq = time.monotonic()
            try:
                response = query(
                    provider,
                    model,
                    SYSTEM,
                    user_message,
                    max_output_tokens=args.max_output_tokens,
                )
                nominal = estimated_usd(
                    provider,
                    model,
                    int(response["input_tokens"]),
                    int(response["output_tokens"]),
                )
                debit = ledger.finalize(job_id, nominal)
                result: dict[str, object] = {
                    "job_id": job_id,
                    "case_id": case.case_id,
                    "provider": provider,
                    "model": model,
                    "condition": condition,
                    "status": "ok",
                    "raw_output": response["text"],
                    "score": score(str(response["text"]), case),
                    "input_tokens": response["input_tokens"],
                    "output_tokens": response["output_tokens"],
                    "latency_seconds": response["latency_seconds"],
                    "nominal_usd": float(nominal),
                    "conservative_debit_usd": str(debit),
                }
            except ProviderError as exc:
                debit = ledger.finalize(job_id, None)
                result = {
                    "job_id": job_id,
                    "case_id": case.case_id,
                    "provider": provider,
                    "model": model,
                    "condition": condition,
                    "status": "error",
                    "error": str(exc),
                    "conservative_debit_usd": str(debit),
                }
            _append_jsonl(results_path, result)
            if result["status"] == "ok":
                completed_ids.add(job_id)
            if index % 10 == 0:
                print(f"Progress: {len(completed_ids)}/{len(jobs)}; last provider={provider}")
            if result["status"] == "error":
                print(f"STOP {provider} after request error: {result['error']}")
                stopped_providers.add(provider)
                stop_reasons[provider] = "provider_error"
    except KeyboardInterrupt:
        print("Interrupted; reservations and completed results are retained for safe resume.")
    finally:
        report = generate_report(output, cases, selected, ledger, manifest, stop_reasons)
        print(f"Report: {output / 'final_report.md'} ({report['report_kind']})")
    return 0 if report["report_kind"] in {"MEASURED", "BUDGET_LIMITED"} else 2


if __name__ == "__main__":
    sys.exit(main())
