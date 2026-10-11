"""Inspect or smoke-test the shared LANTRA provider pool and apply human review.

Curriculum generation remains owned by build_lantra_curriculum; this is diagnostics
and explicit review, never another trainer or inference router.
"""
from __future__ import annotations

import argparse
import json

from pathlib import Path

from src.utils.configuration import DEFAULT_CONFIG_REPOSITORY
from src.training.llm_providers import ProviderPool, providers_from_config
from src.training.supervised_llm_builder import _append_jsonl, _existing_records, _resolve_path, validate_record
from src.training.enrichment_contracts import sha256_payload
from logs.logger import PrettyPrinter


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "smoke", "review"))
    parser.add_argument("--config", default="src/training/configs/lantra_pipeline.yaml")
    parser.add_argument("--candidate-id")
    parser.add_argument("--decision", choices=("accept", "reject"))
    parser.add_argument("--corrected-record", type=Path)
    parser.add_argument("--reviewer")
    args = parser.parse_args(argv)
    cfg = DEFAULT_CONFIG_REPOSITORY.load(Path(args.config))["lantra_pipeline"]["supervised_generation"]
    if args.command == "review":
        if not args.candidate_id or not args.decision or not args.reviewer:
            parser.error("review requires --candidate-id, --decision and --reviewer")
        review_path = _resolve_path(cfg["review_path"])
        candidates = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        candidate = next((row for row in candidates if row["candidate_id"] == args.candidate_id), None)
        if candidate is None:
            parser.error("candidate ID not found")
        decisions_path = review_path.with_name("decisions.jsonl")
        if decisions_path.exists():
            previous = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if any(row["candidate_id"] == args.candidate_id for row in previous):
                parser.error("candidate already reviewed; create a new explicit review revision")
        if args.decision == "accept":
            record = (json.loads(args.corrected_record.read_text(encoding="utf-8"))
                      if args.corrected_record else dict(candidate["candidate"]))
            record["split"] = "train"
            record["metadata"] = {"origin": "validated_teacher_supervision", "teacher_generated": True,
                "teacher_provider": candidate["provider"], "teacher_model": candidate["model"],
                "human_verified": True, "reviewer": args.reviewer, "validation_method": "human_review",
                "source_document_ids": [candidate["source_id"]], "dataset_version": candidate["dataset_version"]}
            record.pop("id", None)
            record = validate_record(record, source_text=candidate["source"], human_verified=True)
            state = json.loads(_resolve_path(cfg["state_path"]).read_text(encoding="utf-8"))
            if state.get("dataset_version") != candidate["dataset_version"]:
                parser.error("review belongs to a different dataset version")
            output = _resolve_path(cfg["output_dir"])
            existing, _ = _existing_records(output)
            if record["id"] not in existing:
                _append_jsonl(output / f"lantra_teacher_{record['task']}.jsonl", record)
        _append_jsonl(decisions_path, {"candidate_id": args.candidate_id, "decision": args.decision, "reviewer": args.reviewer})
        PrettyPrinter().pretty("LANTRA REVIEW", {"candidate_id": args.candidate_id, "decision": args.decision}, "info")
        return 0
    providers = providers_from_config(cfg)
    with ProviderPool(providers, state_path=_resolve_path(cfg["scheduler_state_path"]),
                      limits=cfg.get("provider_limits", {}), retry_limit=0) as pool:
        report = {"configured": [p.name for p in providers], "metrics": pool.metrics()}
        if args.command == "smoke":
            # At most one pass through eligible providers. Never print teacher content.
            outcomes = []
            for index in range(len(providers)):
                result = pool.complete(system="Return strict JSON only.",
                                       prompt=json.dumps({"instruction": "Return {\"ok\":true}", "rotation_probe": index}))
                outcomes.append({"provider": result.provider if result else None,
                                 "status": "cached" if result and result.cached else "passed" if result else "unavailable"})
            report.update(outcomes=outcomes, metrics=pool.metrics())
            report["note"] = "Successful requests verify endpoint/model/authentication; cached results are not fresh API verification."
        PrettyPrinter().pretty("LANTRA PROVIDERS", report, "info")
        return 0 if providers and (args.command != "smoke" or all(r["status"] == "passed" for r in report["outcomes"])) else 2


if __name__ == "__main__":
    raise SystemExit(main())