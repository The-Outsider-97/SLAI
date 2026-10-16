"""Fail-closed quality and leakage gate for generated LANTRA curriculum records."""

from __future__ import annotations

import collections
import importlib
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .enrichment_contracts import (
    CurriculumCompatibilityError,
    CurriculumConfig,
    CurriculumQualityError,
    GateDecision,
    SUPPORTED_TASKS,
    VALID_SPLITS,
    sha256_payload,
)


class TrainingQualityGate:
    """Validate trainer compatibility, source isolation, reasoning state, and duplicates."""

    def __init__(
        self,
        config: CurriculumConfig,
        document_splits: Mapping[str, str],
    ) -> None:
        self.config = config
        self.document_splits = dict(document_splits)
        try:
            trainer = importlib.import_module("train_lantra")
        except Exception as exc:
            raise CurriculumCompatibilityError(
                "Unable to import train_lantra for output schema validation."
            ) from exc
        normalizer = getattr(trainer, "normalize_record", None)
        if not callable(normalizer):
            raise CurriculumCompatibilityError(
                "train_lantra.normalize_record() is required to validate curriculum compatibility."
            )
        self._normalize_record = normalizer
        self._seen_payloads: Dict[str, Tuple[str, str]] = {}
        self.accepted = collections.Counter()
        self.rejected = collections.Counter()

    def evaluate(self, record: Mapping[str, Any]) -> GateDecision:
        try:
            task = str(record.get("task", "")).strip().lower()
            split = str(record.get("split", "")).strip().lower()
            if task not in SUPPORTED_TASKS:
                return self._reject("unsupported_task")
            if split not in VALID_SPLITS:
                return self._reject("missing_or_invalid_split")

            metadata = record.get("metadata", {})
            if not isinstance(metadata, Mapping):
                return self._reject("metadata_not_mapping")

            if not self._source_isolation_valid(split, metadata):
                return self._reject("cross_split_source_leakage")
            if not self._reasoning_gate_valid(metadata):
                return self._reject("reasoning_gold_not_fully_validated")
            if not self._record_size_valid(record):
                return self._reject("record_too_large")

            # The active LANTRA trainer remains the schema authority.
            normalized = self._normalize_record(record, "<agent-enriched>", 1)
            dedupe_key = sha256_payload(
                {
                    "task": normalized.task,
                    "source": normalized.source,
                    "target": normalized.target,
                    "anchor": normalized.anchor,
                    "positive": normalized.positive,
                    "negative": normalized.negative,
                }
            )
            previous = self._seen_payloads.get(dedupe_key)
            if previous is not None:
                previous_id, previous_split = previous
                if previous_split != split:
                    raise CurriculumQualityError(
                        "Exact curriculum duplicate crossed split boundaries: "
                        f"{previous_id} ({previous_split}) vs {record.get('id')} ({split})."
                    )
                return self._reject("exact_duplicate", dedupe_key=dedupe_key)

            self._seen_payloads[dedupe_key] = (str(record.get("id", "")), split)
            self.accepted[(str(metadata.get("curriculum_phase", "unknown")), task, split)] += 1
            return GateDecision(True, "accepted", dedupe_key)
        except CurriculumQualityError:
            raise
        except Exception as exc:
            if self.config.fail_on_agent_error:
                raise CurriculumQualityError(
                    f"Curriculum record validation failed: {type(exc).__name__}: {exc}"
                ) from exc
            return self._reject("schema_validation_error")

    def _source_isolation_valid(self, split: str, metadata: Mapping[str, Any]) -> bool:
        source_ids = metadata.get("source_document_ids")
        if source_ids is None:
            source_id = metadata.get("source_document_id")
            source_ids = [] if source_id in (None, "") else [source_id]
        if isinstance(source_ids, str):
            source_ids = [source_ids]
        if not isinstance(source_ids, Sequence) or isinstance(source_ids, (bytes, bytearray)):
            return False
        if not source_ids:
            return False
        for document_id in source_ids:
            canonical_id = str(document_id)
            expected = self.document_splits.get(canonical_id)
            if expected is None or expected != split:
                return False
        return True

    @staticmethod
    def _reasoning_gate_valid(metadata: Mapping[str, Any]) -> bool:
        reasoning = metadata.get("reasoning")
        if not isinstance(reasoning, Mapping):
            return True
        if bool(reasoning.get("required_gold", False)):
            return bool(
                reasoning.get("decision") == "valid"
                and reasoning.get("combined_valid") is True
                and reasoning.get("validation_status") == "success"
                and reasoning.get("validation_complete") is True
                and reasoning.get("has_conflict") is False
            )
        if bool(reasoning.get("required_unsupported", False)):
            return bool(
                reasoning.get("decision") == "invalid"
                and reasoning.get("combined_valid") is False
                and reasoning.get("validation_status") == "success"
                and reasoning.get("validation_complete") is True
            )
        return True

    def _record_size_valid(self, record: Mapping[str, Any]) -> bool:
        limit = int(self.config.max_record_chars)
        text_fields = (
            "input",
            "prompt",
            "source",
            "target",
            "label",
            "anchor",
            "positive",
            "negative",
            "query",
        )
        total = 0
        for key in text_fields:
            value = record.get(key)
            if isinstance(value, str):
                total += len(value)
        return total <= limit

    def _reject(self, reason: str, *, dedupe_key: Optional[str] = None) -> GateDecision:
        self.rejected[reason] += 1
        return GateDecision(False, reason, dedupe_key)

    def summary(self) -> Dict[str, Any]:
        accepted_rows = [
            {
                "phase": phase,
                "task": task,
                "split": split,
                "count": int(count),
            }
            for (phase, task, split), count in sorted(self.accepted.items())
        ]
        return {
            "accepted": accepted_rows,
            "accepted_total": int(sum(self.accepted.values())),
            "rejected": {str(key): int(value) for key, value in sorted(self.rejected.items())},
            "rejected_total": int(sum(self.rejected.values())),
            "unique_payloads": len(self._seen_payloads),
        }


__all__ = ["TrainingQualityGate"]
