"""Phase-aware scheduling contracts for LANTRA training.

This module deliberately does *not* own a LANTRA model, optimizer, tokenizer,
or agent.  Its single responsibility is to partition already-normalized LANTRA
examples into the evolved training phases from their provenance metadata while
preserving the train/validation/test boundaries established by the source
pipeline.

The root ``train_lantra.py`` remains the optimization authority.  This module is
therefore safe to reuse from tests, analysis scripts, or future training entry
points without importing the language model or any Knowledge/Reasoning/
Perception implementation.
"""

from __future__ import annotations

import collections
import hashlib
import json

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


CURRICULUM_ORIGIN = "agent_enriched"
TRAINABLE_CURRICULUM_PHASES: Tuple[str, ...] = ("2a", "2b", "2c")
NON_TRAINABLE_CURRICULUM_PHASES = frozenset({"2d"})
REAL_SUPERVISED_PHASE = "3"
VALID_PHASES = frozenset((*TRAINABLE_CURRICULUM_PHASES, "2d", REAL_SUPERVISED_PHASE))
VALID_SPLITS = ("train", "validation", "test")


class LantraPhaseScheduleError(RuntimeError):
    """Raised when curriculum metadata cannot be scheduled safely."""


def _stable_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_stable_bytes(value)).hexdigest()


def _metadata(example: Any) -> Mapping[str, Any]:
    value = getattr(example, "metadata", None)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise LantraPhaseScheduleError(
            f"Training example metadata must be a mapping, got {type(value).__name__}."
        )
    return value


def _canonical_example(example: Any) -> Mapping[str, Any]:
    canonical = getattr(example, "canonical_payload", None)
    if callable(canonical):
        value = canonical()
        if isinstance(value, Mapping):
            return dict(value)
    return {
        "id": str(getattr(example, "example_id", "")),
        "task": str(getattr(example, "task", "")),
        "source": getattr(example, "source", None),
        "target": getattr(example, "target", None),
        "anchor": getattr(example, "anchor", None),
        "positive": getattr(example, "positive", None),
        "negative": getattr(example, "negative", None),
        "metadata": dict(_metadata(example)),
    }


def normalize_curriculum_phase(value: Any) -> Optional[str]:
    """Normalize a curriculum phase without inventing one.

    ``None`` means ordinary supervised data.  Only explicit metadata can route
    an example into 2A/2B/2C.  This is intentionally fail-closed for unknown
    phase labels so synthetic examples cannot silently leak into Phase 3.
    """

    if value in (None, ""):
        return None
    phase = str(value).strip().lower().replace("phase", "").replace("_", "")
    phase = phase.replace("-", "").strip()
    aliases = {
        "2a": "2a",
        "2b": "2b",
        "2c": "2c",
        "2d": "2d",
        "3": "3",
    }
    normalized = aliases.get(phase)
    if normalized is None:
        raise LantraPhaseScheduleError(f"Unsupported curriculum phase {value!r}; expected 2a, 2b, 2c, 2d, or 3.")
    return normalized


@dataclass(frozen=True)
class PhasePartition:
    """One phase-specific train/validation/test partition."""

    phase: str
    train: Tuple[Any, ...]
    validation: Tuple[Any, ...]
    test: Tuple[Any, ...]
    fingerprint: str

    @property
    def empty(self) -> bool:
        return not (self.train or self.validation or self.test)

    @property
    def active_tasks(self) -> Tuple[str, ...]:
        tasks = sorted({str(getattr(item, "task", "")) for item in self.train if getattr(item, "task", None)})
        return tuple(tasks)

    def counts(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for split_name, records in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            by_task = collections.Counter(str(getattr(item, "task", "")) for item in records)
            payload[split_name] = {
                "total": len(records),
                "tasks": dict(sorted((task, int(count)) for task, count in by_task.items() if task)),
            }
        return payload

    def summary(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "fingerprint_sha256": self.fingerprint,
            "active_tasks": list(self.active_tasks),
            "counts": self.counts(),
        }


@dataclass(frozen=True)
class LantraPhasePlan:
    """Deterministic routing plan for curriculum and real supervision."""

    phase_2a: PhasePartition
    phase_2b: PhasePartition
    phase_2c: PhasePartition
    phase_3: PhasePartition
    perception_offline_records: int
    excluded_synthetic_phase3: int
    source_fingerprint: str

    def partition(self, phase: str) -> PhasePartition:
        normalized = normalize_curriculum_phase(phase)
        if normalized == "2a":
            return self.phase_2a
        if normalized == "2b":
            return self.phase_2b
        if normalized == "2c":
            return self.phase_2c
        if normalized == "3":
            return self.phase_3
        raise LantraPhaseScheduleError(f"Phase {phase!r} is not an optimizer-training partition.")

    def summary(self) -> Dict[str, Any]:
        return {
            "source_fingerprint": self.source_fingerprint,
            "phase_2a": self.phase_2a.summary(),
            "phase_2b": self.phase_2b.summary(),
            "phase_2c": self.phase_2c.summary(),
            "phase_2d": {
                "mode": "offline_perception_filtering_only",
                "records_marked_with_2d": int(self.perception_offline_records),
                "trainable_representation_loss": False,
            },
            "phase_3": self.phase_3.summary(),
            "excluded_synthetic_phase3": int(self.excluded_synthetic_phase3),
        }


def _partition_fingerprint(phase: str, split_records: Mapping[str, Sequence[Any]]) -> str:
    return _hash_payload(
        {
            "phase": phase,
            "records": {
                split: [_canonical_example(item) for item in split_records.get(split, ())]
                for split in VALID_SPLITS
            },
        }
    )


def partition_lantra_dataset(dataset: Any) -> LantraPhasePlan:
    """Partition an existing LANTRA DatasetSplit by explicit curriculum metadata.

    Rules
    -----
    * 2A/2B/2C require an explicit ``metadata.curriculum_phase``.
    * ``origin=agent_enriched`` without a phase is rejected rather than silently
      entering real supervised Phase 3.
    * 2D is not an optimizer stage in the current contract; records may advertise
      it through ``curriculum_stages`` as evidence of offline Perception filtering.
    * Phase 3 excludes records marked ``synthetic=true``.  This keeps the final
      specialization stage genuinely supervised instead of mixing agent-derived
      curriculum back into it.
    * Existing split containers are authoritative.  This function never resplits
      an example and therefore cannot undo document-level leakage protection.
    """

    buckets: Dict[str, Dict[str, list[Any]]] = {
        phase: {split: [] for split in VALID_SPLITS}
        for phase in (*TRAINABLE_CURRICULUM_PHASES, REAL_SUPERVISED_PHASE)
    }
    perception_offline_records = 0
    excluded_synthetic_phase3 = 0

    for split_name in VALID_SPLITS:
        records = getattr(dataset, split_name, None)
        if records is None:
            raise LantraPhaseScheduleError(f"Dataset object is missing required split attribute {split_name!r}.")
        for example in records:
            meta = _metadata(example)
            phase = normalize_curriculum_phase(meta.get("curriculum_phase"))
            origin = str(meta.get("origin", "")).strip().lower()
            synthetic = bool(meta.get("synthetic", False))

            raw_stages = meta.get("curriculum_stages", ())
            if isinstance(raw_stages, Sequence) and not isinstance(raw_stages, (str, bytes, bytearray)):
                stages = {normalize_curriculum_phase(value) for value in raw_stages if value not in (None, "")}
                if "2d" in stages:
                    perception_offline_records += 1

            if phase == "2d":
                raise LantraPhaseScheduleError(
                    "Found an example with curriculum_phase='2d', but LANTRA has no validated "
                    "Perception representation-distillation objective yet.  Keep 2D as offline "
                    "filtering metadata (curriculum_stages=['2b','2d']) until that loss is implemented."
                )

            if phase in TRAINABLE_CURRICULUM_PHASES:
                buckets[phase][split_name].append(example)
                continue

            if origin == CURRICULUM_ORIGIN and phase is None:
                raise LantraPhaseScheduleError(
                    "Agent-enriched training example is missing metadata.curriculum_phase; refusing "
                    "to route synthetic curriculum into real supervised Phase 3."
                )

            # Explicit Phase 3 is accepted only for non-synthetic data.  Records
            # without a phase are ordinary supervised data and follow the same rule.
            if synthetic:
                excluded_synthetic_phase3 += 1
                continue
            buckets[REAL_SUPERVISED_PHASE][split_name].append(example)

    def make(phase: str) -> PhasePartition:
        split_records = buckets[phase]
        return PhasePartition(
            phase=phase,
            train=tuple(split_records["train"]),
            validation=tuple(split_records["validation"]),
            test=tuple(split_records["test"]),
            fingerprint=_partition_fingerprint(phase, split_records),
        )

    source_fingerprint = str(getattr(dataset, "fingerprint", "")) or _hash_payload(
        {split: [_canonical_example(item) for item in getattr(dataset, split)] for split in VALID_SPLITS}
    )
    return LantraPhasePlan(
        phase_2a=make("2a"),
        phase_2b=make("2b"),
        phase_2c=make("2c"),
        phase_3=make("3"),
        perception_offline_records=perception_offline_records,
        excluded_synthetic_phase3=excluded_synthetic_phase3,
        source_fingerprint=source_fingerprint,
    )



def validate_curriculum_source_splits(
    plan: LantraPhasePlan,
    source_split_by_document_id: Mapping[str, str],
) -> Dict[str, Any]:
    """Fail closed when curriculum provenance conflicts with a raw-corpus split.

    The evolved LANTRA pipeline can train on the same source library in both raw
    denoising and agent-derived curriculum phases.  A document must therefore
    never be ``train`` in one representation and ``validation``/``test`` in the
    other.  This validator compares explicit ``metadata.source_document_ids``
    against the canonical raw-corpus document assignments when those document IDs
    are available.

    Curriculum-only documents that are absent from ``source_split_by_document_id``
    are not errors because they were not consumed by the active raw-corpus phase.
    """

    normalized_source_splits: Dict[str, str] = {}
    for raw_id, raw_split in source_split_by_document_id.items():
        document_id = str(raw_id).strip()
        split = str(raw_split).strip().lower()
        if not document_id:
            continue
        if split not in VALID_SPLITS:
            raise LantraPhaseScheduleError(
                f"Raw source document {document_id!r} has invalid split {raw_split!r}."
            )
        previous = normalized_source_splits.get(document_id)
        if previous is not None and previous != split:
            raise LantraPhaseScheduleError(
                f"Raw source document {document_id!r} occurs in conflicting splits: "
                f"{previous!r} and {split!r}."
            )
        normalized_source_splits[document_id] = split

    checked_references = 0
    matched_documents: set[str] = set()
    missing_provenance_records = 0

    for phase in TRAINABLE_CURRICULUM_PHASES:
        partition = plan.partition(phase)
        for split_name, records in (
            ("train", partition.train),
            ("validation", partition.validation),
            ("test", partition.test),
        ):
            for example in records:
                meta = _metadata(example)
                raw_ids = meta.get("source_document_ids")
                if not isinstance(raw_ids, Sequence) or isinstance(
                    raw_ids, (str, bytes, bytearray)
                ):
                    missing_provenance_records += 1
                    raise LantraPhaseScheduleError(
                        f"Curriculum phase {phase} example "
                        f"{getattr(example, 'example_id', '<unknown>')!r} is missing "
                        "metadata.source_document_ids; cross-phase leakage cannot be verified."
                    )
                document_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
                if not document_ids:
                    missing_provenance_records += 1
                    raise LantraPhaseScheduleError(
                        f"Curriculum phase {phase} example "
                        f"{getattr(example, 'example_id', '<unknown>')!r} has empty "
                        "metadata.source_document_ids."
                    )
                for document_id in document_ids:
                    raw_split = normalized_source_splits.get(document_id)
                    if raw_split is None:
                        continue
                    checked_references += 1
                    matched_documents.add(document_id)
                    if raw_split != split_name:
                        raise LantraPhaseScheduleError(
                            "Cross-phase source leakage detected: curriculum example "
                            f"{getattr(example, 'example_id', '<unknown>')!r} in phase {phase} "
                            f"is assigned to {split_name!r}, but source document {document_id!r} "
                            f"is {raw_split!r} in raw denoising. Rebuild the curriculum with the "
                            "same split seed/fractions/source preparation as train_lantra.py."
                        )

    return {
        "status": "verified",
        "raw_documents": len(normalized_source_splits),
        "matched_documents": len(matched_documents),
        "checked_references": checked_references,
        "missing_provenance_records": missing_provenance_records,
    }

def validate_phase_coverage(
    partition: PhasePartition,
    *,
    min_train_samples_per_task: int = 1,
) -> Dict[str, Any]:
    """Validate one curriculum stage without imposing unrelated task enums."""

    if min_train_samples_per_task <= 0:
        raise LantraPhaseScheduleError("min_train_samples_per_task must be > 0.")
    if partition.empty:
        return {
            "status": "empty",
            "phase": partition.phase,
            "active_tasks": [],
            "counts": partition.counts(),
        }
    if not partition.train:
        raise LantraPhaseScheduleError(
            f"Phase {partition.phase} has held-out records but no training records."
        )

    train_counts = collections.Counter(str(getattr(item, "task", "")) for item in partition.train)
    validation_counts = collections.Counter(str(getattr(item, "task", "")) for item in partition.validation)
    test_counts = collections.Counter(str(getattr(item, "task", "")) for item in partition.test)
    active_tasks = tuple(sorted(task for task, count in train_counts.items() if task and count > 0))
    if not active_tasks:
        raise LantraPhaseScheduleError(f"Phase {partition.phase} has no trainable task labels.")

    missing_validation = [task for task in active_tasks if validation_counts.get(task, 0) <= 0]
    missing_test = [task for task in active_tasks if test_counts.get(task, 0) <= 0]
    undersized = {
        task: int(train_counts[task])
        for task in active_tasks
        if train_counts[task] < min_train_samples_per_task
    }
    if missing_validation or missing_test or undersized:
        parts = []
        if missing_validation:
            parts.append("validation missing: " + ", ".join(missing_validation))
        if missing_test:
            parts.append("test missing: " + ", ".join(missing_test))
        if undersized:
            parts.append(
                "undersized train tasks: "
                + ", ".join(f"{task}={count}" for task, count in sorted(undersized.items()))
            )
        raise LantraPhaseScheduleError(
            f"Phase {partition.phase} does not satisfy held-out coverage: " + "; ".join(parts)
        )

    return {
        "status": "ready",
        "phase": partition.phase,
        "active_tasks": list(active_tasks),
        "counts": partition.counts(),
    }


__all__ = [
    "CURRICULUM_ORIGIN",
    "TRAINABLE_CURRICULUM_PHASES",
    "NON_TRAINABLE_CURRICULUM_PHASES",
    "REAL_SUPERVISED_PHASE",
    "LantraPhaseScheduleError",
    "PhasePartition",
    "LantraPhasePlan",
    "normalize_curriculum_phase",
    "partition_lantra_dataset",
    "validate_curriculum_source_splits",
    "validate_phase_coverage",
]
