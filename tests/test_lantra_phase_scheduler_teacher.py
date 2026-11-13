"""Held-out teacher supervision is admitted only with verifiable source-document provenance."""
from __future__ import annotations

import sys
import unittest

from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training import lantra_phase_scheduler as sched  # noqa: E402


def teacher(example_id: str, document_ids, **extra):
    meta = {
        "origin": sched.TEACHER_ORIGIN, "synthetic": True, "teacher_generated": True,
        "teacher_validated": True, "validation_status": "accepted",
        "validation_method": sorted(sched.TEACHER_VALIDATION_METHODS)[0],
    }
    if document_ids is not None:
        meta["source_document_ids"] = list(document_ids)
    meta.update(extra)
    return SimpleNamespace(example_id=example_id, metadata=meta)


def dataset(train=(), validation=(), test=()):
    return SimpleNamespace(train=list(train), validation=list(validation), test=list(test))


class HeldOutTeacherTests(unittest.TestCase):
    def test_held_out_teacher_examples_with_sources_are_routed_to_phase_3(self) -> None:
        plan = sched.partition_lantra_dataset(dataset(
            train=[teacher("t", ["d1"])], validation=[teacher("v", ["d2"])], test=[teacher("x", ["d3"])],
        ))
        self.assertEqual((len(plan.phase_3.train), len(plan.phase_3.validation), len(plan.phase_3.test)), (1, 1, 1))

    def test_held_out_teacher_example_without_sources_is_rejected(self) -> None:
        with self.assertRaises(sched.LantraPhaseScheduleError):
            sched.partition_lantra_dataset(dataset(validation=[teacher("v", None)]))

    def test_leakage_gate_covers_teacher_examples(self) -> None:
        plan = sched.partition_lantra_dataset(dataset(
            train=[teacher("t", ["d1"])], validation=[teacher("v", ["d2"])],
        ))
        report = sched.validate_curriculum_source_splits(plan, {"d1": "train", "d2": "validation"})
        self.assertEqual(report["checked_references"], 2)
        with self.assertRaises(sched.LantraPhaseScheduleError):  # d2 is a training document in the raw corpus
            sched.validate_curriculum_source_splits(plan, {"d1": "train", "d2": "train"})

    def test_conversational_teacher_data_without_sources_is_not_a_leakage_error(self) -> None:
        plan = sched.partition_lantra_dataset(dataset(train=[teacher("dlg", None)]))
        report = sched.validate_curriculum_source_splits(plan, {"d1": "train"})
        self.assertEqual(report["checked_references"], 0)


if __name__ == "__main__":
    unittest.main()
