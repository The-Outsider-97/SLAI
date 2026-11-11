"""Regression and offline smoke coverage for the opt-in Alignment models.

The labels generated here are deliberately synthetic fixtures, NOT evidence of
model alignment quality or suitable production training material.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from src.agents.alignment.assessment import (
    AlignmentAssessor, AlignmentDimension, AlignmentEvidence, AlignmentStatus,
)
from src.agents.alignment.learned_models import (
    AlignmentModelBundle, RepresentationEncoder, ValueAlignmentScorer,
)
from src.agents.alignment.runtime import AlignmentSubsystem
from src.agents.alignment.train_models import grouped_splits, read_reviewed_pairs, train


def _fixtures(tmp_path):
    data = tmp_path / "reviewed_pairs.jsonl"
    metadata = tmp_path / "dataset_metadata.json"
    examples = []
    for number in range(12):
        for label in (0, 1):
            examples.append({
                "group_id": f"group-{number}",
                "source_id": f"synthetic-fixture-{number}-{label}",
                "value_text": f"value group {number}: consent fairness accountability",
                "action_text": (
                    f"action group {number}: request consent and disclose details {label}"
                    if label else f"action group {number}: disregard consent and hide details"
                ),
                "label": label,
            })
    data.write_text("\n".join(json.dumps(example) for example in examples) + "\n", encoding="utf-8")
    metadata.write_text(json.dumps({
        "license": "test-only synthetic fixture",
        "annotation_guidelines": "synthetic labels for testing program control flow only",
        "source_description": "test-generated; not independently reviewed",
    }), encoding="utf-8")
    return data, metadata


def test_encoder_and_scorer_are_trainable_and_repeatable():
    encoder = RepresentationEncoder().cpu().eval()
    scorer = ValueAlignmentScorer(encoder).eval()
    assert any(p.requires_grad for p in encoder.trainable_parameters())
    values = ["Honor user consent and privacy."]
    actions = ["Ask before sharing personal information."]
    with torch.inference_mode():
        first = encoder(values)
        second = encoder(values)
        scored = scorer(values, actions)
    assert first.shape == (1, encoder.embedding_dim)
    assert torch.allclose(first, second)
    assert scored.shape == (1,)
    assert 0.0 <= scored.item() <= 1.0
    with pytest.raises(ValueError):
        encoder([""])
    with pytest.raises(ValueError):
        scorer(values, [])
    features = encoder.backbone._text_features(values)
    assert torch.equal(features, encoder.backbone._text_features(values))


def test_train_checkpoint_integrity_and_active_approval(tmp_path):
    data, metadata = _fixtures(tmp_path)
    out = tmp_path / "model"
    result = train(data=data, metadata=metadata, out=out,
                   seed=12, encoder_epochs=1, scorer_epochs=2, batch_size=8)
    assert set(result) >= {"validation", "test", "representation_test"}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["approved"] is False
    with pytest.raises(ValueError, match="approved"):
        AlignmentModelBundle.load(out, require_approved=True)
    bundle = AlignmentModelBundle.load(out)
    answer = bundle.score(value_text="respect privacy", action_text="ask consent")
    assert 0.0 <= answer["score"] <= 1.0
    assert answer["calibrated"] is False

    # An explicit review decision is necessary; changing a model checkpoint
    # without matching its digest must never be silently accepted.
    manifest["approved"] = True
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    approved = AlignmentModelBundle.load(out, require_approved=True)
    assert approved.score(value_text="respect privacy", action_text="ask consent")["approved"] is True
    manifest["artifacts"]["representation_encoder.pt"] = "0" * 64
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        AlignmentModelBundle.load(out)


def test_duplicate_pairs_across_groups_rejected(tmp_path):
    data, _ = _fixtures(tmp_path)
    rows = read_reviewed_pairs(data)
    rows[1]["group_id"] = "other-group"
    rows[1]["value_text"] = rows[0]["value_text"]
    rows[1]["action_text"] = rows[0]["action_text"]
    with pytest.raises(ValueError, match="Duplicate"):
        _write_bad_rows(tmp_path, rows)


def _write_bad_rows(tmp_path, rows):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    return read_reviewed_pairs(path)


def test_mandatory_constraint_cannot_be_averaged_away():
    assessor = AlignmentAssessor()
    evidence = [
        AlignmentEvidence(dimension=AlignmentDimension.USER_INTENT, source="a", score=1.0, required=True),
        AlignmentEvidence(dimension=AlignmentDimension.TASK_OBJECTIVE, source="b", score=1.0, required=True),
        AlignmentEvidence(dimension=AlignmentDimension.CONSTRAINT, source="c", score=0.0, required=True),
        AlignmentEvidence(dimension=AlignmentDimension.PREFERENCE, source="model", score=1.0),
        AlignmentEvidence(dimension=AlignmentDimension.TRAJECTORY, source="agent", score=1.0),
        AlignmentEvidence(dimension=AlignmentDimension.ENVIRONMENT, source="agent", score=1.0),
    ]
    assessment = assessor.assess(evidence, record_trajectory=False)
    assert assessment.status == AlignmentStatus.MISALIGNED
    assert assessment.requires_review


class _FakeBundle:
    def score(self, *, value_text, action_text):
        return {
            "score": 0.91, "representation_similarity": 0.4,
            "model_version": "1.0.0", "feature_schema": "value_text.blake2b_v1",
            "calibrated": False, "approved": True,
        }


@pytest.mark.parametrize("mode,expected", [("shadow", 0), ("advisory", 0), ("active", 1)])
def test_learned_model_mode_respects_evidence_boundary(mode, expected):
    runtime = AlignmentSubsystem(
        alignment_memory=object(),
        runtime_config={
            "learned_model_mode": mode,
            "run_statistical_diagnostics": False,
            "memory_logging_enabled": False,
            "enforce_alignment_constraints": False,
        },
    )
    runtime._learned_bundle = _FakeBundle()
    evidence, report = runtime.collect_evidence({
        "enable_ethics_check": False,
        "learned_alignment": {"value_text": "protect privacy", "action_text": "ask consent"},
    })
    assert len(evidence) == expected
    assert report["learned_alignment"]["mode"] == mode
    if expected:
        assert evidence[0].dimension == AlignmentDimension.PREFERENCE
        assert evidence[0].required is False


def test_learned_model_invalid_input_degrades_without_positive_evidence():
    runtime = AlignmentSubsystem(
        alignment_memory=object(),
        runtime_config={"learned_model_mode": "active", "run_statistical_diagnostics": False, "enforce_alignment_constraints": False},
    )
    evidence, report = runtime.collect_evidence({
        "enable_ethics_check": False,
        "learned_alignment": {"value_text": None, "action_text": "act"},
    })
    assert evidence == []
    assert report["learned_alignment"]["status"] == "invalid_input"


def test_untrusted_task_cannot_disable_dynamic_constraints():
    class Constraints:
        def _check_dynamic_constraints(self, context):
            return {"details": ["reviewed"], "violations": ["mandatory-requirement"]}

    runtime = AlignmentSubsystem(
        alignment_memory=object(),
        components={"ethical_constraints": Constraints()},
        runtime_config={"enforce_alignment_constraints": True,
                        "run_statistical_diagnostics": False,
                        "memory_logging_enabled": False},
    )
    evidence, report = runtime.collect_evidence({
        "enable_ethics_check": False,  # untrusted input must be ignored
        "evaluate_alignment_constraints": False,
        "action_context": {"action": "unauthorized"},
    })
    assert report["constraints"]["available"] is True
    assert any(item.dimension == AlignmentDimension.CONSTRAINT and
               item.score == 0.0 and item.required for item in evidence)
