"""Metadata-driven LANTRA checkpoint discovery and stage-resume helpers."""
from __future__ import annotations

import json
import math
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


CheckpointLoader = Callable[[Path], Mapping[str, Any]]


_STAGE_RANK = {
    "glove_semantic_bootstrap": 10,
    "raw_text_denoising_pretraining_interval": 20,
    "raw_text_denoising_pretraining_latest": 22,
    "raw_text_denoising_pretraining_best": 23,
    "curriculum_2a": 30,
    "curriculum_2b": 40,
    "curriculum_2c": 50,
    "supervised": 60,
    "evolved_pretraining_final": 55,
}

_STATUS_RANK = {
    "interrupted": 5,
    "interval": 4,
    "latest": 3,
    "best": 2,
    "final_best": 1,
}


@dataclass(frozen=True)
class CheckpointSummary:
    path: Path
    base_config: Mapping[str, Any]
    metadata: Mapping[str, Any]
    has_optimizer_state: bool
    stage: str
    status: str
    epoch: int
    global_optimizer_step: int
    saved_at: str
    dataset_fingerprint: str
    corpus_fingerprint: str

    @property
    def stage_rank(self) -> int:
        return _stage_rank(self.stage)

    @property
    def resumable_optimizer_state(self) -> bool:
        return self.has_optimizer_state and self.status in {"interrupted", "interval", "latest", "best"}


def _stage_rank(stage: str) -> int:
    value = str(stage or "").strip().casefold()
    if value in _STAGE_RANK:
        return _STAGE_RANK[value]
    if value.startswith("curriculum_2a"):
        return 30
    if value.startswith("curriculum_2b"):
        return 40
    if value.startswith("curriculum_2c"):
        return 50
    if value.startswith("raw_text"):
        return 20
    if value.startswith("supervised"):
        return 60
    return 0


def _metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = payload.get("metadata", {})
    return value if isinstance(value, Mapping) else {}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def summarize_checkpoint(path: Path, payload: Mapping[str, Any]) -> CheckpointSummary:
    metadata = _metadata(payload)
    stage = str(metadata.get("stage") or metadata.get("phase") or "").strip()
    status = str(metadata.get("status") or "").strip().casefold()
    record = metadata.get("record", {})
    if not isinstance(record, Mapping):
        record = {}
    epoch = _int(metadata.get("epoch", record.get("epoch", 0)))
    step = _int(
        metadata.get(
            "global_optimizer_step",
            record.get("global_optimizer_step", 0),
        )
    )
    corpus = metadata.get("corpus", {})
    if not isinstance(corpus, Mapping):
        corpus = {}
    return CheckpointSummary(
        path=Path(path),
        base_config=(
            dict(payload.get("base_config", {}))
            if isinstance(payload.get("base_config"), Mapping)
            else {}
        ),
        metadata=dict(metadata),
        has_optimizer_state=isinstance(payload.get("optimizer_state_dict"), Mapping),
        stage=stage,
        status=status,
        epoch=epoch,
        global_optimizer_step=step,
        saved_at=str(metadata.get("saved_at") or payload.get("saved_at") or ""),
        dataset_fingerprint=str(metadata.get("dataset_fingerprint") or ""),
        corpus_fingerprint=str(
            metadata.get("corpus_fingerprint")
            or corpus.get("fingerprint_sha256")
            or ""
        ),
    )


def compatible_base_config(
    summary: CheckpointSummary,
    *,
    expected: Mapping[str, Any],
    keys: Sequence[str] = (
        "src_vocab_size",
        "tgt_vocab_size",
        "d_model",
        "nhead",
        "num_encoder_layers",
        "num_decoder_layers",
        "dim_feedforward",
        "max_position_embeddings",
        "batch_first",
    ),
) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    for key in keys:
        if key not in expected or key not in summary.base_config:
            continue
        if summary.base_config[key] != expected[key]:
            mismatches.append(
                f"{key}: checkpoint={summary.base_config[key]!r}, expected={expected[key]!r}"
            )
    return not mismatches, mismatches


def _candidate_paths(output_dir: Path, state: Mapping[str, Any]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()

    last = state.get("last_checkpoint")
    if last:
        value = Path(str(last))
        if not value.is_absolute():
            if not value.is_file():
                value = output_dir / value.name
        if value.is_file():
            result.append(value)
            seen.add(str(value.resolve()))

    preferred = (
        "lantra_interrupted.pt",
        "lantra_latest.pt",
        "lantra_best.pt",
        "lantra_raw_pretrain_interrupted.pt",
        "lantra_raw_pretrain_latest.pt",
        "lantra_raw_pretrain_best.pt",
        "lantra_supervised_interrupted.pt",
    )
    for name in preferred:
        value = output_dir / name
        if value.is_file() and str(value.resolve()) not in seen:
            result.append(value)
            seen.add(str(value.resolve()))

    dynamic = sorted(
        (item for item in output_dir.glob("lantra*.pt") if item.is_file()),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    for value in dynamic[:40]:
        key = str(value.resolve())
        if key not in seen:
            result.append(value)
            seen.add(key)
    return result


def select_resume_checkpoint(
    output_dir: Path,
    *,
    continual_state: Mapping[str, Any],
    loader: CheckpointLoader,
    expected_base_config: Mapping[str, Any] | None = None,
) -> tuple[CheckpointSummary | None, list[dict[str, Any]]]:
    """Inspect checkpoint metadata and select the most advanced compatible state.

    Ranking is based on stage, optimizer progress and status rather than a filename
    containing the word latest. A continuation state's explicit checkpoint wins
    ties because it is the last successfully committed training baseline.
    """
    expected = dict(expected_base_config or {})
    state_last = str(continual_state.get("last_checkpoint") or "")
    inspected: list[dict[str, Any]] = []
    candidates: list[CheckpointSummary] = []

    for path in _candidate_paths(Path(output_dir), continual_state):
        try:
            payload = loader(path)
            if not isinstance(payload, Mapping):
                raise ValueError("checkpoint loader did not return a mapping")
            summary = summarize_checkpoint(path, payload)
            compatible, mismatches = compatible_base_config(summary, expected=expected)
            inspected.append(
                {
                    "path": str(path),
                    "stage": summary.stage,
                    "status": summary.status,
                    "epoch": summary.epoch,
                    "global_optimizer_step": summary.global_optimizer_step,
                    "has_optimizer_state": summary.has_optimizer_state,
                    "compatible": compatible,
                    "mismatches": mismatches,
                }
            )
            if compatible:
                candidates.append(summary)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            inspected.append(
                {
                    "path": str(path),
                    "compatible": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if not candidates:
        return None, inspected

    def rank(item: CheckpointSummary) -> tuple[int, int, int, int, str, int]:
        state_bonus = int(
            bool(state_last)
            and str(item.path.resolve()) == str(Path(state_last).resolve())
            if Path(state_last).is_absolute()
            else bool(state_last) and item.path.name == Path(state_last).name
        )
        return (
            item.stage_rank,
            item.epoch,
            item.global_optimizer_step,
            _STATUS_RANK.get(item.status, 0),
            item.saved_at,
            state_bonus,
        )

    return max(candidates, key=rank), inspected


def load_optimizer_resume_state(
    checkpoint: Mapping[str, Any],
    *,
    expected_stage: str,
    dataset_fingerprint: str = "",
    corpus_fingerprint: str = "",
) -> dict[str, Any] | None:
    """Return validated stage-local optimizer continuation metadata."""
    metadata = _metadata(checkpoint)
    stage = str(metadata.get("stage") or metadata.get("phase") or "").strip()
    normalized_expected = str(expected_stage).strip()

    if normalized_expected == "raw_text_denoising_pretraining":
        if not stage.startswith("raw_text_denoising_pretraining"):
            return None
    elif stage != normalized_expected:
        return None

    if not isinstance(checkpoint.get("optimizer_state_dict"), Mapping):
        return None

    checkpoint_dataset = str(metadata.get("dataset_fingerprint") or "")
    if dataset_fingerprint and checkpoint_dataset and checkpoint_dataset != dataset_fingerprint:
        return None

    corpus = metadata.get("corpus", {})
    corpus_mapping = corpus if isinstance(corpus, Mapping) else {}
    checkpoint_corpus = str(
        metadata.get("corpus_fingerprint")
        or corpus_mapping.get("fingerprint_sha256")
        or ""
    )
    if corpus_fingerprint and checkpoint_corpus and checkpoint_corpus != corpus_fingerprint:
        return None

    record = metadata.get("record", {})
    record_mapping = record if isinstance(record, Mapping) else {}
    return {
        "optimizer_state_dict": checkpoint["optimizer_state_dict"],
        "epoch": _int(metadata.get("epoch", record_mapping.get("epoch", 0))),
        "global_optimizer_step": _int(
            metadata.get(
                "global_optimizer_step",
                record_mapping.get("global_optimizer_step", 0),
            )
        ),
        "status": str(metadata.get("status") or ""),
        "metadata": dict(metadata),
    }


__all__ = [
    "CheckpointSummary",
    "compatible_base_config",
    "load_optimizer_resume_state",
    "select_resume_checkpoint",
    "summarize_checkpoint",
]
