from __future__ import annotations

import json
import os
import time

from pathlib import Path

import pytest

import cleaner
import slai_scraper

from src.training.corpus_dedup import (
    NearDuplicateIndex,
    collision_safe_path,
    exact_shingle_similarity,
    sanitize_windows_filename,
)
from src.training.enrichment_contracts import SUPPORTED_TASKS, SourceDocument
from src.training.lantra_resume import (
    compatible_base_config,
    load_optimizer_resume_state,
    select_resume_checkpoint,
    summarize_checkpoint,
)
from src.training.lantra_vocabulary_analysis import (
    VocabularyTracker,
    dictionary_eligible,
    iter_word_tokens,
    normalize_dictionary_token,
)
from src.training.llm_providers import (
    LLMProvider,
    LLMProviderResponseError,
    LLMProviderUnavailable,
    ProviderPool,
    ProviderSpec,
    parse_json_object,
    providers_from_config,
)
from src.training.supervised_llm_builder import (
    build_supervised_dataset,
    validate_record,
)


def _prose(seed: str, count: int = 80) -> str:
    return " ".join(
        f"{seed} sentence number {index} explains a meaningful historical scientific concept clearly."
        for index in range(count)
    )


def _document(index: int, split: str) -> SourceDocument:
    text = _prose(f"topic{index}", 12)
    return SourceDocument(
        document_id=f"doc-{split}-{index}",
        source_path=f"{split}-{index}.txt",
        source_type="txt",
        source_sha256=f"source-{split}-{index}",
        normalized_text_sha256=f"text-{split}-{index}",
        text=text,
        title=f"Document {split} {index}",
        extractor="unit_test",
        logical_index=index,
        metadata={},
        split=split,
    )


def test_recursive_file_discovery(tmp_path: Path) -> None:
    historical = tmp_path / "africa" / "training_text" / "a" / "b"
    historical.mkdir(parents=True)
    (historical / "one.txt").write_text("one", encoding="utf-8")
    (historical / "ignore.md").write_text("ignore", encoding="utf-8")
    found = cleaner.discover_txt_files([tmp_path / "africa" / "training_text"])
    assert found == [historical / "one.txt"]


def test_duplicate_threshold_and_nonduplicate_preservation(tmp_path: Path) -> None:
    original = " ".join(f"token{i}" for i in range(200))
    near = " ".join(
        [*(f"token{i}" for i in range(190)), *(f"replacement{i}" for i in range(10))]
    )
    unrelated = _prose("unrelated", 30)
    assert exact_shingle_similarity(original, near) >= 0.85
    assert exact_shingle_similarity(original, unrelated) < 0.85

    source = tmp_path / "historical.txt"
    source.write_text(original, encoding="utf-8")
    with NearDuplicateIndex(tmp_path / "index.sqlite3", threshold=0.85, bands=16) as index:
        index.add_path(source, source_kind="historical")
        assert index.find_duplicate(near, source_kinds=("historical",)) is not None
        assert index.find_duplicate(unrelated, source_kinds=("historical",)) is None


def test_collision_safe_bin_move_never_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("new", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "source.txt").write_text("existing", encoding="utf-8")
    moved = cleaner._move_duplicate(source, bin_dir)
    assert moved.name == "source (2).txt"
    assert moved.read_text(encoding="utf-8") == "new"
    assert (bin_dir / "source.txt").read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("The Roman Republic", "The Roman Republic"),
        ('A: B / C? * D', "A B C D"),
        ("CON", "_CON"),
    ],
)
def test_windows_title_filename_sanitization(title: str, expected: str) -> None:
    assert sanitize_windows_filename(title) == expected


def test_minimum_fifty_meaningful_words_gate() -> None:
    short = " ".join(["meaningful"] * 49)
    accepted = " ".join(["meaningful"] * 50)
    assert len(slai_scraper.meaningful_words(short)) == 49
    assert len(slai_scraper.meaningful_words(accepted)) == 50


def test_scraper_dedupes_current_session(tmp_path: Path) -> None:
    text = _prose("session", 20)
    first = tmp_path / "first.txt"
    first.write_text(text, encoding="utf-8")
    with NearDuplicateIndex(tmp_path / "index.sqlite3", threshold=0.85, bands=16) as index:
        index.add_path(first, source_kind="session")
        match = index.find_duplicate(text, source_kinds=("session",))
        assert match is not None
        assert match.similarity == pytest.approx(1.0)


def test_target_accounting_is_accepted_written_bytes() -> None:
    state = slai_scraper._new_state(500_000_000)
    state["accepted_bytes"] = 499_999_999
    assert int(state["accepted_bytes"]) < int(state["target_bytes"])
    state["accepted_bytes"] = 500_000_000
    assert int(state["accepted_bytes"]) >= int(state["target_bytes"])


def test_scraper_resume_state_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = slai_scraper._new_state(500_000_000)
    state["accepted_bytes"] = 12_345
    state["processed_source_keys"] = ["wikipedia:page:1"]
    slai_scraper._save_state(path, state)
    loaded = slai_scraper._load_state(path, 500_000_000, reset=False)
    assert loaded["accepted_bytes"] == 12_345
    assert loaded["processed_source_keys"] == ["wikipedia:page:1"]


def test_dictionary_matching_excludes_numbers_and_symbols(tmp_path: Path) -> None:
    dictionary = {"the", "cat", "walked", "home"}
    tracker_path = tmp_path / "vocab.sqlite3"
    with VocabularyTracker(
        tracker_path,
        dictionary_words=dictionary,
        manifest_sha256="manifest",
    ) as tracker:
        tracker.record_document("doc", "The cat walked home 12345 !!! zxqv.")
        summary = tracker.summary()
    assert summary["dictionary_eligible_words"] == 5
    assert summary["dictionary_matched_words"] == 4
    assert summary["dictionary_unmatched_words"] == 1
    assert summary["english_dictionary_percentage"] == pytest.approx(80.0)
    assert not dictionary_eligible(normalize_dictionary_token("12345"))
    assert list(iter_word_tokens("123 !!!")) == ["123"]


def test_unique_word_count_is_distinct_vocabulary(tmp_path: Path) -> None:
    with VocabularyTracker(
        tmp_path / "vocab.sqlite3",
        dictionary_words={"alpha", "beta"},
        manifest_sha256="manifest",
    ) as tracker:
        tracker.record_document("one", "alpha alpha beta")
        tracker.record_document("two", "beta gamma")
        summary = tracker.summary()
    assert summary["total_unique_words"] == 3
    assert summary["dictionary_eligible_words"] == 5


class _Provider(LLMProvider):
    def __init__(self, name: str, behavior: str):
        super().__init__(
            ProviderSpec(name=name, model="unit", api_key="not-secret", timeout_seconds=1)
        )
        self.name = name
        self.behavior = behavior

    def complete(self, *, system: str, prompt: str) -> str:
        del system, prompt
        if self.behavior == "timeout":
            raise LLMProviderUnavailable("timeout")
        if self.behavior == "malformed":
            return "not json"
        return '{"examples": []}'


def test_llm_provider_failure_falls_back() -> None:
    pool = ProviderPool(
        [_Provider("broken", "timeout"), _Provider("working", "ok")],
        retry_limit=0,
        backoff_seconds=0,
        sleep=lambda _: None,
    )
    result = pool.complete(system="system", prompt="prompt")
    assert result is not None
    assert result.provider == "working"


def test_malformed_llm_response_rejected() -> None:
    with pytest.raises(LLMProviderResponseError):
        parse_json_object("not json")


def test_api_timeout_isolated_by_provider_pool() -> None:
    pool = ProviderPool(
        [_Provider("timeout", "timeout")],
        retry_limit=1,
        backoff_seconds=0,
        sleep=lambda _: None,
    )
    assert pool.complete(system="system", prompt="prompt") is None


def test_unavailable_api_key_skips_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANTRA_TEST_MISSING_KEY", raising=False)
    providers = providers_from_config(
        {
            "providers": [
                {
                    "name": "openai",
                    "enabled": True,
                    "api_key_env": "LANTRA_TEST_MISSING_KEY",
                    "default_model": "unit-model",
                }
            ]
        }
    )
    assert providers == []


def test_validation_rejects_un_grounded_record() -> None:
    record = {
        "task": "summarization",
        "split": "train",
        "input": "completely unrelated input words",
        "target": "invented response without source overlap",
        "metadata": {},
    }
    with pytest.raises(Exception):
        validate_record(record, source_text="astronomy telescope planet orbit stellar observation")


def test_supervised_builder_covers_all_existing_tasks_and_resumes(tmp_path: Path) -> None:
    docs = [
        *(_document(index, "train") for index in range(3)),
        _document(100, "validation"),
        _document(200, "test"),
    ]
    cfg = {
        "supervised_generation": {
            "enabled": True,
            "output_dir": str(tmp_path / "generated"),
            "state_path": str(tmp_path / "state.json"),
            "manifest_path": str(tmp_path / "manifest.json"),
            "cache_path": str(tmp_path / "cache.sqlite3"),
            "minimum_train_examples_per_task": 2,
            "minimum_validation_examples_per_task": 1,
            "minimum_test_examples_per_task": 1,
            "external_llm_enabled": False,
            "providers": [],
        }
    }
    first = build_supervised_dataset(
        docs,
        pipeline_config=cfg,
        quality_agent=None,
        enable_external_llm=False,
    )
    for task in SUPPORTED_TASKS:
        assert first.coverage[task]["valid"]["train"] >= 2
        assert first.coverage[task]["valid"]["validation"] >= 1
        assert first.coverage[task]["valid"]["test"] >= 1

    second = build_supervised_dataset(
        docs,
        pipeline_config=cfg,
        quality_agent=None,
        enable_external_llm=False,
    )
    assert second.resumed_records == 28
    assert second.deterministic_fallback_accepted == 0


def _checkpoint_payload(
    *,
    stage: str,
    status: str,
    step: int,
    saved_at: str,
    optimizer: bool = True,
    d_model: int = 256,
) -> dict:
    payload = {
        "base_config": {"d_model": d_model, "nhead": 8},
        "metadata": {
            "stage": stage,
            "status": status,
            "epoch": 2,
            "global_optimizer_step": step,
            "saved_at": saved_at,
            "dataset_fingerprint": "dataset-a",
        },
    }
    if optimizer:
        payload["optimizer_state_dict"] = {"state": {}, "param_groups": []}
    return payload


def test_checkpoint_discovery_uses_metadata_not_filename(tmp_path: Path) -> None:
    older = tmp_path / "lantra_latest.pt"
    newer = tmp_path / "odd_name.pt"
    older.write_bytes(b"a")
    newer.write_bytes(b"b")
    payloads = {
        str(older): _checkpoint_payload(
            stage="supervised", status="latest", step=9000, saved_at="2026-09-29T10:00:00Z"
        ),
        str(newer): _checkpoint_payload(
            stage="raw_text_denoising_pretraining_interval",
            status="interval",
            step=5000,
            saved_at="2026-10-01T10:00:00Z",
        ),
    }

    def loader(path: Path):
        return payloads[str(path)]

    selected, inspected = select_resume_checkpoint(
        tmp_path,
        continual_state={"last_checkpoint": str(newer)},
        loader=loader,
        expected_base_config={"d_model": 256, "nhead": 8},
    )
    assert selected is not None
    assert selected.path == newer
    assert len(inspected) >= 2


def test_checkpoint_compatibility_detects_architecture_change(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"x")
    summary = summarize_checkpoint(
        path,
        _checkpoint_payload(
            stage="supervised",
            status="latest",
            step=10,
            saved_at="2026-10-01T10:00:00Z",
            d_model=256,
        ),
    )
    compatible, mismatches = compatible_base_config(
        summary,
        expected={"d_model": 384, "nhead": 8},
    )
    assert not compatible
    assert any(item.startswith("d_model:") for item in mismatches)


def test_training_resume_state_requires_matching_stage_and_dataset() -> None:
    payload = _checkpoint_payload(
        stage="supervised",
        status="interrupted",
        step=1234,
        saved_at="2026-10-01T10:00:00Z",
    )
    resume = load_optimizer_resume_state(
        payload,
        expected_stage="supervised",
        dataset_fingerprint="dataset-a",
    )
    assert resume is not None
    assert resume["global_optimizer_step"] == 1234
    assert load_optimizer_resume_state(
        payload,
        expected_stage="phase_2b_retrieval",
        dataset_fingerprint="dataset-a",
    ) is None
    assert load_optimizer_resume_state(
        payload,
        expected_stage="supervised",
        dataset_fingerprint="different-dataset",
    ) is None
