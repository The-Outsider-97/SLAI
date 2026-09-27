"""Canonical LANTRA source adapter for curriculum construction.

The adapter reuses :mod:`src.training.lantra_corpus` as the single corpus I/O
authority.  It intentionally does *not* import ``train_lantra.py``: a curriculum
build must not load the language model, optimizer, or PyTorch merely to discover,
extract, hash, and segment source documents.
"""

from __future__ import annotations

import collections

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .enrichment_contracts import (
    CurriculumConfig,
    CurriculumError,
    SourceDocument,
    SourceSegment,
    normalized_text_hash,
    sha256_payload,
    stable_unit_interval,
)
from .lantra_corpus import (
    discover_raw_text_files,
    extract_raw_documents,
    segment_raw_document,
    sha256_file,
)


class CanonicalLantraSourceAdapter:
    """Translate canonical LANTRA corpus records into curriculum contracts."""

    def __init__(self, config: CurriculumConfig) -> None:
        self.config = config

    def discover_files(self) -> List[Path]:
        return list(discover_raw_text_files(self.config.source_paths))

    def inventory_files(self, files: Sequence[Path]) -> Tuple[Dict[str, str], ...]:
        inventory: List[Dict[str, str]] = []
        for path in files:
            try:
                digest = str(sha256_file(path))
            except OSError as exc:
                raise CurriculumError(f"Could not hash source file {path}: {exc}") from exc
            inventory.append({"path": str(path), "sha256": digest})
        return tuple(sorted(inventory, key=lambda item: item["path"]))

    def source_fingerprint(self, inventory: Sequence[Mapping[str, str]]) -> str:
        return sha256_payload({"source_files": list(inventory)})

    def extract_documents(
        self,
        files: Sequence[Path],
        *,
        inventory: Optional[Sequence[Mapping[str, str]]] = None,
    ) -> Tuple[SourceDocument, ...]:
        file_hashes = {
            str(item["path"]): str(item["sha256"])
            for item in (inventory or self.inventory_files(files))
        }

        documents: List[SourceDocument] = []
        seen_normalized: Dict[str, SourceDocument] = {}

        for path in files:
            source_hash = file_hashes.get(str(path)) or str(sha256_file(path))
            try:
                for raw in extract_raw_documents(path, source_sha256=source_hash):
                    text = str(raw.text or "").strip()
                    if len(text) < self.config.min_document_chars:
                        continue

                    normalized_hash = normalized_text_hash(text)
                    document_id = sha256_payload(
                        {
                            "source_sha256": str(raw.source_sha256),
                            "logical_index": int(raw.logical_index),
                            "normalized_text_sha256": normalized_hash,
                        }
                    )[:24]
                    candidate = SourceDocument(
                        document_id=document_id,
                        source_path=str(raw.source_path),
                        source_type=str(raw.source_type),
                        source_sha256=str(raw.source_sha256),
                        normalized_text_sha256=normalized_hash,
                        text=text,
                        title=(str(raw.title).strip() if raw.title else None),
                        extractor=str(raw.extractor),
                        logical_index=int(raw.logical_index),
                        metadata=dict(raw.metadata or {}),
                    )

                    existing = seen_normalized.get(normalized_hash)
                    if existing is None:
                        seen_normalized[normalized_hash] = candidate
                        documents.append(candidate)
                        continue

                    # Ownership must not depend on filesystem traversal order.
                    current_key = (
                        existing.source_sha256,
                        existing.source_path,
                        existing.logical_index,
                    )
                    candidate_key = (
                        candidate.source_sha256,
                        candidate.source_path,
                        candidate.logical_index,
                    )
                    if candidate_key < current_key:
                        documents[documents.index(existing)] = candidate
                        seen_normalized[normalized_hash] = candidate
            except Exception as exc:
                if self.config.fail_on_agent_error:
                    raise CurriculumError(
                        f"Canonical LANTRA extraction failed for {path}: {exc}"
                    ) from exc
                continue

        ordered = sorted(
            documents,
            key=lambda item: (
                stable_unit_interval(
                    item.normalized_text_sha256,
                    self.config.seed + 7,
                ),
                item.normalized_text_sha256,
                item.document_id,
            ),
        )
        if self.config.max_documents > 0:
            ordered = ordered[: self.config.max_documents]
        return tuple(ordered)

    def assign_document_splits(
        self,
        documents: Sequence[SourceDocument],
    ) -> Tuple[SourceDocument, ...]:
        if not documents:
            return ()

        ordered = sorted(
            documents,
            key=lambda item: (
                stable_unit_interval(item.normalized_text_sha256, self.config.seed),
                item.normalized_text_sha256,
                item.document_id,
            ),
        )
        count = len(ordered)
        n_validation = (
            0
            if self.config.validation_fraction <= 0.0
            else max(1, int(round(count * self.config.validation_fraction)))
        )
        n_test = (
            0
            if self.config.test_fraction <= 0.0
            else max(1, int(round(count * self.config.test_fraction)))
        )
        while n_validation + n_test >= count:
            if n_test >= n_validation and n_test > 0:
                n_test -= 1
            elif n_validation > 0:
                n_validation -= 1
            else:
                break

        split_by_id: Dict[str, str] = {}
        for index, document in enumerate(ordered):
            if index < n_test:
                split = "test"
            elif index < n_test + n_validation:
                split = "validation"
            else:
                split = "train"
            split_by_id[document.document_id] = split

        return tuple(
            sorted(
                (
                    document.with_split(split_by_id[document.document_id])
                    for document in documents
                ),
                key=lambda item: item.document_id,
            )
        )

    def segment_documents(
        self,
        documents: Sequence[SourceDocument],
    ) -> Tuple[SourceSegment, ...]:
        """Chunk after split assignment and globally deduplicate exact segments."""

        owner_by_hash: Dict[str, SourceSegment] = {}
        per_document_count: Dict[str, int] = collections.defaultdict(int)

        for document in documents:
            if document.split is None:
                raise CurriculumError(
                    "Documents must be split before segment construction."
                )
            try:
                chunks = segment_raw_document(
                    document.text,
                    min_chars=self.config.segment_min_chars,
                    chunk_chars=self.config.segment_chunk_chars,
                )
                for segment_index, text in enumerate(chunks):
                    if (
                        per_document_count[document.document_id]
                        >= self.config.max_segments_per_document
                    ):
                        break

                    canonical_hash = normalized_text_hash(text)
                    segment_id = sha256_payload(
                        {
                            "document_id": document.document_id,
                            "segment_index": int(segment_index),
                            "normalized_text_sha256": canonical_hash,
                        }
                    )[:24]
                    candidate = SourceSegment(
                        segment_id=segment_id,
                        document_id=document.document_id,
                        split=document.split,
                        segment_index=int(segment_index),
                        text=str(text).strip(),
                        normalized_text_sha256=canonical_hash,
                        title=document.title,
                        source_path=document.source_path,
                    )
                    existing = owner_by_hash.get(canonical_hash)
                    if existing is None:
                        owner_by_hash[canonical_hash] = candidate
                        per_document_count[document.document_id] += 1
                        continue

                    # The same text must never survive in multiple partitions.
                    current_key = stable_unit_interval(
                        existing.document_id + canonical_hash,
                        self.config.seed + 13,
                    )
                    candidate_key = stable_unit_interval(
                        candidate.document_id + canonical_hash,
                        self.config.seed + 13,
                    )
                    if candidate_key < current_key:
                        per_document_count[existing.document_id] = max(
                            0,
                            per_document_count[existing.document_id] - 1,
                        )
                        owner_by_hash[canonical_hash] = candidate
                        per_document_count[candidate.document_id] += 1
            except Exception as exc:
                if self.config.fail_on_agent_error:
                    raise CurriculumError(
                        "LANTRA segment construction failed for document "
                        f"{document.document_id}: {exc}"
                    ) from exc

        retained = sorted(
            owner_by_hash.values(),
            key=lambda item: (
                stable_unit_interval(
                    item.normalized_text_sha256,
                    self.config.seed + 71,
                ),
                item.normalized_text_sha256,
                item.segment_id,
            ),
        )
        if self.config.max_segments > 0:
            retained = retained[: self.config.max_segments]
        return tuple(retained)

    @staticmethod
    def segment_counts_by_split(
        segments: Sequence[SourceSegment],
    ) -> Dict[str, int]:
        counter = collections.Counter(segment.split for segment in segments)
        return {
            name: int(counter.get(name, 0))
            for name in ("train", "validation", "test")
        }


__all__ = ["CanonicalLantraSourceAdapter"]
