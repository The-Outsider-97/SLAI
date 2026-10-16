"""Deduplicate data/library against SLAI's historical LANTRA text corpora.

Run from the repository root with::

    py -m cleaner

Only duplicates found in data/library are moved. Historical scraper output is
read-only. Near-duplicate decisions use the shared persisted MinHash/LSH index
from src.training.corpus_dedup and are verified by normalized token-shingle
Jaccard similarity before a file is moved.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.training.corpus_dedup import NearDuplicateIndex, collision_safe_path
from src.utils.configuration import bind_config


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "src" / "training" / "configs" / "lantra_pipeline.yaml"
_CONFIG = bind_config(DEFAULT_CONFIG)
LOGGER = logging.getLogger("lantra_cleaner")


@dataclass
class CleanerStats:
    historical_documents_scanned: int = 0
    historical_documents_indexed: int = 0
    library_documents_scanned: int = 0
    duplicates_found: int = 0
    files_moved: int = 0
    files_skipped: int = 0
    errors: int = 0
    reclaimed_duplicate_bytes: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_config(path: Path) -> dict[str, Any]:
    loaded = _CONFIG.load(path)
    section = loaded.get("lantra_pipeline", {})
    if not isinstance(section, Mapping):
        raise ValueError("lantra_pipeline configuration must be a mapping.")
    return dict(section)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def discover_txt_files(paths: Iterable[Path]) -> list[Path]:
    """Recursively discover text files without assuming a directory depth."""
    discovered: list[Path] = []
    seen: set[str] = set()
    for root in paths:
        if not root.exists():
            continue
        items = [root] if root.is_file() else root.rglob("*.txt")
        for path in items:
            try:
                if not path.is_file() or path.suffix.casefold() != ".txt":
                    continue
                key = os.path.normcase(str(path.resolve()))
            except OSError:
                continue
            if key not in seen:
                seen.add(key)
                discovered.append(path)
    return sorted(discovered, key=lambda item: os.path.normcase(str(item)))


def _move_duplicate(source: Path, destination_root: Path) -> Path:
    target = collision_safe_path(destination_root, source.name)
    destination_root.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, target)
    except OSError:
        shutil.move(str(source), str(target))
    return target


def clean(
    *,
    config_path: Path = DEFAULT_CONFIG,
    threshold: float | None = None,
    dry_run: bool = False,
) -> CleanerStats:
    config = _load_config(config_path)
    cleaner_cfg = config.get("cleaner", {})
    if not isinstance(cleaner_cfg, Mapping):
        raise ValueError("lantra_pipeline.cleaner must be a mapping.")

    effective_threshold = float(
        threshold
        if threshold is not None
        else config.get("duplicate_similarity_threshold", 0.85)
    )
    shingle_size = int(config.get("duplicate_shingle_size", 5))
    permutations = int(config.get("duplicate_minhash_permutations", 64))
    bands = int(config.get("duplicate_lsh_bands", 8))

    library_root = _resolve(str(cleaner_cfg.get("library_path", "data/library")))
    bin_root = _resolve(str(cleaner_cfg.get("bin_path", "bin/lantra_duplicates")))
    index_path = _resolve(
        str(
            cleaner_cfg.get(
                "dedup_index_path",
                "data/processed/lantra/state/corpus_dedup.sqlite3",
            )
        )
    )
    historical_roots = [
        _resolve(str(value))
        for value in cleaner_cfg.get("historical_corpora", ())
        if str(value).strip()
    ]

    historical_files = discover_txt_files(historical_roots)
    library_files = discover_txt_files([library_root])
    stats = CleanerStats()
    started = time.perf_counter()

    LOGGER.info(
        "CLEANER | historical=%d | library=%d | threshold=%.3f",
        len(historical_files),
        len(library_files),
        effective_threshold,
    )

    with NearDuplicateIndex(
        index_path,
        threshold=effective_threshold,
        shingle_size=shingle_size,
        permutations=permutations,
        bands=bands,
    ) as index:
        for path in historical_files:
            stats.historical_documents_scanned += 1
            try:
                if index.add_path(path, source_kind="historical"):
                    stats.historical_documents_indexed += 1
            except (OSError, UnicodeError, ValueError) as exc:
                stats.errors += 1
                LOGGER.warning("HISTORICAL SKIP | %s | %s: %s", path, type(exc).__name__, exc)

        for path in library_files:
            stats.library_documents_scanned += 1
            try:
                size = path.stat().st_size
                text = path.read_text(encoding="utf-8-sig", errors="replace")
                match = index.find_duplicate(text, source_kinds=("historical",))
                if match is None:
                    stats.files_skipped += 1
                    continue

                stats.duplicates_found += 1
                if dry_run:
                    LOGGER.info(
                        "DUPLICATE DRY-RUN | %.3f | %s | historical=%s",
                        match.similarity,
                        path,
                        match.path,
                    )
                    continue

                destination = _move_duplicate(path, bin_root)
                stats.files_moved += 1
                stats.reclaimed_duplicate_bytes += int(size)
                LOGGER.info(
                    "MOVED | %.3f | %s -> %s | historical=%s",
                    match.similarity,
                    path,
                    destination,
                    match.path,
                )
            except KeyboardInterrupt:
                LOGGER.warning("Cleaner interrupted after %d library documents.", stats.library_documents_scanned)
                raise
            except (OSError, UnicodeError, ValueError, sqlite3.Error) as exc:  # type: ignore[name-defined]
                stats.errors += 1
                LOGGER.warning("LIBRARY ERROR | %s | %s: %s", path, type(exc).__name__, exc)

    stats.elapsed_seconds = time.perf_counter() - started
    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cleaner",
        description="Move data/library TXT documents that are >=85% similar to historical SLAI scraper output.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        stats = clean(
            config_path=Path(args.config),
            threshold=args.threshold,
            dry_run=bool(args.dry_run),
        )
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted safely; files already moved remain in bin and can be resumed.")
        return 130
    except (OSError, ValueError) as exc:
        LOGGER.error("Cleaner configuration/runtime failure: %s", exc)
        return 2

    print(
        "\n".join(
            [
                "LANTRA cleaner summary",
                f"  historical scanned : {stats.historical_documents_scanned:,}",
                f"  historical indexed : {stats.historical_documents_indexed:,}",
                f"  library scanned    : {stats.library_documents_scanned:,}",
                f"  duplicates found   : {stats.duplicates_found:,}",
                f"  files moved        : {stats.files_moved:,}",
                f"  files skipped      : {stats.files_skipped:,}",
                f"  errors             : {stats.errors:,}",
                f"  reclaimed bytes    : {stats.reclaimed_duplicate_bytes:,}",
                f"  elapsed seconds    : {stats.elapsed_seconds:.2f}",
            ]
        )
    )
    return 0 if stats.errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
