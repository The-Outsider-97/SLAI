"""Shared corpus hygiene utilities for the LANTRA data pipeline.

The module is deliberately independent of SLAI agents.  It owns deterministic text
normalisation, content fingerprints, a persisted near-duplicate index, collision-safe
filesystem names, and atomic JSON state writes used by both cleaner.py and
slai_scraper.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import unicodedata

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def normalize_text_for_dedup(text: str) -> str:
    """Normalize formatting noise without erasing lexical differences."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    value = value.casefold()
    chars: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        if char == "'" or category[0] in {"L", "N"}:
            chars.append(char)
        elif char.isspace() or category[0] in {"P", "S", "Z", "C"}:
            chars.append(" ")
        else:
            chars.append(char)
    return " ".join("".join(chars).split())


def meaningful_words(text: str) -> list[str]:
    value = unicodedata.normalize("NFKC", str(text or "")).replace("\u2019", "'")
    words: list[str] = []
    for match in _WORD_RE.finditer(value):
        token = match.group(0).strip("'").casefold()
        if token and any(char.isalpha() for char in token):
            words.append(token)
    return words


def normalized_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text_for_dedup(text).encode("utf-8")).hexdigest()


def _stable_hash64(value: str, seed: int) -> int:
    key = int(seed).to_bytes(8, "little", signed=False)
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8, key=key).digest(),
        "big",
    )


def shingle_hashes(text: str, width: int = 5) -> set[int]:
    tokens = normalize_text_for_dedup(text).split()
    if not tokens:
        return set()
    width = max(1, min(int(width), len(tokens)))
    return {
        _stable_hash64("\x1f".join(tokens[index:index + width]), 0)
        for index in range(0, len(tokens) - width + 1)
    }


def minhash_signature(
    text: str,
    *,
    shingle_size: int = 5,
    permutations: int = 64,
) -> tuple[int, ...]:
    shingles = shingle_hashes(text, shingle_size)
    if not shingles:
        return ()
    result: list[int] = []
    for seed in range(max(1, int(permutations))):
        seed_bytes = seed.to_bytes(8, "little", signed=False)
        minimum = min(
            int.from_bytes(
                hashlib.blake2b(
                    value.to_bytes(8, "big"),
                    digest_size=8,
                    key=seed_bytes,
                ).digest(),
                "big",
            )
            for value in shingles
        )
        result.append(minimum)
    return tuple(result)


def signature_similarity(left: Sequence[int], right: Sequence[int]) -> float:
    if not left or not right:
        return 0.0
    count = min(len(left), len(right))
    if count <= 0:
        return 0.0
    return sum(1 for index in range(count) if left[index] == right[index]) / count


def exact_shingle_similarity(
    left: str,
    right: str,
    *,
    shingle_size: int = 5,
) -> float:
    """Symmetric Jaccard overlap of normalized token shingles."""
    first = shingle_hashes(left, shingle_size)
    second = shingle_hashes(right, shingle_size)
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def sanitize_windows_filename(title: str, *, fallback: str = "") -> str:
    value = unicodedata.normalize("NFKC", str(title or "")).strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", value)
    value = " ".join(value.split()).rstrip(" .")
    if not value:
        value = str(fallback or "").strip()
    if not value:
        raise ValueError("A usable title is required for a training filename.")
    stem = value.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED:
        value = "_" + value
    return value[:180].rstrip(" .")


def collision_safe_path(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    suffix = candidate.suffix
    stem = candidate.stem
    index = 2
    while True:
        candidate = directory / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_text_lossy(path: Path) -> str:
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class DuplicateMatch:
    path: str
    similarity: float
    normalized_sha256: str
    source_kind: str


class NearDuplicateIndex:
    """SQLite-backed MinHash/LSH index with exact verification of candidates.

    MinHash/LSH is only a candidate generator.  A document is declared a
    duplicate only after the actual normalized text shingles reach the configured
    similarity threshold.  This keeps the 0.85 contract meaningful while avoiding
    an O(N^2) all-pairs scan.
    """

    SCHEMA_VERSION = 2

    def __init__(
        self,
        database: Path,
        *,
        threshold: float = 0.85,
        shingle_size: int = 5,
        permutations: int = 64,
        bands: int = 16,
    ) -> None:
        self.database = Path(database)
        self.threshold = float(threshold)
        self.shingle_size = max(1, int(shingle_size))
        self.permutations = max(8, int(permutations))
        self.bands = max(1, int(bands))
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1].")
        if self.permutations % self.bands:
            raise ValueError("permutations must be divisible by bands.")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database)
        self.connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents(
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    source_kind TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    normalized_sha256 TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    signature TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_hash
                    ON documents(normalized_sha256);
                CREATE TABLE IF NOT EXISTS bands(
                    band_key TEXT NOT NULL,
                    document_id INTEGER NOT NULL,
                    PRIMARY KEY(band_key, document_id),
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_bands_key ON bands(band_key);
                """
            )
            row = self.connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            current = int(row[0]) if row else None
            if current not in (None, self.SCHEMA_VERSION):
                self.connection.execute("DELETE FROM bands")
                self.connection.execute("DELETE FROM documents")
            self.connection.execute(
                "INSERT INTO meta(key,value) VALUES('schema_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(self.SCHEMA_VERSION),),
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "NearDuplicateIndex":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @staticmethod
    def _path_key(path: Path) -> str:
        return str(path.resolve())

    def _band_keys(self, signature: Sequence[int]) -> Iterator[str]:
        if not signature:
            return
        rows = self.permutations // self.bands
        for band in range(self.bands):
            values = signature[band * rows:(band + 1) * rows]
            payload = ",".join(str(value) for value in values).encode("ascii")
            digest = hashlib.blake2b(payload, digest_size=12).hexdigest()
            yield f"{band}:{digest}"

    @staticmethod
    def _signature_text(signature: Sequence[int]) -> str:
        return ",".join(str(value) for value in signature)

    @staticmethod
    def _signature_from_text(value: str) -> tuple[int, ...]:
        return tuple(int(item) for item in value.split(",") if item)

    def remove_path(self, path: Path) -> None:
        key = self._path_key(path)
        row = self.connection.execute(
            "SELECT id FROM documents WHERE path=?", (key,)
        ).fetchone()
        if row is None:
            return
        with self.connection:
            self.connection.execute(
                "DELETE FROM bands WHERE document_id=?", (int(row["id"]),)
            )
            self.connection.execute("DELETE FROM documents WHERE id=?", (int(row["id"]),))

    def add_path(self, path: Path, *, source_kind: str) -> bool:
        try:
            stat = path.stat()
        except OSError:
            raise
        key = self._path_key(path)
        existing = self.connection.execute(
            "SELECT id,size,mtime_ns,source_kind FROM documents WHERE path=?", (key,)
        ).fetchone()
        if (
            existing is not None
            and int(existing["size"]) == int(stat.st_size)
            and int(existing["mtime_ns"]) == int(stat.st_mtime_ns)
            and str(existing["source_kind"]) == str(source_kind)
        ):
            return False
        text = read_text_lossy(path)
        self.add_text(
            path=path,
            text=text,
            source_kind=source_kind,
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
        )
        return True

    def add_text(
        self,
        *,
        path: Path,
        text: str,
        source_kind: str,
        size: int | None = None,
        mtime_ns: int | None = None,
    ) -> int:
        key = self._path_key(path)
        signature = minhash_signature(
            text,
            shingle_size=self.shingle_size,
            permutations=self.permutations,
        )
        normalized_hash = normalized_sha256(text)
        token_count = len(normalize_text_for_dedup(text).split())
        if size is None or mtime_ns is None:
            try:
                stat = path.stat()
                size = int(stat.st_size) if size is None else int(size)
                mtime_ns = int(stat.st_mtime_ns) if mtime_ns is None else int(mtime_ns)
            except OSError:
                size = len(text.encode("utf-8")) if size is None else int(size)
                mtime_ns = 0 if mtime_ns is None else int(mtime_ns)
        old = self.connection.execute(
            "SELECT id FROM documents WHERE path=?", (key,)
        ).fetchone()
        with self.connection:
            if old is not None:
                doc_id = int(old["id"])
                self.connection.execute("DELETE FROM bands WHERE document_id=?", (doc_id,))
                self.connection.execute(
                    "UPDATE documents SET source_kind=?,size=?,mtime_ns=?,normalized_sha256=?,"
                    "token_count=?,signature=? WHERE id=?",
                    (
                        str(source_kind), int(size), int(mtime_ns), normalized_hash,
                        token_count, self._signature_text(signature), doc_id,
                    ),
                )
            else:
                cursor = self.connection.execute(
                    "INSERT INTO documents(path,source_kind,size,mtime_ns,normalized_sha256,"
                    "token_count,signature) VALUES(?,?,?,?,?,?,?)",
                    (
                        key, str(source_kind), int(size), int(mtime_ns), normalized_hash,
                        token_count, self._signature_text(signature),
                    ),
                )
                doc_id = int(cursor.lastrowid)
            for band_key in self._band_keys(signature):
                self.connection.execute(
                    "INSERT OR IGNORE INTO bands(band_key,document_id) VALUES(?,?)",
                    (band_key, doc_id),
                )
        return doc_id

    def _candidate_rows(
        self,
        text: str,
        *,
        source_kinds: Iterable[str] | None = None,
    ) -> list[sqlite3.Row]:
        normalized_hash = normalized_sha256(text)
        source_filter = tuple(str(item) for item in (source_kinds or ()))
        params: list[Any] = [normalized_hash]
        sql = "SELECT * FROM documents WHERE normalized_sha256=?"
        if source_filter:
            placeholders = ",".join("?" for _ in source_filter)
            sql += f" AND source_kind IN ({placeholders})"
            params.extend(source_filter)
        exact = self.connection.execute(sql, tuple(params)).fetchall()
        by_id = {int(row["id"]): row for row in exact}

        signature = minhash_signature(
            text,
            shingle_size=self.shingle_size,
            permutations=self.permutations,
        )
        if not signature:
            return list(by_id.values())
        band_keys = list(self._band_keys(signature))
        if not band_keys:
            return list(by_id.values())
        placeholders = ",".join("?" for _ in band_keys)
        query = (
            "SELECT DISTINCT d.* FROM documents d "
            "JOIN bands b ON b.document_id=d.id "
            f"WHERE b.band_key IN ({placeholders})"
        )
        query_params: list[Any] = list(band_keys)
        if source_filter:
            source_placeholders = ",".join("?" for _ in source_filter)
            query += f" AND d.source_kind IN ({source_placeholders})"
            query_params.extend(source_filter)
        for row in self.connection.execute(query, tuple(query_params)).fetchall():
            by_id[int(row["id"])] = row
        return list(by_id.values())

    def find_duplicate(
        self,
        text: str,
        *,
        source_kinds: Iterable[str] | None = None,
    ) -> DuplicateMatch | None:
        normalized = normalize_text_for_dedup(text)
        if not normalized:
            return None
        token_count = len(normalized.split())
        query_signature = minhash_signature(
            text,
            shingle_size=self.shingle_size,
            permutations=self.permutations,
        )
        normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        best: DuplicateMatch | None = None
        for row in self._candidate_rows(text, source_kinds=source_kinds):
            candidate_count = int(row["token_count"])
            if token_count and candidate_count:
                ratio = min(token_count, candidate_count) / max(token_count, candidate_count)
                if ratio < 0.70:
                    continue
            if str(row["normalized_sha256"]) == normalized_hash:
                similarity = 1.0
            else:
                indexed_signature = self._signature_from_text(str(row["signature"]))
                # Avoid disk reads for obvious false-positive LSH collisions.
                if signature_similarity(query_signature, indexed_signature) < max(0.50, self.threshold - 0.20):
                    continue
                candidate_path = Path(str(row["path"]))
                try:
                    candidate_text = read_text_lossy(candidate_path)
                except OSError:
                    continue
                similarity = exact_shingle_similarity(
                    text,
                    candidate_text,
                    shingle_size=self.shingle_size,
                )
            if similarity >= self.threshold and (
                best is None or similarity > best.similarity
            ):
                best = DuplicateMatch(
                    path=str(row["path"]),
                    similarity=float(similarity),
                    normalized_sha256=str(row["normalized_sha256"]),
                    source_kind=str(row["source_kind"]),
                )
        return best

    def index_paths(
        self,
        paths: Iterable[Path],
        *,
        source_kind: str,
    ) -> tuple[int, int]:
        scanned = 0
        refreshed = 0
        for path in paths:
            scanned += 1
            if self.add_path(path, source_kind=source_kind):
                refreshed += 1
        return scanned, refreshed


__all__ = [
    "DuplicateMatch",
    "NearDuplicateIndex",
    "atomic_write_json",
    "collision_safe_path",
    "exact_shingle_similarity",
    "meaningful_words",
    "minhash_signature",
    "normalize_text_for_dedup",
    "normalized_sha256",
    "read_text_lossy",
    "sanitize_windows_filename",
    "signature_similarity",
]
