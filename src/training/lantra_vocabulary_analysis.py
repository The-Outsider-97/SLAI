"""Disk-backed vocabulary statistics for analyze_lantra_library.py."""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata

from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


WORD_RE = re.compile(r"\b\w+(?:['’]\w+)?\b", flags=re.UNICODE)


def normalize_dictionary_token(token: str) -> str:
    value = unicodedata.normalize("NFKC", str(token or ""))
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    value = value.strip(" '").casefold()
    return value


def iter_word_tokens(text: str) -> Iterator[str]:
    for match in WORD_RE.finditer(unicodedata.normalize("NFKC", str(text or ""))):
        token = normalize_dictionary_token(match.group(0))
        if token:
            yield token


def dictionary_eligible(token: str) -> bool:
    return bool(token) and any(char.isalpha() for char in token)


def _collect_json_words(value: Any, output: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = normalize_dictionary_token(str(key))
            if dictionary_eligible(normalized_key) and len(normalized_key) <= 80:
                output.add(normalized_key)
            _collect_json_words(item, output)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _collect_json_words(item, output)
        return
    if isinstance(value, str):
        normalized = normalize_dictionary_token(value)
        if (
            dictionary_eligible(normalized)
            and len(normalized) <= 80
            and len(normalized.split()) == 1
        ):
            output.add(normalized)


def locate_english_wordlists(repo_root: Path) -> tuple[Path, Path]:
    names = ("wordlist_en.json", "structured_wordlist_en.json")
    found: list[Path] = []
    for name in names:
        preferred = repo_root / "src" / "agents" / "language" / "library" / name
        if preferred.is_file():
            found.append(preferred)
            continue
        matches = sorted(repo_root.rglob(name))
        if not matches:
            raise FileNotFoundError(f"Could not locate {name} under {repo_root}.")
        found.append(matches[0])
    return found[0], found[1]


def load_english_dictionary(repo_root: Path) -> tuple[set[str], tuple[str, str]]:
    paths = locate_english_wordlists(repo_root)
    words: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        _collect_json_words(payload, words)
    return words, (str(paths[0]), str(paths[1]))


class VocabularyTracker:
    """Persist unique vocabulary and occurrence counts without a RAM-wide set."""

    SCHEMA = "slai.lantra.vocabulary-analysis.v1"

    def __init__(
        self,
        path: Path,
        *,
        dictionary_words: set[str],
        manifest_sha256: str,
        reset: bool = False,
    ) -> None:
        self.path = Path(path)
        self.dictionary_words = dictionary_words
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()
        current = self._meta("manifest_sha256")
        if reset or current != manifest_sha256:
            with self.connection:
                self.connection.execute("DELETE FROM vocabulary")
                self.connection.execute("DELETE FROM processed_documents")
                self.connection.execute("DELETE FROM meta")
                self.connection.execute(
                    "INSERT INTO meta(key,value) VALUES('schema',?)",
                    (self.SCHEMA,),
                )
                self.connection.execute(
                    "INSERT INTO meta(key,value) VALUES('manifest_sha256',?)",
                    (manifest_sha256,),
                )

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS vocabulary(
                    token TEXT PRIMARY KEY,
                    occurrences INTEGER NOT NULL,
                    eligible INTEGER NOT NULL,
                    matched INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_vocabulary_unmatched
                    ON vocabulary(eligible, matched, occurrences DESC);
                CREATE TABLE IF NOT EXISTS processed_documents(
                    document_hash TEXT PRIMARY KEY
                );
                """
            )

    def _meta(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return str(row[0]) if row else None

    def record_document(self, document_hash: str, text: str, *, batch_size: int = 5000) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM processed_documents WHERE document_hash=?",
            (document_hash,),
        ).fetchone()
        if row is not None:
            return False

        counts: dict[str, int] = {}
        with self.connection:
            for token in iter_word_tokens(text):
                counts[token] = counts.get(token, 0) + 1
                if len(counts) >= batch_size:
                    self._flush_counts(counts)
                    counts.clear()
            if counts:
                self._flush_counts(counts)
            self.connection.execute(
                "INSERT INTO processed_documents(document_hash) VALUES(?)",
                (document_hash,),
            )
        return True

    def _flush_counts(self, counts: Mapping[str, int]) -> None:
        rows = [
            (
                token,
                int(count),
                int(dictionary_eligible(token)),
                int(dictionary_eligible(token) and token in self.dictionary_words),
            )
            for token, count in counts.items()
        ]
        self.connection.executemany(
            """
            INSERT INTO vocabulary(token,occurrences,eligible,matched)
            VALUES(?,?,?,?)
            ON CONFLICT(token) DO UPDATE SET
                occurrences=vocabulary.occurrences + excluded.occurrences,
                eligible=MAX(vocabulary.eligible, excluded.eligible),
                matched=MAX(vocabulary.matched, excluded.matched)
            """,
            rows,
        )

    def summary(self, *, unmatched_limit: int = 50) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT
                COUNT(*) AS unique_words,
                COALESCE(SUM(CASE WHEN eligible=1 THEN occurrences ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN eligible=1 AND matched=1 THEN occurrences ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN eligible=1 AND matched=0 THEN occurrences ELSE 0 END), 0)
            FROM vocabulary
            """
        ).fetchone()
        assert row is not None
        eligible = int(row[1])
        matched = int(row[2])
        unmatched = int(row[3])
        common_unmatched = [
            {"token": str(token), "occurrences": int(count)}
            for token, count in self.connection.execute(
                """
                SELECT token, occurrences
                FROM vocabulary
                WHERE eligible=1 AND matched=0
                ORDER BY occurrences DESC, token ASC
                LIMIT ?
                """,
                (max(0, int(unmatched_limit)),),
            ).fetchall()
        ]
        return {
            "total_unique_words": int(row[0]),
            "dictionary_eligible_words": eligible,
            "dictionary_matched_words": matched,
            "dictionary_unmatched_words": unmatched,
            "english_dictionary_percentage": (
                matched / eligible * 100.0 if eligible else 0.0
            ),
            "common_unmatched_tokens": common_unmatched,
        }

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "VocabularyTracker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = [
    "VocabularyTracker",
    "dictionary_eligible",
    "iter_word_tokens",
    "load_english_dictionary",
    "locate_english_wordlists",
    "normalize_dictionary_token",
]
