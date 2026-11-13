"""Offline tests for the bounded-memory corpus reader in src/training/lantra_corpus.py."""
from __future__ import annotations

import random
import sys
import tempfile
import tracemalloc
import unittest

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training import lantra_corpus as lc  # noqa: E402

ALPHABET = ["a", "b", "word", "\u00e9", "\ufb01", " ", " ", "\t", "\n", "\n", "\r", "\r\n", "\x00", "e\u0301", "\u00a0", "\u00df", "1"]


def random_text(rng: random.Random, pieces: int) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(pieces))


def legacy_normalise(path: Path) -> str:
    return lc._normalize_document_text(lc._decode_document_bytes(path.read_bytes()))


class StreamingNormalisationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._saved = (lc.WHOLE_FILE_DECODE_LIMIT_BYTES, lc.STREAM_READ_BYTES)

    def tearDown(self) -> None:
        lc.WHOLE_FILE_DECODE_LIMIT_BYTES, lc.STREAM_READ_BYTES = self._saved
        self._tmp.cleanup()

    def write(self, text: str, name: str = "doc.txt") -> Path:
        path = self.dir / name
        path.write_bytes(text.encode("utf-8"))
        return path

    def test_small_file_matches_legacy_normalisation_exactly(self) -> None:
        rng = random.Random(7)
        for index in range(200):
            path = self.write(random_text(rng, rng.randint(0, 120)), f"s{index}.txt")
            expected = legacy_normalise(path)
            parts = list(lc.iter_text_file_parts(path))
            self.assertEqual(parts, [expected] if expected else [])

    def test_streamed_decoding_matches_legacy_for_any_chunk_size(self) -> None:
        rng = random.Random(11)
        lc.WHOLE_FILE_DECODE_LIMIT_BYTES = 0  # force the streaming path
        for index in range(200):
            path = self.write(random_text(rng, rng.randint(0, 160)), f"c{index}.txt")
            expected = legacy_normalise(path)
            lc.STREAM_READ_BYTES = rng.randint(1, 23)  # splits multi-byte characters and CRLF pairs
            parts = list(lc.iter_text_file_parts(path))
            self.assertEqual(parts, [expected] if expected else [], msg=repr(path.read_bytes()))

    def test_large_file_is_cut_into_deterministic_parts_that_lose_no_text(self) -> None:
        paragraphs = [" ".join(f"w{p}x{i}" for i in range(40)) for p in range(400)]
        path = self.write("\n\n".join(paragraphs))
        whole = legacy_normalise(path)
        lc.WHOLE_FILE_DECODE_LIMIT_BYTES = 0
        lc.STREAM_READ_BYTES = 997
        parts = list(lc.iter_text_file_parts(path, part_chars=5_000))
        self.assertGreater(len(parts), 3)
        self.assertEqual("\n\n".join(parts), whole)
        self.assertTrue(all(len(part) <= 5_000 * 1.25 + 400 for part in parts))
        self.assertEqual(parts, list(lc.iter_text_file_parts(path, part_chars=5_000)))
        documents = list(lc.extract_raw_documents(path))
        self.assertEqual([doc.logical_index for doc in documents], list(range(len(documents))))

    def test_cp1252_and_utf16_streams_are_decoded(self) -> None:
        lc.WHOLE_FILE_DECODE_LIMIT_BYTES = 0
        lc.STREAM_READ_BYTES = 5
        legacy = self.dir / "cp.txt"
        legacy.write_bytes("caf\xe9 cr\xe8me\nna\xefve".encode("cp1252"))
        self.assertEqual(list(lc.iter_text_file_parts(legacy)), ["caf\u00e9 cr\u00e8me\nna\u00efve"])
        wide = self.dir / "u16.txt"
        wide.write_bytes("hello\nworld".encode("utf-16"))
        self.assertEqual(list(lc.iter_text_file_parts(wide)), ["hello\nworld"])

    def test_peak_memory_is_independent_of_file_size(self) -> None:
        line = ("the quick brown fox jumps over the lazy dog " * 4).strip() + "\n"
        path = self.dir / "big.txt"
        with path.open("w", encoding="utf-8") as handle:
            for index in range(120_000):
                handle.write(line)
                if index % 20 == 0:
                    handle.write("\n")
        size_mb = path.stat().st_size / 1e6
        self.assertGreater(size_mb, 20)
        lc.WHOLE_FILE_DECODE_LIMIT_BYTES = 1_000_000
        lc.STREAM_READ_BYTES = 1_000_000
        tracemalloc.start()
        try:
            total = 0
            for part in lc.iter_text_file_parts(path, part_chars=1_000_000):
                total += len(part)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertGreater(total, 20_000_000)
        self.assertLess(peak / 1e6, size_mb * 0.5, msg=f"peak {peak/1e6:.1f} MB for a {size_mb:.1f} MB file")


class DocumentHeadTests(unittest.TestCase):
    def test_head_is_bounded_and_word_aligned(self) -> None:
        text = " ".join(f"word{i}" for i in range(50_000))
        head = lc.document_head(text, 1000)
        self.assertLessEqual(len(head), 1000)
        self.assertTrue(text.startswith(head))
        self.assertTrue(text[len(head)] == " ")
        self.assertEqual(lc.document_head("short", 1000), "short")


class CurriculumAdapterMemoryTests(unittest.TestCase):
    def test_documents_keep_a_bounded_head_but_report_their_full_size(self) -> None:
        from src.training.enrichment_contracts import CurriculumConfig
        from src.training.source_adapter import CanonicalLantraSourceAdapter

        rng = random.Random(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            body = "\n\n".join(" ".join(f"w{rng.randint(0, 5000)}" for _ in range(60)) + "." for _ in range(900))
            (root / "book.txt").write_text(body, encoding="utf-8")
            (root / "short.txt").write_text("a short but valid document " * 6, encoding="utf-8")
            config = CurriculumConfig(source_paths=(str(root),), seed=3, validation_fraction=0.0, test_fraction=0.0)
            adapter = CanonicalLantraSourceAdapter(config)
            documents = {Path(d.source_path).name: d for d in adapter.extract_documents(adapter.discover_files())}
            book = documents["book.txt"]
            self.assertLessEqual(len(book.text), lc.DOCUMENT_HEAD_CHARS)
            self.assertEqual(book.character_count, len(lc.normalize_document_text(body)))
            self.assertTrue(lc.normalize_document_text(body).startswith(book.text))
            self.assertEqual(book.provenance()["character_count"], book.character_count)
            short = documents["short.txt"]
            self.assertEqual(short.character_count, len(short.text))


def reference_selection(offers, capacity: int, seed: int):
    """Plain implementation of the original load_raw_corpus segment policy."""
    by_hash = {}
    duplicates = 0
    for text, digest, doc_id, index in offers:
        existing = by_hash.get(digest)
        if existing is not None:
            duplicates += 1
            if lc.stable_unit_interval(doc_id + digest, seed) < lc.stable_unit_interval(existing[2] + digest, seed):
                by_hash[digest] = (text, digest, doc_id, index)
        else:
            by_hash[digest] = (text, digest, doc_id, index)
    ranked = sorted(by_hash.values(), key=lambda item: (lc.stable_unit_interval(item[1], seed + 71), item[1]))
    unique = len(ranked)
    if capacity > 0:
        ranked = ranked[:capacity]
    return [(t, h, d, i) for t, h, d, i in ranked], duplicates, unique


class BoundedSegmentSelectorTests(unittest.TestCase):
    def test_matches_the_unbounded_reference_for_every_capacity(self) -> None:
        rng = random.Random(3)
        pool = [lc.normalized_text_hash(f"segment number {i} " * 5) for i in range(300)]
        offers = []
        for n in range(1500):
            digest = rng.choice(pool)
            offers.append((f"text-{digest[:8]}", digest, f"doc{rng.randint(0, 40)}", n))
        for capacity in (0, 1, 7, 100, 299, 300, 1000):
            selector = lc.BoundedSegmentSelector(capacity=capacity, seed=42)
            for offer in offers:
                selector.offer(*offer)
            expected, duplicates, unique = reference_selection(offers, capacity, 42)
            got = [(s.text, s.normalized_sha256, s.document_id, s.segment_index) for s in selector.ranked()]
            self.assertEqual(got, expected, msg=f"capacity={capacity}")
            self.assertEqual(selector.duplicate_segments, duplicates)
            self.assertEqual(selector.unique_segments, unique)
            self.assertEqual(selector.cap_applied, capacity > 0 and unique > capacity)

    def test_result_does_not_depend_on_traversal_order(self) -> None:
        rng = random.Random(5)
        pool = [lc.normalized_text_hash(f"s{i}") for i in range(80)]
        offers = [(f"t{n}", rng.choice(pool), f"doc{rng.randint(0, 9)}", n) for n in range(400)]

        def run(sequence):
            selector = lc.BoundedSegmentSelector(capacity=30, seed=9)
            for offer in sequence:
                selector.offer(*offer)
            return [(s.normalized_sha256, s.document_id) for s in selector.ranked()]

        shuffled = offers[:]
        rng.shuffle(shuffled)
        self.assertEqual(run(offers), run(shuffled))


if __name__ == "__main__":
    unittest.main()
