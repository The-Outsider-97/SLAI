"""Real BEIR/SciFact acquisition, IR candidate construction, validation and metrics.

BEIR qrels are sparse: a document not listed in a qrel is *unjudged*, NOT a
human-labeled negative. Pseudo-negatives here are explicit weak supervision.
Benchmark metrics use the conventional unjudged=0 assumption and disclose it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import urllib.request
import zipfile

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_SOURCE = "beir/scifact"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{i}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{i}")
            yield value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_scifact(destination: Path, *, timeout: int = 60, allow_download: bool = False) -> dict:
    """Fetch original BEIR archive with partial-download recovery and zip-slip limits."""
    if not allow_download:
        raise ValueError("Network download requires explicit --allow-download")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "scifact.zip"
    partial = destination / "scifact.zip.partial"
    if not archive.is_file():
        start = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(SCIFACT_URL, headers={"Range": f"bytes={start}-"} if start else {})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            code = response.status
            if code not in (200, 206):
                raise RuntimeError(f"Dataset download failed: HTTP {code}")
            if code == 206 and start:
                crange = response.headers.get("Content-Range", "")
                if not crange.startswith(f"bytes {start}-"):
                    raise ValueError("Remote server did not honor resume offset")
            if code == 200 and start:
                partial.unlink()
            with partial.open("ab" if code == 206 and start else "wb") as f:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
        # An invalid download is never promoted to the full archive.
        if not zipfile.is_zipfile(partial):
            raise ValueError("Incomplete or invalid SciFact download; partial kept for retry")
        os.replace(partial, archive)
    raw = destination / "raw"
    required = [raw / "corpus.jsonl", raw / "queries.jsonl", raw / "qrels" / "train.tsv", raw / "qrels" / "test.tsv"]
    if not all(p.is_file() for p in required):
        work = destination / "raw.incomplete"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        with zipfile.ZipFile(archive) as zf:
            members = zf.infolist()
            if sum(m.file_size for m in members) > (256 << 20):
                raise ValueError("Archive exceeds SciFact extraction safety limit")
            for item in members:
                name = Path(item.filename)
                if item.is_dir():
                    continue
                parts = name.parts
                if not parts or parts[0] != "scifact":
                    raise ValueError(f"Unexpected archive member: {item.filename}")
                relative = Path(*parts[1:])
                if not relative.parts or ".." in relative.parts or relative.is_absolute():
                    raise ValueError(f"Unsafe archive member: {item.filename}")
                target = work / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(item) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        if not all((work / x.relative_to(raw)).is_file() for x in required):
            raise ValueError("SciFact archive missing expected files")
        if raw.exists():
            shutil.rmtree(raw)
        os.replace(work, raw)
    info = {"dataset": SCIFACT_SOURCE, "source_url": SCIFACT_URL,
            "archive_sha256": _sha256(archive), "archive_bytes": archive.stat().st_size,
            "license_review_required": True, "raw_path": str(raw.resolve())}
    atomic_json(destination / "source_manifest.json", info)
    return info


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    labels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            fields = line.strip().split("\t")
            if index == 0 and fields[0].lower().startswith("query"):
                continue
            if len(fields) != 3:
                raise ValueError(f"Invalid qrel row in {path}: {index+1}")
            qid, did, grade = fields
            val = int(grade)
            if val < 0:
                raise ValueError("Negative relevance label not supported")
            if did in labels[qid] and labels[qid][did] != val:
                raise ValueError("Conflicting relevance judgments")
            labels[qid][did] = val
    return dict(labels)


class BM25:
    """Small-corpus inverted-index BM25 candidate generator."""
    def __init__(self, corpus: dict[str, str]):
        from .relevance_ranker import _terms
        self.corpus = corpus
        self.counts = {did: Counter(_terms(text)) for did, text in corpus.items()}
        self.length = {did: sum(counts.values()) for did, counts in self.counts.items()}
        self.avgdl = sum(self.length.values()) / max(len(corpus), 1)
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        for did, counts in self.counts.items():
            for term, freq in counts.items():
                self.postings[term][did] = freq

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        from .relevance_ranker import _terms
        scores: dict[str, float] = defaultdict(float)
        size = len(self.corpus)
        for term in set(_terms(query)):
            postings = self.postings.get(term, {})
            n = len(postings)
            idf = math.log(1 + (size - n + 0.5) / (n + 0.5))
            for did, tf in postings.items():
                norm = 1.2 * (0.25 + 0.75 * self.length[did] / max(1, self.avgdl))
                scores[did] += idf * tf * 2.2 / (tf + norm)
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]


def _load_beir(raw: Path) -> tuple[dict[str, str], dict[str, str], dict, dict]:
    corpus = {}
    for row in read_jsonl(raw / "corpus.jsonl"):
        did = str(row["_id"])
        if did in corpus:
            raise ValueError(f"Duplicate document {did}")
        corpus[did] = (str(row.get("title") or "") + "\n" + str(row.get("text") or "")).strip()
    queries = {}
    for row in read_jsonl(raw / "queries.jsonl"):
        qid = str(row["_id"])
        if qid in queries:
            raise ValueError(f"Duplicate query {qid}")
        queries[qid] = str(row["text"]).strip()
    train = _read_qrels(raw / "qrels" / "train.tsv")
    test = _read_qrels(raw / "qrels" / "test.tsv")
    if not corpus or not queries or not train or not test:
        raise ValueError("Empty benchmark input")
    if any(did not in corpus for rel in (train, test) for docs in rel.values() for did in docs):
        raise ValueError("Qrel references document missing from corpus")
    if any(qid not in queries for rel in (train, test) for qid in rel):
        raise ValueError("Qrel references query missing from query set")
    if set(train) & set(test):
        raise ValueError("SciFact train/test qrel query overlap")
    return corpus, queries, train, test


def prepare_scifact(raw: Path, output: Path, *, seed: int = 42,
                    negatives: int = 12, eval_pool: int = 100,
                    validation_fraction: float = 0.15) -> dict:
    if negatives < 1 or eval_pool < 10 or not 0.05 <= validation_fraction <= 0.4:
        raise ValueError("Invalid negatives, eval_pool, or validation fraction")
    corpus, queries, original_train, test = _load_beir(raw)
    rng = random.Random(seed)
    ids = sorted(original_train)
    rng.shuffle(ids)
    nval = max(1, int(len(ids) * validation_fraction))
    val_ids = set(ids[:nval])
    train = {qid: v for qid, v in original_train.items() if qid not in val_ids}
    valid = {qid: v for qid, v in original_train.items() if qid in val_ids}
    splits = {"train": train, "validation": valid, "test": test}
    output.mkdir(parents=True, exist_ok=True)
    bm25 = BM25(corpus)
    counts = {}
    for split, qrels in splits.items():
        temp = output / f"{split}.jsonl.tmp"
        count, positive, unjudged = 0, 0, 0
        with temp.open("w", encoding="utf-8", newline="\n") as out:
            for qid in sorted(qrels):
                query = queries[qid]
                judged = qrels[qid]
                found = bm25.search(query, eval_pool if split != "train" else max(100, negatives * 8))
                scores = dict(found)
                if split == "train":
                    relevant = [did for did, grade in judged.items() if grade > 0]
                    if not relevant:
                        continue
                    negative_ids = [did for did, _ in found if did not in judged][:negatives]
                    # BM25 can return fewer than negatives on rare/zero overlap queries.
                    if len(negative_ids) < negatives:
                        remaining = [d for d in sorted(corpus) if d not in judged and d not in negative_ids]
                        negative_ids.extend(rng.sample(remaining, min(negatives - len(negative_ids), len(remaining))))
                    doc_ids = relevant + negative_ids
                else:
                    # Evaluation MUST use retrieval-only candidates, never inject
                    # known positive labels: that would inflate recall artificially.
                    doc_ids = [did for did, _ in found]
                for did in doc_ids:
                    grade = int(judged.get(did, 0))
                    is_judged = did in judged
                    record = {
                        "query_id": qid, "query": query, "document_id": did,
                        "text": corpus[did], "label": grade,
                        "label_source": "benchmark" if is_judged else "unjudged_mined_negative",
                        "source_dataset": SCIFACT_SOURCE,
                        "dataset_version": "BEIR SciFact downloaded archive",
                        "human_review_status": "source_benchmark" if is_judged else "not_reviewed",
                        "annotation_confidence": None,
                        "metadata": {}, "bm25": scores.get(did, 0.0), "split": split,
                    }
                    out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
                    positive += int(grade > 0)
                    unjudged += int(not is_judged)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, output / f"{split}.jsonl")
        atomic_json(output / f"qrels_{split}.json", qrels)
        counts[split] = {"queries": len(qrels), "candidates": count,
                         "positive_in_pool": positive, "unjudged_in_pool": unjudged,
                         "sha256": _sha256(output / f"{split}.jsonl")}
    if (output.parent / "source_manifest.json").exists():
        source = json.loads((output.parent / "source_manifest.json").read_text(encoding="utf-8"))
    else:
        source = {"dataset": SCIFACT_SOURCE, "raw_path": str(raw), "source_unverified": True}
    report = {"source": source, "seed": seed, "negative_count": negatives,
              "evaluation_pool_size": eval_pool, "validation_fraction": validation_fraction,
              "full_corpus_documents": len(corpus), "queries_available": len(queries),
              "splits": counts, "negative_label_policy": "unjudged mined candidates are weak negatives",
              "test_labels_used_in_training": False}
    atomic_json(output / "dataset_manifest.json", report)
    return report


def validate_prepared(data_dir: Path) -> dict:
    manifest = json.loads((data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    sets: dict[str, set[str]] = {}
    pairsets = {}
    counts = {}
    for split in ("train", "validation", "test"):
        file = data_dir / f"{split}.jsonl"
        if _sha256(file) != manifest["splits"][split]["sha256"]:
            raise ValueError(f"Tampered dataset: {file}")
        seen: set[tuple[str, str]] = set()
        query_ids: set[str] = set()
        positives = 0
        count = 0
        for row in read_jsonl(file):
            if row.get("split") != split or row.get("source_dataset") != SCIFACT_SOURCE:
                raise ValueError("Split or dataset provenance mismatch")
            if not isinstance(row.get("query"), str) or not row["query"].strip() or not isinstance(row.get("text"), str):
                raise ValueError("Missing query/document text")
            if type(row.get("label")) is not int or row["label"] < 0:
                raise ValueError("Invalid label")
            pair = (str(row["query_id"]), str(row["document_id"]))
            if pair in seen:
                raise ValueError(f"Duplicate query-document pair: {pair}")
            seen.add(pair)
            query_ids.add(pair[0])
            positives += int(row["label"] > 0)
            count += 1
        if not query_ids or (split == "train" and not positives):
            raise ValueError("Insufficient labeled training data")
        sets[split] = query_ids
        pairsets[split] = seen
        counts[split] = {"queries": len(query_ids), "rows": count, "positives_in_pool": positives}
    if any(sets[a] & sets[b] for a, b in (("train", "test"), ("train", "validation"), ("test", "validation"))):
        raise ValueError("Query leakage between splits")
    if any(pairsets[a] & pairsets[b] for a, b in (("train", "test"), ("train", "validation"), ("test", "validation"))):
        raise ValueError("Query-document leakage")
    return {"valid": True, "counts": counts, "manifest": manifest}


def retrieval_metrics(groups: dict[str, list[tuple[str, float]]],
                      qrels: dict[str, dict[str, int]], k: int = 10) -> dict[str, float]:
    """Standard judged-positive, unjudged-as-zero IR evaluation per query.

    MAP@k uses min(R,k) as denominator; nDCG uses graded gain (2^rel-1).
    Recall@100 is measured for candidates actually retrieved, not imagined
    items outside the pool. A fixed <=100-pool cannot improve Recall@100 by reranking.
    """
    if not groups:
        raise ValueError("No queries to evaluate")
    metrics = defaultdict(float)
    for qid, ranked in groups.items():
        relevant = qrels[qid]
        positives = sum(v > 0 for v in relevant.values())
        if positives == 0:
            continue
        seen = set()
        docs = []
        for did, value in sorted(ranked, key=lambda x: (-x[1], x[0])):
            if did not in seen:
                docs.append(did)
                seen.add(did)
        hits = [int(relevant.get(did, 0) > 0) for did in docs[:100]]
        metrics["Recall@100"] += sum(hits) / positives
        top = docs[:k]
        rel = [int(relevant.get(did, 0)) for did in top]
        count = sum(x > 0 for x in rel)
        metrics["Recall@10"] += count / positives
        metrics["Precision@10"] += count / k
        metrics["HitRate@10"] += float(count > 0)
        ranks = [i + 1 for i, x in enumerate(rel) if x > 0]
        metrics["MRR@10"] += 1 / ranks[0] if ranks else 0
        found, ap = 0, 0.0
        for i, value in enumerate(rel, 1):
            if value > 0:
                found += 1
                ap += found / i
        metrics["MAP@10"] += ap / min(positives, k)
        dcg = sum((2 ** max(v, 0) - 1) / math.log2(i + 2) for i, v in enumerate(rel))
        ideal = sorted(relevant.values(), reverse=True)[:k]
        idcg = sum((2 ** max(v, 0) - 1) / math.log2(i + 2) for i, v in enumerate(ideal))
        metrics["nDCG@10"] += dcg / idcg if idcg else 0
    n = sum(bool(any(v > 0 for v in x.values())) for x in (qrels[qid] for qid in groups))
    if n == 0:
        raise ValueError("No queries with positive judgments")
    return {key: round(metrics[key] / n, 6) for key in (
        "MRR@10", "nDCG@10", "MAP@10", "Recall@10", "Recall@100", "Precision@10", "HitRate@10")}
