"""Reproducible offline training for SLAI alignment representations and scoring.

Usage:
 python -m src.agents.alignment.train_models --data reviewed_pairs.jsonl \
   --metadata dataset_metadata.json --out data/models/alignment/review_001

Input JSONL: group_id, value_text, action_text, label (0/1), source_id.
Metadata JSON must document: license, annotation_guidelines, source_description.
All examples must be independently reviewed. Synthetic fixtures are for tests only.
The resulting checkpoint is NOT approved for active runtime use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from .learned_models import (
    ENCODER_FILE, SCHEMA, SCORER_FILE, RepresentationEncoder,
    ValueAlignmentScorer, _sha256,
)


def read_reviewed_pairs(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    owners: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {number} is not a JSON object")
            for key in ("group_id", "value_text", "action_text", "source_id"):
                value = item.get(key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Line {number} is missing nonempty {key}")
            if type(item.get("label")) is not int or item["label"] not in (0, 1):
                raise ValueError(f"Line {number} label must be integer 0 or 1")
            if any(len(item[key]) > 8192 for key in ("value_text", "action_text")):
                raise ValueError(f"Line {number} has oversized model text")
            # A duplicate pair in different groups would contaminate holdouts.
            pair = "\0".join((item["value_text"].strip().casefold(), item["action_text"].strip().casefold()))
            digest = hashlib.sha256(pair.encode("utf-8")).hexdigest()
            group = item["group_id"]
            if digest in owners and owners[digest] != group:
                raise ValueError(f"Duplicate value/action pair crosses groups at line {number}")
            owners[digest] = group
            records.append(item)
    if len(records) < 12:
        raise ValueError("At least 12 independently labeled pairs are required even for a smoke run")
    return records


def grouped_splits(
    rows: Sequence[Mapping[str, Any]], seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["group_id"]), []).append(dict(row))
    if len(groups) < 9:
        raise ValueError("At least nine independent groups are required")
    group_ids = sorted(groups)
    rng = random.Random(seed)
    for _ in range(256):
        rng.shuffle(group_ids)
        n = len(group_ids)
        ntest, nval = max(2, round(n * 0.20)), max(2, round(n * 0.20))
        keys = (group_ids[: n - nval - ntest], group_ids[n - nval - ntest: n - ntest], group_ids[n - ntest:])
        splits = [[record for key in subset for record in groups[key]] for subset in keys]
        if all(len(part) >= 2 and {int(r["label"]) for r in part} == {0, 1} for part in splits):
            return splits[0], splits[1], splits[2]
    raise ValueError("Cannot form group-disjoint train/validation/test splits containing both classes")


def _batches(rows: Sequence[Mapping[str, Any]], batch_size: int):
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset: offset + batch_size]
        yield (
            [str(r["value_text"]) for r in batch],
            [str(r["action_text"]) for r in batch],
            torch.tensor([int(r["label"]) for r in batch], dtype=torch.float32),
        )


def _metrics(scorer: ValueAlignmentScorer, rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    scorer.eval()
    scores, labels = [], []
    with torch.inference_mode():
        for values, actions, expected in _batches(rows, 32):
            scores.extend(float(x) for x in scorer(values, actions).cpu().tolist())
            labels.extend(float(x) for x in expected.tolist())
    if not scores or not all(math.isfinite(s) for s in scores):
        raise ValueError("Evaluation produced no finite predictions")
    accuracy = sum((s >= 0.5) == (y == 1.0) for s, y in zip(scores, labels)) / len(scores)
    brier = sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(scores)
    tp = sum(s >= 0.5 and y == 1.0 for s, y in zip(scores, labels))
    fn = sum(s < 0.5 and y == 1.0 for s, y in zip(scores, labels))
    fp = sum(s >= 0.5 and y == 0.0 for s, y in zip(scores, labels))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "accuracy": accuracy, "brier": brier,
        "positive_precision": precision, "positive_recall": recall,
        "examples": float(len(scores)),
    }


def _representation_retrieval(
    encoder: RepresentationEncoder, rows: Sequence[Mapping[str, Any]]
) -> Dict[str, float]:
    """Positive-pair retrieval across distinct values/actions, not policy agreement."""
    encoder.eval()
    with torch.inference_mode():
        values = encoder([str(r["value_text"]) for r in rows])
        actions = encoder([str(r["action_text"]) for r in rows])
        similarities = values @ actions.T
        eligible = [i for i, r in enumerate(rows) if int(r["label"]) == 1]
        correct = sum(
            int(int(torch.argmax(similarities[i]).item()) == i)
            for i in eligible
        )
    return {"positive_pair_top1_retrieval": correct / len(eligible) if eligible else 0.0}


def train(
    *, data: Path, metadata: Path, out: Path, seed: int = 42,
    encoder_epochs: int = 5, scorer_epochs: int = 10, batch_size: int = 16
) -> Dict[str, Any]:
    if encoder_epochs < 1 or scorer_epochs < 1 or batch_size < 1:
        raise ValueError("Epoch counts and batch_size must be positive")
    rows = read_reviewed_pairs(data)
    source = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or any(
        not isinstance(source.get(key), str) or not source[key].strip()
        for key in ("license", "annotation_guidelines", "source_description")
    ):
        raise ValueError("Dataset metadata requires license, annotation_guidelines and source_description")
    training, validation, testing = grouped_splits(rows, seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))

    encoder = RepresentationEncoder().cpu()
    scorer = ValueAlignmentScorer(encoder).cpu()
    optimizer = torch.optim.AdamW(encoder.trainable_parameters(), lr=0.0005)
    rng = random.Random(seed)

    # Stage 1: learn reusable value/action text representations using
    # reviewed positive/negative pairs, not a fixed hand-written policy score.
    for _ in range(encoder_epochs):
        encoder.train()
        shuffled = list(training)
        rng.shuffle(shuffled)
        for values, actions, expected in _batches(shuffled, batch_size):
            optimizer.zero_grad(set_to_none=True)
            similarity = F.cosine_similarity(encoder(values), encoder(actions), dim=-1)
            loss = torch.where(expected > 0.5, 1.0 - similarity, F.relu(similarity - 0.20)).mean()
            loss.backward()
            optimizer.step()

    # Stage 2: train the independent compatibility head on fixed representations.
    for parameter in encoder.trainable_parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(scorer.head.parameters(), lr=0.0005)
    positives = sum(int(r["label"]) == 1 for r in training)
    negatives = len(training) - positives
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(negatives / positives))
    )
    best_loss = float("inf")
    best_weights = None
    for _ in range(scorer_epochs):
        scorer.train()
        encoder.eval()
        shuffled = list(training)
        rng.shuffle(shuffled)
        for values, actions, expected in _batches(shuffled, batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = scorer.forward_logits(values, actions)
            loss = criterion(logits, expected)
            loss.backward()
            optimizer.step()
        scorer.eval()
        with torch.inference_mode():
            validation_loss = sum(
                float(criterion(scorer.forward_logits(v, a), y).item()) * len(y)
                for v, a, y in _batches(validation, batch_size)
            ) / len(validation)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_weights = {key: value.clone() for key, value in scorer.export_weights().items()}
    if best_weights is None:
        raise RuntimeError("No valid scorer validation checkpoint was produced")
    scorer.import_weights(best_weights)
    encoder.eval()
    scorer.eval()

    results = {
        "validation": _metrics(scorer, validation),
        "test": _metrics(scorer, testing),
        "representation_test": _representation_retrieval(encoder, testing),
        "majority_test_accuracy_baseline": max(
            sum(int(r["label"]) == 0 for r in testing),
            sum(int(r["label"]) == 1 for r in testing)
        ) / len(testing),
    }
    # Always unapproved: no automatic production promotion from training metrics.
    if out.exists():
        raise FileExistsError("Output model directory exists; choose a new versioned directory")
    out.mkdir(parents=True)
    torch.save(encoder.export_weights(), out / ENCODER_FILE)
    torch.save(scorer.export_weights(), out / SCORER_FILE)
    manifest = {
        "schema": SCHEMA,
        "model_version": "1.0.0",
        "approved": False,
        "calibrated": False,
        "features": encoder.features_manifest(),
        "dataset_sha256": _sha256(data),
        "dataset_metadata_sha256": _sha256(metadata),
        "dataset_metadata": source,
        "splits": {"train": len(training), "validation": len(validation), "test": len(testing)},
        "seed": seed,
        "training": {"encoder_epochs": encoder_epochs, "scorer_epochs": scorer_epochs,
                     "batch_size": batch_size},
        "evaluation": results,
        "artifacts": {ENCODER_FILE: _sha256(out / ENCODER_FILE),
                      SCORER_FILE: _sha256(out / SCORER_FILE)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-epochs", type=int, default=5)
    parser.add_argument("--scorer-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    results = train(
        data=args.data, metadata=args.metadata, out=args.out, seed=args.seed,
        encoder_epochs=args.encoder_epochs, scorer_epochs=args.scorer_epochs,
        batch_size=args.batch_size,
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
