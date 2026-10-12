"""SLAI v2.3 cyber-risk MLP trainer.

No third-party *direct* imports: relies on SLAI's existing CyberSafetyModule
feature extraction and NeuralNetwork implementation (and their installed deps).
The training-only safety subclass NEVER loads a pre-existing model checkpoint.

Example (automatically bootstraps a missing dataset):
    py -m train_safety_risk --data data/training/safety/cyber_risk.jsonl

Synthetic bootstrap is a demonstration/initialization aid, NOT independent
security ground truth. Synthetic evaluation cannot authorize deployment.

JSONL: {"text":"...","context":"api_request","label":0,
        "group_id":"incident-001","split":"train"}
Split: train / val / test; label: 0 benign, 1 malicious.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
from pathlib import Path
import shutil
import sys

from src.agents.safety.cyber_safety import CyberSafetyModule
from src.agents.safety.modules.neural_network import NeuralNetwork
from src.agents.safety.utils.safety_helpers import normalize_identifier


class OfflineMemory:
    """Truthy placeholder: offline feature extraction does not write memories."""


class TrainingCyberSafetyModule(CyberSafetyModule):
    """Use real configured detectors, but skip loading the production MLP."""

    def _load_security_model(self):
        return None


def make_safety_module():
    return TrainingCyberSafetyModule(memory=OfflineMemory())


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def atomic_json(path, document):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False,
                                  allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# The content below is harmless, locally generated test material. Attack-looking
# fragments are used as strings only; this program never executes them.
# Each split uses disjoint example templates to reduce exact template leakage.
_BOOTSTRAP_CASES = {
    "train": {
        1: (
            "GET /search?q=' UNION SELECT name FROM customers -- HTTP/1.1",
            "POST /login user=admin' OR '1'='1' --",
            "GET /files?file=../../../../etc/passwd HTTP/1.1",
            "POST /query statement=SELECT password FROM users WHERE id=1; DROP TABLE audit;",
            "GET /report?sort=1; WAITFOR DELAY '00:00:05' --",
            "POST /form value=<script>alert(1)</script>",
            "POST /upload filename=../../secrets.txt",
            "GET /api?id=1' OR 1=1 --",
            "POST /template expression=${jndi:ldap://127.0.0.1/a}",
            "GET /lookup?id=1 UNION SELECT token FROM sessions",
            "POST /input data=<img src=x onerror=alert(1)>",
            "GET /export?format=csv&path=../../../private/config",
        ),
        0: (
            "GET /health HTTP/1.1 status=200",
            "POST /profile fields=name,email status=accepted",
            "GET /catalog?sort=price HTTP/1.1 status=200",
            "POST /login status=ok method=oauth2",
            "GET /docs/guidelines HTTP/1.1 status=200",
            "POST /orders quantity=2 status=approved",
            "GET /weather?city=Rotterdam status=200",
            "POST /feedback score=5 status=stored",
            "GET /tickets?state=open status=200",
            "POST /events action=scheduled status=created",
            "GET /metrics endpoint=latency status=200",
            "POST /search keyword=report status=ok",
        ),
    },
    "val": {
        1: (
            "GET /items?filter=' UNION SELECT email FROM directory --",
            "POST /authenticate identity=x' OR 'x'='x' --",
            "GET /download?name=../../../../windows/win.ini",
            "POST /notes message=<script>confirm(1)</script>",
        ),
        0: (
            "GET /news HTTP/1.1 status=200",
            "POST /calendar subject=meeting status=accepted",
            "GET /inventory?category=books status=200",
            "POST /documents tag=archive status=stored",
        ),
    },
    "test": {
        1: (
            "GET /people?q=' UNION SELECT id FROM staff --",
            "POST /identity uid=guest' OR 'a'='a' --",
            "GET /preview?path=../../../../var/log/messages",
            "POST /render template=<script>prompt(1)</script>",
        ),
        0: (
            "GET /schedule HTTP/1.1 status=200",
            "POST /preferences locale=nl-NL status=ok",
            "GET /articles?topic=science status=200",
            "POST /comments body=approved status=created",
        ),
    },
}


def bootstrap_dataset(path, safety, seed, *, train_per_class=240,
                      val_per_class=60, test_per_class=100):
    """Create labeled, deliberately synthetic *initialization* examples.

    Values come from template-defined labels, not the detector's predictions;
    SLAI's *actual* feature extractor is applied later by parse_dataset().
    Test metrics describe only this generator and must never pass production
    promotion gates, regardless of apparent accuracy.
    """
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {path}")
    if min(train_per_class, val_per_class, test_per_class) < 5:
        raise ValueError("Synthetic examples per class must be >= 5")
    rng = random.Random(seed)
    contexts = ("api_request", "code_review", "config_file", "log_event", "general")
    services = ("gateway", "worker", "scheduler", "monitor", "backend", "frontend")
    actions = ("inspected", "received", "recorded", "processed", "reviewed", "observed")
    notes = ("routine", "pending", "scheduled", "logged", "transmitted", "queued")
    expected = {"train": train_per_class, "val": val_per_class, "test": test_per_class}
    rows = []
    seen = set()
    for split, quota in expected.items():
        for label in (0, 1):
            anchors = _BOOTSTRAP_CASES[split][label]
            for index in range(quota):
                # Unique randomized log envelope; label is exclusively tied to
                # the template's simulated benign/malicious semantics.
                nonce = rng.getrandbits(72)
                context = rng.choice(contexts)
                anchor = anchors[index % len(anchors)]
                text = (f"service={rng.choice(services)} "
                        f"action={rng.choice(actions)} "
                        f"request_id={nonce:018x} "
                        f"payload={anchor} "
                        f"note={rng.choice(notes)}")
                fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if fingerprint in seen:
                    raise RuntimeError("Bootstrap generator produced duplicate synthetic text")
                seen.add(fingerprint)
                rows.append({"text": text, "context": context, "label": label,
                             "group_id": f"bootstrap-{split}-{label}-{index:06d}",
                             "split": split,
                             "source": "slai_synthetic_bootstrap_v2"})
    rng.shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if path.exists():
            raise FileExistsError(f"Dataset created concurrently; not overwriting {path}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    manifest = {"schema": "slai.cyber.bootstrap.v2", "created_at": utc_now(),
                "origin": "synthetic", "seed": seed,
                "generation_method": "local_template_simulation_not_security_ground_truth",
                "record_count": len(rows), "split_per_class": expected,
                "independent_real_world_test_set": False,
                "warning": "Synthetic validation and test data cannot establish real-world model quality or justify automatic promotion.",
                "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "safety_config_sha256": sha256_json(safety.cyber_config)}
    atomic_json(path.with_suffix(path.suffix + ".manifest.json"), manifest)
    return manifest


def parse_dataset(path, safety, *, max_records=1_000_000, max_record_chars=65536):
    """Read independently labeled data; forbid repeated/grouped cases crossing splits."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Training data does not exist: {path}. Supply labeled JSONL records first.")
    subsets = {"train": [], "val": [], "test": []}
    seen_text = {}
    seen_groups = {}
    seen_samples = set()
    count = 0
    synthetic_sources = 0
    laboratory_sources = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for lineno, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            if len(raw) > max_record_chars * 8:
                raise ValueError(f"Line {lineno}: oversized JSONL record")
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {lineno}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {lineno}: expected a JSON object")
            if record.get("source") == "slai_synthetic_bootstrap_v2":
                synthetic_sources += 1
            if record.get("source") == "csic_2010_lab":
                laboratory_sources += 1
            split = record.get("split")
            label = record.get("label")
            group = record.get("group_id")
            txt = record.get("text")
            ctx = record.get("context", "general")
            if split not in subsets:
                raise ValueError(f"Line {lineno}: split must be train, val, or test")
            if type(label) is not int or label not in (0, 1):
                raise ValueError(f"Line {lineno}: label must be integer 0 or 1")
            if not isinstance(group, str) or not group.strip() or len(group) > 256:
                raise ValueError(f"Line {lineno}: group_id must be a nonempty string (max 256 chars)")
            if not isinstance(ctx, str) or not ctx or len(ctx) > 128:
                raise ValueError(f"Line {lineno}: context must be a nonempty string (max 128 chars)")
            if not isinstance(txt, str) or not txt.strip() or len(txt) > max_record_chars:
                raise ValueError(f"Line {lineno}: text must be nonempty and at most {max_record_chars} chars")

            # Match the production feature extractor's normalization before hashing.
            normalized = safety._stringify_input(txt)
            context = normalize_identifier(ctx, max_length=80)
            signature = sha256_json([normalized, context])
            former = seen_text.get(signature)
            if former is not None and former[0] != split:
                raise ValueError(f"Line {lineno}: exact-example leakage between {former[0]} and {split}")
            if former is not None and former[1] != label:
                raise ValueError(f"Line {lineno}: inconsistent labels for identical input")
            seen_text[signature] = (split, label)
            old_group = seen_groups.get(group)
            if old_group is not None and old_group != split:
                raise ValueError(f"Line {lineno}: incident group {group!r} spans multiple splits")
            seen_groups[group] = split
            # Group labels may differ when cases in an incident include benign and malicious inputs.
            # Such cases are kept within the same split.
            dedup_key = (signature, split)
            if dedup_key in seen_samples:
                continue
            seen_samples.add(dedup_key)
            findings = (safety._detect_rule_findings(normalized)
                        + safety._detect_vulnerability_findings(normalized, context)
                        + safety._detect_contextual_findings(normalized, context))
            features = safety._build_model_features(normalized, findings, context)
            if not features or any(not math.isfinite(float(v)) for v in features):
                raise ValueError(f"Line {lineno}: nonfinite or empty extracted features")
            subsets[split].append((list(map(float, features)), [float(label)]))
            count += 1
            if count > max_records:
                raise ValueError(f"Dataset exceeds max_records={max_records}")
    digest = hashlib.sha256()
    with path.open("rb") as file_for_hash:
        for chunk in iter(lambda: file_for_hash.read(1024 * 1024), b""):
            digest.update(chunk)
    if synthetic_sources:
        origin = "synthetic"
    elif laboratory_sources == count and count:
        origin = "external_laboratory_benchmark"
    elif laboratory_sources:
        origin = "mixed_laboratory_and_other_unverified"
    else:
        origin = "user_provided_unverified"
    return subsets, {"file_sha256": digest.hexdigest(),
                     "origin": origin,
                     "synthetic_source_records": synthetic_sources,
                     "laboratory_source_records": laboratory_sources,
                     "kept": count, "group_count": len(seen_groups),
                     "counts": {k: dict(sorted(Counter(int(y[0]) for _, y in rows).items()))
                                for k, rows in subsets.items()}}


def require_minimums(subsets, train_each=20, validation_each=5, test_each=5):
    for name, required in (("train", train_each), ("val", validation_each), ("test", test_each)):
        counter = Counter(int(y[0]) for _, y in subsets[name])
        for label in (0, 1):
            if counter[label] < required:
                raise ValueError(f"Dataset '{name}' needs at least {required} examples of label {label}; got {counter[label]}")


def model_from_configuration(safety, seed):
    cfg = dict(safety._cfg("model", {}) or {})
    feature_order = cfg.get("feature_order") or []
    n = int(cfg.get("num_inputs", 8))
    if len(feature_order) != n:
        raise ValueError(f"Configured feature_order has {len(feature_order)} elements but model num_inputs is {n}")
    overrides = dict(cfg.get("neural_network_overrides", {}) or {})
    overrides.update({"deterministic_seed": seed, "set_seed_on_init": True,
                      "restore_best_weights": True})
    net = NeuralNetwork(num_inputs=n,
                        layer_config=cfg.get("layer_config"),
                        loss_function_name=cfg.get("loss_function_name", "cross_entropy"),
                        optimizer_name=cfg.get("optimizer_name", "adam"),
                        initialization_method_default=cfg.get("initialization_method_default", "he_normal"),
                        problem_type=cfg.get("problem_type", "binary_classification"),
                        config=overrides)
    return net, cfg


def predictions(model, data):
    scores = []
    labels = []
    for x, y in data:
        probability = model.predict(x)
        score = float(probability[0] if isinstance(probability, (tuple, list)) else probability)
        if not math.isfinite(score) or score < 0 or score > 1:
            raise ValueError("Model produced an invalid probability")
        scores.append(score)
        labels.append(int(y[0]))
    return labels, scores


def metrics(labels, scores, threshold):
    tp = fp = tn = fn = 0
    for y, p in zip(labels, scores):
        positive = p >= threshold
        if positive and y == 1: tp += 1
        elif positive: fp += 1
        elif y == 1: fn += 1
        else: tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    eps = 1e-12
    logloss = sum(-(y * math.log(max(eps, min(1 - eps, p))) +
                    (1 - y) * math.log(max(eps, min(1 - eps, 1 - p))))
                  for y, p in zip(labels, scores)) / len(labels)
    brier = sum((p - y) ** 2 for y, p in zip(labels, scores)) / len(labels)
    return {"threshold": threshold, "samples": len(labels),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall,
            "fpr": fpr, "accuracy": accuracy, "f1": f1,
            "logloss": logloss, "brier": brier}


def auc_roc(labels, scores):
    """Mann–Whitney AUC with average ranks for ties; O(n log n)."""
    ranked = sorted(zip(scores, labels))
    n1 = sum(labels)
    n0 = len(labels) - n1
    if not n1 or not n0:
        raise ValueError("ROC-AUC requires two classes")
    i, rank_sum = 0, 0.0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        positives = sum(y for _, y in ranked[i:j])
        rank_sum += positives * ((i + 1 + j) / 2)
        i = j
    return (rank_sum - n1 * (n1 + 1) / 2) / (n0 * n1)


def select_threshold(labels, scores, min_precision, min_recall, max_fpr):
    """Select on VALIDATION ONLY; do not inspect test scores to choose a threshold."""
    candidate_scores = sorted(set(scores), reverse=True)
    positives, negatives = sum(labels), len(labels) - sum(labels)
    tp = fp = 0
    index = 0
    sorted_items = sorted(zip(scores, labels), reverse=True)
    preferred = None
    fallback = None
    for threshold in sorted(set(candidate_scores + [0.5, 1.0]), reverse=True):
        while index < len(sorted_items) and sorted_items[index][0] >= threshold:
            if sorted_items[index][1] == 1: tp += 1
            else: fp += 1
            index += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        fpr = fp / negatives if negatives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        score_key = (f1, recall, -fpr, threshold)
        if fallback is None or score_key > fallback[0]:
            fallback = (score_key, threshold)
        if precision >= min_precision and recall >= min_recall and fpr <= max_fpr:
            if preferred is None or score_key > preferred[0]:
                preferred = (score_key, threshold)
    return float((preferred or fallback)[1]), preferred is not None


def passes_gates(val, test, options, dataset):
    counts = dataset["counts"]
    enough = (all(int(counts["train"].get(c, 0)) >= 100 for c in (0, 1))
              and all(int(counts["val"].get(c, 0)) >= 40 for c in (0, 1))
              and int(counts["test"].get(0, 0)) >= 80
              and int(counts["test"].get(1, 0)) >= 40)
    def passed(m):
        return (m["precision"] >= options.min_precision
                and m["recall"] >= options.min_recall
                and m["fpr"] <= options.max_fpr
                and m["auc"] >= options.min_auc)
    return enough and passed(val) and passed(test), {
        "sample_minimums_met": enough,
        "validation_gates_met": passed(val),
        "test_gates_met": passed(test),
    }


def promote_candidate(candidate, deployed, cfg):
    """Atomic deployment with full signature/load validation and restoration."""
    candidate = Path(candidate).resolve()
    deployed = Path(deployed).resolve()
    if candidate == deployed:
        raise ValueError("Candidate and production artifact paths must differ")
    overrides = dict(cfg.get("neural_network_overrides", {}) or {})
    expected_n = int(cfg["num_inputs"])
    checked = NeuralNetwork.load_model(str(candidate), custom_config_override=overrides)
    if checked.num_inputs != expected_n:
        raise ValueError("Candidate input dimension mismatch")
    deployed.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = deployed.with_name(deployed.name + "." + stamp + ".bak")
    staged = deployed.with_name(deployed.name + "." + stamp + ".tmp")
    previously_exists = deployed.exists()
    try:
        if previously_exists:
            shutil.copy2(deployed, backup)
        shutil.copy2(candidate, staged)
        os.replace(staged, deployed)
        confirmed = NeuralNetwork.load_model(str(deployed), custom_config_override=overrides)
        if confirmed.num_inputs != expected_n:
            raise ValueError("Installed production model input dimension mismatch")
    except Exception:
        if previously_exists and backup.exists():
            os.replace(backup, deployed)
        elif not previously_exists:
            deployed.unlink(missing_ok=True)
        raise
    finally:
        staged.unlink(missing_ok=True)
    return str(backup) if previously_exists else None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train SLAI cyber-safety MLP from scratch, with offline bootstrap if the data file is absent")
    p.add_argument("--data", type=Path, default=Path("data/training/safety/cyber_risk.jsonl"))
    p.add_argument("--no-bootstrap", action="store_true", help="Require existing labeled training data")
    p.add_argument("--bootstrap-only", action="store_true", help="Generate/check synthetic data; do not train")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-precision", type=float, default=0.85)
    p.add_argument("--min-recall", type=float, default=0.90)
    p.add_argument("--max-fpr", type=float, default=0.05)
    p.add_argument("--min-auc", type=float, default=0.85)
    p.add_argument("--candidate", type=Path, default=Path("artifacts/safety/cyber_safety_candidate.json"))
    p.add_argument("--report", type=Path, default=Path("artifacts/safety/cyber_safety_training_report.json"))
    p.add_argument("--promote", action="store_true")
    p.add_argument("--verified-independent-data", action="store_true",
                   help="Operator attests independent reviewed real-world evaluation evidence; cannot override synthetic or lab benchmark restrictions")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
            raise ValueError("epochs, batch-size and patience must be positive")
        if not (0 < args.learning_rate < 1):
            raise ValueError("learning-rate must be strictly between 0 and 1")
        if not all(0 <= v <= 1 for v in (args.min_precision, args.min_recall, args.max_fpr, args.min_auc)):
            raise ValueError("metric thresholds must be between 0 and 1")
        if args.candidate.resolve() == args.report.resolve():
            raise ValueError("Candidate and report paths must differ")

        print("Initializing offline SLAI Safety feature extractor (no model load)...", flush=True)
        safety = make_safety_module()
        if not args.data.is_file():
            if args.data.exists():
                raise ValueError(f"Dataset path exists but is not a regular file: {args.data}")
            if args.no_bootstrap:
                raise ValueError(f"Missing labeled dataset and --no-bootstrap specified: {args.data}")
            print("No labeled dataset was found. Creating a local SYNTHETIC bootstrap dataset...", flush=True)
            manifest = bootstrap_dataset(args.data, safety, args.seed)
            print(f"Bootstrapped {manifest['record_count']} synthetic records at {args.data.resolve()}", flush=True)
            print("WARNING: bootstrap examples are NOT real attack data; experimental model only.", flush=True)
        if args.bootstrap_only:
            print(f"Dataset ready: {args.data.resolve()} (training skipped)", flush=True)
            return 0
        subsets, data_info = parse_dataset(args.data, safety)
        if data_info["origin"] != "user_provided_unverified":
            print(f"EXPERIMENTAL MODE ({data_info['origin']}): not independently verified live attack data; production promotion disabled.", flush=True)
        require_minimums(subsets)
        model, cfg = model_from_configuration(safety, args.seed)
        n = model.num_inputs
        if any(len(x) != n for records in subsets.values() for x, _ in records):
            raise ValueError("Feature dimension mismatch with configured neural network")
        print(f"Loaded datasets: {data_info['counts']}; features: {n}", flush=True)

        history = model.train(subsets["train"], epochs=args.epochs,
                              initial_learning_rate=args.learning_rate,
                              batch_size=args.batch_size,
                              early_stopping_patience=args.patience,
                              validation_data=subsets["val"],
                              verbose=True)
        val_y, val_p = predictions(model, subsets["val"])
        threshold, feasible = select_threshold(val_y, val_p, args.min_precision,
                                               args.min_recall, args.max_fpr)
        model.binary_threshold = threshold
        val_metrics = metrics(val_y, val_p, threshold)
        val_metrics["auc"] = auc_roc(val_y, val_p)

        # Save candidate with SLAI's signed persistence interface, then reload to
        # check deployment-format integrity before ever opening test predictions.
        candidate = args.candidate.resolve()
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if not model.save_model(str(candidate)):
            raise RuntimeError("NeuralNetwork.save_model returned false")
        verified = NeuralNetwork.load_model(str(candidate),
                    custom_config_override=dict(cfg.get("neural_network_overrides", {}) or {}))
        if verified.num_inputs != n:
            raise ValueError("Saved candidate has invalid input dimension")
        test_y, test_p = predictions(verified, subsets["test"])
        test_metrics = metrics(test_y, test_p, threshold)
        test_metrics["auc"] = auc_roc(test_y, test_p)
        accepted, checks = passes_gates(val_metrics, test_metrics, args, data_info)
        checks["validation_threshold_feasible"] = feasible
        accepted = accepted and feasible
        # A historical laboratory benchmark is NOT independent field evidence.
        # Promotion requires distinct human-reviewed field data and an explicit
        # operator attestation. No annotation can override the lab/synthetic gate.
        independent_field_evidence = (data_info["origin"] == "user_provided_unverified"
                                      and args.verified_independent_data)
        checks["independent_real_world_data"] = independent_field_evidence
        accepted = accepted and independent_field_evidence

        report = {
            "schema_version": "slai.safety_risk.training.v1",
            "completed_at": utc_now(), "candidate_path": str(candidate),
            "production_path": str(Path(cfg.get("model_path", "src/agents/safety/models/cyber_safety_model.json")).resolve()),
            "dataset": data_info, "feature_order": list(cfg.get("feature_order", [])),
            "configuration_sha256": sha256_json(safety.cyber_config),
            "training": {"epochs_requested": args.epochs, "epochs_completed": history.epochs_completed,
                         "early_stopped": history.early_stopped, "best_validation_loss": history.best_validation_loss,
                         "seed": args.seed, "learning_rate": args.learning_rate,
                         "batch_size": args.batch_size, "patience": args.patience},
            "validation_metrics": val_metrics, "test_metrics": test_metrics,
            "requested_gates": {"min_precision": args.min_precision, "min_recall": args.min_recall,
                                "max_fpr": args.max_fpr, "min_auc": args.min_auc},
            "gates": checks, "accepted": accepted, "promoted": False,
            "evaluation_scope": ("externally_labeled_laboratory_benchmark_not_live_field_evidence"
                                 if data_info["origin"] == "external_laboratory_benchmark"
                                 else ("synthetic_only_not_independent" if data_info["origin"] == "synthetic"
                                       else "user_provided_labels_attestation_required")),
            "real_world_performance_established": False,
        }
        print("Validation:", json.dumps(val_metrics, sort_keys=True), flush=True)
        print("Test:", json.dumps(test_metrics, sort_keys=True), flush=True)
        if args.promote:
            if accepted:
                production = Path(cfg["model_path"]).resolve()
                if production == args.report.resolve():
                    raise ValueError("Report path conflicts with production model")
                backup = promote_candidate(candidate, production, cfg)
                report["promoted"] = True
                report["previous_production_backup"] = backup
                print(f"PROMOTED: {production}", flush=True)
            else:
                print("PROMOTION REFUSED: requires independently reviewed field data, --verified-independent-data and passing evaluation gates", file=sys.stderr)
        else:
            print("Candidate saved; production model was NOT modified.", flush=True)
        atomic_json(args.report, report)
        print(f"Report: {args.report.resolve()}", flush=True)
        return 0 if not args.promote or accepted else 3
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"TRAINING FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
