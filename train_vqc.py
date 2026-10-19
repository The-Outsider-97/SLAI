"""Train and benchmark an experimental binary SLAI VQC on a labelled CSV.

Usage:
  python -m train_vqc --csv data.csv --label target --features f1 f2 \\
      --output artifacts/qnn/vqc_candidate.npz

Requires scientific optional dependency: scikit-learn.
The saved artifact is a candidate, NOT automatically an approved deployment.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.agents.qnn_agent import StateVectorCircuit
from src.agents.qnn.variational_classifier import VariationalQuantumClassifier


def read_data(path: Path, columns: list[str], label: str):
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV must have unique named columns")
        if not set(columns + [label]).issubset(reader.fieldnames):
            raise ValueError("CSV missing requested columns")
        if label in columns or len(columns) != len(set(columns)):
            raise ValueError("feature columns must be distinct and exclude the label")
        rows = list(reader)
    if not 20 <= len(rows) <= 20000:
        raise ValueError("CSV requires 20..20000 labelled rows")
    X = np.asarray([[float(row[c]) for c in columns] for row in rows], dtype=np.float64)
    y = np.asarray([float(row[label]) for row in rows], dtype=np.float64)
    if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isin(y, [0., 1.]).all():
        raise ValueError("CSV values must be finite; labels must be binary 0/1")
    return X, y.astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= len(args.features) <= 8 or not 1 <= args.layers <= 8:
        parser.error("Use 1..8 features and 1..8 layers")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y = read_data(args.csv, args.features, args.label)
    if min(np.bincount(y)) < 3:
        raise ValueError("Each class needs at least three observations")
    # Outer hold-out is excluded from both parameter selection and early stopping.
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=args.seed)
    circuit = StateVectorCircuit(num_qubits=len(args.features),
                                 num_layers=args.layers, entanglement="linear")
    rng = np.random.default_rng(args.seed)
    weights = rng.uniform(-0.3, 0.3, circuit.weight_shape)
    vqc = VariationalQuantumClassifier.training_candidate(circuit, weights, X_train)
    stats = vqc.fit(X_train, y_train, epochs=args.epochs,
                    validation_features=X_val, validation_labels=y_val)
    results = {}
    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.seed)),
        "rbf_svc": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", probability=True, random_state=args.seed)),
    }
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        pred = (p >= 0.5).astype(int)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "log_loss": float(log_loss(y_test, np.clip(p, 1e-7, 1 - 1e-7))),
        }
    p, outside = vqc.predict_proba(X_test)
    pred = (p >= 0.5).astype(int)
    results["vqc"] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "log_loss": float(log_loss(y_test, np.clip(p, 1e-7, 1 - 1e-7))),
        "outside_training_feature_range": int(np.sum(outside)),
    }
    provenance = {
        "dataset": str(args.csv), "feature_names": list(args.features),
        "label": args.label, "seed": args.seed,
        "split_sizes": {"train": len(X_train), "validation": len(X_val),
                        "test": len(X_test)},
        "model_selection_split": "validation",
        "test_used_for_training": False,
    }
    vqc.save(args.output, provenance=provenance)
    report = {"schema": "slai.qnn.vqc.experiment.v1",
              "artifact": str(args.output), "train": stats,
              "baselines_and_vqc": results, "provenance": provenance,
              "candidate_only": True}
    report_path = args.output.with_suffix(".evaluation.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
