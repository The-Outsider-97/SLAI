"""VQC contract tests: classifier mechanics, artifact safety and trainer data governance.

Synthetic arrays and archives appear here only to exercise code paths (no network, no training
data for any shipped model); the trainer itself never fabricates rows.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

import train_vqc
from src.agents.qnn.utils.quantum_errors import (
    QNNCheckpointStateError, QNNConfigurationError, QNNDimensionError, QNNError, QNNGradientError,
    QNNInputError, QNNResourceLimitError,
)
from src.agents.qnn.variational_classifier import VariationalQuantumClassifier as VQC
from src.agents.qnn_agent import StateVectorCircuit


def circuit(qubits: int = 2, layers: int = 1, entanglement: str = "linear") -> StateVectorCircuit:
    return StateVectorCircuit(num_qubits=qubits, num_layers=layers, entanglement=entanglement)


def separable(rows: int = 80, seed: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (rows, 2))
    return X, (X[:, 0] + 0.25 * X[:, 1] > 0.0).astype(int)


class ClassifierTests(unittest.TestCase):
    def candidate(self, **kwargs):
        c = circuit(**kwargs)
        X, y = separable()
        weights = np.random.default_rng(0).uniform(-0.3, 0.3, c.weight_shape)
        return c, VQC.training_candidate(c, weights, X), X, y

    def test_batch_route_equals_per_example_forward(self):
        c, model, X, _ = self.candidate()
        states = model._encode(X)
        batch = model._probabilities(states, model.weights)
        single = np.array([model._probabilities(states[i:i + 1], model.weights)[0] for i in range(len(X))])
        np.testing.assert_allclose(batch, single, atol=1e-12)

    def test_parameter_shift_gradient_matches_finite_difference(self):
        _, model, X, y = self.candidate()
        states = model._encode(X)
        p = np.clip(model._probabilities(states, model.weights), 1e-7, 1 - 1e-7)
        gradient = model._gradient(states, (p - y) / (p * (1 - p)))
        loss = lambda w: VQC.binary_log_loss(model._probabilities(states, w), y)
        for index in np.ndindex(model.weights.shape):
            plus, minus = model.weights.copy(), model.weights.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            self.assertAlmostEqual(gradient[index], (loss(plus) - loss(minus)) / 2e-5, places=6)

    def test_fit_improves_loss_and_keeps_best_weights(self):
        _, model, X, y = self.candidate()
        stats = model.fit(X, y, epochs=20, learning_rate=0.2, class_weight="balanced")
        self.assertLess(stats["train_loss"], stats["initial_train_loss"])
        self.assertEqual(model.training_steps, stats["steps"])

    def test_failed_fit_restores_weights_and_steps(self):
        _, model, X, y = self.candidate()
        before, steps = model.weights.copy(), model.training_steps
        calls = {"n": 0}
        original = model._probabilities

        def flaky(states, weights):
            calls["n"] += 1
            if calls["n"] > 40:
                raise RuntimeError("simulated circuit failure")
            return original(states, weights)

        with patch.object(model, "_probabilities", flaky), self.assertRaises(QNNGradientError):
            model.fit(X, y, epochs=10)
        np.testing.assert_array_equal(model.weights, before)
        self.assertEqual(model.training_steps, steps)

    def test_fit_refuses_features_the_readout_cannot_see(self):
        c = circuit(entanglement="none")
        X, y = separable()
        model = VQC.training_candidate(c, np.zeros(c.weight_shape), X)
        with self.assertRaises(QNNConfigurationError):
            model.fit(X, y, epochs=2)

    def test_input_validation_uses_typed_errors(self):
        _, model, X, y = self.candidate()
        with self.assertRaises(QNNDimensionError):
            model.predict_proba([[0.0, 1.0, 2.0]])
        with self.assertRaises(QNNResourceLimitError):
            model.predict_proba(np.zeros((model.limits.max_inference_rows + 1, 2)))
        with self.assertRaises(QNNInputError):
            model.fit(X, np.zeros(len(y)))
        with self.assertRaises(QNNConfigurationError):
            model.fit(X, y, epochs=True)
        with self.assertRaises(QNNConfigurationError):
            model.fit(X, y, optimizer="rmsprop")
        with self.assertRaises(QNNInputError):
            model.fit(X, y, validation_features=X)
        with self.assertRaises(ValueError):  # taxonomy stays ValueError-compatible
            model.predict_proba([[np.nan, 0.0]])

    def test_constructor_rejects_non_circuits_and_missing_config(self):
        with self.assertRaises(QNNConfigurationError):
            VQC(8, 128, [], [])
        c = circuit()
        with self.assertRaises(QNNConfigurationError):
            VQC(c, np.zeros(c.weight_shape), [0, 0], [1, 1], config={"vqc": {}})

    def test_out_of_range_rows_are_flagged(self):
        _, model, _, _ = self.candidate()
        _, outside = model.predict_proba([[0.0, 0.0], [99.0, 0.0]])
        self.assertEqual(outside.tolist(), [False, True])

    def test_save_load_roundtrip_and_tamper_detection(self):
        c, model, X, y = self.candidate()
        model.fit(X, y, epochs=3)
        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "m.npz", provenance={"source": Path("some/where.csv")})
            manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["provenance"]["source"], str(Path("some/where.csv")))
            self.assertEqual(manifest["readout"], "born_p_qubit1_one")
            reloaded = VQC.load(c, path)
            np.testing.assert_allclose(reloaded.predict_proba(X[:6])[0], model.predict_proba(X[:6])[0])
            self.assertEqual(reloaded.training_steps, model.training_steps)

            with self.assertRaises(QNNCheckpointStateError):
                VQC.load(circuit(layers=2), path)  # circuit schema mismatch
            tampered = Path(directory) / "t.npz"
            tampered.write_bytes(path.read_bytes() + b"\0")
            tampered.with_suffix(".json").write_bytes(path.with_suffix(".json").read_bytes())
            with self.assertRaises(QNNCheckpointStateError):
                VQC.load(c, tampered)
            with self.assertRaises(QNNInputError):
                VQC.load(c, Path(directory) / "missing.npz")
            path.with_suffix(".json").write_text("{not json", encoding="utf-8")
            with self.assertRaises(QNNCheckpointStateError):
                VQC.load(c, path)

    def test_save_rejects_bad_path_and_unserialisable_provenance(self):
        _, model, _, _ = self.candidate()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(QNNInputError):
                model.save(Path(directory) / "m.bin")
            with self.assertRaises(QNNInputError):
                model.save(Path(directory) / "m.npz", provenance={"bad": object()})
            self.assertEqual(list(Path(directory).iterdir()), [])  # nothing half-written

    def test_archive_with_unexpected_members_is_rejected(self):
        c, model, X, y = self.candidate()
        model.fit(X, y, epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            path = model.save(Path(directory) / "m.npz")
            buffer = io.BytesIO()
            np.savez_compressed(buffer, weights=model.weights, feature_min=model.feature_min,
                                feature_max=model.feature_max, extra=np.zeros(3))
            forged = buffer.getvalue()
            path.write_bytes(forged)
            manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            manifest["sha256"] = train_vqc.sha(forged)
            path.with_suffix(".json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(QNNCheckpointStateError):
                VQC.load(c, path)


def wine_archive(rows: dict[str, int] | None = None, header: list[str] | None = None) -> bytes:
    rows = rows or train_vqc.WINE["expected_rows"]
    rng = np.random.default_rng(5)
    header = header or train_vqc.WINE["features"] + ["quality"]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for kind, count in rows.items():
            out = io.StringIO()
            writer = csv.writer(out, delimiter=";", lineterminator="\n", quoting=csv.QUOTE_ALL)
            writer.writerow(header)
            for _ in range(count):
                row = [f"{v:.4f}" for v in rng.uniform(0.1, 10.0, len(header) - 1)] + [str(int(rng.integers(3, 9)))]
                out.write(";".join(row) + "\n")
            archive.writestr(f"wine+quality/{train_vqc.WINE['members'][kind]}", out.getvalue())
    return buffer.getvalue()


class TrainerGovernanceTests(unittest.TestCase):
    def namespace(self, root: Path, **overrides) -> argparse.Namespace:
        args = train_vqc.build_parser().parse_args(["collect"])
        args.raw_dir = str(root / "raw")
        args.prepared_dir = str(root / "prepared")
        args.registry = str(Path("data/approved_datasets.json"))
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_parse_wine_archive_validates_structure(self):
        rows, counts = train_vqc.parse_wine_archive(wine_archive(), "combined")
        self.assertEqual(counts, {"red": 1599, "white": 4898})
        self.assertEqual(len(rows), 6497)
        self.assertEqual({r[-1] for r in rows}, {0.0, 1.0})
        red, _ = train_vqc.parse_wine_archive(wine_archive(), "red")
        self.assertEqual(len(red), 1599)
        with self.assertRaises(ValueError):
            train_vqc.parse_wine_archive(wine_archive({"red": 10, "white": 4898}), "combined")
        with self.assertRaises(ValueError):
            train_vqc.parse_wine_archive(wine_archive(header=["a"] * 12), "combined")
        with self.assertRaises(ValueError):
            train_vqc.parse_wine_archive(b"not a zip", "combined")

    def test_download_is_https_and_host_pinned(self):
        with self.assertRaises(ValueError):
            train_vqc.download_archive("http://archive.ics.uci.edu/x.zip", "archive.ics.uci.edu", max_bytes=10, timeout=1)
        with self.assertRaises(ValueError):
            train_vqc.download_archive("https://evil.example/x.zip", "archive.ics.uci.edu", max_bytes=10, timeout=1)

        class Redirected:
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def geturl(self): return "https://evil.example/x.zip"
            def read(self, n): return b"x"

        with patch("urllib.request.urlopen", return_value=Redirected()), self.assertRaises(ValueError):
            train_vqc.download_archive("https://archive.ics.uci.edu/x.zip", "archive.ics.uci.edu", max_bytes=10, timeout=1)

    def test_registry_collect_records_provenance_and_detects_upstream_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "wine.zip"
            archive.write_bytes(wine_archive())
            args = self.namespace(root, archive=str(archive))
            manifest = train_vqc.collect_registry(args)
            self.assertEqual(manifest["license"], "CC BY 4.0")
            self.assertEqual(manifest["subset"], "combined")
            self.assertEqual(manifest["archive_sha256"], train_vqc.sha(archive.read_bytes()))
            self.assertIn("doi.org/10.24432/C56S3T", manifest["attribution"])
            _, columns, table = train_vqc.load_raw(root / "raw")
            self.assertEqual(table.shape, (6497, 13))
            self.assertEqual(columns[-2:], ["quality", "is_white"])

            changed = root / "other.zip"
            changed.write_bytes(wine_archive() + b"\0")  # different bytes, same content
            with self.assertRaises(ValueError):
                train_vqc.collect_registry(self.namespace(root, archive=str(changed)))
            with self.assertRaises(ValueError):
                train_vqc.collect_registry(self.namespace(root, archive=str(archive), expected_sha256="0" * 64, force=True))

    def test_prepare_deduplicates_before_splitting_and_selects_on_train_only(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = rng.normal(size=(150, 5))
            label = (base[:, 0] + 0.3 * rng.normal(size=150) > 0).astype(int)
            table = np.column_stack([base, label])
            table = np.vstack([table, table[:50]])  # 50 exact duplicates
            source = root / "d.csv"
            with source.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([f"f{i}" for i in range(5)] + ["y"])
                writer.writerows(table.tolist())
            args = self.namespace(root, source="csv", csv=str(source), label="y", source_name="synthetic-test",
                                  source_license="internal", features=[f"f{i}" for i in range(5)])
            train_vqc.collect(args)
            args.features = None
            train_vqc.prepare(args)
            manifest, split = train_vqc.load_prepared(root / "prepared")
            self.assertEqual(manifest["duplicates_removed"], 50)
            self.assertEqual(manifest["rows_unique"], 150)
            self.assertEqual(split["train"]["X"].shape[1], 4)
            seen = [set(map(tuple, split[name]["X_all"].round(9))) for name in ("train", "validation", "test")]
            self.assertFalse(seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
            self.assertEqual(manifest["selected_features"][0], "f0")  # the only informative feature

    def test_gate_requires_beating_the_prior_with_confidence(self):
        good = {"test": {"vqc": {"log_loss": 0.5, "outside_fitted_bounds_rate": 0.01},
                         "baselines": {"prior_predictor": {"log_loss": 0.69}},
                         "bootstrap": {"paired_difference_ci95": {"prior_predictor": [0.05, 0.2]}}}}
        self.assertTrue(train_vqc.apply_gate(good, 0.1)["passed"])
        good["test"]["bootstrap"]["paired_difference_ci95"]["prior_predictor"] = [-0.01, 0.2]
        self.assertFalse(train_vqc.apply_gate(good, 0.1)["passed"])

    def test_bootstrap_separates_informative_from_constant_predictions(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 400)
        informative = np.where(y == 1, 0.8, 0.2)
        result = train_vqc.bootstrap(y, informative, {"prior": np.full(400, y.mean())}, draws=300, seed=1)
        self.assertGreater(result["paired_difference_ci95"]["prior"][0], 0.0)

    def test_main_returns_nonzero_with_message_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            code = train_vqc.main(["prepare", "--raw-dir", str(Path(directory) / "nothing")])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
