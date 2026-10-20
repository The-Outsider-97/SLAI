"""Focused supervised QNN tests; execute from SLAI repository root."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.agents.qnn_agent import StateVectorCircuit, QNNAgent
from src.agents.qnn.variational_classifier import VariationalQuantumClassifier


class VQCTests(unittest.TestCase):
    def setUp(self):
        self.circuit = StateVectorCircuit(
            num_qubits=2, num_layers=1, entanglement="linear")
        self.weights = np.zeros(self.circuit.weight_shape)

    def test_born_bounds_and_feature_shapes(self):
        X = np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]])
        model = VariationalQuantumClassifier.training_candidate(
            self.circuit, self.weights, X)
        p, shifted = model.predict_proba(X)
        self.assertEqual(p.shape, (4,))
        self.assertTrue(np.all((p >= 0.) & (p <= 1.)))
        self.assertFalse(shifted.any())
        _, shifted = model.predict_proba([[2., 0.]])
        self.assertTrue(shifted[0])
        with self.assertRaises(ValueError):
            model.predict_proba([[float("nan"), 1.]])
        with self.assertRaises(ValueError):
            model.predict_proba([[0., 1., 2.]])

    def test_binary_gradient_training_and_artifact(self):
        X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        y = np.array([0, 1, 1, 0])
        model = VariationalQuantumClassifier.training_candidate(
            self.circuit, self.weights, X)
        details = model.fit(X, y, epochs=2, patience=2)
        self.assertEqual(details["steps"], 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.npz"
            model.save(path, provenance={"kind": "synthetic_xor_smoke"})
            loaded = VariationalQuantumClassifier.load(self.circuit, path)
            np.testing.assert_allclose(
                model.predict_proba(X)[0], loaded.predict_proba(X)[0])
            raw = path.read_bytes()
            path.write_bytes(raw + b"tampered")
            with self.assertRaises(ValueError):
                VariationalQuantumClassifier.load(self.circuit, path)

    def test_invalid_training_labels(self):
        X = [[0., 0.], [1., 1.]]
        model = VariationalQuantumClassifier.training_candidate(
            self.circuit, self.weights, X)
        with self.assertRaises(ValueError):
            model.fit(X, [0, 2], epochs=1)
        with self.assertRaises(ValueError):
            model.fit(X, [0, 0], epochs=1)

    def test_agent_classification_requires_approved_artifact(self):
        # This is a component-level integration test, not end-to-end orchestration.
        agent = QNNAgent(config={
            "num_qubits": 2, "num_quantum_layers": 1,
            "entanglement": "linear"})
        self.assertEqual(
            agent.perform_task({"mode": "classify", "features": [[0., 1.]]})["status"],
            "unavailable")
        X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        model = VariationalQuantumClassifier.training_candidate(
            agent._circuit, agent.quantum_weights, X)
        model.fit(X, [0, 1, 1, 0], epochs=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.npz"
            model.save(path, provenance={"kind": "synthetic_test"})
            agent.load_classifier(str(path))
            output = agent.perform_task({"mode": "classify", "features": [[0., 1.]]})
            self.assertEqual(output["status"], "success")
            self.assertFalse(output["probabilities_calibrated"])
            self.assertEqual(len(output["labels"]), 1)


if __name__ == "__main__":
    unittest.main()
