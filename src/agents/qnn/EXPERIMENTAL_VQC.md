# SLAI v2.4 — Experimental supervised QNN classifier

This change adds a **CPU-only, classical state-vector simulation** of a supervised
binary variational quantum classifier. It is **not** a quantum-hardware backend,
a quantum advantage claim, a calibrated risk or success prediction, or a
production-approved model. The pre-existing state-vector modes remain intact.

## Scope and boundaries

- `qnn/variational_classifier.py`: angle feature map, one-qubit Born readout,
  differentiable supervised binary log loss, parameter-shift chain rule,
  bounded data validation, and NumPy archive with SHA-256 manifest.
- `qnn_agent.py`: an opt-in `classify` mode, `load_classifier`, and
  `classifier_health`. No inference updates production weights.
- `train_vqc.py`: explicit real-data CSV input; stratified train, validation,
  held-out test; logistic regression and RBF SVM comparisons.
- `tests/test_qnn_vqc.py`: focused initial tests. These tests must be run
  in a compatible SLAI environment before treating this change as verified.

## Execution example

From the root of SLAI with its pinned dependencies installed:

```bash
python -m pytest -q tests/test_qnn_vqc.py
python -m train_vqc --csv data.csv --label target --features feature_a feature_b --output artifacts/qnn/vqc_candidate.npz
```

A caller can then explicitly load an approved artifact and request classification:

```python
from src.agents.agent_factory import AgentFactory

factory = AgentFactory()
qnn = factory.create("qnn", config={"num_qubits": 2, "num_quantum_layers": 2, "entanglement": "linear"})
qnn.load_classifier("artifacts/qnn/vqc_candidate.npz")
result = qnn.perform_task({"mode": "classify", "features": [[0.5, 0.9]]})
```

This snippet illustrates the task contract, not evidence of a passing end-to-end
orchestration test. Direct QNN input still requires features in the same column
order and units as training; use the manifest provenance to validate the consumer.

## Supported collaboration

The existing Orchestrator can delegate a mapping to a core registered specialist
with `specialty="qnn"`, `action="perform_task"`, and
`task_data={"task_data": {"mode": "classify", "features": ...}}`.
The inner `task_data` is a keyword argument for the target method. No new
QNN-specific dispatcher has been introduced. Consumers must treat
`out_of_training_range` as a warning and must not treat Born outputs
as calibrated correctness probabilities.

## Remaining work before production

1. Verify tests and regression suites on the active v2.4 environment.
2. Validate schema-provenance binding in routing; a numeric matrix alone does
   not prove correct semantic feature identity/order.
3. Add model review, independent benchmark evidence and formal promotion.
4. Add failure containment, concurrency policy, secure deployment inventory,
   improved transactional artifact saving and comprehensive negative testing.
5. Demonstrate a real Orchestrator -> QNN -> downstream consumer execution
   trace and compare useful outcomes against classical no-QNN processing.
6. Audit QNN numerical invariants and package-wide documentation drift.
7. Enforce hard runtime budgets for training: dataset size, layers, epochs and
   parameter-shift evaluations jointly determine actual computational cost.

## Change isolation

The new implementation is proposed on
`audit/qnn-v24-integration-20261020` based on the v2.4 branch and does not
rewrite the legacy simulator. It is experimental until acceptance criteria
have been met.
