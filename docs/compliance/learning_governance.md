# Learning governance

Implemented in `governance/learning_governance.py` (`LearningRegistry`, `LearningDeletionAdapter`). Tests: `test_learning_governance.py`, `test_model_governance.py`.

* Learning is a separate purpose (`model_training`, `model_fine_tuning`, `model_evaluation`). A dataset or training run is registered only with an
  ALLOW / ALLOW_WITH_CONTROLS governance decision for that purpose; a serving-purpose approval is refused.
* Lineage: datasets -> training runs -> model versions. Membership is stored as salted digests, never identifiers. Runs with unknown datasets,
  cross-tenant datasets or no datasets are refused.
* **Erasure is not unlearning.** Erasing a subject removes dataset membership and opens an *impact* for every model trained on it. The erasure request
  stays `incomplete` until each impact is closed with recorded evidence: `retrained_without_subject`, `unlearning_verified`, `exempt_documented`
  or `not_personal_data_in_model`. Nothing is ever marked unlearned automatically.
* Model approval requires: no prohibited classification, no open classification questions, no open impacts, registered lineage and (for GPAI)
  complete Art. 53/55 documentation. Unknown training compute is treated as systemic-risk until shown otherwise. An approved model with open impacts reports `remediation_required`.

Not implemented: the actual retraining/unlearning machinery and the call sites in `learning_agent.py` (agents must call the registry); the GPAI documents themselves are organisational.
