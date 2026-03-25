# Data Governance

`data/governance.py` introduces schema enforcement and reproducibility controls for multimodal ingestion.

## Modular Config + Error Utilities

- `data/utils/config_loader.py` for YAML-backed configuration loading.
- `data/utils/data_error.py` for typed data-domain exceptions.
- `data/configs/data_config.yaml` for ingestion/validation/quality/versioning policy.

## Dataset Schemas and Validation Layer

Define modality-specific schemas with:

- `DatasetField(name, expected_type, nullable=False, max_items=None)`
- `DatasetSchema(name, version, modality, fields)`

`DatasetValidator` enforces:

- Required fields and type checks
- Nullability constraints
- Size constraints (`max_records`, per-field `max_items`)
- Modality alignment (`vision/text/audio` length parity)

## Quality Gates

`quality_gate(payload)` performs full validation and returns:

- Total records
- Modalities included
- Null-count diagnostics per modality
- Null ratio
- Validation timestamp

Hard failures raise typed `Data*Error` exceptions and should block ingestion.

## Versioning and Lineage

`DatasetVersionRegistry` writes version entries to `data/processed/dataset_registry.json` with:

- Dataset identity (`dataset_name`, `dataset_version`)
- Source metadata (`source_uri`, `source_commit`, `transform_id`)
- Canonical payload hash (configurable hash algorithm)
- Modalities and record count

This enables reproducibility and traceability of every dataset build.

## Ingestion Contracts

`data/multimodal_dataset.py` enforces ingestion contracts at dataset creation and per-batch iteration:

- Configurable minimum batch size
- Null checks and type constraints (when validator supplied)
- Modality length alignment
- Size and schema conformance
