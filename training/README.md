# SLAI LANTRA Training Enrichment

`src/training/` is an offline integration layer between LANTRA training and SLAI's existing Knowledge, Reasoning, and Perception agents. It does **not** reimplement retrieval, ontology management, reasoning, validation, perception encoders, checkpointing, or LANTRA's document extractors.

## Boundary

```text
data/library + configured corpus paths
        │
        ▼
train_lantra.py canonical discovery/extraction/chunking
        │
        ▼
source-document split assignment (before enriched examples)
        │
        ├───────────────┬──────────────────┐
        ▼               ▼                  ▼
 KnowledgeAgent   ReasoningAgent    PerceptionAgent (optional)
 retrieval        validation        frozen text representation
 ontology         forward chain     semantic-hardness filter
        │               │                  │
        └───────────────┴──────────────────┘
                        │
                        ▼
             LantraCurriculumBuilder
                        │
                        ▼
data/processed/lantra/agent_enriched/
  ├── phase_2a_generation.jsonl
  ├── phase_2b_embedding.jsonl
  ├── phase_2b_reranking.jsonl
  ├── phase_2c_generation.jsonl
  ├── phase_2c_classification.jsonl
  └── manifest.json
                        │
                        ▼
                  train_lantra.py
```

All emitted JSONL records use the existing LANTRA task enum: `generation`, `classification`, `embedding`, or `reranking`. No new task identifier is required.

## Intended phases

- **2A** — ordinary raw denoising remains in `train_lantra.py`; this builder adds ontology-aware span/relation reconstruction as compatible `generation` examples.
- **2B** — KnowledgeAgent mines same-split hard negatives for LANTRA's existing `embedding` and `reranking` objectives.
- **2C** — ontology facts are admitted to factual/reasoning examples only after ReasoningAgent's public validation contract accepts them. Configured forward-chaining rules may add validated inferred conclusions; if no applicable rules exist, no inference examples are fabricated.
- **2D (optional)** — a restored PerceptionAgent checkpoint can filter hard negatives using frozen text representations. The current `train_lantra.py` has no teacher-representation loss, so this package does **not** pretend to perform representation distillation. Teacher similarity is recorded as metadata and used only for curriculum acceptance.
- **3** — existing real supervised specialization remains owned by `train_lantra.py`.

## Leakage prevention

The builder assigns complete source documents to `train`, `validation`, or `test` **before** creating derived examples. Every record contains explicit `split` and `metadata.source_document_ids`.

Pairwise retrieval examples are accepted only when anchor, positive, and negative documents belong to the same partition. Global exact segment deduplication prevents repeated boilerplate from surviving in multiple partitions.

This is important because `train_lantra.py` currently splits ordinary supervised records at example level when no explicit split is supplied. The generated records therefore always carry explicit group-safe splits.

## Quality policy

`TrainingQualityGate` validates every generated record by calling the active `train_lantra.normalize_record()` implementation. The trainer remains the schema authority.

The gate rejects:

- unsupported task schemas;
- missing/invalid explicit splits;
- cross-split source-document references;
- exact duplicate curriculum payloads;
- over-sized records;
- reasoning-gold examples that are not fully validated;
- reasoning-negative examples that are not fully validated as unsupported.

Reasoning status is fail-closed for gold supervision. `partial`, `failed`, `indeterminate`, conflicts, and degraded validation do not become positive labels.

## Run

From the SLAI repository root:

```powershell
py build_lantra_curriculum.py
```

Useful pilot run:

```powershell
py build_lantra_curriculum.py --max-documents 250 --max-segments 5000 --max-retrieval-pairs 1000 --force
```

Inspect source partitioning without constructing agents or writing output:

```powershell
py build_lantra_curriculum.py --dry-run
```

Enable Perception only with a known SLAI checkpoint version:

```powershell
py build_lantra_curriculum.py `
  --enable-perception `
  --perception-checkpoint-version <CHECKPOINT_VERSION>
```

The Perception adapter calls `PerceptionAgent.restore_checkpoint(version, verify_integrity=True)`. It does not accept arbitrary weight-file paths or heuristic parameter conversion.

## Integration with the current trainer

The output directory is under `data/processed/lantra/`, which is already one of `train_lantra.py`'s default supervised discovery roots. Therefore the current trainer can consume the generated JSONL files without task-enum changes.

Current limitation: until `train_lantra.py` gains stage-aware supervised scheduling, the generated 2A/2B/2C examples are read during its supervised multitask stage. `metadata.curriculum_phase` and `metadata.curriculum_stages` preserve the intended future scheduling boundary without falsifying a phase that the current trainer does not yet implement.

## Modules

| Module | Responsibility |
|---|---|
| `enrichment_contracts.py` | typed data/config contracts and deterministic IDs |
| `source_adapter.py` | reuse LANTRA's existing canonical source extraction/chunking |
| `knowledge_adapter.py` | index canonical segments, ontology lookup, hard-negative mining |
| `reasoning_adapter.py` | fail-closed fact validation and optional public forward chaining |
| `perception_adapter.py` | optional frozen representation scoring from restored checkpoint |
| `training_quality_gate.py` | trainer-schema validation, leakage and duplicate guards |
| `curriculum_builder.py` | phase construction, coverage policy, atomic artifact writing |

## Reproducibility and caching

The manifest fingerprints:

- source file SHA-256 values;
- curriculum configuration;
- training-layer code/config resources;
- Knowledge/Reasoning resources;
- configured ontology DB where discoverable;
- Perception agent/checkpoint metadata when enabled.

A matching manifest is reused only when each recorded artifact still matches its SHA-256. Use `--force` to rebuild intentionally.
