# SLAI v2.3 — Provenance Agent and Provenance Subsystem

## 1. Overview

The SLAI v2.3 Provenance capability is the system-wide infrastructure for recording and reconstructing **where artifacts came from and how they were derived**. It consists of the top-level `ProvenanceAgent` orchestration façade and the lower-level modules under `src/agents/provenance/`.

Its governing principle is:

> **Provenance records and reconstructs evidence of derivation. Provenance does not judge that evidence.**

The Provenance Agent coordinates capture and query operations for:

- artifact identity and ancestry;
- source identity and source references;
- derivation relationships;
- dataset lineage;
- model and checkpoint lineage;
- dependency lineage;
- transformation/activity history;
- chain-of-custody;
- durable provenance persistence;
- graph reconstruction and lineage queries;
- reproducibility evidence;
- checkpoint-provenance lifecycle state.

It answers questions such as:

- What produced this artifact?
- What was it derived from?
- Which sources contributed?
- Which transformations occurred?
- Which SLAI components participated?
- Which model or checkpoint contributed?
- Which dependencies were involved?
- What are the artifact's ancestors or descendants?
- Is there a derivation path between two known artifacts?
- Is enough provenance recorded to reconstruct the process?

It does **not** answer whether the evidence is trustworthy, correct, safe, high quality, or operationally healthy. Those judgments belong to other SLAI agents.

Unlike runtime logging or distributed tracing, provenance is artifact- and derivation-centric. Logs answer what a process reported while running; provenance records stable identities and causal derivation evidence that can be reconstructed later.

> **Implementation note**
> `ProvenanceStore` is the authoritative durable store for provenance facts. The Agent coordinates access to it; it does not reimplement storage, graph logic, lineage semantics, or custody rules.

---

## 2. Architectural position inside SLAI

A typical information flow may look like:

```text
Web page
   │
   ▼
BrowserAgent
   │
   ▼
ReaderAgent
   │
   ▼
QualityAgent
   │
   ▼
KnowledgeAgent
   │
   ▼
ReasoningAgent
   │
   ▼
LanguageAgent
   │
   ▼
Final output
```

Provenance runs across that flow rather than replacing any of those agents. The corresponding provenance identities may be recorded as:

```text
source-001
   │
   ▼
browser-artifact-014
   │
   ▼
reader-document-021
   │
   ▼
knowledge-fact-033
   │
   ▼
reasoning-event-104
   │
   ▼
language-output-055
```

The operational architecture is:

```text
Other SLAI Agents
        │
        │ provenance events / references
        ▼
   SharedMemory
        │
        ▼
 ProvenanceAgent
        │
        ├── SourceRegistry
        ├── ProvenanceLineage
        ├── LineageGraph
        ├── ProvenanceCustody
        ├── Reproducibility
        ├── ArtifactLineage
        ├── DatasetLineage
        ├── DependencyLineage
        ├── ModelLineage
        ├── TransformationLineage
        ├── ProvenanceStore
        └── ProvenanceMemory
                 │
                 └── checkpoint / local provenance lifecycle state
```

The ownership boundaries are:

```text
SharedMemory
    transient cross-agent coordination and lightweight references

ProvenanceAgent
    SLAI-wide provenance orchestration façade

ProvenanceStore
    durable provenance facts, relations, and graph evidence

ProvenanceMemory
    bounded local checkpoint-provenance lifecycle/index state

CheckpointManager
    physical/general SLAI checkpoint creation, verification, storage, and restore
```

### Relationship to other agents

| Component | Owns | Provenance relationship |
|---|---|---|
| `BrowserAgent` | Web navigation/acquisition | May emit stable source/artifact references that become provenance evidence. |
| `ReaderAgent` | Parsing and document extraction | Parsed documents can be registered as artifacts derived from acquired sources. |
| `QualityAgent` | Quality, suitability, reliability and trust-oriented assessment | Can consume provenance evidence, but provenance never computes quality or trust scores. |
| `KnowledgeAgent` | Retrieval, knowledge management, semantic search | Knowledge facts can carry derivation references; Provenance records how they were produced. |
| `ReasoningAgent` | Inference and reasoning | Reasoning outputs can be recorded as derived artifacts/activities. |
| `LanguageAgent` | Language generation and text processing | Final outputs can be linked to upstream reasoning, source, model and checkpoint evidence. |
| `VerificationAgent` | Formal properties, satisfiability, model checking | May consume provenance evidence; Provenance does not prove correctness. |
| `EvaluationAgent` | Empirical performance/evaluation | May consume provenance evidence; Provenance does not score performance. |
| `ObservabilityAgent` | Runtime telemetry, traces, health and incident analysis | Complementary to provenance; observability explains runtime behavior, not artifact ancestry. |
| `SafetyAgent` | Safety/risk/policy decisions | May consume provenance evidence; Provenance does not allow/block actions. |
| `CheckpointManager` | Physical SLAI checkpoints | Provenance records lineage *about* checkpoints after/around generic checkpoint operations. |
| `SharedMemory` | Cross-agent coordination | Carries lightweight provenance references/state notifications, not a duplicate provenance database. |

> **Boundary note**
> A source can be registered with an ID, locator, digest, external identifier, and metadata. `SourceRegistry` explicitly rejects fields such as `trust_score`, `reliability_score`, `quality_score`, and `credibility_score` because those meanings belong outside Provenance.

---

## 3. Current repository structure

The current Provenance hierarchy is:

```text
src/agents/
├── provenance_agent.py
└── provenance/
    ├── __init__.py
    ├── README.md
    ├── provenance_store.py
    ├── provenance_lineage.py
    ├── provenance_custody.py
    ├── provenance_memory.py
    ├── provenance_types.py
    │
    ├── configs/
    │   └── provenance_config.yaml
    │
    ├── modules/
    │   ├── __init__.py
    │   ├── lineage_graph.py
    │   ├── source_registry.py
    │   └── reproducibility.py
    │
    ├── lineage/
    │   ├── __init__.py
    │   ├── base_lineage.py
    │   ├── artifact_lineage.py
    │   ├── dataset_lineage.py
    │   ├── dependancy_lineage.py
    │   ├── model_lineage.py
    │   └── transformation_lineage.py
    │
    ├── templates/
    │   ├── alert-calls.json
    │   └── report.json
    │
    └── utils/
        ├── __init__.py
        ├── config_loader.py
        ├── provenance_errors.py
        └── provenance_helpers.py
```

`dependancy_lineage.py` is the current repository filename and is intentionally documented with that spelling for compatibility. The exported Python class is `DependencyLineage`.

### Module responsibilities

| Module | Responsibility |
|---|---|
| `provenance_agent.py` | Thin BaseAgent façade: configuration, SharedMemory coordination, event normalization, public APIs, checkpoint/lifecycle orchestration. |
| `provenance_store.py` | Thread-safe deterministic JSON-backed persistence and indexed retrieval of immutable provenance facts. |
| `provenance_lineage.py` | General lineage façade over the shared store. |
| `provenance_custody.py` | Append-only custody transitions and custody-history retrieval. |
| `provenance_memory.py` | Bounded local checkpoint-provenance manifest/index and restart state. |
| `provenance_types.py` | Frozen typed provenance records and PROV-compatible relation vocabulary. |
| `modules/lineage_graph.py` | Read-only deterministic graph traversal and reconstruction. |
| `modules/source_registry.py` | Stable source identity and metadata registration without quality/trust judgment. |
| `modules/reproducibility.py` | Structured assessment of whether sufficient provenance evidence exists for reconstruction. |
| `lineage/base_lineage.py` | Shared multi-parent lineage contract and PROV-compatible relation persistence. |
| `lineage/artifact_lineage.py` | Artifact-specific ancestry. |
| `lineage/dataset_lineage.py` | Dataset ancestry and transformation history. |
| `lineage/dependancy_lineage.py` | Artifact-to-dependency relationships. |
| `lineage/model_lineage.py` | Model/version/checkpoint/training provenance. |
| `lineage/transformation_lineage.py` | First-class transformations/activities and their input/output derivations. |
| `utils/provenance_errors.py` | Provenance-specific `BaseError` hierarchy. |
| `utils/provenance_helpers.py` | Deterministic timestamps, identifiers, serialization, hashes, normalization and graph-cycle helpers. |
| `utils/config_loader.py` | Provenance subsystem configuration loader. |

---

## 4. Responsibility boundaries

The main distinction is not whether another agent can *use* provenance. Many agents can. The distinction is who owns the semantic judgment.

| System | Primary question | Owns | Does not own |
|---|---|---|---|
| **Provenance** | Where did this artifact/result come from? | Identity, ancestry, derivation, sources, transformations, checkpoints, dependencies, custody, reconstruction evidence. | Trust, correctness, runtime health, safety policy, retrieval decisions. |
| **Observability** | What happened operationally? | Runtime traces, telemetry, health, incidents, operational diagnostics. | Artifact derivation authority. |
| **Knowledge** | What information is available? | Retrieval, knowledge indexing, semantic search, knowledge management. | System-wide derivation lineage. |
| **Quality** | Is this source/data suitable or reliable? | Quality/trust/reliability assessment. | Origin/derivation authority. |
| **Evaluation** | How well did the system perform? | Performance and empirical evaluation. | Provenance persistence. |
| **Verification** | Does the formal model satisfy a property? | Formal proof/refutation/model checking. | Provenance judgment. |
| **Safety** | Should this action/output be allowed? | Safety, risk and policy enforcement. | Provenance capture. |

> **Design rationale**
> Provenance may provide evidence to Quality, Evaluation, Verification, Safety, or Observability, but it must remain evidence infrastructure. This keeps the same derivation record usable by multiple downstream systems without baking one system's judgment into the provenance fact itself.

---

## 5. Provenance semantics

### 5.1 Entity, Activity and Agent

SLAI's typed model is compatible with the core semantic ideas of W3C PROV-DM and the Open Provenance Model without requiring RDF.

**Entity** — `ProvenanceEntity`

An identifiable thing whose provenance can be recorded. Examples include documents, datasets, model outputs, checkpoints, source files, intermediate representations, and software artifacts.

Current fields include:

```text
entity_id
entity_type
label
source_id
digest
created_at
metadata
```

**Activity** — `ProvenanceActivity`

A process/transformation that used or generated entities. Current fields include:

```text
activity_id
activity_type
started_at
ended_at
agent_id
parameters
metadata
```

**Agent reference** — `ProvenanceAgentRef`

A participating SLAI component or other actor reference. Current fields include:

```text
agent_id
agent_type
label
version
metadata
```

### 5.2 Relations

`ProvenanceRelationType` currently includes:

```text
used
wasGeneratedBy
wasDerivedFrom
wasAssociatedWith
wasAttributedTo
specializationOf
alternateOf
dependsOn
```

The subsystem uses these relations as lightweight PROV-compatible semantics. It does not require RDF serialization.

### 5.3 Multi-parent derivation

A `DerivationRecord` / `LineageRecord` contains `parent_artifact_ids`, not only a singular parent. A derived artifact can therefore depend on several inputs:

```text
document-18 ──────┐
                  │
rule-7 ───────────┼──► reasoning-event-103
                  │
checkpoint-4 ─────┘
                         │
                         ▼
                  final-output-42
```

`BaseLineage.record_lineage()` persists the derivation record and corresponding `wasDerivedFrom`, `wasGeneratedBy`, and `wasAttributedTo` relations where those facts are present.

> **Design rationale**
> Provenance is a graph/DAG rather than a simple parent tree because real SLAI outputs may combine several documents, rules, model states, dependencies, or transformations at once. A tree would lose that composition.

The store rejects lineage that would introduce a cycle, and `LineageGraph` independently guards traversal against malformed cyclic persisted data.

---

## 6. Domain-specific lineage

### 6.1 Artifact lineage

`ArtifactLineage` records ancestry for entities such as:

```text
source file
document
intermediate representation
generated response
checkpoint
software artifact
```

It ensures the output and known parents exist as entities, then delegates the actual multi-parent derivation semantics to `BaseLineage`.

### 6.2 Dataset lineage

`DatasetLineage` records dataset ancestry without assessing dataset quality or suitability.

A representative pipeline is:

```text
raw source
    ↓
scraped document
    ↓
normalized corpus
    ↓
deduplicated corpus
    ↓
training split
    ↓
curriculum
```

A dataset lineage record can carry:

```text
dataset_id
parent_dataset_id / parent_dataset_ids
transformation
transformation_id
version
agent_id
source_ids
timestamp
metadata
```

Dataset-specific details are embedded in lineage metadata through `DatasetRecord`; no parallel dataset database is introduced.

### 6.3 Model lineage

`ModelLineage` records model/version ancestry and optional checkpoint provenance. Supported information includes:

```text
model_id
parent_model_id / parent_model_ids
model_version
architecture
checkpoint_id
parent_checkpoint_id
training_run_id
training_dataset_ids
configuration_id
code_version
framework_versions
artifact_id
agent_id
metadata
```

Conceptually:

```text
model
    ├── parent model(s)
    ├── checkpoint
    ├── parent checkpoint
    ├── training run
    ├── training dataset(s)
    ├── configuration identity
    └── code version
```

Performance metrics are not model lineage and are not evaluated here.

### 6.4 Dependency lineage

`DependencyLineage` records immutable dependency relationships with fields such as:

```text
artifact_id
dependency_id
relationship
version
digest
timestamp
metadata
```

Examples:

```text
model       → tokenizer
checkpoint  → framework version
artifact    → package/version
training run→ dataset reference
```

The module creates a `dependsOn` relation but does not install dependencies or scan them for vulnerabilities.

### 6.5 Transformation lineage

`TransformationLineage` treats a transformation as a first-class provenance activity instead of only a free-form string. Supported inputs include:

```text
transformation_id
transformation_type / description
parent_transformation_id / parent_transformation_ids
input_ids
output_ids
agent_id
parameters
timestamp
metadata
```

Typical transformation labels include:

```text
parse
normalize
chunk
summarize
infer
train
fine-tune
generate
merge
convert
serialize
checkpoint
```

These are semantic examples, not an enforced enumeration. The implementation accepts identifiers/descriptions supplied by the caller and records what is actually known.

### 6.6 Chain-of-custody

Custody is append-only and event-oriented. `CustodyRecord` contains:

```text
event_id
artifact_id
previous_custodian
new_custodian
activity
timestamp
artifact_digest
context
```

For example:

```text
BrowserAgent
    ↓
ReaderAgent
    ↓
KnowledgeAgent
    ↓
LanguageAgent
```

Each transfer records a transition rather than replacing the current owner. If the caller supplies `previous_custodian` and it conflicts with the current recorded custodian, `ProvenanceCustodyError` is raised.

`ProvenanceAgent.chain_of_custody()` and `get_custody_history()` are read-only queries; they do not create new custody events.

---

## 7. Memory and persistence architecture

```text
                   ┌─────────────────────┐
                   │     SharedMemory     │
                   │ cross-agent refs /   │
                   │ coordination events  │
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │   ProvenanceAgent    │
                   │ orchestration layer  │
                   └──────┬───────┬──────┘
                          │       │
              ┌───────────┘       └────────────┐
              ▼                                ▼
      ┌────────────────┐              ┌────────────────┐
      │ ProvenanceStore│              │ProvenanceMemory│
      │ durable facts  │              │local checkpoint│
      │ + graph state  │              │lifecycle index │
      └────────────────┘              └────────────────┘
                                               │
                                               │ provenance about
                                               │ checkpoints
                                               ▼
                                      ┌─────────────────┐
                                      │CheckpointManager│
                                      │physical SLAI    │
                                      │checkpoints      │
                                      └─────────────────┘
```

### SharedMemory

Used for transient coordination only. The Agent publishes lightweight messages and references, including:

- event channel: `provenance.events` by default;
- runtime state key: `provenance_agent.state`;
- latest reference key: `provenance_agent.latest_reference`;
- state-update topic: `provenance_agent:state_updated`.

A published provenance reference has schema `provenance_agent.event.v1` and contains:

```json
{
  "schema": "provenance_agent.event.v1",
  "event_type": "artifact_registered",
  "agent_id": "ProvenanceAgent:<instance>",
  "timestamp": "<RFC3339 UTC timestamp>",
  "reference": {
    "artifact_id": "artifact-001",
    "artifact_type": "document"
  }
}
```

The full provenance graph is not copied into SharedMemory.

### ProvenanceStore

The store is authoritative for durable facts. It is thread-safe, JSON-backed, atomically persisted, and maintains lookup indexes for entities, lineage, child relationships, custody, dependencies, transformations, checkpoints and relations.

Its current schema is `slai.provenance.store.v1`. The current logical tables are:

```text
entities
activities
agents
relations
lineage_records
custody_records
sources
checkpoints
dependencies
transformations
```

### ProvenanceMemory

`ProvenanceMemory` is a bounded local checkpoint-provenance index and lifecycle/restart manifest. It is **not** another copy of the provenance graph.

Its current manifest schema is:

```text
slai.provenance.checkpoint-manifest.v2
```

Its compact `snapshot(include_checkpoints=False)` contains:

```text
schema_version
manifest_path
persist
revision
checkpoint_count
latest_checkpoint_id
updated_at
```

The full local checkpoint map is included only when `include_checkpoints=True`.

Current public lifecycle/index methods are:

```text
save_checkpoint(...)
get_checkpoint(...)
list_checkpoints(...)
latest_checkpoint(...)
checkpoint_ancestry(...)
prune_checkpoints(...)
snapshot(...)
restore()
flush()
close()
```

Pruning affects only the local manifest index. It does not delete the authoritative checkpoint fact from `ProvenanceStore`.

### CheckpointManager

`CheckpointManager` remains responsible for generic SLAI physical checkpoint creation, integrity verification, storage, selection and restore. `ProvenanceAgent.save_checkpoint()` delegates the physical save to `BaseAgent`/`CheckpointManager`, then records provenance about the committed checkpoint when `record_agent_checkpoints` is enabled.

> **Operational caution**
> Do not use SharedMemory or BaseAgent checkpoint state as a replacement for `ProvenanceStore`. Those layers carry coordination or restart references; the store remains the durable evidence authority.

---

## 8. ProvenanceAgent public API

The following methods are currently implemented on `ProvenanceAgent`.

### 8.1 Capture methods

| Method | Purpose | Mutates durable provenance? | SharedMemory publication |
|---|---|---:|---:|
| `register_source(source_id, source_info)` | Register/resolve stable source identity metadata. | Yes | Publishes a lightweight `source_registered` reference. |
| `register_artifact(artifact_id, ...)` | Register a typed provenance entity/artifact. | Yes | Publishes `artifact_registered`. |
| `record_derivation(artifact_id, ...)` | Record zero/multi-parent derivation and PROV-compatible relations. | Yes | Publishes `derivation_recorded`. |
| `record_transformation(transformation_id, **kwargs)` | Record a first-class transformation/activity and input/output derivation facts. | Yes | Publishes `transformation_recorded`. |
| `record_dependency(artifact_id, dependency_id, relationship, **kwargs)` | Record artifact dependency evidence and `dependsOn`. | Yes | Publishes `dependency_recorded`. |
| `record_dataset_lineage(dataset_id, ...)` | Record dataset ancestry. | Yes | Publishes `dataset_lineage_recorded`. |
| `record_model_lineage(model_id, **kwargs)` | Record model lineage and optional checkpoint evidence. | Yes | Publishes `model_lineage_recorded`. |
| `record_checkpoint(checkpoint, **kwargs)` | Add checkpoint provenance to the local lifecycle index and durable store. | Yes | Publishes `checkpoint_recorded`. |
| `record_custody(artifact_id, new_custodian, **kwargs)` | Append a custody transition. | Yes | Publishes `custody_recorded`. |
| `record_event(event)` | Normalize a generic cross-agent provenance event and dispatch it to one of the above operations. | Depends on event | Underlying capture method publishes the reference. |

All capture methods preserve the store's immutable stable-identity semantics. Provenance-specific exceptions are not silently converted into generic success values.

### 8.2 Query methods

| Method | Purpose | Mutates provenance? |
|---|---|---:|
| `track_artifact(artifact_id)` | Return all directly associated provenance evidence from `ProvenanceStore.get_provenance()`. | No |
| `get_provenance(artifact_id)` | Alias of `track_artifact()`. | No |
| `get_lineage(artifact_id, limit=None)` | Return direct lineage records, bounded by Agent query limits. | No |
| `get_ancestors(artifact_id, max_depth=None)` | Return deterministic ancestor IDs. | No |
| `get_descendants(artifact_id, max_depth=None)` | Return deterministic descendant IDs. | No |
| `get_derivation_path(source_id, output_id)` | Return one deterministic shortest source→output path, or `[]` when no path exists between known nodes. | No |
| `get_lineage_graph(artifact_id, depth=None, include_descendants=False)` | Return a bounded subgraph with `root`, `nodes`, `edges`, and lineage `records`. | No |
| `chain_of_custody(artifact_id)` | Return custody history. | No |
| `get_custody_history(artifact_id)` | Alias of `chain_of_custody()`. | No |
| `get_reproducibility_report(artifact_id)` | Return structured reconstruction-completeness evidence. | No |
| `provenance_state()` | Return compact Agent orchestration/lifecycle state. | No |
| `capabilities()` | Return advertised capture/query capabilities and SharedMemory/checkpointing availability. | No |

### 8.3 Aggregate artifact provenance

`track_artifact()` / `get_provenance()` currently returns the store's aggregate shape:

```text
artifact_id
entity
parents
children
lineage
custody
dependencies
transformations
checkpoints
sources
agents
activities
relations
```

If no provenance evidence is known for the identifier, `ProvenanceNotFoundError` is raised.

---

## 9. Cross-agent event normalization

`record_event()` accepts a mapping with this envelope:

```json
{
  "event_type": "derivation",
  "event_id": "optional-stable-event-id",
  "payload": {
    "artifact_id": "final-output-42",
    "parent_artifact_ids": ["fact-51", "rule-7"],
    "transformation": "reason",
    "agent_id": "ReasoningAgent"
  }
}
```

If `payload` is omitted, all fields other than `event_type`, `type`, and `event_id` are treated as the payload.

Supported canonical event types and aliases are:

| Canonical dispatch | Accepted event types |
|---|---|
| source | `source`, `source_registration` |
| artifact | `artifact`, `artifact_registration` |
| derivation | `derivation`, `lineage` |
| transformation | `transformation` |
| dependency | `dependency` |
| dataset | `dataset`, `dataset_lineage` |
| model | `model`, `model_lineage` |
| checkpoint | `checkpoint` |
| custody | `custody` |

Unknown event types are rejected with `ProvenanceValidationError`.

### Source event

```json
{
  "event_type": "source",
  "event_id": "source-event-001",
  "payload": {
    "source_id": "source-001",
    "source_info": {
      "source_type": "web",
      "url": "https://example.invalid/document",
      "digest": "sha256:<digest>",
      "external_identifier": null,
      "metadata": {}
    }
  }
}
```

With `strict_event_validation=true`, a source event must not mix `source_info` with extra top-level payload fields.

### Artifact event

```json
{
  "event_type": "artifact",
  "event_id": "artifact-event-014",
  "payload": {
    "artifact_id": "browser-artifact-014",
    "artifact_type": "document",
    "source_id": "source-001",
    "digest": "sha256:<digest>",
    "metadata": {}
  }
}
```

### Derivation / lineage event

```json
{
  "event_type": "derivation",
  "event_id": "reasoning-output-42",
  "payload": {
    "artifact_id": "final-output-42",
    "parent_artifact_ids": [
      "fact-51",
      "rule-7"
    ],
    "transformation": "reason",
    "agent_id": "ReasoningAgent",
    "checkpoint_id": "checkpoint-4",
    "metadata": {}
  }
}
```

### Transformation event

```json
{
  "event_type": "transformation",
  "event_id": "transform-event-001",
  "payload": {
    "transformation_id": "reader-parse-001",
    "transformation_type": "parse",
    "input_ids": ["browser-artifact-014"],
    "output_ids": ["reader-document-021"],
    "agent_id": "ReaderAgent",
    "parameters": {},
    "metadata": {}
  }
}
```

### Dependency event

```json
{
  "event_type": "dependency",
  "event_id": "dependency-event-001",
  "payload": {
    "artifact_id": "model-output-001",
    "dependency_id": "tokenizer-v5",
    "relationship": "tokenizer",
    "version": "5",
    "metadata": {}
  }
}
```

### Dataset-lineage event

```json
{
  "event_type": "dataset_lineage",
  "event_id": "dataset-event-012",
  "payload": {
    "dataset_id": "curriculum-v12",
    "parent_dataset_ids": ["training-split-v11"],
    "transformation": "build_curriculum",
    "version": "12",
    "agent_id": "LearningAgent",
    "source_ids": [],
    "metadata": {}
  }
}
```

### Model-lineage event

```json
{
  "event_type": "model_lineage",
  "event_id": "model-event-005",
  "payload": {
    "model_id": "lantra-v5",
    "parent_model_ids": ["lantra-v4"],
    "model_version": "5",
    "checkpoint_id": "checkpoint-B",
    "parent_checkpoint_id": "checkpoint-A",
    "training_run_id": "train-run-005",
    "training_dataset_ids": ["curriculum-v12"],
    "configuration_id": "training-config-44",
    "code_version": "9f31",
    "framework_versions": {},
    "agent_id": "LearningAgent",
    "metadata": {}
  }
}
```

### Checkpoint event

```json
{
  "event_type": "checkpoint",
  "event_id": "checkpoint-event-B",
  "payload": {
    "checkpoint_id": "checkpoint-B",
    "model_id": "lantra-v5",
    "parent_checkpoint_id": "checkpoint-A",
    "training_run_id": "train-run-005",
    "dataset_ids": ["curriculum-v12"],
    "configuration_id": "training-config-44",
    "code_version": "9f31",
    "framework_versions": {},
    "metadata": {}
  }
}
```

### Custody event

```json
{
  "event_type": "custody",
  "event_id": "custody-event-001",
  "payload": {
    "artifact_id": "reader-document-021",
    "new_custodian": "KnowledgeAgent",
    "previous_custodian": "ReaderAgent",
    "activity": "handoff",
    "context": {}
  }
}
```

> **Operational caution**
> `record_event()` normalizes only known provenance categories. It deliberately does not reinterpret Quality, Observability, Safety, or generic runtime events as provenance facts.

---

## 10. Idempotency, conflicts and graph integrity

Stable identity is an explicit provenance contract:

```text
same stable ID + identical immutable provenance
    → idempotent

same stable ID + conflicting immutable provenance
    → ProvenanceConflictError
```

This matters because cross-agent publication and retry mechanisms may deliver the same event more than once.

### Source identity

When an existing source is retried without a new registration timestamp, `ProvenanceAgent.register_source()` reuses the stored `registered_at` value so an otherwise identical retry remains idempotent.

### Artifact identity

When an existing artifact is retried without a `created_at`, the Agent reuses the stored creation timestamp. Reusing the artifact ID with incompatible content remains a conflict.

### Checkpoint identity

`ProvenanceMemory.save_checkpoint()` similarly reuses the existing creation time for an identical retry when a timestamp was omitted. Conflicting checkpoint provenance raises `ProvenanceConflictError`.

### Event identity

`record_event()` computes a deterministic payload fingerprint. When an explicit or generated `event_id` is found in the bounded event cache:

```text
same event_id + same fingerprint
    → cached result returned

same event_id + different fingerprint
    → ProvenanceConflictError
```

The event cache is an Agent-level retry aid, not the durable provenance database.

### Duplicate lineage and cycles

Stable lineage record IDs and store upsert semantics make identical derivation edges idempotent. The store rejects artifact lineage that would create a cycle. Checkpoint ancestry and transformation ancestry also reject cycles through their respective subsystem validation.

> **Compatibility note**
> Stable IDs should represent stable provenance meaning. Do not use a new random ID merely to bypass a conflict; resolve why two incompatible records are claiming the same identity.

---

## 11. Query behavior and graph safety

`LineageGraph` maintains lightweight parent/child adjacency views from persisted lineage records. It provides subsystem methods for:

```text
direct_parents() / parents()
children()
ancestors()
descendants()
derivation_path()
subgraph()
activities()
sources()
agents_involved()
get_lineage_graph()
```

The top-level Agent exposes bounded ancestry, descendant, derivation-path and subgraph methods and aggregate provenance retrieval.

### Determinism

- Parent and child IDs are sorted.
- Ancestor and descendant results are returned in deterministic sorted order.
- Derivation path uses deterministic breadth-first traversal and therefore returns one deterministic shortest path.
- Subgraph nodes/edges and records are deterministically ordered.

### Query bounds

Agent-level limits are configured through `agents_config.yaml`:

```text
max_query_depth
max_query_results
max_subgraph_depth
```

The subsystem's internal `LineageGraph` also has its own `lineage_graph.query.max_depth` setting in `provenance_config.yaml`.

The Agent never increases the subsystem's graph semantics; it applies an additional orchestration boundary.

### Cycle protection

Before traversal, `LineageGraph` checks the persisted parent graph and raises `ProvenanceGraphError` if a cycle is detected. Iterative traversal also maintains visited sets, so malformed data cannot cause unbounded recursion.

### Unknown artifacts

If an artifact is absent both from graph adjacency and from the entity store, graph queries raise `ProvenanceNotFoundError`.

### Result-size note

`get_ancestors()` and `get_descendants()` truncate returned IDs to `max_query_results`. `get_derivation_path()` rejects a path longer than that bound. `get_lineage()` is similarly bounded. `get_lineage_graph()` delegates to the bounded-depth subgraph query; its result is shaped by depth rather than a second graph implementation in the Agent.

---

## 12. Checkpointing and ProvenanceMemory

### Physical checkpoint versus checkpoint provenance

```text
CheckpointManager
    creates/verifies/stores/restores physical SLAI checkpoints

ProvenanceMemory
    records and indexes provenance about checkpoints
```

A checkpoint provenance record may contain:

```text
checkpoint_id
model_id
parent_checkpoint_id
training_run_id
dataset_ids
configuration_id
code_version
framework_versions
artifact_id
created_at
metadata
```

For example:

```text
checkpoint-C
    │
    ├── parent: checkpoint-B
    ├── model: lantra-v5
    ├── dataset: curriculum-v12
    ├── config: training-config-44
    └── code-version: 9f31...
```

`ProvenanceMemory.checkpoint_ancestry()` follows parent checkpoint IDs with cycle detection and a caller-supplied maximum depth. If a checkpoint was pruned from the local manifest, lookup can fall back to the authoritative `ProvenanceStore`.

### ProvenanceAgent BaseAgent checkpoint state

`ProvenanceAgent` declares `CHECKPOINTING_SUPPORTED = True` and uses BaseAgent's component-based checkpoint contract. The Agent checkpoint schema is:

```text
slai.provenance-agent.state.v3
```

The Agent exports only compact orchestration state:

```text
schema_version
processed_events
failed_events
last_event_id
last_event_timestamp
last_agent_checkpoint_id
local_memory
```

The `local_memory` field is a compact snapshot/reference, not a copy of all checkpoint records. The complete `ProvenanceStore` is intentionally **not** serialized into the generic Agent checkpoint.

When `record_agent_checkpoints=true`, a successfully committed generic SLAI checkpoint is then represented as provenance, linked to the previous Agent checkpoint when available.

> **Implementation note**
> If physical checkpoint creation succeeds but subsequent provenance recording fails, `ProvenanceAgent.save_checkpoint()` raises `ProvenanceStorageError` and marks persistence degraded. This makes the gap visible instead of silently pretending the checkpoint was fully provenance-recorded.

---

## 13. Reproducibility semantics

Within the Provenance subsystem, reproducibility means:

> **Is enough recorded provenance available to reconstruct the artifact or process?**

It does **not** mean:

> Is the artifact correct, high quality, safe, scientifically valid, or performant?

`Reproducibility.reproducibility_report()` currently records these checks:

```text
source_identity_known
source_available
code_version_known
model_checkpoint_known
dataset_version_known
configuration_known
dependencies_known
transformations_complete
environment_known
```

The resulting `ReproducibilityReport` contains:

```text
artifact_id
source_identity_known
source_available
code_version_known
model_checkpoint_known
dataset_version_known
configuration_known
dependencies_known
transformations_complete
environment_known
reproducible
completeness
required_requirements
missing_requirements
evidence
```

Current subsystem configuration requires these fields for the final `reproducible` decision:

```text
source_identity_known
configuration_known
dependencies_known
transformations_complete
environment_known
```

Other checks remain visible evidence even when they are not in the configured required set.

`completeness` is a fraction of required provenance requirements satisfied. It is a **provenance completeness ratio**, not a source-quality or model-evaluation score.

For remote locators such as HTTP/HTTPS, the subsystem does not perform availability probing. Remote availability is true only when availability has been explicitly recorded in provenance metadata. This keeps network/runtime probing outside Provenance.

---

## 14. Error model

All Provenance-specific failures derive from `ProvenanceError`, which itself participates in SLAI's `BaseError` architecture.

| Error | Code | Meaning |
|---|---|---|
| `ProvenanceConfigurationError` | `PROV-1100` | Invalid/inconsistent provenance subsystem configuration. |
| `ProvenanceValidationError` | `PROV-1200` | Malformed or structurally invalid provenance input. |
| `ProvenanceGraphError` | `PROV-1201` | Cyclic, malformed or internally inconsistent provenance graph/ancestry. |
| `ProvenanceConflictError` | `PROV-1202` | Stable identity reused with incompatible immutable content. |
| `ProvenanceNotFoundError` | `PROV-1203` | Requested provenance artifact/source/checkpoint/record is unknown. |
| `ProvenanceTrackingError` | `PROV-1300` | General provenance capture/reconstruction failure. |
| `ProvenanceLineageError` | `PROV-1301` | Lineage capture/reconstruction failure. |
| `ProvenanceSourceError` | `PROV-1302` | Source registration or source-identity failure. |
| `ProvenanceReproducibilityError` | `PROV-1303` | Reproducibility evidence could not be reconstructed/interpreted. |
| `ProvenanceStorageError` | `PROV-1400` | Persistence/retrieval/local-manifest failure. |
| `ProvenanceCustodyError` | `PROV-1500` | Custody continuity/capture/retrieval failure. |

Callers should catch the most specific type they can meaningfully handle. Conflicts, graph violations, and storage failures should not be swallowed as successful retries.

The `alert-calls.json` template maps common provenance-integrity/lifecycle situations to these current errors or to explicit reproducibility/report conditions. It is a catalog only; it does not add an alert execution engine.

---

## 15. Configuration

### Agent-level orchestration configuration

`ProvenanceAgent` reads only the `provenance_agent` section of:

```text
src/agents/base/configs/agents_config.yaml
```

Current fields are:

| Field | Current default/configured value | Purpose |
|---|---:|---|
| `enabled` | `true` | Enable/disable Agent operations. |
| `publish_shared_memory` | `true` | Enable lightweight SharedMemory reference/state publication. |
| `publish_state_updates` | `true` | Publish compact runtime state. |
| `fail_on_shared_memory_error` | `false` | Whether coordination-layer write/publish errors fail the Agent operation. |
| `strict_event_validation` | `true` | Enforce unambiguous generic event shapes, especially source events. |
| `shared_memory_ttl_seconds` | `86400` | TTL for SharedMemory `set()` references; `0` means no explicit TTL. |
| `max_query_depth` | `64` | Agent-level bound for ancestor/descendant traversal. |
| `max_query_results` | `2048` | Agent-level result bound for lineage/ancestor/descendant/path APIs. |
| `max_subgraph_depth` | `16` | Agent-level depth bound for subgraph reconstruction. |
| `local_memory_max_checkpoints` | `1000` | Maximum checkpoint references retained in the local ProvenanceMemory manifest. |
| `event_cache_size` | `256` | Bounded Agent-level event retry/idempotency cache size. |
| `record_agent_checkpoints` | `true` | Record provenance for committed BaseAgent/CheckpointManager checkpoints. |
| `event_channel` | `provenance.events` | SharedMemory pub/sub channel for lightweight provenance references. |
| `state_key` | `provenance_agent.state` | SharedMemory key for compact Agent state. |
| `latest_reference_key` | `provenance_agent.latest_reference` | SharedMemory key for the latest lightweight provenance reference. |

### Subsystem-internal configuration

The lower-level Provenance subsystem uses:

```text
src/agents/provenance/configs/provenance_config.yaml
```

Current sections include:

```text
provenance_store
provenance_memory
provenance_lineage
provenance_custody
lineage
artifact_lineage
dataset_lineage
dependency_lineage
model_lineage
transformation_lineage
source_registry
lineage_graph
reproducibility
```

The separation is intentional:

```text
agents_config.yaml
    → ProvenanceAgent orchestration policy

provenance_config.yaml
    → internal Provenance subsystem implementation settings
```

`provenance_agent.py` does not import or access `provenance_config.yaml` or `provenance.utils.config_loader` directly.

> **Boundary note**
> Do not move graph/storage algorithm settings into `agents_config.yaml` merely because the Agent uses the subsystem. Agent policy and subsystem implementation configuration are separate ownership layers.

---

## 16. Logging

Provenance modules use SLAI's existing logging infrastructure from:

```text
logs/logger.py
```

with:

```python
get_logger
PrettyPrinter
configure_logging
```

Normal imports do not configure a separate logging subsystem. `configure_logging()` is used only in explicit module smoke/entry-point blocks where appropriate.

Operational logs are intended for concise events such as initialization, rejected provenance operations, persistence problems, or SharedMemory coordination degradation. Large provenance payloads should remain in the store/reporting layer rather than being dumped into logs.

Logging and provenance remain distinct:

```text
logging
    operational messages about software execution

provenance persistence
    durable derivation evidence and stable references
```

Observability remains responsible for runtime tracing/telemetry semantics.

---

## 17. Academic foundation

The implementation is academically informed without becoming a literal RDF/PROV stack.

- **Moreau et al. (2011), Open Provenance Model** — provenance graphs, causal process/artifact/agent relationships, interoperability across heterogeneous systems.
- **W3C PROV-DM** — Entity/Activity/Agent semantics and relations such as `used`, `wasGeneratedBy`, `wasDerivedFrom`, `wasAssociatedWith`, and `wasAttributedTo`.
- **W3C PROV Constraints** — consistency, ordering, invalid relations, and cycle/structural validation principles.
- **Cheney, Chiticariu & Tan (2009)** — provenance taxonomy, derivation/origin semantics, and interpretation of source contribution.
- **Buneman, Khanna & Tan (2001)** — why/where provenance, data ancestry, and source origin.
- **Green, Karvounarakis & Tannen (2007)** — compositional/multi-parent provenance; SLAI applies the graph idea without forcing semiring mathematics into the code.
- **Muniswamy-Reddy et al. (2006), PASS** — infrastructure-level persistent provenance associated with stable artifacts.
- **Sandve et al. (2013), ReproZip, and reproducible-build literature** — configuration, dependency, environment and reconstruction metadata.
- **Vartak et al. (2016), ModelDB; Souza et al. / PROV-ML** — model/checkpoint/training-run/data/configuration lineage.
- **Gebru et al., Datasheets for Datasets; Longpre et al.** — dataset creation/source/reuse history, without importing quality judgment into lineage.
- **Software Heritage / Di Cosmo** — stable/content-addressed software artifact identity concepts.
- **in-toto** — explicit artifact transformation chains, actor/process attribution, dependency and custody semantics.

The practical consequence is a lightweight deterministic Python representation that remains compatible with established provenance ideas while fitting SLAI v2.3's existing architecture.

---

## 18. End-to-end example A — web-derived final statement

Assume SLAI acquires one web source, parses it, extracts a fact, reasons over it, and generates a final answer.

```text
URL
 ↓
BrowserAgent
 ↓
browser-artifact-014
 ↓
ReaderAgent
 ↓
reader-document-021
 ↓
KnowledgeAgent
 ↓
knowledge-fact-033
 ↓
ReasoningAgent
 ↓
reasoning-event-104
 ↓
LanguageAgent
 ↓
language-output-055
```

A provenance capture sequence can use stable IDs:

```python
provenance.register_source(
    "source-001",
    {
        "source_type": "web",
        "url": "https://example.invalid/document",
        "digest": "sha256:<digest>",
    },
)

provenance.register_artifact(
    "browser-artifact-014",
    artifact_type="document",
    source_id="source-001",
)

provenance.record_transformation(
    "reader-parse-001",
    transformation_type="parse",
    input_ids=["browser-artifact-014"],
    output_ids=["reader-document-021"],
    agent_id="ReaderAgent",
)

provenance.record_derivation(
    "knowledge-fact-033",
    parent_artifact_ids=["reader-document-021"],
    transformation="extract_fact",
    agent_id="KnowledgeAgent",
    source_ids=["source-001"],
)

provenance.record_derivation(
    "reasoning-event-104",
    parent_artifact_ids=["knowledge-fact-033"],
    transformation="reason",
    agent_id="ReasoningAgent",
)

provenance.record_derivation(
    "language-output-055",
    parent_artifact_ids=["reasoning-event-104"],
    transformation="generate",
    agent_id="LanguageAgent",
)
```

The final output can then be reconstructed through:

```python
provenance.get_ancestors("language-output-055")
provenance.get_derivation_path("browser-artifact-014", "language-output-055")
provenance.track_artifact("language-output-055")
```

`LineageGraph` can additionally expose participating activities, source records, and agent references for the ancestry closure when used by subsystem-level reporting code.

What Provenance can establish is the derivation chain and recorded participants. Whether `source-001` is trustworthy remains a Quality concern; whether the final statement is correct remains an Evaluation/Verification concern.

---

## 19. End-to-end example B — model/checkpoint lineage

A training lineage may be:

```text
raw documents
      ↓
processed dataset
      ↓
LANTRA curriculum
      ↓
training run
      ↓
checkpoint-A
      ↓
checkpoint-B
      ↓
model output
```

Dataset ancestry can be recorded through `DatasetLineage` / `record_dataset_lineage()`. Model/checkpoint ancestry can then be captured with:

```python
provenance.record_model_lineage(
    "lantra-v5",
    parent_model_ids=["lantra-v4"],
    model_version="5",
    architecture="LANTRA",
    checkpoint_id="checkpoint-B",
    parent_checkpoint_id="checkpoint-A",
    training_run_id="train-run-005",
    training_dataset_ids=["curriculum-v12"],
    configuration_id="training-config-44",
    code_version="9f31",
    framework_versions={"python": "3.12"},
    artifact_id="lantra-v5",
    agent_id="LearningAgent",
)
```

The `ModelLineage` service persists model lineage and the checkpoint fact in the shared `ProvenanceStore`. `ProvenanceAgent` also indexes the checkpoint in its `local_memory` for lifecycle/restart use.

Checkpoint ancestry is available through:

```python
provenance.local_memory.checkpoint_ancestry("checkpoint-B")
```

The physical checkpoint file itself remains under `CheckpointManager` ownership. `ProvenanceMemory` only records the identity/ancestry needed to explain that checkpoint.

A reproducibility report for a downstream model output may combine evidence about source identity, configuration, dependencies, transformations, environment, dataset version, code version, and checkpoint presence without rating model quality.

---

## 20. Development and integration guidance

Future SLAI agents that participate in provenance should follow these rules:

1. **Use stable IDs.** Artifact, source, checkpoint and event IDs should identify the same provenance meaning across retries.
2. **Prefer the ProvenanceAgent boundary.** Other agents should normally communicate through public Agent APIs and/or SharedMemory coordination rather than importing internal provenance modules.
3. **Emit references, not payload replicas.** SharedMemory events should carry stable IDs and concise metadata instead of full documents/models/graphs.
4. **Record only known evidence.** If a source, parent, model, or checkpoint is unknown, do not fabricate one to make the graph look complete.
5. **Use multi-parent derivation when appropriate.** Do not collapse several real contributors into one artificial parent.
6. **Keep judgments elsewhere.** Do not attach trust/quality/safety/correctness meaning to provenance fields.
7. **Make retries idempotent.** Retrying the same event with the same stable identity should produce the same provenance fact.
8. **Treat conflicts as data-integrity signals.** A stable ID reused with incompatible evidence should be corrected at the producer, not overwritten.
9. **Keep checkpoint ownership clear.** `CheckpointManager` stores physical checkpoints; Provenance records what produced or relates to them.
10. **Keep observability separate.** Runtime latency, failures, traces and service-health details belong to Observability even when a provenance operation is involved.

> **Implementation note**
> The current `ProvenanceAgent` constructs all provenance services once and injects one shared `ProvenanceStore` instance into them. Future integrations should preserve that single-store authority rather than creating per-call stores or shadow databases.

---

## 21. Operational templates

Two non-executable templates accompany this README:

```text
src/agents/provenance/templates/alert-calls.json
src/agents/provenance/templates/report.json
```

### `alert-calls.json`

A catalog of provenance-integrity and lifecycle conditions that may require operator/agent attention. It maps conditions to the current Provenance error hierarchy or reproducibility-state evidence. It is **not** an Observability incident engine and does not contain runtime-health, CPU, latency, safety or quality alerts.

### `report.json`

A canonical report skeleton for exporting a concise provenance reconstruction. It uses stable IDs/references and mirrors the shapes exposed by `ProvenanceStore`, `LineageGraph`, `ReproducibilityReport`, `CustodyRecord`, `CheckpointRecord`, and `ProvenanceMemory.snapshot()` without embedding whole artifact/model payloads.

Both templates are deliberately data-only JSON and contain no comments, secrets, machine-specific paths, or executable policy.

---

## 22. Summary of invariants

```text
Provenance tells SLAI:

"What produced this?"
"What was it derived from?"
"What transformed it?"
"Which source/model/checkpoint/dependency contributed?"
"Which agent/component participated?"
"Can the derivation path be reconstructed?"
"Is enough provenance present for reconstruction?"

Provenance does NOT tell SLAI:

"Is this source trustworthy?"
"Is this output correct?"
"Was runtime healthy?"
"Should this action be allowed?"
"What knowledge should be retrieved?"
```

That separation is the primary architectural contract for every future Provenance integration in SLAI v2.3.
