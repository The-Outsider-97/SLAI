# SLAI v2.4 Perception — multimodal model training and integration

## Status / scope

Two **model interfaces** are now exposed with exact names:
`MultimodalPerception` and `MultimodalContrastiveEncoder`. They reuse
`TextPerception`, `VisionPerception`, `AudioPerception`,
`PerceptionFusion`, `PerceptionObjectives`, and `PerceptionTrainer`.
They do **not** instantiate duplicate neural backbones. `MultimodalPerception`
supports a fixed-width fusion representation and existing task heads.
`MultimodalContrastiveEncoder` outputs normalized embeddings and computes
symmetric multi-positive InfoNCE via the existing projector layer.

**Do not infer production readiness from import success, training-loop
completion, synthetic fixture tests, or nonzero embeddings.** Fresh agent
initialization uses untrained weights. Real-world performance requires
independent held-out evaluation. The named model response describes whether an
explicit training artifact was loaded. It does not claim metric validity.

Existing responsibilities retained: three modality pipelines; masked
pretraining; supervised downstream heads; temporal objectives; central
CheckpointManager integration; SharedMemory lifecycle; and policy-controlled
routed inference. External agent consumers should use the existing
`PerceptionAgent.perform_task` contract rather than importing modality
internals into Language, Knowledge, Reasoning, or Planning.

## Audit and verified corrections

- **Training/inference mismatch:** the old image inference transform omitted
  ImageNet mean/std normalization used by training. Both now use the same
  deterministic resize/crop/to-tensor/normalize sequence.
- **Invalid width restriction:** agent validation required every encoder's
  `embed_dim` to equal the fusion output size despite `PerceptionFusion`
  already containing projection layers. The contradictory requirement was removed.
- **Missing named model capabilities:** existing contrastive projectors and
  fusion could train, but there were no dedicated, discoverable models exposing
  aligned retrieval embeddings or explicit model selection. Both names now route
  through the Perception Agent and reuse canonical parameters.
- **False-negative supervision:** repeated semantic positives in a paired
  batch were previously treated as independent negatives. Optional
  `group_ids` now supports multi-positive supervised InfoNCE; default
  one-to-one InfoNCE behavior remains backward compatible.
- **Unbounded media inputs:** preprocessing now limits decoded input file size
  and image pixel count before allocating large images.
- **Training provenance and leakage:** opt-in real-data acquisition,
  user-declared rights, split/source group validation, protected relative
  paths, resumable HTTP acquisition, archived SHA-256 reporting, dataset
  fingerprints, checkpoint SHA-256 and weights-only loading are implemented.
- **Dynamic head restoration:** re-created downstream heads now also update
  the model view's reference so it cannot hold stale parameters.

The initial audit also identified older `TrainingManager` and
`InferenceEngine` compatibility classes in `data_loader.py`. They were
not removed because external callers may still use them; the new training
runner uses the canonical `PerceptionTrainer` instead of a duplicate
training owner.

## Real dataset — Oxford-IIIT Pet (optional)

Official source:
https://www.robots.ox.ac.uk/~vgg/data/pets/

The University of Oxford Visual Geometry Group publishes the archive and
annotations for 37 pet breeds (~800 MB). The project page presents a CC BY-SA
4.0 dataset license while stating that the photographs retain their original
owners' copyright. Review the actual source terms and intended use before
downloading or training, and retain required attribution. The runner does not
make a legal judgment. It does not download data without an explicit command.

The official data contains real **images and breed annotations**, not
human-written natural-language descriptions. This adapter turns breed names
into **weak image–text labels**. It can exercise alignment and classification,
but is **not** a substitute for image-caption retrieval benchmarking.
`positive_id` groups same-breed semantics for multi-positive InfoNCE;
`group_id` separates individual source examples across dataset splits.

From the SLAI root with the project virtual environment active:

```powershell
cd G:\GAIA\AI\SLAI
python -m src.agents.perception.train_models acquire-oxford-pets `
    --dataset-root data\training\perception\oxford_pets `
    --confirm-rights --extract

python -m src.agents.perception.train_models oxford-pets-manifest `
    --dataset-root data\training\perception\oxford_pets `
    --output data\training\perception\oxford_pets\paired.jsonl
```

Acquisition logs downloaded archive sizes and SHA-256 digests, but there is
no independently published upstream hash in the adapter. Thus
`upstream_checksum_verified=false`; a computed local digest alone is not
proof of upstream integrity. Extraction rejects links, path traversal and
excessive archive members.

For a **user-supplied approved dataset**, make a UTF-8 JSONL manifest with
one record per paired sample. At least train, validation and test splits are
required:

```json
{"id":"sample-001","group_id":"source-001","positive_id":"class-cat","split":"train","text":"a cat","vision":"images/cat001.jpg","label":0}
```

Keys: `id` is unique; `group_id` tracks source examples and must never
cross splits; `positive_id` optionally groups semantically equivalent
examples within a split, but may recur across splits; `label` is the
integer classification target required for `MultimodalPerception`.
Only relative in-root image/audio paths are accepted. For audio–text
training use `audio` instead of `vision`. For contrastive training, text
pairs must come from actually aligned samples.

Prepare the manifest provenance (example uses the Oxford dataset):

```powershell
python -m src.agents.perception.train_models prepare `
    --manifest data\training\perception\oxford_pets\paired.jsonl `
    --dataset-root data\training\perception\oxford_pets `
    --pair text vision `
    --source-url https://www.robots.ox.ac.uk/~vgg/data/pets/ `
    --license "CC BY-SA 4.0; respect original image rights" `
    --confirm-rights `
    --output data\training\perception\oxford_pets\dataset_declaration.json
```

The declaration records **operator-confirmed rights**, not independent
license verification. The preparation step refuses missing, invalid, or
leaking real examples; it does not invent samples or labels.

## Train and independently evaluate both models

```powershell
$root = "data\training\perception\oxford_pets"
$manifest = "$root\paired.jsonl"
$declaration = "$root\dataset_declaration.json"

python -m src.agents.perception.train_models train `
    --model MultimodalContrastiveEncoder `
    --manifest $manifest --dataset-root $root `
    --dataset-declaration $declaration --pair text vision `
    --device cpu --batch-size 8 --epochs 3 `
    --output "$root\contrastive_model"

python -m src.agents.perception.train_models evaluate `
    --model MultimodalContrastiveEncoder `
    --manifest $manifest --dataset-root $root `
    --dataset-declaration $declaration --pair text vision `
    --checkpoint "$root\contrastive_model" --split test

python -m src.agents.perception.train_models train `
    --model MultimodalPerception `
    --manifest $manifest --dataset-root $root `
    --dataset-declaration $declaration --pair text vision `
    --device cpu --batch-size 8 --epochs 3 `
    --output "$root\perception_model"

python -m src.agents.perception.train_models evaluate `
    --model MultimodalPerception `
    --manifest $manifest --dataset-root $root `
    --dataset-declaration $declaration --pair text vision `
    --checkpoint "$root\perception_model" --split test

python -m src.agents.perception.train_models infer `
    --model MultimodalContrastiveEncoder `
    --checkpoint "$root\contrastive_model" `
    --modality text --text "a siamese cat"
```

For an interrupted job, rerun `train` with `--resume` and `--epochs`
set to the intended **total** epoch count. The implementation checks model
identity, architecture tensor shapes, dataset fingerprint and checkpoint hash
before resuming. Checkpoint files live outside source control.

## Programmatic Perception Agent inference

```python
from src.agents.perception_agent import PerceptionAgent

agent = PerceptionAgent(shared_memory=None, agent_factory=None)
agent.load_training_artifact("data/training/perception/oxford_pets/contrastive_model")
result = agent.perform_task({
    "task_type": "inference",
    "model_name": "MultimodalContrastiveEncoder",
    "modality": "text",
    "input_data": "a cat",
})
embedding = result["output"]["text"]  # torch.Tensor (batch, projection_dim)
```

To perform multimodal task inference, load a compatible
`MultimodalPerception` artifact, pass
`{"modality":"multimodal","input_data":{"text": [...], "vision": tensor}}`,
and optionally name a restored `downstream_task`. For aligned cross-modal
embeddings specify `similarity_pair:["text","vision"]` and inspect the
cosine similarity matrix. A checkpoint trained for one modality pair does not
establish calibrated alignment for other pairs.

## Verification

```powershell
python -m compileall -q src\agents\perception src\agents\perception_agent.py
python -m pytest -q tests\agents\test_perception_models_v24.py
python -m pytest -q tests\agents\test_perception_training_manifest_v24.py
python -m pytest -q
```

These tests check tensor contracts, missing modalities, loss gradients,
multi-positive treatment, numerical learning on synthetic fixtures, and
manifest security. They do not constitute a real-world retrieval or
classification benchmark. Record independent test-set Recall@1/5 or accuracy,
CPU/GPU latency, memory, malformed-input robustness, and SLAI end-to-end
routing evidence before making production claims.

**Remaining blockers:** no actual dataset has been acquired or trained in
this repository change; no measured real-data checkpoint evaluation or
hardware-specific performance claim is supplied. Audio–text and image–audio
training require genuine paired corpora and separate evaluation. Optional
cross-agent consumption remains through established agent envelopes rather
than new direct dependencies.
