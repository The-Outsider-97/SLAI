"""Reproducible, offline-by-default training for the two SLAI perception models.

This runner deliberately uses the existing PerceptionAgent, PerceptionTrainer,
modality preprocessing and checkpoint state contracts. It never manufactures
training labels or claims readiness based on synthetic samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any
import shutil
import tarfile
from urllib.request import Request, urlopen

import torch
from torch.utils.data import Dataset, DataLoader

from ..perception_agent import PerceptionAgent
from .data_loader import MultiModalDataset
from .perception_contracts import Modality


SCHEMA = "slai.perception.training-artifact.v1"
MODALITIES = ("text", "vision", "audio")
MODELS = ("MultimodalPerception", "MultimodalContrastiveEncoder")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError("A real local JSONL dataset manifest is required.")
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL record at line {lineno}.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Record at line {lineno} must be a mapping.")
        records.append(row)
    if not records:
        raise ValueError("Dataset is empty; no synthetic examples will be generated.")
    return records


def _safe_path(root: Path, raw: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ValueError("Modality paths must be non-empty paths relative to the dataset root.")
    target = (root / raw).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file():
        raise ValueError("Missing or out-of-root modality path in dataset.")
    return target


def validate_manifest(manifest: Path, root: Path, modalities: tuple[str, str]) -> tuple[list[dict], dict]:
    if len(set(modalities)) != 2 or any(m not in MODALITIES for m in modalities):
        raise ValueError("Exactly two distinct supported modalities are required.")
    rows = _read_records(manifest)
    counts = {"train": 0, "validation": 0, "test": 0}
    ids: set[str] = set()
    groups: dict[str, str] = {}
    file_splits: dict[str, str] = {}
    labels = set()
    for row in rows:
        split = row.get("split")
        rid = row.get("id")
        group = row.get("group_id", rid)
        positive_id = row.get("positive_id", rid)
        if not isinstance(positive_id, str) or not positive_id:
            raise ValueError("positive_id must be a non-empty string.")
        if split not in counts:
            raise ValueError("Every record needs split=train, validation, or test.")
        if not isinstance(rid, str) or not rid or rid in ids:
            raise ValueError("Every record needs a distinct non-empty string id.")
        if not isinstance(group, str) or not group:
            raise ValueError("group_id must be a non-empty string.")
        if group in groups and groups[group] != split:
            raise ValueError("Group/source leakage between dataset splits.")
        groups[group] = split
        ids.add(rid)
        counts[split] += 1
        for modality in modalities:
            value = row.get(modality)
            if modality == "text":
                if not isinstance(value, str) or not value.strip() or len(value) > 10000:
                    raise ValueError("Text must be non-empty and at most 10,000 characters.")
            else:
                path = _safe_path(root, value)
                if path.stat().st_size > (32 if modality == "vision" else 64) * 1024 * 1024:
                    raise ValueError("An image/audio input exceeds the decoder input size limit.")
                key = str(path)
                if key in file_splits and file_splits[key] != split:
                    raise ValueError("File reused across train/validation/test partitions.")
                file_splits[key] = split
        label = row.get("label")
        if label is not None:
            if isinstance(label, bool) or not isinstance(label, int) or label < 0:
                raise ValueError("Optional classification labels must be non-negative integers.")
            labels.add(label)
    if not counts["train"] or not counts["validation"] or not counts["test"]:
        raise ValueError("Independent train, validation and test splits are required.")
    return rows, {"counts": counts, "num_groups": len(groups),
                  "labels": sorted(labels), "modalities": list(modalities),
                  "manifest_sha256": _sha256(manifest)}


class ManifestPerceptionDataset(MultiModalDataset):
    """Use the existing SLAI image/audio transforms; raw text goes to TextPerception."""

    def __init__(self, records: list[dict], root: Path, split: str, pair: tuple[str, str]):
        super().__init__(mode="val" if split == "validation" else split)
        self.records = [r for r in records if r["split"] == split]
        self.root = root
        self.pair = pair

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        payload = {}
        for modality in self.pair:
            if modality == "text":
                payload["text"] = row["text"]
            elif modality == "vision":
                payload["vision"] = self._process_vision(_safe_path(self.root, row["vision"]))
            else:
                payload["audio"] = self._process_audio(_safe_path(self.root, row["audio"]))
        return {"payload": payload, "id": row["id"],
                "group_id": row.get("group_id", row["id"]),
                "positive_id": row.get("positive_id", row["id"]),
                "label": row.get("label")}


def _collate(samples: list[dict]) -> dict[str, Any]:
    pair = tuple(samples[0]["payload"])
    payload = {}
    for modality in pair:
        values = [record["payload"][modality] for record in samples]
        payload[modality] = values if modality == "text" else torch.stack(values)
    labels = [sample["label"] for sample in samples]
    return {"payload": payload, "group_ids": tuple(str(r["positive_id"]) for r in samples),
            "labels": torch.tensor(labels, dtype=torch.long) if all(isinstance(x, int) for x in labels) else None}


def _loader(records: list[dict], root: Path, split: str, pair: tuple[str, str], batch_size: int):
    data = ManifestPerceptionDataset(records, root, split, pair)
    return DataLoader(data, batch_size=batch_size, shuffle=split == "train",
                      num_workers=0, collate_fn=_collate, drop_last=False)


def _agent(device: str) -> PerceptionAgent:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available.")
    return PerceptionAgent(shared_memory=None, agent_factory=None,
                           config={"perception_agent": {"device": device}})


def _prepare(args: argparse.Namespace) -> None:
    manifest, root = Path(args.manifest).resolve(), Path(args.dataset_root).resolve()
    _, summary = validate_manifest(manifest, root, tuple(args.pair))
    if not args.source_url or not args.license or not args.confirm_rights:
        raise ValueError("Explicit dataset provenance, license and --confirm-rights are required.")
    # User declaration is not an independent license verification.
    summary.update({"source_url": args.source_url, "declared_license": args.license,
                    "rights_confirmed_by_operator": True, "license_independently_verified": False})
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _validated_data(args: argparse.Namespace):
    manifest, root = Path(args.manifest).resolve(), Path(args.dataset_root).resolve()
    pair = tuple(args.pair)
    records, summary = validate_manifest(manifest, root, pair)
    declaration = Path(args.dataset_declaration)
    if not declaration.is_file():
        raise ValueError("Run prepare and supply its dataset declaration first.")
    metadata = json.loads(declaration.read_text(encoding="utf-8"))
    if (metadata.get("manifest_sha256") != summary["manifest_sha256"]
            or metadata.get("modalities") != list(pair)
            or metadata.get("rights_confirmed_by_operator") is not True):
        raise ValueError("Dataset declaration or manifest fingerprint mismatch.")
    summary["dataset_provenance"] = {
        "source_url": metadata.get("source_url"),
        "declared_license": metadata.get("declared_license"),
        "license_independently_verified": False,
    }
    return records, summary, root


def _preprocessing_identity(agent: PerceptionAgent) -> dict:
    """Pin the token vocabulary and lower-level preprocessing/encoder config."""
    tokenizer = agent.text_perception.tokenizer
    vocab = getattr(tokenizer, "vocab", {})
    serialized = json.dumps(vocab, sort_keys=True, ensure_ascii=False).encode("utf-8")
    config_path = Path(__file__).resolve().parent / "configs" / "perception_config.yaml"
    return {
        "tokenizer_vocab_sha256": hashlib.sha256(serialized).hexdigest(),
        "subsystem_config_sha256": _sha256(config_path),
    }


def _save(agent: PerceptionAgent, folder: Path, *, model: str, epoch: int,
          summary: dict, pair: tuple[str, str], metrics: dict,
          training_config: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    state = {"schema": SCHEMA, "model_name": model, "epoch": epoch, "pair": list(pair),
             "manifest_sha256": summary["manifest_sha256"],
             "training_config": training_config,
             "preprocessing_identity": _preprocessing_identity(agent),
             "trained_model_names": (
                 list(MODELS) if model == "MultimodalPerception"
                 and training_config.get("joint_contrastive_weight", 0) > 0
                 else [model]
             ),
             "model_state_dict": agent.state_dict(),
             "trainer_state_dict": agent.trainer.state_dict(),
             "task_head_specs": agent._task_head_specs}
    temporary = folder / "model.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(folder / "model.pt")
    metadata = {"schema": SCHEMA, "model_name": model, "epoch": epoch,
                "pair": list(pair), "manifest_sha256": summary["manifest_sha256"],
                "dataset_provenance": summary.get("dataset_provenance"),
                "training_config": training_config,
                "optimizer_config": agent.trainer.state_dict().get("optimizer_config"),
                "preprocessing_identity": _preprocessing_identity(agent),
                "trained_model_names": (
                    list(MODELS) if model == "MultimodalPerception"
                    and training_config.get("joint_contrastive_weight", 0) > 0
                    else [model]
                ),
                "model_sha256": _sha256(folder / "model.pt"), "metrics": metrics,
                "validated_real_world_performance": False}
    temp_json = folder / "manifest.json.tmp"
    temp_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temp_json.replace(folder / "manifest.json")


def load_into_agent(agent: PerceptionAgent, folder: Path, *,
                    expected_model: str | None = None, expected_dataset: str | None = None,
                    restore_optimizer: bool = False) -> dict:
    metadata = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA or metadata.get("model_name") not in MODELS:
        raise ValueError("Unsupported perception training artifact.")
    if expected_model and expected_model not in metadata.get("trained_model_names", []):
        raise ValueError("Checkpoint does not contain the requested trained model.")
    if expected_dataset and metadata.get("manifest_sha256") != expected_dataset:
        raise ValueError("Checkpoint training data fingerprint mismatch.")
    if metadata.get("preprocessing_identity") != _preprocessing_identity(agent):
        raise ValueError("Tokenizer or subsystem architecture configuration changed since training.")
    if _sha256(folder / "model.pt") != metadata.get("model_sha256"):
        raise ValueError("Training artifact SHA-256 integrity check failed.")
    # weights_only refuses arbitrary pickle object graphs from untrusted data.
    state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    if (state.get("schema") != SCHEMA or state.get("model_name") != metadata["model_name"]
            or state.get("pair") != metadata["pair"] or state.get("epoch") != metadata["epoch"]
            or state.get("preprocessing_identity") != metadata["preprocessing_identity"]
            or state.get("trained_model_names") != metadata["trained_model_names"]):
        raise ValueError("Training artifact metadata disagrees with the checkpoint.")
    if any(name not in MODELS for name in metadata["trained_model_names"]):
        raise ValueError("Training artifact declares an unsupported trained model.")
    # Preflight the complete parameter schema before mutation.
    agent._restore_task_head_structure(state.get("task_head_specs", []))
    target = agent.state_dict()
    candidate = state.get("model_state_dict", {})
    if set(target) != set(candidate) or any(target[k].shape != candidate[k].shape for k in target):
        raise ValueError("Checkpoint is incompatible with the PerceptionAgent architecture.")
    agent.load_state_dict(candidate, strict=True)
    agent.trainer.load_state_dict(state.get("trainer_state_dict", {}), load_optimizer=restore_optimizer)
    # An artifact replaces *all* shared encoder/fusion/projector weights.
    # Older loaded model statuses must never outlive those weights.
    agent._trained_model_artifacts.clear()
    for model_name in metadata["trained_model_names"]:
        agent._trained_model_artifacts[model_name] = {
            "model_sha256": metadata["model_sha256"],
            "pair": list(metadata["pair"]),
            "validated_real_world_performance": metadata.get("validated_real_world_performance", False),
        }
    return metadata


def _evaluate(agent: PerceptionAgent, records: list[dict], root: Path, pair: tuple[str, str],
              model: str, split: str, batch_size: int) -> dict:
    agent.eval()
    all_left, all_right, groups = [], [], []
    correct = total = 0
    with torch.inference_mode():
        for batch in _loader(records, root, split, pair, batch_size):
            payload = batch["payload"]
            if model == "MultimodalContrastiveEncoder":
                embeddings = agent._model_views[model](payload)
                all_left.append(embeddings[Modality.parse(pair[0])].cpu())
                all_right.append(embeddings[Modality.parse(pair[1])].cpu())
                groups.extend(batch["group_ids"])
            else:
                result = agent.perform_task({
                    "task_type": "inference", "model_name": model,
                    "modality": "multimodal", "input_data": payload,
                    "downstream_task": "dataset_classification",
                    "num_classes": max(int(r["label"]) for r in records if r["label"] is not None) + 1,
                })
                logits = result["output"]
                labels = batch["labels"]
                correct += int((logits.argmax(dim=-1) == labels).sum().item())
                total += len(labels)
    if model == "MultimodalPerception":
        return {"split": split, "accuracy": correct / max(1, total), "evaluated_samples": total}
    left, right = torch.cat(all_left), torch.cat(all_right)
    similarities = left @ right.T
    n = similarities.size(0)
    positives = torch.tensor([[a == b for b in groups] for a in groups], dtype=torch.bool)
    metrics = {"split": split, "evaluated_samples": n}
    for k in (1, 5):
        top = similarities.topk(min(k, n), dim=1).indices
        metrics[f"text_to_other_recall_at_{k}"] = float(positives.gather(1, top).any(dim=1).float().mean())
        top = similarities.T.topk(min(k, n), dim=1).indices
        metrics[f"other_to_text_recall_at_{k}"] = float(positives.T.gather(1, top).any(dim=1).float().mean())
    return metrics


def _train(args: argparse.Namespace) -> None:
    records, summary, root = _validated_data(args)
    if args.model == "MultimodalPerception":
        labels = sorted({r.get("label") for r in records if r.get("label") is not None})
        if len(labels) < 2 or labels != list(range(len(labels))) or any(r.get("label") is None for r in records):
            raise ValueError("Supervised training requires complete contiguous integer labels starting at 0.")
    if not 0.0 <= args.joint_contrastive_weight <= 10.0 or not torch.isfinite(torch.tensor(args.joint_contrastive_weight)):
        raise ValueError("--joint-contrastive-weight must be finite in [0,10].")
    if args.model != "MultimodalPerception" and args.joint_contrastive_weight > 0:
        raise ValueError("Joint contrastive loss applies only to MultimodalPerception training.")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    agent = _agent(args.device)
    if args.model == "MultimodalPerception":
        agent._get_task_head("dataset_classification", agent.embed_dim,
                             num_classes=len(labels))
    start = 0
    folder = Path(args.output).resolve()
    if args.resume:
        metadata = load_into_agent(agent, folder, expected_model=args.model,
                                   expected_dataset=summary["manifest_sha256"], restore_optimizer=True)
        previous_weight = metadata.get("training_config", {}).get("joint_contrastive_weight", 0.0)
        if float(previous_weight) != args.joint_contrastive_weight or metadata["model_name"] != args.model:
            raise ValueError("Resume model identity and joint objective weight must match the checkpoint.")
        start = int(metadata["epoch"])
    for epoch in range(start, args.epochs):
        agent.train()
        losses = []
        for batch in _loader(records, root, "train", tuple(args.pair), args.batch_size):
            if len(batch["group_ids"]) < 2 and args.model == "MultimodalContrastiveEncoder":
                continue  # InfoNCE requires at least two examples.
            if args.model == "MultimodalContrastiveEncoder":
                # Canonical model view shares exactly these parameters with the trainer.
                step = agent.trainer.contrastive_step(
                    *[v for m in args.pair for v in (m, batch["payload"][m])],
                    symmetric=True, group_ids=batch["group_ids"],
                )
            else:
                representations = agent._model_views[args.model].encode(batch["payload"])
                head, _ = agent._get_existing_task_head("dataset_classification",
                                                          agent.embed_dim, len(labels))
                if args.joint_contrastive_weight > 0:
                    step = agent.trainer.joint_step(
                        head, representations, batch["labels"].to(agent.device),
                        torch.nn.functional.cross_entropy,
                        args.pair[0], args.pair[1],
                        contrastive_weight=args.joint_contrastive_weight,
                        group_ids=batch["group_ids"],
                    )
                else:
                    fused = agent.fusion(representations)
                    step = agent.trainer.supervised_step(
                        head, fused, batch["labels"].to(agent.device),
                        torch.nn.functional.cross_entropy,
                    )
            losses.append(step.loss)
        if not losses:
            raise ValueError("No usable training batches were available.")
        trained = (
            MODELS if args.model == "MultimodalPerception"
            and args.joint_contrastive_weight > 0 else (args.model,)
        )
        agent._trained_model_artifacts = {
            name: {"pair": list(args.pair), "validated_real_world_performance": False,
                   "training_stage": "in_process"} for name in trained
        }
        metrics = _evaluate(agent, records, root, tuple(args.pair), args.model, "validation", args.batch_size)
        metrics["mean_train_loss"] = sum(losses) / len(losses)
        if args.model == "MultimodalPerception" and args.joint_contrastive_weight > 0:
            alignment_metrics = _evaluate(
                agent, records, root, tuple(args.pair),
                "MultimodalContrastiveEncoder", "validation", args.batch_size
            )
            metrics.update({
                "contrastive_" + k: v for k, v in alignment_metrics.items()
                if k not in {"split", "evaluated_samples"}
            })
        _save(agent, folder, model=args.model, epoch=epoch + 1, summary=summary,
              pair=tuple(args.pair), metrics=metrics,
              training_config={"seed": args.seed, "batch_size": args.batch_size,
                               "device": args.device, "planned_epochs": args.epochs,
                               "joint_contrastive_weight": args.joint_contrastive_weight,
                               "torch_version": str(torch.__version__)})
        print(json.dumps({"epoch": epoch + 1, **metrics}, sort_keys=True))


def _evaluate_saved(args: argparse.Namespace) -> None:
    records, summary, root = _validated_data(args)
    agent = _agent(args.device)
    info = load_into_agent(agent, Path(args.checkpoint), expected_model=args.model,
                           expected_dataset=summary["manifest_sha256"])
    if tuple(info["pair"]) != tuple(args.pair):
        raise ValueError("Evaluation pair does not match the checkpoint.")
    metrics = _evaluate(agent, records, root, tuple(args.pair),
                        args.model, args.split, args.batch_size)
    report = {"model_name": args.model, "checkpoint_sha256": info["model_sha256"],
              "dataset_manifest_sha256": summary["manifest_sha256"], **metrics}
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _infer(args: argparse.Namespace) -> None:
    agent = _agent(args.device)
    info = load_into_agent(agent, Path(args.checkpoint), expected_model=args.model)

    def payload_for(modality: str):
        if modality == "text":
            if not args.text:
                raise ValueError("--text is required for text input.")
            return args.text
        if not args.path:
            raise ValueError("--path is required for image/audio input.")
        base = MultiModalDataset(mode="inference")
        return (base._process_vision(Path(args.path)) if modality == "vision"
                else base._process_audio(Path(args.path)))

    if args.modality == "multimodal":
        pair = info["pair"]
        if len(pair) != 2:
            raise ValueError("Saved artifact does not specify a supported modality pair.")
        data = {modality: payload_for(modality) for modality in pair}
        request = {"task_type": "inference", "model_name": args.model,
                   "modality": "multimodal", "input_data": data}
    else:
        request = {"task_type": "inference", "model_name": args.model,
                   "modality": args.modality, "input_data": payload_for(args.modality)}
    result = agent.perform_task(request)
    output = result["output"]
    if isinstance(output, dict):
        shape = {k: list(v.shape) for k, v in output.items()}
    else:
        shape = list(output.shape)
    print(json.dumps({"model_name": args.model, "output_shape": shape,
                      "checkpoint_sha256": info["model_sha256"]}, indent=2))



# Links copied from the official VGG page, which hosts archive downloads on
# thor.robots.ox.ac.uk (not under the main website's /data/ directory).
OXFORD_PETS_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/"


def _download(url: str, destination: Path, limit_bytes: int) -> dict[str, Any]:
    """Explicit bounded HTTPS download; resume only when Range is honored."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= limit_bytes:
        raise ValueError("Existing partial download exceeds maximum archive size.")
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with urlopen(Request(url, headers=headers), timeout=60) as response:
        if not response.geturl().startswith("https://"):
            raise ValueError("Refusing non-HTTPS dataset redirect.")
        resumed = offset > 0 and response.status == 206
        if resumed and not str(response.headers.get("Content-Range", "")).startswith(f"bytes {offset}-"):
            raise ValueError("Incorrect byte offset in Range response.")
        total = offset if resumed else 0
        declared = response.headers.get("Content-Length")
        if declared and int(declared) + total > limit_bytes:
            raise ValueError("Upstream archive exceeds configured maximum size.")
        with partial.open("ab" if resumed else "wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit_bytes:
                    raise ValueError("Archive exceeded download limit.")
                output.write(chunk)
    partial.replace(destination)
    return {"filename": destination.name, "bytes": destination.stat().st_size,
            "sha256": _sha256(destination), "source_url": url,
            "upstream_checksum_verified": False}


def _safe_extract_tar(archive: Path, destination: Path, *, maximum_bytes: int) -> None:
    """Extract bounded regular files only; reject links and path traversal."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cumulative = 0
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        for item in members:
            if item.isdir():
                continue
            if not item.isfile() or item.size < 0 or item.size > 32 * 1024 * 1024:
                raise ValueError("Unsafe archive member type or excessive member size.")
            cumulative += item.size
            if cumulative > maximum_bytes:
                raise ValueError("Decompressed archive exceeded extraction limit.")
            relative = Path(item.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Archive contains path traversal.")
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive escapes extraction directory.")
            if target.exists():
                raise ValueError("Archive would overwrite an existing file.")
        for item in members:
            if item.isdir():
                continue
            target = (destination / item.name).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            source_file = source.extractfile(item)
            if source_file is None:
                raise ValueError("Unable to extract archive member.")
            with source_file, target.open("xb") as output:
                shutil.copyfileobj(source_file, output, length=1024 * 1024)


def _acquire_oxford_pets(args: argparse.Namespace) -> None:
    """User-triggered official download; no network access on import/training."""
    if not args.confirm_rights:
        raise ValueError("Review license and photograph rights, then pass --confirm-rights.")
    root = Path(args.dataset_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive_info = []
    for name, limit in (("images.tar.gz", 1_200_000_000),
                        ("annotations.tar.gz", 120_000_000)):
        archive = root / name
        # An existing final archive is not implicitly trusted.
        if archive.is_file():
            archive_info.append({"filename": name, "sha256": _sha256(archive),
                                 "bytes": archive.stat().st_size,
                                 "source_url": OXFORD_PETS_URL + name,
                                 "upstream_checksum_verified": False})
        else:
            archive_info.append(_download(OXFORD_PETS_URL + name, archive, limit))
        if args.extract:
            _safe_extract_tar(archive, root, maximum_bytes=1_600_000_000)
    (root / "acquisition.json").write_text(
        json.dumps({"dataset": "Oxford-IIIT Pet", "official_page":
                    "https://www.robots.ox.ac.uk/~vgg/data/pets/",
                    "license_page_claim": "CC BY-SA 4.0",
                    "photograph_rights": "Remain with original owners",
                    "operator_confirmed_rights": True, "archives": archive_info},
                   indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"archives": archive_info, "extracted": bool(args.extract)}, indent=2))


def _oxford_manifest(args: argparse.Namespace) -> None:
    """Produce reproducible *weakly labelled* image/text pairs from official labels."""
    root = Path(args.dataset_root).resolve()
    trainval = root / "annotations" / "trainval.txt"
    test = root / "annotations" / "test.txt"
    if not trainval.is_file() or not test.is_file():
        raise ValueError("Extract official Oxford annotations before preparing the manifest.")

    def records(path: Path) -> list[tuple[str, int]]:
        rows = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.split()
            if len(fields) != 4 or not fields[1].isdecimal():
                raise ValueError("Invalid Oxford annotations structure.")
            name, breed = fields[0], int(fields[1])
            if not name.replace("_", "").isalnum() or not (1 <= breed <= 37):
                raise ValueError("Invalid breed label or image identifier.")
            rows.append((name, breed - 1))
        return rows

    trainval_rows, test_rows = records(trainval), records(test)
    random_generator = random.Random(args.seed)
    per_breed: dict[int, list[str]] = {}
    for name, breed in trainval_rows:
        per_breed.setdefault(breed, []).append(name)
    split_lookup: dict[str, str] = {}
    for names in per_breed.values():
        random_generator.shuffle(names)
        split_at = max(1, round(len(names) * args.validation_fraction))
        for name in names[:split_at]:
            split_lookup[name] = "validation"
        for name in names[split_at:]:
            split_lookup[name] = "train"
    for name, _ in test_rows:
        if name in split_lookup:
            raise ValueError("Same image appears in official trainval and test.")
        split_lookup[name] = "test"

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with out.open("w", encoding="utf-8") as stream:
        for name, breed in trainval_rows + test_rows:
            if name in seen:
                raise ValueError("Duplicate image in Oxford annotations.")
            seen.add(name)
            candidates = [
                root / "images" / (name + extension)
                for extension in (".jpg", ".png")
            ]
            found = [file for file in candidates if file.is_file()]
            if len(found) != 1:
                raise ValueError(f"Expected exactly one JPG/PNG for Oxford image {name!r}.")
            # Breed label text is weak supervision, NOT a human-authored caption.
            breed_text = name.rsplit("_", 1)[0].replace("_", " ").lower()
            row = {
                "id": name, "group_id": name, "positive_id": f"breed:{breed}",
                "split": split_lookup[name], "vision": f"images/{found[0].name}",
                "text": breed_text, "label": breed,
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(out), "samples": len(seen),
                      "caption_annotations": False,
                      "alignment": "weak_breed_label_pairs"}, indent=2))



def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    acquire = subs.add_parser("acquire-oxford-pets")
    acquire.add_argument("--dataset-root", required=True)
    acquire.add_argument("--confirm-rights", action="store_true")
    acquire.add_argument("--extract", action="store_true")
    acquire.set_defaults(fn=_acquire_oxford_pets)

    oxford = subs.add_parser("oxford-pets-manifest")
    oxford.add_argument("--dataset-root", required=True)
    oxford.add_argument("--output", required=True)
    oxford.add_argument("--seed", type=int, default=42)
    oxford.add_argument("--validation-fraction", type=float, default=0.2)
    oxford.set_defaults(fn=_oxford_manifest)

    prepare = subs.add_parser("prepare")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--dataset-root", required=True)
    prepare.add_argument("--pair", nargs=2, choices=MODALITIES, required=True)
    prepare.add_argument("--source-url", required=True)
    prepare.add_argument("--license", required=True)
    prepare.add_argument("--confirm-rights", action="store_true")
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(fn=_prepare)
    for command in ("train", "evaluate"):
        sub = subs.add_parser(command)
        sub.add_argument("--manifest", required=True)
        sub.add_argument("--dataset-root", required=True)
        sub.add_argument("--dataset-declaration", required=True)
        sub.add_argument("--pair", nargs=2, choices=MODALITIES, required=True)
        sub.add_argument("--model", choices=MODELS, required=True)
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--batch-size", type=int, default=4)
        if command == "train":
            sub.add_argument("--epochs", type=int, default=1)
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--joint-contrastive-weight", type=float, default=0.0)
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--output", required=True)
            sub.set_defaults(fn=_train)
        else:
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--split", choices=("validation", "test"), default="test")
            sub.add_argument("--output-json")
            sub.set_defaults(fn=_evaluate_saved)
    infer = subs.add_parser("infer")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--model", choices=MODELS, required=True)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--modality", choices=(*MODALITIES, "multimodal"), required=True)
    infer.add_argument("--text")
    infer.add_argument("--path")
    infer.set_defaults(fn=_infer)
    args = parser.parse_args(argv)
    if hasattr(args, "validation_fraction") and not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be in (0, 0.5).")
    if hasattr(args, "batch_size") and args.batch_size < 2:
        parser.error("batch-size must be at least 2 for contrastive learning.")
    if hasattr(args, "epochs") and args.epochs <= 0:
        parser.error("epochs must be positive.")
    args.fn(args)


if __name__ == "__main__":
    main()
