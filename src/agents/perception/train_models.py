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


def _save(agent: PerceptionAgent, folder: Path, *, model: str, epoch: int,
          summary: dict, pair: tuple[str, str], metrics: dict,
          training_config: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    state = {"schema": SCHEMA, "model_name": model, "epoch": epoch, "pair": list(pair),
             "manifest_sha256": summary["manifest_sha256"],
             "training_config": training_config,
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
    if expected_model and metadata["model_name"] != expected_model:
        raise ValueError("Checkpoint model identity mismatch.")
    if expected_dataset and metadata.get("manifest_sha256") != expected_dataset:
        raise ValueError("Checkpoint training data fingerprint mismatch.")
    if _sha256(folder / "model.pt") != metadata.get("model_sha256"):
        raise ValueError("Training artifact SHA-256 integrity check failed.")
    # weights_only refuses arbitrary pickle object graphs from untrusted data.
    state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    if (state.get("schema") != SCHEMA or state.get("model_name") != metadata["model_name"]
            or state.get("pair") != metadata["pair"] or state.get("epoch") != metadata["epoch"]):
        raise ValueError("Training artifact metadata disagrees with the checkpoint.")
    # Preflight the complete parameter schema before mutation.
    agent._restore_task_head_structure(state.get("task_head_specs", []))
    target = agent.state_dict()
    candidate = state.get("model_state_dict", {})
    if set(target) != set(candidate) or any(target[k].shape != candidate[k].shape for k in target):
        raise ValueError("Checkpoint is incompatible with the PerceptionAgent architecture.")
    agent.load_state_dict(candidate, strict=True)
    agent.trainer.load_state_dict(state.get("trainer_state_dict", {}), load_optimizer=restore_optimizer)
    agent._trained_model_artifacts[metadata["model_name"]] = {
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
                fused = agent.fusion(representations)
                head, _ = agent._get_existing_task_head("dataset_classification",
                                                          agent.embed_dim, len(labels))
                step = agent.trainer.supervised_step(
                    head, fused, batch["labels"].to(agent.device),
                    torch.nn.functional.cross_entropy,
                )
            losses.append(step.loss)
        if not losses:
            raise ValueError("No usable training batches were available.")
        agent._trained_model_artifacts[args.model] = {
            "pair": list(args.pair), "validated_real_world_performance": False,
            "training_stage": "in_process",
        }
        metrics = _evaluate(agent, records, root, tuple(args.pair), args.model, "validation", args.batch_size)
        metrics["mean_train_loss"] = sum(losses) / len(losses)
        _save(agent, folder, model=args.model, epoch=epoch + 1, summary=summary,
              pair=tuple(args.pair), metrics=metrics,
              training_config={"seed": args.seed, "batch_size": args.batch_size,
                               "device": args.device, "planned_epochs": args.epochs,
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
    print(json.dumps({"model_name": info["model_name"], "output_shape": shape,
                      "checkpoint_sha256": info["model_sha256"]}, indent=2))


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
