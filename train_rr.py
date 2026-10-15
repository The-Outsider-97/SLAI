"""SLAI v2.3 RelevanceRanker — real BEIR acquisition, pairwise training and IR evaluation.

Usage: py -m train_rr --help
No dataset or model is bootstrapped synthetically. No provider API is called.
Runs are research artifacts until evaluation and agent integration gates pass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
import numpy as np # type: ignore
import torch # type: ignore
import torch.nn.functional as F # type: ignore

from collections import defaultdict
from pathlib import Path

from src.agents.knowledge.relevance_data import (
    atomic_json, collect_scifact, prepare_scifact, read_jsonl,
    retrieval_metrics, validate_prepared,
)
from src.agents.knowledge.relevance_ranker import (
    FEATURE_NAMES, FEATURE_VERSION, FeatureExtractor, RankerNetwork,
    RelevanceRanker, checkpoint_sha256,
)

DEFAULT_DATA = Path("data/training/knowledge/relevance_ranker/prepared")
DEFAULT_RUN = Path("data/training/knowledge/relevance_ranker/run_v1")
DEFAULT_MODEL = Path("src/agents/knowledge/models/relevance_ranker")


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA selected but unavailable")
    return torch.device(name)


def save_torch(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_torch(path: Path) -> dict:
    # weights_only=True forbids loading arbitrary pickle objects from a checkpoint.
    return torch.load(path, map_location="cpu", weights_only=True)


def load_rows(data: Path, split: str) -> list[dict]:
    rows = list(read_jsonl(data / f"{split}.jsonl"))
    if not rows:
        raise ValueError(f"No rows for {split}")
    return rows


def prepared_signature(data: Path) -> str:
    manifest = json.loads((data / "dataset_manifest.json").read_text(encoding="utf-8"))
    return json.dumps({k: manifest["splits"][k]["sha256"] for k in ("train", "validation", "test")}, sort_keys=True)


def features(rows: list[dict], extractor: FeatureExtractor, batch_size: int = 128) -> torch.Tensor:
    arrays = []
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset:offset + batch_size]
        arrays.append(extractor.transform([r["query"] for r in chunk], [r["text"] for r in chunk]))
    matrix = np.concatenate(arrays, axis=0)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite features")
    return torch.from_numpy(matrix)


def pair_indices(rows: list[dict]) -> list[tuple[int, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["query_id"]].append(i)
    result = []
    for indices in groups.values():
        positive = [i for i in indices if rows[i]["label"] > 0]
        negative = [i for i in indices if rows[i]["label"] == 0]
        for p in positive:
            for n in negative:
                result.append((p, n))
    if not result:
        raise ValueError("No usable positive/negative pairs; check dataset")
    return result


def evaluate_scores(rows: list[dict], scores: list[float], qrels_path: Path) -> dict:
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    ranked: dict[str, list[tuple[str, float]]] = defaultdict(list)
    baseline: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        qid, did = row["query_id"], row["document_id"]
        ranked[qid].append((did, float(score)))
        baseline[qid].append((did, float(row["bm25"])))
    if set(ranked) != set(qrels):
        raise ValueError("Missing query candidate pools")
    return {"reranker": retrieval_metrics(ranked, qrels),
            "bm25": retrieval_metrics(baseline, qrels),
            "queries": len(ranked), "candidate_pairs": len(rows),
            "judgment_policy": "Unjudged pool candidates are treated as nonrelevant for IR metrics"}


def score_model(net, matrix: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                device: torch.device, batch_size: int) -> list[float]:
    net.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(matrix), batch_size):
            x = ((matrix[start:start + batch_size] - mean) / std).clamp(-8, 8).to(device)
            values.extend(net(x).cpu().tolist())
    return [float(v) for v in values]


def _optimizer(name: str, params, lr: float, wd: float):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError("Unsupported optimizer")


def _scheduler(name: str, optimizer, epochs: int):
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)
    raise ValueError("Unsupported scheduler")


def train(args, resume: bool = False) -> None:
    verified = validate_prepared(args.data_dir)
    signature = prepared_signature(args.data_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    last_file = args.run_dir / "last.pt"
    if resume and not last_file.is_file():
        raise FileNotFoundError(f"No resumable checkpoint: {last_file}")
    if not resume and last_file.exists():
        raise FileExistsError("A training run already exists. Use resume or a new --run-dir")
    config_path = args.run_dir / "training_config.json"
    if resume:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        # Restore all effective hyperparameters rather than silently accepting
        # a new, incompatible optimizer/architecture/data configuration.
        for key in ("seed", "width", "lr", "weight_decay", "batch_size", "patience",
                    "optimizer", "scheduler", "device", "encoder_path", "epochs", "gradient_clip",
                    "gradient_accumulation", "amp"):
            setattr(args, key, cfg[key])
        if cfg["signature"] != signature:
            raise ValueError("Cannot resume: source dataset changed")
        print("Resuming with saved hyperparameters", flush=True)
    else:
        cfg = {key: getattr(args, key) for key in (
            "seed", "width", "lr", "weight_decay", "batch_size", "patience", "optimizer",
            "scheduler", "device", "encoder_path", "epochs", "gradient_clip",
            "gradient_accumulation", "amp")}
        cfg["signature"] = signature
        cfg["torch_version"] = torch.__version__
        cfg["numpy_version"] = np.__version__
        atomic_json(config_path, cfg)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = device_from_name(args.device)
    t0 = time.perf_counter()
    extractor = FeatureExtractor(args.encoder_path)
    train_rows = load_rows(args.data_dir, "train")
    valid_rows = load_rows(args.data_dir, "validation")
    x_train = features(train_rows, extractor)
    x_valid = features(valid_rows, extractor)
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False).clamp(min=0.02)
    train_x = ((x_train - mean) / std).clamp(-8, 8)
    pairs = pair_indices(train_rows)
    net = RankerNetwork(width=args.width).to(device)
    optim = _optimizer(args.optimizer, net.parameters(), args.lr, args.weight_decay)
    amp_enabled = bool(args.amp and device.type == "cuda")
    if args.amp and not amp_enabled:
        raise ValueError("Mixed precision requires CUDA")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scheduler = _scheduler(args.scheduler, optim, args.epochs)
    start_epoch, best_ndcg, wait = 0, -1.0, 0
    if resume:
        ckpt = load_torch(last_file)
        if ckpt.get("signature") != signature or ckpt.get("width") != args.width:
            raise ValueError("Checkpoint does not match dataset/model configuration")
        net.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if scheduler is not None and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"])
        best_ndcg, wait = float(ckpt["best_ndcg"]), int(ckpt["wait"])
        mean, std = ckpt["mean"], ckpt["std"]
        train_x = ((x_train - mean) / std).clamp(-8, 8)
        print(f"Restored epoch-boundary checkpoint: completed epochs={start_epoch}")
    history = []
    history_file = args.run_dir / "history.json"
    if history_file.is_file():
        history = json.loads(history_file.read_text(encoding="utf-8"))
    for epoch in range(start_epoch, args.epochs):
        net.train()
        rng = random.Random(args.seed + epoch)
        order = list(range(len(pairs)))
        rng.shuffle(order)
        losses = []
        optim.zero_grad(set_to_none=True)
        total_batches = math.ceil(len(order) / args.batch_size)
        for batch_index, begin in enumerate(range(0, len(order), args.batch_size)):
            group = [pairs[i] for i in order[begin:begin + args.batch_size]]
            left = train_x[[p for p, _ in group]].to(device)
            right = train_x[[n for _, n in group]].to(device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                pos, neg = net(left), net(right)
                loss = F.softplus(neg - pos).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch+1}")
            scaler.scale(loss / args.gradient_accumulation).backward()
            if ((batch_index + 1) % args.gradient_accumulation == 0
                    or batch_index + 1 == total_batches):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.gradient_clip,
                                                error_if_nonfinite=True)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        scores = score_model(net, x_valid, mean, std, device, args.batch_size)
        result = evaluate_scores(valid_rows, scores, args.data_dir / "qrels_validation.json")
        metric = result["reranker"]["nDCG@10"]
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(metric)
            else:
                scheduler.step()
        improved = metric > best_ndcg + 1e-7
        if improved:
            best_ndcg, wait = metric, 0
        else:
            wait += 1
        checkpoint = {
            "model": net.state_dict(), "optimizer": optim.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if amp_enabled else None,
            "epoch": epoch + 1, "best_ndcg": best_ndcg, "wait": wait,
            "mean": mean.detach().cpu(), "std": std.detach().cpu(),
            "signature": signature, "width": args.width,
        }
        if improved:
            save_torch(args.run_dir / "best.pt", checkpoint)
        save_torch(last_file, checkpoint)
        entry = {"epoch": epoch + 1, "pairwise_loss": round(statistics.mean(losses), 7),
                 "validation": result, "best_ndcg": best_ndcg, "improved": improved}
        history.append(entry)
        atomic_json(history_file, history)
        print(json.dumps({"epoch": entry["epoch"], "loss": entry["pairwise_loss"],
                          "nDCG@10": metric, "bm25_nDCG@10": result["bm25"]["nDCG@10"]}), flush=True)
        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    info = {"device": str(device), "model_parameters": sum(p.numel() for p in net.parameters()),
            "completed_epochs": len(history), "best_nDCG@10_validation": best_ndcg,
            "feature_count": len(FEATURE_NAMES), "data_counts": verified["counts"],
            "duration_seconds_this_invocation": round(time.perf_counter() - t0, 3),
            "checkpoint": str((args.run_dir / "best.pt").resolve()),
            "status": "research_checkpoint_not_production_promoted"}
    atomic_json(args.run_dir / "training_report.json", info)
    print(json.dumps(info, indent=2))


def run_evaluation(args) -> dict:
    validate_prepared(args.data_dir)
    config = json.loads((args.run_dir / "training_config.json").read_text(encoding="utf-8"))
    if prepared_signature(args.data_dir) != config["signature"]:
        raise ValueError("Dataset differs from training dataset")
    rows = load_rows(args.data_dir, args.split)
    checkpoint = load_torch(args.run_dir / "best.pt")
    net = RankerNetwork(width=checkpoint["width"]).to(device_from_name(args.device))
    net.load_state_dict(checkpoint["model"])
    extractor = FeatureExtractor(config["encoder_path"])
    x = features(rows, extractor, args.batch_size)
    t0 = time.perf_counter()
    scores = score_model(net, x, checkpoint["mean"], checkpoint["std"],
                         device_from_name(args.device), args.batch_size)
    elapsed = time.perf_counter() - t0
    result = evaluate_scores(rows, scores, args.data_dir / f"qrels_{args.split}.json")
    result["split"] = args.split
    result["model_checkpoint_sha256"] = checkpoint_sha256(args.run_dir / "best.pt")
    result["inference_seconds_excluding_feature_extraction"] = round(elapsed, 4)
    result["candidate_throughput_per_second"] = round(len(rows) / max(elapsed, 1e-6), 3)
    result["candidate_recall_ceiling"] = result["bm25"]["Recall@100"]
    if args.split == "test":
        atomic_json(args.run_dir / "heldout_test_report.json", result)
    else:
        atomic_json(args.run_dir / "validation_benchmark.json", result)
    print(json.dumps(result, indent=2))
    return result


def export_model(args) -> None:
    checkpoint = load_torch(args.run_dir / "best.pt")
    config = json.loads((args.run_dir / "training_config.json").read_text(encoding="utf-8"))
    report_path = args.run_dir / "heldout_test_report.json"
    if not report_path.exists():
        raise ValueError("Run independent test evaluation before export")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["model_checkpoint_sha256"] != checkpoint_sha256(args.run_dir / "best.pt"):
        raise ValueError("Test evaluation checkpoint differs from export checkpoint")
    eligible = (report["reranker"]["nDCG@10"] >= report["bm25"]["nDCG@10"]
                and report["reranker"]["MRR@10"] >= report["bm25"]["MRR@10"] - 0.01)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Output is not empty: refuse to overwrite model artifacts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final = {"model": checkpoint["model"], "mean": checkpoint["mean"], "std": checkpoint["std"]}
    save_torch(args.output_dir / "model.pt", final)
    manifest = {
        "model_id": "knowledge_relevance_ranker_v1", "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES), "width": checkpoint["width"],
        "encoder_path": config["encoder_path"], "sha256": checkpoint_sha256(args.output_dir / "model.pt"),
        "architecture": "pairwise-trained 2-layer neural feature reranker",
        "evaluation_passed_baseline_metrics": eligible,
        "status": "research_candidate_integration_gate_pending",
        "note": "Test result is BM25 candidate-pool reranking, not end-to-end KnowledgeAgent certification",
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    atomic_json(args.output_dir / "metrics.json", report)
    atomic_json(args.output_dir / "training_config.json", config)
    print(json.dumps({"output": str(args.output_dir), "baseline_gate": eligible,
                      "production_ready": False}, indent=2))


def benchmark_inference(args) -> None:
    model = RelevanceRanker(args.model_dir, device=args.device, batch_size=args.batch_size)
    passages = ["Mars has a thin carbon dioxide atmosphere.",
                "Jupiter has large gaseous storms.",
                "Mars atmospheric pressure averages below one percent of Earth's surface pressure."]
    for _ in range(2):
        model.score("What is the atmospheric pressure on Mars?", passages)
    trials = []
    for _ in range(args.repetitions):
        t0 = time.perf_counter()
        model.score("What is the atmospheric pressure on Mars?", passages)
        trials.append((time.perf_counter() - t0) * 1000)
    trials.sort()
    print(json.dumps({"latency_ms_p50": round(statistics.median(trials), 4),
                      "latency_ms_p95": round(trials[min(len(trials)-1, math.ceil(0.95 * len(trials)) - 1)], 4),
                      "queries": args.repetitions, "candidate_count": len(passages),
                      "device": args.device, "model": model.version}, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="Show accessible dataset and run artifacts")
    status.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    status.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    coll = sub.add_parser("collect", help="Download real BEIR SciFact archive (opt-in network)")
    coll.add_argument("--destination", type=Path, default=DEFAULT_DATA.parent)
    coll.add_argument("--allow-download", action="store_true", help="Explicit permission to use public dataset network")
    coll.add_argument("--timeout", type=int, default=60)
    prep = sub.add_parser("prepare", help="Create grouped train/validation/test IR candidate pools")
    prep.add_argument("--raw-dir", type=Path, default=DEFAULT_DATA.parent / "raw")
    prep.add_argument("--output-dir", type=Path, default=DEFAULT_DATA)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--negatives", type=int, default=12)
    prep.add_argument("--eval-pool", type=int, default=100)
    prep.add_argument("--validation-fraction", type=float, default=0.15)
    validate = sub.add_parser("validate-data", help="Check dataset hashes/schema/group leakage")
    validate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    for name in ("train", "resume"):
        p = sub.add_parser(name, help="Train or resume from a complete epoch checkpoint")
        p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
        p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
        p.add_argument("--epochs", type=int, default=25)
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--lr", type=float, default=0.001)
        p.add_argument("--weight-decay", type=float, default=0.0001)
        p.add_argument("--width", type=int, default=64)
        p.add_argument("--patience", type=int, default=5)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        p.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
        p.add_argument("--scheduler", choices=("none", "cosine", "plateau"), default="plateau")
        p.add_argument("--gradient-clip", type=float, default=1.0)
        p.add_argument("--gradient-accumulation", type=int, default=1, help="Optimizer step every N training batches")
        p.add_argument("--amp", action="store_true", help="CUDA mixed-precision training")
        p.add_argument("--encoder-path", default=None, help="EXISTING LOCAL sentence-transformers model directory")
    for name in ("evaluate", "benchmark"):
        p = sub.add_parser(name, help="Evaluate held-out ranking against BM25 on identical pools")
        p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
        p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
        p.add_argument("--split", choices=("validation", "test"), default="test")
        p.add_argument("--batch-size", type=int, default=128)
        p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    exp = sub.add_parser("export", help="Export immutable inference artifact after held-out evaluation")
    exp.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    exp.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL)
    inf = sub.add_parser("infer", help="Score and rank user-supplied passages locally")
    inf.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    inf.add_argument("--query", required=True)
    inf.add_argument("--passage", action="append", required=True, help="Repeat for each candidate passage")
    inf.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    smoke = sub.add_parser("smoke", help="Clean-process model load and repeated deterministic local inference")
    smoke.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    perf = sub.add_parser("inference-benchmark", help="Measure local p50/p95 scorer inference latency")
    perf.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    perf.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    perf.add_argument("--batch-size", type=int, default=64)
    perf.add_argument("--repetitions", type=int, default=100)
    args = ap.parse_args(argv)
    try:
        if args.command == "status":
            print(json.dumps({"dataset": str(args.data_dir), "dataset_exists": (args.data_dir / "dataset_manifest.json").exists(),
                              "best_checkpoint": str(args.run_dir / "best.pt"), "checkpoint_exists": (args.run_dir / "best.pt").exists(),
                              "test_report_exists": (args.run_dir / "heldout_test_report.json").exists()}, indent=2))
        elif args.command == "collect":
            print(json.dumps(collect_scifact(args.destination, timeout=args.timeout,
                                            allow_download=args.allow_download), indent=2))
        elif args.command == "prepare":
            print(json.dumps(prepare_scifact(args.raw_dir, args.output_dir, seed=args.seed,
                                             negatives=args.negatives, eval_pool=args.eval_pool,
                                             validation_fraction=args.validation_fraction), indent=2))
        elif args.command == "validate-data":
            print(json.dumps(validate_prepared(args.data_dir), indent=2))
        elif args.command in ("train", "resume"):
            if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.width < 8 \
               or args.lr <= 0 or args.weight_decay < 0 or args.gradient_clip <= 0 \
               or args.gradient_accumulation < 1:
                raise ValueError("Invalid training hyperparameter")
            train(args, resume=args.command == "resume")
        elif args.command in ("evaluate", "benchmark"):
            run_evaluation(args)
        elif args.command == "export":
            export_model(args)
        elif args.command == "infer":
            model = RelevanceRanker(args.model_dir, device=args.device)
            scores = model.score(args.query, args.passage)
            ranked = sorted(zip(scores, args.passage), key=lambda x: -x[0])
            print(json.dumps([{"score": score, "text": text} for score, text in ranked], indent=2, ensure_ascii=False))
        elif args.command == "smoke":
            model = RelevanceRanker(args.model_dir, device=args.device)
            candidates = [(0.9, {"doc_id": "a", "text": "Titan's atmosphere contains nitrogen", "metadata": {"source": "a"}}),
                          (0.4, {"doc_id": "b", "text": "Titan surface ice", "metadata": {"source": "b"}})]
            a = model.rank("What gas dominates Titan's atmosphere?", candidates)
            b = model.rank("What gas dominates Titan's atmosphere?", candidates)
            assert a == b and {v[1]["doc_id"] for v in a} == {"a", "b"}
            assert all(v[1]["metadata"] for v in a)
            print(json.dumps({"status": "PASS", "order": [v[1]["doc_id"] for v in a],
                              "model": model.version}))
        elif args.command == "inference-benchmark":
            if args.repetitions < 2:
                raise ValueError("repetitions must be >=2")
            benchmark_inference(args)
        return 0
    except (OSError, ValueError, RuntimeError, FloatingPointError) as exc:
        print(f"TRAIN_RR ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
