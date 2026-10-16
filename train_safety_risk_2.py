"""SLAI v2.3 consolidated native cyber-risk trainer (experimental, HTTP v2).

REPLACES train_safety_risk_2.py. No new auxiliary Python modules are created.
Integrates existing SLAI Safety / NeuralNetwork, and optional QualityAgent /
ProvenanceAgent through their public APIs. Builds model from scratch, fits
normalization on TRAIN ONLY, compares native MLP variants using VAL ONLY,
loads a signed SLAI checkpoint, and reports TEST once after model selection.

Uses only Python stdlib + already installed SLAI imports. Does not promote to
production: current CyberSafetyModule uses eight, not 32, input features.
Example:
 py -m train_safety_risk_2 --data data/training/safety/cyber_risk_csic2010.jsonl \
     --epochs 200 --batch-size 32 --learning-rate 0.0005 --patience 20 \
     --seed 42 --quality-agent --provenance-agent
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re

from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from urllib.parse import urlsplit, unquote, parse_qsl
from unittest.mock import patch

from train_safety_risk import (
    TrainingCyberSafetyModule,
    OfflineMemory,
    parse_dataset,
    require_minimums,
    predictions,
    metrics,
    auc_roc,
    utc_now,
    atomic_json,
)
from src.agents.safety.modules.neural_network import NeuralNetwork

# HTTP v2 lexical feature extraction remains in THIS trainer (not a new module).
# All features are label-independent. This exact implementation must be
# migrated into the production runtime before any v2 deployment.
HEX = re.compile(r"%[0-9a-fA-F]{2}")
SQL = re.compile(r"\b(?:union\s+(?:all\s+)?select|select\s+.{0,50}\s+from|drop\s+table|or\s+['\d][^\r\n]{0,20}=|waitfor\s+delay|sleep\s*\()", re.I)
XSS = re.compile(r"<\s*(?:script|svg|iframe)\b|\bonerror\s*=|javascript\s*:", re.I)
TRAVERSAL = re.compile(r"(?:^|[/\\])\.\.(?:[/\\]|$)|(?:%2e){2}|%252e", re.I)
SHELL = re.compile(r"(?:;|\|\||&&)\s*(?:cat|sh|bash|curl|wget|powershell|cmd)(?:\s|$)|\$\(|`[^`]+`", re.I)
SENSITIVE = re.compile(r"(?:/etc/(?:passwd|shadow)|win\.ini|web\.config|\.env\b|/proc/self/|/\.git/)", re.I)
HTTP_LINE = re.compile(r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|TRACE|CONNECT)\s+(\S+)(?:\s+HTTP/\d(?:\.\d)?)?", re.I)


def clamp(x, cap=1.0):
    return max(0.0, min(float(x), cap))


def scale(x, divisor):
    return clamp(x / divisor)


def entropy(s):
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((v / n) * math.log2(v / n) for v in counts.values()) / 8.0


def feature_map(request):
    """Parse actual HTTP request text. Does not depend on dataset labels."""
    lines = request.splitlines()
    first = lines[0].strip() if lines else ""
    match = HTTP_LINE.match(first)
    if match:
        method, target = match.group(1).upper(), match.group(2)
    else:
        method, target = "OTHER", first.split(" ", 1)[-1][:65536]
    parts = urlsplit(target)
    path, query = parts.path or "", parts.query or ""
    body = request.split("\n\n", 1)[-1] if "\n\n" in request else ""
    # Use URL and body only; shared benchmark headers (e.g. User-Agent) dominate
    # the raw text and are not discriminating HTTP-request attack indicators.
    payload = path + ("?" + query if query else "") + ("\n" + body if body else "")
    decoded = unquote(payload, errors="replace")
    decoded_twice = unquote(decoded, errors="replace")
    pct = HEX.findall(payload)
    param_pairs = parse_qsl(query, keep_blank_values=True, max_num_fields=1024)
    all_symbols = sum(ch in "'\"<>;()[]{}\\|`$" for ch in decoded)
    digits = sum(ch.isdigit() for ch in decoded)
    nonsafe = sum(not (ch.isalnum() or ch in "-_.~/?:=&%+") for ch in payload)
    high = sum(ord(ch) > 127 for ch in decoded)
    segments = [s for s in path.split("/") if s]
    uncommon_ext = bool(re.search(r"\.(?:bak|old|inc|ini|log|sql|sh|env|config|zip|tar|php\d?|asp|aspx)(?:$|[?;/])", path, re.I))
    encoded_reserved = sum(int(int(x[1:], 16) in {0x2F, 0x5C, 0x3C, 0x3E, 0x22, 0x27, 0x00, 0x3B, 0x25}) for x in pct)
    return {
        "method_post": float(method == "POST"),
        "method_other": float(method not in {"GET", "POST"}),
        "request_length": scale(len(request), 2048),
        "target_length": scale(len(target), 256),
        "path_length": scale(len(path), 150),
        "query_length": scale(len(query), 256),
        "body_length": scale(len(body), 512),
        "param_count": scale(len(param_pairs), 24),
        "path_depth": scale(len(segments), 15),
        "longest_segment": scale(max(map(len, segments), default=0), 100),
        "percent_count": scale(len(pct), 32),
        "percent_density": scale(len(pct) / max(1, len(payload)), 0.25),
        "encoded_reserved_count": scale(encoded_reserved, 8),
        "double_encoding": float("%" in decoded and "%" in decoded_twice and decoded != decoded_twice),
        "target_entropy": entropy(target),
        "query_entropy": entropy(query),
        "body_entropy": entropy(body),
        "digit_fraction": scale(digits / max(1, len(decoded)), 0.6),
        "symbol_fraction": scale(all_symbols / max(1, len(decoded)), 0.4),
        "non_safe_fraction": scale(nonsafe / max(1, len(payload)), 0.25),
        "single_quote_count": scale(decoded.count("'"), 12),
        "double_quote_count": scale(decoded.count('"'), 12),
        "angle_bracket_count": scale(decoded.count("<") + decoded.count(">"), 8),
        "semicolon_count": scale(decoded.count(";"), 10),
        "backslash_count": scale(decoded.count("\\"), 10),
        "non_ascii_count": scale(high, 16),
        "path_traversal": float(bool(TRAVERSAL.search(decoded_twice))),
        "sql_pattern": float(bool(SQL.search(decoded_twice))),
        "xss_pattern": float(bool(XSS.search(decoded_twice))),
        "shell_pattern": float(bool(SHELL.search(decoded_twice))),
        "sensitive_path": float(bool(SENSITIVE.search(decoded_twice))),
        "uncommon_extension": float(uncommon_ext),
    }

FEATURE_NAMES = tuple(feature_map("GET / HTTP/1.1").keys())



class HTTPTrainingSafetyModule(TrainingCyberSafetyModule):
    """Cold-start SLAI Safety adapter; preserve original HTTP body separators.

    Do not call runtime CyberSafetyModule for 32-input inference; this is an
    offline, HTTP-specific experiment. Rule signals are not silently relabeled.
    """
    def _stringify_input(self, input_data):
        if not isinstance(input_data, str):
            raise TypeError("Expected HTTP request text")
        limit = int(self._cfg("max_input_text_length", 65536))
        if len(input_data) > limit:
            raise ValueError("HTTP request exceeds configured input length")
        return input_data.replace("\r\n", "\n").replace("\r", "\n")

    def _build_model_features(self, text, findings, context):
        mapping = feature_map(text)
        return [float(mapping[name]) for name in FEATURE_NAMES]


class InProcessCoordination:
    """Lightweight coordination surface for explicitly opted-in agent audits.

    This is not an implementation of QualityMemory/ProvenanceStore. Their own
    configured persistence remains authoritative, if they are enabled.
    """
    def __init__(self):
        self.values = {}

    def set(self, key, value, **kwargs):
        self.values[str(key)] = value
        return True

    def get(self, key, default=None, **kwargs):
        return self.values.get(str(key), default)

    def publish(self, channel, payload, **kwargs):
        return True

    def subscribe(self, *args, **kwargs):
        return None


def mean_std(train):
    if not train:
        raise ValueError("Empty training data")
    dim = len(FEATURE_NAMES)
    n = len(train)
    means = [sum(float(x[j]) for x, _ in train) / n for j in range(dim)]
    std = [(sum((float(x[j]) - means[j]) ** 2 for x, _ in train) / n) ** 0.5
           for j in range(dim)]
    # Matches earlier standard-library logistic diagnostic contract.
    return means, [max(0.01, item) for item in std]


def standardize(partition, mean, std):
    return [([max(-8.0, min(8.0, (float(x[j]) - mean[j]) / std[j]))
              for j in range(len(mean))], list(y)) for x, y in partition]


def split_diagnostics(data):
    d = len(FEATURE_NAMES)
    constant = []
    for j in range(d):
        values = [x[j] for x, _ in data]
        if max(values) - min(values) < 1e-12:
            constant.append(FEATURE_NAMES[j])
    return {"constant_train_features": constant,
            "train_feature_count": d, "train_examples": len(data)}


def quality_audit(subsets, dataset_digest, seed, sample_each=64):
    """TRAIN-only advisory using the REAL SLAI QualityAgent with ephemeral memory.

    SLAI's QualityAgent v2.3 constructs its QualityMemory directly and does
    not expose constructor injection. We override just that construction,
    scoped to this single offline QualityAgent initialization. Neither the
    operational SQLite backend nor its settings/history are modified.
    """
    from src.agents import quality_agent as qa_module
    from src.agents.quality import quality_memory as qm_module

    configured = qm_module.get_config_section("quality_memory") or {}
    if not isinstance(configured, dict):
        raise TypeError("SLAI quality_memory config must be a mapping")

    offline_config = deepcopy(configured)
    offline_config["backend"] = "memory"
    offline_config["auto_persist"] = False
    offline_config["bootstrap_on_start"] = False
    offline_config["journaling"] = {**dict(offline_config.get("journaling") or {}), "enabled": False}

    offline_memory = qm_module.QualityMemory(memory_config_override=offline_config)
    if getattr(offline_memory, "backend", None) != "memory":
        raise RuntimeError("Offline QualityMemory must use in-memory backend")

    # QualityAgent plus FOUR of its existing subcomponents independently call
    # QualityMemory() in their constructors. Patch each module's locally
    # imported class symbol so they all receive ONE isolated memory instance.
    # The real class symbols are restored immediately afterwards. This CLI
    # constructs the agent synchronously; never use this scope across threads.
    modules = [qa_module] + [
        import_module("src.agents.quality." + name)
        for name in ("structural_quality", "statistical_quality", "semantic_quality", "workflow_control")
    ]
    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(patch.object(module, "QualityMemory", side_effect=lambda: offline_memory))
        agent = qa_module.QualityAgent(
            shared_memory=InProcessCoordination(), agent_factory=None,
            config={"auto_route_via_workflow": False,
                    "publish_to_shared_memory": False,
                    "shared_memory": {"enabled": False,
                                      "publish_notifications": False}},
        )

    if getattr(agent.quality_memory, "backend", None) != "memory":
        raise RuntimeError("QualityAgent did not receive isolated offline memory")
    for child_name in ("structural_quality", "statistical_quality", "semantic_quality", "workflow_control"):
        child = getattr(agent, child_name)
        if child.memory is not agent.quality_memory:
            raise RuntimeError("QualityAgent subsystem memory is not shared: " + child_name)
    rng = random.Random(seed)
    outcome = {}
    for split in ("train",):
        rows = []
        for label in (0, 1):
            bucket = [(x, y) for x, y in subsets[split] if int(y[0]) == label]
            rng.shuffle(bucket)
            for features, target in bucket[:sample_each]:
                row = {name: float(value) for name, value in zip(FEATURE_NAMES, features)}
                row.update({"label": int(target[0])})
                rows.append(row)
        result = agent.evaluate_batch(
            rows,
            dataset_id="safety-http32-" + dataset_digest[:16],
            source_id="csic_or_supplied_labeled_http_requests",
            batch_id="offline_" + split + "_" + dataset_digest[:12],
            label_field="label", feature_fields=list(FEATURE_NAMES),
            use_case="model_training",
            context={"pipeline": "offline_safety_training", "split": split},
        )
        if not isinstance(result, dict):
            raise TypeError("QualityAgent.evaluate_batch must return a mapping")
        outcome[split] = {
            "verdict": str(result.get("verdict", "unknown")),
            "batch_score": result.get("batch_score"),
            "quarantine_count": result.get("quarantine_count"),
            "flags": list(result.get("flags") or [])[:24],
            "examined": len(rows),
            "storage": "isolated_in_memory_not_production_sqlite",
        }
    return outcome


def provenance_record(dataset_digest, dataset_path, candidate, feature_digest, metadata):
    """ProvenanceAgent PUBLIC register_artifact/record_derivation.

    The agent owns its configured persistent store; no shadow provenance DB.
    No request payloads or security keys are registered.
    """
    from src.agents.provenance_agent import ProvenanceAgent
    agent = ProvenanceAgent(shared_memory=InProcessCoordination(), agent_factory=None,
                            config={"publish_shared_memory": False,
                                    "publish_state_updates": False})
    dataset_id = "safety-dataset-" + dataset_digest[:24]
    model_digest = sha256_path(candidate)
    model_id = "safety-mlp32-" + model_digest[:24]
    agent.register_artifact(
        dataset_id, artifact_type="dataset", label="safety HTTP labeled dataset",
        digest=dataset_digest,
        metadata={"path_name": dataset_path.name, "study": "offline_experiment"},
    )
    agent.register_artifact(
        model_id, artifact_type="model", label="SLAI native MLP 32 features",
        digest=model_digest,
        metadata={"feature_contract_digest": feature_digest,
                  "candidate_name": candidate.name, "selected_variant": metadata["variant"]},
    )
    record = agent.record_model_lineage(
        model_id,
        transformation="train",
        training_dataset_ids=[dataset_id],
        architecture="32-16-8-1",
        model_version="safety_http32_experimental_v2",
        agent_id="safety_training",
        artifact_id=model_id,
        metadata={"feature_count": len(FEATURE_NAMES),
                  "model_sha256": model_digest},
    )
    return {"dataset_id": dataset_id, "model_id": model_id,
            "model_sha256": model_digest,
            "model_lineage_recorded": isinstance(record, dict)}


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(b)
    return digest.hexdigest()


def require_source_manifest(path):
    p = Path(str(path) + ".manifest.json")
    if not p.is_file():
        return {"manifest": "missing", "source": "unverified"}
    meta = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("Dataset manifest must be an object")
    digest = meta.get("jsonl_sha256") or meta.get("dataset_sha256")
    if digest and digest != sha256_path(path):
        raise ValueError("Dataset manifest SHA256 does not match data file")
    return {"manifest": "verified" if digest else "missing_hash",
            "source": str(meta.get("dataset_kind") or meta.get("origin") or "unverified"),
            "independent_live_field_data": bool(meta.get("independent_live_field_data", False))}


def threshold_search(labels, scores, min_precision, min_recall, max_fpr):
    pairs = sorted(zip(scores, labels), reverse=True)
    thresholds = [math.nextafter(pairs[0][0], math.inf)] + sorted(set(scores), reverse=True)
    candidates = []
    for t in thresholds:
        m = metrics(labels, scores, t)
        if m["fpr"] <= max_fpr:
            candidates.append((m["recall"], m["precision"], -m["fpr"], t, m))
    if not candidates:
        raise RuntimeError("No validation operating point under FPR cap")
    best = max(candidates)
    passing = [c for c in candidates if c[1] >= min_precision and c[0] >= min_recall]
    good = max(passing) if passing else None
    return {"best_at_fpr_cap": {"threshold": best[3], "metrics": best[4]},
            "feasible": good is not None,
            "threshold": (good or best)[3]}


def architecture_for(cfg, variant):
    layers = [dict(row) for row in cfg["layer_config"]]
    for layer in layers[:-1]:
        if variant == "plain":
            layer["dropout"] = 0.0
            layer["batch_norm"] = False
        elif variant == "regularized":
            layer["dropout"] = 0.05
            layer["batch_norm"] = True
        else:
            raise ValueError(f"Unknown variant: {variant}")
    return layers


def make_network(cfg, seed, variant):
    overrides = dict(cfg.get("neural_network_overrides", {}) or {})
    overrides.update({
        "deterministic_seed": seed,
        "set_seed_on_init": True,
        "restore_best_weights": True,
        "weight_decay_lambda": 0.0002 if variant == "plain" else 0.00001,
    })
    net = NeuralNetwork(
        num_inputs=len(FEATURE_NAMES),
        layer_config=architecture_for(cfg, variant),
        loss_function_name=cfg.get("loss_function_name", "cross_entropy"),
        optimizer_name=cfg.get("optimizer_name", "adam"),
        initialization_method_default=cfg.get("initialization_method_default", "he_normal"),
        problem_type="binary_classification",
        config=overrides,
    )
    return net, overrides


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/training/safety/cyber_risk_csic2010.jsonl"))
    p.add_argument("--candidate", type=Path, default=Path("artifacts/safety/cyber_safety_v2_integrated_candidate.json"))
    p.add_argument("--report", type=Path, default=Path("artifacts/safety/cyber_safety_v2_integrated_report.json"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.0005)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variants", default="plain,regularized", help="Validation-only search among native MLP variants")
    p.add_argument("--min-precision", type=float, default=0.90)
    p.add_argument("--min-recall", type=float, default=0.95)
    p.add_argument("--max-fpr", type=float, default=0.03)
    p.add_argument("--min-auc", type=float, default=0.90)
    p.add_argument("--quality-agent", action="store_true", help="Enable real QualityAgent advisory audit with isolated in-memory QualityMemory")
    p.add_argument("--provenance-agent", action="store_true", help="Enable actual ProvenanceAgent durable lineage")
    p.add_argument("--enforce-quality", action="store_true", help="Abort if QualityAgent issues a blocking verdict")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if min(a.epochs, a.batch_size, a.patience) < 1 or not 0 < a.learning_rate < 1:
        raise ValueError("Invalid training arguments")
    if any(not 0 <= x <= 1 for x in (a.min_precision, a.min_recall, a.max_fpr, a.min_auc)):
        raise ValueError("Metrics constraints must be in [0,1]")
    if a.enforce_quality and not a.quality_agent:
        raise ValueError("--enforce-quality requires --quality-agent")
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    if len(set(variants)) != len(variants) or not variants or any(x not in ("plain", "regularized") for x in variants):
        raise ValueError("--variants must be a unique subset of plain,regularized")
    candidate = a.candidate.resolve()
    report_file = a.report.resolve()
    if candidate == report_file:
        raise ValueError("Candidate and report path must be different")
    if not a.data.is_file():
        raise FileNotFoundError(a.data)
    manifest = require_source_manifest(a.data)
    print("SLAI v2 native model training; no production modification", flush=True)
    print("QualityAgent:", a.quality_agent, "ProvenanceAgent:", a.provenance_agent, flush=True)
    safety = HTTPTrainingSafetyModule(memory=OfflineMemory())
    subsets, info = parse_dataset(a.data, safety)
    require_minimums(subsets)
    if any(len(x) != len(FEATURE_NAMES) for rows in subsets.values() for x, _ in rows):
        raise ValueError("Wrong feature dimensionality")
    digest = info["file_sha256"]
    check = split_diagnostics(subsets["train"])
    print("Samples:", info["counts"], "Nonconstant features:", len(FEATURE_NAMES)-len(check["constant_train_features"]), flush=True)
    qa = None
    if a.quality_agent:
        qa = quality_audit(subsets, digest, a.seed)
        print("QualityAgent (isolated memory) verdicts:", {k: v["verdict"] for k, v in qa.items()}, flush=True)
        if a.enforce_quality and any(v["verdict"].lower() in ("block", "quarantine", "fail") for v in qa.values()):
            raise RuntimeError("QualityAgent blocked data. Aborting before training.")
    mean, std = mean_std(subsets["train"])
    prepared = {key: standardize(rows, mean, std) for key, rows in subsets.items()}
    cfg = dict(safety._cfg("model", {}) or {})
    if not cfg.get("layer_config"):
        raise RuntimeError("Missing native SLAI model.layer_config")
    print("Training native variants:", variants, "(selection uses only validation)", flush=True)
    candidates, trial_summaries = [], []
    for variant in variants:
        net, overrides = make_network(cfg, a.seed, variant)
        summary = net.train(
            prepared["train"], epochs=a.epochs, initial_learning_rate=a.learning_rate,
            batch_size=a.batch_size, early_stopping_patience=a.patience,
            validation_data=prepared["val"], verbose=False,
        )
        y_val, p_val = predictions(net, prepared["val"])
        roc = auc_roc(y_val, p_val)
        threshold_info = threshold_search(y_val, p_val, a.min_precision, a.min_recall, a.max_fpr)
        chosen = threshold_info["threshold"]
        val = dict(metrics(y_val, p_val, chosen), auc=roc)
        best = threshold_info["best_at_fpr_cap"]["metrics"]
        # Primary objective: recall at fixed FPR; secondary: AUC. These
        # selection dimensions are established before test is touched.
        ordering = (int(threshold_info["feasible"]), best["recall"], roc,
                    best["precision"], -val["logloss"])
        trial_summaries.append({"variant": variant, "epochs_completed": summary.epochs_completed,
                                "best_val_loss": summary.best_validation_loss,
                                "validation": val, "validation_threshold_feasible": threshold_info["feasible"]})
        candidates.append((ordering, variant, net, overrides, threshold_info))
        print(variant, "val_auc", round(roc, 5), "recall_at_fpr_cap", round(best["recall"], 5), flush=True)
    _, variant, model, overrides, chosen_info = max(candidates, key=lambda x: x[0])
    chosen = chosen_info["threshold"]
    model.binary_threshold = chosen
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if not model.save_model(str(candidate)):
        raise RuntimeError("Could not save native SLAI model")
    reloaded = NeuralNetwork.load_model(str(candidate), custom_config_override=overrides)
    if reloaded.num_inputs != len(FEATURE_NAMES) or abs(reloaded.binary_threshold - chosen) > 1e-8:
        raise RuntimeError("Signed checkpoint reload mismatch")
    contract_path = candidate.with_suffix(candidate.suffix + ".features.json")
    feature_contract = {
        "schema": "slai.cyber.http_32.standardized.v3.experimental",
        "names": list(FEATURE_NAMES), "num_inputs": len(FEATURE_NAMES),
        "normalization": {"mean": mean, "std": std, "clip": [-8.0, 8.0],
                          "fit_on": "train_only"},
        "source_hash": digest,
        "model_sha256": sha256_path(candidate),
        "selected_variant": variant,
        "note": "Required at inference; current production CyberSafetyModule is not compatible",
    }
    atomic_json(contract_path, feature_contract)
    feature_hash = sha256_path(contract_path)
    # Test predictions/metrics are inspected only AFTER candidate choice.
    # Note: the historic CSIC test set has already been reused in prior runs,
    # so this is exploratory, not an independent final generalization test.
    y_val, p_val = predictions(reloaded, prepared["val"])
    y_test, p_test = predictions(reloaded, prepared["test"])
    val = dict(metrics(y_val, p_val, chosen), auc=auc_roc(y_val, p_val))
    test = dict(metrics(y_test, p_test, chosen), auc=auc_roc(y_test, p_test))
    eligible = lambda d: d["precision"] >= a.min_precision and d["recall"] >= a.min_recall and d["fpr"] <= a.max_fpr and d["auc"] >= a.min_auc
    lab_passed = bool(chosen_info["feasible"] and eligible(val) and eligible(test))
    prov = None
    if a.provenance_agent:
        prov = provenance_record(digest, a.data, candidate, feature_hash, {"variant": variant})
    report = {
        "schema": "slai.cyber_safety.v2.integrated_experiment.v1",
        "created_at": utc_now(), "dataset": info, "manifest": manifest,
        "feature_contract": str(contract_path), "feature_contract_sha256": feature_hash,
        "dataset_diagnostics": check,
        "agent_contributions": {
            "Safety/CyberSafetyModule": "Cold-start initialization, configured safety context; HTTP-specific offline features",
            "QualityAgent": qa if qa is not None else "not_requested",
            "ProvenanceAgent": prov if prov is not None else "not_requested",
            "EvaluationAgent": "not_integrated: custom binary classification metrics fit the actual task",
            "LearningAgent/DQN/MAML": "not_integrated: reinforcement learning not justified for labeled supervised data",
            "OptimizationAgent": "not_integrated: native, reproducible validation-only MLP variant comparison",
        },
        "variants": trial_summaries, "selected_variant": variant,
        "checkpoint": str(candidate), "signed_reload_passed": True,
        "selected_validation_threshold": chosen,
        "validation_best_under_fpr": chosen_info["best_at_fpr_cap"],
        "validation_feasible": chosen_info["feasible"],
        "validation_metrics": val, "test_metrics": test,
        "lab_gates_passed": lab_passed,
        "production_eligible": False,
        "independent_test_status": "exploratory_reused_CSIC_test_from_previous_experiments",
        "reason": "Lab dataset and incompatible production feature contract; external field validation required",
        "warning": "Never install this candidate into 8-input CyberSafetyModule",
    }
    atomic_json(report_file, report)
    print("SELECTED:", variant, flush=True)
    print("VALIDATION:", json.dumps(val, sort_keys=True), flush=True)
    print("TEST:", json.dumps(test, sort_keys=True), flush=True)
    print("LAB GATES:", lab_passed, "PRODUCTION: DISABLED", flush=True)
    print("ARTIFACT:", candidate, "CONTRACT:", contract_path, "REPORT:", report_file, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
