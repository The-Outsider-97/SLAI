"""Trainable, preview-only Python test generation for the Evaluation Agent.

A small conditional byte-level encoder/decoder learns from *verified* source/test
pairs. Generated code is never executed or written to a repository here. Static
screening is advisory, not a security sandbox or proof of behavioral correctness.
Torch is imported only for training or when a checkpoint is actually used.

Operational events are emitted through the shared ``logger`` (installed by the
embedding application via ``logs.logger.configure_logging``). Semantic failures
raise typed exceptions from :mod:`..utils.evaluation_errors`. CLI output is JSON
by default; ``--format pretty`` swaps in :class:`PrettyPrinter` for human review.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import tempfile
import threading

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils.evaluation_errors import *
from ..utils.evaluation_helpers import *
from logs.logger import get_logger, PrettyPrinter  # type: ignore


logger = get_logger("Test generator")
printer = PrettyPrinter()


SCHEMA = "slai.evaluation.test_generator.v1"
PAD, BOS, EOS, OFFSET, VOCAB = 0, 1, 2, 4, 260
CATEGORIES = frozenset({
    "unit", "boundary", "regression", "integration", "property",
    "failure_driven", "parameterized", "improvement",
})
MODES = frozenset({"disabled", "shadow", "advisory", "active"})
FORBIDDEN_IMPORTS = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "ctypes", "importlib",
    "multiprocessing", "urllib", "requests", "http", "ftplib", "pathlib",
    "pickle", "marshal", "builtins", "signal",
})
FORBIDDEN_CALLS = frozenset({
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "getattr", "setattr", "delattr", "globals", "locals",
})
MAX_SOURCE_BYTES = 16384
MAX_TEST_BYTES = 8192


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes(text: str, maximum: int) -> list[int]:
    return [b + OFFSET for b in text.encode("utf-8")[:maximum]]


def _decode(tokens: Sequence[int]) -> str:
    return bytes(n - OFFSET for n in tokens if OFFSET <= n < VOCAB).decode(
        "utf-8", errors="replace"
    )


def _source_prompt(record: Mapping[str, Any]) -> str:
    source = record.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValidationFailureError(
            "source_non_empty", source, "non-empty Python source string"
        )
    source_size = len(source.encode("utf-8"))
    if source_size > MAX_SOURCE_BYTES:
        raise ValidationFailureError(
            "source_max_bytes", source_size, f"<= {MAX_SOURCE_BYTES} bytes"
        )
    ast.parse(source)
    parts = [source]
    for label in ("requirements", "verified_failure"):
        val = record.get(label, "")
        if val:
            if not isinstance(val, str) or len(val) > 4000:
                raise ValidationFailureError(
                    f"{label}_text_max_4000",
                    type(val).__name__,
                    "text up to 4000 characters",
                )
            parts.append("\n# " + label + ": " + val.replace("\n", " ") + "\n")
    return "\n".join(parts)


def validate_candidate(
    test_source: str,
    *,
    target_symbol: str = "",
    target_module: str = "",
) -> dict[str, Any]:
    """Conservative static lint only; NEVER interprets or imports a candidate."""
    issues: list[str] = []
    if (
        not isinstance(test_source, str)
        or len(test_source.encode("utf-8")) > MAX_TEST_BYTES
    ):
        return {"status": "rejected", "issues": ["invalid_test_source_size"]}
    try:
        tree = ast.parse(test_source)
    except (SyntaxError, ValueError) as exc:
        return {
            "status": "rejected",
            "issues": ["syntax_error:" + type(exc).__name__],
        }
    test_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    if not test_functions:
        issues.append("no_pytest_test_function")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for module in modules:
                if module.split(".")[0] in FORBIDDEN_IMPORTS:
                    issues.append("unsafe_import:" + module)
                if isinstance(node, ast.ImportFrom) and node.level:
                    issues.append("relative_import_requires_review")
        if isinstance(node, ast.Call):
            fun = node.func
            if isinstance(fun, ast.Name) and fun.id in FORBIDDEN_CALLS:
                issues.append("unsafe_call:" + fun.id)
            if isinstance(fun, ast.Attribute) and fun.attr in {
                "system", "popen", "run", "Popen", "remove", "unlink",
                "rmtree", "write_text", "write_bytes", "connect", "request",
                "urlopen",
            }:
                issues.append("unsafe_method:" + fun.attr)
        if isinstance(node, ast.Assert) and (
            isinstance(node.test, ast.Constant)
            or isinstance(node.test, ast.Name)
            and node.test.id in {"True", "False"}
        ):
            issues.append("trivial_assertion")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(
                isinstance(decorator, ast.Attribute)
                and decorator.attr in {"skip", "xfail"}
                for decorator in node.decorator_list
            ):
                issues.append("skipped_or_expected_failure")
    assertions = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    raises = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "raises"
    ]
    if not assertions and not raises:
        issues.append("no_meaningful_assertion")
    if target_symbol and not any(
        isinstance(n, (ast.Name, ast.Attribute))
        and (n.id if isinstance(n, ast.Name) else n.attr) == target_symbol
        for n in ast.walk(tree)
    ):
        issues.append("target_not_referenced")
    if target_module:
        if not re.fullmatch(
            r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", target_module
        ):
            issues.append("invalid_target_module")
        else:
            imports = [
                n.module
                for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom)
            ]
            imports += [
                a.name
                for n in ast.walk(tree)
                if isinstance(n, ast.Import)
                for a in n.names
            ]
            if not any(
                n == target_module or n.startswith(target_module + ".")
                for n in imports
                if n
            ):
                issues.append("target_module_import_not_found")
    issues = sorted(set(issues))
    return {
        "status": "draft_requires_review" if issues else "static_candidate",
        "issues": issues,
        "assertions": len(assertions),
        "raises_checks": len(raises),
        "test_functions": len(test_functions),
        "execution": "not_executed",
        "behavioral_oracle_verified": False,
    }


def _examples(path: Path) -> list[dict[str, Any]]:
    logger.debug("Loading verified examples from %s", path)
    examples: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataIntegrityError(
                    f"dataset line {line_no}",
                    f"invalid JSON: {exc}",
                    {"line": line_no},
                ) from exc
            if not isinstance(record, dict):
                raise DataIntegrityError(
                    f"dataset line {line_no}",
                    "expected JSON object",
                    {"line": line_no},
                )
            # All supervised examples must be traceable and explicitly verified.
            if record.get("verified") is not True:
                continue
            for key in ("license", "provenance", "group", "target_symbol"):
                if not isinstance(record.get(key), str) or not record[key].strip():
                    raise DataIntegrityError(
                        f"dataset line {line_no}",
                        f"missing or invalid required field '{key}'",
                        {"line": line_no, "field": key},
                    )
            prompt = _source_prompt(record)
            target = record.get("test")
            if not isinstance(target, str) or not target.strip():
                raise DataIntegrityError(
                    f"dataset line {line_no}",
                    "test must be a non-empty string",
                    {"line": line_no},
                )
            if len(target.encode("utf-8")) > MAX_TEST_BYTES:
                raise DataIntegrityError(
                    f"dataset line {line_no}",
                    f"test exceeds {MAX_TEST_BYTES} bytes",
                    {"line": line_no},
                )
            check = validate_candidate(
                target,
                target_symbol=str(record["target_symbol"]),
                target_module=str(record.get("target_module", "")),
            )
            if check["status"] != "static_candidate":
                raise DataIntegrityError(
                    f"dataset line {line_no}",
                    f"verified test failed static screening: {check['issues']}",
                    {"line": line_no, "issues": check["issues"]},
                )
            fingerprint = _digest((prompt + "\0" + target).encode("utf-8"))
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            examples.append(
                {
                    "source": prompt,
                    "test": target,
                    "group": record["group"],
                    "provenance": record["provenance"],
                    "license": record["license"],
                    "target_symbol": record["target_symbol"],
                    "target_module": record.get("target_module", ""),
                    "fingerprint": fingerprint,
                    "synthetic": bool(record.get("synthetic", False)),
                }
            )
    logger.debug("Loaded %d unique verified examples", len(examples))
    return examples


def prepare_dataset(source_file: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Split by group, not by example; never create fictitious training labels."""
    source_path = Path(source_file)
    logger.info("Preparing TestGenerator dataset from %s", source_path)
    items = _examples(source_path)
    groups = sorted(
        {r["group"] for r in items}, key=lambda g: _digest(g.encode())
    )
    if len(groups) < 5 or len(items) < 10:
        raise DataIntegrityError(
            "dataset",
            "need >=10 verified unique examples from >=5 independent groups",
            {"examples": len(items), "groups": len(groups)},
        )
    group_partition: dict[str, str] = {}
    n = len(groups)
    for i, group in enumerate(groups):
        group_partition[group] = (
            "test"
            if i < max(1, n // 5)
            else ("validation"
                  if i < max(1, n // 5) + max(1, n // 5)
                  else "train")
        )
    partitions = {
        name: [r for r in items if group_partition[r["group"]] == name]
        for name in ("train", "validation", "test")
    }
    if any(not rows for rows in partitions.values()):
        raise DataIntegrityError(
            "dataset",
            "insufficient independent groups for all splits",
            {"group_counts": {k: len(v) for k, v in partitions.items()}},
        )
    dest = Path(output_dir)
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CheckpointError(
            "create", str(dest), f"could not create output directory: {exc}"
        ) from exc
    hashes = {}
    for name, rows in partitions.items():
        payload = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        )
        encoded = payload.encode("utf-8")
        # Write bytes, not text: text mode turns "\n" into "\r\n" on Windows,
        # so the file on disk would no longer match the recorded digest.
        (dest / (name + ".jsonl")).write_bytes(encoded)
        hashes[name] = _digest(encoded)
    summary = {
        "schema": SCHEMA,
        "counts": {k: len(v) for k, v in partitions.items()},
        "groups": {
            k: len({r["group"] for r in v}) for k, v in partitions.items()
        },
        "sha256": hashes,
        "synthetic_count": sum(r["synthetic"] for r in items),
    }
    (dest / "dataset_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info(
        "Prepared TestGenerator dataset: counts=%s groups=%s",
        summary["counts"],
        summary["groups"],
    )
    return summary


def _network(embed_dim: int, hidden_dim: int):
    """Small conditional generative model; no torch dependency at module import."""
    from torch import nn  # type: ignore

    class ByteSeq2Seq(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(VOCAB, embed_dim, padding_idx=PAD)
            self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
            self.decoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, VOCAB)

        def forward(self, source, target):
            _, state = self.encoder(self.embedding(source))
            predictions, _ = self.decoder(self.embedding(target), state)
            return self.head(predictions)

        def generate(self, source, max_bytes: int) -> list[int]:
            import torch  # type: ignore

            _, state = self.encoder(self.embedding(source))
            token = torch.tensor(
                [[BOS]], dtype=torch.long, device=source.device
            )
            result = []
            for _ in range(max_bytes):
                value, state = self.decoder(self.embedding(token), state)
                token = self.head(value[:, -1:, :]).argmax(dim=-1)
                index = int(token.item())
                if index == EOS:
                    break
                result.append(index)
            return result

    return ByteSeq2Seq()


def _read_prepared(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DataIntegrityError(
            "prepared_data", f"not valid UTF-8: {path}"
        ) from exc
    try:
        rows = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as exc:
        raise DataIntegrityError(
            "prepared_data", f"invalid JSON in {path}: {exc}"
        ) from exc
    if any(
        not isinstance(r, dict) or "source" not in r or "test" not in r
        for r in rows
    ):
        raise DataIntegrityError(
            "prepared_data", f"missing required fields in {path}"
        )
    return rows


def train_model(
    prepared_dir: str | Path,
    checkpoint_dir: str | Path,
    *,
    epochs: int = 3,
    seed: int = 42,
    embed_dim: int = 48,
    hidden_dim: int = 96,
    max_source_bytes: int = 512,
    max_target_bytes: int = 768,
) -> dict[str, Any]:
    """Explicit offline training. Saves *unapproved* weights; never promotes a run."""
    # Validate numeric configuration up-front (fails fast, before torch import).
    epochs = _require_positive_int(epochs, "epochs")
    embed_dim = _require_positive_int(embed_dim, "embed_dim")
    hidden_dim = _require_positive_int(hidden_dim, "hidden_dim")
    max_source_bytes = _require_positive_int(
        max_source_bytes, "max_source_bytes"
    )
    max_target_bytes = _require_positive_int(
        max_target_bytes, "max_target_bytes"
    )
    if not 1 <= epochs <= 100:
        raise ConfigLoadError(
            "<config>", "epochs", f"expected 1 <= epochs <= 100, got {epochs}"
        )
    if not 8 <= embed_dim <= 512:
        raise ConfigLoadError(
            "<config>",
            "embed_dim",
            f"expected 8 <= embed_dim <= 512, got {embed_dim}",
        )
    if not 16 <= hidden_dim <= 1024:
        raise ConfigLoadError(
            "<config>",
            "hidden_dim",
            f"expected 16 <= hidden_dim <= 1024, got {hidden_dim}",
        )
    if not 32 <= max_source_bytes <= 4096:
        raise ConfigLoadError(
            "<config>",
            "max_source_bytes",
            f"expected 32 <= max_source_bytes <= 4096, got {max_source_bytes}",
        )
    if not 64 <= max_target_bytes <= 4096:
        raise ConfigLoadError(
            "<config>",
            "max_target_bytes",
            f"expected 64 <= max_target_bytes <= 4096, got {max_target_bytes}",
        )

    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    source_dir = Path(prepared_dir)
    manifest_path = source_dir / "dataset_manifest.json"
    try:
        dataset_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except FileNotFoundError as exc:
        raise CheckpointError(
            "load", str(manifest_path), "dataset manifest not found"
        ) from exc
    except json.JSONDecodeError as exc:
        raise DataIntegrityError(
            "dataset_manifest",
            f"invalid JSON in {manifest_path}: {exc}",
            {"path": str(manifest_path)},
        ) from exc
    if dataset_manifest.get("schema") != SCHEMA:
        raise DataIntegrityError(
            "dataset_manifest",
            f"incompatible schema (expected {SCHEMA!r})",
            {"path": str(manifest_path)},
        )
    for partition in ("train", "validation", "test"):
        partition_path = source_dir / (partition + ".jsonl")
        expected_digest = dataset_manifest.get("sha256", {}).get(partition)
        if not expected_digest:
            raise DataIntegrityError(
                f"prepared_data/{partition}",
                "missing sha256 in dataset manifest",
                {"path": str(manifest_path)},
            )
        try:
            actual_digest = _file_hash(partition_path)
        except FileNotFoundError as exc:
            raise DataIntegrityError(
                f"prepared_data/{partition}",
                f"file not found: {partition_path}",
            ) from exc
        if actual_digest != expected_digest:
            raise DataIntegrityError(
                f"prepared_data/{partition}",
                "sha256 mismatch between manifest and on-disk file",
                {"path": str(partition_path)},
            )

    records = _read_prepared(source_dir / "train.jsonl")
    validation = _read_prepared(source_dir / "validation.jsonl")
    holdout = _read_prepared(source_dir / "test.jsonl")
    if not records or not validation or not holdout:
        raise DataIntegrityError(
            "prepared_data", "requires non-empty train/validation/test splits"
        )
    if {r["group"] for r in records} & {
        r["group"] for r in validation + holdout
    }:
        raise DataIntegrityError(
            "prepared_data", "train/holdout group contamination detected"
        )
    if {r["group"] for r in validation} & {r["group"] for r in holdout}:
        raise DataIntegrityError(
            "prepared_data", "validation/test group contamination detected"
        )

    logger.info(
        "Training TestGenerator: train=%d val=%d test=%d epochs=%d seed=%d",
        len(records),
        len(validation),
        len(holdout),
        epochs,
        seed,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    model = _network(embed_dim, hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    best_state = None
    best_loss = math.inf
    history: list[dict[str, float]] = []

    def measure(rows: Sequence[Mapping[str, Any]], training: bool) -> float:
        model.train(training)
        accumulated = 0.0
        count = 0
        order = list(rows)
        if training:
            random.shuffle(order)
        with torch.enable_grad() if training else torch.no_grad():
            for row in order:
                src = torch.tensor(
                    [_bytes(str(row["source"]), max_source_bytes) or [EOS]],
                    dtype=torch.long,
                )
                target = (
                    [BOS]
                    + _bytes(str(row["test"]), max_target_bytes - 2)
                    + [EOS]
                )
                inp = torch.tensor([target[:-1]], dtype=torch.long)
                expected = torch.tensor(target[1:], dtype=torch.long)
                logits = model(src, inp)
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB), expected
                )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                accumulated += float(loss.detach().item())
                count += 1
        return accumulated / max(count, 1)

    for epoch_index in range(epochs):
        training_loss = measure(records, training=True)
        validation_loss = measure(validation, training=False)
        history.append(
            {"train_loss": training_loss, "validation_loss": validation_loss}
        )
        logger.debug(
            "Epoch %d/%d train_loss=%.4f val_loss=%.4f",
            epoch_index + 1,
            epochs,
            training_loss,
            validation_loss,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is None or not math.isfinite(best_loss):
        raise ModelRuntimeError(
            "training", "TestGenerator", "no valid model state was produced"
        )

    output = Path(checkpoint_dir)
    try:
        output.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CheckpointError(
            "create", str(output), f"could not create checkpoint directory: {exc}"
        ) from exc
    checkpoint = output / "weights.pt"
    # No pickled module objects. Inference only uses torch.load(weights_only=True).
    with tempfile.NamedTemporaryFile(
        dir=output, suffix=".pt", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(best_state, temporary)
        os.replace(temporary, checkpoint)
    except OSError as exc:
        raise CheckpointError(
            "save", str(checkpoint), f"could not write checkpoint: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "schema": SCHEMA,
        "model": "TestGenerator",
        "model_type": "byte_gru_seq2seq",
        "promotion": "experimental",
        "checkpoint": "weights.pt",
        "checkpoint_sha256": _file_hash(checkpoint),
        "hyperparameters": {
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "max_source_bytes": max_source_bytes,
            "max_target_bytes": max_target_bytes,
        },
        "seed": seed,
        "epochs": epochs,
        "train_count": len(records),
        "validation_count": len(validation),
        "reserved_test_count": len(holdout),
        "validation_loss": best_loss,
        "history": history,
        "data_sha256": {
            part: _file_hash(source_dir / (part + ".jsonl"))
            for part in ("train", "validation", "test")
        },
        "automatic_execution_authorized": False,
    }
    temp_manifest = output / "manifest.json.tmp"
    temp_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temp_manifest, output / "manifest.json")
    logger.info(
        "TestGenerator trained: best_validation_loss=%.4f checkpoint=%s",
        best_loss,
        checkpoint,
    )
    return manifest


class TestGenerator:
    """Versioned locally trained generator; no code execution or repository writes."""

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path | None = None,
        mode: str = "disabled",
    ) -> None:
        if mode not in MODES:
            raise UnsupportedOperationError(
                "TestGenerator mode",
                f"got {mode!r}, expected one of {sorted(MODES)}",
            )
        self.mode = mode
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._lock = threading.RLock()
        self._network = None
        self._manifest: dict[str, Any] | None = None

    def health(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "mode": self.mode,
            "loaded": self._network is not None,
            "checkpoint_configured": self.checkpoint_dir is not None,
            "execution_authorized": False,
        }

    def _load(self) -> None:
        if self._network is not None:
            return
        if self.checkpoint_dir is None:
            raise CheckpointError(
                "load",
                "<unconfigured>",
                "no TestGenerator checkpoint directory configured",
            )
        root = self.checkpoint_dir.resolve()
        manifest_path = root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CheckpointError(
                "load", str(manifest_path), "manifest not found"
            ) from exc
        except json.JSONDecodeError as exc:
            raise DataIntegrityError(
                "checkpoint_manifest",
                f"invalid JSON in {manifest_path}: {exc}",
                {"path": str(manifest_path)},
            ) from exc
        if (
            manifest.get("schema") != SCHEMA
            or manifest.get("model") != "TestGenerator"
        ):
            raise DataIntegrityError(
                "checkpoint_manifest",
                "incompatible schema or model identifier",
                {"path": str(manifest_path)},
            )
        if manifest.get("promotion") not in ("experimental", "approved"):
            raise DataIntegrityError(
                "checkpoint_manifest",
                "invalid promotion state",
                {"promotion": manifest.get("promotion")},
            )
        if manifest.get("checkpoint") != "weights.pt":
            raise DataIntegrityError(
                "checkpoint_manifest",
                "unsafe checkpoint filename",
                {"checkpoint": manifest.get("checkpoint")},
            )
        params = manifest.get("hyperparameters")
        if not isinstance(params, dict):
            raise DataIntegrityError(
                "checkpoint_manifest", "missing hyperparameters object"
            )
        embed = params.get("embed_dim")
        hidden = params.get("hidden_dim")
        if (
            type(embed) is not int
            or type(hidden) is not int
            or not 8 <= embed <= 512
            or not 16 <= hidden <= 1024
        ):
            raise DataIntegrityError(
                "checkpoint_manifest",
                "invalid model dimensions",
                {"embed_dim": embed, "hidden_dim": hidden},
            )
        weights = root / "weights.pt"
        if not weights.is_file():
            raise CheckpointError(
                "load", str(weights), "checkpoint file not found"
            )
        if _file_hash(weights) != manifest.get("checkpoint_sha256"):
            raise CheckpointError(
                "load", str(weights), "checkpoint integrity verification failed"
            )
        import torch  # type: ignore

        model = _network(embed, hidden)
        try:
            state = torch.load(weights, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise SerializationError(
                "checkpoint",
                f"torch.load failed: {exc}",
                {"path": str(weights)},
            ) from exc
        if not isinstance(state, dict):
            raise SerializationError(
                "checkpoint",
                "checkpoint payload is not a state dict",
                {"path": str(weights)},
            )
        try:
            model.load_state_dict(state, strict=True)
        except (RuntimeError, ValueError) as exc:
            raise SerializationError(
                "checkpoint",
                f"state dict incompatible with model: {exc}",
                {"path": str(weights)},
            ) from exc
        model.eval()
        self._network = model
        self._manifest = manifest
        logger.info(
            "TestGenerator checkpoint loaded (sha256=%s…)",
            str(manifest["checkpoint_sha256"])[:12],
        )

    def generate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise TypeError("TestGenerator request must be a mapping")
        if self.mode == "disabled":
            return {
                "schema": SCHEMA,
                "status": "disabled",
                "execution": "not_executed",
            }
        if str(request.get("framework", "pytest")) != "pytest":
            return {
                "schema": SCHEMA,
                "status": "unsupported_framework",
                "execution": "not_executed",
            }
        category = str(request.get("category", "unit"))
        if category not in CATEGORIES:
            return {
                "schema": SCHEMA,
                "status": "unsupported_category",
                "execution": "not_executed",
            }
        target_symbol = str(request.get("target_symbol", ""))
        target_module = str(request.get("target_module", ""))

        try:
            prompt = _source_prompt(request)
            if not target_symbol or not target_symbol.isidentifier():
                raise ValueError(
                    "target_symbol must identify a Python callable"
                )
        except (SyntaxError, TypeError, ValueError, EvaluationError) as exc:
            error_detail = (
                exc.to_dict() if isinstance(exc, EvaluationError) else None
            )
            logger.warning(
                "TestGenerator.generate rejected invalid request: %s", exc
            )
            return {
                "schema": SCHEMA,
                "status": "invalid_request",
                "error": str(exc),
                "error_detail": error_detail,
                "execution": "not_executed",
            }

        try:
            with self._lock:
                self._load()
                assert self._manifest is not None and self._network is not None
                if (
                    self.mode in ("advisory", "active")
                    and self._manifest["promotion"] != "approved"
                ):
                    return {
                        "schema": SCHEMA,
                        "status": "checkpoint_not_approved",
                        "execution": "not_executed",
                    }
                import torch  # type: ignore

                max_source = self._manifest["hyperparameters"]["max_source_bytes"]
                max_target = self._manifest["hyperparameters"]["max_target_bytes"]
                source = torch.tensor(
                    [_bytes(prompt, max_source) or [EOS]], dtype=torch.long
                )
                with torch.no_grad():
                    output = self._network.generate(source, max_target)
                candidate = _decode(output)
                validation = validate_candidate(
                    candidate,
                    target_symbol=target_symbol,
                    target_module=target_module,
                )
                result_status = (
                    "shadow"
                    if self.mode == "shadow"
                    else validation["status"]
                )
                logger.debug(
                    "TestGenerator.generate mode=%s category=%s status=%s",
                    self.mode,
                    category,
                    result_status,
                )
                return {
                    "schema": SCHEMA,
                    "status": result_status,
                    "category": category,
                    "framework": "pytest",
                    "model": "TestGenerator",
                    "checkpoint_sha256": self._manifest["checkpoint_sha256"],
                    "generated_test": candidate,
                    "validation": validation,
                    "assumptions": [
                        "expected_behavior_not_independently_verified"
                    ],
                    "execution": "not_executed",
                    "integration": "not_applied",
                    "review_required": True,
                }
        except Exception as exc:
            # Untrusted/corrupt checkpoints must fail closed, including malformed
            # torch payloads that may raise errors outside ValueError/OSError.
            error_detail = (exc.to_dict() if isinstance(exc, EvaluationError) else None)
            logger.exception("TestGenerator.generate failed closed (mode=%s)", self.mode)
            return {
                "schema": SCHEMA,
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_detail": error_detail,
                "execution": "not_executed",
            }


def evaluate_model(prepared_dir: str | Path, checkpoint_dir: str | Path, *, limit: int = 25) -> dict[str, Any]:
    """Read-only held-out *static* evaluation; no generated tests are run."""
    limit = _require_positive_int(limit, "limit")
    generator = TestGenerator(checkpoint_dir=checkpoint_dir, mode="shadow")
    rows = _read_prepared(Path(prepared_dir) / "test.jsonl")
    statuses: dict[str, int] = {}
    duplicates: set[str] = set()
    seen: set[str] = set()
    for row in rows[:limit]:
        result = generator.generate(
            {
                "source": row["source"],
                "target_symbol": row["target_symbol"],
                "target_module": row.get("target_module", ""),
                "category": "unit",
            }
        )
        validation = result.get("validation", {})
        status = str(validation.get("status", result.get("status", "unknown")))
        statuses[status] = statuses.get(status, 0) + 1
        output = str(result.get("generated_test", ""))
        if output in seen:
            duplicates.add(output)
        seen.add(output)
    summary = {
        "schema": SCHEMA,
        "evaluated": min(limit, len(rows)),
        "static_status_counts": statuses,
        "duplicate_generated_outputs": len(duplicates),
        "execution": "not_executed",
        "behavioral_correctness": "unverified",
    }
    logger.info(
        "TestGenerator static evaluation: evaluated=%d statuses=%s duplicates=%d",
        summary["evaluated"],
        statuses,
        summary["duplicate_generated_outputs"],
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _report_cli_error(exc: BaseException, fmt: str) -> None:
    detail: dict[str, Any] = {
        "status": "error",
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    if isinstance(exc, EvaluationError):
        detail["detail"] = exc.to_dict()
    if fmt == "pretty":
        printer.status("error", str(exc), "error")
        if isinstance(exc, EvaluationError):
            printer.pretty("detail", exc.to_dict(), status="error")
    else:
        print(json.dumps(detail, indent=2, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SLAI TestGenerator (offline training and inspection)")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--format",
        choices=("json", "pretty"),
        default="json",
        help="Output format for the result payload (default: json)",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    p = subs.add_parser("prepare", parents=[common])
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    t = subs.add_parser("train", parents=[common])
    t.add_argument("--prepared", required=True)
    t.add_argument("--checkpoint", required=True)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--embed-dim", type=int, default=48)
    t.add_argument("--hidden-dim", type=int, default=96)

    e = subs.add_parser("evaluate", parents=[common])
    e.add_argument("--prepared", required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--limit", type=int, default=25)

    args = parser.parse_args(argv)
    fmt = getattr(args, "format", "json")

    try:
        if args.command == "prepare":
            logger.info("CLI prepare: input=%s output=%s", args.input, args.output)
            result = prepare_dataset(args.input, args.output)
        elif args.command == "train":
            logger.info(
                "CLI train: prepared=%s checkpoint=%s",
                args.prepared,
                args.checkpoint,
            )
            result = train_model(
                args.prepared,
                args.checkpoint,
                epochs=args.epochs,
                seed=args.seed,
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
            )
        else:
            logger.info(
                "CLI evaluate: prepared=%s checkpoint=%s limit=%d",
                args.prepared,
                args.checkpoint,
                args.limit,
            )
            result = evaluate_model(args.prepared, args.checkpoint, limit=args.limit)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        logger.exception("TestGenerator CLI failed for command=%s", args.command)
        _report_cli_error(exc, fmt)
        return 1

    if fmt == "pretty":
        printer.section_header(f"TestGenerator {args.command} result")
        printer.pretty("result", result)
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
