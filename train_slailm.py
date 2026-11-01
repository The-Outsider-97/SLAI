"""Train SLAILM_84M using SLAI's native LanguageTransformer and LanguageTokenizer.

This module intentionally does not introduce a second tokenizer, model, or
application-agent framework. It consumes curated LANTRA JSONL, performs
sequence-to-sequence and contrastive training, and writes checkpoints readable
by LanguageTransformer.load_language_model / LantraRuntime.

Run from the SLAI repository root: ``python -m train_slailm --dry-run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


TASKS = frozenset({
    "dialogue", "generation", "classification", "translation",
    "summarization", "embedding", "reranking",
})
PAIR_TASKS = frozenset({"embedding", "reranking"})
TARGET_PARAMETERS = 84_000_000
DEFAULT_DATA = "data/processed/lantra/supervised"
DEFAULT_OUTPUT = "src/agents/language/checkpoints/slailm_84m"
ARCHITECTURES = {
    "84m": {"d_model": 384, "nhead": 8, "num_encoder_layers": 6,
            "num_decoder_layers": 6, "dim_feedforward": 1792},
    "smoke": {"d_model": 64, "nhead": 4, "num_encoder_layers": 1,
              "num_decoder_layers": 1, "dim_feedforward": 256},
}
BENCHMARK_DIALOGUE_PROMPTS = (
    "hi", "hello", "how are you?", "the sky is blue",
    "the dog is black", "I want a bulldog", "I am a human, not a dog.",
)
ARCHITECTURE_KEYS = (
    "d_model", "nhead", "num_encoder_layers", "num_decoder_layers",
    "dim_feedforward", "norm_first", "src_vocab_size", "tgt_vocab_size",
    "pad_token_id", "bos_token_id", "eos_token_id",
    "tie_embeddings", "tie_output_projection", "max_position_embeddings",
)


@dataclass(frozen=True)
class Example:
    task: str
    split: str
    source: str
    target: str
    negative: str
    group: str
    record_id: str


def _require_string(value: Any, field: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: missing or empty '{field}'")
    return value.strip()


def _split_for(record: Mapping[str, Any], group: str) -> str:
    if "split" in record:
        split = str(record["split"]).lower().strip()
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split {split!r} for group {group!r}")
        return split
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if bucket < 10 else "validation" if bucket < 20 else "train"


def _unapproved_teacher_reason(row: Mapping[str, Any]) -> str | None:
    """Exclude teacher data until both review and output-use rights are recorded."""
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, Mapping) or not metadata.get("teacher_provider"):
        return None
    if metadata.get("review_status") != "approved" or not metadata.get("reviewed_by"):
        return "missing_independent_review"
    if not metadata.get("license_review_reference"):
        return "missing_output_use_rights"
    return None


def normalize_record(row: Mapping[str, Any], where: str) -> Example:
    """Use LANTRA v3's seven supervised contracts; reject undocumented fields.

    If the local train_lantra.py has further schema revisions, update this
    mapping after comparing its normalize_record contract.
    """
    task = _require_string(row.get("task"), "task", where).lower()
    if task not in TASKS:
        raise ValueError(f"{where}: unsupported task {task!r}; LSTP training is gated")
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{where}: metadata must be an object")
    teacher_reason = _unapproved_teacher_reason(row)
    if teacher_reason:
        raise ValueError(f"{where}: teacher data is not approved ({teacher_reason})")
    record_id = str(row.get("id") or where)
    if task == "dialogue":
        history = row.get("history")
        if isinstance(history, list):
            turns: list[str] = []
            for turn in history:
                if not isinstance(turn, Mapping):
                    raise ValueError(f"{where}: dialogue history must contain role/content objects")
                role = _require_string(turn.get("role"), "role", where).lower()
                content = _require_string(turn.get("content"), "content", where)
                turns.append(f"{role}: {content}")
            history_text = "\n".join(turns)
        else:
            history_text = _require_string(history, "history", where)
        source = f"task: dialogue\nconversation:\n{history_text}\nassistant:"
        target = _require_string(row.get("target"), "target", where)
        negative = ""
    elif task == "generation":
        source = f"task: generation\nprompt:\n{_require_string(row.get('input'), 'input', where)}\nresponse:"
        target = _require_string(row.get("target"), "target", where)
        negative = ""
    elif task == "classification":
        source = f"task: classification\ntext:\n{_require_string(row.get('input'), 'input', where)}\nlabel:"
        target = _require_string(row.get("label"), "label", where)
        negative = ""
    elif task == "translation":
        src_lang = _require_string(row.get("source_language"), "source_language", where)
        tgt_lang = _require_string(row.get("target_language"), "target_language", where)
        source = (f"task: translation\nsource_language: {src_lang}\n"
                  f"target_language: {tgt_lang}\nsource:\n"
                  f"{_require_string(row.get('source'), 'source', where)}\ntranslation:")
        target = _require_string(row.get("target"), "target", where)
        negative = ""
    elif task == "summarization":
        source = f"task: summarization\ndocument:\n{_require_string(row.get('input'), 'input', where)}\nsummary:"
        target = _require_string(row.get("target"), "target", where)
        negative = ""
    else:
        text = _require_string(row.get("anchor" if task == "embedding" else "query"),
                               "anchor" if task == "embedding" else "query", where)
        source = (f"task: embedding\ntext:\n{text}" if task == "embedding"
                  else f"task: reranking\nquery:\n{text}")
        target = _require_string(row.get("positive"), "positive", where)
        negative = _require_string(row.get("negative"), "negative", where)
        if target == negative:
            raise ValueError(f"{where}: contrastive positive and negative are identical")
    group = str(metadata.get("source_group") or row.get("source_group") or record_id)
    return Example(task, _split_for(row, group), source, target, negative, group, record_id)


def resolve_inputs(locations: Sequence[str]) -> list[Path]:
    files: set[Path] = set()
    for location in locations:
        path = Path(location).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".jsonl":
            files.add(path)
        elif path.is_dir():
            files.update(item.resolve() for item in path.rglob("*.jsonl") if item.is_file())
        else:
            raise FileNotFoundError(f"Expected an existing JSONL file or directory: {path}")
    if not files:
        raise ValueError("No JSONL records found in --data inputs")
    return sorted(files)


def iter_examples(files: Sequence[Path], split: str | None = None,
                  *, teacher_policy: str = "skip",
                  excluded: Counter[str] | None = None) -> Iterator[Example]:
    if teacher_policy not in {"skip", "error"}:
        raise ValueError(f"Unsupported teacher policy: {teacher_policy}")
    for path in files:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                where = f"{path}:{line_number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{where}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{where}: each line must contain a JSON object")
                reason = _unapproved_teacher_reason(row)
                if reason:
                    if teacher_policy == "error":
                        raise ValueError(f"{where}: teacher data is not approved ({reason})")
                    if excluded is not None:
                        excluded[f"{path}::{reason}"] += 1
                    continue
                result = normalize_record(row, where)
                if split is None or split == result.split:
                    yield result


def audit_data(files: Sequence[Path], require_all_tasks: bool,
               teacher_policy: str = "skip") -> dict[str, Any]:
    counts: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    groups: dict[str, str] = {}
    ids: set[str] = set()
    content_splits: dict[str, str] = {}
    manifest = hashlib.sha256()
    for item in iter_examples(files, teacher_policy=teacher_policy, excluded=excluded):
        # Only examples actually admitted to training affect the resume identity.
        # Pending teacher candidates may change without invalidating a checkpoint.
        manifest.update(json.dumps([item.task, item.split, item.source, item.target,
                                     item.negative, item.group, item.record_id],
                                    ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        manifest.update(b"\n")
        counts[f"{item.task}/{item.split}"] += 1
        if item.record_id in ids:
            raise ValueError(f"Duplicate record id: {item.record_id}")
        ids.add(item.record_id)
        previous = groups.setdefault(item.group, item.split)
        if previous != item.split:
            raise ValueError(f"Source-group leakage: {item.group} is in {previous} and {item.split}")
        content_hash = hashlib.sha256(
            json.dumps([item.task, item.source, item.target, item.negative],
                       ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        previous_split = content_splits.setdefault(content_hash, item.split)
        if previous_split != item.split:
            raise ValueError(f"Exact-content leakage across {previous_split} and {item.split}")
    if not any(key.endswith("/train") for key in counts):
        raise ValueError("No training records")
    if not any(key.endswith("/validation") for key in counts):
        raise ValueError("No validation records; curate a held-out set first")
    if require_all_tasks:
        missing = sorted(task for task in TASKS if counts[f"{task}/train"] == 0 or
                         counts[f"{task}/validation"] == 0)
        if missing:
            raise ValueError(f"Train/validation coverage missing for: {', '.join(missing)}")
    return {"fingerprint": manifest.hexdigest(), "counts": dict(sorted(counts.items())),
            "records": len(ids), "groups": len(groups), "files": [str(p) for p in files],
            "skipped_unapproved_teacher_records": sum(excluded.values()),
            "skipped_teacher_by_file_and_reason": dict(sorted(excluded.items()))}


PROVIDERS: dict[str, tuple[str, str, str, str]] = {
    "openai": ("OPENAI_API_KEY", "OPENAI_URL", "https://api.openai.com/v1/chat/completions", "api.openai.com"),
    "groq": ("GROQ_API_KEY", "GROQ_URL", "https://api.groq.com/openai/v1/chat/completions", "api.groq.com"),
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions", "openrouter.ai"),
    "gemini": ("GEMINI_API_KEY", "GEMINI_URL", "https://generativelanguage.googleapis.com/v1beta", "generativelanguage.googleapis.com"),
}
TEACHER_FIELDS: dict[str, tuple[str, ...]] = {
    "generation": ("input", "target"),
    "dialogue": ("history", "target"),
    "classification": ("input", "label"),
    "translation": ("source", "source_language", "target_language", "target"),
    "summarization": ("input", "target"),
    "embedding": ("anchor", "positive", "negative"),
    "reranking": ("query", "positive", "negative"),
}


def read_env_config(dotenv: Path) -> dict[str, str]:
    """Read only relevant credentials/URLs; never print or serialize secrets."""
    allowed = {name for item in PROVIDERS.values() for name in item[:2]}
    values: dict[str, str] = {}
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8-sig").splitlines():
            match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.*?)\s*$", line)
            if match and match.group(1) in allowed:
                raw = match.group(2)
                if raw.startswith(('"', "'")) and raw.endswith(raw[0]):
                    raw = raw[1:-1]
                values[match.group(1)] = raw
    values.update({name: os.environ[name] for name in allowed if os.environ.get(name)})
    return values


def teacher_endpoint(provider: str, model: str, values: Mapping[str, str]) -> str:
    from urllib.parse import quote, urlsplit

    _, url_name, default, host = PROVIDERS[provider]
    url = values.get(url_name) or default
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != host or parsed.username or parsed.password:
        raise ValueError(f"Untrusted {url_name} hostname or insecure scheme; refusing to send API key")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{url_name} must not contain query parameters or fragments")
    if provider == "gemini":
        if "/models/" in parsed.path:
            raise ValueError("GEMINI_URL must be an API base URL, not a model-specific endpoint")
        return url.rstrip("/") + "/models/" + quote(model, safe="-._") + ":generateContent"
    if parsed.path.endswith("/chat/completions"):
        return url
    return url.rstrip("/") + "/chat/completions"


def teacher_request(provider: str, model: str, prompt: str, key: str, endpoint: str,
                    output_tokens: int, timeout: int) -> tuple[str, Mapping[str, Any]]:
    """One non-retried request. A retry could incur duplicate API charges."""
    import requests

    if provider == "gemini":
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                   "generationConfig": {"maxOutputTokens": output_tokens,
                                        "responseMimeType": "application/json"}}
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    else:
        payload = {"model": model, "messages": [
            {"role": "system", "content": "Produce one original, valid JSON training example. No markdown."},
            {"role": "user", "content": prompt},
        ]}
        payload["max_completion_tokens" if provider == "openai" else "max_tokens"] = output_tokens
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise RuntimeError(f"{provider} returned HTTP {status}; no response body or credentials logged") from None
    except (requests.RequestException, ValueError):
        raise RuntimeError(f"{provider} request failed; details suppressed to protect credentials") from None
    try:
        if provider == "gemini":
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
        else:
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
    except (KeyError, IndexError, TypeError):
        raise ValueError(f"{provider} response lacks a text candidate") from None
    if not isinstance(content, str):
        raise ValueError(f"{provider} response was not text")
    return content.strip(), usage if isinstance(usage, Mapping) else {}


def _teacher_seeds(files: Sequence[Path], limit: int) -> list[tuple[Mapping[str, Any], Example]]:
    """Choose only held-in train seeds, balanced across task types."""
    pools: dict[str, list[tuple[Mapping[str, Any], Example]]] = defaultdict(list)
    for path in files:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or _unapproved_teacher_reason(row):
                    continue
                item = normalize_record(row, f"{path}:{line_number}")
                if item.split == "train" and not (row.get("metadata") or {}).get("teacher_provider"):
                    if len(pools[item.task]) < limit:
                        pools[item.task].append((row, item))
    selected: list[tuple[Mapping[str, Any], Example]] = []
    ordered_tasks = sorted(pools)
    for index in range(limit):
        task = ordered_tasks[index % len(ordered_tasks)]
        choices = pools[task]
        selected.append(choices[(index // len(ordered_tasks)) % len(choices)])
    return selected


def teacher_generate(args: argparse.Namespace, files: Sequence[Path]) -> None:
    if not args.teacher_model or not args.teacher_output:
        raise ValueError("Teacher mode requires --teacher-model and --teacher-output")
    if args.teacher_max_requests < 1 or args.teacher_max_requests > 1000:
        raise ValueError("--teacher-max-requests must be between 1 and 1000")
    if not args.teacher_consent:
        raise ValueError("Explicit --teacher-consent required to send training examples to an external API")
    if args.teacher_provider != "openrouter":
        raise ValueError("This provider's terms restrict use for training a separate generative LM. "
                         "Teacher-training mode is disabled; ordinary application inference is unaffected")
    if args.teacher_model.lower().startswith(("openai/", "google/", "anthropic/")):
        raise ValueError("The selected routed model has separate output-training restrictions")
    if not args.teacher_terms_reviewed or not args.teacher_license_reference:
        raise ValueError("OpenRouter model-specific output-training rights must be reviewed first; "
                         "set --teacher-terms-reviewed and --teacher-license-reference")
    values = read_env_config(Path(args.dotenv))
    key_name, _, _, _ = PROVIDERS[args.teacher_provider]
    key = values.get(key_name, "").strip()
    if not key:
        raise ValueError(f"No configured {key_name} found; API call cancelled")
    endpoint = teacher_endpoint(args.teacher_provider, args.teacher_model, values)
    seeds = _teacher_seeds(files, args.teacher_max_requests)
    output = Path(args.teacher_output).resolve()
    if output.suffix.lower() != ".jsonl":
        raise ValueError("Teacher output must be a .jsonl file")
    if any(output == source for source in files):
        raise ValueError("Teacher output must not overwrite the source dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if output.is_file():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing.add(str(json.loads(line).get("id", "")))
    completed = 0
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for row, source in seeds:
            request_id = "teacher-" + hashlib.sha256(
                f"{args.teacher_provider}|{args.teacher_model}|{source.record_id}".encode("utf-8")
            ).hexdigest()[:20]
            if request_id in existing:
                continue
            seed_fields = {field: row.get(field) for field in TEACHER_FIELDS[source.task]}
            seed_json = json.dumps(seed_fields, ensure_ascii=False)
            if len(seed_json) > args.teacher_max_chars:
                continue
            prompt = ("You are preparing a candidate for a human-reviewed training corpus. "
                      "Make ONE substantively different, factually coherent example for the SAME "
                      f"task '{source.task}'. Return only a JSON object containing EXACTLY these "
                      f"fields: {', '.join(TEACHER_FIELDS[source.task])}. "
                      "Avoid names, email addresses, secrets, personal data, false citations, and "
                      "claims of factual verification you cannot provide. For dialogue use a list of "
                      "role/content objects. For embedding/reranking make the negative meaningfully "
                      "wrong, the positive accurate. Keep translations semantically equivalent. "
                      "Do not reproduce the example verbatim.\n\nSeed example:\n" + seed_json)
            try:
                content, usage = teacher_request(
                    args.teacher_provider, args.teacher_model, prompt, key, endpoint,
                    args.teacher_output_tokens, args.teacher_timeout)
                content = content.strip()
                if content.startswith("```"):
                    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
                generated = json.loads(content)
                if not isinstance(generated, dict):
                    raise ValueError("Teacher response was not a JSON object")
                candidate = {field: generated.get(field) for field in TEACHER_FIELDS[source.task]}
                candidate.update({"task": source.task, "split": "train", "id": request_id,
                                  "metadata": {"teacher_provider": args.teacher_provider,
                                               "teacher_model": args.teacher_model,
                                               "license_review_reference": args.teacher_license_reference,
                                               "seed_record_id": source.record_id,
                                               "source_group": source.group,
                                               "review_status": "pending",
                                               "reviewed_by": None,
                                               "usage": {key: value for key, value in usage.items()
                                                         if key in {"prompt_tokens", "completion_tokens", "total_tokens",
                                                                    "promptTokenCount", "candidatesTokenCount", "totalTokenCount"}}}})
                # Temporarily mark as reviewed for schema validation only.
                inspect_row = dict(candidate)
                inspect_row["metadata"] = {**candidate["metadata"],
                                           "review_status": "approved", "reviewed_by": "schema-check"}
                normalized = normalize_record(inspect_row, request_id)
                if (normalized.source, normalized.target, normalized.negative) == \
                        (source.source, source.target, source.negative):
                    raise ValueError("Teacher repeated its seed exactly")
            except (ValueError, RuntimeError) as exc:
                print(f"Teacher candidate rejected ({source.task}): {type(exc).__name__}", file=sys.stderr)
                # Do not retry automatically; the API may still have charged.
                continue
            stream.write(json.dumps(candidate, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            existing.add(request_id)
            completed += 1
            print(json.dumps({"teacher_candidate": request_id, "task": source.task,
                              "review_status": "pending"}), flush=True)
    print(f"Saved {completed} unapproved teacher candidates to {output}; manual review required")


def estimate_parameters(vocab: int, arch: Mapping[str, int]) -> int:
    """Exact count for the standard untied nn.Transformer encoder-decoder stack."""
    d, f = arch["d_model"], arch["dim_feedforward"]
    e, z = arch["num_encoder_layers"], arch["num_decoder_layers"]
    encoder = 4 * d * d + 2 * d * f + 9 * d + f
    decoder = 8 * d * d + 2 * d * f + 15 * d + f
    return 3 * vocab * d + e * encoder + z * decoder + 4 * d


def checkpoint_manifest(torch: Any, path: Path, *, trust: bool = False) -> dict[str, Any]:
    """Read-only fingerprint and architecture inventory for an existing checkpoint."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    checkpoint = _load_checkpoint(torch, str(path), trust)
    state = checkpoint["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError(f"Checkpoint state_dict is not a mapping: {path}")
    tensors = {name: list(value.shape) for name, value in state.items()
               if isinstance(name, str) and isinstance(value, torch.Tensor)}
    base = checkpoint.get("base_config") or {}
    metadata = checkpoint.get("metadata") or {}
    if not isinstance(base, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError(f"Malformed checkpoint configuration/metadata: {path}")
    return {
        "path": str(path.resolve()), "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
        "architecture": {key: base.get(key) for key in ARCHITECTURE_KEYS},
        "parameter_elements": sum(math.prod(shape) for shape in tensors.values()),
        "tensor_shapes": tensors,
        "training_metadata": {key: metadata.get(key) for key in
                              ("model_id", "source_framework", "trainer_schema",
                               "audit_fingerprint", "tokenizer_fingerprint", "training", "transfer", "lstp")
                              if key in metadata},
    }


def compare_architectures(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Report rather than guess whether layer tensors can transfer exactly."""
    first = reference["tensor_shapes"]
    second = candidate["tensor_shapes"]
    matched = [name for name, shape in first.items() if second.get(name) == shape]
    total = sum(math.prod(shape) for shape in second.values())
    copied = sum(math.prod(second[name]) for name in matched)
    return {
        "reference": reference["path"], "candidate": candidate["path"],
        "architecture_differences": {
            key: {"reference": reference["architecture"].get(key),
                  "candidate": candidate["architecture"].get(key)}
            for key in ARCHITECTURE_KEYS
            if reference["architecture"].get(key) != candidate["architecture"].get(key)
        },
        "exact_shape_matched_parameters": copied,
        "candidate_parameters": total,
        "exact_shape_coverage": copied / total if total else 0.0,
        "exact_shape_matched_tensors": len(matched),
        "candidate_tensors": len(second),
        "note": "Shape compatibility is necessary but does not prove tokenizer or functional compatibility.",
    }


def inspect_checkpoints(args: argparse.Namespace) -> None:
    import torch  # type: ignore

    manifests = [checkpoint_manifest(torch, Path(path), trust=args.trust_checkpoint)
                 for path in args.inspect_checkpoints]
    comparisons = [compare_architectures(manifests[0], item)
                   for item in manifests[1:]]
    print(json.dumps({"checkpoints": manifests, "comparisons": comparisons}, indent=2))


def benchmark_checkpoints(args: argparse.Namespace) -> None:
    """Evaluate raw LANTRA outputs; application NLU/NLG fallback is intentionally absent."""
    from src.agents.language.lantra_runtime import LantraRuntime

    results: list[dict[str, Any]] = []
    for name in args.benchmark_checkpoints:
        checkpoint = Path(name)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        runtime = LantraRuntime({
            "checkpoint_path": str(checkpoint.resolve()),
            "device": args.device,
            "inference": {"generation_strategy": "greedy", "source_max_length": 256,
                          "target_max_length": args.benchmark_max_length,
                          "embedding_max_length": 128},
        })
        prompts = []
        for prompt in BENCHMARK_DIALOGUE_PROMPTS:
            started = time.perf_counter()
            response = runtime.dialogue([{"role": "user", "content": prompt}],
                                        max_length=args.benchmark_max_length)
            value = response.text.strip()
            prompts.append({"input": prompt, "output": value,
                            "empty": not value,
                            "degenerate": not value or len(value.split()) < 2,
                            "latency_ms": round((time.perf_counter() - started) * 1000, 2)})
        results.append({"checkpoint": str(checkpoint.resolve()),
                        "prompts": prompts,
                        "empty_count": sum(item["empty"] for item in prompts),
                        "degenerate_count": sum(item["degenerate"] for item in prompts),
                        "note": "Direct model benchmark: no template fallback. Short replies can be valid; inspect individual results."})
    print(json.dumps({"benchmark": results}, indent=2, ensure_ascii=False))


def artifact_vocab_size(path: str) -> int:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Expected a nonempty tokenizer vocabulary mapping")
    return len(payload)


def report_plan(args: argparse.Namespace, audit: Mapping[str, Any]) -> None:
    vocab = artifact_vocab_size(args.vocab_json)
    arch = ARCHITECTURES[args.architecture]
    params = estimate_parameters(vocab, arch)
    print(json.dumps({"model": "SLAILM_84M" if args.architecture == "84m" else "SLAILM_SMOKE",
                      "architecture": arch, "vocabulary_size": vocab,
                      "estimated_parameters": params, "target_parameters": TARGET_PARAMETERS,
                      "data": audit, "lstp_training": "disabled until contract-readiness evidence"},
                     indent=2))


def batches(items: Iterable[Example], batch_size: int) -> Iterator[list[Example]]:
    pending: list[Example] = []
    for item in items:
        pending.append(item)
        if len(pending) == batch_size:
            yield pending
            pending = []
    if pending:
        yield pending


def shuffle_buffered(items: Iterable[Example], seed: int, capacity: int) -> Iterator[Example]:
    """Deterministic bounded-memory shuffle, reproducible from an epoch cursor."""
    rng = random.Random(seed)
    buffer: list[Example] = []
    for item in items:
        if len(buffer) < capacity:
            buffer.append(item)
        else:
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = item
    while buffer:
        yield buffer.pop(rng.randrange(len(buffer)))


def learning_rate(base_lr: float, step: int, warmup: int, total: int) -> float:
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return base_lr * (0.1 + 0.9 * (1.0 + math.cos(math.pi * progress)) / 2.0)


def _data_batch_loss(torch: Any, model: Any, tokenizer: Any,
                     records: Sequence[Example], src_limit: int, tgt_limit: int,
                     device: Any, contrastive_temperature: float) -> Any:
    import torch.nn.functional as F

    pad = int(tokenizer.pad_token_id)

    def ids(text: str, limit: int) -> list[int]:
        result = tokenizer.encode(text, add_special_tokens=True, truncation=True,
                                  max_length=limit, padding=False,
                                  return_attention_mask=False, return_tokens=False,
                                  return_special_tokens_mask=False,
                                  return_offsets_mapping=False,
                                  return_token_metadata=False, return_tensors=None)
        return list(result["input_ids"])

    def tensor(texts: Sequence[str], limit: int) -> Any:
        sequences = [ids(text, limit) for text in texts]
        length = max(len(tokens) for tokens in sequences)
        return torch.tensor([tokens + [pad] * (length - len(tokens)) for tokens in sequences],
                            dtype=torch.long, device=device)

    by_task: dict[str, list[Example]] = defaultdict(list)
    for record in records:
        by_task[record.task].append(record)
    weighted_loss = None
    for task, subset in by_task.items():
        source = tensor([item.source for item in subset], src_limit)
        padding_mask = source.eq(pad)
        if task in PAIR_TASKS:
            prefix = "task: embedding\ntext:\n" if task == "embedding" else "task: reranking\ndocument:\n"
            positive = tensor([prefix + item.target for item in subset], src_limit)
            negative = tensor([prefix + item.negative for item in subset], src_limit)

            def pooled(encoded: Any) -> Any:
                valid = encoded.ne(pad)
                representation = model.encode(encoded, src_key_padding_mask=~valid)
                pooled_vectors = (representation * valid.unsqueeze(-1)).sum(1) / valid.sum(1).clamp_min(1).unsqueeze(-1)
                return F.normalize(pooled_vectors, dim=-1)

            anchor = pooled(source)
            pos = pooled(positive)
            neg = pooled(negative)
            scores = torch.stack(((anchor * pos).sum(-1), (anchor * neg).sum(-1)), dim=-1)
            objective = F.cross_entropy(scores / contrastive_temperature,
                                        torch.zeros(len(subset), device=device, dtype=torch.long))
        else:
            target = tensor([item.target for item in subset], tgt_limit)
            decoder_input = target[:, :-1]
            labels = target[:, 1:]
            causal = model.generate_square_subsequent_mask(decoder_input.shape[1], device=device).isneginf()
            logits = model(source, decoder_input, tgt_mask=causal,
                           src_key_padding_mask=padding_mask,
                           memory_key_padding_mask=padding_mask,
                           tgt_key_padding_mask=decoder_input.eq(pad))
            if isinstance(logits, dict):
                logits = logits["logits"]
            objective = model.compute_loss(logits, labels, ignore_index=pad)
        value = objective * (len(subset) / len(records))
        weighted_loss = value if weighted_loss is None else weighted_loss + value
    return weighted_loss


def evaluate(torch: Any, model: Any, tokenizer: Any, files: Sequence[Path],
             args: argparse.Namespace, device: Any, split: str = "validation") -> dict[str, Any]:
    model.eval()
    total, count = 0.0, 0
    by_task_total: Counter[str] = Counter()
    by_task_count: Counter[str] = Counter()
    task_batches_seen: Counter[str] = Counter()
    with torch.no_grad():
        for batch in batches(iter_examples(files, split), args.batch_size):
            grouped: dict[str, list[Example]] = defaultdict(list)
            for item in batch:
                grouped[item.task].append(item)
            for task, subset in grouped.items():
                # Input files are often sorted by task. Limiting the first N
                # batches globally silently drops later tasks from validation.
                if args.validation_batches and task_batches_seen[task] >= args.validation_batches:
                    continue
                task_batches_seen[task] += 1
                objective = _data_batch_loss(torch, model, tokenizer, subset,
                                             args.source_length, args.target_length,
                                             device, args.contrastive_temperature)
                weighted = float(objective.detach().cpu().item()) * len(subset)
                total += weighted
                count += len(subset)
                by_task_total[task] += weighted
                by_task_count[task] += len(subset)
    model.train()
    if count == 0:
        raise ValueError(f"No examples for {split}")
    return {"aggregate_objective": total / count,
            "tasks": {task: {"objective": by_task_total[task] / number,
                              "examples": number,
                              "objective_type": "contrastive" if task in PAIR_TASKS else "token_cross_entropy"}
                      for task, number in sorted(by_task_count.items())},
            "note": "Cross-entropy and contrastive objectives have different scales. "
                    "Select best checkpoints using a single task-specific objective."}


def selection_objective(metrics: Mapping[str, Any], task: str) -> float:
    """Avoid comparing an arbitrary mixture of contrastive and token losses."""
    tasks = metrics.get("tasks") or {}
    if task not in tasks:
        raise ValueError(f"Selected validation task {task!r} has no examples")
    value = float(tasks[task]["objective"])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {task} validation objective")
    return value


def _load_checkpoint(torch: Any, path: str, trust: bool) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=not trust)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Incompatible checkpoint structure: {path}")
    return checkpoint


def transfer_weights(model: Any, checkpoint: Mapping[str, Any]) -> dict[str, int]:
    """Widen embeddings/FFN safely and preserve separate attention Q/K/V rows.

    A widening warm start is *not* function-preserving; it requires training and
    evaluation. Checkpoint optimizer state is deliberately never transferred.
    """
    import torch

    donor = checkpoint["state_dict"]
    current = model.state_dict()
    copied, tensors, omitted = 0, 0, 0
    with torch.no_grad():
        for name, target in current.items():
            source = donor.get(name)
            if not isinstance(source, torch.Tensor) or source.ndim != target.ndim:
                omitted += target.numel()
                continue
            if source.shape == target.shape:
                target.copy_(source)
                copied += source.numel()
                tensors += 1
                continue
            if any(old > new for old, new in zip(source.shape, target.shape)):
                omitted += target.numel()
                continue
            if name.endswith("in_proj_weight") and source.shape[0] % 3 == 0:
                source_width = source.shape[0] // 3
                target_width = target.shape[0] // 3
                for index in range(3):
                    target[index * target_width:index * target_width + source_width,
                           :source.shape[1]].copy_(source[index * source_width:(index + 1) * source_width])
            elif name.endswith("in_proj_bias") and source.numel() % 3 == 0:
                source_width = source.numel() // 3
                target_width = target.numel() // 3
                for index in range(3):
                    target[index * target_width:index * target_width + source_width].copy_(
                        source[index * source_width:(index + 1) * source_width])
            else:
                slices = tuple(slice(0, n) for n in source.shape)
                target[slices].copy_(source)
            copied += source.numel()
            omitted += target.numel() - source.numel()
            tensors += 1
    model.load_state_dict(current, strict=True)
    if copied == 0:
        raise ValueError("Initialization checkpoint has no transferable model parameters")
    return {"copied_elements": copied, "newly_initialized_elements": omitted,
            "tensors_reused": tensors}


def _checkpoint_metadata(torch: Any, args: argparse.Namespace, audit: Mapping[str, Any],
                         epoch: int, seen: int, step: int, best: float,
                         transfer: Mapping[str, int] | None, tokenizer_fingerprint: str) -> dict[str, Any]:
    return {
        "model_id": "SLAILM_84M" if args.architecture == "84m" else "SLAILM_SMOKE",
        "source_framework": "SLAI-v2.4/LanguageTransformer",
        "trainer_schema": "slailm-trainer-v1",
        "audit_fingerprint": audit["fingerprint"],
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "training": {"epoch": epoch, "records_seen_in_epoch": seen, "global_step": step,
                     "best_validation_loss": best, "selection_task": args.selection_task,
                     "configuration": _resume_config(args)},
        "transfer": dict(transfer or {}),
        "lstp": "disabled: readiness gates not independently verified",
        "rng_state": {"python": random.getstate(), "torch": torch.get_rng_state(),
                      "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    }


def _resume_config(args: argparse.Namespace) -> dict[str, Any]:
    return {key: getattr(args, key) for key in (
        "architecture", "batch_size", "accumulate", "epochs", "max_steps",
        "source_length", "target_length", "lr", "weight_decay", "warmup_steps",
        "contrastive_temperature", "seed", "precision", "shuffle_buffer", "selection_task",
    )}


def run_training(args: argparse.Namespace, files: Sequence[Path], audit: Mapping[str, Any]) -> None:
    try:
        import torch
        from src.agents.language.modules.language_tokenizer import LanguageTokenizer
        from src.agents.language.modules.language_transformer import LanguageTransformer
    except ImportError as exc:
        raise RuntimeError("Training needs SLAI's neural dependencies; install torch and the project dependencies") from exc

    if args.resume and args.init_from:
        raise ValueError("--resume and --init-from are mutually exclusive")
    if audit["counts"].get(f"{args.selection_task}/validation", 0) == 0:
        raise ValueError(f"No held-out validation records for --selection-task {args.selection_task}")
    if args.precision in {"bf16", "fp16"} and not torch.cuda.is_available():
        raise ValueError("Mixed precision requires CUDA; choose --precision fp32 for CPU")
    device = torch.device(args.device if args.device != "auto" else
                          "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device is unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = LanguageTokenizer()
    if not tokenizer.is_trained:
        raise ValueError("SLAI LanguageTokenizer does not have trained BPE artifacts")
    vocab = tokenizer.get_vocab_size()
    actual_artifact_size = artifact_vocab_size(args.vocab_json)
    if vocab < actual_artifact_size or vocab - actual_artifact_size > 32:
        raise ValueError(f"Tokenizer vocab={vocab} differs unexpectedly from artifact={actual_artifact_size}")
    tokenizer_fingerprint = hashlib.sha256(json.dumps(
        tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False,
        separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    arch = {**ARCHITECTURES[args.architecture], "src_vocab_size": vocab, "tgt_vocab_size": vocab,
            "pad_token_id": tokenizer.pad_token_id, "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id, "tie_embeddings": False,
            "tie_output_projection": False, "batch_first": True,
            # Maintain LANTRA runtime's default source/target context contract.
            "max_len": max(512, args.source_length, args.target_length),
            "max_position_embeddings": max(512, args.source_length, args.target_length),
            "lr": args.lr, "weight_decay": args.weight_decay,
            "record_history": False, "enable_memory_integration": False}
    model = LanguageTransformer(**arch)
    parameters = sum(p.numel() for p in model.parameters())
    if args.architecture == "84m" and abs(parameters - TARGET_PARAMETERS) > TARGET_PARAMETERS * 0.05:
        raise ValueError(f"Expected ~84M parameters; actual: {parameters:,}")
    mismatches = {key: (value, getattr(model.config, key)) for key, value in arch.items()
                  if hasattr(model.config, key) and getattr(model.config, key) != value}
    if mismatches:
        raise ValueError(f"LanguageTransformer did not honor architecture overrides: {mismatches}")

    transfer: dict[str, int] | None = None
    start_epoch, start_seen, step, best = 0, 0, 0, float("inf")
    checkpoint = None
    if args.resume:
        checkpoint = _load_checkpoint(torch, args.resume, args.trust_checkpoint)
        saved = checkpoint.get("metadata", {}).get("training", {})
        saved_config = dict(saved.get("configuration") or {})
        legacy_objective = "selection_task" not in saved_config
        saved_config.setdefault("selection_task", "dialogue")
        if saved_config != _resume_config(args):
            raise ValueError("Resume configuration differs from the saved training configuration")
        if checkpoint.get("metadata", {}).get("audit_fingerprint") != audit["fingerprint"]:
            raise ValueError("Dataset fingerprint changed since checkpoint; refusing unsafe resume")
        if checkpoint.get("metadata", {}).get("tokenizer_fingerprint") != tokenizer_fingerprint:
            raise ValueError("Tokenizer vocabulary mapping changed since checkpoint")
        if checkpoint.get("base_config") != model.config.to_dict():
            raise ValueError("Checkpoint architecture differs; use --init-from to warm start instead")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        start_epoch, start_seen, step = (int(saved[key]) for key in
                                          ("epoch", "records_seen_in_epoch", "global_step"))
        # Legacy trainers selected against mixed CE + contrastive loss. That
        # number is not comparable with task-specific dialogue loss.
        best = float("inf") if legacy_objective else float(saved.get("best_validation_loss", best))
    elif args.init_from:
        donor = _load_checkpoint(torch, args.init_from, args.trust_checkpoint)
        if donor.get("base_config", {}).get("src_vocab_size") != vocab or \
                donor.get("base_config", {}).get("tgt_vocab_size") != vocab:
            raise ValueError("Tokenizer vocabulary dimensions differ; unsafe warm start")
        donor_fingerprint = donor.get("metadata", {}).get("tokenizer_fingerprint")
        if donor_fingerprint != tokenizer_fingerprint and not (
            donor_fingerprint is None and args.allow_unverified_tokenizer
        ):
            raise ValueError("Donor tokenizer mapping is unverified or different; supply matching artifacts"
                             " (legacy checkpoints require --allow-unverified-tokenizer)")
        if args.init_strategy == "strict":
            current = model.state_dict()
            candidate = donor["state_dict"]
            donor_config = donor.get("base_config") or {}
            target_config = model.config.to_dict()
            differences = {key: {"donor": donor_config.get(key), "target": target_config.get(key)}
                           for key in ARCHITECTURE_KEYS
                           if donor_config.get(key) != target_config.get(key)}
            incompatible = {name: {"expected": tuple(value.shape),
                                   "donor": tuple(candidate[name].shape) if name in candidate else None}
                            for name, value in current.items()
                            if name not in candidate or value.shape != candidate[name].shape}
            unexpected = sorted(set(candidate) - set(current))
            if differences or incompatible or unexpected:
                raise ValueError(
                    "Initialization checkpoint architecture is incompatible: "
                    f"{len(incompatible)} mismatched/missing tensors, {len(unexpected)} extra tensors. "
                    f"Config differences: {differences}. Tensor examples: {list(incompatible.items())[:4]}. "
                    "Use teacher-student distillation for 512→384 dimensions, "
                    "or --init-strategy partial only for explicitly reviewed exploratory warm starts."
                )
            model.load_state_dict(candidate, strict=True)
            transfer = {"copied_elements": sum(value.numel() for value in current.values()),
                        "newly_initialized_elements": 0, "tensors_reused": len(current)}
        else:
            transfer = transfer_weights(model, donor)
        print(f"Warm start: {transfer}")

    model.to(device)
    optimizer = model.configure_optimizer()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.precision == "fp16"))
    if checkpoint is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("Resume checkpoint lacks optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state = checkpoint.get("metadata", {}).get("scaler_state")
        if scaler.is_enabled():
            if not isinstance(scaler_state, dict):
                raise ValueError("FP16 resume checkpoint lacks GradScaler state")
            scaler.load_state_dict(scaler_state)
        rng = checkpoint.get("metadata", {}).get("rng_state")
        if not isinstance(rng, dict):
            raise ValueError("Resume checkpoint lacks RNG state")
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda"):
            torch.cuda.set_rng_state_all(rng["cuda"])
    if step >= args.max_steps:
        print(f"Training already reached max_steps={args.max_steps}; checkpoint restored successfully")
        return

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = "slailm_84m" if args.architecture == "84m" else "slailm_smoke"
    train_records = sum(number for task_split, number in audit["counts"].items()
                        if task_split.endswith("/train"))
    print(f"SLAILM trainer: {parameters:,} parameters | device={device} | "
          f"train={train_records} records | accepted_total={audit['records']}")

    def save(name: str, epoch: int, seen: int, current_step: int) -> None:
        meta = _checkpoint_metadata(torch, args, audit, epoch, seen, current_step, best,
                                    transfer, tokenizer_fingerprint)
        meta["scaler_state"] = scaler.state_dict()
        model.save_language_model(output / name, optimizer=optimizer, lang_metadata=meta)

    model.train()
    final_epoch, final_seen = start_epoch, start_seen
    for epoch in range(start_epoch, args.epochs):
        seen = 0
        accumulated = 0
        optimizer.zero_grad(set_to_none=True)
        examples = shuffle_buffered(iter_examples(files, "train"), args.seed + epoch,
                                    args.shuffle_buffer)
        for batch in batches(examples, args.batch_size):
            next_seen = seen + len(batch)
            if next_seen <= start_seen and epoch == start_epoch:
                seen = next_seen
                continue
            if seen < start_seen and epoch == start_epoch:
                raise ValueError("Resume cursor is not aligned with a minibatch")
            seen = next_seen
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(args.lr, step, args.warmup_steps, args.max_steps)
            use_amp = device.type == "cuda" and args.precision in {"fp16", "bf16"}
            # PyTorch autocast is used only for CUDA. FP16 requires GradScaler;
            # choose BF16 by default on compatible GPUs or FP32 everywhere.
            with torch.autocast(device_type="cuda", enabled=use_amp,
                                dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16):
                loss = _data_batch_loss(torch, model, tokenizer, batch,
                                        args.source_length, args.target_length,
                                        device, args.contrastive_temperature)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss; aborting before optimizer update")
                loss = loss / args.accumulate
            scaler.scale(loss).backward()
            accumulated += 1
            if accumulated < args.accumulate:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad,
                                           error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            step += 1
            print(json.dumps({"step": step, "epoch": epoch + 1,
                              "train_loss_scaled": float(loss.detach().cpu()) * args.accumulate,
                              "lr": optimizer.param_groups[0]["lr"]}), flush=True)
            if step % args.eval_every == 0:
                metrics = evaluate(torch, model, tokenizer, files, args, device)
                val = selection_objective(metrics, args.selection_task)
                print(json.dumps({"step": step, "validation": metrics}), flush=True)
                if val < best:
                    best = val
                    save(f"{prefix}_best.pt", epoch, seen, step)
            if step % args.checkpoint_every == 0:
                save(f"{prefix}_latest.pt", epoch, seen, step)
            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            final_epoch, final_seen = epoch, seen
            save(f"{prefix}_latest.pt", epoch, seen, step)
            break
        if accumulated:
            # Last partial gradient accumulation of an epoch.
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(args.accumulate / accumulated)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad,
                                           error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
        start_seen = 0
        final_epoch, final_seen = epoch + 1, 0
        save(f"{prefix}_latest.pt", epoch + 1, 0, step)

    final_metrics = evaluate(torch, model, tokenizer, files, args, device)
    final_loss = selection_objective(final_metrics, args.selection_task)
    if final_loss < best:
        best = final_loss
        save(f"{prefix}_best.pt", final_epoch, final_seen, step)
    save(f"{prefix}_final.pt", final_epoch, final_seen, step)
    print(json.dumps({"status": "training_complete", "global_step": step,
                      "selection_task": args.selection_task,
                      "final_validation_loss": final_loss, "best_validation_loss": best,
                      "validation_details": final_metrics,
                      "output_dir": str(output.resolve())}, indent=2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SLAI-native SLAILM_84M trainer")
    parser.add_argument("--inspect-checkpoints", nargs="+", metavar="PT",
                        help="Read-only SHA-256, architecture and parameter-shape audit")
    parser.add_argument("--benchmark-checkpoints", nargs="+", metavar="PT",
                        help="Compare raw LANTRA dialogue outputs without application fallbacks")
    parser.add_argument("--benchmark-max-length", type=int, default=64)
    parser.add_argument("--data", nargs="+", default=[DEFAULT_DATA], help="Curated LANTRA JSONL files/directories")
    parser.add_argument("--teacher-policy", choices=("skip", "error"), default="skip",
                        help="Exclude unapproved teacher examples with counts, or stop strictly")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and estimate model size without torch")
    parser.add_argument("--teacher-generate", action="store_true",
                        help="Create review-pending teacher examples; never train on them automatically")
    parser.add_argument("--teacher-provider", choices=tuple(PROVIDERS), default="openai")
    parser.add_argument("--teacher-model", help="Explicit provider model identifier (no hardcoded default)")
    parser.add_argument("--teacher-output", help="Separate JSONL output for unapproved candidate examples")
    parser.add_argument("--dotenv", default=".env", help="Local credentials file; never copied into artifacts")
    parser.add_argument("--teacher-consent", action="store_true",
                        help="Confirm selected train examples may be sent to the chosen external service")
    parser.add_argument("--teacher-terms-reviewed", action="store_true",
                        help="Confirm the selected routed model permits training on its outputs")
    parser.add_argument("--teacher-license-reference",
                        help="Evidence URL/reference for model-specific output training rights")
    parser.add_argument("--teacher-max-requests", type=int, default=0,
                        help="Maximum billable API requests (default zero)")
    parser.add_argument("--teacher-output-tokens", type=int, default=512)
    parser.add_argument("--teacher-max-chars", type=int, default=4000)
    parser.add_argument("--teacher-timeout", type=int, default=45)
    parser.add_argument("--require-all-supervised-tasks", action="store_true")
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), default="84m")
    parser.add_argument("--vocab-json", default="data/bpe_200d_50k_vocab.json")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--init-from", help="LANTRA checkpoint: explicitly warm-start compatible/expanded weights")
    parser.add_argument("--init-strategy", choices=("strict", "partial"), default="strict",
                        help="Strict exact tensor shapes by default; partial warm starts are experimental")
    parser.add_argument("--allow-unverified-tokenizer", action="store_true",
                        help="Explicitly accept a legacy donor lacking tokenizer mapping hash")
    parser.add_argument("--resume", help="Exact trainer checkpoint, with optimizer/RNG/cursor")
    parser.add_argument("--trust-checkpoint", action="store_true", help="Allow legacy pickle checkpoint loading (only trusted files)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--shuffle-buffer", type=int, default=2048)
    parser.add_argument("--source-length", type=int, default=256)
    parser.add_argument("--target-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--validation-batches", type=int, default=0,
                        help="Limit validation batches per task; 0 evaluates all records")
    parser.add_argument("--selection-task", choices=tuple(sorted(TASKS)), default="dialogue",
                        help="Task-specific objective for best checkpoint selection")
    args = parser.parse_args(argv)
    for field in ("epochs", "max_steps", "batch_size", "accumulate", "shuffle_buffer", "source_length", "target_length",
                  "eval_every", "checkpoint_every"):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.target_length < 2:
        parser.error("--target-length must be at least 2")
    if args.lr <= 0 or args.contrastive_temperature <= 0 or args.clip_grad <= 0:
        parser.error("learning rate, contrastive temperature and clip grad must be positive")
    if args.warmup_steps < 0 or args.validation_batches < 0:
        parser.error("warmup steps and validation batches cannot be negative")
    if args.teacher_output_tokens < 64 or args.teacher_max_chars < 256 or args.teacher_timeout < 1:
        parser.error("Invalid teacher token, character, or timeout limits")
    if args.benchmark_max_length < 2:
        parser.error("--benchmark-max-length must be at least 2")
    if args.inspect_checkpoints and args.benchmark_checkpoints:
        parser.error("Choose --inspect-checkpoints or --benchmark-checkpoints, not both")
    if args.architecture == "smoke" and args.output_dir == DEFAULT_OUTPUT:
        args.output_dir = "src/agents/language/checkpoints/slailm_smoke"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.inspect_checkpoints:
            inspect_checkpoints(args)
            return 0
        if args.benchmark_checkpoints:
            benchmark_checkpoints(args)
            return 0
        files = resolve_inputs(args.data)
        audit = audit_data(files, args.require_all_supervised_tasks, args.teacher_policy)
        skipped = audit["skipped_unapproved_teacher_records"]
        if skipped:
            print(f"SLAILM trainer: excluded {skipped} unapproved teacher-generated record(s); "
                  "review/license evidence is required before inclusion", file=sys.stderr)
        if args.dry_run:
            report_plan(args, audit)
        elif args.teacher_generate:
            teacher_generate(args, files)
        else:
            run_training(args, files, audit)
    except (OSError, ValueError, RuntimeError, FloatingPointError) as exc:
        print(f"SLAILM trainer error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
