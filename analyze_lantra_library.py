from __future__ import annotations

"""Size LANTRA from the actual text in ``data/library``.

Run from the SLAI repository root:

    py analyze_lantra_library.py

The script reuses ``train_lantra.py`` for document extraction, uses SLAI's BPE
LanguageTokenizer when available, and recommends the nearest balanced
encoder-decoder LANTRA profile from the amount of unique training text.

Files are analyzed from smallest to largest. Scan progress is checkpointed after
each fully processed file, allowing a restart to resume from the last committed file.

It does not modify SLAI configuration. It writes a JSON report and a suggested
YAML snippet under:

    src/agents/language/artifacts/training/lantra/library_analysis/
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from src.training.lantra_vocabulary_analysis import VocabularyTracker, load_english_dictionary


LIBRARY = Path("data/library")
OUTPUT_DIR = Path("src/agents/language/artifacts/training/lantra/library_analysis")
DEFAULT_TOKENS_PER_PARAMETER = 20.0
DEFAULT_CONTEXT = 512
TOKENIZER_CHUNK_CHARS = 50_000
CHECKPOINT_FILENAME = "lantra_library_scan_checkpoint.json"
CHECKPOINT_SCHEMA = "slai.lantra.library-scan-checkpoint.v2"

# name, d_model, nhead, encoder layers, decoder layers
PROFILE_LADDER = (
    ("lantra_7m", 128, 4, 2, 2),
    ("lantra_13m", 192, 3, 3, 3),
    ("lantra_20m", 256, 8, 4, 4),
    ("lantra_44m", 384, 6, 6, 6),
    ("lantra_84m", 512, 8, 8, 8),
    ("lantra_124m", 640, 10, 8, 8),
    ("lantra_204m", 768, 12, 10, 10),
    ("lantra_315m", 896, 14, 12, 12),
    ("lantra_404m", 1024, 16, 12, 12),
    ("lantra_707m", 1280, 20, 14, 14),
    ("lantra_1b", 1536, 24, 14, 14),
    ("lantra_1_5b", 1792, 28, 16, 16),
    ("lantra_2b", 2048, 32, 16, 16),
    ("lantra_2_5b", 2304, 36, 16, 16),
    ("lantra_3b", 2560, 40, 16, 16),
)

COMPARE_SIZES = (
    ("45M", 45_000_000),
    ("150M", 150_000_000),
    ("350M", 350_000_000),
    ("1B", 1_000_000_000),
    ("2B", 2_000_000_000),
    ("3B", 3_000_000_000),
)


def human_number(value: float | int) -> str:
    value = float(value)
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def text_chunks(text: str, max_chars: int = TOKENIZER_CHUNK_CHARS) -> Iterator[str]:
    """Split on whitespace without overlap so token counts are not duplicated."""
    text = text.strip()
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        start = max(end, start + 1)


def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+(?:['’]\w+)?\b", text, flags=re.UNICODE))


def load_lantra_extractor():
    try:
        import train_lantra  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Could not import train_lantra.py. Run this script from the SLAI root. "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    extractor = getattr(train_lantra, "extract_raw_documents", None)
    extensions = getattr(train_lantra, "RAW_TEXT_EXTENSIONS", None)
    if not callable(extractor) or not extensions:
        raise RuntimeError(
            "train_lantra.py is missing extract_raw_documents or RAW_TEXT_EXTENSIONS."
        )
    return extractor, frozenset(str(item).lower() for item in extensions)


def load_tokenizer():
    try:
        from src.agents.language.modules.language_tokenizer import LanguageTokenizer

        tokenizer = LanguageTokenizer()
        if not getattr(tokenizer, "is_trained", False):
            return None, "Configured BPE resources were not loaded; using characters/4 estimate."
        return tokenizer, None
    except Exception as exc:
        return None, f"Tokenizer unavailable ({type(exc).__name__}: {exc}); using characters/4 estimate."


def count_bpe_tokens(tokenizer: Any, text: str) -> int:
    total = 0
    for chunk in text_chunks(text):
        encoded = tokenizer.encode(
            chunk,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_tensors=None,
            return_attention_mask=False,
            return_special_tokens_mask=False,
            return_offsets_mapping=False,
            return_token_metadata=False,
            return_tokens=False,
        )
        ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes, bytearray)):
            raise RuntimeError("LanguageTokenizer returned invalid input_ids")
        total += len(ids)
    return total


def estimate_parameters(
    vocab: int,
    d_model: int,
    enc_layers: int,
    dec_layers: int,
    ffn: int,
) -> int:
    """Parameter count for the current tied-weight PyTorch Transformer layout."""
    # One shared source/target/output vocabulary matrix.
    embeddings = vocab * d_model

    # nn.TransformerEncoderLayer:
    # self-attn + FFN + biases + two layer norms.
    encoder_layer = (
        4 * d_model * d_model
        + 2 * d_model * ffn
        + ffn
        + 9 * d_model
    )

    # nn.TransformerDecoderLayer:
    # self-attn + cross-attn + FFN + biases + three layer norms.
    decoder_layer = (
        8 * d_model * d_model
        + 2 * d_model * ffn
        + ffn
        + 15 * d_model
    )

    # Final encoder and decoder LayerNorm modules.
    final_norms = 4 * d_model

    return int(
        embeddings
        + enc_layers * encoder_layer
        + dec_layers * decoder_layer
        + final_norms
    )


def build_profiles(vocab: int) -> list[dict[str, Any]]:
    profiles = []
    for name, d_model, nhead, enc, dec in PROFILE_LADDER:
        ffn = d_model * 4
        profiles.append(
            {
                "name": name,
                "d_model": d_model,
                "nhead": nhead,
                "encoder_layers": enc,
                "decoder_layers": dec,
                "dim_feedforward": ffn,
                "vocab_size": vocab,
                "estimated_parameters": estimate_parameters(vocab, d_model, enc, dec, ffn),
            }
        )
    return profiles


def nearest_profile(profiles: Sequence[Mapping[str, Any]], target: float) -> dict[str, Any]:
    target = max(1.0, target)
    return dict(
        min(
            profiles,
            key=lambda p: abs(math.log(float(p["estimated_parameters"]) / target)),
        )
    )


def suggested_config(profile: Mapping[str, Any], context: int) -> dict[str, Any]:
    return {
        "language_transformer": {
            "task_type": "generation",
            "dropout": 0.1,
            "base_overrides": {
                "src_vocab_size": profile["vocab_size"],
                "tgt_vocab_size": profile["vocab_size"],
                "d_model": profile["d_model"],
                "nhead": profile["nhead"],
                "num_encoder_layers": profile["encoder_layers"],
                "num_decoder_layers": profile["decoder_layers"],
                "dim_feedforward": profile["dim_feedforward"],
                "activation": "gelu",
                "dropout": 0.1,
                "max_position_embeddings": context,
                "batch_first": True,
                "norm_first": True,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "tie_embeddings": True,
                "tie_output_projection": True,
            },
        }
    }


def simple_yaml(data: Mapping[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    pad = " " * indent
    for key, value in data.items():
        if isinstance(value, Mapping):
            lines.append(f"{pad}{key}:")
            lines.append(simple_yaml(value, indent + 2))
        elif isinstance(value, bool):
            lines.append(f"{pad}{key}: {'true' if value else 'false'}")
        else:
            lines.append(f"{pad}{key}: {value}")
    return "\n".join(lines)


def _path_key(path: Path) -> str:
    """Return a deterministic normalized path key for manifests and progress logs."""
    return str(path).replace("\\", "/").casefold()


def discover_library_files(library: Path, extensions: frozenset[str]) -> list[Path]:
    """Discover supported files and order them from smallest to largest."""
    candidates = [
        path
        for path in library.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    # Size is the primary key as requested. Path is only a deterministic tie-breaker.
    return sorted(candidates, key=lambda p: (p.stat().st_size, _path_key(p)))


def build_manifest(library: Path, files: Sequence[Path]) -> tuple[list[dict[str, Any]], str]:
    """Build a lightweight scan manifest used to validate resume checkpoints."""
    entries: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        try:
            rel = path.relative_to(library)
        except ValueError:
            rel = path
        entry = {
            "path": str(rel).replace("\\", "/"),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        entries.append(entry)
        digest.update(json.dumps(entry, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return entries, digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace a JSON file so a shutdown cannot leave partial JSON behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _archive_stale_checkpoint(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = path.with_name(f"{path.stem}.stale-{stamp}{path.suffix}")
    counter = 1
    while archived.exists():
        archived = path.with_name(f"{path.stem}.stale-{stamp}-{counter}{path.suffix}")
        counter += 1
    os.replace(path, archived)
    return archived


def _new_scan_state(
    *,
    library: Path,
    manifest_sha256: str,
    files_discovered: int,
    estimate_tokens: bool,
    vocab: int,
    method: str,
    tokenizer_note: str | None,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "library": str(library.resolve()),
        "manifest_sha256": manifest_sha256,
        "files_discovered": files_discovered,
        "estimate_tokens": bool(estimate_tokens),
        "vocab_size": int(vocab),
        "token_count_method": method,
        "tokenizer_note": tokenizer_note,
        "next_file_index": 0,
        "totals": {"files_discovered": files_discovered},
        "seen_file_hashes": [],
        "seen_document_hashes": [],
        "by_type": {},
        "failures": [],
        "elapsed_seconds": 0.0,
        "last_completed_path": None,
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def _checkpoint_matches(
    state: Mapping[str, Any],
    *,
    library: Path,
    manifest_sha256: str,
    estimate_tokens: bool,
    vocab: int,
    method: str,
) -> bool:
    return (
        state.get("schema") == CHECKPOINT_SCHEMA
        and state.get("library") == str(library.resolve())
        and state.get("manifest_sha256") == manifest_sha256
        and bool(state.get("estimate_tokens")) == bool(estimate_tokens)
        and int(state.get("vocab_size", -1)) == int(vocab)
        and state.get("token_count_method") == method
    )


def _load_or_create_scan_state(
    checkpoint_path: Path,
    *,
    library: Path,
    manifest_sha256: str,
    files_discovered: int,
    estimate_tokens: bool,
    vocab: int,
    method: str,
    tokenizer_note: str | None,
) -> tuple[dict[str, Any], bool]:
    if checkpoint_path.exists():
        try:
            state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except Exception as exc:
            archived = _archive_stale_checkpoint(checkpoint_path)
            print(
                f"[CHECKPOINT] Unreadable checkpoint archived to {archived}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        else:
            if _checkpoint_matches(
                state,
                library=library,
                manifest_sha256=manifest_sha256,
                estimate_tokens=estimate_tokens,
                vocab=vocab,
                method=method,
            ):
                next_index = int(state.get("next_file_index", 0))
                if 0 <= next_index <= files_discovered:
                    return dict(state), True
            archived = _archive_stale_checkpoint(checkpoint_path)
            print(
                "[CHECKPOINT] Existing checkpoint does not match the current library/tokenizer "
                f"state and was archived to {archived}. Starting a clean scan.",
                file=sys.stderr,
                flush=True,
            )

    state = _new_scan_state(
        library=library,
        manifest_sha256=manifest_sha256,
        files_discovered=files_discovered,
        estimate_tokens=estimate_tokens,
        vocab=vocab,
        method=method,
        tokenizer_note=tokenizer_note,
    )
    atomic_write_json(checkpoint_path, state)
    return state, False


def _counter_map(raw: Mapping[str, Any] | None) -> dict[str, Counter]:
    result: dict[str, Counter] = {}
    for key, values in (raw or {}).items():
        if isinstance(values, Mapping):
            result[str(key)] = Counter({str(k): int(v) for k, v in values.items()})
    return result


def _serialize_scan_state(
    state: dict[str, Any],
    *,
    next_file_index: int,
    totals: Counter,
    seen_files: set[str],
    seen_docs: set[str],
    by_type: Mapping[str, Counter],
    failures: Sequence[Mapping[str, str]],
    elapsed_seconds: float,
    last_completed_path: Path | None,
) -> dict[str, Any]:
    state.update(
        {
            "next_file_index": int(next_file_index),
            "totals": dict(totals),
            "seen_file_hashes": sorted(seen_files),
            "seen_document_hashes": sorted(seen_docs),
            "by_type": {key: dict(value) for key, value in sorted(by_type.items())},
            "failures": [dict(item) for item in failures],
            "elapsed_seconds": float(elapsed_seconds),
            "last_completed_path": str(last_completed_path) if last_completed_path else None,
            "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
    )
    return state


def scan_library(
    library: Path,
    *,
    estimate_tokens: bool,
    progress_every: int,
    checkpoint_path: Path,
    reset_progress: bool = False,
) -> dict[str, Any]:
    extractor, extensions = load_lantra_extractor()
    files = discover_library_files(library, extensions)
    if not files:
        raise RuntimeError(f"No LANTRA-supported files found in {library}")

    tokenizer, tokenizer_note = (None, "BPE disabled by --estimate-tokens")
    if not estimate_tokens:
        tokenizer, tokenizer_note = load_tokenizer()

    vocab = len(getattr(tokenizer, "vocab", {})) if tokenizer is not None else 50_000
    method = "slai_bpe" if tokenizer is not None else "characters_div_4"
    manifest, manifest_sha256 = build_manifest(library, files)
    repo_root = Path(__file__).resolve().parent
    english_dictionary, dictionary_sources = load_english_dictionary(repo_root)
    vocabulary_path = checkpoint_path.with_name("lantra_library_vocabulary.sqlite3")
    vocabulary = VocabularyTracker(
        vocabulary_path,
        dictionary_words=english_dictionary,
        manifest_sha256=manifest_sha256,
        reset=reset_progress,
    )

    if reset_progress:
        checkpoint_path.unlink(missing_ok=True)
        checkpoint_path.with_name(checkpoint_path.name + ".tmp").unlink(missing_ok=True)

    state, resumed = _load_or_create_scan_state(
        checkpoint_path,
        library=library,
        manifest_sha256=manifest_sha256,
        files_discovered=len(files),
        estimate_tokens=estimate_tokens,
        vocab=vocab,
        method=method,
        tokenizer_note=tokenizer_note,
    )

    next_index = int(state.get("next_file_index", 0))
    seen_files = {str(item) for item in state.get("seen_file_hashes", [])}
    seen_docs = {str(item) for item in state.get("seen_document_hashes", [])}
    by_type = _counter_map(state.get("by_type") if isinstance(state.get("by_type"), Mapping) else {})
    failures = [
        {str(k): str(v) for k, v in item.items()}
        for item in state.get("failures", [])
        if isinstance(item, Mapping)
    ]
    totals = Counter(
        {str(k): int(v) for k, v in state.get("totals", {}).items()}
        if isinstance(state.get("totals"), Mapping)
        else {"files_discovered": len(files)}
    )
    totals["files_discovered"] = len(files)
    prior_elapsed = float(state.get("elapsed_seconds", 0.0))
    started = time.perf_counter()

    smallest = manifest[0]["size"]
    largest = manifest[-1]["size"]
    print(
        f"[ORDER] {len(files):,} supported files sorted smallest -> largest "
        f"({human_number(smallest)}B -> {human_number(largest)}B).",
        flush=True,
    )
    if resumed:
        if next_index < len(files):
            print(
                f"[RESUME] Continuing at file {next_index + 1:,}/{len(files):,}: "
                f"{files[next_index]} ({files[next_index].stat().st_size:,} bytes).",
                flush=True,
            )
        else:
            print(
                f"[RESUME] Checkpoint already contains all {len(files):,} files; "
                "reusing completed scan state.",
                flush=True,
            )
    else:
        print(f"[CHECKPOINT] Progress file: {checkpoint_path}", flush=True)

    for zero_index in range(next_index, len(files)):
        path = files[zero_index]
        i = zero_index + 1
        ext = path.suffix.lower().lstrip(".") or "unknown"
        source_size = int(path.stat().st_size)

        # Per-file transaction. Nothing below is merged into global state until the
        # entire file finishes, so a crash retries only this file without double-counting.
        delta = Counter()
        delta["source_bytes"] = source_size
        bucket_delta = Counter()
        bucket_delta["files"] = 1
        pending_file_hashes: set[str] = set()
        pending_doc_hashes: set[str] = set()
        pending_failures: list[dict[str, str]] = []

        try:
            file_hash = sha256_file(path)
            if file_hash in seen_files:
                delta["duplicate_files"] += 1
            else:
                pending_file_hashes.add(file_hash)

                # ``extract_raw_documents`` is imported dynamically, so static
                # analysis cannot infer that its result is iterable.
                extracted = extractor(path, source_sha256=file_hash)
                documents = list(extracted) if isinstance(extracted, Iterable) else []
                if not documents:
                    delta["extraction_failures"] += 1
                    pending_failures.append({"path": str(path), "error": "NoTextExtracted"})
                else:
                    delta["files_accepted"] += 1
                    bucket_delta["accepted_files"] += 1

                    for document in documents:
                        delta["documents_extracted"] += 1
                        text = str(getattr(document, "text", "") or "").strip()
                        if not text:
                            continue

                        doc_hash = canonical_hash(text)
                        if doc_hash in seen_docs or doc_hash in pending_doc_hashes:
                            delta["duplicate_documents"] += 1
                            continue

                        chars = len(text)
                        words = count_words(text)
                        if tokenizer is not None:
                            try:
                                tokens = count_bpe_tokens(tokenizer, text)
                            except Exception as exc:
                                tokens = max(1, math.ceil(chars / 4))
                                pending_failures.append(
                                    {
                                        "path": str(path),
                                        "error": (
                                            "TokenizerFallback: "
                                            f"{type(exc).__name__}: {exc}"
                                        ),
                                    }
                                )
                        else:
                            tokens = max(1, math.ceil(chars / 4))

                        vocabulary.record_document(doc_hash, text)
                        pending_doc_hashes.add(doc_hash)
                        delta["unique_documents"] += 1
                        delta["characters"] += chars
                        delta["words"] += words
                        delta["tokens"] += tokens
                        bucket_delta["documents"] += 1
                        bucket_delta["characters"] += chars
                        bucket_delta["words"] += words
                        bucket_delta["tokens"] += tokens

        except Exception as exc:
            # The failed file is still considered completed for this scan. Its failure
            # is checkpointed and the next run resumes after it, matching prior behavior.
            delta["extraction_failures"] += 1
            pending_failures.append(
                {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )

        # Commit the complete file atomically to in-memory state, then persist it.
        totals.update(delta)
        seen_files.update(pending_file_hashes)
        seen_docs.update(pending_doc_hashes)
        by_type.setdefault(ext, Counter()).update(bucket_delta)
        failures.extend(pending_failures)

        elapsed = prior_elapsed + (time.perf_counter() - started)
        _serialize_scan_state(
            state,
            next_file_index=i,
            totals=totals,
            seen_files=seen_files,
            seen_docs=seen_docs,
            by_type=by_type,
            failures=failures,
            elapsed_seconds=elapsed,
            last_completed_path=path,
        )
        atomic_write_json(checkpoint_path, state)

        if progress_every and (i % progress_every == 0 or i == len(files)):
            print(
                f"[{i:>4}/{len(files)}] files | "
                f"size={human_number(source_size)}B | "
                f"docs={totals['unique_documents']:,} | "
                f"tokens={totals['tokens']:,} | "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    total_elapsed = prior_elapsed + (time.perf_counter() - started)
    vocabulary_summary = vocabulary.summary()
    vocabulary.close()
    if totals["tokens"] <= 0:
        raise RuntimeError("No usable training text was extracted")

    return {
        **dict(totals),
        "total_documents": int(totals.get("unique_documents", 0)),
        "total_bytes": int(totals.get("source_bytes", 0)),
        "total_words": int(totals.get("words", 0)),
        **vocabulary_summary,
        "dictionary_sources": list(dictionary_sources),
        "vocabulary_database": str(vocabulary_path),
        "vocab_size": vocab,
        "token_count_method": method,
        "tokenizer_note": tokenizer_note,
        "scan_order": "file_size_ascending",
        "checkpoint_path": str(checkpoint_path),
        "resumed_from_checkpoint": resumed,
        "by_type": {key: dict(value) for key, value in sorted(by_type.items())},
        "failures": failures,
        "elapsed_seconds": total_elapsed,
    }

def build_report(
    corpus: Mapping[str, Any],
    *,
    library: Path,
    tokens_per_parameter: float,
    planned_epochs: int,
    context: int,
) -> dict[str, Any]:
    tokens = int(corpus["tokens"])
    target = tokens / tokens_per_parameter
    profiles = build_profiles(int(corpus["vocab_size"]))
    recommended = nearest_profile(profiles, target)

    comparisons = []
    for label, params in COMPARE_SIZES:
        needed = int(params * tokens_per_parameter)
        comparisons.append(
            {
                "model": label,
                "parameters": params,
                "tokens_required": needed,
                "coverage": tokens / max(1, needed),
                "token_shortfall": max(0, needed - tokens),
            }
        )

    if target < profiles[0]["estimated_parameters"]:
        note = (
            "The corpus is smaller than the reference token budget for even the smallest "
            "profile. The selected profile is an architectural floor; add more unique text "
            "before scaling upward."
        )
    else:
        note = (
            "The selected profile is the nearest balanced encoder-decoder configuration to "
            "the token-derived target. Validate it with held-out loss and scaling experiments."
        )

    return {
        "schema": "slai.lantra.library-size-analysis.v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "library": str(library),
        "corpus": dict(corpus),
        "heuristic": {
            "tokens_per_parameter": tokens_per_parameter,
            "unique_token_parameter_target": int(round(target)),
            "parameter_band_15_to_30_tokens_per_parameter": {
                "lower": int(round(tokens / 30.0)),
                "upper": int(round(tokens / 15.0)),
            },
            "planned_epochs": planned_epochs,
            "token_exposures_if_repeated": tokens * planned_epochs,
            "note": (
                "Repeated epochs increase optimization exposure but do not create new information, "
                "so unique corpus tokens determine the primary recommendation."
            ),
        },
        "recommendation": {
            "profile": recommended,
            "note": note,
            "config": suggested_config(recommended, context),
        },
        "profile_ladder": profiles,
        "model_size_comparison": comparisons,
        "methodology_note": (
            "This is a planning heuristic, not proof of an optimal model size. LANTRA is an "
            "encoder-decoder denoising/multitask model, so verify the result empirically before "
            "committing large compute."
        ),
    }


def write_outputs(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "lantra_library_analysis.json"
    config_path = output_dir / "lantra_recommended_config.yaml"

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    profile = report["recommendation"]["profile"]
    header = (
        "# Generated by analyze_lantra_library.py\n"
        f"# Unique corpus tokens: {report['corpus']['tokens']:,}\n"
        f"# Estimated parameters: {profile['estimated_parameters']:,}\n"
        "# Review before copying into language_config.yaml.\n\n"
    )
    config_path.write_text(
        header + simple_yaml(report["recommendation"]["config"]) + "\n",
        encoding="utf-8",
    )
    return report_path, config_path


def print_summary(report: Mapping[str, Any], report_path: Path, config_path: Path) -> None:
    corpus = report["corpus"]
    heuristic = report["heuristic"]
    profile = report["recommendation"]["profile"]

    print("\n" + "=" * 72)
    print("LANTRA LIBRARY ANALYSIS")
    print("=" * 72)
    print(f"Files discovered:      {corpus['files_discovered']:,}")
    print(f"Files accepted:        {corpus['files_accepted']:,}")
    print(f"Unique documents:      {corpus['unique_documents']:,}")
    print(f"Duplicate files:       {int(corpus.get('duplicate_files', 0)):,}")
    print(f"Duplicate documents:   {int(corpus.get('duplicate_documents', 0)):,}")
    print(f"Extraction failures:   {int(corpus.get('extraction_failures', 0)):,}")
    print(f"Characters:            {corpus['characters']:,}")
    print(f"Words:                 {corpus['total_words']:,}")
    print(f"Unique words:          {corpus['total_unique_words']:,}")
    print(
        f"English dictionary:    {corpus['english_dictionary_percentage']:.2f}% "
        f"({corpus['dictionary_matched_words']:,}/{corpus['dictionary_eligible_words']:,} eligible occurrences)"
    )
    print(f"Dictionary unmatched:  {corpus['dictionary_unmatched_words']:,}")
    print(f"Training tokens:       {corpus['tokens']:,} ({corpus['token_count_method']})")
    if corpus.get("tokenizer_note"):
        print(f"Tokenizer note:        {corpus['tokenizer_note']}")

    print("\nSizing")
    print("-" * 72)
    print(f"Reference ratio:       {heuristic['tokens_per_parameter']:g} tokens/parameter")
    print(f"Token-derived target:  {human_number(heuristic['unique_token_parameter_target'])}")
    band = heuristic["parameter_band_15_to_30_tokens_per_parameter"]
    print(f"Reference band:        {human_number(band['lower'])} - {human_number(band['upper'])}")

    print("\nRecommended profile")
    print("-" * 72)
    print(f"Name:                  {profile['name']}")
    print(f"Estimated parameters:  {human_number(profile['estimated_parameters'])}")
    print(f"d_model / heads:       {profile['d_model']} / {profile['nhead']}")
    print(f"Encoder / decoder:     {profile['encoder_layers']} / {profile['decoder_layers']}")
    print(f"FFN width:             {profile['dim_feedforward']}")

    print("\n1B-3B data coverage")
    print("-" * 72)
    for row in report["model_size_comparison"]:
        if row["parameters"] < 1_000_000_000:
            continue
        print(
            f"{row['model']:>3}: need {human_number(row['tokens_required']):>9} tokens | "
            f"coverage {row['coverage'] * 100:8.4f}% | "
            f"shortfall {human_number(row['token_shortfall'])}"
        )

    print("\nOutputs")
    print("-" * 72)
    print(report_path)
    print(config_path)
    print("=" * 72)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze data/library and recommend a LANTRA parameter/config profile."
    )
    parser.add_argument("--library", default=str(LIBRARY))
    parser.add_argument(
        "--tokens-per-parameter",
        type=float,
        default=DEFAULT_TOKENS_PER_PARAMETER,
        help="Planning heuristic; default: 20 unique tokens per parameter.",
    )
    parser.add_argument("--planned-epochs", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--estimate-tokens",
        action="store_true",
        help="Use characters/4 instead of BPE token counting.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help=(
            "Resume checkpoint path. Default: <output-dir>/"
            f"{CHECKPOINT_FILENAME}"
        ),
    )
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Discard the matching resume checkpoint and scan the library from the beginning.",
    )
    args = parser.parse_args(argv)

    if args.tokens_per_parameter <= 0:
        parser.error("--tokens-per-parameter must be > 0")
    if args.planned_epochs < 1:
        parser.error("--planned-epochs must be >= 1")
    if args.context_length < 16:
        parser.error("--context-length must be >= 16")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")

    library = Path(args.library)
    if not library.is_dir():
        parser.error(f"Library directory not found: {library}")

    try:
        output_dir = Path(args.output_dir)
        checkpoint_path = (
            Path(args.checkpoint_file)
            if args.checkpoint_file
            else output_dir / CHECKPOINT_FILENAME
        )
        corpus = scan_library(
            library,
            estimate_tokens=args.estimate_tokens,
            progress_every=args.progress_every,
            checkpoint_path=checkpoint_path,
            reset_progress=args.reset_progress,
        )
        report = build_report(
            corpus,
            library=library,
            tokens_per_parameter=args.tokens_per_parameter,
            planned_epochs=args.planned_epochs,
            context=args.context_length,
        )
        report_path, config_path = write_outputs(report, output_dir)
        print_summary(report, report_path, config_path)
        return 0
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nAnalysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
