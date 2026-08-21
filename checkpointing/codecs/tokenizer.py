"""Deterministic tokenizer persistence without a transformers dependency.

Three explicit strategies are supported, in order for ``auto`` selection:
``save_pretrained``, a ``word_to_id`` vocabulary, and ``state_dict``.  The
codec writes a small descriptor plus the chosen payload, and restoration is an
explicit operation rather than an implicit side effect of decoding.
"""

from __future__ import annotations

import os
import shutil
import stat
import uuid

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..checkpoint_errors import *
from ..checkpoint_storage import *
from ..checkpoint_types import *
from .base import *
from .torch import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]


logger = get_logger("Checkpoint Codec Tokenizer")
printer = PrettyPrinter()


_FORMAT = "slai.checkpoint.tokenizer"
_SCHEMA_VERSION = 1
_DESCRIPTOR_NAME = "tokenizer.codec.json"
_VOCABULARY_NAME = "vocabulary.json"
_STATE_NAME = "state.pt"
_PRETRAINED_DIRECTORY = "pretrained"


class TokenizerPersistenceKind(str, Enum):
    """Supported tokenizer serialization interfaces."""

    PRETRAINED = "save_pretrained"
    VOCABULARY = "word_to_id"
    STATE_DICT = "state_dict"


@dataclass(frozen=True, slots=True)
class TokenizerPayload:
    """Decoded tokenizer payload awaiting explicit application to a target."""

    kind: TokenizerPersistenceKind
    payload_path: Path
    vocabulary: Mapping[str, int] | None = None
    state: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", TokenizerPersistenceKind(self.kind))
        object.__setattr__(self, "payload_path", Path(self.payload_path))
        if self.vocabulary is not None:
            object.__setattr__(
                self,
                "vocabulary",
                MappingProxyType(_validate_vocabulary(self.vocabulary)),
            )
        if self.state is not None and not isinstance(self.state, Mapping):
            raise TypeError("tokenizer state must be a mapping")
        if self.state is not None:
            object.__setattr__(self, "state", MappingProxyType(dict(self.state)))
        if self.kind is TokenizerPersistenceKind.VOCABULARY:
            if self.vocabulary is None or self.state is not None:
                raise ValueError("vocabulary payload must contain only a vocabulary")
        elif self.kind is TokenizerPersistenceKind.STATE_DICT:
            if self.state is None or self.vocabulary is not None:
                raise ValueError("state-dict payload must contain only state")
        elif self.vocabulary is not None or self.state is not None:
            raise ValueError("pretrained payload state is represented by its directory")


def _validate_vocabulary(value: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("tokenizer vocabulary must be a non-empty mapping")
    vocabulary: dict[str, int] = {}
    identifiers: set[int] = set()
    for token, identifier in value.items():
        if not isinstance(token, str):
            raise ValueError("tokenizer vocabulary tokens must be strings")
        if token in vocabulary:
            raise ValueError("tokenizer vocabulary contains duplicate tokens")
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier < 0
        ):
            raise ValueError("tokenizer vocabulary IDs must be non-negative integers")
        if identifier in identifiers:
            raise ValueError("tokenizer vocabulary IDs must be unique")
        identifiers.add(identifier)
        vocabulary[token] = identifier
    return vocabulary


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "application/json"
    if suffix in {".txt", ".vocab", ".merges"}:
        return "text/plain"
    if suffix in {".pt", ".pth", ".bin"}:
        return "application/octet-stream"
    return "application/octet-stream"


class TokenizerCheckpointCodec(BaseCheckpointCodec):
    """Persist tokenizer assets through a capability-based explicit strategy."""

    def __init__(
        self,
        *,
        max_files: int = 4096,
        max_directories: int = 4096,
        max_total_bytes: int = 2 * 1024 * 1024 * 1024,
        max_descriptor_bytes: int = 1024 * 1024,
        allow_unsafe_pickle: bool = False,
    ) -> None:
        super().__init__(
            "tokenizer",
            "1",
            (StandardComponent.TOKENIZER.value,),
            multi_file=True,
        )
        for name, value in (
            ("max_files", max_files),
            ("max_directories", max_directories),
            ("max_total_bytes", max_total_bytes),
            ("max_descriptor_bytes", max_descriptor_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.max_files = max_files
        self.max_directories = max_directories
        self.max_total_bytes = max_total_bytes
        self.max_descriptor_bytes = max_descriptor_bytes
        self._torch_codec = TorchCheckpointCodec(
            components=(StandardComponent.TOKENIZER.value,),
            save_on_cpu=True,
            allow_unsafe_pickle=allow_unsafe_pickle,
            max_source_bytes=max_total_bytes,
        )

    def _supports_value(self, value: Any | None) -> bool:
        return value is None or any(
            (
                callable(getattr(value, "save_pretrained", None)),
                isinstance(getattr(value, "word_to_id", None), Mapping),
                callable(getattr(value, "state_dict", None)),
            )
        )

    @staticmethod
    def _choose_strategy(value: Any, requested: str | None) -> TokenizerPersistenceKind:
        available: list[TokenizerPersistenceKind] = []
        if callable(getattr(value, "save_pretrained", None)):
            available.append(TokenizerPersistenceKind.PRETRAINED)
        if isinstance(getattr(value, "word_to_id", None), Mapping):
            available.append(TokenizerPersistenceKind.VOCABULARY)
        if callable(getattr(value, "state_dict", None)):
            available.append(TokenizerPersistenceKind.STATE_DICT)
        if not available:
            raise CheckpointTokenizerError(
                "unsupported tokenizer serialization interface",
                tokenizer_type=type(value).__name__,
                expected_methods=["save_pretrained", "word_to_id", "state_dict"],
            )
        if requested is None or requested == "auto":
            return available[0]
        try:
            selected = TokenizerPersistenceKind(requested)
        except ValueError as exc:
            raise CheckpointTokenizerError(
                f"unsupported tokenizer strategy {requested!r}",
                tokenizer_type=type(value).__name__,
            ) from exc
        if selected not in available:
            raise CheckpointTokenizerError(
                f"tokenizer does not implement requested strategy {selected.value!r}",
                tokenizer_type=type(value).__name__,
                expected_methods=[selected.value],
            )
        return selected

    def _secure_files(
        self,
        root: Path,
        *,
        durable: bool,
        set_permissions: bool,
    ) -> tuple[Path, ...]:
        files: list[Path] = []
        directory_count = 0
        total_bytes = 0
        for path in root.rglob("*"):
            if path.is_symlink():
                raise CheckpointTokenizerError(
                    "tokenizer output contains a symbolic link",
                    path=path,
                )
            mode = path.stat().st_mode
            if stat.S_ISDIR(mode):
                directory_count += 1
                if directory_count > self.max_directories:
                    raise CheckpointTokenizerError(
                        "tokenizer output contains too many directories",
                        path=root,
                        details={
                            "directory_count": directory_count,
                            "max_directories": self.max_directories,
                        },
                    )
                if set_permissions:
                    os.chmod(path, 0o700)
                continue
            if not stat.S_ISREG(mode):
                raise CheckpointTokenizerError(
                    "tokenizer output contains a non-regular file",
                    path=path,
                )
            files.append(path)
            total_bytes += path.stat().st_size
            if len(files) > self.max_files or total_bytes > self.max_total_bytes:
                raise CheckpointTokenizerError(
                    "tokenizer output exceeds configured resource limits",
                    path=root,
                    details={
                        "file_count": len(files),
                        "total_bytes": total_bytes,
                        "max_files": self.max_files,
                        "max_total_bytes": self.max_total_bytes,
                    },
                )
            if set_permissions:
                os.chmod(path, 0o600)
        if durable:
            fsync_tree(root, strict=True)
        return tuple(sorted(files))

    def _encode(
        self,
        value: Any,
        destination: Path,
        *,
        context: CodecContext,
    ) -> Sequence[CodecOutput]:
        if destination.exists() or destination.is_symlink():
            raise CheckpointTokenizerError(
                "tokenizer destination must not already exist",
                path=destination,
            )
        strategy = self._choose_strategy(
            value,
            metadata_string(context, "tokenizer_strategy", default="auto"),
        )
        durable = metadata_bool(context, "durable", default=True)
        temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.tokenizer-tmp"
        )
        temporary.mkdir(parents=True, mode=0o700)
        try:
            if strategy is TokenizerPersistenceKind.PRETRAINED:
                payload = temporary / _PRETRAINED_DIRECTORY
                payload.mkdir(mode=0o700)
                value.save_pretrained(str(payload))
                relative_payload = _PRETRAINED_DIRECTORY
            elif strategy is TokenizerPersistenceKind.VOCABULARY:
                payload = temporary / _VOCABULARY_NAME
                vocabulary = _validate_vocabulary(value.word_to_id)
                write_json_object(
                    {"vocabulary": vocabulary},
                    payload,
                    durable=False,
                    max_bytes=self.max_total_bytes,
                )
                relative_payload = _VOCABULARY_NAME
            else:
                payload = temporary / _STATE_NAME
                inner_metadata = dict(context.metadata)
                inner_metadata["durable"] = False
                inner_context = CodecContext(
                    checkpoint_id=context.checkpoint_id,
                    version=context.version,
                    component=context.component,
                    metadata=inner_metadata,
                )
                self._torch_codec.encode(
                    value.state_dict(),
                    payload,
                    context=inner_context,
                )
                relative_payload = _STATE_NAME

            descriptor = {
                "format": _FORMAT,
                "schema_version": _SCHEMA_VERSION,
                "codec_version": self.codec_version,
                "kind": strategy.value,
                "payload": relative_payload,
            }
            write_json_object(
                descriptor,
                temporary / _DESCRIPTOR_NAME,
                durable=False,
                max_bytes=self.max_descriptor_bytes,
            )
            files = self._secure_files(
                temporary,
                durable=durable,
                set_permissions=True,
            )
            if not files:
                raise CheckpointTokenizerError(
                    "tokenizer serializer produced no files",
                    path=temporary,
                )
            if strategy is TokenizerPersistenceKind.PRETRAINED and not any(
                file.is_relative_to(payload) for file in files
            ):
                raise CheckpointTokenizerError(
                    "save_pretrained() produced no tokenizer files",
                    tokenizer_type=type(value).__name__,
                    path=payload,
                )
            os.replace(temporary, destination)
            if durable:
                fsync_directory(destination.parent, strict=True)
            published = tuple(
                destination / file.relative_to(temporary) for file in files
            )
            return tuple(
                CodecOutput(
                    path=file,
                    media_type=_media_type(file),
                    metadata={
                        "strategy": strategy.value,
                        "descriptor": file == destination / _DESCRIPTOR_NAME,
                    },
                )
                for file in published
            )
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _descriptor(self, source: Path) -> tuple[TokenizerPersistenceKind, Path]:
        descriptor_path = source / _DESCRIPTOR_NAME
        descriptor = read_json_object(
            descriptor_path,
            max_bytes=self.max_descriptor_bytes,
        )
        allowed = {"format", "schema_version", "codec_version", "kind", "payload"}
        if set(descriptor) != allowed:
            raise CheckpointTokenizerError(
                "tokenizer descriptor has an invalid field set",
                path=descriptor_path,
                details={
                    "missing": sorted(allowed - set(descriptor)),
                    "unknown": sorted(set(descriptor) - allowed),
                },
            )
        if descriptor["format"] != _FORMAT:
            raise CheckpointTokenizerError(
                "tokenizer descriptor has an unexpected format marker",
                path=descriptor_path,
            )
        if descriptor["schema_version"] != _SCHEMA_VERSION:
            raise CheckpointTokenizerError(
                "unsupported tokenizer descriptor schema version",
                path=descriptor_path,
                details={"actual": descriptor["schema_version"]},
            )
        if descriptor["codec_version"] != self.codec_version:
            raise CheckpointTokenizerError(
                "unsupported tokenizer codec payload version",
                path=descriptor_path,
                details={"actual": descriptor["codec_version"]},
            )
        try:
            kind = TokenizerPersistenceKind(descriptor["kind"])
            relative = validate_relative_path(descriptor["payload"])
        except (TypeError, ValueError) as exc:
            raise CheckpointTokenizerError(
                "tokenizer descriptor contains invalid kind or payload path",
                path=descriptor_path,
            ) from exc
        expected_payload = {
            TokenizerPersistenceKind.PRETRAINED: _PRETRAINED_DIRECTORY,
            TokenizerPersistenceKind.VOCABULARY: _VOCABULARY_NAME,
            TokenizerPersistenceKind.STATE_DICT: _STATE_NAME,
        }[kind]
        if relative != expected_payload:
            raise CheckpointTokenizerError(
                "tokenizer descriptor payload path does not match its strategy",
                path=descriptor_path,
                details={"expected": expected_payload, "actual": relative},
            )
        payload = source / relative
        try:
            resolved_source = source.resolve(strict=True)
            resolved_payload = payload.resolve(strict=True)
            resolved_payload.relative_to(resolved_source)
        except (OSError, ValueError) as exc:
            raise CheckpointTokenizerError(
                "tokenizer payload is missing or escapes its codec directory",
                path=payload,
            ) from exc
        if payload.is_symlink():
            raise CheckpointTokenizerError(
                "tokenizer payload cannot be a symbolic link",
                path=payload,
            )
        return kind, payload

    def _decode(self, source: Path, *, context: CodecContext) -> TokenizerPayload:
        files = self._secure_files(source, durable=False, set_permissions=False)
        kind, payload = self._descriptor(source)
        if kind is TokenizerPersistenceKind.PRETRAINED:
            if not payload.is_dir():
                raise CheckpointTokenizerError(
                    "pretrained tokenizer payload must be a directory",
                    path=payload,
                )
            if not any(file.is_relative_to(payload) for file in files):
                raise CheckpointTokenizerError(
                    "pretrained tokenizer payload contains no files",
                    path=payload,
                )
            return TokenizerPayload(kind, payload)
        if not payload.is_file():
            raise CheckpointTokenizerError(
                "tokenizer payload must be a regular file",
                path=payload,
            )
        if kind is TokenizerPersistenceKind.VOCABULARY:
            document = read_json_object(payload, max_bytes=self.max_total_bytes)
            if set(document) != {"vocabulary"}:
                raise CheckpointTokenizerError(
                    "tokenizer vocabulary document has an invalid field set",
                    path=payload,
                )
            vocabulary = _validate_vocabulary(document["vocabulary"])
            return TokenizerPayload(kind, payload, vocabulary=vocabulary)
        state = self._torch_codec.decode(payload, context=context)
        return TokenizerPayload(kind, payload, state=state)

    def restore(self, target: Any, payload: TokenizerPayload) -> Any:
        """Explicitly apply a decoded payload and return the effective tokenizer."""

        if not isinstance(payload, TokenizerPayload):
            raise TypeError("payload must be TokenizerPayload")
        if payload.kind is TokenizerPersistenceKind.PRETRAINED:
            loader = getattr(target, "from_pretrained", None)
            if not callable(loader):
                raise CheckpointTokenizerError(
                    "tokenizer target does not expose from_pretrained()",
                    tokenizer_type=type(target).__name__,
                    path=payload.payload_path,
                )
            try:
                return loader(str(payload.payload_path))
            except Exception as exc:
                raise CheckpointTokenizerError(
                    "tokenizer from_pretrained() restoration failed",
                    tokenizer_type=type(target).__name__,
                    path=payload.payload_path,
                    details={"error_type": type(exc).__name__},
                ) from exc
        if payload.kind is TokenizerPersistenceKind.STATE_DICT:
            loader = getattr(target, "load_state_dict", None)
            if not callable(loader):
                raise CheckpointTokenizerError(
                    "tokenizer target does not expose load_state_dict()",
                    tokenizer_type=type(target).__name__,
                    path=payload.payload_path,
                )
            try:
                loader(payload.state)
            except Exception as exc:
                raise CheckpointTokenizerError(
                    "tokenizer load_state_dict() restoration failed",
                    tokenizer_type=type(target).__name__,
                    path=payload.payload_path,
                    details={"error_type": type(exc).__name__},
                ) from exc
            return target

        if payload.vocabulary is None:
            raise CheckpointTokenizerError(
                "decoded vocabulary payload is missing its vocabulary",
                path=payload.payload_path,
            )
        vocabulary = dict(payload.vocabulary)
        updates = {
            "word_to_id": vocabulary,
            "id_to_word": {
                identifier: token for token, identifier in vocabulary.items()
            },
            "vocab_size": len(vocabulary),
        }
        absent = object()
        try:
            previous = {
                name: getattr(target, name, absent)
                for name in updates
            }
        except Exception as exc:
            raise CheckpointTokenizerError(
                "tokenizer vocabulary target could not be inspected",
                tokenizer_type=type(target).__name__,
                path=payload.payload_path,
                details={"error_type": type(exc).__name__},
            ) from exc
        applied: list[str] = []
        try:
            for name, value in updates.items():
                setattr(target, name, value)
                applied.append(name)
        except Exception as exc:
            rollback_failures: list[str] = []
            for name in reversed(applied):
                try:
                    old_value = previous[name]
                    if old_value is absent:
                        delattr(target, name)
                    else:
                        setattr(target, name, old_value)
                except Exception:
                    rollback_failures.append(name)
            raise CheckpointTokenizerError(
                "tokenizer target cannot accept vocabulary attributes",
                tokenizer_type=type(target).__name__,
                path=payload.payload_path,
                details={
                    "rolled_back": not rollback_failures,
                    "rollback_failures": rollback_failures,
                },
            ) from exc
        return target


__all__ = [
    "TokenizerCheckpointCodec",
    "TokenizerPayload",
    "TokenizerPersistenceKind",
]

if __name__ == "__main__":
    print("\n=== Running Tokenizer Checkpoint Codec Comprehensive Self-Test ===\n")
    import tempfile

    printer.status("TEST", "Starting tokenizer codec tests", "info")
    from src.agents.perception.modules.tokenizer import Tokenizer # type: ignore

    codec = TokenizerCheckpointCodec()
    printer.status("CODEC", f"created {codec.codec_id} v{codec.codec_version}", "success")

    tokenizer = Tokenizer()
    ctx = CodecContext(checkpoint_id="test", version="v1", component="tokenizer")

    # Test each strategy via metadata
    strategies = [
        ("save_pretrained", {}),
        ("vocabulary", {"tokenizer_strategy": "vocabulary"}),
        ("state_dict", {"tokenizer_strategy": "state_dict"}),
    ]

    for strategy_name, metadata_override in strategies:
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "tokenizer"
            context = CodecContext(
                checkpoint_id="test",
                version="v1",
                component="tokenizer",
                metadata=metadata_override,
            )
            outputs = codec.encode(tokenizer, dest, context=context)
            assert outputs, "encode produced no outputs"
            # Decode
            payload = codec.decode(dest, context=context)
            assert payload.kind.value == strategy_name
            # Restore into a new tokenizer
            new_tokenizer = Tokenizer()
            restored = codec.restore(new_tokenizer, payload)
            if strategy_name == "save_pretrained":
                assert restored.save_pretrained_called is False  # from_pretrained created new
                # but we can't easily verify, so just check it's a MockTokenizer
                assert isinstance(restored, Tokenizer)
            elif strategy_name == "vocabulary":
                assert restored.word_to_id == tokenizer.word_to_id
            else:  # state_dict
                assert restored.load_state_dict_called is True
                assert restored._state == tokenizer
    printer.status("STRATEGIES", "all three persistence strategies work", "success")

    # Error: unsupported strategy
    try:
        codec.encode(tokenizer, Path("/dev/null"), context=CodecContext(
            checkpoint_id="test", version="v1", component="tokenizer",
            metadata={"tokenizer_strategy": "unsupported"}
        ))
        assert False, "Should have raised"
    except CheckpointTokenizerError as e:
        assert "unsupported tokenizer strategy" in str(e)
    printer.status("ERRORS", "strategy validation works", "success")

    print("\n=== All tokenizer tests passed ===\n")