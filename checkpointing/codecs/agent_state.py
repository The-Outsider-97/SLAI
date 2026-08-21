"""Deterministic JSON codec for transparent, framework-neutral agent state.

This codec intentionally accepts only mappings composed of JSON-native values.
Tensor state, tokenizer assets, and RNG state belong to their dedicated codecs;
silently stringifying arbitrary Python objects would make recovery opaque and
non-reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..checkpoint_errors import *
from ..checkpoint_types import *
from .base import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]


logger = get_logger("Checkpoint Codec Agent State")
printer = PrettyPrinter()

_FORMAT = "slai.checkpoint.agent-state"
_SCHEMA_VERSION = 1


class AgentStateCodec(BaseCheckpointCodec):
    """Persist auditable agent state as a schema-versioned JSON object."""

    def __init__(self, *, max_bytes: int = 64 * 1024 * 1024) -> None:
        super().__init__(
            "agent-state",
            "1",
            (StandardComponent.AGENT_STATE.value,),
        )
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.max_bytes = max_bytes
        logger.info("Codec Descriptor initialized with", self.max_bytes)

    def _supports_value(self, value: Any | None) -> bool:
        return value is None or isinstance(value, Mapping)

    def _encode(
        self,
        value: Any,
        destination: Path,
        *,
        context: CodecContext,
    ) -> Sequence[CodecOutput]:
        if not isinstance(value, Mapping):
            raise TypeError("agent state must be a JSON-compatible mapping")
        envelope = {
            "format": _FORMAT,
            "schema_version": _SCHEMA_VERSION,
            "codec_version": self.codec_version,
            "component": context.component,
            "state": dict(value),
        }
        durable = metadata_bool(context, "durable", default=True)
        write_json_object(
            envelope,
            destination,
            durable=durable,
            max_bytes=self.max_bytes,
        )
        return (
            CodecOutput(
                path=destination,
                media_type="application/json",
                metadata={"schema_version": _SCHEMA_VERSION},
            ),
        )

    def _decode(self, source: Path, *, context: CodecContext) -> Mapping[str, Any]:
        envelope = read_json_object(source, max_bytes=self.max_bytes)
        allowed = {
            "format",
            "schema_version",
            "codec_version",
            "component",
            "state",
        }
        unknown = set(envelope) - allowed
        missing = allowed - set(envelope)
        if unknown or missing:
            raise CheckpointCodecError(
                "agent-state envelope has an invalid field set",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={"missing": sorted(missing), "unknown": sorted(unknown)},
            )
        if envelope["format"] != _FORMAT:
            raise CheckpointCodecError(
                "agent-state payload has an unexpected format marker",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={"actual": envelope["format"], "expected": _FORMAT},
            )
        if envelope["schema_version"] != _SCHEMA_VERSION:
            raise CheckpointCodecError(
                "unsupported agent-state schema version",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={
                    "actual": envelope["schema_version"],
                    "expected": _SCHEMA_VERSION,
                },
            )
        if envelope["codec_version"] != self.codec_version:
            raise CheckpointCodecError(
                "unsupported agent-state codec payload version",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={
                    "actual": envelope["codec_version"],
                    "expected": self.codec_version,
                },
            )
        if envelope["component"] != context.component:
            raise CheckpointCodecError(
                "agent-state component does not match decode context",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
                details={
                    "actual": envelope["component"],
                    "expected": context.component,
                },
            )
        state = envelope["state"]
        if not isinstance(state, dict):
            raise CheckpointCodecError(
                "agent-state payload must contain a state object",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        frozen = freeze_json(state, _path="$.agent_state")
        if not isinstance(frozen, Mapping):
            raise CheckpointCodecError(
                "agent-state payload must contain a state object",
                stage=CheckpointStage.DESERIALIZATION,
                path=source,
            )
        return frozen


__all__ = ["AgentStateCodec"]


if __name__ == "__main__":
    print("\n=== Running Checkpoint Codec Agent State Comprehensive Self-Test ===\n")
    printer.status("TEST", "Starting agent-state codec tests", "info")

    codec = AgentStateCodec()
    printer.status("CODEC", f"created {codec.codec_id} v{codec.codec_version}", "success")

    # Test supports
    assert codec.supports("agent_state") is True
    assert codec.supports("unknown") is False
    assert codec.supports("agent_state", value={"a": 1}) is True
    assert codec.supports("agent_state", value=42) is True  # value is Mapping or None, so 42 is False
    assert codec._supports_value(42) is False

    # Full encode/decode cycle
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        path = Path(tmp.name)
        test_state = {"step": 100, "data": [1, 2, 3]}
        ctx = CodecContext(checkpoint_id="test", version="v1", component="agent_state")
        outputs = codec.encode(test_state, path, context=ctx)
        assert len(outputs) == 1 and outputs[0].path == path

        decoded = codec.decode(path, context=ctx)
        assert decoded == test_state

    printer.status("ROUNDTRIP", "encode/decode passed", "success")

    # Test error cases: wrong component, invalid envelope
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        path = Path(tmp.name)
        # Write an invalid envelope
        import json
        path.write_text(json.dumps({"format": "bad", "state": {}}))
        ctx_wrong = CodecContext(checkpoint_id="test", version="v1", component="agent_state")
        try:
            codec.decode(path, context=ctx_wrong)
            assert False, "Should have raised"
        except CheckpointCodecError as e:
            assert "format marker" in str(e)

    printer.status("ERRORS", "error handling works", "success")
    print("\n=== All agent-state tests passed ===\n")