"""Composition entry point for checkpoint codec implementations.

Concrete serialization behavior lives exclusively in ``checkpointing.codecs``.
This module assembles a deterministic default registry and provides the stable
top-level imports consumed by the future checkpoint manager.  It creates no
global registry and imports PyTorch only when a torch operation is executed.
"""

from __future__ import annotations

from .checkpoint_types import StandardComponent
from .codecs import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]


logger = get_logger("Checkpoint Codecs")
printer = PrettyPrinter()


def build_default_codec_registry(
    *,
    include_torch: bool = True,
    freeze: bool = True,
) -> CodecRegistry:
    """Build an isolated registry with one explicit default per component.

    ``include_torch`` controls registration, not module import: constructing a
    ``TorchCheckpointCodec`` remains safe on systems without PyTorch because
    the framework dependency is loaded lazily.  Set ``freeze=False`` only when
    application startup must register additional codecs before use.
    """

    if not isinstance(include_torch, bool) or not isinstance(freeze, bool):
        raise TypeError("include_torch and freeze must be booleans")
    registry = CodecRegistry()
    numpy_codec = NumpyCheckpointCodec()
    registry.register(numpy_codec)

    if include_torch:
        registry.register(
            TorchCheckpointCodec(),
            default_for=(
                StandardComponent.MODEL.value,
                StandardComponent.OPTIMIZER.value,
                StandardComponent.SCHEDULER.value,
                StandardComponent.SCALER.value,
            ),
        )
    else:
        registry.set_default(StandardComponent.MODEL.value, numpy_codec.codec_id)

    registry.register(
        TokenizerCheckpointCodec(),
        default_for=(StandardComponent.TOKENIZER.value,),
    )
    registry.register(
        RNGStateCodec(include_torch=include_torch),
        default_for=(StandardComponent.RNG.value,),
    )
    registry.register(
        AgentStateCodec(),
        default_for=(StandardComponent.AGENT_STATE.value,),
    )
    if freeze:
        registry.freeze()
    return registry


__all__ = [
    "AgentStateCodec",
    "BaseCheckpointCodec",
    "CodecDescriptor",
    "CodecRegistry",
    "CodecRegistrySnapshot",
    "NumpyCheckpointCodec",
    "RNGRestoreReport",
    "RNGStateCodec",
    "RNGStateSnapshot",
    "ShapeMismatch",
    "StateLoadReport",
    "TokenizerCheckpointCodec",
    "TokenizerPayload",
    "TokenizerPersistenceKind",
    "TorchCheckpointCodec",
    "build_default_codec_registry",
]