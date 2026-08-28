"""Text perception pipeline for SLAI.

This module composes the current ``main`` branch ``Tokenizer``, ``TextEncoder``,
and optional ``TextDecoder`` behind the internal modality contract.  It does
not become an independently routable SLAI agent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from ..decoders.text_decoder import TextDecoder
from ..encoders.text_encoder import TextEncoder
from ..modules.tokenizer import Tokenizer
from ..perception_contracts import *
from ..utils.perception_errors import *
from ..utils.perception_helpers import *
from .base import BasePerceptionModality
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("Tekst Perception")
printer = PrettyPrinter()


class TextPerception(BasePerceptionModality):
    """Vertical text-perception component: tokenize -> encode -> optional reconstruct."""

    def __init__(
        self,
        *,
        encoder: Optional[TextEncoder] = None,
        decoder: Optional[TextDecoder] = None,
        tokenizer: Optional[Tokenizer] = None,
        enable_decoder: bool = False,
        config: Optional[Mapping[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        active_encoder = encoder or TextEncoder()
        active_tokenizer = tokenizer or Tokenizer()
        active_decoder = decoder
        if active_decoder is None and enable_decoder:
            active_decoder = TextDecoder(encoder=active_encoder)

        super().__init__(
            modality=Modality.TEXT,
            encoder=active_encoder,
            decoder=active_decoder,
            config=config,
            device=device,
        )
        self.tokenizer = active_tokenizer
        self.vocab_size = int(self.tokenizer.get_vocab_size())
        if self.vocab_size <= 0:
            raise ModalityInputError(
                "Tokenizer vocabulary must be non-empty.",
                component="text_perception",
            )
        self.prediction_head = nn.Linear(self.embed_dim, self.vocab_size).to(self.device)

    def _prepare_payload(self, payload: Any) -> Dict[str, Any]:
        if isinstance(payload, str):
            prepared = dict(self.tokenizer(payload))
        elif isinstance(payload, torch.Tensor):
            prepared = {"input_ids": payload}
        elif isinstance(payload, Mapping):
            prepared = canonicalize_payload_aliases(Modality.TEXT, payload)
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            if payload and all(isinstance(item, str) for item in payload):
                prepared = dict(self.tokenizer(list(payload)))
            else:
                prepared = {"input_ids": torch.as_tensor(payload, dtype=torch.long)}
        else:
            raise ModalityInputError(
                "Text input must be text, token IDs, a batch of texts, or a mapping.",
                component="text_perception",
                details={"input_type": type(payload).__name__},
            )

        if "input_ids" not in prepared:
            raise ModalityInputError(
                "Text payload does not contain canonical 'input_ids'.",
                component="text_perception",
                details={"keys": sorted(prepared.keys())},
            )

        input_ids = prepared["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2:
            raise ModalityInputError(
                "Text input_ids must have shape (B, L).",
                component="text_perception",
                details={"shape": list(input_ids.shape)},
            )
        prepared["input_ids"] = input_ids

        attention_mask = prepared.get("attention_mask")
        if attention_mask is None:
            try:
                pad_token_id = int(self.tokenizer.token_to_id(self.tokenizer.pad_token))
                attention_mask = input_ids.ne(pad_token_id)
            except Exception:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        attention_mask = normalize_attention_mask(
            attention_mask,
            batch_size=input_ids.size(0),
            seq_len=input_ids.size(1),
            device=self.device,
        )
        prepared["attention_mask"] = attention_mask
        return prepared

    def encode(
        self,
        payload: Any,
        *,
        style_id: Optional[Any] = None,
        pooling: Optional[str] = None,
    ) -> ModalityRepresentation:
        prepared = self._prepare_payload(payload)
        input_ids = prepared["input_ids"]
        attention_mask = prepared["attention_mask"]
        style = self._normalize_style_id(
            prepared.get("style_id", style_id),
            input_ids.size(0),
        )
        try:
            encoded = self.encoder(
                input_ids,
                attention_mask=attention_mask,
                style_id=style,
                output_type="sequence",
            )
        except Exception as exc:
            raise wrap_exception(
                exc,
                ModalityEncodingError,
                "Text encoder failed.",
                component="text_perception",
                details={"input_shape": list(input_ids.shape)},
            ) from exc

        return self._build_representation(
            encoded,
            attention_mask=attention_mask,
            source_shape=tuple(input_ids.shape),
            metadata={"token_count": int(attention_mask.sum().item())},
            pooling=pooling,
        )

    def masked_prediction(
        self,
        payload: Any,
        *,
        mask_ratio: float = 0.15,
        style_id: Optional[Any] = None,
        **_: Any,
    ) -> MaskedPrediction:
        if not 0.0 <= float(mask_ratio) <= 1.0:
            raise ModalityInputError(
                "mask_ratio must be in [0, 1].",
                component="text_perception",
                details={"mask_ratio": mask_ratio},
            )

        prepared = self._prepare_payload(payload)
        input_ids = prepared["input_ids"]
        attention_mask = prepared["attention_mask"]
        style = self._normalize_style_id(
            prepared.get("style_id", style_id),
            input_ids.size(0),
        )

        try:
            mask_token_id = int(self.tokenizer.token_to_id(self.tokenizer.mask_token))
            special_ids = {
                int(self.tokenizer.token_to_id(self.tokenizer.pad_token)),
                int(self.tokenizer.token_to_id(self.tokenizer.cls_token)),
                int(self.tokenizer.token_to_id(self.tokenizer.sep_token)),
            }
        except Exception as exc:
            raise ModalityInputError(
                "Tokenizer special-token IDs are unavailable for masked text learning.",
                component="text_perception",
                cause=exc,
            ) from exc

        eligible = attention_mask.clone()
        for token_id in special_ids:
            eligible &= input_ids.ne(token_id)

        sampled = torch.rand(input_ids.shape, device=self.device) < float(mask_ratio)
        mask = sampled & eligible
        masked_input_ids = input_ids.clone()
        masked_input_ids[mask] = mask_token_id

        try:
            encoded = self.encoder(
                masked_input_ids,
                attention_mask=attention_mask,
                style_id=style,
                output_type="sequence",
            )
            predictions = self.prediction_head(encoded)
        except Exception as exc:
            raise wrap_exception(
                exc,
                ModalityEncodingError,
                "Masked text prediction failed.",
                component="text_perception",
            ) from exc

        representation = self._build_representation(
            encoded,
            attention_mask=attention_mask,
            source_shape=tuple(input_ids.shape),
            metadata={"masked_tokens": int(mask.sum().item())},
        )
        return MaskedPrediction(
            modality=Modality.TEXT,
            predictions=predictions,
            targets=input_ids,
            mask=mask,
            representation=representation,
            metadata={"masked_tokens": int(mask.sum().item())},
        )

    def reconstruct(
        self,
        representation: Union[ModalityRepresentation, torch.Tensor],
        *,
        strategy: str = "greedy",
        style_id: Optional[Any] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        decoder = self.require_decoder()
        if isinstance(representation, ModalityRepresentation):
            memory = representation.sequence
            if memory is None:
                memory = representation.pooled.unsqueeze(1)
            if memory_mask is None:
                memory_mask = representation.attention_mask
        else:
            memory = require_tensor(representation, "representation", component="text_perception")
            if memory.dim() == 2:
                memory = memory.unsqueeze(1)
        if memory.dim() != 3 or memory.size(-1) != self.embed_dim:
            raise PerceptionDimensionError(
                "Text decoder memory must have shape (B, L, embed_dim).",
                component="text_perception",
                details={"shape": list(memory.shape), "embed_dim": self.embed_dim},
            )
        style = self._normalize_style_id(style_id, memory.size(0))
        try:
            return decoder.inference(
                memory,
                memory_mask=memory_mask,
                style_id=style,
                strategy=strategy,
            )
        except Exception as exc:
            raise wrap_exception(
                exc,
                ModalityDecodingError,
                "Text decoder failed.",
                component="text_perception",
            ) from exc

    def decode_tokens(self, token_ids: torch.Tensor, *, skip_special_tokens: bool = True) -> Union[str, list[str]]:
        token_ids = require_tensor(token_ids, "token_ids", component="text_perception")
        if token_ids.dim() == 1:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        if token_ids.dim() == 2:
            return [
                self.tokenizer.decode(row, skip_special_tokens=skip_special_tokens)
                for row in token_ids
            ]
        raise ModalityInputError(
            "token_ids must have shape (L,) or (B,L).",
            component="text_perception",
            details={"shape": list(token_ids.shape)},
        )


__all__ = ["TextPerception"]
